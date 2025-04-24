from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy import signal
import os
import logging
import librosa
from typing import Optional, Tuple, List, Any, Union, Dict, Callable
import gc
import tempfile
import io
import soundfile as sf
import traceback
import time
import psutil

from src.utils.config import ConfigManager
from src.utils.logging_utils import get_logger
from src.utils.cache import cached_result, numpy_cache
from src.utils.progress import track_progress, update_progress, complete_progress

logger = get_logger(__name__)

class AudioProcessor:
    """Audio processing utilities with memory-efficient operations."""
    
    def __init__(self, config_path: str = "../configs/default.yaml"):
        """Initialize audio processor with configuration."""
        self.config = ConfigManager(config_path)
        self.sample_rate = self.config.get('audio', 'sample_rate', 44100)
        self.normalize = self.config.get('audio', 'normalize', True)
        
        # Use simple relative path for cache
        self.cache_dir = "../cache"
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Memory management settings
        self.chunk_size = self.config.get('processing', 'chunk_size', 4096)
        self.max_file_size_mb = self.config.get('processing', 'max_file_size_mb', 100)
        
        # Performance settings
        self.use_memmap = self.config.get('processing', 'use_memmap', True)
        self.use_parallel_loading = self.config.get('processing', 'use_parallel_loading', True)
        self.n_fft = self.config.get('audio', 'n_fft', 2048)
        self.hop_length = self.config.get('audio', 'hop_length', 512)
        
        # Multithreading settings
        self.num_threads = min(os.cpu_count() or 4, self.config.get('processing', 'num_threads', 4))
        
        # Ensure max_file_size_mb is not None
        if self.max_file_size_mb is None:
            self.max_file_size_mb = 100
        
        # Ensure sample_rate is not None
        if self.sample_rate is None:
            self.sample_rate = 44100
        
        # Ensure normalize is not None
        if self.normalize is None:
            self.normalize = True
        
        logger.info(f"AudioProcessor initialized (sr={self.sample_rate}, normalize={self.normalize})")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Using {self.num_threads} threads for parallel processing")
    
    @cached_result(ttl=86400)  # Cache audio for 24 hours
    def load(self, file_path: str, start_time: float = 0.0, 
             end_time: Optional[float] = None) -> np.ndarray:
        """
        Load audio file with memory optimization for large files.
        
        Args:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds (or None for full file)
            
        Returns:
            numpy.ndarray: Audio data
        """
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Loading audio file: {file_path} ({file_size_mb:.2f} MB)")
        
        # Track progress for large files
        if file_size_mb > 10:
            progress = track_progress(f"load_{os.path.basename(file_path)}", 100, 
                                      f"Loading {os.path.basename(file_path)}")
        
        try:
            # Optimize loading strategy based on file size
            if file_size_mb > self.max_file_size_mb:
                logger.info(f"Large file detected, using memory-efficient loading")
                audio_data = self._load_large_file(file_path, start_time, end_time)
            else:
                # For smaller files, use standard loading
                audio_data = self._load_standard(file_path, start_time, end_time)
                
            if file_size_mb > 10:
                complete_progress(f"load_{os.path.basename(file_path)}")
                
            # Additional memory cleanup
            gc.collect()
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error loading audio file: {str(e)}")
            if file_size_mb > 10:
                complete_progress(f"load_{os.path.basename(file_path)}")
            raise
    
    def _load_standard(self, file_path: str, start_time: float = 0.0,
                      end_time: Optional[float] = None) -> np.ndarray:
        """
        Standard audio loading for smaller files.
        
        Args:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds (or None for full file)
            
        Returns:
            numpy.ndarray: Audio data
        """
        try:
            y, sr = librosa.load(
                file_path, 
                sr=self.sample_rate,
                offset=start_time,
                duration=None if end_time is None else (end_time - start_time),
                mono=True
            )
            
            logger.info(f"Audio loaded: {len(y)} samples, {len(y)/sr:.2f} seconds")
            
            if self.normalize:
                y = librosa.util.normalize(y)
                
            return y
            
        except Exception as e:
            # Try alternative loading with soundfile
            logger.warning(f"Standard loading failed, trying soundfile: {str(e)}")
            try:
                with sf.SoundFile(file_path) as sound_file:
                    sr = sound_file.samplerate
                    
                    # Calculate positions
                    start_frame = int(start_time * sr)
                    if end_time is not None:
                        end_frame = int(end_time * sr)
                        frame_count = end_frame - start_frame
                    else:
                        frame_count = -1  # Read all frames
                    
                    # Seek to start position
                    sound_file.seek(start_frame)
                    
                    # Read frames
                    y = sound_file.read(frames=frame_count, dtype='float32')
                    
                    # Convert to mono if needed
                    if sound_file.channels > 1:
                        y = np.mean(y, axis=1)
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
                    
                    if self.normalize:
                        y = librosa.util.normalize(y)
                        
                    return y
                
            except Exception as e2:
                logger.error(f"All loading methods failed: {str(e2)}")
                raise
    
    def _load_large_file(self, file_path: str, start_time: float = 0.0,
                        end_time: Optional[float] = None) -> np.ndarray:
        """
        Memory-efficient loading for large audio files.
        Uses temporary files and streaming where possible.
        
        Args:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds (or None for full file)
            
        Returns:
            numpy.ndarray: Audio data
        """
        try:
            # Get duration without loading full file
            duration = librosa.get_duration(path=file_path)
            logger.info(f"Full file duration: {duration:.2f} seconds")
            
            # Calculate effective duration
            if end_time is None:
                end_time = duration
                
            effective_duration = end_time - start_time
            logger.info(f"Loading segment: {start_time:.2f}s - {end_time:.2f}s "
                        f"({effective_duration:.2f}s)")
            
            # For very large files or long durations, use chunked loading
            if effective_duration > 600:  # More than 10 minutes
                return self._chunked_load(file_path, start_time, end_time)
            
            # For moderately large files, use memory-mapped loading if possible
            if self.use_memmap:
                try:
                    return self._memmap_load(file_path, start_time, end_time)
                except Exception as e:
                    logger.warning(f"Memory-mapped loading failed: {str(e)}, falling back to standard loading")
            
            # For moderately large files, use standard loading with immediate GC
            y, sr = librosa.load(
                file_path, 
                sr=self.sample_rate,
                offset=start_time,
                duration=effective_duration,
                mono=True
            )
            
            # Check if audio is empty
            if len(y) == 0:
                logger.warning(f"Empty audio loaded from file: {file_path}")
                raise ValueError(f"File {file_path} loaded with 0 samples. Check if file exists and is not corrupted.")
            
            if self.normalize:
                y = librosa.util.normalize(y)
                
            # Force garbage collection
            gc.collect()
            
            return y
            
        except Exception as e:
            logger.error(f"Error in memory-efficient loading: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _memmap_load(self, file_path: str, start_time: float = 0.0, 
                    end_time: Optional[float] = None) -> np.ndarray:
        """
        Load audio using memory mapping for more efficient memory usage.
        
        Args:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds (or None for full file)
            
        Returns:
            numpy.ndarray: Audio data
        """
        # Create a temp file for the resampled audio if needed
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            # Get the info without loading the audio
            info = sf.info(file_path)
            sr = info.samplerate
            
            # If the sample rate matches and we want the whole file, we can use direct memory mapping
            if sr == self.sample_rate and start_time == 0.0 and (end_time is None or end_time >= info.duration):
                logger.info(f"Using direct memory mapping for audio file")
                
                # Use soundfile to create a memory-mapped array
                with sf.SoundFile(file_path) as sound_file:
                    # Calculate frames and channels
                    frames = sound_file.frames
                    channels = sound_file.channels
                    
                    # Create memory map
                    memmap = np.memmap(temp_path, dtype='float32', mode='w+', 
                                      shape=(frames, channels))
                    
                    # Fill the memory map
                    chunk_size = min(16384, frames)
                    progress = track_progress(f"memmap_{os.path.basename(file_path)}", 
                                             frames // chunk_size + 1, 
                                             "Memory mapping audio")
                    
                    for i in range(0, frames, chunk_size):
                        sound_file.seek(i)
                        chunk_frames = min(chunk_size, frames - i)
                        chunk = sound_file.read(chunk_frames, dtype='float32')
                        memmap[i:i+chunk_frames] = chunk
                        update_progress(f"memmap_{os.path.basename(file_path)}")
                    
                    # Flush to disk
                    memmap.flush()
                    
                    # Convert to mono if needed
                    if channels > 1:
                        result = np.mean(memmap, axis=1)
                    else:
                        result = memmap[:, 0] if channels == 1 else memmap
                    
                    # Create a copy to avoid issues with the memmap being closed
                    result = np.array(result)
                    
                    # Normalize if needed
                    if self.normalize:
                        result = librosa.util.normalize(result)
                    
                    complete_progress(f"memmap_{os.path.basename(file_path)}")
                    return result
            else:
                # We need to resample or extract a portion, so use a hybrid approach
                logger.info(f"Using hybrid memory mapping approach for audio file")
                
                # Calculate sample positions
                start_frame = int(start_time * sr)
                end_frame = int(end_time * sr) if end_time is not None else None
                
                with sf.SoundFile(file_path) as sound_file:
                    # Calculate frames to read
                    if end_frame is not None:
                        frames_to_read = end_frame - start_frame
                    else:
                        frames_to_read = sound_file.frames - start_frame
                    
                    # Seek to start position
                    sound_file.seek(start_frame)
                    
                    # Create a memory-mapped file for intermediate storage
                    channels = sound_file.channels
                    memmap = np.memmap(temp_path, dtype='float32', mode='w+', 
                                      shape=(frames_to_read, channels))
                    
                    # Fill the memory map in chunks
                    chunk_size = min(16384, frames_to_read)
                    progress = track_progress(f"memmap_{os.path.basename(file_path)}", 
                                             frames_to_read // chunk_size + 1, 
                                             "Memory mapping audio segment")
                    
                    for i in range(0, frames_to_read, chunk_size):
                        chunk_frames = min(chunk_size, frames_to_read - i)
                        chunk = sound_file.read(chunk_frames, dtype='float32')
                        memmap[i:i+chunk_frames] = chunk
                        update_progress(f"memmap_{os.path.basename(file_path)}")
                    
                    # Flush to disk
                    memmap.flush()
                    
                    # Convert to mono if needed
                    if channels > 1:
                        audio = np.mean(memmap, axis=1)
                    else:
                        audio = memmap[:, 0] if channels == 1 else memmap
                    
                    # Resample if needed
                    if sr != self.sample_rate:
                        logger.info(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    
                    # Normalize if needed
                    if self.normalize:
                        audio = librosa.util.normalize(audio)
                    
                    # Create a copy to avoid issues with the memmap being closed
                    result = np.array(audio)
                    
                    complete_progress(f"memmap_{os.path.basename(file_path)}")
                    return result
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    def _chunked_load(self, file_path: str, start_time: float = 0.0,
                     end_time: Optional[float] = None) -> np.ndarray:
        """
        Load audio in chunks for very large files.
        
        Args:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds (or None for full file)
            
        Returns:
            numpy.ndarray: Audio data
        """
        if end_time is None:
            end_time = librosa.get_duration(path=file_path)
            
        # Calculate duration
        duration = end_time - start_time
        
        # Determine optimal chunk size based on available memory
        mem_info = psutil.virtual_memory()
        available_memory = mem_info.available * 0.5  # Use 50% of available memory
        
        # Estimate memory needed per second of audio
        # 1 second of 44.1kHz mono audio is about 176KB
        memory_per_second = 176 * 1024
        
        # Calculate optimal chunk size
        max_chunk_seconds = available_memory / memory_per_second
        chunk_seconds = min(max_chunk_seconds, 60)  # Cap at 60 seconds per chunk
        
        # Calculate number of chunks
        num_chunks = int(np.ceil(duration / chunk_seconds))
        logger.info(f"Loading file in {num_chunks} chunks ({chunk_seconds:.1f}s each)")
        
        # Initialize output array
        chunk_samples = int(chunk_seconds * self.sample_rate)
        output = np.zeros(int(duration * self.sample_rate), dtype=np.float32)
        actual_samples = 0
        
        # Create progress tracker
        progress = track_progress(f"chunked_load_{os.path.basename(file_path)}", 
                                 num_chunks, "Loading audio in chunks")
        
        # Process file in chunks
        for i in range(num_chunks):
            chunk_start = start_time + (i * chunk_seconds)
            chunk_end = min(end_time, chunk_start + chunk_seconds)
            
            logger.info(f"Loading chunk {i+1}/{num_chunks}: "
                       f"{chunk_start:.2f}s - {chunk_end:.2f}s")
            
            # Load chunk
            y_chunk, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                offset=chunk_start,
                duration=chunk_end - chunk_start,
                mono=True
            )
            
            # Store in output array
            chunk_samples = len(y_chunk)
            if actual_samples + chunk_samples > len(output):
                # Expand output array if needed
                output = np.resize(output, actual_samples + chunk_samples)
                
            output[actual_samples:actual_samples + chunk_samples] = y_chunk
            actual_samples += chunk_samples
            
            # Update progress
            update_progress(f"chunked_load_{os.path.basename(file_path)}")
            
            # Force garbage collection after each chunk
            del y_chunk
            gc.collect()
        
        # Trim output to actual size
        output = output[:actual_samples]
        
        # Check if trimmed output is empty
        if len(output) == 0:
            logger.warning(f"Empty audio loaded from file after chunking")
            raise ValueError(f"Audio file loaded with 0 samples after chunking. Check if file exists and is not corrupted.")
        
        # Normalize if needed
        if self.normalize:
            output = librosa.util.normalize(output)
        
        # Complete progress tracking
        complete_progress(f"chunked_load_{os.path.basename(file_path)}")
        
        return output
    
    def process_segments(self, audio: np.ndarray, segment_length_ms: int = 1000, 
                         hop_length_ms: int = 500) -> List[np.ndarray]:
        """
        Split audio into overlapping segments for efficient processing.
        
        Args:
            audio: Audio data
            segment_length_ms: Segment length in milliseconds
            hop_length_ms: Hop length in milliseconds
            
        Returns:
            List of audio segments
        """
        samples_per_ms = self.sample_rate / 1000
        segment_length = int(segment_length_ms * samples_per_ms)
        hop_length = int(hop_length_ms * samples_per_ms)
        
        # Calculate number of segments
        num_segments = 1 + int((len(audio) - segment_length) / hop_length)
        logger.info(f"Splitting audio into {num_segments} segments "
                   f"(length: {segment_length_ms}ms, hop: {hop_length_ms}ms)")
        
        # For large files, use a more memory-efficient approach
        if len(audio) > 10 * self.sample_rate:  # More than 10 seconds
            # Process segments in batches
            return self._batch_process_segments(audio, segment_length, hop_length, num_segments)
        
        # For smaller files, use standard approach
        segments = []
        for i in range(num_segments):
            start = i * hop_length
            end = start + segment_length
            
            # Ensure we don't go beyond audio length
            if end <= len(audio):
                segments.append(audio[start:end])
        
        return segments
    
    def _batch_process_segments(self, audio: np.ndarray, segment_length: int, 
                               hop_length: int, num_segments: int) -> List[np.ndarray]:
        """
        Process segments in batches to reduce memory usage.
        
        Args:
            audio: Audio data
            segment_length: Length of each segment in samples
            hop_length: Hop length in samples
            num_segments: Number of segments to extract
            
        Returns:
            List of audio segments
        """
        batch_size = min(100, num_segments)  # Process 100 segments at a time
        num_batches = (num_segments + batch_size - 1) // batch_size
        
        segments = []
        progress = track_progress("segment_extraction", num_batches, "Extracting audio segments")
        
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min(start_idx + batch_size, num_segments)
            
            batch_segments = []
            for i in range(start_idx, end_idx):
                start = i * hop_length
                end = start + segment_length
                
                # Ensure we don't go beyond audio length
                if end <= len(audio):
                    batch_segments.append(audio[start:end])
            
            segments.extend(batch_segments)
            update_progress("segment_extraction")
        
        complete_progress("segment_extraction")
        return segments
    
    @cached_result(ttl=3600)  # Cache for 1 hour
    def compute_features(self, audio: np.ndarray, feature_type: str = 'mfcc', 
                         **kwargs) -> np.ndarray:
        """
        Compute audio features with optimized memory usage.
        
        Args:
            audio: Audio data
            feature_type: Type of feature to compute ('mfcc', 'mel', 'chroma', etc.)
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            numpy.ndarray: Extracted features
        """
        logger.info(f"Computing {feature_type} features for audio with {len(audio)} samples")
        
        # For long audio, use chunked computation
        if len(audio) > 30 * self.sample_rate:  # More than 30 seconds
            return self._chunked_feature_extraction(audio, feature_type, **kwargs)
        
        try:
            if feature_type == 'mfcc':
                n_mfcc = kwargs.get('n_mfcc', 13)
                features = librosa.feature.mfcc(
                    y=audio, 
                    sr=self.sample_rate, 
                    n_mfcc=n_mfcc,
                    **{k: v for k, v in kwargs.items() if k != 'n_mfcc'}
                )
                
            elif feature_type == 'mel':
                n_mels = kwargs.get('n_mels', 128)
                features = librosa.feature.melspectrogram(
                    y=audio, 
                    sr=self.sample_rate,
                    n_mels=n_mels, 
                    **{k: v for k, v in kwargs.items() if k != 'n_mels'}
                )
                
            elif feature_type == 'chroma':
                n_chroma = kwargs.get('n_chroma', 12)
                features = librosa.feature.chroma_stft(
                    y=audio, 
                    sr=self.sample_rate,
                    n_chroma=n_chroma,
                    **{k: v for k, v in kwargs.items() if k != 'n_chroma'}
                )
                
            elif feature_type == 'spectral_contrast':
                n_bands = kwargs.get('n_bands', 6)
                features = librosa.feature.spectral_contrast(
                    y=audio,
                    sr=self.sample_rate,
                    n_bands=n_bands,
                    **{k: v for k, v in kwargs.items() if k != 'n_bands'}
                )
                
            elif feature_type == 'tonnetz':
                features = librosa.feature.tonnetz(
                    y=audio,
                    sr=self.sample_rate,
                    **kwargs
                )
                
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
                
            return features
            
        except Exception as e:
            logger.error(f"Error computing {feature_type} features: {str(e)}")
            raise
    
    def _chunked_feature_extraction(self, audio: np.ndarray, feature_type: str, 
                                   **kwargs) -> np.ndarray:
        """
        Compute features in chunks for memory efficiency.
        
        Args:
            audio: Audio data
            feature_type: Type of feature to compute
            **kwargs: Additional parameters for feature extraction
            
        Returns:
            numpy.ndarray: Extracted features
        """
        logger.info(f"Using chunked computation for {feature_type} features")
        
        # Compute features for first chunk to determine feature dimensions
        mem_info = psutil.virtual_memory()
        available_memory = mem_info.available * 0.5  # Use 50% of available memory
        
        # Adjust chunk size based on available memory and feature type
        if feature_type in ['mel', 'spectrogram']:
            # These features use more memory
            chunk_seconds = 10
        else:
            chunk_seconds = 30
            
        chunk_size = int(chunk_seconds * self.sample_rate)
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing audio in {num_chunks} chunks ({chunk_seconds}s each)")
        
        # Process first chunk to determine feature dimensions
        first_chunk = audio[:min(chunk_size, len(audio))]
        first_features = self.compute_features(first_chunk, feature_type, **kwargs)
        
        # Get feature dimension
        feature_dim, time_dim = first_features.shape
        
        # Initialize output features array
        estimated_time_dim = int(time_dim * (len(audio) / len(first_chunk)))
        all_features = np.zeros((feature_dim, estimated_time_dim), dtype=np.float32)
        
        # Copy first chunk features
        all_features[:, :time_dim] = first_features
        current_time_idx = time_dim
        
        # Create progress tracker
        progress = track_progress(f"feature_extraction_{feature_type}", 
                                 num_chunks, f"Computing {feature_type} features")
        update_progress(f"feature_extraction_{feature_type}")  # Mark first chunk as complete
        
        # Process remaining chunks
        for i in range(1, num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(audio))
            
            logger.info(f"Processing chunk {i+1}/{num_chunks}")
            
            chunk = audio[start_idx:end_idx]
            chunk_features = self.compute_features(chunk, feature_type, **kwargs)
            
            # Append to output
            _, chunk_time_dim = chunk_features.shape
            if current_time_idx + chunk_time_dim > all_features.shape[1]:
                # Expand if needed
                all_features = np.hstack((
                    all_features, 
                    np.zeros((feature_dim, chunk_time_dim), dtype=np.float32)
                ))
                
            all_features[:, current_time_idx:current_time_idx+chunk_time_dim] = chunk_features
            current_time_idx += chunk_time_dim
            
            # Update progress
            update_progress(f"feature_extraction_{feature_type}")
            
            # Clean up
            del chunk, chunk_features
            gc.collect()
        
        # Trim to actual size
        all_features = all_features[:, :current_time_idx]
        
        # Complete progress
        complete_progress(f"feature_extraction_{feature_type}")
        
        return all_features
    
    def batch_process(self, audio_list: List[np.ndarray], 
                     process_func: Callable[[np.ndarray], Any],
                     batch_size: int = 16,
                     show_progress: bool = True) -> List[Any]:
        """
        Process multiple audio arrays in batches.
        
        Args:
            audio_list: List of audio arrays
            process_func: Function to apply to each audio array
            batch_size: Number of arrays to process in each batch
            show_progress: Whether to show progress
            
        Returns:
            List of results from applying process_func to each array
        """
        results = []
        num_items = len(audio_list)
        num_batches = (num_items + batch_size - 1) // batch_size
        
        if show_progress:
            progress = track_progress("batch_processing", 
                                     num_batches, 
                                     f"Processing {num_items} audio segments")
        
        for i in range(0, num_items, batch_size):
            batch = audio_list[i:i+batch_size]
            
            # Process batch
            batch_results = []
            for audio in batch:
                result = process_func(audio)
                batch_results.append(result)
            
            # Add to results
            results.extend(batch_results)
            
            if show_progress:
                update_progress("batch_processing")
            
            # Free memory
            gc.collect()
        
        if show_progress:
            complete_progress("batch_processing")
            
        return results
    
    def resampled_length(self, original_length: int, original_sr: int) -> int:
        """Calculate resampled length without loading the audio."""
        return int(original_length * (self.sample_rate / original_sr))
    
    def trim_silence(self, audio: np.ndarray, threshold_db: float = -60) -> np.ndarray:
        """Trim silence from audio using memory-efficient approach."""
        # For large arrays, process in chunks
        if len(audio) > self.sample_rate * 60:  # More than 1 minute
            return self._chunked_trim_silence(audio, threshold_db)
            
        # For smaller arrays, use standard approach
        trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db))
        return trimmed
    
    def _chunked_trim_silence(self, audio: np.ndarray, threshold_db: float = -60) -> np.ndarray:
        """
        Trim silence in chunks for large audio files.
        
        Args:
            audio: Audio data
            threshold_db: Threshold in dB
            
        Returns:
            Trimmed audio data
        """
        logger.info(f"Using chunked silence trimming for {len(audio)} samples")
        
        # First, find non-silent segments
        non_silent = librosa.effects.split(audio, top_db=abs(threshold_db))
        
        if len(non_silent) == 0:
            # No non-silent segments found
            return np.zeros(0, dtype=audio.dtype)
            
        # Concatenate non-silent segments
        trimmed_audio = []
        for start, end in non_silent:
            segment = audio[start:end]
            trimmed_audio.append(segment)
            
        # Combine segments
        return np.concatenate(trimmed_audio)
    
    def save_audio(self, audio: np.ndarray, filename: str) -> str:
        """Save audio to file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            # Save file using scipy.io.wavfile
            import scipy.io.wavfile as wavfile
            # Ensure audio is in the right range for int16 (-32768 to 32767)
            if self.normalize or np.max(np.abs(audio)) <= 1.0:
                # If normalized or already in [-1.0, 1.0] range, scale to int16
                audio_int = (audio * 32767).astype(np.int16)
            else:
                # If already in appropriate range, just convert
                audio_int = audio.astype(np.int16)
                
            wavfile.write(filename, self.sample_rate, audio_int)
            logger.info(f"Audio saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving audio file: {str(e)}")
            # Try fallback approach
            try:
                # Use librosa with temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Save using librosa utilities
                y_scaled = audio
                if self.normalize and np.max(np.abs(audio)) > 1.0:
                    y_scaled = librosa.util.normalize(audio)
                
                # Use scipy again as fallback method
                wavfile.write(filename, self.sample_rate, (y_scaled * 32767).astype(np.int16))
                logger.info(f"Audio saved to {filename} using fallback method")
                return filename
            except Exception as e2:
                logger.error(f"All fallback approaches failed: {str(e2)}")
                raise


class BatchProcessor:
    """Process multiple audio files in parallel with optimized memory usage."""
    
    def __init__(self, num_workers: int = 4, max_chunk_size_mb: int = 100, config_path: str = "../configs/default.yaml"):
        """Initialize the batch processor.
        
        Args:
            num_workers: Number of worker processes
            max_chunk_size_mb: Maximum chunk size in MB
            config_path: Path to configuration file
        """
        self.num_workers = min(os.cpu_count() or 4, num_workers)
        self.processor = AudioProcessor(config_path)
        self.max_chunk_size_mb = max_chunk_size_mb
        
        logger.info(f"BatchProcessor initialized with {self.num_workers} workers")

    def process_batch(self, file_paths: List[str], process_func: Optional[Callable] = None) -> List[Any]:
        """Process multiple audio files in parallel.
        
        Args:
            file_paths: List of audio file paths
            process_func: Optional function to apply to each loaded audio
            
        Returns:
            List of processed audio data or results of process_func
        """
        # Create progress tracker
        progress = track_progress("batch_processing", 
                                 len(file_paths), 
                                 f"Processing {len(file_paths)} audio files")
            
        # For small batches, use sequential processing to avoid overhead
        if len(file_paths) <= 2 or self.num_workers <= 1:
            results = []
            for path in file_paths:
                try:
                    audio = self.processor.load(path)
                    if process_func:
                        result = process_func(audio)
                    else:
                        result = audio
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {path}: {str(e)}")
                    results.append(None)
                update_progress("batch_processing")
            
            complete_progress("batch_processing")
            return results
        
        # For larger batches, use parallel processing
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            if process_func:
                # Load and process
                futures = [
                    executor.submit(self._load_and_process, path, process_func) 
                    for path in file_paths
                ]
            else:
                # Just load
                futures = [
                    executor.submit(self._process_file, path) 
                    for path in file_paths
                ]
            
            # Process results as they complete
            results = [None] * len(file_paths)
            for i, future in enumerate(futures):
                try:
                    results[i] = future.result()
                except Exception as e:
                    logger.error(f"Error in worker process: {str(e)}")
                update_progress("batch_processing")
        
        complete_progress("batch_processing")
        return results

    def _process_file(self, file_path: str) -> np.ndarray:
        """Process a single audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            np.ndarray: Processed audio data
        """
        try:
            return self.processor.load(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
    
    def _load_and_process(self, file_path: str, process_func: Callable) -> Any:
        """Load and process a single audio file.
        
        Args:
            file_path: Path to the audio file
            process_func: Function to apply to the loaded audio
            
        Returns:
            Result of process_func
        """
        try:
            audio = self.processor.load(file_path)
            return process_func(audio)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            raise
