from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy import signal
import os
import logging
import librosa
from typing import Optional, Tuple, List, Any, Union
import gc
import tempfile

from utils.config import ConfigManager
from utils.logging_utils import get_logger

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
        
        # Optimize loading strategy based on file size
        if file_size_mb > self.max_file_size_mb:
            logger.info(f"Large file detected, using memory-efficient loading")
            return self._load_large_file(file_path, start_time, end_time)
        
        # For smaller files, use standard loading
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
            logger.error(f"Error loading audio file: {str(e)}")
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
            raise
    
    def _chunked_load(self, file_path: str, start_time: float, 
                     end_time: float) -> np.ndarray:
        """
        Load an audio file in chunks to minimize memory usage.
        
        Args:
            file_path: Path to the audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            numpy.ndarray: Audio data
        """
        logger.info("Using chunked loading for very large file")
        
        # Calculate total duration and expected size
        duration = end_time - start_time
        chunk_duration = 60  # 1-minute chunks
        num_chunks = int(np.ceil(duration / chunk_duration))
        
        logger.info(f"Splitting into {num_chunks} chunks of {chunk_duration}s each")
        
        # Pre-allocate output array (approximate size)
        expected_samples = int(duration * self.sample_rate)
        output = np.zeros(expected_samples, dtype=np.float32)
        actual_samples = 0
        
        # Process file in chunks
        for i in range(num_chunks):
            chunk_start = start_time + (i * chunk_duration)
            chunk_end = min(end_time, chunk_start + chunk_duration)
            
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
        
        segments = []
        for i in range(num_segments):
            start = i * hop_length
            end = start + segment_length
            
            # Ensure we don't go beyond audio length
            if end <= len(audio):
                segments.append(audio[start:end])
        
        return segments
    
    def compute_features(self, audio: np.ndarray, feature_type: str = 'mfcc', 
                         **kwargs) -> np.ndarray:
        """
        Compute audio features with memory optimization.
        
        Args:
            audio: Audio data
            feature_type: Type of feature to compute ('mfcc', 'mel', 'chroma', etc.)
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            numpy.ndarray: Feature matrix
        """
        if len(audio) > self.sample_rate * 300:  # More than 5 minutes
            logger.info(f"Computing {feature_type} features on large audio using chunked processing")
            return self._chunked_feature_extraction(audio, feature_type, **kwargs)
            
        logger.info(f"Computing {feature_type} features")
        
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
        Compute features in chunks to reduce memory usage.
        
        Args:
            audio: Audio data
            feature_type: Type of feature to compute
            **kwargs: Additional arguments for feature extraction
            
        Returns:
            numpy.ndarray: Combined feature matrix
        """
        # Split audio into manageable chunks (e.g., 30 seconds)
        chunk_size = self.sample_rate * 30
        num_chunks = int(np.ceil(len(audio) / chunk_size))
        
        logger.info(f"Processing {feature_type} features in {num_chunks} chunks")
        
        # Process first chunk to get feature dimensions
        first_chunk = audio[:min(chunk_size, len(audio))]
        first_features = self.compute_features(first_chunk, feature_type, **kwargs)
        
        # Initialize output based on first chunk's shape
        feature_dim, time_dim = first_features.shape
        expected_time_dim = int(np.ceil(time_dim * (len(audio) / len(first_chunk))))
        all_features = np.zeros((feature_dim, expected_time_dim), dtype=np.float32)
        
        # Copy first chunk results
        all_features[:, :time_dim] = first_features
        current_time_idx = time_dim
        
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
            
            # Clean up
            del chunk, chunk_features
            gc.collect()
        
        # Trim to actual size
        all_features = all_features[:, :current_time_idx]
        
        return all_features
        
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
        
    def _chunked_trim_silence(self, audio: np.ndarray, threshold_db: float) -> np.ndarray:
        """Trim silence in chunks for large files."""
        logger.info("Using chunked silence trimming for large audio")
        
        # Find non-silent regions
        non_silent = librosa.effects.split(audio, top_db=abs(threshold_db))
        
        # Allocate output array (maximum possible size is original audio)
        output = np.zeros_like(audio)
        current_idx = 0
        
        # Copy non-silent regions
        for start, end in non_silent:
            region_length = end - start
            output[current_idx:current_idx+region_length] = audio[start:end]
            current_idx += region_length
            
        # Trim output to actual size
        return output[:current_idx]
        
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
                import tempfile
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
    def __init__(self, num_workers: int = 4, max_chunk_size_mb: int = 100):
        self.num_workers = num_workers
        self.processor = AudioProcessor(max_chunk_size_mb)

    def process_batch(self, file_paths: list) -> list:
        """Process multiple audio files in parallel.
        
        Args:
            file_paths (list): List of audio file paths
            
        Returns:
            list: List of processed audio data
        """
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_file, path) for path in file_paths]
            results = [f.result() for f in futures]
        return results

    def _process_file(self, file_path: str) -> np.ndarray:
        """Process a single audio file.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Processed audio data
        """
        audio = self.processor.load_audio(file_path)
        return self.processor.process_audio(audio)
