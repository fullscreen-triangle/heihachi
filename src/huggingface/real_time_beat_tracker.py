"""
Real-time beat tracking module using BEAST (Beat Event-prediction with Adaptive Spectral Tracking) models.

This module provides functionality for real-time beat tracking in audio streams
with low latency using pre-trained BEAST models.
"""

import torch
import numpy as np
import librosa
import logging
import time
import threading
import queue
import sounddevice as sd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple, Callable
import os
from collections import deque
from transformers import AutoModelForAudioFrameClassification, AutoFeatureExtractor

logger = logging.getLogger(__name__)

class RealTimeBeatTracker:
    """Real-time beat tracking using BEAST models."""
    
    def __init__(self, model_name: str = "beast-team/beast-dione", 
                 use_cuda: bool = True,
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the real-time beat tracker.
        
        Args:
            model_name: Model name/identifier (default: beast-team/beast-dione)
            use_cuda: Whether to use CUDA if available
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            api_key: HuggingFace API key for accessing gated models
            config: Additional configuration settings
        """
        self.model_name = model_name
        self.config = config or {}
        self.api_key = api_key
        
        # Set up device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = device or ('cuda' if self.use_cuda else 'cpu')
        
        # Set the API key environment variable if provided
        if api_key:
            os.environ["HF_API_TOKEN"] = api_key
            logger.info("HuggingFace API token set from configuration")
        
        # Configuration 
        self.sample_rate = self.config.get("sample_rate", 44100)
        self.buffer_size = self.config.get("buffer_size", 2)  # Buffer size in seconds
        self.hop_size = self.config.get("hop_size", 0.01)  # Hop size in seconds
        self.activation_threshold = self.config.get("activation_threshold", 0.5)
        self.min_interval = self.config.get("min_interval", 0.20)  # Minimum interval between beats in seconds
        self.history_size = self.config.get("history_size", 8)  # Number of beats to keep for tempo estimation
        
        # Initialize state variables
        self.is_running = False
        self.audio_buffer = np.array([])
        self.beat_times = []
        self.last_beat_time = 0
        self.tempo = 120.0
        self.beat_interval_history = deque(maxlen=self.history_size)
        
        # Processing thread
        self.processing_thread = None
        self.audio_queue = queue.Queue()
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the BEAST model for beat tracking."""
        logger.info(f"Loading beat tracking model: {self.model_name}")
        try:
            # Load feature extractor and model
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioFrameClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Get required sample rate for model
            self.model_sr = self.feature_extractor.sampling_rate
            if self.model_sr != self.sample_rate:
                logger.info(f"Model requires {self.model_sr}Hz, but initialized with {self.sample_rate}Hz. Will resample.")
            
            logger.info(f"Beat tracking model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load beat tracking model: {str(e)}")
            raise ValueError(f"Failed to load beat tracking model: {str(e)}")
    
    def start(self):
        """Start real-time beat tracking."""
        if self.is_running:
            logger.warning("Beat tracker is already running.")
            return
        
        logger.info("Starting real-time beat tracker")
        self.is_running = True
        self.audio_buffer = np.array([])
        self.beat_times = []
        self.last_beat_time = 0
        self.tempo = 120.0
        self.beat_interval_history = deque(maxlen=self.history_size)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop(self):
        """Stop real-time beat tracking."""
        if not self.is_running:
            logger.warning("Beat tracker is not running.")
            return
        
        logger.info("Stopping real-time beat tracker")
        self.is_running = False
        
        # Wait for processing thread to stop
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def add_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int = None):
        """Add a new chunk of audio to the processing queue.
        
        Args:
            audio_chunk: NumPy array of audio samples
            sample_rate: Sample rate of the audio chunk (defaults to self.sample_rate)
        """
        if not self.is_running:
            logger.warning("Beat tracker is not running. Call start() first.")
            return
        
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=self.sample_rate)
        
        # Add to processing queue
        self.audio_queue.put(audio_chunk)
    
    def _processing_loop(self):
        """Main processing loop for the beat tracker thread."""
        logger.info("Beat tracking processing thread started")
        
        while self.is_running:
            try:
                # Get audio from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Add to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                
                # Process buffer if enough data
                buffer_samples = int(self.buffer_size * self.sample_rate)
                if len(self.audio_buffer) >= buffer_samples:
                    # Process the buffer
                    self._process_buffer()
                    
                    # Keep part of the buffer for overlap
                    overlap_samples = int(self.buffer_size * 0.5 * self.sample_rate)
                    self.audio_buffer = self.audio_buffer[-overlap_samples:] if overlap_samples > 0 else np.array([])
            
            except Exception as e:
                logger.error(f"Error in beat tracking thread: {str(e)}")
        
        logger.info("Beat tracking processing thread stopped")
    
    def _process_buffer(self):
        """Process the current audio buffer to detect beats."""
        if len(self.audio_buffer) == 0:
            return
        
        # Resample to model sample rate if needed
        if self.sample_rate != self.model_sr:
            audio = librosa.resample(self.audio_buffer, orig_sr=self.sample_rate, target_sr=self.model_sr)
        else:
            audio = self.audio_buffer
        
        try:
            # Extract features
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=self.model_sr,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get frame-level predictions
            logits = outputs.logits.cpu().numpy()[0, :, 0]  # Assuming binary classification (beat/no beat)
            
            # Convert to probabilities
            probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
            
            # Get time points for each frame
            hop_length = int(self.hop_size * self.model_sr)
            frames = np.arange(len(probabilities))
            frame_times = librosa.frames_to_time(frames, sr=self.model_sr, hop_length=hop_length)
            
            # Find beats in the buffer
            current_time = time.time()  # Current real-time timestamp
            buffer_start_time = current_time - self.buffer_size  # Approximate start time of buffer
            
            for i, (frame_time, prob) in enumerate(zip(frame_times, probabilities)):
                if prob >= self.activation_threshold:
                    # Check if enough time has passed since the last beat
                    if not self.beat_times or (frame_time - (self.beat_times[-1] % self.buffer_size)) > self.min_interval:
                        # Calculate real-time timestamp for this beat
                        beat_time = buffer_start_time + frame_time
                        
                        # Add to beat list
                        self.beat_times.append(beat_time)
                        self.last_beat_time = beat_time
                        
                        # Update beat intervals and estimate tempo
                        if len(self.beat_times) > 1:
                            interval = self.beat_times[-1] - self.beat_times[-2]
                            if 0.2 < interval < 2.0:  # Filter out unreasonable intervals
                                self.beat_interval_history.append(interval)
                                self._update_tempo()
                        
                        logger.debug(f"Beat detected at {beat_time:.3f}s with confidence {prob:.2f}")
                        
                        # Ensure we only get one beat per peak
                        break_count = 0
                        while i + break_count + 1 < len(probabilities) and probabilities[i + break_count + 1] >= self.activation_threshold:
                            break_count += 1
                        i += break_count
        
        except Exception as e:
            logger.error(f"Error processing audio buffer: {str(e)}")
    
    def _update_tempo(self):
        """Update tempo estimate based on recent beat intervals."""
        if not self.beat_interval_history:
            return
        
        # Calculate median beat interval
        median_interval = np.median(self.beat_interval_history)
        
        # Convert to BPM
        if median_interval > 0:
            self.tempo = 60.0 / median_interval
        
        logger.debug(f"Tempo updated to {self.tempo:.1f} BPM")
    
    def get_recent_beats(self, seconds: float = 5.0) -> List[float]:
        """Get recent beat times within the specified time window.
        
        Args:
            seconds: Time window in seconds
            
        Returns:
            List of beat timestamps within the window
        """
        if not self.beat_times:
            return []
        
        current_time = time.time()
        recent_beats = [t for t in self.beat_times if (current_time - t) <= seconds]
        return recent_beats
    
    def get_tempo(self) -> float:
        """Get the current tempo estimate in BPM.
        
        Returns:
            Tempo in beats per minute
        """
        return self.tempo
    
    def process_file(self, audio_path: str, simulate_realtime: bool = False, 
                      chunk_size: float = 0.1) -> Dict:
        """Process a complete audio file and return beat timings.
        
        Args:
            audio_path: Path to audio file
            simulate_realtime: Whether to simulate real-time processing
            chunk_size: Size of audio chunks in seconds (for simulation)
            
        Returns:
            Dictionary with beat times and estimated tempo
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = len(audio) / sr
            
            # Process the file
            if simulate_realtime:
                # Start tracker
                self.start()
                
                # Process in chunks to simulate real-time
                chunk_samples = int(chunk_size * sr)
                for i in range(0, len(audio), chunk_samples):
                    chunk = audio[i:i+chunk_samples]
                    self.add_audio_chunk(chunk, sr)
                    # Slight delay to allow processing
                    time.sleep(chunk_size * 0.5)
                
                # Wait for processing to complete
                time.sleep(0.5)
                
                # Stop tracker
                self.stop()
                
                # Collect results
                beat_times = self.beat_times
                tempo = self.tempo
                
            else:
                # Process the whole file at once
                # Prepare the model
                if self.sample_rate != self.model_sr:
                    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=self.model_sr)
                else:
                    audio_resampled = audio
                
                # Extract features
                inputs = self.feature_extractor(
                    audio_resampled, 
                    sampling_rate=self.model_sr,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Get frame-level predictions
                logits = outputs.logits.cpu().numpy()[0, :, 0]
                probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid
                
                # Get beat frames
                hop_length = int(self.hop_size * self.model_sr)
                frames = np.arange(len(probabilities))
                frame_times = librosa.frames_to_time(frames, sr=self.model_sr, hop_length=hop_length)
                
                # Find beats
                beat_frames = []
                i = 0
                while i < len(probabilities):
                    if probabilities[i] >= self.activation_threshold:
                        if not beat_frames or (frame_times[i] - frame_times[beat_frames[-1]]) > self.min_interval:
                            beat_frames.append(i)
                            
                            # Skip ahead to avoid multiple detections of same beat
                            break_count = 0
                            while i + break_count + 1 < len(probabilities) and probabilities[i + break_count + 1] >= self.activation_threshold:
                                break_count += 1
                            i += break_count
                    i += 1
                
                # Convert frames to times
                beat_times = [frame_times[i] for i in beat_frames]
                
                # Estimate tempo
                if len(beat_times) > 1:
                    intervals = np.diff(beat_times)
                    valid_intervals = intervals[(intervals > 0.2) & (intervals < 2.0)]
                    if len(valid_intervals) > 0:
                        median_interval = np.median(valid_intervals)
                        tempo = 60.0 / median_interval
                    else:
                        tempo = 120.0
                else:
                    tempo = 120.0
            
            # Package results
            result = {
                "file_path": audio_path,
                "duration": duration,
                "beats": beat_times,
                "tempo": float(tempo),
                "num_beats": len(beat_times)
            }
            
            logger.info(f"Detected {len(beat_times)} beats, estimated tempo: {tempo:.1f} BPM")
            return result
            
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return {"error": str(e)}
    
    def register_beat_callback(self, callback: Callable[[float, float], None]):
        """Register a callback function to be called when a beat is detected.
        
        Args:
            callback: Function to call with args (beat_time, confidence)
        """
        self.beat_callback = callback
    
    def run_real_time_demo(self, duration: float = 20.0, visualize: bool = True):
        """Run a real-time demo that captures audio from a microphone and processes it.
        
        Args:
            duration: Duration of the demo in seconds
            visualize: Whether to visualize the waveform and beats
        """
        try:
            # Initialize variables
            self.start()
            chunk_size = 0.05  # 50ms chunks
            chunk_samples = int(chunk_size * self.sample_rate)
            
            # Setup audio capture
            audio_buffer = []
            
            # Setup visualization if requested
            if visualize:
                plt.figure(figsize=(10, 6))
                plt.ion()  # Enable interactive mode
                display_buffer = np.zeros(int(1.5 * self.sample_rate))  # 1.5 seconds of display buffer
                beat_markers = []
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio capture status: {status}")
                
                # Get audio data (first channel only if stereo)
                chunk = indata[:, 0] if indata.shape[1] > 1 else indata[:, 0]
                
                # Add to local buffer for visualization
                audio_buffer.extend(chunk)
                
                # Process the chunk
                self.add_audio_chunk(chunk, self.sample_rate)
            
            # Start audio stream
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=chunk_samples
            )
            
            with stream:
                logger.info(f"Starting real-time demo for {duration} seconds")
                
                start_time = time.time()
                last_beat_count = 0
                
                while time.time() - start_time < duration:
                    # Update visualization
                    if visualize and len(audio_buffer) > 0:
                        # Update display buffer
                        new_samples = min(len(audio_buffer), len(display_buffer))
                        display_buffer = np.roll(display_buffer, -new_samples)
                        display_buffer[-new_samples:] = audio_buffer[:new_samples]
                        audio_buffer = audio_buffer[new_samples:]
                        
                        # Clear and redraw plot
                        plt.clf()
                        plt.plot(display_buffer)
                        
                        # Mark beats
                        current_beats = self.beat_times
                        new_beats = current_beats[last_beat_count:]
                        beat_markers.extend(new_beats)
                        last_beat_count = len(current_beats)
                        
                        # Only show recent beat markers
                        current_time = time.time()
                        recent_markers = [b for b in beat_markers if current_time - b < 1.5]
                        beat_markers = recent_markers
                        
                        for beat_time in beat_markers:
                            beat_age = current_time - beat_time
                            # Convert to display buffer index
                            beat_idx = len(display_buffer) - int(beat_age * self.sample_rate)
                            if 0 <= beat_idx < len(display_buffer):
                                plt.axvline(x=beat_idx, color='r', alpha=max(0.2, 1.0 - beat_age/1.5))
                        
                        # Add tempo information
                        plt.title(f"Real-time Beat Tracking - Tempo: {self.tempo:.1f} BPM")
                        plt.ylabel("Amplitude")
                        plt.xlabel("Sample")
                        plt.tight_layout()
                        plt.pause(0.01)
                    
                    # Sleep to avoid hogging CPU
                    time.sleep(0.01)
                
                logger.info(f"Demo completed. Detected {len(self.beat_times)} beats with tempo {self.tempo:.1f} BPM")
            
            # Stop tracking
            self.stop()
            
            if visualize:
                plt.ioff()  # Turn off interactive mode
                plt.show()
            
            return {
                "beats": self.beat_times,
                "tempo": self.tempo,
                "duration": duration
            }
            
        except Exception as e:
            logger.error(f"Error running real-time demo: {str(e)}")
            self.stop()
            return {"error": str(e)}
    
    def save_beats_to_file(self, beats: List[float], output_path: str, format: str = "csv"):
        """Save beat times to a file in various formats.
        
        Args:
            beats: List of beat times
            output_path: Path to save the output file
            format: Output format (csv, txt, json)
            
        Returns:
            Path to the saved file
        """
        try:
            if format.lower() == "csv":
                with open(output_path, 'w') as f:
                    f.write("beat_time\n")
                    for beat in beats:
                        f.write(f"{beat:.6f}\n")
            
            elif format.lower() == "txt":
                with open(output_path, 'w') as f:
                    for beat in beats:
                        f.write(f"{beat:.6f}\n")
            
            elif format.lower() == "json":
                import json
                with open(output_path, 'w') as f:
                    json.dump({"beats": beats, "tempo": self.tempo}, f, indent=2)
            
            else:
                logger.error(f"Unsupported output format: {format}")
                return {"error": f"Unsupported output format: {format}"}
            
            logger.info(f"Saved {len(beats)} beats to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving beats to file: {str(e)}")
            return {"error": str(e)} 