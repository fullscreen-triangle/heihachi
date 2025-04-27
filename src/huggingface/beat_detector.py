"""
Beat detection module using specialized beat tracking models.

This module provides beat and downbeat tracking functionality using
transformer-based models like Beat-Transformer.
"""

import torch
import numpy as np
import librosa
import logging
from typing import Dict, List, Union, Optional, Tuple
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class BeatDetector:
    """Beat and downbeat detection using transformer models."""
    
    def __init__(self, model_name: str = "nicolaus625/cmi", 
                 use_cuda: bool = True, 
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the beat detector with a transformer model.
        
        Args:
            model_name: Model name/path (default: nicolaus625/cmi for Beat-Transformer)
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
        
        # Default configuration
        self.sr = self.config.get("sample_rate", 44100)
        self.hop_length = self.config.get("hop_length", 512)
        self.n_mels = self.config.get("n_mels", 128)
        self.patch_size = self.config.get("patch_size", (128, 128))
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the beat detection model."""
        logger.info(f"Loading beat detection model: {self.model_name}")
        try:
            # For Beat-Transformer model
            if "nicolaus625/cmi" in self.model_name:
                # Import transformers here
                from transformers import AutoModelForImageClassification, AutoFeatureExtractor
                
                # Load model and processor
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Beat-Transformer model loaded successfully on {self.device}")
            else:
                # For other beat detection models
                # This could be extended for other models like BEAST when available
                logger.warning(f"Model {self.model_name} not specifically supported. Attempting generic loading.")
                from transformers import AutoModelForImageClassification, AutoFeatureExtractor
                
                self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load beat detection model: {str(e)}")
            self.processor = None
            self.model = None
            raise ValueError(f"Failed to load beat detection model: {str(e)}")
    
    def detect(self, audio_path: Optional[str] = None, 
              audio_array: Optional[np.ndarray] = None, 
              sample_rate: Optional[int] = None,
              return_downbeats: bool = True,
              visualize: bool = False,
              output_path: Optional[str] = None) -> Dict:
        """Detect beats and downbeats in audio.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            return_downbeats: Whether to return downbeat positions
            visualize: Whether to generate visualization
            output_path: Path to save visualization if enabled
            
        Returns:
            Dictionary containing beat and downbeat positions in seconds
        """
        # Load audio if path is provided
        if audio_path is not None:
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=self.sr, mono=True)
                logger.info(f"Loaded audio: {audio_path}, {len(audio_array)/sample_rate:.2f}s @ {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to load audio from {audio_path}: {str(e)}")
                raise ValueError(f"Audio loading error: {str(e)}")
        elif audio_array is None or sample_rate is None:
            raise ValueError("Either audio_path or (audio_array and sample_rate) must be provided")
            
        # Use provided sample rate or default
        self.sr = sample_rate or self.sr
        
        try:
            # Extract mel spectrogram
            mel_spec = self._extract_mel_spectrogram(audio_array)
            
            # Process with the model
            beat_mask, downbeat_mask = self._process_spectrogram(mel_spec)
            
            # Convert masks to time positions
            beat_positions = self._mask_to_positions(beat_mask, len(audio_array), sample_rate)
            downbeat_positions = self._mask_to_positions(downbeat_mask, len(audio_array), sample_rate) if return_downbeats else []
            
            # Calculate tempo
            tempo = self._estimate_tempo(beat_positions)
            
            # Prepare result
            result = {
                "beats": beat_positions,
                "tempo": tempo,
                "sample_rate": sample_rate,
                "audio_length": len(audio_array) / sample_rate
            }
            
            if return_downbeats:
                result["downbeats"] = downbeat_positions
            
            # Generate visualization if requested
            if visualize:
                self._visualize_beats(audio_array, sample_rate, beat_positions, 
                                     downbeat_positions if return_downbeats else None,
                                     output_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Beat detection error: {str(e)}")
            raise ValueError(f"Beat detection failed: {str(e)}")
    
    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio.
        
        Args:
            audio: Audio signal
            
        Returns:
            Mel spectrogram as numpy array
        """
        # Calculate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_mels=self.n_mels, 
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1] range for the model
        mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min())
        
        return mel_spec
    
    def _process_spectrogram(self, mel_spec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Process mel spectrogram through the model to get beat predictions.
        
        Args:
            mel_spec: Mel spectrogram
            
        Returns:
            Tuple of (beat_mask, downbeat_mask)
        """
        # For Beat-Transformer, we need to process in overlapping patches
        # since it expects 128x128 mel patches
        
        # Ensure spectrogram has enough frames
        if mel_spec.shape[1] < self.patch_size[1]:
            # Pad if too short
            pad_width = ((0, 0), (0, self.patch_size[1] - mel_spec.shape[1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
        
        # Initialize output masks
        beat_mask = np.zeros(mel_spec.shape[1])
        downbeat_mask = np.zeros(mel_spec.shape[1])
        
        # Process in overlapping patches
        stride = self.patch_size[1] // 2  # 50% overlap
        
        with torch.no_grad():
            for i in range(0, mel_spec.shape[1] - self.patch_size[1] + 1, stride):
                # Extract patch
                patch = mel_spec[:, i:i+self.patch_size[1]]
                
                # Ensure patch is the right size
                if patch.shape != (self.patch_size[0], self.patch_size[1]):
                    continue
                
                # Convert to batch for model
                patch = np.expand_dims(patch, axis=0)  # Add batch dimension
                
                # Process through model
                inputs = self.processor(images=patch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                
                # Get predictions
                logits = outputs.logits.cpu().numpy()
                
                # Assuming model output: [non-beat, beat, downbeat]
                # Get argmax for each position
                predictions = np.argmax(logits, axis=-1)
                
                # Create local beat and downbeat masks
                local_beat_mask = np.zeros(self.patch_size[1])
                local_downbeat_mask = np.zeros(self.patch_size[1])
                
                for j, pred in enumerate(predictions[0]):
                    if pred == 1:  # Beat
                        local_beat_mask[j] = 1
                    elif pred == 2:  # Downbeat
                        local_beat_mask[j] = 1  # Downbeats are also beats
                        local_downbeat_mask[j] = 1
                
                # Apply to global masks with overlap handling
                # For overlapping regions, take maximum
                beat_mask[i:i+self.patch_size[1]] = np.maximum(
                    beat_mask[i:i+self.patch_size[1]], local_beat_mask)
                downbeat_mask[i:i+self.patch_size[1]] = np.maximum(
                    downbeat_mask[i:i+self.patch_size[1]], local_downbeat_mask)
        
        return beat_mask, downbeat_mask
    
    def _mask_to_positions(self, mask: np.ndarray, audio_length: int, sample_rate: int) -> List[float]:
        """Convert a binary mask to time positions.
        
        Args:
            mask: Binary mask indicating beats/downbeats
            audio_length: Length of audio in samples
            sample_rate: Sample rate of audio
            
        Returns:
            List of time positions in seconds
        """
        # Find indices where mask is 1
        indices = np.where(mask > 0.5)[0]
        
        # Convert frame indices to time positions
        hop_time = self.hop_length / sample_rate
        positions = indices * hop_time
        
        # Filter out positions beyond audio length
        max_time = audio_length / sample_rate
        positions = [p for p in positions if p < max_time]
        
        return positions
    
    def _estimate_tempo(self, beat_positions: List[float]) -> float:
        """Estimate tempo from beat positions.
        
        Args:
            beat_positions: List of beat time positions in seconds
            
        Returns:
            Estimated tempo in BPM
        """
        if len(beat_positions) < 2:
            return 0.0
            
        # Calculate beat intervals
        intervals = np.diff(beat_positions)
        
        # Remove outliers (e.g., skipped beats)
        intervals = intervals[intervals < 2.0]  # Assume no beats slower than 30 BPM
        
        if len(intervals) == 0:
            return 0.0
            
        # Calculate median interval for robustness
        median_interval = np.median(intervals)
        
        # Convert to BPM
        tempo = 60.0 / median_interval
        
        return tempo
    
    def _visualize_beats(self, audio: np.ndarray, sample_rate: int, 
                        beat_positions: List[float], downbeat_positions: Optional[List[float]] = None,
                        output_path: Optional[str] = None):
        """Visualize detected beats on audio waveform.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            beat_positions: Beat positions in seconds
            downbeat_positions: Downbeat positions in seconds
            output_path: Path to save visualization
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot waveform
            times = np.arange(len(audio)) / sample_rate
            plt.plot(times, audio, alpha=0.5, color='blue')
            
            # Plot beats
            beat_y = np.zeros_like(beat_positions)
            plt.scatter(beat_positions, beat_y, color='green', marker='|', s=100, label='Beats')
            
            # Plot downbeats if available
            if downbeat_positions and len(downbeat_positions) > 0:
                downbeat_y = np.zeros_like(downbeat_positions)
                plt.scatter(downbeat_positions, downbeat_y, color='red', marker='|', s=200, label='Downbeats')
            
            # Add labels and title
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Beat Detection - Tempo: {self._estimate_tempo(beat_positions):.1f} BPM')
            plt.legend()
            plt.tight_layout()
            
            # Save or show
            if output_path:
                plt.savefig(output_path)
                logger.info(f"Visualization saved to {output_path}")
            else:
                plt.show()
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {str(e)}")
            # Continue without visualization
            pass 