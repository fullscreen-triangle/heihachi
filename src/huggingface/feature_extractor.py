"""
Feature extraction module using HuggingFace transformer models.

This module provides implementations for various specialized audio feature extraction
models from HuggingFace, focusing on models like BEATs and Whisper that provide
rich embeddings for audio analysis.
"""

import torch
import numpy as np
import librosa
from typing import Dict, List, Union, Optional, Tuple
import logging
from transformers import AutoModel, AutoProcessor, AutoFeatureExtractor
import os

from ..utils.env_loader import load_dotenv, get_api_key

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Audio feature extraction using specialized HuggingFace models."""
    
    def __init__(self, model: str = "microsoft/BEATs-base", api_key: Optional[str] = None,
                 use_cuda: bool = True, device: Optional[str] = None, config: Optional[Dict] = None,
                 load_from_env: bool = True):
        """Initialize the feature extractor with a specified model.
        
        Args:
            model: Model name/path for feature extraction (default: microsoft/BEATs-base)
            api_key: HuggingFace API key for accessing gated models
            use_cuda: Whether to use CUDA if available
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            config: Additional configuration settings
            load_from_env: Whether to load API key from .env file if not explicitly provided
        """
        self.model_name = model
        self.config = config or {}
        
        # Try to load .env file if requested
        if load_from_env:
            load_dotenv()
        
        # Get API key with priority:
        # 1. Explicitly provided api_key parameter
        # 2. Environment variables from .env file (HF_API_TOKEN, HUGGINGFACE_API_KEY)
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = get_api_key(
                key_name="HF_API_TOKEN", 
                fallback_names=["HUGGINGFACE_API_KEY", "HF_TOKEN"]
            )
        
        # Set up device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = device or ('cuda' if self.use_cuda else 'cpu')
        
        # Set the API key environment variable if available
        if self.api_key:
            os.environ["HF_API_TOKEN"] = self.api_key
            logger.info("HuggingFace API token set")
        
        # Load the model and processor
        logger.info(f"Loading feature extraction model: {model}")
        try:
            # For BEATs models
            if "BEATs" in model:
                self.processor = AutoFeatureExtractor.from_pretrained(model)
                self.model = AutoModel.from_pretrained(model)
            # For Whisper models (encoder only)
            elif "whisper" in model:
                self.processor = AutoProcessor.from_pretrained(model)
                self.model = AutoModel.from_pretrained(model)
            else:
                # Default approach for other models
                self.processor = AutoProcessor.from_pretrained(model)
                self.model = AutoModel.from_pretrained(model)
                
            # Move model to the specified device
            self.model = self.model.to(self.device)
            logger.info(f"Model loaded successfully on {self.device}")
            
            # Set eval mode for inference
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            self.processor = None
            self.model = None
            raise ValueError(f"Failed to load model {model}: {str(e)}")
    
    def extract(self, audio_path: Optional[str] = None, 
                audio_array: Optional[np.ndarray] = None, 
                sample_rate: Optional[int] = None,
                return_timestamps: bool = True,
                max_length_seconds: float = 30.0,
                batch_size_seconds: float = 10.0) -> Dict:
        """Extract features from audio using the loaded model.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            return_timestamps: Whether to include timestamps in the output
            max_length_seconds: Maximum audio length to process
            batch_size_seconds: Size of processing batches for long audio
            
        Returns:
            Dictionary containing extracted features and metadata
        """
        if self.model is None or self.processor is None:
            raise ValueError("Model or processor not initialized")
            
        # Load audio if path is provided
        if audio_path is not None:
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=None)
                logger.info(f"Loaded audio: {audio_path}, {len(audio_array)/sample_rate:.2f}s @ {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to load audio from {audio_path}: {str(e)}")
                raise ValueError(f"Audio loading error: {str(e)}")
        elif audio_array is None or sample_rate is None:
            raise ValueError("Either audio_path or (audio_array and sample_rate) must be provided")
        
        # Convert to mono if needed
        if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)
            logger.info("Converted stereo audio to mono")
            
        # Limit audio length if needed
        max_samples = int(max_length_seconds * sample_rate)
        if len(audio_array) > max_samples:
            logger.warning(f"Audio exceeds maximum length of {max_length_seconds}s, truncating")
            audio_array = audio_array[:max_samples]
            
        try:
            # Process in batches if audio is long
            if len(audio_array) > int(batch_size_seconds * sample_rate):
                features = self._extract_in_batches(audio_array, sample_rate, 
                                                  batch_size_seconds, return_timestamps)
            else:
                features = self._extract_single(audio_array, sample_rate, return_timestamps)
                
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            raise ValueError(f"Feature extraction failed: {str(e)}")
            
    def _extract_single(self, audio_array: np.ndarray, sample_rate: int, 
                       return_timestamps: bool) -> Dict:
        """Extract features from a single audio segment.
        
        Args:
            audio_array: Audio data
            sample_rate: Sample rate of the audio
            return_timestamps: Whether to include timestamps
            
        Returns:
            Dictionary of extracted features
        """
        # Process audio through the model
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Extract relevant features based on model type
        if "BEATs" in self.model_name:
            # For BEATs models, extract the hidden states
            features = outputs.extract_features.cpu().numpy()
            frame_features = outputs.last_hidden_state.cpu().numpy()
        elif "whisper" in self.model_name:
            # For Whisper models, extract encoder features
            features = outputs.last_hidden_state.cpu().numpy()
            frame_features = features  # Same as main features for Whisper
        else:
            # Default approach for other models
            features = outputs.last_hidden_state.cpu().numpy()
            frame_features = features
            
        # Create timestamps if requested
        timestamps = None
        if return_timestamps:
            # Calculate approximate timestamps based on model architecture
            if "BEATs" in self.model_name:
                # BEATs uses ~20ms hop size
                hop_time = 0.02  # 20ms
            elif "whisper" in self.model_name:
                # Whisper uses 50Hz sampling (20ms)
                hop_time = 0.02
            else:
                # Default assumption
                hop_time = 0.02
                
            # Generate timestamps for each frame
            num_frames = frame_features.shape[1]
            timestamps = np.arange(num_frames) * hop_time
            
        # Prepare the result
        result = {
            "embedding": features.squeeze(),
            "frame_features": frame_features.squeeze(),
            "sample_rate": sample_rate,
            "model": self.model_name,
            "embedding_dim": features.shape[-1]
        }
        
        if timestamps is not None:
            result["timestamps"] = timestamps
            
        return result
    
    def _extract_in_batches(self, audio_array: np.ndarray, sample_rate: int,
                           batch_size_seconds: float, return_timestamps: bool) -> Dict:
        """Extract features from long audio in batches.
        
        Args:
            audio_array: Audio data
            sample_rate: Sample rate of the audio
            batch_size_seconds: Batch size in seconds
            return_timestamps: Whether to include timestamps
            
        Returns:
            Dictionary of extracted features
        """
        logger.info(f"Processing audio in batches of {batch_size_seconds}s")
        
        batch_samples = int(sample_rate * batch_size_seconds)
        all_features = []
        all_frame_features = []
        all_timestamps = []
        
        # Process each batch
        for i in range(0, len(audio_array), batch_samples):
            batch = audio_array[i:min(i+batch_samples, len(audio_array))]
            logger.debug(f"Processing batch {i//batch_samples + 1}, " +
                        f"samples {i} to {i+len(batch)} ({len(batch)/sample_rate:.2f}s)")
            
            # Extract features for this batch
            batch_result = self._extract_single(batch, sample_rate, return_timestamps)
            
            # Collect the features
            all_features.append(batch_result["embedding"])
            all_frame_features.append(batch_result["frame_features"])
            
            if return_timestamps and "timestamps" in batch_result:
                # Adjust timestamps to account for batch position
                batch_timestamps = batch_result["timestamps"] + (i / sample_rate)
                all_timestamps.append(batch_timestamps)
            
            # Clear GPU cache if using CUDA
            if self.use_cuda:
                torch.cuda.empty_cache()
        
        # Combine features from all batches
        # For global features, take average
        combined_features = np.mean(np.stack(all_features), axis=0)
        # For frame features, concatenate
        combined_frame_features = np.concatenate(all_frame_features, axis=0)
        
        # Combine timestamps if available
        combined_timestamps = None
        if all_timestamps:
            combined_timestamps = np.concatenate(all_timestamps)
            
        # Prepare the result
        result = {
            "embedding": combined_features,
            "frame_features": combined_frame_features,
            "sample_rate": sample_rate,
            "model": self.model_name,
            "embedding_dim": combined_features.shape[-1],
            "processed_in_batches": True,
            "num_batches": len(all_features)
        }
        
        if combined_timestamps is not None:
            result["timestamps"] = combined_timestamps
            
        return result 