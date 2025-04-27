"""
Audio source separation module using Demucs models.

This module provides functionality to separate audio into stems (vocals, drums, bass, other)
using the Demucs model from Facebook Research.
"""

import torch
import numpy as np
import librosa
import logging
from typing import Dict, List, Union, Optional, Tuple
import os
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class StemSeparator:
    """Audio source separation using Demucs."""
    
    def __init__(self, model_name: str = "facebook/demucs", 
                 use_cuda: bool = True, 
                 device: Optional[str] = None,
                 num_stems: int = 4,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the source separator with a Demucs model.
        
        Args:
            model_name: Model name/identifier (default: facebook/demucs)
            use_cuda: Whether to use CUDA if available
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            num_stems: Number of stems to separate (4 or 6)
            api_key: HuggingFace API key for accessing gated models
            config: Additional configuration settings
        """
        self.model_name = model_name
        self.num_stems = num_stems
        self.config = config or {}
        self.api_key = api_key
        
        # Set up device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = device or ('cuda' if self.use_cuda else 'cpu')
        
        # Set the API key environment variable if provided
        if api_key:
            os.environ["HF_API_TOKEN"] = api_key
            logger.info("HuggingFace API token set from configuration")
        
        # Define stem mappings
        if num_stems == 4:
            self.stem_names = ["drums", "bass", "other", "vocals"]
            logger.info("Using 4-stem separation: drums, bass, other, vocals")
        elif num_stems == 6:
            self.stem_names = ["drums", "bass", "other", "vocals", "guitar", "piano"]
            logger.info("Using 6-stem separation: drums, bass, other, vocals, guitar, piano")
        else:
            raise ValueError(f"Invalid num_stems value: {num_stems}. Must be 4 or 6.")
        
        # Load the model lazily to save memory
        self._model = None
    
    def _load_model(self):
        """Lazy-load the Demucs model when needed."""
        if self._model is not None:
            return
            
        logger.info(f"Loading Demucs model: {self.model_name}")
        try:
            # Import dependencies here to avoid loading them if not needed
            from demucs.pretrained import get_model
            from demucs.apply import apply_model
            
            # Determine the specific model to load based on num_stems
            if self.num_stems == 4:
                # Standard 4-stem Demucs model
                model_name = "htdemucs"
            else:
                # 6-stem Demucs model
                model_name = "htdemucs_6s"
                
            # Load the model
            self._model = get_model(model_name)
            self._model.to(self.device)
            
            # Store apply function
            self._apply_model = apply_model
            
            logger.info(f"Model {model_name} loaded successfully on {self.device}")
            
        except ImportError:
            logger.error("Demucs package not installed. Please install with 'pip install demucs'")
            raise ImportError("Demucs package is required but not installed. Install with 'pip install demucs'")
        except Exception as e:
            logger.error(f"Failed to load Demucs model: {str(e)}")
            raise ValueError(f"Failed to load Demucs model: {str(e)}")
    
    def separate(self, audio_path: Optional[str] = None, 
                audio_array: Optional[np.ndarray] = None, 
                sample_rate: Optional[int] = None,
                output_dir: Optional[str] = None,
                save_stems: bool = False,
                shifts: int = 1,
                overlap: float = 0.25,
                split: bool = True) -> Dict[str, np.ndarray]:
        """Separate audio into stems.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            output_dir: Directory to save separated stems if save_stems is True
            save_stems: Whether to save separated stems to disk
            shifts: Number of random shifts for equivariant stabilization
            overlap: Overlap between window splits
            split: Whether to split audio into windows for processing
            
        Returns:
            Dictionary mapping stem names to audio arrays
        """
        # Load the model if not already loaded
        self._load_model()
        
        # Load audio if path is provided
        if audio_path is not None:
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=None, mono=False)
                # Convert to the format expected by Demucs: (channels, samples)
                if audio_array.ndim == 1:
                    audio_array = audio_array[np.newaxis, :]
                if audio_array.shape[0] > 2:  # More than 2 channels
                    audio_array = audio_array[:2]  # Keep only first two channels
                
                logger.info(f"Loaded audio: {audio_path}, " +
                          f"{audio_array.shape[1]/sample_rate:.2f}s @ {sample_rate}Hz, " +
                          f"{audio_array.shape[0]} channels")
                
            except Exception as e:
                logger.error(f"Failed to load audio from {audio_path}: {str(e)}")
                raise ValueError(f"Audio loading error: {str(e)}")
                
        elif audio_array is None or sample_rate is None:
            raise ValueError("Either audio_path or (audio_array and sample_rate) must be provided")
        
        # Convert to proper format for Demucs if needed
        if audio_array.ndim == 1:
            audio_array = audio_array[np.newaxis, :]
        
        # Convert numpy array to tensor
        audio_tensor = torch.tensor(audio_array, device=self.device)
        
        try:
            # Apply the model for separation
            logger.info("Separating audio into stems...")
            separated = self._apply_model(
                self._model, 
                audio_tensor, 
                shifts=shifts, 
                split=split, 
                overlap=overlap
            )
            
            # Convert tensor output to numpy arrays
            result = {}
            for i, name in enumerate(self.stem_names):
                stem_audio = separated[i].cpu().numpy()
                result[name] = stem_audio
                logger.debug(f"Separated stem: {name}, shape: {stem_audio.shape}")
            
            # Save stems if requested
            if save_stems and output_dir:
                self._save_stems(result, sample_rate, output_dir, audio_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Stem separation error: {str(e)}")
            raise ValueError(f"Stem separation failed: {str(e)}")
    
    def _save_stems(self, stems: Dict[str, np.ndarray], sample_rate: int, 
                   output_dir: str, audio_path: Optional[str] = None):
        """Save separated stems to disk.
        
        Args:
            stems: Dictionary of separated stems
            sample_rate: Sample rate to use for saving
            output_dir: Directory to save stems
            audio_path: Original audio path (used for naming)
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine base filename
            if audio_path:
                base_name = os.path.splitext(os.path.basename(audio_path))[0]
            else:
                base_name = f"separated_{int(time.time())}"
            
            # Save each stem
            saved_paths = {}
            for name, audio in stems.items():
                # Ensure audio is in the right format for librosa
                if audio.ndim > 1 and audio.shape[0] <= 2:
                    # Convert from (channels, samples) to (samples, channels)
                    audio = audio.T
                
                # Create filename for the stem
                stem_path = os.path.join(output_dir, f"{base_name}_{name}.wav")
                
                # Save the audio
                librosa.output.write_wav(stem_path, audio, sample_rate)
                saved_paths[name] = stem_path
                logger.info(f"Saved {name} stem to {stem_path}")
            
            # Save metadata
            metadata = {
                "original": audio_path,
                "sample_rate": sample_rate,
                "stems": list(stems.keys()),
                "paths": saved_paths,
                "model": self.model_name
            }
            
            metadata_path = os.path.join(output_dir, f"{base_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved separation metadata to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save stems: {str(e)}") 