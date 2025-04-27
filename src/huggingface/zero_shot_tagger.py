"""
Zero-shot audio tagging module using OpenJMLA models.

This module provides functionality for zero-shot audio tagging
using the OpenJMLA model from UniMus.
"""

import torch
import numpy as np
import librosa
import logging
from typing import Dict, List, Union, Optional, Tuple
import os
import json
from pathlib import Path
from transformers import AutoModelForAudioClassification, AutoProcessor

logger = logging.getLogger(__name__)

class ZeroShotTagger:
    """Zero-shot audio tagging using OpenJMLA models."""
    
    def __init__(self, model_name: str = "UniMus/OpenJMLA", 
                 use_cuda: bool = True,
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the zero-shot tagger with an OpenJMLA model.
        
        Args:
            model_name: Model name/identifier (default: UniMus/OpenJMLA)
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
        
        # Load the model and processor
        self._load_model()
    
    def _load_model(self):
        """Load the OpenJMLA model and processor."""
        logger.info(f"Loading zero-shot tagging model: {self.model_name}")
        try:
            # For OpenJMLA model
            if "UniMus/OpenJMLA" in self.model_name:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Zero-shot tagging model loaded successfully on {self.device}")
            else:
                # For other zero-shot audio tagging models
                logger.warning(f"Model {self.model_name} not specifically supported as zero-shot tagger. Attempting generic loading.")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            logger.error(f"Failed to load zero-shot tagging model: {str(e)}")
            raise ValueError(f"Failed to load zero-shot tagging model: {str(e)}")
    
    def tag(self, audio_path: Optional[str] = None, 
           audio_array: Optional[np.ndarray] = None, 
           sample_rate: Optional[int] = None,
           tags: Optional[List[str]] = None,
           top_k: int = 10,
           threshold: float = 0.5) -> Dict:
        """Tag audio with zero-shot classification.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            tags: Custom list of tags to score against. If None, uses default tags.
            top_k: Number of top tags to return
            threshold: Confidence threshold for including tags
            
        Returns:
            Dictionary with tag scores and metadata
        """
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
        
        # Default genres/tags if none provided
        if tags is None:
            tags = self._get_default_tags()
        
        # Process audio with zero-shot tagging
        try:
            inputs = self.processor(
                audio_array, 
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # For OpenJMLA, we need to separately score each tag
                tag_scores = {}
                
                for tag in tags:
                    # Format the tag for the model
                    text_inputs = self.processor.tokenizer(
                        tag,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Run inference with the audio and text
                    outputs = self.model(**inputs, **text_inputs)
                    
                    # Extract the relevance score
                    logits = outputs.logits.cpu().numpy().squeeze()
                    score = float(torch.sigmoid(outputs.logits).cpu().numpy().squeeze())
                    
                    tag_scores[tag] = score
            
            # Filter and sort results
            filtered_tags = {tag: score for tag, score in tag_scores.items() if score >= threshold}
            sorted_tags = sorted(filtered_tags.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # Prepare results
            results = {
                "tags": [{"tag": tag, "score": score} for tag, score in sorted_tags],
                "model": self.model_name
            }
            
            if audio_path:
                results["audio_file"] = audio_path
            
            return results
            
        except Exception as e:
            logger.error(f"Zero-shot tagging error: {str(e)}")
            raise ValueError(f"Zero-shot tagging failed: {str(e)}")
    
    def _get_default_tags(self) -> List[str]:
        """Get default tags for zero-shot classification."""
        # Basic electronic music genre tags
        genre_tags = [
            "techno", "house", "ambient", "drum and bass", "dubstep", 
            "trance", "breakbeat", "electro", "acid", "chillout",
            "downtempo", "minimal", "grime", "garage", "jungle",
            "IDM", "industrial", "glitch", "hardstyle", "hardcore",
            "dub", "footwork", "trap", "synthwave", "vaporwave"
        ]
        
        # Sound characteristic tags
        characteristic_tags = [
            "distorted", "reverb", "delay", "synthesizer", "sampled",
            "bass-heavy", "bright", "dark", "atmospheric", "noisy",
            "melodic", "rhythmic", "percussive", "repetitive", "layered",
            "textured", "clean", "lo-fi", "hi-fi", "compressed"
        ]
        
        # Musical attribute tags
        attribute_tags = [
            "fast", "slow", "uplifting", "melancholic", "aggressive",
            "relaxing", "energetic", "hypnotic", "groovy", "dreamy",
            "experimental", "minimal", "complex", "simple", "dissonant",
            "harmonic", "vocal", "instrumental", "acoustic", "electronic"
        ]
        
        # Production technique tags
        technique_tags = [
            "side-chain compression", "filter sweep", "chopped vocals", "granular synthesis",
            "time-stretching", "pitch-shifting", "resampling", "looping", "saturation",
            "resonance", "modulation", "arpeggiation", "sequenced", "phaser",
            "flanger", "chorus", "distortion", "bit-crushing", "automation", "mastering"
        ]
        
        # Combine all categories
        return genre_tags + characteristic_tags + attribute_tags + technique_tags
    
    def tag_with_custom_categories(self, audio_path: str, 
                                 categories: Dict[str, List[str]],
                                 threshold: float = 0.5) -> Dict:
        """Tag audio with custom categories of tags.
        
        Args:
            audio_path: Path to audio file
            categories: Dictionary mapping category names to lists of tags
            threshold: Confidence threshold for including tags
            
        Returns:
            Dictionary with tags organized by category
        """
        # Load audio
        audio_array, sample_rate = librosa.load(audio_path, sr=None)
        
        # Prepare results
        results = {
            "audio_file": audio_path,
            "model": self.model_name,
            "categories": {}
        }
        
        # Process each category
        for category_name, tags in categories.items():
            # Get tag scores for this category
            category_scores = self.tag(
                audio_array=audio_array,
                sample_rate=sample_rate,
                tags=tags,
                threshold=threshold,
                top_k=len(tags)  # Return all tags above threshold
            )
            
            # Add to results
            results["categories"][category_name] = category_scores["tags"]
        
        return results 