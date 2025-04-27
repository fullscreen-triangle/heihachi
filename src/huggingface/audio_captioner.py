"""
Audio captioning module using specialized models.

This module provides functionality for generating textual descriptions
of audio content using specialized audio captioning models.
"""

import torch
import numpy as np
import librosa
import logging
from typing import Dict, List, Union, Optional, Tuple
import os
import json
from pathlib import Path
from transformers import AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)

class AudioCaptioner:
    """Audio captioning for generating textual descriptions of audio."""
    
    def __init__(self, model_name: str = "slseanwu/beats-conformer-bart-audio-captioner", 
                 use_cuda: bool = True,
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the audio captioner with a captioning model.
        
        Args:
            model_name: Model name/identifier (default: slseanwu/beats-conformer-bart-audio-captioner)
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
        
        # Configuration for captioning
        self.max_length = self.config.get("max_length", 50)
        self.num_beams = self.config.get("num_beams", 5)
        self.segment_length = self.config.get("segment_length", 10.0)  # seconds
        self.overlap = self.config.get("overlap", 2.0)  # seconds
        
        # Load the model and processor
        self._load_model()
    
    def _load_model(self):
        """Load the audio captioning model and processor."""
        logger.info(f"Loading audio captioning model: {self.model_name}")
        try:
            # For beats-conformer-bart captioning model
            if "beats-conformer-bart" in self.model_name:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Audio captioning model loaded successfully on {self.device}")
            else:
                # For other audio captioning models
                logger.warning(f"Model {self.model_name} not specifically supported for audio captioning. Attempting generic loading.")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
        except Exception as e:
            logger.error(f"Failed to load audio captioning model: {str(e)}")
            raise ValueError(f"Failed to load audio captioning model: {str(e)}")
    
    def caption(self, audio_path: Optional[str] = None, 
               audio_array: Optional[np.ndarray] = None, 
               sample_rate: Optional[int] = None,
               segment_audio: bool = True,
               return_all_captions: bool = False) -> Dict:
        """Generate captions for audio.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            segment_audio: Whether to segment audio into smaller chunks
            return_all_captions: Whether to return captions for all segments
            
        Returns:
            Dictionary with generated captions and metadata
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
        
        # Process audio to generate captions
        try:
            audio_duration = len(audio_array) / sample_rate
            
            # For short audio, process as a single segment
            if audio_duration <= self.segment_length or not segment_audio:
                caption = self._generate_caption(audio_array, sample_rate)
                
                return {
                    "caption": caption,
                    "duration": audio_duration,
                    "model": self.model_name
                }
            else:
                # For longer audio, segment and caption each part
                segment_captions = self._caption_segments(audio_array, sample_rate)
                
                # Generate summary caption (combining the most relevant segments)
                summary_caption = self._generate_summary(segment_captions)
                
                result = {
                    "caption": summary_caption,
                    "duration": audio_duration,
                    "model": self.model_name
                }
                
                # Include all segment captions if requested
                if return_all_captions:
                    result["segments"] = segment_captions
                
                return result
                
        except Exception as e:
            logger.error(f"Audio captioning error: {str(e)}")
            raise ValueError(f"Audio captioning failed: {str(e)}")
    
    def _generate_caption(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Generate a caption for a single audio segment.
        
        Args:
            audio_array: Audio data
            sample_rate: Sample rate of the audio
            
        Returns:
            Generated caption text
        """
        # Process audio through the model
        inputs = self.processor(
            audios=audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate caption
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True
            )
        
        # Decode the generated text
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def _caption_segments(self, audio_array: np.ndarray, sample_rate: int) -> List[Dict]:
        """Segment audio and caption each segment.
        
        Args:
            audio_array: Audio data
            sample_rate: Sample rate of the audio
            
        Returns:
            List of segment captions with timestamps
        """
        segment_length_samples = int(self.segment_length * sample_rate)
        overlap_samples = int(self.overlap * sample_rate)
        hop_length = segment_length_samples - overlap_samples
        
        # Initialize segments
        segments = []
        
        # Split audio into overlapping segments
        for start in range(0, len(audio_array) - segment_length_samples + 1, hop_length):
            end = start + segment_length_samples
            segment = audio_array[start:end]
            
            # Generate caption for this segment
            caption = self._generate_caption(segment, sample_rate)
            
            # Calculate timestamps
            start_time = start / sample_rate
            end_time = end / sample_rate
            
            segments.append({
                "start_time": start_time,
                "end_time": end_time,
                "caption": caption
            })
        
        # Handle last segment if it would be cut off
        if len(audio_array) > start + segment_length_samples:
            last_segment = audio_array[-segment_length_samples:]
            last_caption = self._generate_caption(last_segment, sample_rate)
            last_start_time = (len(audio_array) - segment_length_samples) / sample_rate
            last_end_time = len(audio_array) / sample_rate
            
            segments.append({
                "start_time": last_start_time,
                "end_time": last_end_time,
                "caption": last_caption
            })
        
        return segments
    
    def _generate_summary(self, segment_captions: List[Dict]) -> str:
        """Generate a summary caption from multiple segment captions.
        
        Args:
            segment_captions: List of segment captions
            
        Returns:
            Summary caption
        """
        # Simple approach: concatenate captions with transitions
        if len(segment_captions) == 0:
            return "No audio content detected."
        elif len(segment_captions) == 1:
            return segment_captions[0]["caption"]
        
        # For multiple segments, create a narrative
        summary_parts = []
        
        # Beginning
        summary_parts.append(f"The audio begins with {segment_captions[0]['caption'].lower()}")
        
        # Middle (limited to a few segments to avoid too long summaries)
        middle_segments = segment_captions[1:-1]
        if len(middle_segments) > 0:
            if len(middle_segments) <= 2:
                # For just 1-2 middle segments, include all
                for segment in middle_segments:
                    summary_parts.append(f"Then {segment['caption'].lower()}")
            else:
                # For many segments, sample a few representative ones
                sample_indices = [len(middle_segments) // 3, 2 * len(middle_segments) // 3]
                for idx in sample_indices:
                    summary_parts.append(f"Then {middle_segments[idx]['caption'].lower()}")
                
        # End
        if len(segment_captions) > 1:
            summary_parts.append(f"Finally, {segment_captions[-1]['caption'].lower()}")
        
        # Join with appropriate transitions
        summary = " ".join(summary_parts)
        
        # Clean up any redundant or repeated phrases
        summary = summary.replace(" , ", ", ")
        summary = summary.replace(" .", ".")
        
        return summary
    
    def caption_with_sentiment(self, audio_path: str) -> Dict:
        """Generate captions with sentiment analysis.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with captions and sentiment analysis
        """
        # First generate standard caption
        caption_result = self.caption(audio_path=audio_path, return_all_captions=True)
        
        # For sentiment analysis, we'll need a text model - normally would import here
        # but for demonstration, we'll use a simple rule-based approach
        sentiment_words = {
            "positive": ["happy", "upbeat", "cheerful", "joyful", "pleasant", "bright", 
                        "energetic", "uplifting", "exciting", "relaxing", "beautiful",
                        "harmonious", "smooth", "peaceful"],
            "negative": ["sad", "depressing", "dark", "gloomy", "ominous", "tense",
                        "aggressive", "harsh", "noisy", "chaotic", "disturbing", 
                        "anxious", "frightening", "distorted"],
            "neutral": ["ambient", "calm", "steady", "neutral", "moderate", "average",
                        "standard", "typical", "ordinary", "regular"]
        }
        
        # Extract captions for analysis
        captions_to_analyze = [caption_result["caption"]]
        if "segments" in caption_result:
            for segment in caption_result["segments"]:
                captions_to_analyze.append(segment["caption"])
        
        # Analyze sentiment of captions
        sentiment_scores = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
        
        for caption in captions_to_analyze:
            caption_lower = caption.lower()
            for sentiment, words in sentiment_words.items():
                for word in words:
                    if word in caption_lower:
                        sentiment_scores[sentiment] += 1
        
        # Determine overall sentiment
        dominant_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        if sentiment_scores[dominant_sentiment] == 0:
            dominant_sentiment = "neutral"  # Default if no sentiment words found
        
        # Add sentiment analysis to result
        caption_result["sentiment"] = {
            "dominant": dominant_sentiment,
            "scores": sentiment_scores
        }
        
        return caption_result
    
    def generate_mix_notes(self, audio_path: str, include_timestamps: bool = True) -> Dict:
        """Generate mix notes suitable for DJs or producers.
        
        Args:
            audio_path: Path to audio file
            include_timestamps: Whether to include timestamps in the notes
            
        Returns:
            Dictionary with mix notes and sections
        """
        # Generate captions with segments
        caption_result = self.caption(audio_path=audio_path, return_all_captions=True)
        
        # Extract audio features for additional analysis
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Calculate tempo/BPM
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Estimate key (simple implementation - would use a dedicated key detection model in practice)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        key_profile = np.mean(chroma, axis=1)
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        estimated_key = key_names[np.argmax(key_profile)]
        
        # Estimate if major or minor based on relative minor third presence
        key_index = np.argmax(key_profile)
        minor_third = (key_index + 3) % 12
        major_third = (key_index + 4) % 12
        
        if key_profile[minor_third] > key_profile[major_third]:
            estimated_key += "m"  # minor
        
        # Format mix notes
        mix_notes = {
            "title": f"Mix Notes for {os.path.basename(audio_path)}",
            "summary": caption_result["caption"],
            "technical_details": {
                "estimated_bpm": float(f"{tempo:.1f}"),
                "estimated_key": estimated_key,
                "duration": caption_result["duration"]
            },
            "sections": []
        }
        
        # Add sections based on segments
        if "segments" in caption_result:
            for i, segment in enumerate(caption_result["segments"]):
                section = {
                    "section": f"Section {i+1}",
                    "description": segment["caption"]
                }
                
                if include_timestamps:
                    section["start_time"] = f"{segment['start_time']:.1f}s"
                    section["end_time"] = f"{segment['end_time']:.1f}s"
                    
                mix_notes["sections"].append(section)
        
        return mix_notes 