"""
Drum sound analysis module using wav2vec2 models.

This module provides functionality for drum onset detection 
and kit piece identification using specialized wav2vec2 models.
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
from transformers import AutoModelForAudioClassification, AutoProcessor

logger = logging.getLogger(__name__)

class DrumAnalyzer:
    """Drum sound analysis using wav2vec2 models."""
    
    def __init__(self, model_name: str = "DunnBC22/wav2vec2-base-Drum_Kit_Sounds", 
                 use_cuda: bool = True,
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the drum analyzer with a wav2vec2 model.
        
        Args:
            model_name: Model name/identifier (default: DunnBC22/wav2vec2-base-Drum_Kit_Sounds)
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
        
        # Configuration for onset detection
        self.onset_threshold = self.config.get("onset_threshold", 0.5)
        self.min_silence_duration = self.config.get("min_silence_duration", 0.05)  # seconds
        self.hop_length = self.config.get("hop_length", 512)
        
        # Load the model and processor
        self._load_model()
    
    def _load_model(self):
        """Load the drum classification model."""
        logger.info(f"Loading drum analysis model: {self.model_name}")
        try:
            # For wav2vec2 drum kit model
            if "wav2vec2" in self.model_name and "Drum" in self.model_name:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Get class labels
                self.id2label = self.model.config.id2label
                self.labels = list(self.id2label.values())
                
                logger.info(f"Drum analysis model loaded successfully on {self.device} with labels: {self.labels}")
            else:
                # For other drum sound models
                logger.warning(f"Model {self.model_name} not specifically supported for drum analysis. Attempting generic loading.")
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Get class labels
                self.id2label = self.model.config.id2label
                self.labels = list(self.id2label.values())
                
        except Exception as e:
            logger.error(f"Failed to load drum analysis model: {str(e)}")
            raise ValueError(f"Failed to load drum analysis model: {str(e)}")
    
    def analyze(self, audio_path: Optional[str] = None, 
               audio_array: Optional[np.ndarray] = None, 
               sample_rate: Optional[int] = None,
               detect_onsets: bool = True,
               visualize: bool = False,
               output_path: Optional[str] = None) -> Dict:
        """Analyze audio for drum sounds.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            detect_onsets: Whether to detect onsets before classification
            visualize: Whether to generate visualization
            output_path: Path to save visualization if enabled
            
        Returns:
            Dictionary with drum sound analysis results
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
        
        # Detect onsets if requested
        if detect_onsets:
            # Detect onsets to isolate drum hits
            onsets, onset_boundaries = self._detect_onsets(audio_array, sample_rate)
            logger.info(f"Detected {len(onsets)} onsets in audio")
            
            # Analyze each onset segment
            drum_hits = []
            
            for i, (start_idx, end_idx) in enumerate(onset_boundaries):
                if end_idx - start_idx < 10:  # Ensure minimum segment length
                    continue
                    
                # Extract segment
                segment = audio_array[start_idx:end_idx]
                
                # Classify segment
                classification = self._classify_segment(segment, sample_rate)
                
                # Add result with timestamp
                timestamp = start_idx / sample_rate
                duration = (end_idx - start_idx) / sample_rate
                
                drum_hits.append({
                    "timestamp": timestamp,
                    "duration": duration,
                    "instrument": classification["instrument"],
                    "score": classification["score"],
                    "all_scores": classification["all_scores"]
                })
            
            # Prepare results
            results = {
                "num_hits": len(drum_hits),
                "hits": drum_hits,
                "sample_rate": sample_rate,
                "model": self.model_name
            }
        else:
            # Classify the entire audio as a single segment
            classification = self._classify_segment(audio_array, sample_rate)
            
            # Prepare results for whole audio
            results = {
                "whole_audio": {
                    "instrument": classification["instrument"],
                    "score": classification["score"],
                    "all_scores": classification["all_scores"]
                },
                "sample_rate": sample_rate,
                "model": self.model_name
            }
        
        # Generate visualization if requested
        if visualize:
            if detect_onsets:
                self._visualize_drum_hits(audio_array, sample_rate, results["hits"], output_path)
            else:
                logger.warning("Visualization requires onset detection. No visualization generated.")
        
        return results
    
    def _detect_onsets(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Detect onsets in audio to isolate drum hits.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            
        Returns:
            Tuple of (onset_frames, onset_boundaries) where onset_boundaries is a list of (start, end) indices
        """
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(
            y=audio, 
            sr=sample_rate,
            hop_length=self.hop_length
        )
        
        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=self.hop_length,
            threshold=self.onset_threshold
        )
        
        # Convert frames to sample indices
        onset_samples = librosa.frames_to_samples(onset_frames, hop_length=self.hop_length)
        
        # Calculate onset boundaries (start and end indices for each hit)
        onset_boundaries = []
        
        for i, onset in enumerate(onset_samples):
            # Start at onset
            start_idx = onset
            
            # End at next onset or end of audio
            if i < len(onset_samples) - 1:
                # Minimum distance between onsets is min_silence_duration
                min_distance = int(self.min_silence_duration * sample_rate)
                if onset_samples[i+1] - onset > min_distance:
                    end_idx = onset_samples[i+1]
                else:
                    end_idx = onset + min_distance
            else:
                # For the last onset, take a fixed segment
                end_idx = onset + int(0.2 * sample_rate)  # 200ms
            
            # Ensure we don't go beyond the audio length
            end_idx = min(end_idx, len(audio))
            
            onset_boundaries.append((start_idx, end_idx))
        
        return onset_frames, onset_boundaries
    
    def _classify_segment(self, audio_segment: np.ndarray, sample_rate: int) -> Dict:
        """Classify a single audio segment using the drum classification model.
        
        Args:
            audio_segment: Audio segment to classify
            sample_rate: Sample rate of the audio
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Process audio through the model
            inputs = self.processor(
                audio_segment, 
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get logits and convert to probabilities
            logits = outputs.logits.cpu().numpy()[0]
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0).numpy()
            
            # Get top prediction
            top_idx = np.argmax(probs)
            top_label = self.id2label[top_idx]
            top_score = float(probs[top_idx])
            
            # Get all scores
            all_scores = []
            for i, label in self.id2label.items():
                all_scores.append({
                    "instrument": label,
                    "score": float(probs[i])
                })
            
            # Sort by score
            all_scores.sort(key=lambda x: x["score"], reverse=True)
            
            return {
                "instrument": top_label,
                "score": top_score,
                "all_scores": all_scores
            }
        
        except Exception as e:
            logger.error(f"Error classifying audio segment: {str(e)}")
            return {
                "instrument": "unknown",
                "score": 0.0,
                "all_scores": [],
                "error": str(e)
            }
    
    def _visualize_drum_hits(self, audio: np.ndarray, sample_rate: int, hits: List[Dict], output_path: Optional[str] = None):
        """Visualize detected drum hits on audio waveform.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of audio
            hits: List of detected drum hits
            output_path: Path to save visualization
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot waveform
            times = np.arange(len(audio)) / sample_rate
            plt.subplot(2, 1, 1)
            plt.plot(times, audio, alpha=0.6, color='blue')
            plt.title('Audio Waveform with Detected Drum Hits')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            
            # Add markers for drum hits
            instrument_colors = {
                "kick": "red",
                "snare": "green",
                "hi-hat": "orange",
                "tom": "purple",
                "cymbal": "cyan",
                "overhead": "magenta",
                "clap": "yellow",
                "rim": "brown"
            }
            
            # Define fallback colors for any other instruments
            fallback_colors = ["pink", "gray", "olive", "teal", "navy"]
            
            # Add markers for each hit
            for hit in hits:
                timestamp = hit["timestamp"]
                instrument = hit["instrument"].lower()
                
                # Get color for this instrument
                color = None
                for key in instrument_colors:
                    if key in instrument:
                        color = instrument_colors[key]
                        break
                
                # If no matching color found, use hash-based fallback
                if color is None:
                    hash_val = hash(instrument) % len(fallback_colors)
                    color = fallback_colors[hash_val]
                
                # Plot marker
                plt.axvline(x=timestamp, color=color, alpha=0.7, linestyle='--')
            
            # Plot drum hit distribution by instrument type
            plt.subplot(2, 1, 2)
            
            # Count hits by instrument
            instrument_counts = {}
            for hit in hits:
                instrument = hit["instrument"]
                instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
            
            # Sort by count
            sorted_instruments = sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Plot bar chart
            instruments = [item[0] for item in sorted_instruments]
            counts = [item[1] for item in sorted_instruments]
            
            plt.bar(instruments, counts, color='skyblue')
            plt.title('Drum Hit Distribution')
            plt.xlabel('Instrument')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            
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
    
    def analyze_pattern(self, audio_path: str, pattern_length: float = 4.0, 
                      bpm: Optional[float] = None) -> Dict:
        """Analyze a drum pattern, identifying repetitions and structure.
        
        Args:
            audio_path: Path to audio file
            pattern_length: Expected pattern length in seconds (default: 4.0)
            bpm: Beats per minute (if None, attempts to detect)
            
        Returns:
            Dictionary with pattern analysis results
        """
        # First, perform regular analysis to get all drum hits
        results = self.analyze(audio_path=audio_path, detect_onsets=True)
        
        # If no BPM provided, attempt to detect
        # (Note: In a real implementation, you might want to use a dedicated BPM detector)
        if bpm is None:
            # Simple BPM detection based on onset intervals
            if len(results["hits"]) > 1:
                hit_times = [hit["timestamp"] for hit in results["hits"]]
                intervals = np.diff(hit_times)
                if len(intervals) > 0:
                    # Calculate median interval for robustness
                    median_interval = np.median(intervals)
                    detected_bpm = 60.0 / median_interval
                    
                    # Check if the detected BPM is reasonable
                    if 60 <= detected_bpm <= 200:
                        bpm = detected_bpm
                    else:
                        # If outside reasonable range, try adjusting
                        while detected_bpm > 200:
                            detected_bpm /= 2
                        while detected_bpm < 60:
                            detected_bpm *= 2
                        bpm = detected_bpm
            
            # Default fallback
            if bpm is None:
                bpm = 120.0
                logger.warning(f"Could not detect BPM, using default: {bpm}")
        
        # Calculate beats per pattern
        beat_duration = 60.0 / bpm
        beats_per_pattern = pattern_length / beat_duration
        
        # Organize hits by pattern position
        pattern_hits = []
        
        for hit in results["hits"]:
            # Calculate position within pattern
            timestamp = hit["timestamp"]
            pattern_position = (timestamp % pattern_length) / pattern_length
            beat_position = (pattern_position * beats_per_pattern)
            
            # Add pattern information
            pattern_hit = hit.copy()
            pattern_hit["pattern_position"] = float(pattern_position)
            pattern_hit["beat_position"] = float(beat_position)
            pattern_hit["pattern_number"] = int(timestamp / pattern_length)
            
            pattern_hits.append(pattern_hit)
        
        # Group hits by instrument
        instrument_patterns = {}
        
        for hit in pattern_hits:
            instrument = hit["instrument"]
            if instrument not in instrument_patterns:
                instrument_patterns[instrument] = []
            instrument_patterns[instrument].append(hit)
        
        # Analyze pattern density
        pattern_count = max([hit["pattern_number"] for hit in pattern_hits]) + 1 if pattern_hits else 0
        total_duration = pattern_count * pattern_length
        density = len(results["hits"]) / total_duration if total_duration > 0 else 0
        
        # Prepare results
        pattern_results = {
            "bpm": float(bpm),
            "pattern_length_seconds": pattern_length,
            "beats_per_pattern": beats_per_pattern,
            "total_patterns": pattern_count,
            "hits_per_pattern": len(results["hits"]) / pattern_count if pattern_count > 0 else 0,
            "density": density,
            "pattern_hits": pattern_hits,
            "instrument_patterns": {k: len(v) for k, v in instrument_patterns.items()},
            "model": self.model_name
        }
        
        return pattern_results 