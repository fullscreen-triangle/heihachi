"""
Drum sound analysis module using Wav2Vec2 models.

This module provides functionality for classifying and analyzing drum sounds
in audio recordings using fine-tuned Wav2Vec2 models.
"""

import torch
import numpy as np
import librosa
import logging
import time
from typing import Dict, List, Union, Optional, Tuple
import os
import json
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class DrumSoundAnalyzer:
    """Drum sound analyzer using Wav2Vec2 models."""
    
    def __init__(self, model_name: str = "JackArt/wav2vec2-for-drum-classification", 
                 use_cuda: bool = True,
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the drum sound analyzer.
        
        Args:
            model_name: Model name/identifier (default: JackArt/wav2vec2-for-drum-classification)
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
        self.sample_rate = self.config.get("sample_rate", 16000)  # Wav2Vec2 typically uses 16kHz
        self.segment_length = self.config.get("segment_length", 0.4)  # Length in seconds for analysis
        self.overlap = self.config.get("overlap", 0.2)  # Overlap between segments
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)  # Threshold for detection
        
        # Drum class labels based on the model used
        self.drum_classes = self.config.get("drum_classes", [
            "kick", "snare", "closed_hi_hat", "open_hi_hat", "tom", "cymbal", "clap", "rim", "percussion", "other"
        ])
        
        # Ensure the labels match the model's output size
        self.class_mapping = {i: label for i, label in enumerate(self.drum_classes)}
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the Wav2Vec2 model for drum classification."""
        logger.info(f"Loading drum sound analysis model: {self.model_name}")
        try:
            # Load feature extractor and model
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Update class labels if available in the model config
            if hasattr(self.model.config, "id2label") and self.model.config.id2label:
                self.class_mapping = self.model.config.id2label
                self.drum_classes = list(self.class_mapping.values())
                logger.info(f"Using model's drum classes: {', '.join(self.drum_classes)}")
            
            logger.info(f"Drum sound analysis model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load drum analysis model: {str(e)}")
            raise ValueError(f"Failed to load drum analysis model: {str(e)}")
    
    def load_audio(self, audio_path: str, target_sr: Optional[int] = None) -> np.ndarray:
        """Load audio from file and convert to mono at target sample rate.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate (defaults to model's sample rate)
            
        Returns:
            NumPy array of audio data
        """
        if target_sr is None:
            target_sr = self.sample_rate
            
        logger.info(f"Loading audio: {audio_path}")
        try:
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            duration = len(audio) / sr
            logger.info(f"Loaded audio: {duration:.2f}s @ {sr}Hz")
            return audio
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise ValueError(f"Failed to load audio: {str(e)}")
    
    def analyze_drum_hit(self, audio: Union[str, np.ndarray], 
                           start_time: float = 0.0,
                           duration: Optional[float] = None) -> Dict:
        """Analyze a single drum hit to identify its type and characteristics.
        
        Args:
            audio: Audio file path or NumPy array
            start_time: Start time in seconds for analysis
            duration: Duration in seconds for analysis (default to segment_length)
            
        Returns:
            Dictionary with drum hit analysis results
        """
        # Load audio if string path provided
        if isinstance(audio, str):
            audio_data = self.load_audio(audio)
        else:
            audio_data = audio
            
        # Set duration if not provided
        if duration is None:
            duration = self.segment_length
            
        # Extract segment for analysis
        sr = self.sample_rate
        start_sample = int(start_time * sr)
        end_sample = min(len(audio_data), start_sample + int(duration * sr))
        
        if start_sample >= len(audio_data) or start_sample >= end_sample:
            logger.warning(f"Invalid time range: {start_time}s to {start_time + duration}s")
            return {"error": "Invalid time range"}
        
        segment = audio_data[start_sample:end_sample]
        
        # Ensure minimum length
        if len(segment) < 0.05 * sr:  # At least 50ms
            logger.warning(f"Segment too short: {len(segment)/sr:.3f}s")
            return {"error": "Segment too short"}
            
        # Process with Wav2Vec2
        try:
            # Prepare input
            inputs = self.feature_extractor(
                segment, 
                sampling_rate=sr,
                padding=True,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Process results
            logits = outputs.logits.cpu()
            probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_class_idx]
            drum_type = self.class_mapping[predicted_class_idx]
            
            # Extract audio features for further analysis
            features = self._extract_drum_features(segment, sr)
            
            # Combine results
            result = {
                "drum_type": drum_type,
                "confidence": float(confidence),
                "probabilities": {self.class_mapping[i]: float(prob) for i, prob in enumerate(probabilities)},
                "start_time": start_time,
                "duration": (end_sample - start_sample) / sr,
                "features": features
            }
            
            logger.debug(f"Detected {drum_type} with confidence {confidence:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing drum hit: {str(e)}")
            return {"error": str(e)}
    
    def _extract_drum_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract acoustic features for a drum hit.
        
        Args:
            audio: Audio samples
            sr: Sample rate
            
        Returns:
            Dictionary of acoustic features
        """
        features = {}
        
        try:
            # Calculate energy/loudness
            rms = librosa.feature.rms(y=audio)[0]
            features["peak_amplitude"] = float(np.max(np.abs(audio)))
            features["rms_energy"] = float(np.mean(rms))
            
            # Calculate spectral centroid (brightness)
            cent = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features["spectral_centroid"] = float(np.mean(cent))
            
            # Calculate spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            features["spectral_bandwidth"] = float(np.mean(bandwidth))
            
            # Calculate spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            features["spectral_contrast"] = float(np.mean(contrast))
            
            # Calculate attack time
            envelope = np.abs(audio)
            attack_time = np.argmax(envelope) / sr
            features["attack_time"] = float(attack_time)
            
            # Calculate decay time
            peak_idx = np.argmax(envelope)
            decay_threshold = 0.1 * envelope[peak_idx]
            decay_samples = np.where(envelope[peak_idx:] < decay_threshold)[0]
            if len(decay_samples) > 0:
                decay_time = decay_samples[0] / sr
            else:
                decay_time = (len(envelope) - peak_idx) / sr
            features["decay_time"] = float(decay_time)
            
            # Get tempo-related info if the segment is long enough
            if len(audio) > 0.1 * sr:  # At least 100ms
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                features["estimated_tempo"] = float(tempo)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting drum features: {str(e)}")
            return {"error": str(e)}
    
    def detect_drum_hits(self, audio: Union[str, np.ndarray], 
                          threshold: Optional[float] = None,
                          min_duration: float = 0.05,
                          max_duration: float = 0.3) -> List[Dict]:
        """Detect and analyze all drum hits in an audio file.
        
        Args:
            audio: Audio file path or NumPy array
            threshold: Energy threshold for onset detection (None for automatic)
            min_duration: Minimum duration for a valid drum hit
            max_duration: Maximum duration for a valid drum hit
            
        Returns:
            List of dictionaries with drum hit analysis
        """
        # Load audio if string path provided
        if isinstance(audio, str):
            audio_data = self.load_audio(audio)
        else:
            audio_data = audio
            
        sr = self.sample_rate
        logger.info(f"Detecting drum hits in {len(audio_data)/sr:.2f}s audio")
        
        try:
            # Detect onsets
            onset_envelope = librosa.onset.onset_strength(
                y=audio_data, 
                sr=sr,
                hop_length=512
            )
            
            # Use dynamic threshold if not specified
            if threshold is None:
                threshold = np.mean(onset_envelope) + 0.5 * np.std(onset_envelope)
            
            # Find onset locations
            onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_envelope,
                sr=sr,
                hop_length=512,
                threshold=threshold
            )
            
            # Convert frames to time
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            
            logger.info(f"Detected {len(onset_times)} potential drum hits")
            
            # Analyze each drum hit
            results = []
            for i, onset_time in enumerate(onset_times):
                # Determine segment duration
                if i < len(onset_times) - 1:
                    # Use time to next onset, capped by max_duration
                    segment_duration = min(onset_times[i+1] - onset_time, max_duration)
                else:
                    # Use fixed duration for last onset
                    segment_duration = max_duration
                
                # Skip segments that are too short
                if segment_duration < min_duration:
                    continue
                
                # Analyze the drum hit
                hit_result = self.analyze_drum_hit(
                    audio_data, 
                    start_time=onset_time, 
                    duration=segment_duration
                )
                
                # Add to results if confidence threshold met
                if "error" not in hit_result and hit_result["confidence"] >= self.confidence_threshold:
                    # Add onset index
                    hit_result["onset_index"] = i
                    results.append(hit_result)
            
            logger.info(f"Successfully analyzed {len(results)} drum hits")
            return results
            
        except Exception as e:
            logger.error(f"Error detecting drum hits: {str(e)}")
            return [{"error": str(e)}]
    
    def create_drum_pattern(self, hits: List[Dict], 
                             quantize: bool = True,
                             tempo: float = 120.0,
                             grid_resolution: int = 16) -> Dict:
        """Create a structured drum pattern from detected hits.
        
        Args:
            hits: List of drum hit analyses
            quantize: Whether to quantize to musical grid
            tempo: Tempo in BPM (for quantization)
            grid_resolution: Grid resolution for quantization (16 = 16th notes)
            
        Returns:
            Dictionary representing the drum pattern
        """
        if not hits:
            return {"error": "No hits provided"}
            
        try:
            # Calculate beat duration in seconds
            beat_duration = 60.0 / tempo
            
            # Initialize pattern
            pattern = {
                "tempo": tempo,
                "resolution": grid_resolution,
                "total_beats": 0,
                "tracks": {}
            }
            
            # Initialize tracks for each drum type
            drum_types = set(hit["drum_type"] for hit in hits if "drum_type" in hit)
            for drum_type in drum_types:
                pattern["tracks"][drum_type] = []
            
            # Get the total pattern duration
            if hits:
                last_hit = max(hits, key=lambda x: x.get("start_time", 0) + x.get("duration", 0))
                total_duration = last_hit["start_time"] + last_hit["duration"]
                total_beats = np.ceil(total_duration / beat_duration)
                pattern["total_beats"] = int(total_beats)
            else:
                pattern["total_beats"] = 4  # Default to one bar
            
            # Process each hit
            for hit in hits:
                if "drum_type" not in hit:
                    continue
                    
                drum_type = hit["drum_type"]
                start_time = hit["start_time"]
                
                # Convert to musical position
                position_in_beats = start_time / beat_duration
                
                if quantize:
                    # Quantize to the specified grid
                    ticks_per_beat = grid_resolution
                    position_in_ticks = position_in_beats * ticks_per_beat
                    quantized_ticks = round(position_in_ticks)
                    quantized_beats = quantized_ticks / ticks_per_beat
                    position = quantized_beats
                else:
                    position = position_in_beats
                
                # Add hit to the track
                pattern["tracks"][drum_type].append({
                    "position": float(position),
                    "velocity": min(1.0, hit.get("features", {}).get("peak_amplitude", 0.8) * 1.25),
                    "confidence": hit.get("confidence", 1.0)
                })
            
            # Sort hits by position
            for drum_type in pattern["tracks"]:
                pattern["tracks"][drum_type].sort(key=lambda x: x["position"])
            
            return pattern
            
        except Exception as e:
            logger.error(f"Error creating drum pattern: {str(e)}")
            return {"error": str(e)}
    
    def export_pattern(self, pattern: Dict, output_format: str = "json", file_path: Optional[str] = None) -> Union[str, Dict]:
        """Export drum pattern to various formats.
        
        Args:
            pattern: Drum pattern dictionary
            output_format: Output format (json, midi, etc.)
            file_path: Path to save the output file
            
        Returns:
            Exported pattern or path to exported file
        """
        if "error" in pattern:
            return pattern
            
        try:
            if output_format.lower() == "json":
                if file_path:
                    with open(file_path, 'w') as f:
                        json.dump(pattern, f, indent=2)
                    return file_path
                else:
                    return pattern
                    
            elif output_format.lower() == "midi":
                # Create MIDI file
                try:
                    import mido
                    from mido import Message, MidiFile, MidiTrack
                    
                    # Create a new MIDI file with a single track
                    mid = MidiFile()
                    track = MidiTrack()
                    mid.tracks.append(track)
                    
                    # Set tempo
                    tempo_in_microseconds = int(60000000 / pattern["tempo"])
                    track.append(mido.MetaMessage('set_tempo', tempo=tempo_in_microseconds))
                    
                    # Standard MIDI drum mapping
                    drum_map = {
                        "kick": 36,
                        "snare": 38,
                        "closed_hi_hat": 42,
                        "open_hi_hat": 46,
                        "tom": 45,  # Low tom
                        "cymbal": 49,  # Crash cymbal
                        "clap": 39,
                        "rim": 37,
                        "percussion": 56,  # Cowbell as generic percussion
                        "other": 75,  # Claves as generic
                    }
                    
                    # Add each drum hit as a MIDI note
                    ticks_per_beat = mid.ticks_per_beat
                    
                    for drum_type, hits in pattern["tracks"].items():
                        # Get MIDI note number for this drum type
                        note = drum_map.get(drum_type, 75)  # Default to claves if not mapped
                        
                        for hit in hits:
                            # Calculate position in ticks
                            position_in_ticks = int(hit["position"] * ticks_per_beat)
                            
                            # Calculate velocity (0-127)
                            velocity = int(hit["velocity"] * 127)
                            
                            # Add note on and note off
                            track.append(Message('note_on', note=note, velocity=velocity, time=position_in_ticks))
                            track.append(Message('note_off', note=note, velocity=0, time=10))  # Short duration
                    
                    # Save the MIDI file
                    if file_path:
                        mid.save(file_path)
                        return file_path
                    else:
                        import tempfile
                        temp_file = tempfile.NamedTemporaryFile(suffix='.mid', delete=False)
                        mid.save(temp_file.name)
                        return temp_file.name
                        
                except ImportError:
                    logger.error("mido library required for MIDI export")
                    return {"error": "mido library required for MIDI export"}
                    
            else:
                logger.error(f"Unsupported output format: {output_format}")
                return {"error": f"Unsupported output format: {output_format}"}
                
        except Exception as e:
            logger.error(f"Error exporting pattern: {str(e)}")
            return {"error": str(e)}
    
    def recommend_drum_replacements(self, reference_hits: List[Dict], 
                                     sample_library_path: str,
                                     top_n: int = 3) -> Dict[str, List[Dict]]:
        """Recommend replacements for detected drum hits from a sample library.
        
        Args:
            reference_hits: List of analyzed drum hits
            sample_library_path: Path to sample library directory
            top_n: Number of top recommendations per hit
            
        Returns:
            Dictionary mapping drum types to lists of recommended samples
        """
        if not reference_hits:
            return {"error": "No reference hits provided"}
            
        try:
            # Scan sample library
            sample_files = []
            for root, _, files in os.walk(sample_library_path):
                for file in files:
                    if file.endswith(('.wav', '.mp3', '.aiff', '.ogg')):
                        sample_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(sample_files)} samples in library")
            
            if not sample_files:
                return {"error": "No audio samples found in library"}
            
            # Group reference hits by drum type
            hits_by_type = {}
            for hit in reference_hits:
                drum_type = hit.get("drum_type", "unknown")
                if drum_type not in hits_by_type:
                    hits_by_type[drum_type] = []
                hits_by_type[drum_type].append(hit)
            
            # Process each sample in the library
            sample_analyses = []
            for sample_path in sample_files:
                try:
                    # Load and analyze sample
                    audio = self.load_audio(sample_path)
                    analysis = self.analyze_drum_hit(audio)
                    
                    if "error" not in analysis:
                        analysis["file_path"] = sample_path
                        analysis["file_name"] = os.path.basename(sample_path)
                        sample_analyses.append(analysis)
                except Exception as e:
                    logger.warning(f"Error analyzing sample {sample_path}: {str(e)}")
            
            logger.info(f"Successfully analyzed {len(sample_analyses)} samples")
            
            # Find the best matches for each drum type
            recommendations = {}
            
            for drum_type, hits in hits_by_type.items():
                recommendations[drum_type] = []
                
                # Get samples of the same drum type
                matching_samples = [s for s in sample_analyses if s.get("drum_type") == drum_type]
                
                if not matching_samples:
                    logger.warning(f"No matching samples found for {drum_type}")
                    continue
                
                # For each hit, find the most similar samples
                for hit in hits:
                    hit_features = hit.get("features", {})
                    
                    if not hit_features or "error" in hit_features:
                        continue
                    
                    # Calculate similarity scores
                    scores = []
                    for sample in matching_samples:
                        sample_features = sample.get("features", {})
                        
                        if not sample_features or "error" in sample_features:
                            continue
                        
                        # Create feature vectors (normalize values)
                        hit_vector = []
                        sample_vector = []
                        
                        for key in ["spectral_centroid", "spectral_bandwidth", "attack_time", "decay_time"]:
                            if key in hit_features and key in sample_features:
                                hit_vector.append(hit_features[key])
                                sample_vector.append(sample_features[key])
                        
                        if hit_vector and sample_vector:
                            # Calculate cosine similarity
                            similarity = cosine_similarity([hit_vector], [sample_vector])[0][0]
                            scores.append((sample, similarity))
                    
                    # Sort by similarity
                    scores.sort(key=lambda x: x[1], reverse=True)
                    
                    # Get top N recommendations
                    top_recommendations = []
                    for sample, score in scores[:top_n]:
                        recommendation = {
                            "file_path": sample["file_path"],
                            "file_name": sample["file_name"],
                            "similarity_score": float(score),
                            "features": sample.get("features", {})
                        }
                        top_recommendations.append(recommendation)
                    
                    # Add to results
                    if top_recommendations:
                        hit_info = {
                            "hit": {
                                "start_time": hit["start_time"],
                                "duration": hit["duration"],
                                "features": hit["features"]
                            },
                            "recommendations": top_recommendations
                        }
                        recommendations[drum_type].append(hit_info)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error recommending drum replacements: {str(e)}")
            return {"error": str(e)}
    
    def analyze_file(self, audio_path: str, output_path: Optional[str] = None) -> Dict:
        """Analyze a complete audio file and output comprehensive drum analysis.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save results (JSON)
            
        Returns:
            Dictionary with comprehensive drum analysis
        """
        try:
            # Load audio
            audio = self.load_audio(audio_path)
            duration = len(audio) / self.sample_rate
            
            logger.info(f"Analyzing drums in {os.path.basename(audio_path)} ({duration:.2f}s)")
            
            # Detect and analyze drum hits
            hits = self.detect_drum_hits(audio)
            
            # Create drum pattern
            pattern = self.create_drum_pattern(hits)
            
            # Count hits by type
            hit_counts = {}
            for hit in hits:
                drum_type = hit.get("drum_type")
                if drum_type:
                    hit_counts[drum_type] = hit_counts.get(drum_type, 0) + 1
            
            # Calculate distribution percentages
            total_hits = sum(hit_counts.values())
            distribution = {drum_type: count / total_hits for drum_type, count in hit_counts.items()}
            
            # Analyze timing
            timing_analysis = self._analyze_timing(hits)
            
            # Package results
            results = {
                "file_path": audio_path,
                "duration": duration,
                "total_hits": len(hits),
                "hit_counts": hit_counts,
                "distribution": {k: float(f"{v:.2f}") for k, v in distribution.items()},
                "pattern": pattern,
                "timing_analysis": timing_analysis,
                "drum_hits": hits[:10]  # Include first 10 hits for reference
            }
            
            # Save results if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Drum analysis saved to: {output_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_timing(self, hits: List[Dict]) -> Dict:
        """Analyze timing characteristics of drum hits.
        
        Args:
            hits: List of analyzed drum hits
            
        Returns:
            Dictionary with timing analysis
        """
        if not hits:
            return {"error": "No hits to analyze"}
            
        try:
            # Sort hits by start time
            sorted_hits = sorted(hits, key=lambda x: x.get("start_time", 0))
            
            # Calculate inter-onset intervals
            intervals = []
            for i in range(1, len(sorted_hits)):
                interval = sorted_hits[i]["start_time"] - sorted_hits[i-1]["start_time"]
                if 0.05 < interval < 2.0:  # Filter out very short or long intervals
                    intervals.append(interval)
            
            if not intervals:
                return {"error": "Insufficient data for timing analysis"}
            
            # Calculate statistics
            mean_ioi = np.mean(intervals)
            median_ioi = np.median(intervals)
            std_ioi = np.std(intervals)
            
            # Estimate tempo
            estimated_tempo = 60.0 / median_ioi
            
            # Analyze by drum type
            timing_by_type = {}
            for drum_type in set(hit.get("drum_type") for hit in hits if "drum_type" in hit):
                # Get hits of this type
                type_hits = [hit for hit in hits if hit.get("drum_type") == drum_type]
                
                if len(type_hits) < 2:
                    continue
                
                # Sort by start time
                type_hits.sort(key=lambda x: x.get("start_time", 0))
                
                # Calculate intervals
                type_intervals = []
                for i in range(1, len(type_hits)):
                    interval = type_hits[i]["start_time"] - type_hits[i-1]["start_time"]
                    if 0.05 < interval < 2.0:
                        type_intervals.append(interval)
                
                if type_intervals:
                    timing_by_type[drum_type] = {
                        "mean_interval": float(np.mean(type_intervals)),
                        "median_interval": float(np.median(type_intervals)),
                        "std_interval": float(np.std(type_intervals))
                    }
            
            # Calculate timing consistency score (lower std_ioi relative to mean_ioi is better)
            if mean_ioi > 0:
                consistency_score = 1.0 - min(1.0, std_ioi / mean_ioi)
            else:
                consistency_score = 0.0
            
            # Create timing analysis
            timing_analysis = {
                "mean_interval": float(mean_ioi),
                "median_interval": float(median_ioi),
                "std_interval": float(std_ioi),
                "estimated_tempo": float(f"{estimated_tempo:.1f}"),
                "consistency_score": float(f"{consistency_score:.2f}"),
                "by_drum_type": timing_by_type
            }
            
            return timing_analysis
            
        except Exception as e:
            logger.error(f"Error in timing analysis: {str(e)}")
            return {"error": str(e)} 