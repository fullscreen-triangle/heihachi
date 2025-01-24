import numpy as np
import librosa
import torch
from scipy.stats import entropy
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor


class GrooveAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100
        self.hop_length = 512
        self.segment_duration = 30  # analyze in 30-second segments
        self.overlap = 5  # 5-second overlap between segments

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze groove characteristics of the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing groove features
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Split into segments
        segment_samples = self.sr * self.segment_duration
        overlap_samples = self.sr * self.overlap
        segments = []
        
        for start in range(0, len(audio), segment_samples - overlap_samples):
            end = min(start + segment_samples, len(audio))
            if end - start >= self.sr * 5:  # Minimum 5 seconds
                segments.append(audio[start:end])

        # Analyze segments in parallel
        with ThreadPoolExecutor() as executor:
            segment_results = list(executor.map(self.analyze_segment, segments))

        # Combine results
        return self._combine_segment_results(segment_results)

    def analyze_segment(self, audio: np.ndarray) -> Dict:
        """Analyze a single segment of audio."""
        try:
            onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
            tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)

            if len(beats) < 2:
                return self._get_fallback_features()

            features = {
                'microtiming': self._analyze_microtiming(audio, beats),
                'syncopation': self._analyze_syncopation(onset_env, beats),
                'swing_ratio': self._compute_swing_ratio(onset_env, beats),
                'groove_density': self._compute_groove_density(onset_env, beats),
                'rhythmic_complexity': self._compute_rhythmic_complexity(onset_env)
            }

            # Validate features
            return self._validate_features(features)

        except Exception as e:
            print(f"Error in groove analysis: {str(e)}")
            return self._get_fallback_features()

    def _get_fallback_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'microtiming': np.array([0.0]),
            'syncopation': 0.0,
            'swing_ratio': 1.0,
            'groove_density': 0.0,
            'rhythmic_complexity': 0.0
        }

    def _validate_features(self, features: Dict) -> Dict:
        """Validate and clean feature values."""
        validated = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                # Remove NaN/Inf values
                value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
                validated[key] = value.tolist()  # Convert to list for JSON
            else:
                # Clean scalar values
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                validated[key] = float(value)
        return validated

    def _combine_segment_results(self, results: list) -> Dict:
        """Combine results from multiple segments."""
        if not results:
            return self._get_fallback_features()

        combined = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            if isinstance(values[0], list):
                # For array features, concatenate and compute statistics
                all_values = np.concatenate([np.array(v) for v in values])
                combined[key] = {
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'max': float(np.max(all_values)),
                    'min': float(np.min(all_values))
                }
            else:
                # For scalar features, compute statistics
                combined[key] = float(np.mean(values))

        return combined

    def _analyze_microtiming(self, audio: np.ndarray, beats: np.ndarray) -> np.ndarray:
        """Analyze microtiming deviations."""
        if len(beats) < 2:
            return np.array([0.0])

        beat_times = librosa.frames_to_time(beats, sr=self.sr)
        deviations = []
        
        for i in range(1, len(beat_times)):
            start_idx = int(beat_times[i-1] * self.sr)
            end_idx = int(beat_times[i] * self.sr)
            if start_idx < end_idx and end_idx <= len(audio):
                local_window = audio[start_idx:end_idx]
                if len(local_window) > 0:
                    peak = np.argmax(np.abs(local_window))
                    expected = len(local_window) / 2
                    deviation = abs(peak - expected) / len(local_window)
                    deviations.append(deviation)

        return np.array(deviations) if deviations else np.array([0.0])

    def _analyze_syncopation(self, onset_env: np.ndarray, beats: np.ndarray) -> float:
        """Analyze syncopation level."""
        if len(beats) < 2 or len(onset_env) <= np.max(beats):
            return 0.0

        beat_strength = onset_env[beats]
        # Ensure we don't exceed array bounds
        offset = len(onset_env) // 2
        valid_beats = beats[beats + offset < len(onset_env)]
        if len(valid_beats) == 0:
            return 0.0

        offbeat_strength = onset_env[valid_beats + offset]
        ratio = np.mean(offbeat_strength) / (np.mean(beat_strength) + 1e-8)
        return min(ratio, 10.0)  # Cap the ratio to avoid extreme values

    def _compute_swing_ratio(self, onset_env: np.ndarray, beats: np.ndarray) -> float:
        """Compute swing ratio."""
        if len(beats) < 3:
            return 1.0

        even_beats = onset_env[beats[::2]]
        odd_beats = onset_env[beats[1::2]]
        
        if len(even_beats) == 0 or len(odd_beats) == 0:
            return 1.0

        ratio = np.mean(odd_beats) / (np.mean(even_beats) + 1e-8)
        return min(ratio, 5.0)  # Cap the ratio to avoid extreme values

    def _compute_groove_density(self, onset_env: np.ndarray, beats: np.ndarray) -> float:
        """Compute groove density."""
        if len(beats) < 2:
            return 0.0

        beat_windows = np.array_split(onset_env, len(beats))
        densities = []
        
        for window in beat_windows:
            if len(window) > 0:
                threshold = np.mean(window)
                density = np.sum(window > threshold) / len(window)
                densities.append(density)

        return float(np.mean(densities)) if densities else 0.0

    def _compute_rhythmic_complexity(self, onset_env: np.ndarray) -> float:
        """Compute rhythmic complexity."""
        if len(onset_env) < 2:
            return 0.0

        # Compute tempogram
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=self.sr)
        
        if tempogram.size == 0:
            return 0.0

        # Calculate entropy of temporal patterns
        pattern_entropy = entropy(tempogram.mean(axis=1) + 1e-8)
        return min(pattern_entropy, 5.0)  # Cap complexity to avoid extreme values
