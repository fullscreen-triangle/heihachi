import numpy as np
import librosa
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.signal import find_peaks, butter, filtfilt
from concurrent.futures import ThreadPoolExecutor


@dataclass
class RhythmInfo:
    groove_pattern: np.ndarray
    syncopation_score: float
    rhythm_complexity: float
    pulse_clarity: float
    onset_patterns: Dict[str, np.ndarray]
    meter_strength: float
    subdivisions: List[float]


class RhythmAnalyzer:
    def __init__(self):
        self.sr = 44100
        self.hop_length = 512
        self.segment_duration = 30  # 30-second segments
        self.overlap = 5  # 5-second overlap
        
        self.rhythm_bands = {
            'low': (20, 200),  # Bass/kick
            'mid': (200, 2000),  # Snare/clap
            'high': (2000, 8000)  # Hi-hats/cymbals
        }

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze rhythmic characteristics of the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing rhythmic features
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
            segment_results = list(executor.map(self._analyze_segment, segments))

        # Combine results
        return self._combine_segment_results(segment_results)

    def _analyze_segment(self, audio: np.ndarray) -> Dict:
        """Analyze a single segment of audio."""
        try:
            # Get onset envelopes for different frequency bands
            onset_patterns = self._get_multiband_onsets(audio)
            
            if not any(len(env) > 0 for env in onset_patterns.values()):
                return self._get_fallback_features()

            # Detect main groove pattern
            groove_pattern = self._extract_groove_pattern(onset_patterns)

            # Calculate rhythm metrics
            info = RhythmInfo(
                groove_pattern=groove_pattern,
                syncopation_score=self._calculate_syncopation(onset_patterns),
                rhythm_complexity=self._analyze_rhythm_complexity(onset_patterns),
                pulse_clarity=self._measure_pulse_clarity(onset_patterns),
                onset_patterns=onset_patterns,
                meter_strength=self._analyze_meter_strength(onset_patterns),
                subdivisions=self._detect_subdivisions(onset_patterns)
            )

            return self._convert_to_dict(info)

        except Exception as e:
            print(f"Error in rhythmic analysis: {str(e)}")
            return self._get_fallback_features()

    def _get_fallback_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'groove_pattern': np.zeros(32).tolist(),
            'syncopation_score': 0.0,
            'rhythm_complexity': 0.0,
            'pulse_clarity': 0.0,
            'onset_patterns': {band: np.zeros(32).tolist() for band in self.rhythm_bands},
            'meter_strength': 0.0,
            'subdivisions': [0.0] * 6  # Default subdivisions
        }

    def _convert_to_dict(self, info: RhythmInfo) -> Dict:
        """Convert RhythmInfo to dictionary with JSON-serializable values."""
        return {
            'groove_pattern': info.groove_pattern.tolist(),
            'syncopation_score': float(info.syncopation_score),
            'rhythm_complexity': float(info.rhythm_complexity),
            'pulse_clarity': float(info.pulse_clarity),
            'onset_patterns': {k: v.tolist() for k, v in info.onset_patterns.items()},
            'meter_strength': float(info.meter_strength),
            'subdivisions': [float(x) for x in info.subdivisions]
        }

    def _combine_segment_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple segments."""
        if not results:
            return self._get_fallback_features()

        # Average numerical metrics
        metrics = ['syncopation_score', 'rhythm_complexity', 'pulse_clarity', 'meter_strength']
        averaged_metrics = {
            metric: float(np.mean([r[metric] for r in results]))
            for metric in metrics
        }

        # Combine patterns using weighted average based on pulse clarity
        weights = [r['pulse_clarity'] for r in results]
        total_weight = sum(weights) or 1.0

        # Combine groove patterns
        groove_patterns = np.array([r['groove_pattern'] for r in results])
        combined_groove = np.average(groove_patterns, weights=weights, axis=0)

        # Combine onset patterns
        combined_onsets = {}
        for band in self.rhythm_bands:
            patterns = np.array([r['onset_patterns'][band] for r in results])
            combined_onsets[band] = np.average(patterns, weights=weights, axis=0).tolist()

        # Combine subdivisions using weighted average
        subdivs = np.array([r['subdivisions'] for r in results])
        combined_subdivs = np.average(subdivs, weights=weights, axis=0)

        return {
            'groove_pattern': combined_groove.tolist(),
            'onset_patterns': combined_onsets,
            'subdivisions': combined_subdivs.tolist(),
            **averaged_metrics
        }

    def _get_multiband_onsets(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Get onset envelopes for different frequency bands."""
        onset_patterns = {}

        for band_name, (low, high) in self.rhythm_bands.items():
            try:
                # Apply bandpass filter using scipy
                nyquist = self.sr // 2
                b, a = butter(4, [low/nyquist, high/nyquist], btype='band')
                filtered_audio = filtfilt(b, a, audio)
                
                # Trim the filtered audio
                y_band = librosa.effects.trim(filtered_audio)[0]

                # Get onset strength envelope
                onset_env = librosa.onset.onset_strength(
                    y=y_band,
                    sr=self.sr,
                    hop_length=self.hop_length
                )

                onset_patterns[band_name] = onset_env

            except Exception as e:
                print(f"Error in band {band_name}: {str(e)}")
                onset_patterns[band_name] = np.zeros(128)  # Fallback pattern

        return onset_patterns

    def _extract_groove_pattern(self, onset_patterns):
        # Combine onset patterns with weights
        combined = (
                0.4 * librosa.util.normalize(onset_patterns['low']) +
                0.3 * librosa.util.normalize(onset_patterns['mid']) +
                0.3 * librosa.util.normalize(onset_patterns['high'])
        )

        # Find typical pattern length (in frames)
        ac = librosa.autocorrelate(combined)
        peaks = find_peaks(ac)[0]
        if len(peaks) > 0:
            pattern_length = peaks[0]
            # Extract representative pattern
            num_patterns = len(combined) // pattern_length
            patterns = combined[:num_patterns * pattern_length].reshape(-1, pattern_length)
            return np.median(patterns, axis=0)
        return combined[:64]  # fallback to first 64 frames

    def _calculate_syncopation(self, onset_patterns):
        # Calculate syncopation based on onset strength at weak beats
        combined = np.sum([env for env in onset_patterns.values()], axis=0)

        # Get beat positions
        _, beat_frames = librosa.beat.beat_track(
            onset_envelope=combined,
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Compare onset strength at beat vs. off-beat positions
        beat_strength = np.mean(combined[beat_frames])
        offbeat_strength = np.mean(np.delete(combined, beat_frames))

        return offbeat_strength / (beat_strength + 1e-8)

    def _analyze_rhythm_complexity(self, onset_patterns):
        # Measure complexity through onset pattern entropy and variability
        complexities = []

        for env in onset_patterns.values():
            # Calculate entropy of onset pattern
            hist, _ = np.histogram(env, bins=20, density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))

            # Calculate variability
            variability = np.std(env) / (np.mean(env) + 1e-8)

            complexities.append(entropy * variability)

        return np.mean(complexities)

    def _measure_pulse_clarity(self, onset_patterns):
        # Measure how clear and regular the pulse is
        combined = np.sum([env for env in onset_patterns.values()], axis=0)

        # Compute autocorrelation
        ac = librosa.autocorrelate(combined)

        # Find peaks in autocorrelation
        peaks = find_peaks(ac)[0]
        if len(peaks) > 0:
            # Measure prominence of first few peaks
            peak_values = ac[peaks[:5]]
            return np.mean(peak_values) / ac[0]
        return 0.0

    def _analyze_meter_strength(self, onset_patterns):
        # Analyze strength of metrical structure
        combined = np.sum([env for env in onset_patterns.values()], axis=0)

        # Get tempo and beats
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=combined,
            sr=self.sr,
            hop_length=self.hop_length
        )

        if len(beats) > 0:
            # Calculate regularity of beat intervals
            intervals = np.diff(beats)
            regularity = 1.0 - (np.std(intervals) / np.mean(intervals))

            # Calculate average onset strength at beat positions
            beat_strength = np.mean(combined[beats])

            return regularity * beat_strength
        return 0.0

    def _detect_subdivisions(self, onset_patterns):
        # Detect common rhythmic subdivisions (e.g., 1/4, 1/8, 1/16 notes)
        combined = np.sum([env for env in onset_patterns.values()], axis=0)

        # Get beat positions
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=combined,
            sr=self.sr,
            hop_length=self.hop_length
        )

        if len(beats) < 2:
            return []

        beat_duration = np.median(np.diff(beats))
        subdivs = []

        # Check common subdivisions
        for div in [1, 2, 3, 4, 6, 8]:
            subdev_length = beat_duration / div
            score = self._evaluate_subdivision(combined, beats, subdev_length)
            subdivs.append(score)

        return subdivs

    def _evaluate_subdivision(self, onset_env, beats, subdiv_length):
        # Evaluate how well a subdivision fits the onset pattern
        score = 0
        count = 0

        for i in range(len(beats) - 1):
            start = beats[i]
            end = beats[i + 1]

            # Check onset strength at subdivision points
            for j in range(1, int(end - start) // int(subdiv_length)):
                point = start + j * int(subdiv_length)
                if point < len(onset_env):
                    score += onset_env[point]
                    count += 1

        return score / (count + 1e-8) if count > 0 else 0
