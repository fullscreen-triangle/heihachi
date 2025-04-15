import numpy as np
import librosa
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor


@dataclass
class DrumEvent:
    type: str  # 'kick', 'snare', 'hihat', etc.
    time: float
    confidence: float
    velocity: float


@dataclass
class PercussionInfo:
    events: List[DrumEvent]
    pattern_length: float
    density: float
    complexity: float
    patterns: Dict[str, np.ndarray]
    timing_stability: float
    ghost_notes: List[DrumEvent]


class PercussionAnalyzer:
    def __init__(self):
        self.sr = 44100
        self.hop_length = 512
        self.segment_duration = 30  # 30-second segments
        self.overlap = 5  # 5-second overlap

        # Frequency ranges for different drum types
        self.drum_ranges = {
            'kick': (20, 150),
            'snare': (150, 500),
            'hihat': (5000, 10000),
            'tom': (200, 800),
            'cymbal': (8000, 16000)
        }

        # Minimum time between consecutive hits (in seconds)
        self.min_interval = {
            'kick': 0.1,
            'snare': 0.1,
            'hihat': 0.05,
            'tom': 0.1,
            'cymbal': 0.1
        }

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze percussion in the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing percussion information
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
                segments.append((audio[start:end], start / self.sr))

        # Analyze segments in parallel
        with ThreadPoolExecutor() as executor:
            segment_results = list(executor.map(
                lambda x: self._analyze_segment(*x), segments
            ))

        # Combine results
        return self._combine_segment_results(segment_results)

    def _analyze_segment(self, audio: np.ndarray, start_time: float) -> Dict:
        """Analyze a single segment of audio."""
        try:
            # Extract drum events
            events = self._detect_drum_events(audio, start_time)
            
            if not events:
                return self._get_fallback_features()

            # Analyze patterns
            patterns = self._extract_patterns(events)
            
            # Calculate metrics
            info = PercussionInfo(
                events=events,
                pattern_length=self._calculate_pattern_length(events),
                density=self._calculate_density(events),
                complexity=self._analyze_complexity(events, patterns),
                patterns=patterns,
                timing_stability=self._analyze_timing_stability(events),
                ghost_notes=self._detect_ghost_notes(audio, events)
            )

            return self._convert_to_dict(info)

        except Exception as e:
            print(f"Error in percussion analysis: {str(e)}")
            return self._get_fallback_features()

    def _get_fallback_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'events': [],
            'pattern_length': 0.0,
            'density': 0.0,
            'complexity': 0.0,
            'patterns': {},
            'timing_stability': 0.0,
            'ghost_notes': []
        }

    def _convert_to_dict(self, info: PercussionInfo) -> Dict:
        """Convert PercussionInfo to dictionary with JSON-serializable values."""
        return {
            'events': [
                {
                    'type': e.type,
                    'time': float(e.time),
                    'confidence': float(e.confidence),
                    'velocity': float(e.velocity)
                }
                for e in info.events
            ],
            'pattern_length': float(info.pattern_length),
            'density': float(info.density),
            'complexity': float(info.complexity),
            'patterns': {
                k: v.tolist() for k, v in info.patterns.items()
            },
            'timing_stability': float(info.timing_stability),
            'ghost_notes': [
                {
                    'type': e.type,
                    'time': float(e.time),
                    'confidence': float(e.confidence),
                    'velocity': float(e.velocity)
                }
                for e in info.ghost_notes
            ]
        }

    def _combine_segment_results(self, results: List[Dict]) -> Dict:
        """Combine results from multiple segments."""
        if not results:
            return self._get_fallback_features()

        # Merge events and ghost notes
        all_events = []
        all_ghost_notes = []
        for r in results:
            all_events.extend(r['events'])
            all_ghost_notes.extend(r['ghost_notes'])

        # Sort by time
        all_events.sort(key=lambda x: x['time'])
        all_ghost_notes.sort(key=lambda x: x['time'])

        # Combine patterns
        combined_patterns = {}
        for r in results:
            for drum_type, pattern in r['patterns'].items():
                if drum_type not in combined_patterns:
                    combined_patterns[drum_type] = []
                combined_patterns[drum_type].extend(pattern)

        # Average metrics
        metrics = ['pattern_length', 'density', 'complexity', 'timing_stability']
        averaged_metrics = {
            metric: np.mean([r[metric] for r in results])
            for metric in metrics
        }

        return {
            'events': all_events,
            'ghost_notes': all_ghost_notes,
            'patterns': combined_patterns,
            **averaged_metrics
        }

    def _detect_drum_events(self, audio, start_time):
        events = []

        for drum_type, (low_freq, high_freq) in self.drum_ranges.items():
            # Filter audio to drum frequency range
            y_filtered = librosa.effects.trim(
                librosa.filtfilt(
                    audio,
                    self.sr,
                    low_freq,
                    high_freq
                )
            )[0]

            # Get onset envelope
            onset_env = librosa.onset.onset_strength(
                y=y_filtered,
                sr=self.sr,
                hop_length=self.hop_length
            )

            # Find peaks in onset envelope
            peaks, properties = find_peaks(
                onset_env,
                distance=int(self.min_interval[drum_type] * self.sr / self.hop_length),
                height=0.1,
                prominence=0.1
            )

            # Convert peaks to drum events
            for peak, prom in zip(peaks, properties['prominences']):
                time = librosa.frames_to_time(peak, sr=self.sr, hop_length=self.hop_length) + start_time
                confidence = min(1.0, prom / np.max(properties['prominences']))
                velocity = onset_env[peak]

                events.append(DrumEvent(
                    type=drum_type,
                    time=time,
                    confidence=confidence,
                    velocity=velocity
                ))

        # Sort events by time
        events.sort(key=lambda x: x.time)
        return events

    def _extract_patterns(self, events: List[DrumEvent]) -> Dict[str, np.ndarray]:
        patterns = {}

        for drum_type in self.drum_ranges.keys():
            # Get events for this drum type
            type_events = [e for e in events if e.type == drum_type]

            if not type_events:
                continue

            # Convert to onset sequence
            times = np.array([e.time for e in type_events])
            velocities = np.array([e.velocity for e in type_events])

            # Find pattern length through autocorrelation
            if len(times) > 1:
                intervals = np.diff(times)
                pattern_length = self._find_pattern_length(intervals)

                # Create grid representation of pattern
                grid = np.zeros(32)  # 32nd note resolution
                positions = ((times % pattern_length) / pattern_length * 32).astype(int)
                for pos, vel in zip(positions, velocities):
                    if pos < 32:
                        grid[pos] = vel

                patterns[drum_type] = grid

        return patterns

    def _find_pattern_length(self, intervals):
        if len(intervals) < 2:
            return 1.0

        # Use autocorrelation to find repeating pattern
        ac = librosa.autocorrelate(intervals)
        peaks = find_peaks(ac)[0]

        if len(peaks) > 0:
            return intervals[:peaks[0]].sum()
        return intervals.mean() * 4  # fallback to average bar length

    def _calculate_pattern_length(self, events):
        if not events:
            return 0.0

        # Find most common interval between similar events
        pattern_lengths = []

        for drum_type in self.drum_ranges.keys():
            type_events = [e for e in events if e.type == drum_type]
            if len(type_events) > 1:
                intervals = np.diff([e.time for e in type_events])
                pattern_lengths.append(self._find_pattern_length(intervals))

        return np.median(pattern_lengths) if pattern_lengths else 0.0

    def _calculate_density(self, events):
        if not events:
            return 0.0

        # Calculate events per second
        duration = max(e.time for e in events)
        return len(events) / duration if duration > 0 else 0.0

    def _analyze_complexity(self, events, patterns):
        if not events or not patterns:
            return 0.0

        complexity_scores = []

        for pattern in patterns.values():
            # Calculate pattern entropy
            hist, _ = np.histogram(pattern, bins=8, density=True)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))

            # Calculate syncopation
            strong_beats = pattern[::8]  # Quarter notes
            weak_beats = np.delete(pattern, slice(None, None, 8))
            syncopation = np.mean(weak_beats) / (np.mean(strong_beats) + 1e-8)

            complexity_scores.append(entropy * syncopation)

        return np.mean(complexity_scores)

    def _analyze_timing_stability(self, events):
        if not events:
            return 0.0

        stability_scores = []

        for drum_type in self.drum_ranges.keys():
            type_events = [e for e in events if e.type == drum_type]
            if len(type_events) > 1:
                times = np.array([e.time for e in type_events])
                intervals = np.diff(times)

                # Calculate coefficient of variation of intervals
                cv = np.std(intervals) / np.mean(intervals)
                stability_scores.append(1.0 / (1.0 + cv))

        return np.mean(stability_scores) if stability_scores else 0.0

    def _detect_ghost_notes(self, audio, events):
        ghost_notes = []

        for drum_type in ['snare', 'hihat']:
            main_events = [e for e in events if e.type == drum_type]
            if not main_events:
                continue

            # Get average velocity for this drum type
            avg_velocity = np.mean([e.velocity for e in main_events])

            # Find quieter hits that might be ghost notes
            low_freq, high_freq = self.drum_ranges[drum_type]
            y_filtered = librosa.effects.trim(
                librosa.filtfilt(
                    audio,
                    self.sr,
                    low_freq,
                    high_freq
                )
            )[0]

            onset_env = librosa.onset.onset_strength(
                y=y_filtered,
                sr=self.sr,
                hop_length=self.hop_length
            )

            peaks, properties = find_peaks(
                onset_env,
                height=0.05 * avg_velocity,  # Lower threshold for ghost notes
                prominence=0.05
            )

            for peak, prom in zip(peaks, properties['prominences']):
                time = librosa.frames_to_time(peak, sr=self.sr, hop_length=self.hop_length)

                # Check if this is not already a main hit
                if not any(abs(e.time - time) < 0.03 for e in main_events):
                    velocity = onset_env[peak]
                    if velocity < 0.7 * avg_velocity:  # Ghost note threshold
                        ghost_notes.append(DrumEvent(
                            type=f"ghost_{drum_type}",
                            time=time,
                            confidence=prom / np.max(properties['prominences']),
                            velocity=velocity
                        ))

        return ghost_notes
