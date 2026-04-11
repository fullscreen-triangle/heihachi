"""
Signal Recording - Captures audio signals and prepares them for
categorical analysis. Handles loading, preprocessing, and segmentation.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class RecordedSignal:
    """A recorded or loaded audio signal ready for categorical analysis."""
    signal: np.ndarray
    sample_rate: int
    duration: float
    n_samples: int
    source_path: Optional[str]
    metadata: Dict[str, object]


class SignalRecorder:
    """Loads, preprocesses, and segments audio signals for categorical analysis.

    Supports:
    - Loading from audio files (WAV, FLAC, MP3 via librosa)
    - Generating synthetic test signals
    - Segmenting long recordings into analysis frames
    - Normalizing and preprocessing
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def load(
        self,
        path: str,
        duration: Optional[float] = None,
        offset: float = 0.0,
        mono: bool = True,
    ) -> RecordedSignal:
        """Load an audio file."""
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for audio file loading")

        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        signal, sr = librosa.load(
            str(filepath),
            sr=self.sample_rate,
            mono=mono,
            duration=duration,
            offset=offset,
        )

        return RecordedSignal(
            signal=signal,
            sample_rate=sr,
            duration=len(signal) / sr,
            n_samples=len(signal),
            source_path=str(filepath),
            metadata={
                'filename': filepath.name,
                'format': filepath.suffix,
                'mono': mono,
            },
        )

    def generate_sine(
        self,
        frequency: float = 440.0,
        duration: float = 1.0,
        amplitude: float = 1.0,
        phase: float = 0.0,
    ) -> RecordedSignal:
        """Generate a pure sine wave test signal."""
        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        return RecordedSignal(
            signal=signal,
            sample_rate=self.sample_rate,
            duration=duration,
            n_samples=len(signal),
            source_path=None,
            metadata={'type': 'sine', 'frequency': frequency, 'amplitude': amplitude},
        )

    def generate_multimode(
        self,
        frequencies: List[float],
        amplitudes: Optional[List[float]] = None,
        duration: float = 1.0,
    ) -> RecordedSignal:
        """Generate a multi-mode test signal (sum of sinusoids)."""
        if amplitudes is None:
            amplitudes = [1.0 / (i + 1) for i in range(len(frequencies))]

        t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
        signal = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))

        return RecordedSignal(
            signal=signal,
            sample_rate=self.sample_rate,
            duration=duration,
            n_samples=len(signal),
            source_path=None,
            metadata={'type': 'multimode', 'frequencies': frequencies, 'amplitudes': amplitudes},
        )

    def generate_amen_pattern(self, duration: float = 2.0, bpm: float = 136.7) -> RecordedSignal:
        """Generate a synthetic amen-break-like pattern for testing.

        Creates a drum pattern with kick, snare, and hi-hat at the
        characteristic amen break timing.
        """
        n_samples = int(duration * self.sample_rate)
        signal = np.zeros(n_samples)
        t = np.arange(n_samples) / self.sample_rate

        beat_dur = 60.0 / bpm
        sixteenth = beat_dur / 4

        # Amen break pattern (simplified): K=kick, S=snare, H=hihat
        # Pattern: K-H-S-H-K-K-S-H-K-H-S-H-K-H-S-H (one bar of 16ths)
        kick_times = [0, 4, 5, 8, 12]      # 16th note positions
        snare_times = [2, 6, 10, 14]
        hihat_times = [1, 3, 7, 9, 11, 13, 15]

        n_bars = int(duration / (beat_dur * 4)) + 1

        for bar in range(n_bars):
            bar_offset = bar * beat_dur * 4

            for pos in kick_times:
                onset = bar_offset + pos * sixteenth
                if onset < duration:
                    idx = int(onset * self.sample_rate)
                    # Kick: low frequency burst
                    kick_len = min(int(0.05 * self.sample_rate), n_samples - idx)
                    if kick_len > 0:
                        t_kick = np.arange(kick_len) / self.sample_rate
                        kick = np.sin(2 * np.pi * 60 * t_kick) * np.exp(-t_kick * 40)
                        signal[idx:idx + kick_len] += kick * 0.8

            for pos in snare_times:
                onset = bar_offset + pos * sixteenth
                if onset < duration:
                    idx = int(onset * self.sample_rate)
                    snare_len = min(int(0.04 * self.sample_rate), n_samples - idx)
                    if snare_len > 0:
                        t_snare = np.arange(snare_len) / self.sample_rate
                        snare = (
                            np.sin(2 * np.pi * 200 * t_snare) * np.exp(-t_snare * 50)
                            + np.random.randn(snare_len) * 0.3 * np.exp(-t_snare * 30)
                        )
                        signal[idx:idx + snare_len] += snare * 0.6

            for pos in hihat_times:
                onset = bar_offset + pos * sixteenth
                if onset < duration:
                    idx = int(onset * self.sample_rate)
                    hh_len = min(int(0.02 * self.sample_rate), n_samples - idx)
                    if hh_len > 0:
                        t_hh = np.arange(hh_len) / self.sample_rate
                        hihat = np.random.randn(hh_len) * np.exp(-t_hh * 100)
                        signal[idx:idx + hh_len] += hihat * 0.3

        # Normalize
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal /= max_val

        return RecordedSignal(
            signal=signal,
            sample_rate=self.sample_rate,
            duration=duration,
            n_samples=n_samples,
            source_path=None,
            metadata={'type': 'amen_pattern', 'bpm': bpm},
        )

    def segment(
        self,
        recorded: RecordedSignal,
        segment_duration: float = 0.5,
        hop_duration: float = 0.25,
    ) -> List[RecordedSignal]:
        """Segment a recording into overlapping analysis frames."""
        seg_samples = int(segment_duration * recorded.sample_rate)
        hop_samples = int(hop_duration * recorded.sample_rate)

        segments = []
        start = 0
        while start + seg_samples <= recorded.n_samples:
            seg_signal = recorded.signal[start:start + seg_samples]
            segments.append(RecordedSignal(
                signal=seg_signal,
                sample_rate=recorded.sample_rate,
                duration=segment_duration,
                n_samples=seg_samples,
                source_path=recorded.source_path,
                metadata={
                    **recorded.metadata,
                    'segment_start': start / recorded.sample_rate,
                    'segment_index': len(segments),
                },
            ))
            start += hop_samples

        return segments

    def normalize(self, recorded: RecordedSignal) -> RecordedSignal:
        """Peak-normalize a recorded signal to [-1, 1]."""
        max_val = np.max(np.abs(recorded.signal))
        if max_val > 0:
            normalized = recorded.signal / max_val
        else:
            normalized = recorded.signal.copy()

        return RecordedSignal(
            signal=normalized,
            sample_rate=recorded.sample_rate,
            duration=recorded.duration,
            n_samples=recorded.n_samples,
            source_path=recorded.source_path,
            metadata={**recorded.metadata, 'normalized': True},
        )
