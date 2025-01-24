import numpy as np
import torch
import librosa
from scipy.signal import butter, filtfilt, find_peaks
from dataclasses import dataclass
from typing import Dict


@dataclass
class BasslineInfo:
    energy: np.ndarray
    peak_frequencies: np.ndarray
    intensity: float
    pattern_strength: float


class BasslineAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.bass_freq_range = (20, 250)  # Hz

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze bassline characteristics in the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing bassline features
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Extract bass frequencies
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Get bass frequency mask
        bass_mask = (freqs >= self.bass_freq_range[0]) & (freqs <= self.bass_freq_range[1])
        bass_spec = np.abs(D[bass_mask])
        
        # Compute features
        bass_energy = np.mean(bass_spec ** 2, axis=0)
        bass_variance = np.var(bass_spec, axis=0)
        bass_peaks = self._detect_bass_peaks(bass_energy)
        
        return {
            'energy': bass_energy,
            'variance': bass_variance,
            'peaks': bass_peaks,
            'mean_energy': float(np.mean(bass_energy)),
            'peak_count': len(bass_peaks)
        }
    
    def _detect_bass_peaks(self, bass_energy: np.ndarray) -> np.ndarray:
        """Detect significant peaks in bass energy."""
        peaks = librosa.util.peak_pick(
            bass_energy,
            pre_max=20,
            post_max=20,
            pre_avg=20,
            post_avg=20,
            delta=0.5,
            wait=20
        )
        return peaks

    def _bandpass_filter(self, audio):
        nyquist = self.sr // 2
        b, a = butter(4, [self.low_cut / nyquist, self.high_cut / nyquist], btype='band')
        return filtfilt(b, a, audio)

    def _find_peak_frequencies(self, spec):
        return librosa.hz_to_midi(
            librosa.frequency_to_hz(
                np.argmax(spec[:50], axis=0)
            )
        )

    def _calculate_pattern_strength(self, energy):
        ac = librosa.autocorrelate(energy)
        peaks = find_peaks(ac)[0]
        if len(peaks) > 0:
            return np.max(ac[peaks]) / ac[0]
        return 0.0
