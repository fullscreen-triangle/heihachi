import numpy as np
import librosa
import torch
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor
from typing import Dict


class DrumAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100  # Match pipeline sample rate

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze drum patterns in the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing drum patterns and characteristics
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        return self.analyze_drums(audio)

    def analyze_drums(self, audio):
        harmonic, percussive = librosa.effects.hpss(audio)

        with ThreadPoolExecutor() as executor:
            futures = {
                'kick': executor.submit(self._analyze_kick, percussive),
                'snare': executor.submit(self._analyze_snare, percussive),
                'hihat': executor.submit(self._analyze_hihat, percussive),
                'pattern': executor.submit(self._analyze_pattern, percussive)
            }

        results = {k: v.result() for k, v in futures.items()}
        
        # Convert numpy arrays to lists for JSON serialization
        for k, v in results.items():
            if k != 'pattern':
                results[k]['times'] = results[k]['times'].tolist()
        results['pattern'] = results['pattern'].tolist()
        
        return results

    def _analyze_kick(self, audio):
        spec = np.abs(librosa.stft(audio))
        low_freq = spec[:20, :]  # Focus on low frequencies
        kick_times = find_peaks(low_freq.mean(axis=0))[0]
        return {
            'times': kick_times,
            'strength': spec[:20, kick_times].mean(),
            'regularity': np.std(np.diff(kick_times))
        }

    def _analyze_snare(self, audio):
        spec = np.abs(librosa.stft(audio))
        mid_freq = spec[20:50, :]  # Focus on mid frequencies
        snare_times = find_peaks(mid_freq.mean(axis=0))[0]
        return {
            'times': snare_times,
            'strength': spec[20:50, snare_times].mean(),
            'regularity': np.std(np.diff(snare_times))
        }

    def _analyze_hihat(self, audio):
        spec = np.abs(librosa.stft(audio))
        high_freq = spec[50:, :]  # Focus on high frequencies
        hihat_times = find_peaks(high_freq.mean(axis=0))[0]
        return {
            'times': hihat_times,
            'strength': spec[50:, hihat_times].mean(),
            'regularity': np.std(np.diff(hihat_times))
        }

    def _analyze_pattern(self, audio):
        onset_env = librosa.onset.onset_strength(y=audio)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env)
        pattern = np.zeros((len(beats), 3))  # kick, snare, hihat

        for i, beat in enumerate(beats):
            window = onset_env[max(0, beat - 2):min(len(onset_env), beat + 3)]
            pattern[i, 0] = np.max(window[:2])  # kick
            pattern[i, 1] = np.max(window[1:3])  # snare
            pattern[i, 2] = np.max(window[2:])  # hihat

        return pattern
