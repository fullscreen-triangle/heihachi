import numpy as np
import librosa
from dataclasses import dataclass
from typing import Dict


@dataclass
class BPMInfo:
    bpm: float
    confidence: float
    stability: float
    beat_positions: np.ndarray


class BPMAnalyzer:
    def __init__(self):
        self.sr = 44100  # Match pipeline sample rate
        self.hop_length = 512

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze BPM characteristics of the audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing BPM information
        """
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        onset_env = librosa.onset.onset_strength(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Get tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length
        )

        # Calculate confidence and stability
        confidence = self._calculate_confidence(onset_env, beat_frames)
        stability = self._calculate_stability(beat_frames)

        info = BPMInfo(
            bpm=float(tempo),
            confidence=float(confidence),
            stability=float(stability),
            beat_positions=librosa.frames_to_time(beat_frames, sr=self.sr)
        )

        return {
            'bpm': info.bpm,
            'confidence': info.confidence,
            'stability': info.stability,
            'beat_positions': info.beat_positions.tolist()
        }

    def _calculate_confidence(self, onset_env, beat_frames):
        beat_strengths = onset_env[beat_frames]
        return np.mean(beat_strengths) / np.mean(onset_env)

    def _calculate_stability(self, beat_frames):
        intervals = np.diff(beat_frames)
        return 1.0 - (np.std(intervals) / np.mean(intervals))
