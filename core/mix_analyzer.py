import numpy as np
import torch
from concurrent.futures import  ThreadPoolExecutor
from typing import Dict, List, Tuple
from dataclasses import dataclass



@dataclass
class MixFeatures:
    spectral: torch.Tensor
    temporal: torch.Tensor
    rhythmic: torch.Tensor
    alignment: float
    segment_info: Dict


class MixAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = 44100 * 10  # 10 second chunks
        self.hop_length = 44100 * 5  # 5 second overlap
        self.n_workers = 4

    def analyze(self, audio: np.ndarray, sr: int = 44100) -> Dict:
        """Analyze audio mix.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            sr (int, optional): Sample rate. Defaults to 44100.
            
        Returns:
            Dict: Analysis results containing spectral, temporal, rhythmic and global features
        """
        return self.analyze_mix(audio, sr)

    @torch.no_grad()
    def analyze_mix(self, audio: np.ndarray, sr: int = 44100) -> Dict:
        audio_tensor = torch.from_numpy(audio).to(self.device)
        chunks = self._split_into_chunks(audio_tensor)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            spectral_future = executor.submit(self._analyze_spectral, chunks)
            temporal_future = executor.submit(self._analyze_temporal, chunks)
            rhythmic_future = executor.submit(self._analyze_rhythmic, chunks)

        return {
            'spectral': spectral_future.result(),
            'temporal': temporal_future.result(),
            'rhythmic': rhythmic_future.result(),
            'global_features': self._compute_global_features(chunks)
        }

    def _split_into_chunks(self, audio: torch.Tensor) -> List[torch.Tensor]:
        chunks = []
        for i in range(0, len(audio) - self.chunk_size, self.hop_length):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                chunks.append(chunk)
        return chunks

    def _analyze_spectral(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        features = []
        for chunk in chunks:
            stft = torch.stft(chunk, n_fft=2048, hop_length=512)
            mag = torch.abs(stft)
            features.append(self._compute_spectral_features(mag))
        return torch.stack(features)

    def _analyze_temporal(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        features = []
        for chunk in chunks:
            rms = torch.sqrt(torch.mean(chunk ** 2))
            zcr = torch.mean((chunk[:-1] * chunk[1:]) < 0).float()
            features.append(torch.stack([rms, zcr]))
        return torch.stack(features)

    def _analyze_rhythmic(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        features = []
        for chunk in chunks:
            onset_env = self._compute_onset_envelope(chunk)
            tempo_features = self._extract_tempo_features(onset_env)
            features.append(tempo_features)
        return torch.stack(features)

    def _compute_spectral_features(self, mag: torch.Tensor) -> torch.Tensor:
        centroid = torch.sum(mag * torch.arange(mag.shape[1], device=self.device)) / torch.sum(mag)
        bandwidth = torch.sqrt(
            torch.sum(((torch.arange(mag.shape[1], device=self.device) - centroid) ** 2) * mag) / torch.sum(mag))
        flatness = torch.exp(torch.mean(torch.log(mag + 1e-6))) / torch.mean(mag)
        return torch.stack([centroid, bandwidth, flatness])

    def _compute_onset_envelope(self, audio: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(audio, n_fft=2048, hop_length=512)
        mag = torch.abs(stft)
        return torch.sum(torch.diff(mag, dim=1) ** 2, dim=0)

    def _extract_tempo_features(self, onset_env: torch.Tensor) -> torch.Tensor:
        acf = torch.correlate(onset_env, onset_env, mode='full')
        acf = acf[len(acf) // 2:]
        peaks = torch.where(torch.diff(torch.sign(torch.diff(acf))) < 0)[0] + 1
        if len(peaks) > 0:
            tempo = 60 * 44100 / (512 * peaks[0])
            tempo_strength = acf[peaks[0]]
        else:
            tempo, tempo_strength = torch.tensor(0.), torch.tensor(0.)
        return torch.stack([tempo, tempo_strength])

    def _compute_global_features(self, chunks: List[torch.Tensor]) -> Dict:
        all_features = []
        for chunk in chunks:
            features = {
                'loudness': torch.mean(torch.abs(chunk)),
                'dynamic_range': torch.max(chunk) - torch.min(chunk),
                'crest_factor': torch.max(torch.abs(chunk)) / torch.sqrt(torch.mean(chunk ** 2))
            }
            all_features.append(features)

        return {k: torch.mean(torch.tensor([f[k] for f in all_features]))
                for k in all_features[0].keys()}
