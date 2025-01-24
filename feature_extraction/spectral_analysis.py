import librosa
import torch
import numpy as np
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor


class SpectralFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sr = 44100
        self.n_fft = 2048
        self.hop_length = 512
        self.segment_duration = 30  # 30-second segments
        self.overlap = 5  # 5-second overlap
        self.n_mels = 128
        self.n_mfcc = 20

    def analyze(self, audio: np.ndarray) -> Dict:
        """Extract spectral features from audio.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing spectral features
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
            # Convert to tensor and move to GPU
            audio_tensor = torch.from_numpy(audio).to(self.device)

            # Compute STFT
            stft = torch.stft(
                audio_tensor,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            mag = torch.abs(stft)

            features = {
                'centroid': self._spectral_centroid(mag),
                'bandwidth': self._spectral_bandwidth(mag),
                'flatness': self._spectral_flatness(mag),
                'rolloff': self._spectral_rolloff(mag),
                'contrast': self._spectral_contrast(mag),
                'mel_features': self._mel_features(mag),
                'mfcc': self._mfcc(mag),
                'chroma': self._chroma_features(mag)
            }

            return self._validate_features(features)

        except Exception as e:
            print(f"Error in spectral analysis: {str(e)}")
            return self._get_fallback_features()

    def _get_fallback_features(self) -> Dict:
        """Return default features when analysis fails."""
        return {
            'centroid': np.zeros(128).tolist(),
            'bandwidth': np.zeros(128).tolist(),
            'flatness': np.zeros(128).tolist(),
            'rolloff': np.zeros(128).tolist(),
            'contrast': np.zeros((6, 128)).tolist(),
            'mel_features': np.zeros((self.n_mels, 128)).tolist(),
            'mfcc': np.zeros((self.n_mfcc, 128)).tolist(),
            'chroma': np.zeros((12, 128)).tolist()
        }

    def _validate_features(self, features: Dict) -> Dict:
        """Validate and clean feature values."""
        validated = {}
        for key, value in features.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                # Convert to numpy if tensor
                if isinstance(value, torch.Tensor):
                    value = value.cpu().numpy()
                # Remove NaN/Inf values
                value = np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=0.0)
                validated[key] = value.tolist()
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
            # Stack arrays along time axis
            arrays = [np.array(r[key]) for r in results]
            if len(arrays[0].shape) == 1:
                # For 1D features, compute statistics
                values = np.concatenate(arrays)
                combined[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'max': float(np.max(values)),
                    'min': float(np.min(values))
                }
            else:
                # For 2D features, concatenate along time axis and compute statistics
                values = np.concatenate(arrays, axis=1)
                combined[key] = {
                    'mean': np.mean(values, axis=1).tolist(),
                    'std': np.std(values, axis=1).tolist(),
                    'max': np.max(values, axis=1).tolist(),
                    'min': np.min(values, axis=1).tolist()
                }

        return combined

    def _spectral_centroid(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute spectral centroid."""
        freqs = torch.linspace(0, 1, mag.shape[0], device=self.device)
        norm_mag = mag / (torch.sum(mag, dim=0, keepdim=True) + 1e-8)
        centroid = torch.sum(norm_mag * freqs.unsqueeze(1), dim=0)
        return centroid

    def _spectral_bandwidth(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute spectral bandwidth."""
        centroid = self._spectral_centroid(mag)
        freqs = torch.linspace(0, 1, mag.shape[0], device=self.device)
        norm_mag = mag / (torch.sum(mag, dim=0, keepdim=True) + 1e-8)
        bandwidth = torch.sqrt(
            torch.sum(norm_mag * (freqs.unsqueeze(1) - centroid.unsqueeze(0)) ** 2, dim=0)
        )
        return bandwidth

    def _spectral_flatness(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute spectral flatness."""
        geometric_mean = torch.exp(torch.mean(torch.log(mag + 1e-8), dim=0))
        arithmetic_mean = torch.mean(mag, dim=0)
        flatness = geometric_mean / (arithmetic_mean + 1e-8)
        return flatness

    def _spectral_rolloff(self, mag: torch.Tensor, percentile: float = 0.85) -> torch.Tensor:
        """Compute spectral rolloff."""
        total_energy = torch.cumsum(mag, dim=0)
        threshold = percentile * total_energy[-1]
        indices = torch.argmax(total_energy >= threshold.unsqueeze(0), dim=0)
        return indices.float() / mag.shape[0]

    def _spectral_contrast(self, mag: torch.Tensor, n_bands: int = 6) -> torch.Tensor:
        """Compute spectral contrast."""
        bands = torch.chunk(mag, n_bands, dim=0)
        contrasts = []

        for band in bands:
            peak = torch.max(band, dim=0)[0]
            valley = torch.min(band, dim=0)[0]
            contrast = torch.log(peak / (valley + 1e-8))
            contrasts.append(contrast)

        return torch.stack(contrasts)

    def _mel_features(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram."""
        mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )
        mel_basis = torch.from_numpy(mel_basis).to(self.device)
        mel_spec = torch.matmul(mel_basis, mag)
        return torch.log(mel_spec + 1e-8)

    def _mfcc(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute MFCC."""
        mel_spec = self._mel_features(mag)
        mfcc = torch.from_numpy(
            librosa.feature.mfcc(
                S=mel_spec.cpu().numpy(),
                n_mfcc=self.n_mfcc
            )
        ).to(self.device)
        return mfcc

    def _chroma_features(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute chroma features."""
        chroma = torch.from_numpy(
            librosa.feature.chroma_stft(
                S=mag.cpu().numpy(),
                sr=self.sr,
                n_fft=self.n_fft
            )
        ).to(self.device)
        return chroma
