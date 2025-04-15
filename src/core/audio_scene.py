import numpy as np
import torch
from sklearn.decomposition import NMF
import librosa
from typing import Dict, List, Tuple


class AudioSceneAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.nmf = NMF(n_components=4, max_iter=200)
        self.sr = 44100
        self.chunk_size = 44100 * 5  # 5 seconds
        self.hop_length = 44100 * 2  # 2 seconds overlap
        self.n_fft = 2048

    def analyze(self, audio: np.ndarray) -> Dict:
        """Analyze audio scene characteristics.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            
        Returns:
            Dict: Analysis results containing spatial characteristics and background analysis
        """
        chunks = self._split_chunks(audio)
        spatial_analysis = self._analyze_spatial(audio)
        background = self._extract_background(audio)
        
        return {
            'spatial': spatial_analysis,
            'background': background,
            'chunks': len(chunks)
        }

    def _split_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into overlapping chunks for parallel processing."""
        chunks = []
        for i in range(0, len(audio) - self.chunk_size, self.hop_length):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:  # Only use complete chunks
                chunks.append(chunk)
        return chunks

    def _reconstruct_source(self, W: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Reconstruct audio source from NMF components."""
        # Reconstruct spectrogram
        S_reconstructed = np.outer(W, H)

        # Get phase from original STFT
        phase = np.exp(2j * np.pi * np.random.rand(*S_reconstructed.shape))

        # Inverse STFT
        reconstructed = librosa.istft(S_reconstructed * phase,
                                      hop_length=self.n_fft // 4,
                                      win_length=self.n_fft)

        return reconstructed

    def _analyze_spatial(self, audio: np.ndarray) -> Dict:
        """Analyze spatial characteristics of the audio."""
        if audio.ndim == 1:
            return self._analyze_mono_spatial(audio)
        else:
            return self._analyze_stereo_spatial(audio)

    def _analyze_mono_spatial(self, audio: np.ndarray) -> Dict:
        """Analyze spatial characteristics of mono audio."""
        # Compute spectral centroid
        centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)

        # Compute spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)

        # Compute spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)

        return {
            'centroid': np.mean(centroid),
            'bandwidth': np.mean(bandwidth),
            'rolloff': np.mean(rolloff)
        }

    def _analyze_stereo_spatial(self, audio: np.ndarray) -> Dict:
        """Analyze spatial characteristics of stereo audio."""
        # Compute phase correlation between channels
        correlation = np.correlate(audio[0], audio[1], mode='full')

        # Compute stereo width
        width = self._compute_stereo_width(audio)

        # Compute pan position
        pan = self._compute_pan_position(audio)

        return {
            'correlation': np.max(correlation),
            'width': width,
            'pan': pan
        }

    def _compute_stereo_width(self, audio: np.ndarray) -> float:
        """Compute stereo width of audio."""
        # Compute mid and side channels
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2

        # Compute RMS of mid and side
        mid_rms = np.sqrt(np.mean(mid ** 2))
        side_rms = np.sqrt(np.mean(side ** 2))

        # Calculate width as ratio of side to mid
        width = side_rms / (mid_rms + 1e-8)

        return float(width)

    def _compute_pan_position(self, audio: np.ndarray) -> float:
        """Compute overall pan position of audio."""
        # Compute RMS of left and right channels
        left_rms = np.sqrt(np.mean(audio[0] ** 2))
        right_rms = np.sqrt(np.mean(audio[1] ** 2))

        # Calculate pan position (-1 to 1)
        pan = (right_rms - left_rms) / (right_rms + left_rms + 1e-8)

        return float(pan)

    def _extract_background(self, audio: np.ndarray) -> Dict:
        """Extract background components from audio."""
        # Compute STFT
        D = librosa.stft(audio, n_fft=self.n_fft)
        S = np.abs(D)

        # Median filtering for background estimation
        background_stft = self._median_filter_stft(S)

        # Reconstruct background audio
        background_audio = librosa.istft(background_stft * np.exp(1j * np.angle(D)))

        # Compute background features
        background_features = self._analyze_background_features(background_audio)

        return {
            'audio': background_audio,
            'features': background_features
        }

    def _median_filter_stft(self, S: np.ndarray) -> np.ndarray:
        """Apply median filtering to STFT magnitude."""
        filtered = np.zeros_like(S)
        kernel_size = 11

        for i in range(S.shape[0]):
            filtered[i] = np.convolve(S[i],
                                      np.ones(kernel_size) / kernel_size,
                                      mode='same')

        return filtered

    def _analyze_background_features(self, background: np.ndarray) -> Dict:
        """Analyze features of extracted background."""
        # Compute spectral features
        spectral = {
            'flatness': np.mean(librosa.feature.spectral_flatness(y=background)),
            'rolloff': np.mean(librosa.feature.spectral_rolloff(y=background, sr=self.sr)),
            'contrast': np.mean(librosa.feature.spectral_contrast(y=background, sr=self.sr))
        }

        # Compute temporal features
        temporal = {
            'rms': np.sqrt(np.mean(background ** 2)),
            'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(background))
        }

        return {
            'spectral': spectral,
            'temporal': temporal
        }

    def _aggregate_features(self, features: List[Dict]) -> Dict:
        """Aggregate features from multiple chunks."""
        aggregated = {
            'sources': [],
            'spatial': {
                'mean': {},
                'std': {}
            },
            'background': {
                'mean': {},
                'std': {}
            }
        }

        # Aggregate sources
        for chunk_features in features:
            aggregated['sources'].extend(chunk_features['sources'].values())

        # Aggregate spatial features
        spatial_features = [f['spatial'] for f in features]
        for key in spatial_features[0].keys():
            values = [f[key] for f in spatial_features]
            aggregated['spatial']['mean'][key] = np.mean(values)
            aggregated['spatial']['std'][key] = np.std(values)

        # Aggregate background features
        bg_features = [f['background']['features'] for f in features]
        for category in ['spectral', 'temporal']:
            for key in bg_features[0][category].keys():
                values = [f[category][key] for f in bg_features]
                aggregated['background']['mean'][f"{category}_{key}"] = np.mean(values)
                aggregated['background']['std'][f"{category}_{key}"] = np.std(values)

        return aggregated
