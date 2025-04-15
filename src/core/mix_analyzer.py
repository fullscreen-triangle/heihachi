import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import gc
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MixFeatures:
    spectral: torch.Tensor
    temporal: torch.Tensor
    rhythmic: torch.Tensor
    alignment: float
    segment_info: Dict


class MixAnalyzer:
    def __init__(self, use_gpu: bool = True):
        """Initialize the mix analyzer with configurable GPU usage and chunk sizes.
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.default_chunk_seconds = 10  # Default 10 second chunks
        self.default_hop_ratio = 0.5     # Default 50% overlap
        
        # Log device info
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
        self.n_workers = 4

    def analyze(self, audio: np.ndarray, sr: int = 44100) -> Dict:
        """Analyze audio mix with automatic chunk size adjustment.
        
        Args:
            audio (np.ndarray): Audio data to analyze
            sr (int, optional): Sample rate. Defaults to 44100.
            
        Returns:
            Dict: Analysis results containing spectral, temporal, rhythmic and global features
        """
        # Automatically adjust chunk size based on audio length and device memory
        self._adjust_chunk_size(audio, sr)
        
        return self.analyze_mix(audio, sr)

    def _adjust_chunk_size(self, audio: np.ndarray, sr: int) -> None:
        """Automatically adjust chunk and hop size based on audio length and available memory.
        
        Args:
            audio (np.ndarray): Audio data
            sr (int): Sample rate
        """
        audio_duration = len(audio) / sr
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # For very short clips, analyze the whole thing
        if audio_duration < 30:
            self.chunk_size = len(audio)
            self.hop_length = len(audio)
            logger.info(f"Short audio: using entire file as single chunk")
            return
            
        # For GPU memory constraint: estimate max chunk size
        if self.device.type == 'cuda':
            # Estimate available memory (70% of total to be safe)
            available_memory = torch.cuda.get_device_properties(0).total_memory * 0.7
            
            # Estimate memory needed per sample (conservatively)
            # Each sample requires about 4 bytes for float32, plus overhead for processing
            bytes_per_sample = 20  # Conservative estimate including all processing
            
            # Calculate max samples that can fit in memory
            max_samples = int(available_memory / bytes_per_sample)
            
            # Convert to seconds
            max_chunk_seconds = max_samples / sr
            logger.info(f"GPU memory constraint: max chunk size = {max_chunk_seconds:.2f} seconds")
            
            # Cap chunk size
            chunk_seconds = min(self.default_chunk_seconds, max_chunk_seconds)
        else:
            # On CPU, we can use larger chunks but still be reasonable
            chunk_seconds = min(self.default_chunk_seconds, audio_duration / 10)
            
        # For long audio, adjust hop length to ensure reasonable number of chunks
        if audio_duration > 600:  # >10 min
            # For very long files, use less overlap
            hop_ratio = 0.8  # 20% overlap
        else:
            hop_ratio = self.default_hop_ratio
            
        # Calculate chunk and hop size in samples
        self.chunk_size = int(chunk_seconds * sr)
        self.hop_length = int(self.chunk_size * hop_ratio)
        
        logger.info(f"Using chunk size: {chunk_seconds:.2f} seconds ({self.chunk_size} samples)")
        logger.info(f"Using hop length: {self.hop_length / sr:.2f} seconds ({self.hop_length} samples)")

    @torch.no_grad()
    def analyze_mix(self, audio: np.ndarray, sr: int = 44100) -> Dict:
        """Analyze the audio mix by splitting into chunks and processing in parallel.
        
        Args:
            audio (np.ndarray): Audio data
            sr (int): Sample rate
            
        Returns:
            Dict: Analysis results
        """
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
        
        # Split into chunks for analysis
        chunks = self._split_into_chunks(audio_tensor)
        logger.info(f"Split audio into {len(chunks)} chunks for analysis")
        
        if not chunks:
            logger.warning("No valid chunks to analyze")
            return {
                'spectral': torch.zeros((1, 3)),
                'temporal': torch.zeros((1, 2)),
                'rhythmic': torch.zeros((1, 2)),
                'global_features': {
                    'loudness': 0.0,
                    'dynamic_range': 0.0,
                    'crest_factor': 0.0
                }
            }
            
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            spectral_future = executor.submit(self._analyze_spectral, chunks)
            temporal_future = executor.submit(self._analyze_temporal, chunks)
            rhythmic_future = executor.submit(self._analyze_rhythmic, chunks)
            
            # Free memory while waiting for results
            del audio_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Get results
            spectral = spectral_future.result()
            temporal = temporal_future.result()
            rhythmic = rhythmic_future.result()
            
        # Calculate global features
        global_features = self._compute_global_features(chunks)
            
        # Free memory after processing
        del chunks
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'spectral': spectral,
            'temporal': temporal,
            'rhythmic': rhythmic,
            'global_features': global_features
        }

    def _split_into_chunks(self, audio: torch.Tensor) -> List[torch.Tensor]:
        """Split audio into overlapping chunks.
        
        Args:
            audio (torch.Tensor): Audio tensor
            
        Returns:
            List[torch.Tensor]: List of audio chunks
        """
        chunks = []
        for i in range(0, len(audio) - self.chunk_size + 1, self.hop_length):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                chunks.append(chunk)
        
        # Handle remainder if needed and audio is long enough
        if chunks and len(audio) > self.chunk_size and len(audio) % self.hop_length != 0:
            last_chunk = audio[-self.chunk_size:]
            if len(last_chunk) == self.chunk_size and not torch.allclose(last_chunk, chunks[-1]):
                chunks.append(last_chunk)
                
        return chunks

    def _analyze_spectral(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Extract spectral features from audio chunks.
        
        Args:
            chunks (List[torch.Tensor]): List of audio chunks
            
        Returns:
            torch.Tensor: Spectral features for each chunk
        """
        features = []
        for i, chunk in enumerate(chunks):
            # Log progress for long analyses
            if i % 10 == 0 and len(chunks) > 20:
                logger.debug(f"Spectral analysis: {i}/{len(chunks)} chunks")
                
            # Process chunk
            stft = torch.stft(chunk, n_fft=2048, hop_length=512, return_complex=True)
            mag = torch.abs(stft)
            features.append(self._compute_spectral_features(mag))
            
            # Free memory for large files
            del stft, mag
            if i % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
                
        return torch.stack(features)

    def _analyze_temporal(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Extract temporal features from audio chunks.
        
        Args:
            chunks (List[torch.Tensor]): List of audio chunks
            
        Returns:
            torch.Tensor: Temporal features for each chunk
        """
        features = []
        for chunk in chunks:
            rms = torch.sqrt(torch.mean(chunk ** 2))
            zcr = torch.mean(((chunk[:-1] * chunk[1:]) < 0).float())
            features.append(torch.stack([rms, zcr]))
        return torch.stack(features)

    def _analyze_rhythmic(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """Extract rhythmic features from audio chunks.
        
        Args:
            chunks (List[torch.Tensor]): List of audio chunks
            
        Returns:
            torch.Tensor: Rhythmic features for each chunk
        """
        features = []
        for chunk in chunks:
            onset_env = self._compute_onset_envelope(chunk)
            tempo_features = self._extract_tempo_features(onset_env)
            features.append(tempo_features)
            
            # Free memory
            del onset_env
        return torch.stack(features)

    def _compute_spectral_features(self, mag: torch.Tensor) -> torch.Tensor:
        """Compute spectral features from magnitude spectrum.
        
        Args:
            mag (torch.Tensor): Magnitude spectrum
            
        Returns:
            torch.Tensor: Spectral features (centroid, bandwidth, flatness)
        """
        # Handle empty or zero magnitudes
        if mag.numel() == 0 or torch.sum(mag) == 0:
            return torch.zeros(3, device=self.device)
            
        # Compute spectral centroid
        freq_range = torch.arange(mag.shape[1], device=self.device)
        centroid = torch.sum(mag * freq_range) / torch.sum(mag)
        
        # Compute spectral bandwidth
        squared_diff = (freq_range - centroid) ** 2
        bandwidth = torch.sqrt(torch.sum(squared_diff * mag) / torch.sum(mag))
        
        # Compute spectral flatness
        flatness = torch.exp(torch.mean(torch.log(mag + 1e-6))) / torch.mean(mag)
        
        return torch.stack([centroid, bandwidth, flatness])

    def _compute_onset_envelope(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute onset envelope for rhythm analysis.
        
        Args:
            audio (torch.Tensor): Audio chunk
            
        Returns:
            torch.Tensor: Onset envelope
        """
        # Compute STFT
        stft = torch.stft(audio, n_fft=2048, hop_length=512, return_complex=True)
        mag = torch.abs(stft)
        
        # Compute first-order difference across time
        diff = torch.diff(mag, dim=1)
        
        # Only keep increases in energy (half-wave rectification)
        diff = torch.maximum(diff, torch.zeros_like(diff))
        
        # Sum across frequency bins
        onset_env = torch.sum(diff ** 2, dim=0)
        
        return onset_env

    def _extract_tempo_features(self, onset_env: torch.Tensor) -> torch.Tensor:
        """Extract tempo-related features from onset envelope.
        
        Args:
            onset_env (torch.Tensor): Onset envelope
            
        Returns:
            torch.Tensor: Tempo features (tempo, strength)
        """
        # Compute autocorrelation
        if len(onset_env) > 1:
            # Move to CPU for numpy operation since torch.correlate isn't available
            onset_env_np = onset_env.cpu().numpy()
            
            # Compute autocorrelation using numpy
            acf_np = np.correlate(onset_env_np, onset_env_np, mode='full')
            acf_np = acf_np[len(acf_np) // 2:]  # Only use positive lags
            
            # Convert back to torch tensor on the same device
            acf = torch.from_numpy(acf_np).to(self.device)
            
            # Find peaks in autocorrelation
            if len(acf) > 2:
                # Compute derivative and find zero-crossings from + to -
                diff = torch.diff(acf)
                peaks = torch.where(torch.diff(torch.sign(diff)) < 0)[0] + 1
                
                if len(peaks) > 0:
                    # Convert first peak to BPM
                    first_peak_idx = peaks[0]
                    # Skip peaks that are too early (less than 40 BPM)
                    sr = 44100  # Sample rate
                    hop_length = 512  # STFT hop length
                    min_peak_idx = int((60 / 250) * (sr / hop_length))  # 250 BPM max
                    
                    valid_peaks = peaks[peaks >= min_peak_idx]
                    if len(valid_peaks) > 0:
                        first_valid_peak = valid_peaks[0]
                        tempo = 60 * (sr / hop_length) / first_valid_peak
                        tempo_strength = acf[first_valid_peak]
                        return torch.tensor([tempo, tempo_strength], device=self.device)
        
        # Default values if no peaks found
        return torch.tensor([0.0, 0.0], device=self.device)

    def _compute_global_features(self, chunks: List[torch.Tensor]) -> Dict:
        """Compute global features across all chunks.
        
        Args:
            chunks (List[torch.Tensor]): List of audio chunks
            
        Returns:
            Dict: Global features
        """
        # Initialize accumulators
        total_loudness = 0.0
        max_peak = -float('inf')
        min_peak = float('inf')
        total_rms = 0.0
        
        # Process each chunk
        for chunk in chunks:
            # Loudness (mean absolute amplitude)
            chunk_loudness = torch.mean(torch.abs(chunk)).item()
            total_loudness += chunk_loudness
            
            # Peak values
            chunk_max = torch.max(chunk).item()
            chunk_min = torch.min(chunk).item()
            max_peak = max(max_peak, chunk_max)
            min_peak = min(min_peak, chunk_min)
            
            # RMS
            chunk_rms = torch.sqrt(torch.mean(chunk ** 2)).item()
            total_rms += chunk_rms
            
        # Avoid division by zero
        n_chunks = max(1, len(chunks))
        
        # Calculate averages
        avg_loudness = total_loudness / n_chunks
        avg_rms = total_rms / n_chunks
        
        # Calculate global features
        dynamic_range = max_peak - min_peak
        crest_factor = max(abs(max_peak), abs(min_peak)) / (avg_rms + 1e-8)
        
        return {
            'loudness': avg_loudness,
            'dynamic_range': dynamic_range,
            'crest_factor': crest_factor,
            'peak_amplitude': max(abs(max_peak), abs(min_peak)),
            'rms_level': avg_rms
        }
