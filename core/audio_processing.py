from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from scipy import signal
import soundfile as sf


class AudioProcessor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = 1024 * 1024  # 1MB chunks

    def load(self, file_path):
        """Load audio file from path.
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            np.ndarray: Loaded audio data
        """
        return self.load_audio(file_path)

    def load_audio(self, file_path):
        return self._load_in_chunks(file_path)

    def _load_in_chunks(self, file_path):
        with sf.SoundFile(file_path) as f:
            chunks = []
            while f.tell() < f.frames:
                chunk = f.read(self.chunk_size)
                chunks.append(chunk)
        return np.concatenate(chunks)

    @torch.no_grad()
    def process_audio(self, audio):
        # Convert to tensor and move to GPU if available
        audio_tensor = torch.from_numpy(audio).to(self.device)

        # Apply preprocessing
        normalized = self._normalize(audio_tensor)
        filtered = self._apply_filters(normalized)

        return filtered.cpu().numpy()

    def _normalize(self, audio):
        return audio / torch.max(torch.abs(audio))

    def _apply_filters(self, audio):
        # Apply various filters in parallel on GPU
        filtered = audio.clone()

        # High-pass filter
        filtered = self._highpass_filter(filtered, cutoff=20)

        # De-noise
        filtered = self._denoise(filtered)

        return filtered

    def _highpass_filter(self, audio, cutoff):
        nyquist = 22050
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        return torch.from_numpy(signal.filtfilt(b, a, audio.cpu().numpy())).to(self.device)

    def _denoise(self, audio):
        # Spectral subtraction-based denoising
        stft = torch.stft(audio, n_fft=2048, hop_length=512)
        mag = torch.abs(stft)
        phase = torch.angle(stft)

        # Estimate noise floor
        noise_floor = torch.mean(mag[:100], dim=0, keepdim=True)

        # Subtract noise
        mag = torch.maximum(mag - noise_floor, torch.zeros_like(mag))

        # Reconstruct
        stft_denoised = mag * torch.exp(1j * phase)
        return torch.istft(stft_denoised, n_fft=2048, hop_length=512)




class BatchProcessor:
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.processor = AudioProcessor()

    def process_batch(self, file_paths):
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._process_file, path) for path in file_paths]
            results = [f.result() for f in futures]
        return results

    def _process_file(self, file_path):
        audio = self.processor.load_audio(file_path)
        return self.processor.process_audio(audio)
