"""
Sinc Interpolation - Standard Nyquist-Shannon bandlimited reconstruction.

Implements the Whittaker-Shannon interpolation formula:
  x(t) = sum_n x[n] * sinc((t - n*T_s) / T_s)

This serves as the baseline against which categorical trajectory
reconstruction is compared.
"""

import numpy as np
from typing import Optional, Tuple


class SincInterpolator:
    """Standard bandlimited sinc interpolation (Nyquist reconstruction).

    Reconstructs the continuous waveform from discrete samples using
    the sinc function. This is the theoretically optimal reconstruction
    for bandlimited signals, but it:
    1. Cannot resolve inter-sample dynamics
    2. Is subject to the Gabor uncertainty limit
    3. Produces artifacts for non-bandlimited signals (Gibbs ringing)
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.T_s = 1.0 / sample_rate

    def interpolate(
        self,
        samples: np.ndarray,
        target_times: np.ndarray,
        window_size: int = 64,
    ) -> np.ndarray:
        """Reconstruct signal at arbitrary time points using windowed sinc.

        Args:
            samples: discrete signal samples x[n]
            target_times: continuous time points to evaluate (seconds)
            window_size: number of samples on each side for truncated sinc

        Returns:
            interpolated signal values at target_times
        """
        N = len(samples)
        result = np.zeros(len(target_times))

        for i, t in enumerate(target_times):
            # Find nearest sample index
            center = t / self.T_s
            n_start = max(0, int(np.floor(center)) - window_size)
            n_end = min(N, int(np.ceil(center)) + window_size + 1)

            val = 0.0
            for n in range(n_start, n_end):
                sinc_arg = (t - n * self.T_s) / self.T_s
                # Windowed sinc (Hann window)
                if abs(sinc_arg) < window_size:
                    sinc_val = np.sinc(sinc_arg)
                    hann = 0.5 * (1 + np.cos(np.pi * sinc_arg / window_size))
                    val += samples[n] * sinc_val * hann
            result[i] = val

        return result

    def upsample(
        self,
        samples: np.ndarray,
        factor: int = 4,
        window_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Upsample signal by integer factor using sinc interpolation.

        Returns (upsampled_signal, time_points).
        """
        N = len(samples)
        n_out = N * factor
        t_out = np.arange(n_out) * self.T_s / factor
        upsampled = self.interpolate(samples, t_out, window_size)
        return upsampled, t_out

    def reconstruction_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> dict:
        """Compute reconstruction error metrics.

        Returns dict with MSE, SNR, max error, and normalized error.
        """
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]

        error = orig - recon
        mse = float(np.mean(error ** 2))
        signal_power = float(np.mean(orig ** 2))

        if signal_power > 0:
            snr = 10 * np.log10(signal_power / max(mse, 1e-30))
        else:
            snr = 0.0

        return {
            'mse': mse,
            'snr_db': snr,
            'max_error': float(np.max(np.abs(error))),
            'normalized_error': mse / max(signal_power, 1e-30),
        }

    def gabor_limited_resolution(self, frequency: float) -> dict:
        """Compute the Gabor-limited time-frequency resolution.

        delta_t * delta_f >= 1/(4*pi)
        At the sample rate, delta_t_min = T_s.
        """
        delta_t = self.T_s
        delta_f_min = 1.0 / (4 * np.pi * delta_t)

        return {
            'delta_t': delta_t,
            'delta_f_min': delta_f_min,
            'gabor_product': delta_t * delta_f_min,
            'gabor_limit': 1.0 / (4 * np.pi),
            'frequency_resolution_at_nyquist': self.sample_rate / 2.0,
        }
