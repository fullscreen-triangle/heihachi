"""
Reconstruction Difference - Compares sinc interpolation vs categorical
trajectory recovery to quantify the information gained from the
categorical channel.

The key comparison:
  - Sinc: uses only physical channel (C_phys = B * log2(1 + SNR))
  - Categorical: uses both channels (C_total = C_phys + C_cat)

The difference reveals the orthogonal information content.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .sinc_interpolation import SincInterpolator
from .categorical_trajectory import CategoricalTrajectoryRecovery, CategoricalReconstruction

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transplanckian.virtual_spectrometer import VirtualSpectrometer, CategoricalMeasurement


@dataclass
class ReconstructionComparison:
    """Results of comparing sinc vs categorical reconstruction."""
    sinc_signal: np.ndarray
    categorical_signal: np.ndarray
    difference_signal: np.ndarray
    time_points: np.ndarray
    sinc_error: dict
    categorical_error: dict
    information_gain: dict
    gabor_bypass: dict


class ReconstructionDifference:
    """Compares sinc interpolation against categorical trajectory recovery.

    Generates a high-resolution ground truth (either from a known analytic
    signal or from a very high sample rate recording), then compares:
    1. Standard sinc reconstruction from downsampled signal
    2. Categorical trajectory recovery from same downsampled signal

    The difference quantifies the information content of the categorical channel.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        partition_depth: int = 32,
        n_modes: int = 64,
        upsample_factor: int = 8,
    ):
        self.sample_rate = sample_rate
        self.sinc = SincInterpolator(sample_rate)
        self.categorical = CategoricalTrajectoryRecovery(
            sample_rate=sample_rate,
            partition_depth=partition_depth,
            n_modes=n_modes,
            upsample_factor=upsample_factor,
        )
        self.spectrometer = VirtualSpectrometer(
            sample_rate=sample_rate,
            n_modes=n_modes,
            partition_depth=partition_depth,
        )
        self.upsample_factor = upsample_factor

    def compare(
        self,
        signal: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        ground_truth_times: Optional[np.ndarray] = None,
    ) -> ReconstructionComparison:
        """Compare sinc and categorical reconstruction.

        If ground_truth is not provided, uses the upsampled sinc reconstruction
        as an approximate ground truth for error computation.
        """
        N = len(signal)

        # Categorical reconstruction
        cat_recon = self.categorical.recover_trajectory(signal)
        t_out = cat_recon.time_points
        cat_signal = cat_recon.reconstructed_signal

        # Sinc reconstruction at same time points
        sinc_signal = self.sinc.interpolate(signal, t_out)

        # Difference signal (what categorical captures that sinc misses)
        diff_signal = cat_signal - sinc_signal

        # Error metrics
        if ground_truth is not None:
            # Resample ground truth to match t_out
            if ground_truth_times is not None:
                gt_interp = np.interp(t_out, ground_truth_times, ground_truth)
            else:
                gt_interp = np.interp(
                    t_out,
                    np.linspace(0, N / self.sample_rate, len(ground_truth)),
                    ground_truth,
                )
            sinc_err = self.sinc.reconstruction_error(gt_interp, sinc_signal)
            cat_err = self.sinc.reconstruction_error(gt_interp, cat_signal)
        else:
            sinc_err = {'mse': 0, 'snr_db': 0, 'max_error': 0, 'normalized_error': 0}
            cat_err = {'mse': 0, 'snr_db': 0, 'max_error': 0, 'normalized_error': 0}

        # Information gain from categorical channel
        measurement = self.spectrometer.measure(signal)
        c_cat = self.spectrometer.categorical_capacity()
        c_phys = self.spectrometer.physical_capacity()

        info_gain = {
            'C_phys_bits_per_s': c_phys,
            'C_cat_bits': c_cat,
            'C_total': c_phys + c_cat,
            'categorical_fraction': c_cat / (c_phys + c_cat) if (c_phys + c_cat) > 0 else 0,
            'difference_energy': float(np.mean(diff_signal ** 2)),
            'difference_max': float(np.max(np.abs(diff_signal))),
        }

        # Gabor bypass metrics
        gabor = self.categorical.gabor_bypass_resolution(measurement)

        return ReconstructionComparison(
            sinc_signal=sinc_signal,
            categorical_signal=cat_signal,
            difference_signal=diff_signal,
            time_points=t_out,
            sinc_error=sinc_err,
            categorical_error=cat_err,
            information_gain=info_gain,
            gabor_bypass=gabor,
        )

    def generate_test_signal(
        self,
        duration: float = 1.0,
        frequencies: Optional[list] = None,
        include_transient: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a test signal with known analytic form for validation.

        Returns (signal_at_sample_rate, high_res_ground_truth).
        """
        if frequencies is None:
            frequencies = [440.0, 880.0, 1320.0, 2200.0]

        # High-resolution ground truth
        hr_rate = self.sample_rate * self.upsample_factor
        t_hr = np.arange(int(duration * hr_rate)) / hr_rate

        signal_hr = np.zeros_like(t_hr)
        for i, f in enumerate(frequencies):
            amp = 1.0 / (i + 1)
            signal_hr += amp * np.sin(2 * np.pi * f * t_hr + np.random.uniform(0, 2 * np.pi))

        if include_transient:
            # Add a sharp transient (click) at t=0.3s
            click_idx = int(0.3 * hr_rate)
            click_width = int(0.001 * hr_rate)
            if click_idx + click_width < len(signal_hr):
                signal_hr[click_idx:click_idx + click_width] += 2.0 * np.hanning(click_width)

        # Downsample to standard rate
        signal_lr = signal_hr[::self.upsample_factor]

        return signal_lr, signal_hr
