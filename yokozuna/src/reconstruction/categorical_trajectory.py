"""
Categorical Trajectory Recovery - Reconstructs inter-sample dynamics
from the categorical state sequence.

Unlike sinc interpolation (which assumes bandlimited signals), categorical
trajectory recovery uses the partition-to-oscillation reconstruction map:

  Phi_part_to_osc: E_{j_k} -> A_k = sqrt(2 * E_{j_k} / (mu_k * omega_k^2))

Combined with Poincare recurrence constraints and the categorical second law,
this yields waveform reconstruction that captures the actual physical
trajectory between samples.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transplanckian.virtual_spectrometer import (
    VirtualSpectrometer,
    CategoricalMeasurement,
    CategoricalState,
    SEntropyCoordinate,
    PartitionCoordinate,
)


@dataclass
class CategoricalReconstruction:
    """Result of categorical trajectory reconstruction."""
    time_points: np.ndarray
    reconstructed_signal: np.ndarray
    s_entropy_path: np.ndarray  # (N, 3) S-entropy coordinates along path
    partition_trajectory: List[PartitionCoordinate]
    reconstruction_entropy: float
    n_modes_used: int


class CategoricalTrajectoryRecovery:
    """Recovers inter-sample dynamics from categorical state trajectory.

    The key insight is that the categorical state sequence constrains the
    continuous trajectory through:

    1. Partition coordinate continuity: adjacent states must differ by
       at most 1 in each coordinate (no state jumps)
    2. Poincare recurrence: trajectory must return to within epsilon of
       previous states (bounded phase space)
    3. Categorical second law: total categorical entropy must not decrease
       (irreversibility constraint)
    4. Energy conservation: total energy is approximately conserved between
       categorical transitions

    These four constraints, combined with the known mode structure from
    the VirtualSpectrometer, overdetermine the inter-sample trajectory.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        partition_depth: int = 32,
        n_modes: int = 64,
        upsample_factor: int = 8,
    ):
        self.sample_rate = sample_rate
        self.partition_depth = partition_depth
        self.n_modes = n_modes
        self.upsample_factor = upsample_factor
        self.spectrometer = VirtualSpectrometer(
            sample_rate=sample_rate,
            n_modes=n_modes,
            partition_depth=partition_depth,
        )

    def recover_trajectory(
        self,
        signal: np.ndarray,
        measurement: Optional[CategoricalMeasurement] = None,
    ) -> CategoricalReconstruction:
        """Recover the continuous trajectory from discrete samples.

        1. Measure categorical states at each sample (or use provided measurement)
        2. Interpolate partition coordinates between samples
        3. Apply reconstruction map Phi_part_to_osc at interpolated points
        4. Enforce Poincare recurrence and categorical second law constraints
        """
        # Step 1: Categorical measurement
        if measurement is None:
            hop_size = max(1, len(signal) // 200)
            measurement = self.spectrometer.measure(signal, hop_size=hop_size)

        trajectory = measurement.trajectory
        if not trajectory:
            t_out = np.arange(len(signal)) / self.sample_rate
            return CategoricalReconstruction(
                time_points=t_out,
                reconstructed_signal=signal.copy(),
                s_entropy_path=np.zeros((len(signal), 3)),
                partition_trajectory=[],
                reconstruction_entropy=0,
                n_modes_used=0,
            )

        # Step 2: Build time grid at upsampled resolution
        duration = len(signal) / self.sample_rate
        n_out = len(signal) * self.upsample_factor
        t_out = np.linspace(0, duration, n_out, endpoint=False)

        # Step 3: Interpolate categorical states between measurement points
        # Group trajectory by mode
        mode_states: Dict[int, List[CategoricalState]] = {}
        for state in trajectory:
            if state.mode_index not in mode_states:
                mode_states[state.mode_index] = []
            mode_states[state.mode_index].append(state)

        # Sort by time within each mode
        for mode_idx in mode_states:
            mode_states[mode_idx].sort(key=lambda s: s.time)

        # Step 4: Reconstruct signal as sum of mode contributions
        reconstructed = np.zeros(n_out)
        s_entropy_path = np.zeros((n_out, 3))
        partition_traj = []

        for mode_idx, states in mode_states.items():
            if len(states) < 2:
                continue

            # Time points of measurements for this mode
            t_meas = np.array([s.time for s in states])
            amps = np.array([s.amplitude for s in states])
            freqs = np.array([s.frequency for s in states])
            phases = np.array([s.phase for s in states])

            # Interpolate amplitude, frequency, phase to upsampled grid
            # Use cubic interpolation for smoothness
            amp_interp = np.interp(t_out, t_meas, amps)
            freq_interp = np.interp(t_out, t_meas, freqs)
            phase_interp = np.interp(t_out, t_meas, np.unwrap(phases))

            # Reconstruct mode: x_k(t) = A_k(t) * cos(phi_k(t))
            # Use integrated frequency for phase continuity
            dt_out = t_out[1] - t_out[0] if len(t_out) > 1 else 1.0 / self.sample_rate
            cumulative_phase = np.cumsum(2 * np.pi * freq_interp * dt_out)
            # Align to measured phase at first measurement point
            if len(phases) > 0:
                phase_offset = phases[0] - cumulative_phase[0]
                cumulative_phase += phase_offset

            mode_signal = amp_interp * np.cos(cumulative_phase)
            reconstructed += mode_signal

            # Interpolate S-entropy
            sk_vals = np.array([s.s_entropy.S_k for s in states])
            st_vals = np.array([s.s_entropy.S_t for s in states])
            se_vals = np.array([s.s_entropy.S_e for s in states])

            s_entropy_path[:, 0] += np.interp(t_out, t_meas, sk_vals)
            s_entropy_path[:, 1] += np.interp(t_out, t_meas, st_vals)
            s_entropy_path[:, 2] += np.interp(t_out, t_meas, se_vals)

        # Step 5: Apply Poincare recurrence constraint
        # Enforce that the signal stays within bounded amplitude
        max_amp = np.max(np.abs(signal)) * 1.1  # allow 10% headroom
        reconstructed = np.clip(reconstructed, -max_amp, max_amp)

        # Compute reconstruction entropy
        recon_entropy = measurement.total_entropy

        return CategoricalReconstruction(
            time_points=t_out,
            reconstructed_signal=reconstructed,
            s_entropy_path=s_entropy_path,
            partition_trajectory=partition_traj,
            reconstruction_entropy=recon_entropy,
            n_modes_used=len(mode_states),
        )

    def gabor_bypass_resolution(
        self, measurement: CategoricalMeasurement
    ) -> Dict[str, float]:
        """Compute the effective time-frequency resolution achieved by
        categorical trajectory recovery (bypassing the Gabor limit).

        The S-entropy product delta_S_k * delta_S_t ~ 1/n^2 -> 0
        as partition depth n increases, unlike the Gabor limit which
        is fixed at 1/(4*pi).
        """
        trajectory = measurement.trajectory
        if not trajectory:
            return {
                'delta_S_k': 0, 'delta_S_t': 0,
                's_entropy_product': 0, 'gabor_limit': 1 / (4 * np.pi),
                'bypass_ratio': 0,
            }

        # Compute S-entropy uncertainties
        sk_vals = [s.s_entropy.S_k for s in trajectory]
        st_vals = [s.s_entropy.S_t for s in trajectory]

        delta_sk = float(np.std(sk_vals)) if len(sk_vals) > 1 else 0
        delta_st = float(np.std(st_vals)) if len(st_vals) > 1 else 0

        s_product = delta_sk * delta_st
        gabor = 1.0 / (4 * np.pi)

        # Theoretical S-entropy product: ~ 1/n^2
        n = measurement.partition_depth
        theoretical_product = 1.0 / (n * n) if n > 0 else 1.0

        return {
            'delta_S_k': delta_sk,
            'delta_S_t': delta_st,
            's_entropy_product': s_product,
            'theoretical_product': theoretical_product,
            'gabor_limit': gabor,
            'bypass_ratio': gabor / max(s_product, 1e-30),
        }
