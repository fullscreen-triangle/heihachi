"""
Categorical State Ensemble - Manages the ensemble of categorical states
across all oscillatory modes and computes aggregate thermodynamic quantities.

The ensemble represents the full categorical state of an audio signal:
Omega = n^M total states for M modes at partition depth n.
Entropy: S = k_B * M * ln(n)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from .virtual_spectrometer import (
    CategoricalState,
    CategoricalMeasurement,
    SEntropyCoordinate,
    PartitionCoordinate,
)


@dataclass
class EnsembleStatistics:
    """Aggregate statistics for a categorical state ensemble."""
    n_modes: int
    partition_depth: int
    total_states: float
    entropy: float
    mean_s_entropy: SEntropyCoordinate
    std_s_entropy: SEntropyCoordinate
    entropy_rate: float
    irreversibility_fraction: float
    categorical_capacity: float


class CategoricalEnsemble:
    """Manages categorical state ensemble and computes thermodynamic quantities.

    Given a CategoricalMeasurement (from VirtualSpectrometer), this class
    computes:
      - Total accessible states Omega = n^M
      - Categorical entropy S = k_B * M * ln(n)
      - S-entropy statistics across modes and time
      - Entropy production rate (categorical second law: delta_S_cat > 0)
      - State occupation probabilities
      - Partition function Z = sum exp(-E_j / k_B T)
    """

    K_B = 1.380649e-23

    def __init__(self, measurement: CategoricalMeasurement):
        self.measurement = measurement
        self.n_modes = measurement.n_modes
        self.partition_depth = measurement.partition_depth
        self._build_state_index()

    def _build_state_index(self) -> None:
        """Index states by (mode_index, time) for fast lookup."""
        self._mode_trajectories: Dict[int, List[CategoricalState]] = {}
        for state in self.measurement.trajectory:
            if state.mode_index not in self._mode_trajectories:
                self._mode_trajectories[state.mode_index] = []
            self._mode_trajectories[state.mode_index].append(state)

        for mode_idx in self._mode_trajectories:
            self._mode_trajectories[mode_idx].sort(key=lambda s: s.time)

    @property
    def total_states(self) -> float:
        """Total accessible categorical states: Omega = n^M."""
        return float(self.partition_depth ** self.n_modes)

    @property
    def entropy(self) -> float:
        """Categorical entropy: S = k_B * M * ln(n)."""
        return self.K_B * self.n_modes * np.log(self.partition_depth)

    @property
    def categorical_capacity(self) -> float:
        """Categorical channel capacity: C_cat = M * log2(n) bits."""
        return self.n_modes * np.log2(self.partition_depth)

    def compute_statistics(self) -> EnsembleStatistics:
        """Compute aggregate ensemble statistics."""
        trajectory = self.measurement.trajectory
        transitions = self.measurement.transitions

        if not trajectory:
            zero_s = SEntropyCoordinate(0, 0, 0)
            return EnsembleStatistics(
                n_modes=self.n_modes,
                partition_depth=self.partition_depth,
                total_states=self.total_states,
                entropy=self.entropy,
                mean_s_entropy=zero_s,
                std_s_entropy=zero_s,
                entropy_rate=0,
                irreversibility_fraction=0,
                categorical_capacity=self.categorical_capacity,
            )

        sk_vals = [s.s_entropy.S_k for s in trajectory]
        st_vals = [s.s_entropy.S_t for s in trajectory]
        se_vals = [s.s_entropy.S_e for s in trajectory]

        mean_s = SEntropyCoordinate(
            S_k=float(np.mean(sk_vals)),
            S_t=float(np.mean(st_vals)),
            S_e=float(np.mean(se_vals)),
        )
        std_s = SEntropyCoordinate(
            S_k=float(np.std(sk_vals)),
            S_t=float(np.std(st_vals)),
            S_e=float(np.std(se_vals)),
        )

        duration = self.measurement.duration
        entropy_rate = (
            self.measurement.entropy_production / duration if duration > 0 else 0
        )

        if transitions:
            n_irreversible = sum(1 for t in transitions if t.is_irreversible)
            irrev_frac = n_irreversible / len(transitions)
        else:
            irrev_frac = 0

        return EnsembleStatistics(
            n_modes=self.n_modes,
            partition_depth=self.partition_depth,
            total_states=self.total_states,
            entropy=self.entropy,
            mean_s_entropy=mean_s,
            std_s_entropy=std_s,
            entropy_rate=entropy_rate,
            irreversibility_fraction=irrev_frac,
            categorical_capacity=self.categorical_capacity,
        )

    def state_occupation_histogram(self) -> Dict[int, int]:
        """Count how many times each flat partition index is occupied."""
        histogram: Dict[int, int] = {}
        for state in self.measurement.trajectory:
            idx = state.partition.flat_index
            histogram[idx] = histogram.get(idx, 0) + 1
        return histogram

    def occupation_entropy(self) -> float:
        """Shannon entropy of state occupation: H = -sum p_j * ln(p_j)."""
        histogram = self.state_occupation_histogram()
        total = sum(histogram.values())
        if total == 0:
            return 0.0
        entropy = 0.0
        for count in histogram.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log(p)
        return entropy

    def partition_function(self, temperature: float = 1.0) -> float:
        """Compute partition function Z = sum exp(-E_j / k_B T)."""
        if not self.measurement.trajectory:
            return 1.0
        energies = []
        for state in self.measurement.trajectory:
            E = 0.5 * state.amplitude ** 2 * (2 * np.pi * state.frequency) ** 2
            energies.append(E)
        energies = np.array(energies)
        beta = 1.0 / (self.K_B * temperature)
        E_min = np.min(energies)
        Z = np.sum(np.exp(-beta * (energies - E_min)))
        return float(Z)

    def mode_trajectory(self, mode_index: int) -> List[CategoricalState]:
        """Get the categorical trajectory for a specific mode."""
        return self._mode_trajectories.get(mode_index, [])

    def s_entropy_trajectory(self) -> np.ndarray:
        """Get S-entropy coordinates as (N, 3) array [S_k, S_t, S_e]."""
        if not self.measurement.trajectory:
            return np.empty((0, 3))
        return np.array([
            s.s_entropy.as_array() for s in self.measurement.trajectory
        ])

    def time_series(self) -> np.ndarray:
        """Get timestamps for all trajectory states."""
        return np.array([s.time for s in self.measurement.trajectory])

    def categorical_second_law_verification(self) -> Dict[str, float]:
        """Verify the categorical second law: delta_S_cat > 0."""
        transitions = self.measurement.transitions
        if not transitions:
            return {
                'n_transitions': 0, 'n_positive': 0,
                'n_negative': 0, 'n_zero': 0,
                'mean_delta_s': 0, 'fraction_positive': 0,
                'total_production': 0,
            }
        deltas = [t.delta_s_entropy for t in transitions]
        n_pos = sum(1 for d in deltas if d > 0)
        n_neg = sum(1 for d in deltas if d < 0)
        n_zero = sum(1 for d in deltas if d == 0)
        return {
            'n_transitions': len(transitions),
            'n_positive': n_pos,
            'n_negative': n_neg,
            'n_zero': n_zero,
            'mean_delta_s': float(np.mean(deltas)),
            'fraction_positive': n_pos / len(transitions),
            'total_production': float(np.sum([d for d in deltas if d > 0])),
        }
