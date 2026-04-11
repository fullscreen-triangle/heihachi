"""
Oscillatory Entropy S_osc - Computes entropy from oscillatory phase state counting.

S_osc = k_B * M * ln(n)

where M is the number of independent oscillatory modes and n is the partition depth
(number of distinguishable phase states per mode).

This is one of three equivalent entropy formulations proven identical by the
Triple Equivalence Theorem.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class OscillatoryEntropy:
    """Computes oscillatory entropy from phase space trajectory.

    For an audio signal decomposed into M modes, each mode traces a trajectory
    in its 2D phase space (q_k, p_k). The oscillatory entropy counts the number
    of distinguishable phase states visited.

    Phi_osc_to_cat: phi_k(t) -> j_k = floor(n * phi_k(t) / (2*pi)) mod n
    """

    K_B = 1.380649e-23

    def __init__(self, partition_depth: int = 32, sample_rate: int = 44100):
        self.partition_depth = partition_depth
        self.sample_rate = sample_rate

    def extract_instantaneous_phase(self, signal: np.ndarray) -> np.ndarray:
        """Extract instantaneous phase via analytic signal (Hilbert transform)."""
        N = len(signal)
        spec = np.fft.fft(signal)
        spec[0] = 0
        n_half = N // 2
        spec[1:n_half] *= 2
        spec[n_half + 1:] = 0
        analytic = np.fft.ifft(spec)
        return np.unwrap(np.angle(analytic))

    def phase_to_categorical_state(self, phase: np.ndarray) -> np.ndarray:
        """Map continuous phase to discrete categorical state index.

        j_k = floor(n * phi_k / (2*pi)) mod n
        """
        n = self.partition_depth
        return (np.floor(n * phase / (2 * np.pi)) % n).astype(int)

    def count_phase_states(self, signal: np.ndarray) -> int:
        """Count number of distinct phase states visited by the signal."""
        phase = self.extract_instantaneous_phase(signal)
        states = self.phase_to_categorical_state(phase)
        return len(np.unique(states))

    def compute_entropy(self, signal: np.ndarray, n_modes: int = 1) -> float:
        """Compute oscillatory entropy S_osc = k_B * M * ln(n).

        For a single-mode signal, M=1.
        For a multi-mode signal, pass n_modes = number of decomposed modes.
        """
        n = self.count_phase_states(signal)
        if n < 1:
            n = 1
        return self.K_B * n_modes * np.log(n)

    def compute_multimode_entropy(
        self, mode_signals: List[np.ndarray]
    ) -> float:
        """Compute oscillatory entropy for M independent mode signals.

        S_osc = k_B * sum_k ln(n_k) where n_k is states visited by mode k.
        """
        total = 0.0
        for mode_signal in mode_signals:
            n_k = self.count_phase_states(mode_signal)
            if n_k > 0:
                total += np.log(n_k)
        return self.K_B * total

    def phase_state_trajectory(self, signal: np.ndarray) -> np.ndarray:
        """Return the full categorical state trajectory j(t)."""
        phase = self.extract_instantaneous_phase(signal)
        return self.phase_to_categorical_state(phase)

    def state_transition_matrix(self, signal: np.ndarray) -> np.ndarray:
        """Compute n x n transition matrix between phase states.

        T[i,j] = P(state j at t+1 | state i at t)
        """
        states = self.phase_state_trajectory(signal)
        n = self.partition_depth
        T = np.zeros((n, n))
        for i in range(len(states) - 1):
            T[states[i], states[i + 1]] += 1

        # Normalize rows
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T /= row_sums
        return T

    def entropy_rate(self, signal: np.ndarray) -> float:
        """Compute entropy rate from transition matrix.

        h = -sum_i pi_i sum_j T[i,j] * ln(T[i,j])

        where pi is the stationary distribution.
        """
        T = self.state_transition_matrix(signal)
        states = self.phase_state_trajectory(signal)
        n = self.partition_depth

        # Empirical stationary distribution
        pi = np.zeros(n)
        for s in states:
            pi[s] += 1
        pi /= pi.sum() if pi.sum() > 0 else 1

        h = 0.0
        for i in range(n):
            for j in range(n):
                if T[i, j] > 0 and pi[i] > 0:
                    h -= pi[i] * T[i, j] * np.log(T[i, j])
        return float(h)
