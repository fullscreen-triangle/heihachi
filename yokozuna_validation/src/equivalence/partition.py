"""
Partition Function Entropy S_part - Computes entropy from the statistical
mechanics partition function.

S_part = k_B * M * ln(n)

The partition function Z = sum_j exp(-E_j / k_B T) connects the microscopic
categorical states to macroscopic thermodynamic quantities.

This is one of three equivalent entropy formulations proven identical by the
Triple Equivalence Theorem.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class PartitionEntropy:
    """Computes entropy from the partition function formalism.

    Maps categorical state labels to energy levels and computes
    thermodynamic quantities via the partition function.

    Phi_part_to_osc: E_{j_k} -> A_k = sqrt(2 * E_{j_k} / (mu_k * omega_k^2))

    The partition coordinate (n, l, m, s) with degeneracy g_n = 2n^2
    determines the density of states.
    """

    K_B = 1.380649e-23

    def __init__(self, partition_depth: int = 32, sample_rate: int = 44100):
        self.partition_depth = partition_depth
        self.sample_rate = sample_rate

    def degeneracy(self, n: int) -> int:
        """Compute degeneracy at partition depth n: g_n = 2n^2."""
        return 2 * n * n

    def energy_levels(self, max_amplitude: float, frequency: float) -> np.ndarray:
        """Compute energy levels for partition states.

        E_j = j * E_max / n for j in [0, n-1]
        where E_max = 0.5 * A_max^2 * omega^2
        """
        omega = 2 * np.pi * frequency
        E_max = 0.5 * max_amplitude ** 2 * omega ** 2
        n = self.partition_depth
        return np.array([j * E_max / n for j in range(n)])

    def partition_function(
        self,
        energies: np.ndarray,
        temperature: float = 1.0,
    ) -> float:
        """Compute partition function Z = sum_j g_j * exp(-E_j / k_B T).

        Includes degeneracy g_n = 2n^2 for each level.
        """
        beta = 1.0 / (self.K_B * temperature)
        E_min = np.min(energies)
        n = self.partition_depth
        Z = 0.0
        for j, E in enumerate(energies):
            g = self.degeneracy(min(j + 1, n))
            Z += g * np.exp(-beta * (E - E_min))
        return Z

    def compute_entropy(
        self,
        signal: np.ndarray,
        frequency: float = 440.0,
        temperature: float = 1.0,
        n_modes: int = 1,
    ) -> float:
        """Compute partition entropy S_part = k_B * M * ln(n).

        Also computable as S = k_B * (ln Z + beta * <E>).
        """
        max_amp = np.max(np.abs(signal))
        if max_amp < 1e-15:
            return 0.0

        energies = self.energy_levels(max_amp, frequency)
        Z = self.partition_function(energies, temperature)

        if Z <= 0:
            return 0.0

        # S = k_B * ln(Z) + <E>/T
        beta = 1.0 / (self.K_B * temperature)
        E_min = np.min(energies)

        mean_E = 0.0
        for j, E in enumerate(energies):
            g = self.degeneracy(min(j + 1, self.partition_depth))
            p_j = g * np.exp(-beta * (E - E_min)) / Z
            mean_E += p_j * E

        S = self.K_B * np.log(Z) + mean_E / temperature
        return float(S * n_modes)

    def free_energy(
        self,
        energies: np.ndarray,
        temperature: float = 1.0,
    ) -> float:
        """Helmholtz free energy F = -k_B T ln Z."""
        Z = self.partition_function(energies, temperature)
        if Z <= 0:
            return 0.0
        return -self.K_B * temperature * np.log(Z)

    def mean_energy(
        self,
        energies: np.ndarray,
        temperature: float = 1.0,
    ) -> float:
        """Mean energy <E> = sum_j p_j * E_j."""
        Z = self.partition_function(energies, temperature)
        if Z <= 0:
            return 0.0

        beta = 1.0 / (self.K_B * temperature)
        E_min = np.min(energies)

        mean_E = 0.0
        for j, E in enumerate(energies):
            g = self.degeneracy(min(j + 1, self.partition_depth))
            p_j = g * np.exp(-beta * (E - E_min)) / Z
            mean_E += p_j * E
        return float(mean_E)

    def occupation_probabilities(
        self,
        energies: np.ndarray,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """Boltzmann occupation probabilities p_j = g_j * exp(-beta E_j) / Z."""
        Z = self.partition_function(energies, temperature)
        if Z <= 0:
            return np.zeros(len(energies))

        beta = 1.0 / (self.K_B * temperature)
        E_min = np.min(energies)
        probs = np.zeros(len(energies))
        for j, E in enumerate(energies):
            g = self.degeneracy(min(j + 1, self.partition_depth))
            probs[j] = g * np.exp(-beta * (E - E_min)) / Z
        return probs

    def energy_to_amplitude(
        self, energy: float, frequency: float, effective_mass: float = 1.0
    ) -> float:
        """Reconstruction map: E -> A = sqrt(2E / (mu * omega^2)).

        This is Phi_part_to_osc from the triple equivalence.
        """
        omega = 2 * np.pi * frequency
        denominator = effective_mass * omega ** 2
        if denominator < 1e-30:
            return 0.0
        return np.sqrt(2 * abs(energy) / denominator)
