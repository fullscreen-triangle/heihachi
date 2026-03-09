"""
Categorical Entropy S_cat - Computes entropy from categorical state label counting.

S_cat = k_B * M * ln(n)

Categorical states are labels for phase space regions. The categorical observable
O_cat acts on the categorical Hilbert space H_cat and commutes with all physical
observables: [O_cat, O_phys] = 0.

This is one of three equivalent entropy formulations proven identical by the
Triple Equivalence Theorem.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple


class CategoricalEntropy:
    """Computes categorical entropy from state label counting.

    The categorical approach labels phase space regions and counts
    distinguishable labels. Unlike oscillatory entropy (which tracks phase),
    categorical entropy operates on the abstract state space directly.

    Phi_cat_to_part: j_k -> E_{j_k} = j_k * E_max / n
    """

    K_B = 1.380649e-23

    def __init__(self, partition_depth: int = 32, sample_rate: int = 44100):
        self.partition_depth = partition_depth
        self.sample_rate = sample_rate

    def assign_categorical_labels(
        self, signal: np.ndarray, n_bins: Optional[int] = None
    ) -> np.ndarray:
        """Assign categorical state labels to signal samples.

        Maps amplitude values to discrete bins in [0, n-1].
        """
        n = n_bins or self.partition_depth
        sig_min = np.min(signal)
        sig_max = np.max(signal)
        sig_range = sig_max - sig_min
        if sig_range < 1e-15:
            return np.zeros(len(signal), dtype=int)
        normalized = (signal - sig_min) / sig_range  # [0, 1]
        labels = np.clip(np.floor(normalized * n).astype(int), 0, n - 1)
        return labels

    def count_categorical_states(self, signal: np.ndarray) -> int:
        """Count number of distinct categorical labels occupied."""
        labels = self.assign_categorical_labels(signal)
        return len(np.unique(labels))

    def compute_entropy(self, signal: np.ndarray, n_modes: int = 1) -> float:
        """Compute categorical entropy S_cat = k_B * M * ln(n).

        n = number of distinct categorical states occupied.
        """
        n = self.count_categorical_states(signal)
        if n < 1:
            n = 1
        return self.K_B * n_modes * np.log(n)

    def label_distribution(self, signal: np.ndarray) -> np.ndarray:
        """Compute probability distribution over categorical labels."""
        labels = self.assign_categorical_labels(signal)
        n = self.partition_depth
        counts = np.zeros(n)
        for l in labels:
            counts[l] += 1
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts

    def shannon_entropy(self, signal: np.ndarray) -> float:
        """Shannon entropy of categorical label distribution.

        H = -sum p_j * ln(p_j)
        """
        dist = self.label_distribution(signal)
        entropy = 0.0
        for p in dist:
            if p > 0:
                entropy -= p * np.log(p)
        return float(entropy)

    def mutual_information(
        self, signal_a: np.ndarray, signal_b: np.ndarray
    ) -> float:
        """Mutual information between categorical labels of two signals.

        I(A;B) = H(A) + H(B) - H(A,B)
        """
        labels_a = self.assign_categorical_labels(signal_a)
        labels_b = self.assign_categorical_labels(signal_b)

        # Ensure same length
        min_len = min(len(labels_a), len(labels_b))
        labels_a = labels_a[:min_len]
        labels_b = labels_b[:min_len]

        n = self.partition_depth

        # Joint distribution
        joint = np.zeros((n, n))
        for la, lb in zip(labels_a, labels_b):
            joint[la, lb] += 1
        joint /= joint.sum() if joint.sum() > 0 else 1

        # Marginals
        pa = joint.sum(axis=1)
        pb = joint.sum(axis=0)

        # MI = sum p(a,b) * ln(p(a,b) / (p(a)*p(b)))
        mi = 0.0
        for i in range(n):
            for j in range(n):
                if joint[i, j] > 0 and pa[i] > 0 and pb[j] > 0:
                    mi += joint[i, j] * np.log(joint[i, j] / (pa[i] * pb[j]))
        return float(mi)

    def categorical_channel_capacity(self, n_modes: int) -> float:
        """Categorical channel capacity: C_cat = M * log2(n) bits."""
        return n_modes * np.log2(self.partition_depth)

    def commutation_test(
        self, signal: np.ndarray, physical_observable: np.ndarray
    ) -> float:
        """Test [O_cat, O_phys] = 0 numerically.

        Measures the correlation between categorical labels and a physical
        observable. Should be near zero for orthogonal channels.
        Returns normalized correlation coefficient.
        """
        labels = self.assign_categorical_labels(signal).astype(float)
        min_len = min(len(labels), len(physical_observable))
        labels = labels[:min_len]
        phys = physical_observable[:min_len]

        if np.std(labels) < 1e-15 or np.std(phys) < 1e-15:
            return 0.0

        correlation = np.abs(np.corrcoef(labels, phys)[0, 1])
        return float(correlation)
