"""
Harmonic Coincidence Network — Detects and exploits harmonic relationships
between oscillatory modes for enhanced categorical resolution.

When multiple modes share integer frequency ratios (harmonics), their
categorical states become correlated, creating a coincidence network
that amplifies categorical information by ~10^3.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from itertools import combinations


@dataclass
class HarmonicRelation:
    """A detected harmonic relationship between two modes."""
    mode_i: int
    mode_j: int
    ratio: float
    nearest_integer: int
    deviation: float
    strength: float


@dataclass
class CoincidenceCluster:
    """A group of harmonically related modes forming a coincidence cluster."""
    fundamental_idx: int
    fundamental_freq: float
    member_indices: List[int]
    member_freqs: List[float]
    harmonic_numbers: List[int]
    cluster_entropy: float
    coincidence_enhancement: float


class HarmonicCoincidenceNetwork:
    """Detects harmonic relationships and computes coincidence enhancement.

    The harmonic coincidence mechanism exploits the fact that harmonically
    related modes traverse their categorical states in locked patterns.
    When mode j = k * mode i (integer harmonic), mode j completes k
    categorical cycles for every one cycle of mode i. This correlation
    constrains the joint state space, concentrating categorical information.

    Enhancement ~ product over clusters of (harmonic_order)^(cluster_size - 1)
    """

    def __init__(
        self,
        tolerance: float = 0.02,
        min_strength: float = 0.1,
        max_harmonic: int = 16,
    ):
        self.tolerance = tolerance
        self.min_strength = min_strength
        self.max_harmonic = max_harmonic

    def detect_harmonics(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
    ) -> List[HarmonicRelation]:
        """Detect all pairwise harmonic relationships among modes."""
        n = len(frequencies)
        relations = []

        for i, j in combinations(range(n), 2):
            fi, fj = frequencies[i], frequencies[j]
            ai, aj = amplitudes[i], amplitudes[j]

            if fi > fj:
                i, j = j, i
                fi, fj = fj, fi
                ai, aj = aj, ai

            if fi < 1e-10:
                continue

            ratio = fj / fi

            for k in range(1, self.max_harmonic + 1):
                deviation = abs(ratio - k) / k
                if deviation < self.tolerance:
                    max_amp = max(amplitudes)
                    if max_amp > 0:
                        strength = np.sqrt(ai * aj) / max_amp
                    else:
                        strength = 0

                    if strength >= self.min_strength:
                        relations.append(HarmonicRelation(
                            mode_i=i, mode_j=j,
                            ratio=ratio,
                            nearest_integer=k,
                            deviation=deviation,
                            strength=strength,
                        ))
                    break

        return relations

    def build_clusters(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        relations: Optional[List[HarmonicRelation]] = None,
    ) -> List[CoincidenceCluster]:
        """Group harmonically related modes into coincidence clusters."""
        if relations is None:
            relations = self.detect_harmonics(frequencies, amplitudes)

        n = len(frequencies)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for rel in relations:
            union(rel.mode_i, rel.mode_j)

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        clusters = []
        for root, members in groups.items():
            if len(members) < 2:
                continue

            members.sort(key=lambda i: frequencies[i])
            fund_idx = members[0]
            fund_freq = frequencies[fund_idx]

            harmonic_numbers = []
            for idx in members:
                if fund_freq > 0:
                    ratio = frequencies[idx] / fund_freq
                    harmonic_numbers.append(max(1, round(ratio)))
                else:
                    harmonic_numbers.append(1)

            k_B = 1.380649e-23
            cluster_entropy = k_B * sum(
                np.log(max(h, 1)) for h in harmonic_numbers
            )

            enhancement = float(np.prod([
                max(h, 1) for h in harmonic_numbers[1:]
            ]))

            clusters.append(CoincidenceCluster(
                fundamental_idx=fund_idx,
                fundamental_freq=fund_freq,
                member_indices=members,
                member_freqs=[frequencies[i] for i in members],
                harmonic_numbers=harmonic_numbers,
                cluster_entropy=cluster_entropy,
                coincidence_enhancement=enhancement,
            ))

        return clusters

    def total_coincidence_enhancement(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
    ) -> float:
        """Compute total harmonic coincidence enhancement factor."""
        clusters = self.build_clusters(frequencies, amplitudes)
        if not clusters:
            return 1.0
        total = 1.0
        for cluster in clusters:
            total *= cluster.coincidence_enhancement
        return total

    def coincidence_matrix(
        self,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        """Build M x M coincidence matrix. C[i,j] = harmonic strength."""
        n = len(frequencies)
        C = np.eye(n)
        relations = self.detect_harmonics(frequencies, amplitudes)
        for rel in relations:
            C[rel.mode_i, rel.mode_j] = rel.strength
            C[rel.mode_j, rel.mode_i] = rel.strength
        return C

    def phase_locking_value(
        self,
        phases_i: np.ndarray,
        phases_j: np.ndarray,
        harmonic_ratio: int,
    ) -> float:
        """Compute phase locking value: PLV = |<exp(j*(phi_j - k*phi_i))>|."""
        phase_diff = phases_j - harmonic_ratio * phases_i
        return float(np.abs(np.mean(np.exp(1j * phase_diff))))
