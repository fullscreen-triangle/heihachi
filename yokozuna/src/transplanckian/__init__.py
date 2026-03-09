"""
Transplanckian module - Hardware clock, virtual spectrometer, and
categorical state measurement for trans-Nyquist audio resolution.
"""

from .virtual_spectrometer import (
    HardwareClock,
    VirtualSpectrometer,
    PartitionCoordinate,
    SEntropyCoordinate,
    CategoricalState,
    CategoricalTransition,
    CategoricalMeasurement,
)
from .harmonics import HarmonicCoincidenceNetwork, HarmonicRelation, CoincidenceCluster
from .ensemble import CategoricalEnsemble, EnsembleStatistics

__all__ = [
    'HardwareClock',
    'VirtualSpectrometer',
    'PartitionCoordinate',
    'SEntropyCoordinate',
    'CategoricalState',
    'CategoricalTransition',
    'CategoricalMeasurement',
    'HarmonicCoincidenceNetwork',
    'HarmonicRelation',
    'CoincidenceCluster',
    'CategoricalEnsemble',
    'EnsembleStatistics',
]
