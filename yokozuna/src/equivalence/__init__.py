"""
Equivalence module - Triple equivalence theorem verification.

Implements the three equivalent entropy formulations:
  S_osc = S_cat = S_part = k_B * M * ln(n)

And the equivalence maps:
  Phi_osc_to_cat: phase -> categorical state index
  Phi_cat_to_part: state index -> energy level
  Phi_part_to_osc: energy -> amplitude (reconstruction)
"""

from .oscillatory import OscillatoryEntropy
from .categorical import CategoricalEntropy
from .partition import PartitionEntropy

__all__ = [
    'OscillatoryEntropy',
    'CategoricalEntropy',
    'PartitionEntropy',
]
