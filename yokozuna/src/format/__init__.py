"""
Yokozuna Format (.ykz) — Container format for categorical audio.

A .ykz file is a zip-based container that stores:
  - The original audio signal (PCM samples)
  - The full categorical state trajectory
  - S-entropy and partition coordinate streams
  - Harmonic coincidence network
  - Analysis metadata

This provides both the physical channel (audio samples) and the
categorical channel (orthogonal information) in a single file,
enabling reconstruction that exceeds Nyquist-Shannon limits.
"""

from .encoder import YokozunaEncoder
from .decoder import YokozunaDecoder

__all__ = ['YokozunaEncoder', 'YokozunaDecoder']
