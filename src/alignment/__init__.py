"""
Alignment modules for audio pattern detection and matching.

This package contains components for aligning audio patterns, detecting 
similarities, and analyzing specific audio characteristics such as the Amen break.
"""

from .alignment import SequenceAligner, AmenBreakTemplate
from .similarity import SimilarityAnalyzer
from .composite_similarity import CompositeSimilarity
from .prior_subspace_analysis import PriorSubspaceAnalysis

__all__ = [
    'SequenceAligner',
    'AmenBreakTemplate',
    'SimilarityAnalyzer',
    'CompositeSimilarity',
    'PriorSubspaceAnalysis',
]
