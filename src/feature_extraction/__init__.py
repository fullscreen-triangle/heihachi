"""
Feature extraction modules for audio analysis.

This package contains various feature extraction methods for extracting
meaningful features from audio signals.
"""

from .spectral_analysis import SpectralFeatureExtractor

# Create the necessary function that's imported in pipeline.py
_extractor = SpectralFeatureExtractor()

def feature_extractor(audio):
    """
    Extract spectral features from audio data.
    
    This is a convenience function that wraps the SpectralFeatureExtractor class.
    
    Args:
        audio: Audio data as numpy array
        
    Returns:
        Dict of extracted features
    """
    return _extractor.analyze(audio)
