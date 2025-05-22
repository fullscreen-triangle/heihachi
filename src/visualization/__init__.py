"""
Visualization modules for audio analysis.

This package contains various visualization tools for displaying
analysis results and audio features.
"""

from .plots import plot_waveform, plot_spectrogram

# Create the visualizer object that's imported in pipeline.py
class Visualizer:
    """
    Visualization helper class for audio analysis.
    """
    @staticmethod
    def plot_waveform(*args, **kwargs):
        return plot_waveform(*args, **kwargs)
        
    @staticmethod
    def plot_spectrogram(*args, **kwargs):
        return plot_spectrogram(*args, **kwargs)

visualizer = Visualizer() 