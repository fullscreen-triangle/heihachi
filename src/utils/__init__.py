"""
Utility modules for configuration, logging, visualization, and storage.

This package contains helper functions and classes used throughout the project.
"""

from .config import ConfigManager
from .logging_utils import get_logger, setup_logging
from .storage import Storage, StorageFormat
from .visualization import MixVisualizer, AnalysisVisualizer
from .visualization_optimization import VisualizationOptimizer

# Extract common visualization functions from VisualizationOptimizer
_viz_optimizer = VisualizationOptimizer()
create_figure = _viz_optimizer.create_figure
save_figure = _viz_optimizer.save_figure
plot_spectrogram = _viz_optimizer.plot_spectrogram

__all__ = [
    'ConfigManager',
    'get_logger',
    'setup_logging',
    'Storage',
    'StorageFormat',
    'MixVisualizer',
    'AnalysisVisualizer',
    'create_figure',
    'save_figure',
    'plot_spectrogram',
]
