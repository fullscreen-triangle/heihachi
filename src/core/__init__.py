"""
Core audio processing and analysis modules.

This package contains the main processing pipeline and audio analysis components.
"""

from .audio_processing import AudioProcessor
from .pipeline import Pipeline
from .mix_analyzer import MixAnalyzer
from .audio_scene import AudioSceneAnalyzer

__all__ = [
    'AudioProcessor',
    'Pipeline',
    'MixAnalyzer',
    'AudioSceneAnalyzer',
]
"""
Core module for Heihachi.

This module contains the core components and pipeline for the Heihachi audio analysis framework.
"""

# Import core components to make them available at package level
from .pipeline import Pipeline

__all__ = ['Pipeline']
"""
Core module for Heihachi.

This module contains the core components and pipeline for the Heihachi audio analysis framework.
"""

# Import core components to make them available at package level
from .pipeline import Pipeline

__all__ = ['Pipeline']