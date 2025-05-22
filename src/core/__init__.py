"""
Core audio processing and analysis modules.

This package contains the main processing pipeline and audio analysis components.
"""

# Define module exports without directly importing them
# This avoids circular imports
__all__ = [
    'AudioProcessor',
    'MixAnalyzer',
    'AudioSceneAnalyzer',
    'Pipeline',
]

# Note: Import these classes directly from their modules, e.g.:
# from src.core.audio_processing import AudioProcessor
# from src.core.mix_analyzer import MixAnalyzer
# from src.core.audio_scene import AudioSceneAnalyzer
# from src.core.pipeline import Pipeline