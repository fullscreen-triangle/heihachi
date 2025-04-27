"""
Hugging Face integration for the Heihachi audio analysis framework.

This module provides integration with Hugging Face's Transformers library
for advanced audio analysis, genre classification, and instrument detection.
"""

from .audio_analyzer import HuggingFaceAudioAnalyzer
from .feature_extractor import FeatureExtractor
from .stem_separator import StemSeparator
from .beat_detector import BeatDetector
from .drum_analyzer import DrumAnalyzer
from .wav2vec2_drum_analyzer import DrumSoundAnalyzer
from .similarity_analyzer import SimilarityAnalyzer
from .zero_shot_tagger import ZeroShotTagger
from .audio_captioner import AudioCaptioner
from .real_time_beat_tracker import RealTimeBeatTracker
