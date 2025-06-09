"""
Semantic Analysis Module for Heihachi

This module provides semantic analysis capabilities including emotional feature mapping,
vector-based search, and LLM-powered analysis of audio content.
"""

from .semantic_analyzer import SemanticAnalyzer
from .feature_mapping import EmotionalFeatureMapper
from .embedding_generator import EmbeddingGenerator
from .vector_store import VectorStore
from .main import SemanticSearch

# Import query_processor and chat_interface if they exist
try:
    from .query_processor import QueryProcessor
    from .chat_interface import ChatInterface
except ImportError:
    QueryProcessor = None
    ChatInterface = None

__all__ = [
    'SemanticAnalyzer',
    'EmotionalFeatureMapper', 
    'EmbeddingGenerator',
    'VectorStore',
    'SemanticSearch',
    'QueryProcessor',
    'ChatInterface'
]
