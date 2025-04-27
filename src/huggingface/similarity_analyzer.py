"""
Multimodal similarity module using CLAP models.

This module provides functionality for audio-text similarity analysis 
using Contrastive Language-Audio Pretraining (CLAP) models.
"""

import torch
import numpy as np
import librosa
import logging
from typing import Dict, List, Union, Optional, Tuple
import os
import json
from pathlib import Path
import open_clip
import transformers

logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    """Multimodal similarity analysis using CLAP models."""
    
    def __init__(self, model_name: str = "laion/clap-htsat-fused", 
                 use_cuda: bool = True,
                 device: Optional[str] = None,
                 api_key: Optional[str] = None,
                 config: Optional[Dict] = None):
        """Initialize the similarity analyzer with a CLAP model.
        
        Args:
            model_name: Model name/identifier (default: laion/clap-htsat-fused)
            use_cuda: Whether to use CUDA if available
            device: Specific device to use (e.g., 'cuda:0', 'cpu')
            api_key: HuggingFace API key for accessing gated models
            config: Additional configuration settings
        """
        self.model_name = model_name
        self.config = config or {}
        self.api_key = api_key
        
        # Set up device
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = device or ('cuda' if self.use_cuda else 'cpu')
        
        # Set the API key environment variable if provided
        if api_key:
            os.environ["HF_API_TOKEN"] = api_key
            logger.info("HuggingFace API token set from configuration")
        
        # Load the model and processor
        self._load_model()
    
    def _load_model(self):
        """Load the CLAP model and processor."""
        logger.info(f"Loading CLAP model: {self.model_name}")
        try:
            # For LAION CLAP model
            if "laion/clap" in self.model_name:
                # Use transformers implementation
                self.processor = transformers.AutoProcessor.from_pretrained(self.model_name)
                self.model = transformers.ClapModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = 512  # CLAP embedding dimension
                
                logger.info(f"CLAP model loaded successfully on {self.device}")
            else:
                # For other similarity models
                logger.warning(f"Model {self.model_name} not specifically supported as similarity model. Attempting generic loading.")
                self.processor = transformers.AutoProcessor.from_pretrained(self.model_name)
                self.model = transformers.AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                self.embedding_dim = self.model.config.hidden_size
                
        except Exception as e:
            logger.error(f"Failed to load similarity model: {str(e)}")
            raise ValueError(f"Failed to load similarity model: {str(e)}")
    
    def embed_audio(self, audio_path: Optional[str] = None, 
                   audio_array: Optional[np.ndarray] = None, 
                   sample_rate: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for audio input.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            
        Returns:
            Embedding vector as numpy array
        """
        # Load audio if path is provided
        if audio_path is not None:
            try:
                audio_array, sample_rate = librosa.load(audio_path, sr=None)
                logger.info(f"Loaded audio: {audio_path}, {len(audio_array)/sample_rate:.2f}s @ {sample_rate}Hz")
            except Exception as e:
                logger.error(f"Failed to load audio from {audio_path}: {str(e)}")
                raise ValueError(f"Audio loading error: {str(e)}")
        elif audio_array is None or sample_rate is None:
            raise ValueError("Either audio_path or (audio_array and sample_rate) must be provided")
        
        # Convert to mono if needed
        if len(audio_array.shape) > 1 and audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)
            logger.info("Converted stereo audio to mono")
        
        # Process audio through the model
        try:
            inputs = self.processor(
                audios=audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_audio_features(**inputs)
                
            # Extract audio embeddings
            if hasattr(outputs, "audio_embeds"):
                embeddings = outputs.audio_embeds.cpu().numpy()
            else:
                embeddings = outputs.cpu().numpy()
                
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings.squeeze()
            
        except Exception as e:
            logger.error(f"Error generating audio embeddings: {str(e)}")
            raise ValueError(f"Audio embedding generation failed: {str(e)}")
    
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text input.
        
        Args:
            text: Text prompt or list of text prompts
            
        Returns:
            Embedding vector(s) as numpy array
        """
        try:
            # Format input
            if isinstance(text, str):
                text = [text]
                
            # Process text through the model
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.get_text_features(**inputs)
                
            # Extract text embeddings
            if hasattr(outputs, "text_embeds"):
                embeddings = outputs.text_embeds.cpu().numpy()
            else:
                embeddings = outputs.cpu().numpy()
                
            # Normalize embeddings
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            # Return single embedding if only one text was provided
            if len(text) == 1:
                return embeddings.squeeze()
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            raise ValueError(f"Text embedding generation failed: {str(e)}")
    
    def compute_similarity(self, audio_embedding: np.ndarray, 
                          text_embeddings: np.ndarray) -> np.ndarray:
        """Compute similarity between audio and text embeddings.
        
        Args:
            audio_embedding: Audio embedding vector
            text_embeddings: Text embedding vector(s)
            
        Returns:
            Similarity scores as numpy array
        """
        # Ensure embeddings are normalized
        audio_embedding = audio_embedding / np.linalg.norm(audio_embedding)
        
        # If text_embeddings is a 1D array, add batch dimension
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.reshape(1, -1)
            
        # Normalize text embeddings
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity = np.matmul(text_embeddings, audio_embedding)
        
        return similarity
    
    def match_text_to_audio(self, audio_path: Optional[str] = None, 
                           audio_array: Optional[np.ndarray] = None, 
                           sample_rate: Optional[int] = None,
                           text_queries: List[str]) -> Dict:
        """Match text queries to audio and rank by similarity.
        
        Args:
            audio_path: Path to audio file (alternative to audio_array)
            audio_array: NumPy array of audio data (alternative to audio_path)
            sample_rate: Sample rate of audio_array (required if audio_array is provided)
            text_queries: List of text queries to match against the audio
            
        Returns:
            Dictionary with ranked matches and scores
        """
        # Get audio embedding
        audio_embedding = self.embed_audio(audio_path, audio_array, sample_rate)
        
        # Get text embeddings for all queries
        text_embeddings = self.embed_text(text_queries)
        
        # Compute similarity scores
        similarity_scores = self.compute_similarity(audio_embedding, text_embeddings)
        
        # Rank results
        ranked_indices = np.argsort(similarity_scores)[::-1]
        
        # Prepare results
        results = {
            "matches": [
                {
                    "query": text_queries[idx],
                    "score": float(similarity_scores[idx])
                }
                for idx in ranked_indices
            ],
            "audio_embedding": audio_embedding,
            "model": self.model_name
        }
        
        return results
    
    def search_with_audio(self, query_audio_path: str, 
                         reference_audios: List[str],
                         top_k: int = 5) -> Dict:
        """Search for similar audio files using a query audio.
        
        Args:
            query_audio_path: Path to query audio file
            reference_audios: List of paths to reference audio files
            top_k: Number of top matches to return
            
        Returns:
            Dictionary with ranked matches and scores
        """
        # Get query audio embedding
        query_embedding = self.embed_audio(audio_path=query_audio_path)
        
        # Get embeddings for all reference audios
        reference_embeddings = []
        for audio_path in reference_audios:
            try:
                embedding = self.embed_audio(audio_path=audio_path)
                reference_embeddings.append({
                    "path": audio_path,
                    "embedding": embedding
                })
            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {str(e)}")
        
        # Compute similarity scores
        results = []
        for ref in reference_embeddings:
            similarity = np.dot(query_embedding, ref["embedding"])
            results.append({
                "path": ref["path"],
                "score": float(similarity)
            })
        
        # Rank results
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k results
        return {
            "query_audio": query_audio_path,
            "matches": results[:top_k],
            "model": self.model_name
        }
    
    def find_timestamps_for_text(self, audio_path: str, text_query: str, 
                               segment_length: float = 3.0,
                               overlap: float = 1.5,
                               threshold: float = 0.3) -> Dict:
        """Find timestamps in audio that best match the text query.
        
        Args:
            audio_path: Path to audio file
            text_query: Text query to match
            segment_length: Length of audio segments in seconds
            overlap: Overlap between segments in seconds
            threshold: Minimum similarity score threshold
            
        Returns:
            Dictionary with matching segments and scores
        """
        # Get text embedding
        text_embedding = self.embed_text(text_query)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)
        
        # Calculate segment parameters
        samples_per_segment = int(segment_length * sr)
        hop_length = int((segment_length - overlap) * sr)
        
        # Split audio into overlapping segments
        segments = []
        timestamps = []
        
        for start in range(0, len(audio) - samples_per_segment + 1, hop_length):
            segment = audio[start:start + samples_per_segment]
            segments.append(segment)
            timestamps.append(start / sr)
        
        # Process each segment
        matches = []
        
        for i, (segment, timestamp) in enumerate(zip(segments, timestamps)):
            try:
                # Get embedding for segment
                segment_embedding = self.embed_audio(audio_array=segment, sample_rate=sr)
                
                # Compute similarity with text query
                similarity = np.dot(segment_embedding, text_embedding)
                
                # Add to matches if above threshold
                if similarity > threshold:
                    matches.append({
                        "timestamp": timestamp,
                        "duration": segment_length,
                        "end_time": timestamp + segment_length,
                        "score": float(similarity)
                    })
            except Exception as e:
                logger.warning(f"Failed to process segment at {timestamp}s: {str(e)}")
        
        # Sort matches by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "query": text_query,
            "matches": matches,
            "audio_file": audio_path,
            "model": self.model_name
        } 