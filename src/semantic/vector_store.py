import numpy as np
import os
import json
import pickle
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime
import shutil
import threading

class VectorStore:
    """
    Simple vector store for track embeddings with search capabilities.
    Stores track data and embeddings for efficient semantic search.
    """
    
    def __init__(self, storage_dir: str = "data/vector_store"):
        """
        Initialize the vector store.
        
        Args:
            storage_dir: Directory to store vector data
        """
        self.storage_dir = storage_dir
        self.index_file = os.path.join(storage_dir, "index.json")
        self.embeddings_file = os.path.join(storage_dir, "embeddings.npy")
        self.metadata_dir = os.path.join(storage_dir, "metadata")
        
        # In-memory data
        self.track_ids = []
        self.embeddings = None
        self.index = {}
        self.lock = threading.RLock()  # For thread safety
        
        # Create directories if they don't exist
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Load existing data if available
        self._load_store()
    
    def add_track(self, track_id: str, track_info: Dict[str, Any], 
                  embedding: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a track to the vector store.
        
        Args:
            track_id: Unique identifier for the track
            track_info: Basic track information (title, artist, etc.)
            embedding: Track embedding vector
            metadata: Additional metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Check if track already exists
                if track_id in self.index:
                    return self.update_track(track_id, track_info, embedding, metadata)
                
                # Add to in-memory index
                self.track_ids.append(track_id)
                self.index[track_id] = {
                    "info": track_info,
                    "added": datetime.now().isoformat(),
                    "updated": datetime.now().isoformat(),
                    "embedding_index": len(self.track_ids) - 1
                }
                
                # Add embedding to array
                if self.embeddings is None:
                    self.embeddings = embedding.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, embedding])
                
                # Save metadata if provided
                if metadata:
                    self._save_metadata(track_id, metadata)
                
                # Save to disk
                self._save_store()
                return True
                
            except Exception as e:
                print(f"Error adding track to vector store: {str(e)}")
                return False
    
    def update_track(self, track_id: str, track_info: Optional[Dict[str, Any]] = None,
                    embedding: Optional[np.ndarray] = None, 
                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing track in the vector store.
        
        Args:
            track_id: Unique identifier for the track
            track_info: Updated track information (optional)
            embedding: Updated embedding vector (optional)
            metadata: Updated metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                if track_id not in self.index:
                    return False
                
                # Update track info if provided
                if track_info:
                    self.index[track_id]["info"] = track_info
                
                # Update embedding if provided
                if embedding is not None:
                    embedding_index = self.index[track_id]["embedding_index"]
                    self.embeddings[embedding_index] = embedding
                
                # Update metadata if provided
                if metadata:
                    self._save_metadata(track_id, metadata)
                
                # Update timestamp
                self.index[track_id]["updated"] = datetime.now().isoformat()
                
                # Save to disk
                self._save_store()
                return True
                
            except Exception as e:
                print(f"Error updating track in vector store: {str(e)}")
                return False
    
    def remove_track(self, track_id: str) -> bool:
        """
        Remove a track from the vector store.
        
        Args:
            track_id: Unique identifier for the track
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                if track_id not in self.index:
                    return False
                
                # Get embedding index
                embedding_index = self.index[track_id]["embedding_index"]
                
                # Remove from track_ids and update index
                self.track_ids.remove(track_id)
                del self.index[track_id]
                
                # Remove embedding and update indices
                self.embeddings = np.delete(self.embeddings, embedding_index, axis=0)
                
                # Update embedding indices for all tracks
                for tid, data in self.index.items():
                    if data["embedding_index"] > embedding_index:
                        self.index[tid]["embedding_index"] -= 1
                
                # Remove metadata file
                metadata_file = os.path.join(self.metadata_dir, f"{track_id}.json")
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)
                
                # Save to disk
                self._save_store()
                return True
                
            except Exception as e:
                print(f"Error removing track from vector store: {str(e)}")
                return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for tracks similar to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of track data with similarity scores
        """
        with self.lock:
            try:
                if self.embeddings is None or len(self.track_ids) == 0:
                    return []
                
                # Calculate cosine similarity
                similarities = self._cosine_similarity(query_embedding, self.embeddings)
                
                # Get top k indices
                if threshold > 0:
                    # Filter by threshold first
                    indices = np.where(similarities >= threshold)[0]
                    # Sort by similarity
                    indices = indices[np.argsort(-similarities[indices])]
                    # Limit to top_k
                    indices = indices[:top_k]
                else:
                    # Get top k indices directly
                    indices = np.argsort(-similarities)[:top_k]
                
                # Build result list
                results = []
                for idx in indices:
                    track_id = self.track_ids[idx]
                    track_data = self.index[track_id]
                    
                    result = {
                        "track_id": track_id,
                        "info": track_data["info"],
                        "similarity": float(similarities[idx]),
                        "added": track_data["added"],
                        "updated": track_data["updated"]
                    }
                    
                    # Add metadata if available
                    metadata = self._load_metadata(track_id)
                    if metadata:
                        result["metadata"] = metadata
                    
                    results.append(result)
                
                return results
                
            except Exception as e:
                print(f"Error searching vector store: {str(e)}")
                return []
    
    def get_track(self, track_id: str) -> Optional[Dict[str, Any]]:
        """
        Get track data by ID.
        
        Args:
            track_id: Unique identifier for the track
            
        Returns:
            Track data or None if not found
        """
        with self.lock:
            if track_id not in self.index:
                return None
            
            track_data = self.index[track_id]
            result = {
                "track_id": track_id,
                "info": track_data["info"],
                "added": track_data["added"],
                "updated": track_data["updated"]
            }
            
            # Add metadata if available
            metadata = self._load_metadata(track_id)
            if metadata:
                result["metadata"] = metadata
            
            return result
    
    def get_all_tracks(self) -> List[Dict[str, Any]]:
        """
        Get all tracks in the store.
        
        Returns:
            List of all track data
        """
        with self.lock:
            results = []
            for track_id in self.track_ids:
                track_data = self.get_track(track_id)
                if track_data:
                    results.append(track_data)
            return results
    
    def count(self) -> int:
        """
        Get the number of tracks in the store.
        
        Returns:
            Number of tracks
        """
        with self.lock:
            return len(self.track_ids)
    
    def clear(self) -> bool:
        """
        Clear all data from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            try:
                # Clear in-memory data
                self.track_ids = []
                self.embeddings = None
                self.index = {}
                
                # Clear files
                if os.path.exists(self.index_file):
                    os.remove(self.index_file)
                
                if os.path.exists(self.embeddings_file):
                    os.remove(self.embeddings_file)
                
                # Clear metadata directory
                if os.path.exists(self.metadata_dir):
                    shutil.rmtree(self.metadata_dir)
                    os.makedirs(self.metadata_dir, exist_ok=True)
                
                return True
                
            except Exception as e:
                print(f"Error clearing vector store: {str(e)}")
                return False
    
    def _save_store(self) -> None:
        """Save vector store data to disk."""
        try:
            # Save index
            with open(self.index_file, 'w') as f:
                json.dump({
                    "track_ids": self.track_ids,
                    "index": self.index
                }, f)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
                
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")
    
    def _load_store(self) -> None:
        """Load vector store data from disk."""
        try:
            # Load index
            if os.path.exists(self.index_file):
                with open(self.index_file, 'r') as f:
                    data = json.load(f)
                    self.track_ids = data.get("track_ids", [])
                    self.index = data.get("index", {})
            
            # Load embeddings
            if os.path.exists(self.embeddings_file):
                self.embeddings = np.load(self.embeddings_file)
                
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            # Initialize empty store
            self.track_ids = []
            self.embeddings = None
            self.index = {}
    
    def _save_metadata(self, track_id: str, metadata: Dict[str, Any]) -> None:
        """Save track metadata to disk."""
        try:
            metadata_file = os.path.join(self.metadata_dir, f"{track_id}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            print(f"Error saving metadata for track {track_id}: {str(e)}")
    
    def _load_metadata(self, track_id: str) -> Optional[Dict[str, Any]]:
        """Load track metadata from disk."""
        try:
            metadata_file = os.path.join(self.metadata_dir, f"{track_id}.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            return None
                
        except Exception as e:
            print(f"Error loading metadata for track {track_id}: {str(e)}")
            return None
    
    def _cosine_similarity(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate cosine similarity between query and vectors.
        
        Args:
            query: Query vector
            vectors: Matrix of vectors to compare against
            
        Returns:
            Array of similarity scores
        """
        # Normalize query
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm
        
        # Normalize vectors (row-wise)
        vectors_norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors_norm[vectors_norm == 0] = 1  # Avoid division by zero
        normalized_vectors = vectors / vectors_norm
        
        # Calculate dot product (cosine similarity for normalized vectors)
        return np.dot(normalized_vectors, query)
