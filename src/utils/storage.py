import pickle
import json
import logging
import zlib
import numpy as np
import torch
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from datetime import datetime
import os

logger = logging.getLogger("mix_analyzer")

class AnalysisVersion:
    MAJOR = 1
    MINOR = 0
    PATCH = 0

    @classmethod
    def to_string(cls) -> str:
        return f"{cls.MAJOR}.{cls.MINOR}.{cls.PATCH}"

class AudioCache:
    def __init__(self, cache_dir: str = "../cache/", compression_level: int = 6):
        """Initialize the audio cache.
        
        Args:
            cache_dir: Path to the cache directory
            compression_level: Compression level for storing cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
        self.compression_level = compression_level
        self._stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0
        }
        
        logger.info(f"AudioCache initialized with cache directory: {self.cache_dir}")

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with performance tracking."""
        # Check memory cache first
        if key in self._memory_cache:
            self._stats['hits'] += 1
            self._stats['memory_hits'] += 1
            return self._memory_cache[key]

        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl.gz"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    compressed_data = f.read()
                    decompressed_data = zlib.decompress(compressed_data)
                    value = pickle.loads(decompressed_data)
                self._memory_cache[key] = value
                self._stats['hits'] += 1
                self._stats['disk_hits'] += 1
                return value
            except Exception as e:
                logger.warning(f"Failed to load cache for key {key}: {e}")
                self._stats['misses'] += 1
                return None
        self._stats['misses'] += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Store item with compression."""
        self._memory_cache[key] = value
        cache_file = self.cache_dir / f"{key}.pkl.gz"
        try:
            serialized_data = pickle.dumps(value)
            compressed_data = zlib.compress(serialized_data, self.compression_level)
            with open(cache_file, 'wb') as f:
                f.write(compressed_data)
        except Exception as e:
            logger.warning(f"Failed to save cache for key {key}: {e}")

    def clear(self) -> None:
        """Clear cache and reset statistics."""
        self._memory_cache.clear()
        self._stats = {k: 0 for k in self._stats}
        for cache_file in self.cache_dir.glob("*.pkl.gz"):
            cache_file.unlink()

    def get_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            **self._stats,
            'hit_rate': self._stats['hits'] / (self._stats['hits'] + self._stats['misses']) 
            if (self._stats['hits'] + self._stats['misses']) > 0 else 0
        }

class FeatureStorage:
    def __init__(self, storage_dir: str = "../results/"):
        """Initialize the feature storage.
        
        Args:
            storage_dir: Path to the storage directory for results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.json"
        self._load_metadata()
        
        logger.info(f"FeatureStorage initialized with storage directory: {self.storage_dir}")

    def save_results(self, analysis_id: str, results: Dict[str, Any], 
                    metadata: Optional[Dict] = None) -> None:
        """Save analysis results with versioning and metadata."""
        result_file = self.storage_dir / f"{analysis_id}.json"
        try:
            # Add version and timestamp
            versioned_results = {
                'version': AnalysisVersion.to_string(),
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata or {},
                'results': self._make_serializable(results)
            }
            
            with open(result_file, 'w') as f:
                json.dump(versioned_results, f, indent=2)
            
            # Update metadata
            self._update_metadata(analysis_id, versioned_results)
            
        except Exception as e:
            logger.error(f"Failed to save results for {analysis_id}: {e}")

    def load_results(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Load analysis results with version checking."""
        result_file = self.storage_dir / f"{analysis_id}.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                # Version compatibility check
                if 'version' in data:
                    stored_version = data['version'].split('.')
                    current_version = AnalysisVersion.to_string().split('.')
                    
                    if stored_version[0] != current_version[0]:
                        logger.warning(
                            f"Major version mismatch for {analysis_id}. "
                            f"Stored: {data['version']}, Current: {AnalysisVersion.to_string()}"
                        )
                
                return data
            except Exception as e:
                logger.error(f"Failed to load results for {analysis_id}: {e}")
                return None
        return None

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON serializable format with enhanced type support."""
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (datetime, Path)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        return obj

    def _load_metadata(self) -> None:
        """Load storage metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)
            except Exception:
                self._metadata = {'analyses': {}}
        else:
            self._metadata = {'analyses': {}}

    def _update_metadata(self, analysis_id: str, results: Dict) -> None:
        """Update metadata with new analysis information."""
        self._metadata['analyses'][analysis_id] = {
            'timestamp': results['timestamp'],
            'version': results['version'],
            'metadata': results['metadata']
        }
        
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update metadata: {e}")

    def get_analysis_history(self) -> Dict[str, List[Dict]]:
        """Get historical analysis information."""
        return self._metadata['analyses']




