#!/usr/bin/env python3
"""
Storage utilities for the Heihachi framework.

This module provides classes and functions for storing and retrieving data
in various formats, with a focus on audio analysis results.
"""

import os
import json
import yaml
import pickle
import numpy as np
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Union, Optional, List, Tuple, BinaryIO

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class StorageFormat(Enum):
    """Supported storage formats for data serialization."""
    JSON = "json"
    YAML = "yaml"
    PICKLE = "pickle"
    NUMPY = "npy"
    CSV = "csv"
    MARKDOWN = "md"
    HTML = "html"


class Storage:
    """Flexible storage utility for saving and loading data in different formats.
    
    Provides an interface for saving and loading data with format detection,
    automatic directory creation, and error handling.
    """
    
    @staticmethod
    def save(data: Any, file_path: Union[str, Path], 
            format: Optional[Union[StorageFormat, str]] = None,
            pretty: bool = True,
            compress: bool = False) -> Path:
        """Save data to file in the specified format.
        
        Args:
            data: Data to save
            file_path: Path to save the data to
            format: Format to use (if None, inferred from file extension)
            pretty: Whether to use pretty formatting for JSON and YAML
            compress: Whether to compress the data
            
        Returns:
            Path to the saved file
        
        Raises:
            ValueError: If format cannot be determined or is unsupported
        """
        file_path = Path(file_path)
        
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format if not specified
        if format is None:
            suffix = file_path.suffix.lower().lstrip('.')
            try:
                format = StorageFormat(suffix)
            except ValueError:
                raise ValueError(f"Could not determine format from file extension: {suffix}")
        elif isinstance(format, str):
            try:
                format = StorageFormat(format.lower())
            except ValueError:
                raise ValueError(f"Unsupported format: {format}")
        
        # Apply compression if requested
        if compress:
            import gzip
            open_func = gzip.open
            if not file_path.suffix.endswith('.gz'):
                file_path = file_path.with_suffix(file_path.suffix + '.gz')
        else:
            open_func = open
        
        # Save data in the specified format
        try:
            if format == StorageFormat.JSON:
                with open_func(file_path, 'wt') as f:
                    indent = 2 if pretty else None
                    json.dump(data, f, indent=indent, default=lambda o: str(o) if isinstance(o, np.ndarray) else o)
            
            elif format == StorageFormat.YAML:
                with open_func(file_path, 'wt') as f:
                    yaml.dump(data, f, default_flow_style=not pretty, sort_keys=False)
            
            elif format == StorageFormat.PICKLE:
                with open_func(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            elif format == StorageFormat.NUMPY:
                if isinstance(data, np.ndarray):
                    if compress:
                        np.savez_compressed(file_path, data=data)
                    else:
                        np.save(file_path, data)
                else:
                    with open_func(file_path, 'wb') as f:
                        np.savez(f, **data if isinstance(data, dict) else {'data': data})
            
            elif format == StorageFormat.CSV:
                if isinstance(data, np.ndarray):
                    np.savetxt(file_path, data, delimiter=',')
                elif isinstance(data, (list, tuple)) and all(isinstance(x, (list, tuple)) for x in data):
                    with open_func(file_path, 'wt') as f:
                        for row in data:
                            f.write(','.join(str(x) for x in row) + '\n')
                else:
                    import csv
                    with open_func(file_path, 'wt', newline='') as f:
                        if isinstance(data, dict):
                            writer = csv.DictWriter(f, fieldnames=data.keys())
                            writer.writeheader()
                            if isinstance(list(data.values())[0], list):
                                # Transpose data if values are lists
                                rows = []
                                for i in range(len(list(data.values())[0])):
                                    row = {k: v[i] if i < len(v) else None for k, v in data.items()}
                                    rows.append(row)
                                writer.writerows(rows)
                            else:
                                writer.writerow(data)
                        else:
                            writer = csv.writer(f)
                            writer.writerows(data)
            
            elif format == StorageFormat.MARKDOWN:
                with open_func(file_path, 'wt') as f:
                    if isinstance(data, dict):
                        f.write("# Analysis Results\n\n")
                        for section, content in data.items():
                            f.write(f"## {section}\n\n")
                            if isinstance(content, dict):
                                for key, value in content.items():
                                    f.write(f"- **{key}**: {value}\n")
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict):
                                        for k, v in item.items():
                                            f.write(f"- **{k}**: {v}\n")
                                    else:
                                        f.write(f"- {item}\n")
                            else:
                                f.write(f"{content}\n")
                            f.write("\n")
                    else:
                        f.write(str(data))
            
            elif format == StorageFormat.HTML:
                with open_func(file_path, 'wt') as f:
                    f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                    f.write("<title>Analysis Results</title>\n")
                    f.write("<style>\n")
                    f.write("body { font-family: Arial, sans-serif; margin: 20px; }\n")
                    f.write("h1 { color: #333; }\n")
                    f.write("h2 { color: #555; margin-top: 20px; }\n")
                    f.write("table { border-collapse: collapse; margin: 15px 0; }\n")
                    f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n")
                    f.write("th { background-color: #f2f2f2; }\n")
                    f.write("</style>\n</head>\n<body>\n")
                    
                    f.write("<h1>Analysis Results</h1>\n")
                    
                    if isinstance(data, dict):
                        for section, content in data.items():
                            f.write(f"<h2>{section}</h2>\n")
                            
                            if isinstance(content, dict):
                                f.write("<table>\n<tr><th>Property</th><th>Value</th></tr>\n")
                                for key, value in content.items():
                                    f.write(f"<tr><td>{key}</td><td>{value}</td></tr>\n")
                                f.write("</table>\n")
                            
                            elif isinstance(content, list):
                                if content and isinstance(content[0], dict):
                                    # Table with headers from dict keys
                                    headers = set()
                                    for item in content:
                                        headers.update(item.keys())
                                    
                                    f.write("<table>\n<tr>")
                                    for header in headers:
                                        f.write(f"<th>{header}</th>")
                                    f.write("</tr>\n")
                                    
                                    for item in content:
                                        f.write("<tr>")
                                        for header in headers:
                                            f.write(f"<td>{item.get(header, '')}</td>")
                                        f.write("</tr>\n")
                                    
                                    f.write("</table>\n")
                                else:
                                    # Simple list
                                    f.write("<ul>\n")
                                    for item in content:
                                        f.write(f"<li>{item}</li>\n")
                                    f.write("</ul>\n")
                            else:
                                f.write(f"<p>{content}</p>\n")
                    
                    f.write("</body>\n</html>")
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.debug(f"Data saved to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def load(file_path: Union[str, Path], 
            format: Optional[Union[StorageFormat, str]] = None) -> Any:
        """Load data from file in the specified format.
        
        Args:
            file_path: Path to load the data from
            format: Format to use (if None, inferred from file extension)
            
        Returns:
            Loaded data
        
        Raises:
            ValueError: If format cannot be determined or is unsupported
            FileNotFoundError: If the file does not exist
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        # Determine if file is compressed
        is_compressed = file_path.suffix.lower() == '.gz'
        if is_compressed:
            import gzip
            open_func = gzip.open
            # Use previous suffix for format detection
            file_suffix = file_path.stem.split('.')[-1]
        else:
            open_func = open
            file_suffix = file_path.suffix.lower().lstrip('.')
        
        # Determine format if not specified
        if format is None:
            try:
                format = StorageFormat(file_suffix)
            except ValueError:
                raise ValueError(f"Could not determine format from file extension: {file_suffix}")
        elif isinstance(format, str):
            try:
                format = StorageFormat(format.lower())
            except ValueError:
                raise ValueError(f"Unsupported format: {format}")
        
        # Load data in the specified format
        try:
            if format == StorageFormat.JSON:
                with open_func(file_path, 'rt') as f:
                    return json.load(f)
            
            elif format == StorageFormat.YAML:
                with open_func(file_path, 'rt') as f:
                    return yaml.safe_load(f)
            
            elif format == StorageFormat.PICKLE:
                with open_func(file_path, 'rb') as f:
                    return pickle.load(f)
            
            elif format == StorageFormat.NUMPY:
                if file_path.suffix.lower() == '.npz':
                    with np.load(file_path) as data:
                        # If it has only 'data', return that directly
                        if len(data.files) == 1 and data.files[0] == 'data':
                            return data['data']
                        # Otherwise return a dict of all arrays
                        return {k: data[k] for k in data.files}
                else:
                    return np.load(file_path, allow_pickle=True)
            
            elif format == StorageFormat.CSV:
                try:
                    return np.loadtxt(file_path, delimiter=',')
                except:
                    import csv
                    with open_func(file_path, 'rt', newline='') as f:
                        reader = csv.reader(f)
                        return list(reader)
            
            else:
                raise ValueError(f"Loading from {format} format is not supported")
            
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def exists(file_path: Union[str, Path]) -> bool:
        """Check if a file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if the file exists, False otherwise
        """
        return Path(file_path).exists()
    
    @staticmethod
    def get_path(base_dir: Union[str, Path], 
                file_name: str,
                format: Union[StorageFormat, str] = StorageFormat.JSON,
                create_dir: bool = True) -> Path:
        """Get a path for storing data with proper extension.
        
        Args:
            base_dir: Base directory
            file_name: File name without extension
            format: Storage format
            create_dir: Whether to create the directory if it doesn't exist
            
        Returns:
            Path object with proper extension
        """
        base_dir = Path(base_dir)
        
        # Create directory if needed
        if create_dir:
            base_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we have the format as enum
        if isinstance(format, str):
            try:
                format = StorageFormat(format.lower())
            except ValueError:
                raise ValueError(f"Unsupported format: {format}")
        
        # Ensure filename has correct extension
        if not file_name.endswith(f".{format.value}"):
            file_name = f"{file_name}.{format.value}"
        
        return base_dir / file_name
    
    @staticmethod
    def list_files(directory: Union[str, Path], 
                  format: Optional[Union[StorageFormat, str]] = None,
                  recursive: bool = False) -> List[Path]:
        """List files in a directory with optional format filtering.
        
        Args:
            directory: Directory to list files from
            format: Filter by format (if None, list all files)
            recursive: Whether to recursively list files in subdirectories
            
        Returns:
            List of file paths
        """
        directory = Path(directory)
        
        # Get format extension if specified
        if format:
            if isinstance(format, str):
                try:
                    format = StorageFormat(format.lower())
                except ValueError:
                    raise ValueError(f"Unsupported format: {format}")
            extension = f".{format.value}"
        else:
            extension = None
        
        # Find files
        if recursive:
            if extension:
                return list(directory.glob(f"**/*{extension}"))
            else:
                return [f for f in directory.glob("**/*") if f.is_file()]
        else:
            if extension:
                return list(directory.glob(f"*{extension}"))
            else:
                return [f for f in directory.iterdir() if f.is_file()]


# Add the missing classes needed by pipeline.py

class AnalysisVersion:
    """Version information for analysis results."""
    
    MAJOR = 1
    MINOR = 0
    PATCH = 0
    
    @classmethod
    def to_string(cls) -> str:
        """Convert version to string.
        
        Returns:
            Version string in the format "MAJOR.MINOR.PATCH"
        """
        return f"{cls.MAJOR}.{cls.MINOR}.{cls.PATCH}"
    
    @classmethod
    def from_string(cls, version_str: str) -> Tuple[int, int, int]:
        """Parse version string.
        
        Args:
            version_str: Version string in the format "MAJOR.MINOR.PATCH"
            
        Returns:
            Tuple of (MAJOR, MINOR, PATCH)
        """
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
        
        try:
            major = int(parts[0])
            minor = int(parts[1])
            patch = int(parts[2])
            return (major, minor, patch)
        except ValueError:
            raise ValueError(f"Invalid version string: {version_str}")
    
    @classmethod
    def is_compatible(cls, version_str: str) -> bool:
        """Check if a version is compatible with the current version.
        
        Args:
            version_str: Version string to check
            
        Returns:
            Whether the version is compatible
        """
        try:
            major, _, _ = cls.from_string(version_str)
            return major == cls.MAJOR
        except ValueError:
            return False


class AudioCache:
    """Cache for storing processed audio data to avoid redundant processing."""
    
    def __init__(self, cache_dir: str = "./cache", compression_level: int = 6):
        """Initialize the audio cache.
        
        Args:
            cache_dir: Directory to store cached audio files
            compression_level: Compression level for stored files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.compression_level = compression_level
        
        logger.info(f"Audio cache initialized with directory: {cache_dir}")
    
    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache path for an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Path to the cached file
        """
        # Create hash of the file path
        file_hash = self._compute_file_hash(audio_path)
        return self.cache_dir / f"{file_hash}.json.gz"
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hash of the file
        """
        import hashlib
        
        # Include the file path in the hash
        path_hash = hashlib.md5(file_path.encode()).hexdigest()
        
        # Include the file modification time
        try:
            mtime = os.path.getmtime(file_path)
            mtime_str = str(mtime)
            return hashlib.md5((path_hash + mtime_str).encode()).hexdigest()
        except OSError:
            # If file doesn't exist, just use the path hash
            return path_hash
    
    def get(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Get cached results for an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Cached results if available, None otherwise
        """
        cache_path = self._get_cache_path(audio_path)
        
        if not cache_path.exists():
            logger.debug(f"No cache found for {audio_path}")
            return None
        
        try:
            import gzip
            with gzip.open(cache_path, 'rt') as f:
                data = json.load(f)
            
            logger.info(f"Loaded cached results for {audio_path}")
            return data
        except Exception as e:
            logger.warning(f"Error loading cache for {audio_path}: {str(e)}")
            return None
    
    def set(self, audio_path: str, data: Dict[str, Any]) -> None:
        """Cache results for an audio file.
        
        Args:
            audio_path: Path to the audio file
            data: Analysis results to cache
        """
        cache_path = self._get_cache_path(audio_path)
        
        try:
            import gzip
            with gzip.open(cache_path, 'wt', compresslevel=self.compression_level) as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Cached results for {audio_path}")
        except Exception as e:
            logger.warning(f"Error caching results for {audio_path}: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached results."""
        try:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            logger.info("Audio cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing cache: {str(e)}")


class FeatureStorage:
    """Storage for audio analysis features and results."""
    
    def __init__(self, storage_dir: str = "./results"):
        """Initialize the feature storage.
        
        Args:
            storage_dir: Directory to store analysis results
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Feature storage initialized with directory: {storage_dir}")
    
    def save_results(self, audio_path: str, results: Dict[str, Any], 
                    metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save analysis results for an audio file.
        
        Args:
            audio_path: Path to the audio file
            results: Analysis results
            metadata: Additional metadata
            
        Returns:
            Path to the saved file
        """
        # Extract file name without extension
        file_name = Path(audio_path).stem
        
        # Create result object with metadata
        result_obj = {
            'audio_path': audio_path,
            'timestamp': time.time(),
            'metadata': metadata or {},
            'results': results
        }
        
        # Save to file
        results_path = self.storage_dir / f"{file_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump(result_obj, f, indent=2)
        
        logger.info(f"Saved analysis results for {audio_path} to {results_path}")
        return results_path
    
    def load_results(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Load analysis results for an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Analysis results if available, None otherwise
        """
        # Extract file name without extension
        file_name = Path(audio_path).stem
        
        # Check if results file exists
        results_path = self.storage_dir / f"{file_name}_results.json"
        if not results_path.exists():
            logger.debug(f"No results found for {audio_path}")
            return None
        
        try:
            with open(results_path, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded analysis results for {audio_path}")
            return data
        except Exception as e:
            logger.warning(f"Error loading results for {audio_path}: {str(e)}")
            return None
    
    def list_results(self) -> List[Dict[str, Any]]:
        """List all available analysis results.
        
        Returns:
            List of analysis result metadata
        """
        results = []
        
        for file_path in self.storage_dir.glob('*_results.json'):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract basic metadata
                results.append({
                    'file_name': file_path.name,
                    'audio_path': data.get('audio_path', ''),
                    'timestamp': data.get('timestamp', 0),
                    'metadata': data.get('metadata', {})
                })
            except Exception as e:
                logger.warning(f"Error reading results file {file_path}: {str(e)}")
        
        return results


# Import at the end to avoid issues with circular imports
import time  # For timestamp generation




