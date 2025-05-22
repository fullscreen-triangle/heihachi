#!/usr/bin/env python3
"""
Script for summarizing large audio analysis results.

This script generates statistical summaries of the analysis results 
without loading the entire dataset into memory.
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Summarize large audio analysis results')
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True, 
        help='Path to the analysis results file (JSON or HDF5)'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='./analysis_summary.json', 
        help='Path to save the summary report'
    )
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=1000000, 
        help='Chunk size for loading large arrays'
    )
    
    return parser.parse_args()

def get_feature_chunks(file_path: str, feature_name: str, chunk_size: int):
    """Generator to yield chunks of a feature from an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        feature_name: Name of the feature to load
        chunk_size: Size of each chunk to load
        
    Yields:
        Chunks of the feature data as numpy arrays
    """
    with h5py.File(file_path, 'r') as f:
        if feature_name in f:
            dataset = f[feature_name]
            total_size = dataset.shape[0]
            
            for i in range(0, total_size, chunk_size):
                end = min(i + chunk_size, total_size)
                yield dataset[i:end]
        else:
            logger.warning(f"Feature '{feature_name}' not found in HDF5 file")
            return

def calculate_feature_stats(file_path: str, feature_name: str, chunk_size: int) -> Dict[str, Any]:
    """Calculate statistics for a feature in chunks.
    
    Args:
        file_path: Path to the HDF5 file
        feature_name: Name of the feature to analyze
        chunk_size: Size of each chunk to load
        
    Returns:
        Dictionary of statistics
    """
    logger.info(f"Calculating statistics for feature: {feature_name}")
    
    stats = {'feature': feature_name}
    
    with h5py.File(file_path, 'r') as f:
        if feature_name in f:
            dataset = f[feature_name]
            
            # Get metadata
            stats['shape'] = dataset.shape
            stats['dtype'] = str(dataset.dtype)
            stats['size_mb'] = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
            
            # Calculate statistics in chunks
            min_vals, max_vals = [], []
            sum_vals, sum_squared = 0, 0
            count = 0
            n_chunks = 0
            
            for chunk in get_feature_chunks(file_path, feature_name, chunk_size):
                min_vals.append(np.min(chunk))
                max_vals.append(np.max(chunk))
                sum_vals += np.sum(chunk)
                sum_squared += np.sum(np.square(chunk))
                count += len(chunk)
                n_chunks += 1
            
            if count > 0:
                stats['min'] = float(np.min(min_vals))
                stats['max'] = float(np.max(max_vals))
                stats['mean'] = float(sum_vals / count)
                
                # Calculate variance and standard deviation
                variance = (sum_squared / count) - (stats['mean'] ** 2)
                stats['variance'] = float(variance)
                stats['std'] = float(np.sqrt(variance))
                
                # Calculate sparsity (percentage of zero values)
                # We need another pass through the data
                zero_count = 0
                for chunk in get_feature_chunks(file_path, feature_name, chunk_size):
                    zero_count += np.sum(chunk == 0)
                
                stats['sparsity'] = float(zero_count / count)
                stats['n_chunks_processed'] = n_chunks
            
            return stats
        else:
            logger.warning(f"Feature '{feature_name}' not found in HDF5 file")
            return {'feature': feature_name, 'error': 'Feature not found'}

def summarize_json_results(file_path: str) -> Dict[str, Any]:
    """Summarize results from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the summary
    """
    logger.info(f"Summarizing JSON results: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        summary = {
            'file_format': 'JSON',
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'metadata': {},
            'features': {}
        }
        
        # Extract metadata (non-feature fields)
        for key, value in results.items():
            if key != 'features':
                summary['metadata'][key] = value
        
        # Summarize features
        if 'features' in results:
            for feature_name, feature_data in results['features'].items():
                if isinstance(feature_data, (list, np.ndarray)) and len(feature_data) > 0:
                    # Convert to numpy array for statistics
                    if not isinstance(feature_data, np.ndarray):
                        feature_data = np.array(feature_data)
                    
                    summary['features'][feature_name] = {
                        'shape': feature_data.shape,
                        'dtype': str(feature_data.dtype),
                        'size_mb': feature_data.size * feature_data.dtype.itemsize / (1024 * 1024),
                        'min': float(np.min(feature_data)),
                        'max': float(np.max(feature_data)),
                        'mean': float(np.mean(feature_data)),
                        'std': float(np.std(feature_data)),
                        'sparsity': float(np.sum(feature_data == 0) / feature_data.size)
                    }
                else:
                    summary['features'][feature_name] = {
                        'type': str(type(feature_data)),
                        'summary': str(feature_data)[:100] + '...' if isinstance(feature_data, str) and len(str(feature_data)) > 100 else str(feature_data)
                    }
        
        return summary
    
    except Exception as e:
        logger.error(f"Error summarizing JSON file: {e}")
        return {'error': str(e)}

def summarize_hdf5_results(file_path: str, chunk_size: int) -> Dict[str, Any]:
    """Summarize results from an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        chunk_size: Size of each chunk to load
        
    Returns:
        Dictionary containing the summary
    """
    logger.info(f"Summarizing HDF5 results: {file_path}")
    
    try:
        summary = {
            'file_format': 'HDF5',
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'metadata': {},
            'features': {}
        }
        
        with h5py.File(file_path, 'r') as f:
            # Extract metadata (attributes)
            for key in f.attrs:
                summary['metadata'][key] = f.attrs[key]
            
            # List available features
            features = list(f.keys())
            summary['feature_count'] = len(features)
            
            # Calculate total size
            total_size_mb = 0
            for feature_name in features:
                dataset = f[feature_name]
                size_mb = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
                total_size_mb += size_mb
            
            summary['total_data_size_mb'] = total_size_mb
        
        # Process each feature to get statistics
        for feature_name in features:
            feature_stats = calculate_feature_stats(file_path, feature_name, chunk_size)
            summary['features'][feature_name] = feature_stats
        
        return summary
    
    except Exception as e:
        logger.error(f"Error summarizing HDF5 file: {e}")
        return {'error': str(e)}

def main():
    """Main function."""
    args = parse_args()
    
    # Determine file type
    file_ext = os.path.splitext(args.input)[1].lower()
    
    if file_ext == '.json':
        summary = summarize_json_results(args.input)
    elif file_ext in ['.h5', '.hdf5']:
        summary = summarize_hdf5_results(args.input, args.chunk_size)
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        sys.exit(1)
    
    # Save summary to file
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {args.output}")
    
    # Print a brief summary to console
    print("\nSUMMARY OF ANALYSIS RESULTS")
    print("=" * 40)
    print(f"File: {args.input}")
    print(f"File Format: {summary.get('file_format', 'Unknown')}")
    print(f"File Size: {summary.get('file_size_mb', 0):.2f} MB")
    
    if 'metadata' in summary:
        print("\nMetadata:")
        for key, value in summary['metadata'].items():
            print(f"  {key}: {value}")
    
    if 'features' in summary:
        print("\nFeatures:")
        for feature_name, feature_stats in summary['features'].items():
            shape_str = f"Shape: {feature_stats.get('shape', 'Unknown')}"
            size_str = f"Size: {feature_stats.get('size_mb', 0):.2f} MB"
            print(f"  {feature_name}: {shape_str}, {size_str}")
    
    print("\nFull summary saved to:", args.output)

if __name__ == "__main__":
    main() 