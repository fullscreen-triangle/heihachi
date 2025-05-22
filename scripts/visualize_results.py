#!/usr/bin/env python3
"""
Script for visualizing large audio analysis results.

This script loads analysis results in chunks/batches to handle very large data files
and generates visualizations using the existing visualization module.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import h5py

# Add the project root to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.plots import (
    plot_waveform,
    plot_spectrogram,
    plot_feature_comparison,
    plot_feature_distribution,
    create_summary_plots
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize large audio analysis results')
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True, 
        help='Path to the analysis results file (JSON or HDF5)'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default='./visualizations', 
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--feature', '-f', 
        type=str, 
        default=None, 
        help='Specific feature to visualize (if not provided, all features will be visualized)'
    )
    parser.add_argument(
        '--chunk-size', 
        type=int, 
        default=1000000, 
        help='Chunk size for loading large arrays'
    )
    parser.add_argument(
        '--dpi', 
        type=int, 
        default=300, 
        help='DPI for saving figures'
    )
    parser.add_argument(
        '--format', 
        type=str, 
        default='png', 
        choices=['png', 'jpg', 'svg', 'pdf'],
        help='Format for saving figures'
    )
    
    return parser.parse_args()

def load_json_results(file_path: str) -> Dict[str, Any]:
    """Load analysis results from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the analysis results
    """
    logger.info(f"Loading results from JSON: {file_path}")
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")
        sys.exit(1)

def load_hdf5_results(file_path: str) -> Dict[str, Any]:
    """Load analysis results from an HDF5 file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing analysis results metadata
    """
    logger.info(f"Loading results from HDF5: {file_path}")
    try:
        with h5py.File(file_path, 'r') as f:
            # Extract metadata
            metadata = {}
            
            # Get all attributes at root level
            for key in f.attrs:
                metadata[key] = f.attrs[key]
            
            # Get group structure
            metadata['features'] = list(f.keys())
            
            return metadata
    except Exception as e:
        logger.error(f"Error loading HDF5 file: {e}")
        sys.exit(1)

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

def visualize_feature(feature_name: str, 
                     file_path: str, 
                     output_dir: str,
                     chunk_size: int,
                     dpi: int,
                     format: str):
    """Visualize a specific feature from the analysis results.
    
    Args:
        feature_name: Name of the feature to visualize
        file_path: Path to the analysis results file
        output_dir: Directory to save visualizations
        chunk_size: Size of each chunk to load
        dpi: DPI for saving figures
        format: Format for saving figures
    """
    logger.info(f"Visualizing feature: {feature_name}")
    
    # Create output directory for this feature
    feature_dir = os.path.join(output_dir, feature_name)
    os.makedirs(feature_dir, exist_ok=True)
    
    # Determine file type
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.json':
        # Load JSON results
        results = load_json_results(file_path)
        if 'features' in results and feature_name in results['features']:
            feature_data = results['features'][feature_name]
            
            # Simple visualization for small features
            if isinstance(feature_data, (list, np.ndarray)) and len(feature_data) > 0:
                sample_rate = results.get('sample_rate', 44100)
                
                # Create figure based on feature type
                if feature_name == 'waveform':
                    fig = plot_waveform(feature_data, sample_rate)
                elif feature_name == 'spectrogram':
                    fig = plot_spectrogram(feature_data, sample_rate)
                else:
                    # Generic time series plot
                    fig, ax = plt.subplots(figsize=(10, 4))
                    duration = results.get('duration', len(feature_data) / sample_rate)
                    time = np.linspace(0, duration, len(feature_data))
                    ax.plot(time, feature_data, color='blue', alpha=0.7)
                    ax.set_title(f"{feature_name.capitalize()}")
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel(feature_name)
                    ax.grid(True, alpha=0.3)
                
                # Save figure
                fig.savefig(os.path.join(feature_dir, f"{feature_name}.{format}"), dpi=dpi)
                plt.close(fig)
        else:
            logger.warning(f"Feature '{feature_name}' not found in JSON results")
    
    elif file_ext in ['.h5', '.hdf5']:
        # Process HDF5 file
        with h5py.File(file_path, 'r') as f:
            if feature_name in f:
                dataset = f[feature_name]
                
                # Get metadata
                sample_rate = f.attrs.get('sample_rate', 44100)
                
                # Handle different feature types
                if feature_name == 'waveform':
                    # For waveform, we can process in chunks and create multiple plots
                    chunk_count = 0
                    for chunk_idx, chunk in enumerate(get_feature_chunks(file_path, feature_name, chunk_size)):
                        fig = plot_waveform(
                            chunk, 
                            sample_rate, 
                            title=f"Waveform (Chunk {chunk_idx+1})"
                        )
                        fig.savefig(os.path.join(feature_dir, f"waveform_chunk_{chunk_idx+1}.{format}"), dpi=dpi)
                        plt.close(fig)
                        chunk_count += 1
                    
                    logger.info(f"Created {chunk_count} waveform chunk visualizations")
                
                elif feature_name == 'spectrogram':
                    # For spectrogram, process chunks if it's too large
                    if dataset.shape[0] > chunk_size:
                        for chunk_idx, chunk in enumerate(get_feature_chunks(file_path, feature_name, chunk_size)):
                            fig = plot_spectrogram(
                                chunk, 
                                sample_rate, 
                                title=f"Spectrogram (Chunk {chunk_idx+1})"
                            )
                            fig.savefig(os.path.join(feature_dir, f"spectrogram_chunk_{chunk_idx+1}.{format}"), dpi=dpi)
                            plt.close(fig)
                    else:
                        # Load entire spectrogram if it's small enough
                        fig = plot_spectrogram(dataset[:], sample_rate)
                        fig.savefig(os.path.join(feature_dir, f"spectrogram.{format}"), dpi=dpi)
                        plt.close(fig)
                
                else:
                    # For other features, process in chunks and create summary stats
                    # Also create visualization of first chunk for preview
                    chunk_data = next(get_feature_chunks(file_path, feature_name, chunk_size))
                    
                    # Create preview visualization
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(chunk_data, alpha=0.7)
                    ax.set_title(f"{feature_name.capitalize()} (Preview)")
                    ax.set_xlabel('Sample')
                    ax.set_ylabel(feature_name)
                    ax.grid(True, alpha=0.3)
                    fig.savefig(os.path.join(feature_dir, f"{feature_name}_preview.{format}"), dpi=dpi)
                    plt.close(fig)
                    
                    # Calculate and save statistics
                    stats = {
                        'feature': feature_name,
                        'shape': dataset.shape,
                        'dtype': str(dataset.dtype),
                    }
                    
                    # Calculate statistics in chunks
                    min_vals, max_vals, sum_vals, count = [], [], 0, 0
                    for chunk in get_feature_chunks(file_path, feature_name, chunk_size):
                        min_vals.append(np.min(chunk))
                        max_vals.append(np.max(chunk))
                        sum_vals += np.sum(chunk)
                        count += len(chunk)
                    
                    stats['min'] = float(np.min(min_vals))
                    stats['max'] = float(np.max(max_vals))
                    stats['mean'] = float(sum_vals / count) if count > 0 else 0
                    
                    # Save statistics
                    with open(os.path.join(feature_dir, 'stats.json'), 'w') as f:
                        json.dump(stats, f, indent=2)
                    
                    logger.info(f"Calculated and saved statistics for {feature_name}")
            else:
                logger.warning(f"Feature '{feature_name}' not found in HDF5 file")
    else:
        logger.error(f"Unsupported file format: {file_ext}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Determine file type
    file_ext = os.path.splitext(args.input)[1].lower()
    
    if file_ext == '.json':
        # Load entire JSON file
        results = load_json_results(args.input)
        
        # If a specific feature is requested, visualize only that feature
        if args.feature:
            visualize_feature(
                args.feature, 
                args.input, 
                args.output, 
                args.chunk_size,
                args.dpi,
                args.format
            )
        else:
            # Visualize all features
            if 'features' in results:
                for feature_name in results['features']:
                    visualize_feature(
                        feature_name, 
                        args.input, 
                        args.output, 
                        args.chunk_size,
                        args.dpi,
                        args.format
                    )
            else:
                logger.error("No features found in JSON results")
    
    elif file_ext in ['.h5', '.hdf5']:
        # Get available features from HDF5 file
        with h5py.File(args.input, 'r') as f:
            available_features = list(f.keys())
        
        logger.info(f"Available features: {available_features}")
        
        # If a specific feature is requested, visualize only that feature
        if args.feature:
            if args.feature in available_features:
                visualize_feature(
                    args.feature, 
                    args.input, 
                    args.output, 
                    args.chunk_size,
                    args.dpi,
                    args.format
                )
            else:
                logger.error(f"Feature '{args.feature}' not found in HDF5 file")
        else:
            # Visualize all features
            for feature_name in available_features:
                visualize_feature(
                    feature_name, 
                    args.input, 
                    args.output, 
                    args.chunk_size,
                    args.dpi,
                    args.format
                )
    
    else:
        logger.error(f"Unsupported file format: {file_ext}")
        sys.exit(1)
    
    logger.info(f"Visualizations saved to: {args.output}")

if __name__ == "__main__":
    main() 