#!/usr/bin/env python3
"""
Script for subsampling large audio analysis results.

This script extracts selected features and/or time ranges from large analysis files
and creates smaller, more manageable files for visualization and further analysis.
"""

import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

# Add the project root to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Subsample large audio analysis results')
    parser.add_argument(
        '--input', '-i', 
        type=str, 
        required=True, 
        help='Path to the analysis results file (JSON or HDF5)'
    )
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        required=True, 
        help='Path to save the subsampled results'
    )
    parser.add_argument(
        '--features', '-f', 
        type=str, 
        nargs='+',
        help='List of features to extract (if not provided, all features will be subsampled)'
    )
    parser.add_argument(
        '--start-time', 
        type=float, 
        default=None, 
        help='Start time in seconds (if not provided, starts from the beginning)'
    )
    parser.add_argument(
        '--end-time', 
        type=float, 
        default=None, 
        help='End time in seconds (if not provided, ends at the end)'
    )
    parser.add_argument(
        '--downsample-factor', 
        type=int, 
        default=1, 
        help='Factor by which to downsample the data (e.g., 10 means keep every 10th sample)'
    )
    parser.add_argument(
        '--max-samples', 
        type=int, 
        default=None, 
        help='Maximum number of samples to keep per feature'
    )
    
    return parser.parse_args()

def time_to_index(time_seconds: float, sample_rate: int, hop_length: Optional[int] = None) -> int:
    """Convert time in seconds to index based on sample rate and hop length.
    
    Args:
        time_seconds: Time in seconds
        sample_rate: Sample rate of the audio
        hop_length: Hop length for frame-based features (if None, uses sample_rate)
        
    Returns:
        Index corresponding to the time
    """
    if hop_length is None:
        # For waveform, each sample is at sample_rate
        return int(time_seconds * sample_rate)
    else:
        # For frame-based features
        return int(time_seconds * sample_rate / hop_length)

def subsample_json(input_path: str, output_path: str, args: argparse.Namespace) -> None:
    """Subsample data from a JSON file.
    
    Args:
        input_path: Path to the input JSON file
        output_path: Path to save the subsampled JSON file
        args: Command line arguments
    """
    logger.info(f"Subsampling JSON data from {input_path}")
    
    try:
        # Load the input JSON file
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Create a new dictionary for the subsampled data
        subsampled_data = {}
        
        # Copy metadata
        for key, value in data.items():
            if key != 'features':
                subsampled_data[key] = value
        
        # Get the sample rate
        sample_rate = data.get('sample_rate', 44100)
        
        # Initialize features dictionary
        subsampled_data['features'] = {}
        
        # Get the features to extract
        features_to_extract = args.features if args.features else list(data.get('features', {}).keys())
        
        # Process each feature
        for feature_name in features_to_extract:
            if feature_name in data.get('features', {}):
                feature_data = data['features'][feature_name]
                
                # Skip if not a list or numpy array
                if not isinstance(feature_data, (list, np.ndarray)):
                    subsampled_data['features'][feature_name] = feature_data
                    continue
                
                # Convert to numpy array if it's a list
                if isinstance(feature_data, list):
                    feature_data = np.array(feature_data)
                
                # Determine hop length (if applicable)
                hop_length = None  # Default for waveform
                if feature_name != 'waveform':
                    # Estimate hop length based on feature length and duration
                    duration = data.get('duration', len(feature_data) / sample_rate)
                    if feature_data.shape[0] < duration * sample_rate:
                        hop_length = int(sample_rate * duration / feature_data.shape[0])
                
                # Calculate start and end indices
                start_idx = 0
                end_idx = len(feature_data)
                
                if args.start_time is not None:
                    start_idx = time_to_index(args.start_time, sample_rate, hop_length)
                    start_idx = max(0, min(start_idx, len(feature_data) - 1))
                
                if args.end_time is not None:
                    end_idx = time_to_index(args.end_time, sample_rate, hop_length)
                    end_idx = max(start_idx + 1, min(end_idx, len(feature_data)))
                
                # Extract the time range
                extracted_data = feature_data[start_idx:end_idx]
                
                # Apply downsampling if requested
                if args.downsample_factor > 1:
                    extracted_data = extracted_data[::args.downsample_factor]
                
                # Apply max samples limit if requested
                if args.max_samples is not None and len(extracted_data) > args.max_samples:
                    # Calculate the new downsample factor to get max_samples
                    new_factor = len(extracted_data) // args.max_samples + 1
                    extracted_data = extracted_data[::new_factor]
                
                # Convert back to list for JSON serialization
                subsampled_data['features'][feature_name] = extracted_data.tolist()
                
                logger.info(f"Subsampled {feature_name}: {len(feature_data)} -> {len(extracted_data)} samples")
            else:
                logger.warning(f"Feature '{feature_name}' not found in input data")
        
        # Update duration if we extracted a time range
        if args.start_time is not None or args.end_time is not None:
            start_time = args.start_time or 0
            end_time = args.end_time or data.get('duration', 0)
            subsampled_data['duration'] = end_time - start_time
            subsampled_data['extracted_time_range'] = [start_time, end_time]
        
        # Save the subsampled data
        with open(output_path, 'w') as f:
            json.dump(subsampled_data, f, indent=2)
        
        logger.info(f"Subsampled data saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error subsampling JSON data: {e}")
        sys.exit(1)

def subsample_hdf5(input_path: str, output_path: str, args: argparse.Namespace) -> None:
    """Subsample data from an HDF5 file.
    
    Args:
        input_path: Path to the input HDF5 file
        output_path: Path to save the subsampled HDF5 file
        args: Command line arguments
    """
    logger.info(f"Subsampling HDF5 data from {input_path}")
    
    try:
        # Open the input file
        with h5py.File(input_path, 'r') as in_file:
            # Get the sample rate
            sample_rate = in_file.attrs.get('sample_rate', 44100)
            
            # Get features to extract
            available_features = list(in_file.keys())
            features_to_extract = args.features if args.features else available_features
            
            # Create a new HDF5 file for the subsampled data
            with h5py.File(output_path, 'w') as out_file:
                # Copy attributes
                for key, value in in_file.attrs.items():
                    out_file.attrs[key] = value
                
                # Add subsampling metadata
                out_file.attrs['subsampled'] = True
                out_file.attrs['original_file'] = input_path
                if args.start_time is not None:
                    out_file.attrs['start_time'] = args.start_time
                if args.end_time is not None:
                    out_file.attrs['end_time'] = args.end_time
                if args.downsample_factor > 1:
                    out_file.attrs['downsample_factor'] = args.downsample_factor
                
                # Process each feature
                for feature_name in features_to_extract:
                    if feature_name in in_file:
                        in_dataset = in_file[feature_name]
                        
                        # Determine hop length (if applicable)
                        hop_length = None  # Default for waveform
                        if feature_name != 'waveform':
                            # Estimate hop length based on feature length and duration
                            duration = in_file.attrs.get('duration', in_dataset.shape[0] / sample_rate)
                            if in_dataset.shape[0] < duration * sample_rate:
                                hop_length = int(sample_rate * duration / in_dataset.shape[0])
                        
                        # Calculate start and end indices
                        start_idx = 0
                        end_idx = in_dataset.shape[0]
                        
                        if args.start_time is not None:
                            start_idx = time_to_index(args.start_time, sample_rate, hop_length)
                            start_idx = max(0, min(start_idx, in_dataset.shape[0] - 1))
                        
                        if args.end_time is not None:
                            end_idx = time_to_index(args.end_time, sample_rate, hop_length)
                            end_idx = max(start_idx + 1, min(end_idx, in_dataset.shape[0]))
                        
                        # Calculate the final indices after downsampling
                        if args.downsample_factor > 1:
                            indices = np.arange(start_idx, end_idx, args.downsample_factor)
                        else:
                            indices = np.arange(start_idx, end_idx)
                        
                        # Apply max samples limit if requested
                        if args.max_samples is not None and len(indices) > args.max_samples:
                            new_factor = len(indices) // args.max_samples + 1
                            indices = indices[::new_factor]
                        
                        # Create a new dataset with the extracted data
                        if len(in_dataset.shape) == 1:
                            # 1D data
                            out_dataset = out_file.create_dataset(
                                feature_name, 
                                shape=(len(indices),), 
                                dtype=in_dataset.dtype,
                                compression="gzip", 
                                compression_opts=9
                            )
                            # Read and write in chunks to avoid loading the entire dataset
                            chunk_size = 10000
                            for i in range(0, len(indices), chunk_size):
                                chunk_indices = indices[i:i+chunk_size]
                                out_dataset[i:i+len(chunk_indices)] = in_dataset[chunk_indices]
                        else:
                            # Multi-dimensional data
                            out_shape = (len(indices),) + in_dataset.shape[1:]
                            out_dataset = out_file.create_dataset(
                                feature_name, 
                                shape=out_shape, 
                                dtype=in_dataset.dtype,
                                compression="gzip", 
                                compression_opts=9
                            )
                            # Read and write in chunks to avoid loading the entire dataset
                            chunk_size = 1000
                            for i in range(0, len(indices), chunk_size):
                                chunk_indices = indices[i:i+chunk_size]
                                out_dataset[i:i+len(chunk_indices)] = in_dataset[chunk_indices]
                        
                        # Copy attributes
                        for key, value in in_dataset.attrs.items():
                            out_dataset.attrs[key] = value
                        
                        # Add subsampling info to dataset attributes
                        out_dataset.attrs['original_size'] = in_dataset.shape[0]
                        out_dataset.attrs['subsampled_size'] = len(indices)
                        out_dataset.attrs['reduction_factor'] = in_dataset.shape[0] / len(indices)
                        
                        logger.info(f"Subsampled {feature_name}: {in_dataset.shape[0]} -> {len(indices)} samples")
                    else:
                        logger.warning(f"Feature '{feature_name}' not found in input data")
                
                # Update duration if we extracted a time range
                if args.start_time is not None or args.end_time is not None:
                    start_time = args.start_time or 0
                    end_time = args.end_time or in_file.attrs.get('duration', 0)
                    out_file.attrs['duration'] = end_time - start_time
                    out_file.attrs['original_duration'] = in_file.attrs.get('duration', 0)
        
        logger.info(f"Subsampled data saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error subsampling HDF5 data: {e}")
        sys.exit(1)

def main():
    """Main function."""
    args = parse_args()
    
    # Determine file type
    input_ext = os.path.splitext(args.input)[1].lower()
    output_ext = os.path.splitext(args.output)[1].lower()
    
    # Check that input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Check that input and output extensions match
    if input_ext != output_ext:
        logger.error(f"Input and output file extensions must match: {input_ext} vs {output_ext}")
        sys.exit(1)
    
    # Process based on file type
    if input_ext == '.json':
        subsample_json(args.input, args.output, args)
    elif input_ext in ['.h5', '.hdf5']:
        subsample_hdf5(args.input, args.output, args)
    else:
        logger.error(f"Unsupported file format: {input_ext}")
        sys.exit(1)

if __name__ == "__main__":
    main() 