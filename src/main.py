#!/usr/bin/env python3
"""
Heihachi - Neural Processing of Electronic Music

Command-line interface entry point for running the audio analysis pipeline.
"""

import argparse
import os
import sys
import logging
from typing import Optional, List

# Make sure the package is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components from the package
from src.core.pipeline import Pipeline
from src.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Heihachi - Neural Processing of Electronic Music",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to input audio file or directory"
    )
    
    parser.add_argument(
        "-c", "--config", 
        default=os.path.join(os.path.dirname(__file__), "configs", "default.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "-o", "--output", 
        default="../results",
        help="Path to output directory"
    )
    
    parser.add_argument(
        "--cache-dir",
        default="../cache",
        help="Path to cache directory"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def process_file(input_file: str, config_path: str, output_dir: str, cache_dir: str) -> bool:
    """
    Process a single audio file.
    
    Args:
        input_file: Path to audio file
        config_path: Path to config file
        output_dir: Path to output directory
        cache_dir: Path to cache directory
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    try:
        # Initialize pipeline
        pipeline = Pipeline(config_path)
        
        # Process the file
        results = pipeline.process(input_file)
        
        # Print a summary of results
        print(f"\nResults for {os.path.basename(input_file)}:")
        print(f"  Analysis completed successfully")
        print(f"  Results saved to {output_dir}")
        
        return True
    except Exception as e:
        logging.error(f"Error processing {input_file}: {str(e)}")
        return False


def process_directory(input_dir: str, config_path: str, output_dir: str, cache_dir: str) -> List[str]:
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Path to directory containing audio files
        config_path: Path to config file
        output_dir: Path to output directory
        cache_dir: Path to cache directory
        
    Returns:
        List[str]: List of files that failed processing
    """
    # Get all audio files in the directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.ogg']
    audio_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and 
        os.path.splitext(f)[1].lower() in audio_extensions
    ]
    
    if not audio_files:
        logging.warning(f"No audio files found in {input_dir}")
        return []
    
    # Process each file
    failed_files = []
    for i, file_path in enumerate(audio_files):
        print(f"\nProcessing file {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        success = process_file(file_path, config_path, output_dir, cache_dir)
        if not success:
            failed_files.append(file_path)
    
    return failed_files


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Ensure cache directory exists
    os.makedirs(args.cache_dir, exist_ok=True)
    
    print("Heihachi Audio Analysis")
    print("======================\n")
    
    # Check if input is a file or directory
    if os.path.isfile(args.input_file):
        # Process a single file
        success = process_file(args.input_file, args.config, args.output, args.cache_dir)
        sys.exit(0 if success else 1)
    elif os.path.isdir(args.input_file):
        # Process all files in the directory
        failed_files = process_directory(args.input_file, args.config, args.output, args.cache_dir)
        
        if failed_files:
            print(f"\nFailed to process {len(failed_files)} files:")
            for file_path in failed_files:
                print(f"  - {file_path}")
            sys.exit(1)
        else:
            print("\nAll files processed successfully!")
            sys.exit(0)
    else:
        logging.error(f"Input path not found: {args.input_file}")
        sys.exit(1)


if __name__ == "__main__":
    main() 