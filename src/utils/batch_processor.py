#!/usr/bin/env python3
"""
Batch processor for handling multiple audio files with different configuration settings.

This module provides utilities for batch processing audio files using different
configuration profiles.
"""

import os
import time
import json
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from src.utils.logging_utils import get_logger
from src.core.processor import AudioProcessor
from src.utils.export import export_results
from src.utils.progress import track_progress, ProgressContext, ProgressType

logger = get_logger(__name__)

@dataclass
class BatchTaskResult:
    """Class to store the result of a batch processing task."""
    file_path: str
    config_name: str
    status: str = "pending"
    error: Optional[str] = None
    processing_time: float = 0.0
    results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "file_path": self.file_path,
            "config_name": self.config_name,
            "status": self.status,
            "error": self.error,
            "processing_time": self.processing_time,
            "results": self.results
        }


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    name: str
    config: Dict[str, Any]
    file_patterns: List[str] = field(default_factory=list)
    files: List[str] = field(default_factory=list)


class BatchProcessor:
    """Processor for handling batch analysis of audio files."""
    
    def __init__(self, 
                output_dir: Union[str, Path] = "results",
                max_workers: int = None,
                progress_type: Union[str, ProgressType] = ProgressType.BAR):
        """Initialize batch processor.
        
        Args:
            output_dir: Directory to save results
            max_workers: Maximum number of parallel workers (None = CPU count)
            progress_type: Type of progress indicator to display
        """
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.max_workers = max_workers
        self.progress_type = progress_type
        self.configs: List[BatchConfig] = []
        self.results: List[BatchTaskResult] = []
        self.stats = {
            "total": 0,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "total_time": 0.0,
        }
    
    def add_config(self, 
                  name: str, 
                  config: Dict[str, Any],
                  file_patterns: List[str] = None,
                  files: List[str] = None) -> None:
        """Add a configuration profile to the batch processor.
        
        Args:
            name: Name of the configuration
            config: Configuration dictionary
            file_patterns: Glob patterns for matching files
            files: Explicit list of files to process
        """
        self.configs.append(BatchConfig(
            name=name,
            config=config,
            file_patterns=file_patterns or [],
            files=files or []
        ))
        logger.info(f"Added config '{name}' with {len(file_patterns or [])} patterns and {len(files or [])} explicit files")
    
    def _resolve_files(self) -> List[Dict[str, Any]]:
        """Resolve all files to process based on patterns and explicit files.
        
        Returns:
            List of dictionaries with file_path and config_name
        """
        tasks = []
        
        with ProgressContext(
            total=len(self.configs),
            desc="Resolving files",
            progress_type=self.progress_type
        ) as progress:
            for config in self.configs:
                # Add explicit files
                for file_path in config.files:
                    if os.path.isfile(file_path):
                        tasks.append({
                            "file_path": file_path,
                            "config_name": config.name
                        })
                    else:
                        logger.warning(f"File not found (skipping): {file_path}")
                
                # Resolve patterns
                for pattern in config.file_patterns:
                    for path in Path().glob(pattern):
                        if path.is_file():
                            tasks.append({
                                "file_path": str(path),
                                "config_name": config.name
                            })
                            
                progress.update(1)
        
        # Remove duplicates (same file/config combination)
        unique_tasks = []
        seen = set()
        
        for task in tasks:
            key = f"{task['file_path']}:{task['config_name']}"
            if key not in seen:
                seen.add(key)
                unique_tasks.append(task)
        
        logger.info(f"Resolved {len(unique_tasks)} unique file/config tasks from {len(tasks)} total matches")
        return unique_tasks
    
    def _process_file(self, file_path: str, config_name: str) -> BatchTaskResult:
        """Process a single file with the given configuration.
        
        Args:
            file_path: Path to the audio file
            config_name: Name of the configuration to use
            
        Returns:
            BatchTaskResult: Result of the processing
        """
        result = BatchTaskResult(file_path=file_path, config_name=config_name)
        
        # Find the configuration
        config_data = None
        for config in self.configs:
            if config.name == config_name:
                config_data = config.config
                break
        
        if not config_data:
            result.status = "failed"
            result.error = f"Configuration '{config_name}' not found"
            return result
        
        # Process the file
        start_time = time.time()
        
        try:
            # Create processor with config
            processor = AudioProcessor(**config_data)
            
            # Process the file
            analysis_result = processor.process_file(file_path)
            
            # Update result
            result.status = "success"
            result.results = analysis_result
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            logger.error(f"Error processing file {file_path} with config {config_name}: {e}")
        
        # Calculate processing time
        result.processing_time = time.time() - start_time
        
        return result
    
    def process(self, 
               parallel: bool = True, 
               export_format: str = "json",
               summary_file: str = "batch_summary.json") -> Dict[str, Any]:
        """Process all files with their respective configurations.
        
        Args:
            parallel: Whether to process files in parallel
            export_format: Format for exporting individual results (json, csv, yaml)
            summary_file: Filename for the batch summary
            
        Returns:
            Dictionary with batch processing statistics and results
        """
        tasks = self._resolve_files()
        self.stats["total"] = len(tasks)
        
        logger.info(f"Starting batch processing of {len(tasks)} tasks")
        self.results = []
        
        batch_start_time = time.time()
        
        if parallel and self.max_workers != 1:
            # Process in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_file, task["file_path"], task["config_name"]): task
                    for task in tasks
                }
                
                # Track progress of completed futures
                with ProgressContext(
                    total=len(futures),
                    desc="Processing files",
                    progress_type=self.progress_type,
                    unit="files"
                ) as progress:
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        self._handle_result(result, export_format)
                        
                        # Update progress
                        progress.update(1)
                        if hasattr(progress, 'set_postfix'):
                            progress.set_postfix(
                                success=f"{self.stats['success']}/{self.stats['processed']}",
                                failed=self.stats['failed']
                            )
        else:
            # Process sequentially
            for task in track_progress(
                tasks,
                desc="Processing files",
                progress_type=self.progress_type,
                unit="files"
            ):
                result = self._process_file(task["file_path"], task["config_name"])
                self._handle_result(result, export_format)
        
        # Update final stats
        self.stats["total_time"] = time.time() - batch_start_time
        
        # Create summary and save it
        logger.info("Creating batch summary...")
        summary = self._create_summary()
        summary_path = self.output_dir / summary_file
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Batch processing completed. Results saved to {self.output_dir}")
        logger.info(f"Processed {self.stats['total']} files: {self.stats['success']} successful, {self.stats['failed']} failed")
        logger.info(f"Total processing time: {self.stats['total_time']:.2f} seconds")
        
        return summary
    
    def _handle_result(self, result: BatchTaskResult, export_format: str) -> None:
        """Handle a completed task result.
        
        Args:
            result: Result of a processing task
            export_format: Format for exporting individual results
        """
        self.results.append(result)
        
        # Update stats
        self.stats["processed"] += 1
        if result.status == "success":
            self.stats["success"] += 1
        else:
            self.stats["failed"] += 1
        
        # Export individual result if successful
        if result.status == "success" and result.results:
            file_name = os.path.basename(result.file_path)
            base_name, _ = os.path.splitext(file_name)
            export_path = self.output_dir / f"{base_name}_{result.config_name}.{export_format}"
            
            export_results(result.results, str(export_path), export_format)
    
    def _create_summary(self) -> Dict[str, Any]:
        """Create a summary of the batch processing.
        
        Returns:
            Dictionary with batch processing summary
        """
        return {
            "stats": self.stats,
            "results": [result.to_dict() for result in self.results]
        }


def process_batch_file(batch_file: str, 
                     output_dir: str = "results",
                     parallel: bool = True,
                     export_format: str = "json",
                     max_workers: int = None,
                     progress_type: Union[str, ProgressType] = ProgressType.BAR) -> Dict[str, Any]:
    """Process a batch file with multiple configurations.
    
    Args:
        batch_file: Path to the batch configuration file (JSON)
        output_dir: Directory to save results
        parallel: Whether to process files in parallel
        export_format: Format for exporting individual results
        max_workers: Maximum number of parallel workers
        progress_type: Type of progress display to use
        
    Returns:
        Dictionary with batch processing statistics
        
    Example batch file format:
    {
        "configs": [
            {
                "name": "default",
                "config": {
                    "sample_rate": 44100,
                    "features": ["rms", "spectral_centroid"]
                },
                "file_patterns": ["audio/*.wav"],
                "files": ["path/to/specific.wav"]
            },
            {
                "name": "high_quality",
                "config": {
                    "sample_rate": 48000,
                    "features": ["rms", "spectral_centroid", "spectral_contrast", "mfcc"]
                },
                "files": ["path/to/another.wav"]
            }
        ]
    }
    """
    # Load batch file
    try:
        with open(batch_file, 'r') as f:
            batch_config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading batch file: {e}")
        raise
    
    # Create batch processor
    processor = BatchProcessor(
        output_dir=output_dir, 
        max_workers=max_workers,
        progress_type=progress_type
    )
    
    # Add configurations
    for config in batch_config.get("configs", []):
        processor.add_config(
            name=config.get("name", "default"),
            config=config.get("config", {}),
            file_patterns=config.get("file_patterns", []),
            files=config.get("files", [])
        )
    
    # Process batch
    return processor.process(
        parallel=parallel,
        export_format=export_format
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch process audio files with different configurations")
    parser.add_argument("batch_file", help="Batch configuration file (JSON)")
    parser.add_argument("--output-dir", "-o", default="results", help="Output directory for results")
    parser.add_argument("--format", "-f", choices=["json", "csv", "yaml"], default="json",
                        help="Export format for individual results")
    parser.add_argument("--sequential", "-s", action="store_true", help="Process files sequentially (not in parallel)")
    parser.add_argument("--workers", "-w", type=int, default=None, 
                        help="Maximum number of parallel workers (default: CPU count)")
    parser.add_argument("--progress", "-p", choices=["bar", "spinner", "silent"], default="bar",
                        help="Type of progress display")
    
    args = parser.parse_args()
    
    try:
        process_batch_file(
            batch_file=args.batch_file,
            output_dir=args.output_dir,
            parallel=not args.sequential,
            export_format=args.format,
            max_workers=args.workers,
            progress_type=args.progress
        )
    except Exception as e:
        logger.error(f"Batch processing failed: {e}") 