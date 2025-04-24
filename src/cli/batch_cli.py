#!/usr/bin/env python3
"""
Command-line interface for batch processing audio files.

This module provides a command-line interface for processing multiple audio files
with different configurations, using the progress utilities for visual feedback.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logging_utils import get_logger, configure_logging
from src.utils.progress import ProgressManager, cli_progress_bar
from src.utils.file_utils import ensure_directory, load_json, find_files
from src.core.pipeline import Pipeline
from src.utils.config import load_config
from src.utils.batch_processor import BatchProcessor, process_batch_file
from src.utils.export import export_results

logger = get_logger(__name__)
console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the batch CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Batch process audio files with different configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all audio files in a directory with a single configuration
  python -m src.cli.batch_cli --input-dir data/audio --config configs/default.json

  # Process with multiple configurations
  python -m src.cli.batch_cli --input-dir data/audio --config configs/config1.json configs/config2.json

  # Process with a batch specification file
  python -m src.cli.batch_cli --batch-file batch_spec.json

  # Export results in multiple formats
  python -m src.cli.batch_cli --input-dir data/audio --config configs/default.json --export json csv
"""
    )
    
    # Input specification
    input_group = parser.add_argument_group("Input Options")
    input_source = input_group.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--input-dir", type=str, 
                           help="Directory containing audio files to process")
    input_source.add_argument("--batch-file", type=str,
                           help="JSON file containing batch processing specification")
    
    input_group.add_argument("--pattern", type=str, default="*.wav",
                         help="File pattern to match (default: *.wav)")
    
    input_group.add_argument("--max-files", type=int, 
                         help="Maximum number of files to process")
    
    # Configuration
    config_group = parser.add_argument_group("Configuration Options")
    config_group.add_argument("--config", type=str, nargs="+",
                           help="Configuration file(s) to use (required with --input-dir)")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str, default="results",
                           help="Directory to save results (default: results)")
    
    output_group.add_argument("--export", type=str, nargs="+", 
                           choices=["json", "csv", "yaml"],
                           default=["json"],
                           help="Export formats (default: json)")
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument("--parallel", action="store_true", default=True,
                         help="Process files in parallel (default: True)")
    
    proc_group.add_argument("--no-parallel", action="store_false", dest="parallel",
                         help="Disable parallel processing")
    
    proc_group.add_argument("--workers", type=int,
                         help="Number of worker processes (default: CPU count)")
    
    # Display options
    display_group = parser.add_argument_group("Display Options")
    display_group.add_argument("--progress", choices=["bar", "rich", "simple", "none"],
                            default="rich",
                            help="Progress display type (default: rich)")
    
    display_group.add_argument("--log-level", type=str, 
                            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                            default="INFO", 
                            help="Set logging level (default: INFO)")
    
    display_group.add_argument("--quiet", action="store_true",
                            help="Minimize output (overrides --progress)")
    
    return parser


def process_directory(args: argparse.Namespace) -> int:
    """Process audio files from an input directory with specified configs.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    # Validate arguments
    if not args.config:
        console.print("[red]Error: --config is required when using --input-dir[/red]")
        return 1
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        return 1
    
    # Validate configuration files
    configs = []
    for config_path in args.config:
        config_file = Path(config_path)
        if not config_file.exists():
            console.print(f"[red]Error: Configuration file not found: {config_file}[/red]")
            return 1
        
        try:
            config = load_config(config_file)
            configs.append((config_file.stem, config))
        except Exception as e:
            console.print(f"[red]Error loading configuration {config_file}: {e}[/red]")
            return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    
    # Find audio files matching pattern
    file_pattern = args.pattern
    try:
        audio_files = find_files(input_dir, file_pattern)
        if args.max_files:
            audio_files = audio_files[:args.max_files]
            
        if not audio_files:
            console.print(f"[yellow]No files matching '{file_pattern}' found in {input_dir}[/yellow]")
            return 0
            
        console.print(f"Found {len(audio_files)} audio files to process")
    except Exception as e:
        console.print(f"[red]Error finding audio files: {e}[/red]")
        return 1
    
    # Determine progress type
    progress_type = args.progress
    if args.quiet:
        progress_type = "none"
    
    # Initialize batch processor
    processor = BatchProcessor(
        output_dir=output_dir,
        max_workers=args.workers,
        progress_type=progress_type
    )
    
    # Add configurations
    for config_name, config in configs:
        processor.add_config(
            name=config_name,
            config=config,
            files=audio_files
        )
    
    # Process files
    try:
        start_time = time.time()
        
        # Show batch processing info
        console.print(Panel.fit(
            f"Batch processing audio files\n\n"
            f"Input directory: [blue]{input_dir}[/blue]\n"
            f"Output directory: [blue]{output_dir}[/blue]\n"
            f"Files to process: [blue]{len(audio_files)}[/blue]\n"
            f"Configurations: [blue]{len(configs)}[/blue]\n"
            f"Total tasks: [blue]{len(audio_files) * len(configs)}[/blue]",
            title="Heihachi Batch Processing",
            border_style="green"
        ))
        
        # Run processing
        results = processor.process(
            parallel=args.parallel,
            export_format=args.export[0],  # Use first format for initial export
            summary_file="batch_summary.json"
        )
        
        processing_time = time.time() - start_time
        
        # Export in additional formats if needed
        if len(args.export) > 1:
            with cli_progress_bar(total=len(results["results"]), desc="Exporting results", unit="file") as pbar:
                for result in results["results"]:
                    if result["status"] == "success" and result["results"]:
                        base_path = Path(output_dir) / f"{Path(result['file_path']).stem}_{result['config_name']}"
                        for export_format in args.export[1:]:  # Skip first format, already exported
                            export_path = f"{base_path}.{export_format}"
                            export_results(result["results"], export_path, export_format)
                    pbar.update(1)
        
        # Display summary
        display_batch_summary(results, processing_time)
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing interrupted by user.[/yellow]")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        console.print(f"[red]Error in batch processing: {e}[/red]")
        return 1


def process_batch_specification(args: argparse.Namespace) -> int:
    """Process audio files according to a batch specification file.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    batch_file = Path(args.batch_file)
    if not batch_file.exists():
        console.print(f"[red]Error: Batch file not found: {batch_file}[/red]")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    ensure_directory(output_dir)
    
    # Determine progress type
    progress_type = args.progress
    if args.quiet:
        progress_type = "none"
    
    try:
        # Load and validate batch file
        with open(batch_file, 'r') as f:
            batch_spec = json.load(f)
        
        if not isinstance(batch_spec, dict):
            console.print(f"[red]Error: Invalid batch file format. Expected JSON object.[/red]")
            return 1
        
        # Show batch file info
        console.print(Panel.fit(
            f"Processing batch file\n\n"
            f"Batch file: [blue]{batch_file}[/blue]\n"
            f"Output directory: [blue]{output_dir}[/blue]",
            title="Heihachi Batch Processing",
            border_style="green"
        ))
        
        # Process batch file
        start_time = time.time()
        results = process_batch_file(
            batch_file=str(batch_file),
            output_dir=str(output_dir),
            parallel=args.parallel,
            export_format=args.export[0],  # Use first format for initial export
            max_workers=args.workers,
            progress_type=progress_type
        )
        
        processing_time = time.time() - start_time
        
        # Export in additional formats if needed
        if len(args.export) > 1:
            with cli_progress_bar(total=len(results["results"]), desc="Exporting results", unit="file") as pbar:
                for result in results["results"]:
                    if result["status"] == "success" and result["results"]:
                        base_path = Path(output_dir) / f"{Path(result['file_path']).stem}_{result['config_name']}"
                        for export_format in args.export[1:]:  # Skip first format, already exported
                            export_path = f"{base_path}.{export_format}"
                            export_results(result["results"], export_path, export_format)
                    pbar.update(1)
        
        # Display summary
        display_batch_summary(results, processing_time)
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing interrupted by user.[/yellow]")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        console.print(f"[red]Error in batch processing: {e}[/red]")
        return 1


def display_batch_summary(results: Dict[str, Any], processing_time: float) -> None:
    """Display a summary of batch processing results.
    
    Args:
        results: Batch processing results
        processing_time: Total processing time in seconds
    """
    # Create summary table
    table = Table(title="Batch Processing Summary")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percentage", style="blue")
    
    # Count results by status
    total = len(results["results"])
    status_counts = {}
    
    for result in results["results"]:
        status = result["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Add rows to table
    for status, count in status_counts.items():
        percentage = (count / total) * 100 if total > 0 else 0
        status_color = "green" if status == "success" else "red"
        table.add_row(
            f"[{status_color}]{status}[/{status_color}]",
            str(count),
            f"{percentage:.1f}%"
        )
    
    # Add total row
    table.add_row("Total", str(total), "100.0%", style="bold")
    
    # Print summary
    console.print("\n")
    console.print(table)
    console.print(f"\nTotal processing time: {processing_time:.2f} seconds")
    
    # Print result file location
    console.print(f"\nDetailed results saved to: [blue]{results['summary_file']}[/blue]")
    
    # Suggest next steps
    console.print("\n[green]Batch processing complete![/green]")
    console.print("\nNext steps:")
    console.print("  - Use the interactive explorer to examine results:")
    console.print(f"    [blue]python -m src.cli.interactive_cli --results-dir {results.get('output_dir', 'results')}[/blue]")
    console.print("  - Use the web UI to visualize results:")
    console.print(f"    [blue]python -m src.cli.interactive_cli --web --results-dir {results.get('output_dir', 'results')}[/blue]")


def main() -> int:
    """Main entry point for the batch CLI.
    
    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=args.log_level)
    
    # Process based on input source
    if args.input_dir:
        if not args.config:
            console.print("[red]Error: --config is required when using --input-dir[/red]")
            return 1
        return process_directory(args)
    elif args.batch_file:
        return process_batch_specification(args)
    else:
        # This should not happen due to required args
        console.print("[red]Error: Either --input-dir or --batch-file is required[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 