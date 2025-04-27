#!/usr/bin/env python3
"""
Command-line interface for the Heihachi audio analysis framework.
Provides commands for processing, analyzing, and visualizing audio files.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

from ..core.pipeline import Pipeline
from ..core.batch_processing import BatchProcessor
from ..configs import load_config
from ..utils.profiling import Profiler
from ..utils.logging_utils import setup_logging, MemoryMonitor
from ..utils.visualization_optimization import VisualizationOptimizer


def show_examples():
    """Display usage examples for the Heihachi CLI."""
    examples = """
Examples:
    # Process a single audio file
    heihachi process audio.wav --output results/
    
    # Process a directory of audio files
    heihachi process audio_dir/ --output results/
    
    # Batch processing with different configurations
    heihachi batch audio_dir/ --config configs/performance.yaml
    
    # Interactive mode with processed results
    heihachi interactive results/
    
    # Compare multiple results
    heihachi compare results1/ results2/
    
    # Export results to different formats
    heihachi export results/ --format json
    
    # Process with GPU acceleration and 4 workers
    heihachi process audio.wav --gpu --workers 4
    """
    print(examples)


def setup_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser.
    
    Args:
        parser: The argparse parser to add arguments to
    """
    parser.add_argument("--verbose", "-v", action="count", default=0,
                        help="Increase verbosity level (use -v, -vv, or -vvv)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress all non-error output")
    parser.add_argument("--config", type=str,
                        help="Path to configuration file")
    parser.add_argument("--profile", action="store_true",
                        help="Enable performance profiling")
    parser.add_argument("--monitor-memory", action="store_true",
                        help="Monitor memory usage during processing")
    parser.add_argument("--log-file", type=str,
                        help="Path to log file")
    parser.add_argument("--workers", type=int,
                        help="Number of worker processes")
    parser.add_argument("--memory-limit", type=int,
                        help="Memory limit in MB")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU acceleration")


def create_process_parser(subparsers) -> None:
    """Create the 'process' command parser.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    process_parser = subparsers.add_parser(
        "process", 
        help="Process audio files",
        description="Process single audio files or directories with configurable options."
    )
    
    process_parser.add_argument(
        "input",
        type=str,
        help="Input audio file or directory path"
    )
    
    process_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for results"
    )
    
    process_parser.add_argument(
        "--format",
        choices=["json", "csv", "yaml", "all"],
        default="json",
        help="Output format for results (default: json)"
    )
    
    process_parser.add_argument(
        "--extensions",
        type=str,
        default="wav,mp3,flac,ogg,m4a",
        help="Comma-separated list of file extensions to process (default: wav,mp3,flac,ogg,m4a)"
    )
    
    process_parser.add_argument(
        "--pipeline",
        type=str,
        help="Path to pipeline configuration file"
    )
    
    process_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of results"
    )
    
    process_parser.add_argument(
        "--batch-size",
        type=int,
        help="Number of files to process in each batch"
    )
    
    process_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have results"
    )
    
    # HuggingFace integration options
    process_parser.add_argument(
        "--hf-models",
        action="store_true",
        help="Enable HuggingFace models for audio analysis"
    )
    
    process_parser.add_argument(
        "--hf-api-key",
        help="HuggingFace API key for accessing models"
    )
    
    process_parser.add_argument(
        "--hf-model-name",
        help="HuggingFace model name (default: facebook/wav2vec2-base-960h)"
    )
    
    process_parser.add_argument(
        "--hf-genre-model",
        help="Specific model for genre classification"
    )
    
    process_parser.add_argument(
        "--hf-instrument-model",
        help="Specific model for instrument detection"
    )
    
    process_parser.add_argument(
        "--no-hf-genre",
        action="store_true",
        help="Disable genre classification"
    )
    
    process_parser.add_argument(
        "--no-hf-instruments",
        action="store_true",
        help="Disable instrument detection"
    )
    
    setup_common_args(process_parser)


def create_batch_parser(subparsers) -> None:
    """Create the 'batch' command parser.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    batch_parser = subparsers.add_parser(
        "batch", 
        help="Process multiple files with different configurations",
        description="Batch processing with different configurations for each file or file groups."
    )
    
    batch_parser.add_argument(
        "input",
        type=str,
        help="Input directory containing audio files"
    )
    
    batch_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for results"
    )
    
    batch_parser.add_argument(
        "--config-dir",
        type=str,
        help="Directory containing configuration files"
    )
    
    batch_parser.add_argument(
        "--config",
        type=str,
        help="Path to batch configuration file"
    )
    
    batch_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted batch processing"
    )
    
    batch_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations of results"
    )
    
    batch_parser.add_argument(
        "--extensions",
        type=str,
        default="wav,mp3,flac,ogg,m4a",
        help="Comma-separated list of file extensions to process (default: wav,mp3,flac,ogg,m4a)"
    )
    
    setup_common_args(batch_parser)


def create_interactive_parser(subparsers) -> None:
    """Create the 'interactive' command parser.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    interactive_parser = subparsers.add_parser(
        "interactive", 
        help="Start interactive mode for exploring results",
        description="Interactive mode for exploring and visualizing analysis results."
    )
    
    interactive_parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing analysis results"
    )
    
    interactive_parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the interactive web server (default: 8050)"
    )
    
    interactive_parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host for the interactive web server (default: localhost)"
    )
    
    interactive_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open a browser window automatically"
    )


def create_export_parser(subparsers) -> None:
    """Create the 'export' command parser.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    export_parser = subparsers.add_parser(
        "export", 
        help="Export analysis results to different formats",
        description="Export analysis results to various formats like JSON, CSV, or Markdown."
    )
    
    export_parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing analysis results"
    )
    
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for exported files"
    )
    
    export_parser.add_argument(
        "--format",
        choices=["json", "csv", "yaml", "markdown", "html", "all"],
        default="all",
        help="Format for exported results (default: all)"
    )
    
    export_parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for exported filenames"
    )


def create_compare_parser(subparsers) -> None:
    """Create the 'compare' command parser.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    compare_parser = subparsers.add_parser(
        "compare", 
        help="Compare multiple analysis results",
        description="Compare analysis results from multiple files or directories."
    )
    
    compare_parser.add_argument(
        "results",
        type=str,
        nargs="+",
        help="Paths to result files or directories to compare"
    )
    
    compare_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for comparison results"
    )
    
    compare_parser.add_argument(
        "--format",
        choices=["json", "csv", "yaml", "html", "all"],
        default="html",
        help="Format for comparison results (default: html)"
    )
    
    compare_parser.add_argument(
        "--metrics",
        type=str,
        help="Comma-separated list of metrics to compare"
    )
    
    compare_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive mode after comparison"
    )


def create_visualize_parser(subparsers) -> None:
    """Create the 'visualize' command parser.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    visualize_parser = subparsers.add_parser(
        "visualize", 
        help="Generate visualizations from analysis results",
        description="Create spectrograms, waveforms, and other visualizations from analysis results."
    )
    
    visualize_parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing analysis results"
    )
    
    visualize_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output directory for visualizations"
    )
    
    visualize_parser.add_argument(
        "--type",
        choices=["spectrogram", "waveform", "features", "all"],
        default="all",
        help="Type of visualization to generate (default: all)"
    )
    
    visualize_parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output images (default: 150)"
    )
    
    visualize_parser.add_argument(
        "--format",
        choices=["png", "svg", "pdf", "jpg"],
        default="png",
        help="Image format for visualizations (default: png)"
    )
    
    visualize_parser.add_argument(
        "--max-points",
        type=int,
        default=10000,
        help="Maximum number of points to plot (default: 10000)"
    )


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Heihachi: Advanced Audio Analysis Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run 'heihachi <command> --help' for more information on a command."
    )
    
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit"
    )
    
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples and exit"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create parsers for each command
    create_process_parser(subparsers)
    create_batch_parser(subparsers)
    create_interactive_parser(subparsers)
    create_export_parser(subparsers)
    create_compare_parser(subparsers)
    create_visualize_parser(subparsers)
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Show examples if requested
    if parsed_args.examples:
        show_examples()
        sys.exit(0)
    
    # Show version if requested
    if parsed_args.version:
        from .. import __version__
        print(f"Heihachi Audio Analysis Framework v{__version__}")
        sys.exit(0)
    
    # Require a command
    if parsed_args.command is None:
        parser.print_help()
        sys.exit(1)
    
    return parsed_args


def get_log_level(verbosity: int) -> int:
    """Convert verbosity count to logging level.
    
    Args:
        verbosity: Verbosity count (0-3)
    
    Returns:
        Logging level
    """
    if verbosity == 0:
        return logging.WARNING
    elif verbosity == 1:
        return logging.INFO
    else:
        return logging.DEBUG


def process_command(args: argparse.Namespace) -> int:
    """Process audio files based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Determine input/output paths
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    # Create output directory if needed
    if output_path and not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    config = {}
    if args.config:
        config = load_config(args.config)
    
    # Configure pipeline
    pipeline_config = {}
    if args.pipeline:
        pipeline_config = load_config(args.pipeline)
    
    # Apply HuggingFace settings if provided
    if args.hf_models or args.hf_api_key or args.hf_model_name:
        if 'huggingface' not in config:
            config['huggingface'] = {}
        
        # Enable HuggingFace
        config['huggingface']['enable'] = True
        
        # Set API key if provided
        if args.hf_api_key:
            config['huggingface']['api_key'] = args.hf_api_key
            
        # Set model name if provided
        if args.hf_model_name:
            config['huggingface']['model_name'] = args.hf_model_name
            
        # Set genre classification settings
        if args.hf_genre_model:
            config['huggingface']['genre_model'] = args.hf_genre_model
            
        # Set instrument detection settings
        if args.hf_instrument_model:
            config['huggingface']['instrument_model'] = args.hf_instrument_model
            
        # Disable features if requested
        if args.no_hf_genre:
            config['huggingface']['genre_classification'] = False
            
        if args.no_hf_instruments:
            config['huggingface']['instrument_detection'] = False
    
    # Set up pipeline
    pipeline = Pipeline(config=config)
    
    # Process single file or directory
    try:
        if input_path.is_file():
            result = pipeline.process_file(str(input_path))
            
            if output_path:
                result_file = output_path / f"{input_path.stem}_result.json"
                with open(result_file, 'w') as f:
                    import json
                    json.dump(result, f, indent=2)
                
                logging.info(f"Results saved to {result_file}")
            
            if args.visualize:
                visualizer = VisualizationOptimizer(
                    output_dir=str(output_path) if output_path else None,
                    dpi=150,
                    max_points=10000
                )
                visualizer.generate_visualizations(result, input_path.stem)
                
                logging.info(f"Visualizations generated in {output_path}")
            
            return 0
            
        elif input_path.is_dir():
            extensions = args.extensions.split(',')
            
            batch_processor = BatchProcessor(
                config_path=args.config,
            )
            
            results = batch_processor.process_directory(
                directory=str(input_path),
                file_extensions=extensions,
                output_dir=str(output_path) if output_path else None,
                pipeline_config=pipeline_config
            )
            
            logging.info(f"Processed {len(results)} files")
            
            if args.visualize:
                visualizer = VisualizationOptimizer(
                    output_dir=str(output_path) if output_path else None,
                    dpi=150,
                    max_points=10000
                )
                visualizer.generate_batch_visualizations(results)
                
                logging.info(f"Visualizations generated in {output_path}")
            
            return 0
        else:
            logging.error(f"Input path {input_path} does not exist or is not a file/directory")
            return 1
            
    except Exception as e:
        logging.error(f"Error processing {input_path}: {e}")
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


def batch_command(args: argparse.Namespace) -> int:
    """Run batch processing based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Create output directory if needed
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(
        config_path=args.config,
    )
    
    # Process the directory
    try:
        extensions = args.extensions.split(',')
        
        results = batch_processor.process_directory(
            directory=str(input_path),
            file_extensions=extensions,
            output_dir=str(output_path),
            resume=args.resume
        )
        
        logging.info(f"Processed {len(results)} files")
        
        if args.visualize:
            visualizer = VisualizationOptimizer(
                output_dir=str(output_path),
                dpi=150,
                max_points=10000
            )
            visualizer.generate_batch_visualizations(results)
            
            logging.info(f"Visualizations generated in {output_path}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error during batch processing: {e}")
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


def interactive_command(args: argparse.Namespace) -> int:
    """Start interactive mode based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        from ..interactive.app import start_interactive_app
        
        return start_interactive_app(
            results_dir=args.results_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser
        )
    except ImportError:
        logging.error("Interactive mode requires additional dependencies. Install with 'pip install heihachi[interactive]'")
        return 1
    except Exception as e:
        logging.error(f"Error starting interactive mode: {e}")
        import traceback
        traceback.print_exc()
        return 1


def export_command(args: argparse.Namespace) -> int:
    """Export results based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    from ..utils.exporters import export_results
    
    try:
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output) if args.output else results_dir / "exports"
        
        # Create output directory if needed
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine formats to export
        formats = [args.format]
        if args.format == "all":
            formats = ["json", "csv", "yaml", "markdown", "html"]
        
        # Export results
        exported_files = export_results(
            results_dir=str(results_dir),
            output_dir=str(output_dir),
            formats=formats,
            prefix=args.prefix
        )
        
        for fmt, files in exported_files.items():
            logging.info(f"Exported {len(files)} files in {fmt} format")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error exporting results: {e}")
        import traceback
        traceback.print_exc()
        return 1


def compare_command(args: argparse.Namespace) -> int:
    """Compare results based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    from ..utils.comparison import compare_results
    
    try:
        results_paths = [Path(path) for path in args.results]
        output_path = Path(args.output) if args.output else Path("comparison_results")
        
        # Create output directory if needed
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine metrics to compare
        metrics = args.metrics.split(',') if args.metrics else None
        
        # Compare results
        comparison = compare_results(
            results_paths=[str(path) for path in results_paths],
            output_dir=str(output_path),
            format=args.format,
            metrics=metrics
        )
        
        logging.info(f"Comparison results saved to {output_path}")
        
        # Start interactive mode if requested
        if args.interactive:
            from ..interactive.app import start_interactive_comparison
            
            return start_interactive_comparison(
                comparison_results=comparison,
                host="localhost",
                port=8050,
                open_browser=True
            )
        
        return 0
        
    except Exception as e:
        logging.error(f"Error comparing results: {e}")
        import traceback
        traceback.print_exc()
        return 1


def visualize_command(args: argparse.Namespace) -> int:
    """Generate visualizations based on command-line arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        results_dir = Path(args.results_dir)
        output_dir = Path(args.output) if args.output else results_dir / "visualizations"
        
        # Create output directory if needed
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualizer
        visualizer = VisualizationOptimizer(
            output_dir=str(output_dir),
            dpi=args.dpi,
            max_points=args.max_points
        )
        
        # Load results
        import json
        results = []
        
        if results_dir.is_file() and results_dir.suffix == '.json':
            with open(results_dir, 'r') as f:
                data = json.load(f)
                results.append((results_dir.stem, data))
        else:
            for result_file in results_dir.glob('*.json'):
                with open(result_file, 'r') as f:
                    try:
                        data = json.load(f)
                        results.append((result_file.stem, data))
                    except json.JSONDecodeError:
                        logging.warning(f"Could not parse {result_file} as JSON")
        
        # Generate visualizations
        visualization_types = [args.type]
        if args.type == "all":
            visualization_types = ["spectrogram", "waveform", "features"]
        
        for name, data in results:
            for viz_type in visualization_types:
                if viz_type == "spectrogram":
                    visualizer.plot_spectrogram(data, name, fmt=args.format)
                elif viz_type == "waveform":
                    visualizer.plot_waveform(data, name, fmt=args.format)
                elif viz_type == "features":
                    visualizer.plot_features(data, name, fmt=args.format)
        
        logging.info(f"Generated visualizations in {output_dir}")
        
        return 0
        
    except Exception as e:
        logging.error(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command-line arguments (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    # Parse arguments
    parsed_args = parse_args(args)
    
    # Set up logging
    log_level = get_log_level(parsed_args.verbose)
    if parsed_args.quiet:
        log_level = logging.ERROR
    
    setup_logging(level=log_level, log_file=parsed_args.log_file)
    
    # Start memory monitoring if requested
    memory_monitor = None
    if hasattr(parsed_args, 'monitor_memory') and parsed_args.monitor_memory:
        memory_monitor = MemoryMonitor()
        memory_monitor.start()
    
    # Start profiling if requested
    profiler = None
    if hasattr(parsed_args, 'profile') and parsed_args.profile:
        profiler = Profiler()
        profiler.start()
    
    # Dispatch command
    try:
        if parsed_args.command == "process":
            exit_code = process_command(parsed_args)
        elif parsed_args.command == "batch":
            exit_code = batch_command(parsed_args)
        elif parsed_args.command == "interactive":
            exit_code = interactive_command(parsed_args)
        elif parsed_args.command == "export":
            exit_code = export_command(parsed_args)
        elif parsed_args.command == "compare":
            exit_code = compare_command(parsed_args)
        elif parsed_args.command == "visualize":
            exit_code = visualize_command(parsed_args)
        else:
            logging.error(f"Unknown command: {parsed_args.command}")
            exit_code = 1
    except Exception as e:
        logging.error(f"Error executing command: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    # Stop profiling if started
    if profiler:
        profiler.stop()
        profile_output = f"profile_{parsed_args.command}.prof"
        profiler.save(profile_output)
        logging.info(f"Profile saved to {profile_output}")
    
    # Stop memory monitoring if started
    if memory_monitor:
        memory_monitor.stop()
        memory_output = f"memory_{parsed_args.command}.json"
        memory_monitor.save_stats(memory_output)
        logging.info(f"Memory statistics saved to {memory_output}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 