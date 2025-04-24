#!/usr/bin/env python3
"""
Enhanced CLI for Heihachi - Neural Processing of Electronic Music

This module provides an improved command-line interface for the Heihachi
audio analysis framework with better help messages, examples, and
error handling.
"""

import os
import sys
import argparse
import logging
import json
import time
import shutil
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import readline
import glob

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.pipeline import Pipeline
from src.core.batch_processing import BatchProcessor, resume_processing
from src.utils.logging_utils import setup_logging, get_logger
from src.utils.profiling import global_profiler
from src.utils.cache import result_cache
from src.utils.export import export_results
from src.utils.result_viewer import ResultViewer
from src.utils.error_handler import ErrorHandler, handle_error, suggest_solution
from src.utils.progress import ProgressTracker, track_progress, update_progress, complete_progress

logger = get_logger(__name__)

# Terminal colors for pretty output
class TermColors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class HeihachiArgumentParser(argparse.ArgumentParser):
    """Enhanced argument parser with better formatting and examples."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with improved help formatting."""
        if 'formatter_class' not in kwargs:
            kwargs['formatter_class'] = lambda prog: argparse.RawDescriptionHelpFormatter(
                prog, max_help_position=30, width=100)
        super().__init__(*args, **kwargs)
    
    def error(self, message):
        """Override error method to provide more helpful error messages."""
        self.print_usage(sys.stderr)
        error_msg = f'{TermColors.RED}{TermColors.BOLD}Error:{TermColors.ENDC} {message}\n'
        suggestion = suggest_solution(message)
        if suggestion:
            error_msg += f"\n{TermColors.YELLOW}Suggestion:{TermColors.ENDC} {suggestion}\n"
        error_msg += f"\nTry '{TermColors.BOLD}{self.prog} --help{TermColors.ENDC}' for more information."
        self.exit(2, error_msg)
    
    def add_examples(self):
        """Add usage examples to the parser."""
        examples = f"""
{TermColors.BOLD}Examples:{TermColors.ENDC}
  # Process a single audio file with default settings
  {TermColors.GREEN}heihachi track.mp3{TermColors.ENDC}
  
  # Process a file with custom configuration and output directory
  {TermColors.GREEN}heihachi track.mp3 -c custom_config.yaml -o results/{TermColors.ENDC}
  
  # Process all audio files in a directory
  {TermColors.GREEN}heihachi audio_folder/ --batch{TermColors.ENDC}
  
  # Resume interrupted batch processing
  {TermColors.GREEN}heihachi audio_folder/ --resume{TermColors.ENDC}
  
  # Process with performance profiling enabled
  {TermColors.GREEN}heihachi track.mp3 --profile{TermColors.ENDC}
  
  # Export results in different formats
  {TermColors.GREEN}heihachi track.mp3 --export csv,json{TermColors.ENDC}
  
  # Use interactive mode to explore results
  {TermColors.GREEN}heihachi track.mp3 --interactive{TermColors.ENDC}
  
  # Process files with different configurations
  {TermColors.GREEN}heihachi audio_folder/ --batch --configs configs/dance.yaml,configs/ambient.yaml{TermColors.ENDC}
  
  # Enable progress bar display
  {TermColors.GREEN}heihachi track.mp3 --progress-bar{TermColors.ENDC}
  
  # Compare multiple audio files
  {TermColors.GREEN}heihachi --compare track1.mp3,track2.mp3,track3.mp3{TermColors.ENDC}
  
  # Use a specific configuration profile
  {TermColors.GREEN}heihachi track.mp3 --profile-name "dance-music"{TermColors.ENDC}
        """
        self.epilog = examples


def create_parser() -> HeihachiArgumentParser:
    """Create enhanced argument parser with improved help and examples."""
    description = f"""
{TermColors.BOLD}Heihachi Audio Analysis Framework{TermColors.ENDC}
--------------------------------
A comprehensive tool for analyzing audio files with a focus on electronic music.
Extracts features, segments tracks, and provides detailed analysis.
    """
    
    parser = HeihachiArgumentParser(
        description=textwrap.dedent(description),
        prog='heihachi'
    )
    
    # Main arguments with improved help messages
    parser.add_argument(
        "input",
        help="Input audio file or directory of audio files to process"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output directory for results (default: './results/')"
    )
    
    parser.add_argument(
        "-c", "--config",
        default="configs/performance.yaml",
        help="Configuration file path (default: configs/performance.yaml)"
    )
    
    # Operation modes with improved descriptions
    operation_group = parser.add_argument_group(f'{TermColors.BOLD}Operation Modes{TermColors.ENDC}')
    
    operation_group.add_argument(
        "--batch",
        action="store_true",
        help="Process all audio files in the input directory"
    )
    
    operation_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted batch processing from the last saved state"
    )
    
    operation_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling and generate a detailed report"
    )
    
    operation_group.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive mode after processing to explore results"
    )
    
    # Export options
    export_group = parser.add_argument_group(f'{TermColors.BOLD}Export Options{TermColors.ENDC}')
    
    export_group.add_argument(
        "--export",
        help="Export results in specified formats (comma-separated: json,csv,yaml,md)"
    )
    
    export_group.add_argument(
        "--export-dir",
        help="Directory for exported results (default: <output>/exports/)"
    )
    
    # Batch configuration
    batch_group = parser.add_argument_group(f'{TermColors.BOLD}Batch Processing{TermColors.ENDC}')
    
    batch_group.add_argument(
        "--configs",
        help="Comma-separated list of config files to use for batch processing"
    )
    
    batch_group.add_argument(
        "--file-pattern",
        help="Pattern to match filenames for processing (e.g., '*.wav,*.mp3')"
    )
    
    batch_group.add_argument(
        "--max-files",
        type=int,
        help="Maximum number of files to process in batch mode"
    )
    
    # Performance options with better descriptions
    perf_group = parser.add_argument_group(f'{TermColors.BOLD}Performance Options{TermColors.ENDC}')
    
    perf_group.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes/threads for parallel processing"
    )
    
    perf_group.add_argument(
        "--memory-limit",
        type=int,
        help="Memory usage limit in MB"
    )
    
    perf_group.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU usage if available"
    )
    
    perf_group.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage even if available"
    )

    # User Experience enhancements
    ux_group = parser.add_argument_group(f'{TermColors.BOLD}User Experience{TermColors.ENDC}')
    
    ux_group.add_argument(
        "--progress-bar",
        action="store_true",
        help="Display progress bars for long-running operations"
    )
    
    ux_group.add_argument(
        "--compare",
        help="Compare multiple audio files (comma-separated list of files)"
    )
    
    ux_group.add_argument(
        "--profile-name",
        help="Use a specific named configuration profile"
    )
    
    ux_group.add_argument(
        "--web-ui",
        action="store_true",
        help="Launch web interface after processing to visualize results"
    )
    
    ux_group.add_argument(
        "--completion-install",
        action="store_true",
        help="Install command completion for your shell"
    )
    
    # Add examples to help output
    parser.add_examples()
    
    return parser


def parse_export_formats(export_arg: str) -> List[str]:
    """Parse export formats from command line argument.
    
    Args:
        export_arg: Comma-separated list of export formats
        
    Returns:
        List of valid export formats
    """
    if not export_arg:
        return []
        
    valid_formats = ['json', 'csv', 'yaml', 'md', 'markdown', 'html', 'xml']
    requested_formats = [fmt.strip().lower() for fmt in export_arg.split(',')]
    
    # Validate formats
    invalid_formats = [fmt for fmt in requested_formats if fmt not in valid_formats]
    if invalid_formats:
        logger.warning(f"Invalid export format(s): {', '.join(invalid_formats)}. "
                      f"Valid formats are: {', '.join(valid_formats)}")
        
    # Return only valid formats
    return [fmt for fmt in requested_formats if fmt in valid_formats]


def parse_config_list(configs_arg: str) -> List[str]:
    """Parse list of configuration files.
    
    Args:
        configs_arg: Comma-separated list of config files
        
    Returns:
        List of config file paths
    """
    if not configs_arg:
        return []
        
    return [path.strip() for path in configs_arg.split(',')]


def process_file_with_config(file_path: str, config_path: str, 
                           output_dir: Optional[Path], 
                           export_formats: List[str],
                           export_dir: Optional[Path]) -> dict:
    """Process a single file with the specified configuration.
    
    Args:
        file_path: Path to the audio file
        config_path: Path to the configuration file
        output_dir: Output directory for results
        export_formats: List of export formats
        export_dir: Directory for exported results
        
    Returns:
        Processing results
    """
    try:
        # Initialize pipeline with configuration
        pipeline = Pipeline(config_path)
        
        # Process the file
        logger.info(f"Processing file: {file_path}")
        start_time = time.time()
        result = pipeline.process(file_path)
        processing_time = time.time() - start_time
        
        # Add metadata to result
        result['metadata'] = {
            'file_path': file_path,
            'config_path': config_path,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
        
        # Write result if output directory specified
        if output_dir:
            output_file = output_dir / f"{Path(file_path).stem}_result.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results written to {output_file}")
        
        # Export results if formats specified
        if export_formats and output_dir:
            if export_dir is None:
                export_dir = output_dir / "exports"
            
            export_dir.mkdir(exist_ok=True, parents=True)
            
            for fmt in export_formats:
                try:
                    export_path = export_results(result, fmt, export_dir, Path(file_path).stem)
                    logger.info(f"Exported results in {fmt} format to {export_path}")
                except Exception as e:
                    logger.error(f"Failed to export in {fmt} format: {str(e)}")
        
        logger.info(f"Processing complete in {processing_time:.2f}s")
        return result
    
    except Exception as e:
        # Enhanced error handling
        error_type = type(e).__name__
        error_msg = str(e)
        
        logger.error(f"Error processing {file_path}: {error_type}: {error_msg}")
        
        # Get suggestion for fixing the error
        suggestion = suggest_solution(e)
        if suggestion:
            logger.info(f"Suggestion: {suggestion}")
        
        # Return error information
        return {
            'error': {
                'type': error_type,
                'message': error_msg,
                'suggestion': suggestion
            },
            'status': 'failed',
            'file_path': file_path
        }


def process_batch_with_configs(input_dir: Path, configs: List[str], 
                             output_dir: Optional[Path], 
                             file_pattern: Optional[str],
                             max_files: Optional[int],
                             export_formats: List[str],
                             export_dir: Optional[Path],
                             args: argparse.Namespace) -> dict:
    """Process a batch of files with multiple configurations.
    
    Args:
        input_dir: Input directory containing audio files
        configs: List of configuration files
        output_dir: Output directory for results
        file_pattern: Pattern to match filenames
        max_files: Maximum number of files to process
        export_formats: List of export formats
        export_dir: Directory for exported results
        args: Command line arguments
        
    Returns:
        Batch processing results
    """
    batch_results = {
        'configs': configs,
        'results': [],
        'stats': {
            'total': 0,
            'success': 0,
            'failed': 0,
            'total_time': 0
        }
    }
    
    # Ensure we have at least one config
    if not configs:
        configs = [args.config]
    
    # Create batch processor
    batch_processor = BatchProcessor(args.config)
    
    # Override configuration with command line args
    if args.workers:
        batch_processor.num_workers = args.workers
    if args.memory_limit:
        batch_processor.memory_limit_mb = args.memory_limit
    
    # Process with each configuration
    for config_path in configs:
        logger.info(f"Processing batch with configuration: {config_path}")
        
        config_output_dir = output_dir / Path(config_path).stem if output_dir else None
        if config_output_dir:
            config_output_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Process directory with this configuration
            results = batch_processor.process_directory(
                str(input_dir),
                file_extension=file_pattern.split(',') if file_pattern else None,
                output_dir=str(config_output_dir) if config_output_dir else None,
                pipeline_configs=[{'config_path': config_path}]
            )
            
            # Update stats
            batch_results['stats']['total'] += results['stats']['total']
            batch_results['stats']['success'] += results['stats']['success']
            batch_results['stats']['failed'] += results['stats']['failed']
            batch_results['stats']['total_time'] += results['stats']['total_time']
            
            # Add results
            batch_results['results'].append({
                'config': config_path,
                'results': results
            })
            
            # Export summary if needed
            if export_formats and config_output_dir:
                config_export_dir = export_dir / Path(config_path).stem if export_dir else config_output_dir / "exports"
                config_export_dir.mkdir(exist_ok=True, parents=True)
                
                for fmt in export_formats:
                    try:
                        export_path = export_results(results, fmt, config_export_dir, "batch_summary")
                        logger.info(f"Exported batch summary in {fmt} format to {export_path}")
                    except Exception as e:
                        logger.error(f"Failed to export batch summary in {fmt} format: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error in batch processing with config {config_path}: {str(e)}")
            batch_results['results'].append({
                'config': config_path,
                'error': str(e),
                'status': 'failed'
            })
    
    return batch_results


def start_interactive_mode(results: Dict, result_file: Optional[Path] = None) -> None:
    """Start interactive mode for exploring results.
    
    Args:
        results: Processing results
        result_file: Path to result file for loading additional data
    """
    try:
        from src.utils.result_viewer import ResultViewer
        
        logger.info("Starting interactive result explorer...")
        viewer = ResultViewer(results, result_file)
        viewer.start()
    except ImportError as e:
        logger.error(f"Failed to start interactive mode: {str(e)}")
        logger.error("Interactive mode requires additional dependencies.")
        logger.error("Install them with: pip install heihachi[interactive]")
    except Exception as e:
        logger.error(f"Error in interactive mode: {str(e)}")


def run() -> int:
    """Main CLI entry point with enhanced features.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Parse arguments with enhanced parser
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log_file if args.log_file else None
    setup_logging(log_level=log_level, log_file=log_file)
    
    # Parse export formats
    export_formats = parse_export_formats(args.export) if args.export else []
    
    # Parse config list for batch processing
    configs = parse_config_list(args.configs) if args.configs else []
    
    # Start memory monitoring if requested
    memory_monitor = None
    if args.memory_monitor:
        try:
            from src.utils.logging_utils import start_memory_monitoring
            memory_monitor = start_memory_monitoring()
            logger.info("Memory monitoring started")
        except ImportError:
            logger.warning("Memory monitoring requires 'psutil'. Install with: pip install psutil")
    
    # Handle cache settings
    if args.no_cache:
        result_cache.clear()
        logger.info("Result cache cleared")
    
    # Start profiling if requested
    if args.profile:
        global_profiler.start("main")
        logger.info("Performance profiling started")
    
    try:
        # Setup paths
        input_path = Path(args.input)
        output_dir = Path(args.output) if args.output else Path("./results")
        export_dir = Path(args.export_dir) if args.export_dir else output_dir / "exports"
        
        # Create output directory
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if input exists
        if not input_path.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1
        
        # Process based on input type and mode
        if input_path.is_dir() or args.batch:
            # Directory or batch mode
            if args.resume:
                # Resume interrupted batch processing
                logger.info(f"Resuming batch processing from {input_path}")
                results = resume_processing(
                    str(input_path),
                    output_dir=str(output_dir),
                    config_path=args.config
                )
                
                print(f"Processing complete: {results['stats']['success']} successful, "
                     f"{results['stats']['failed']} failed")
                
                # Export results if requested
                if export_formats:
                    for fmt in export_formats:
                        try:
                            export_path = export_results(results, fmt, export_dir, "resumed_batch_summary")
                            logger.info(f"Exported resumed batch summary in {fmt} format to {export_path}")
                        except Exception as e:
                            logger.error(f"Failed to export resumed batch summary in {fmt} format: {str(e)}")
                
                # Start interactive mode if requested
                if args.interactive and results['stats']['success'] > 0:
                    start_interactive_mode(results)
                
            else:
                # Process with multiple configurations
                if configs:
                    logger.info(f"Processing directory with {len(configs)} different configurations")
                    results = process_batch_with_configs(
                        input_path,
                        configs,
                        output_dir,
                        args.file_pattern,
                        args.max_files,
                        export_formats,
                        export_dir,
                        args
                    )
                    
                    print(f"Multi-config batch processing complete:")
                    print(f"  Total files: {results['stats']['total']}")
                    print(f"  Successful: {results['stats']['success']}")
                    print(f"  Failed: {results['stats']['failed']}")
                    print(f"  Total time: {results['stats']['total_time']:.2f}s")
                    
                    # Start interactive mode if requested
                    if args.interactive and results['stats']['success'] > 0:
                        start_interactive_mode(results)
                    
                else:
                    # Standard batch processing
                    logger.info(f"Processing directory: {input_path}")
                    batch_processor = BatchProcessor(args.config)
                    
                    # Override configuration with command line args
                    if args.workers:
                        batch_processor.num_workers = args.workers
                    if args.memory_limit:
                        batch_processor.memory_limit_mb = args.memory_limit
                    
                    # Process directory
                    results = batch_processor.process_directory(
                        str(input_path),
                        file_extension=args.file_pattern.split(',') if args.file_pattern else None,
                        output_dir=str(output_dir)
                    )
                    
                    print(f"Processing complete: {results['stats']['success']} successful, "
                         f"{results['stats']['failed']} failed, "
                         f"total time: {results['stats']['total_time']:.2f}s")
                    
                    # Export results if requested
                    if export_formats:
                        for fmt in export_formats:
                            try:
                                export_path = export_results(results, fmt, export_dir, "batch_summary")
                                logger.info(f"Exported batch summary in {fmt} format to {export_path}")
                            except Exception as e:
                                logger.error(f"Failed to export batch summary in {fmt} format: {str(e)}")
                    
                    # Start interactive mode if requested
                    if args.interactive and results['stats']['success'] > 0:
                        start_interactive_mode(results)
        
        else:
            # Single file mode
            logger.info(f"Processing file: {input_path}")
            
            # Process with multiple configurations if specified
            if configs:
                logger.info(f"Processing file with {len(configs)} different configurations")
                all_results = []
                
                for config in configs:
                    config_output_dir = output_dir / Path(config).stem
                    config_output_dir.mkdir(exist_ok=True, parents=True)
                    
                    result = process_file_with_config(
                        str(input_path),
                        config,
                        config_output_dir,
                        export_formats,
                        export_dir / Path(config).stem if export_dir else None
                    )
                    
                    all_results.append({
                        'config': config,
                        'result': result
                    })
                
                # Print summary
                print(f"Multi-configuration processing complete")
                for res in all_results:
                    print(f"  Config: {res['config']}")
                    if 'error' in res['result']:
                        print(f"    Status: Failed - {res['result']['error']['type']}")
                    else:
                        print(f"    Status: Success")
                
                # Start interactive mode if requested
                if args.interactive:
                    start_interactive_mode({'multi_config_results': all_results})
            
            else:
                # Single configuration
                result = process_file_with_config(
                    str(input_path),
                    args.config,
                    output_dir,
                    export_formats,
                    export_dir
                )
                
                # Start interactive mode if requested
                if args.interactive and 'error' not in result:
                    result_file = output_dir / f"{input_path.stem}_result.json" if output_dir else None
                    start_interactive_mode(result, result_file)
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        return 1
    
    finally:
        # Stop profiling if active
        if args.profile:
            global_profiler.stop("main")
            profile_file = output_dir / "profile_summary.txt" if 'output_dir' in locals() else None
            global_profiler.generate_summary(str(profile_file) if profile_file else None)
            logger.info(f"Performance profile generated{f' at {profile_file}' if profile_file else ''}")
        
        # Stop memory monitoring if active
        if memory_monitor:
            memory_monitor.stop()
            logger.info("Memory monitoring stopped")


if __name__ == "__main__":
    sys.exit(run()) 