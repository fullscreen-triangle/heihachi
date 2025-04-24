#!/usr/bin/env python3
"""
Heihachi - Neural Processing of Electronic Music

Command-line interface entry point for running the audio analysis pipeline.
"""

import os
import sys
import argparse
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.pipeline import Pipeline
from src.core.batch_processing import BatchProcessor, resume_processing
from src.utils.logging_utils import setup_logging, start_memory_monitoring, MemoryMonitor
from src.utils.profiling import global_profiler, Profiler
from src.utils.cache import result_cache, start_cleanup_thread
from src.commands import compare, interactive

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Heihachi Audio Analysis Framework",
        epilog="For more information, visit the project documentation."
    )
    
    # Main arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input audio file or directory'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./configs/default.yaml',
        help='Configuration file path (default: ./configs/default.yaml)'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command')
    
    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Process audio files',
        description='Process audio files with the Heihachi framework'
    )
    
    process_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input audio file or directory'
    )
    
    process_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for results (default: ./output)'
    )
    
    process_parser.add_argument(
        '--config', '-c',
        type=str,
        default='./configs/default.yaml',
        help='Configuration file path (default: ./configs/default.yaml)'
    )
    
    process_parser.add_argument(
        '--performance-config', '-p',
        type=str,
        default='./configs/performance.yaml',
        help='Performance configuration file path (default: ./configs/performance.yaml)'
    )
    
    process_parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous run'
    )
    
    process_parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
        help='File extensions to process (default: wav mp3 flac m4a ogg)'
    )
    
    process_parser.set_defaults(func=process_command)
    
    # Setup compare command
    compare.setup_parser(subparsers)
    
    # Setup interactive command
    interactive.setup_parser(subparsers)
    
    # Debug options
    debug_group = parser.add_argument_group('Debug options')
    debug_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    debug_group.add_argument(
        '--profile',
        action='store_true',
        help='Enable performance profiling'
    )
    
    debug_group.add_argument(
        '--monitor-memory',
        action='store_true',
        help='Monitor memory usage'
    )
    
    return parser.parse_args()

def process_command(args: argparse.Namespace) -> None:
    """Process audio files command handler."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize profiler if enabled
    profiler = None
    if args.profile:
        profiler = Profiler(output_dir=args.output_dir)
        profiler.start()
    
    # Initialize memory monitor if enabled
    memory_monitor = None
    if args.monitor_memory:
        memory_monitor = MemoryMonitor(
            log_interval=10,  # seconds
            output_dir=args.output_dir
        )
        memory_monitor.start()
    
    try:
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Process single file
            logging.info(f"Processing single file: {input_path}")
            
            pipeline = Pipeline(config_path=args.config)
            result = pipeline.process_file(str(input_path))
            
            # Save result
            output_file = os.path.join(
                args.output_dir, 
                f"{input_path.stem}_result.json"
            )
            with open(output_file, 'w') as f:
                import json
                json.dump(result, f, indent=2)
                
            logging.info(f"Results saved to: {output_file}")
            
        elif input_path.is_dir():
            # Process directory
            logging.info(f"Processing directory: {input_path}")
            
            # Initialize batch processor
            batch_processor = BatchProcessor(
                config_path=args.config,
                performance_config=args.performance_config
            )
            
            if args.resume:
                # Resume from previous run
                logging.info("Resuming from previous run")
                results = batch_processor.resume_batch_processing(
                    str(input_path),
                    args.output_dir,
                    file_extensions=args.extensions
                )
            else:
                # Start new processing
                results = batch_processor.process_directory(
                    str(input_path),
                    args.output_dir,
                    file_extensions=args.extensions
                )
            
            # Calculate success/failure
            success_count = sum(1 for r in results.values() if r.get('status') == 'success')
            failure_count = sum(1 for r in results.values() if r.get('status') == 'error')
            
            logging.info(f"Batch processing completed: {success_count} succeeded, {failure_count} failed")
            
        else:
            logging.error(f"Input path does not exist: {input_path}")
            return
        
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        if args.debug:
            traceback.print_exc()
    
    finally:
        # Stop profiling and monitoring
        if profiler:
            profiler.stop()
            logging.info(f"Profiling results saved to: {profiler.output_file}")
        
        if memory_monitor:
            memory_monitor.stop()
            logging.info(f"Memory monitoring stopped")

def main():
    """Main entry point for the application."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Handle subcommands
    if args.command and hasattr(args, 'func'):
        args.func(args)
    else:
        # For backward compatibility, default to process mode
        if args.input:
            process_args = argparse.Namespace(
                input=args.input,
                output_dir=args.output_dir,
                config=args.config,
                performance_config=getattr(args, 'performance_config', args.config),
                resume=False,
                extensions=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
                debug=args.debug,
                profile=getattr(args, 'profile', False),
                monitor_memory=getattr(args, 'monitor_memory', False)
            )
            process_command(process_args)
        else:
            # No input specified, show help
            parse_args()
            print("\nError: No input file or command specified.")
            print("Use --input to specify an input file/directory or choose a command.")
            print("Run with --help for more information.")

if __name__ == "__main__":
    main()