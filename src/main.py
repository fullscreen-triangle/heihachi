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
from src.cli import huggingface_commands
from src.semantic import SemanticSearch, EmotionalFeatureMapper

def setup_semantic_commands(subparsers):
    """Setup semantic analysis commands."""
    semantic_parser = subparsers.add_parser(
        'semantic',
        help='Semantic analysis commands',
        description='Semantic analysis and search functionality'
    )
    
    semantic_subparsers = semantic_parser.add_subparsers(dest='semantic_command')
    
    # Index command
    index_parser = semantic_subparsers.add_parser(
        'index',
        help='Index audio files for semantic search'
    )
    index_parser.add_argument('input', help='Audio file or directory to index')
    index_parser.add_argument('--title', help='Track title (for single files)')
    index_parser.add_argument('--artist', help='Artist name (for single files)')
    index_parser.set_defaults(func=semantic_index_command)
    
    # Search command
    search_parser = semantic_subparsers.add_parser(
        'search',
        help='Search indexed tracks semantically'
    )
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--no-enhance', action='store_true', help='Disable query enhancement')
    search_parser.set_defaults(func=semantic_search_command)
    
    # Emotions command
    emotions_parser = semantic_subparsers.add_parser(
        'emotions',
        help='Extract emotional features from audio'
    )
    emotions_parser.add_argument('input', help='Audio file to analyze')
    emotions_parser.add_argument('--output', help='Output file for results')
    emotions_parser.set_defaults(func=semantic_emotions_command)
    
    # Stats command
    stats_parser = semantic_subparsers.add_parser(
        'stats',
        help='Show semantic search statistics'
    )
    stats_parser.set_defaults(func=semantic_stats_command)

def semantic_index_command(args):
    """Index audio files for semantic search."""
    try:
        semantic_search = SemanticSearch()
        
        input_path = Path(args.input)
        
        if input_path.is_file():
            # Index single file
            track_info = {
                'title': args.title or input_path.stem,
                'artist': args.artist or 'Unknown Artist',
                'path': str(input_path)
            }
            
            success = semantic_search.analyze_and_index_audio(str(input_path), track_info)
            if success:
                logging.info(f"Successfully indexed: {track_info['title']}")
            else:
                logging.error(f"Failed to index: {track_info['title']}")
                
        elif input_path.is_dir():
            # Index directory
            audio_files = []
            extensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg']
            
            for ext in extensions:
                audio_files.extend(input_path.glob(f"*.{ext}"))
                audio_files.extend(input_path.glob(f"*.{ext.upper()}"))
            
            if not audio_files:
                logging.error(f"No audio files found in {input_path}")
                return
            
            results = semantic_search.batch_analyze_and_index([str(f) for f in audio_files])
            logging.info(f"Indexed {results['success_count']}/{results['total']} files successfully")
            
        else:
            logging.error(f"Input path not found: {input_path}")
            
    except Exception as e:
        logging.error(f"Error in semantic indexing: {str(e)}")

def semantic_search_command(args):
    """Search indexed tracks semantically."""
    try:
        semantic_search = SemanticSearch()
        
        results = semantic_search.search(
            args.query, 
            top_k=args.top_k, 
            enhance_query=not args.no_enhance
        )
        
        print(f"\nSearch results for: '{args.query}'")
        print("=" * 50)
        
        if 'tracks' in results and results['tracks']:
            for i, track in enumerate(results['tracks'], 1):
                print(f"{i}. {track['title']} - {track['artist']}")
                print(f"   Similarity: {track.get('similarity', 0):.3f}")
                if 'emotions' in track:
                    top_emotions = sorted(track['emotions'].items(), key=lambda x: x[1], reverse=True)[:3]
                    emotions_str = ", ".join([f"{e}: {v:.1f}" for e, v in top_emotions])
                    print(f"   Emotions: {emotions_str}")
                print()
        else:
            print("No results found.")
            
    except Exception as e:
        logging.error(f"Error in semantic search: {str(e)}")

def semantic_emotions_command(args):
    """Extract emotional features from audio."""
    try:
        from src.core.pipeline import Pipeline
        
        # Process audio
        pipeline = Pipeline()
        analysis_result = pipeline.process_file(args.input)
        
        # Extract emotions
        emotion_mapper = EmotionalFeatureMapper()
        emotions = emotion_mapper.map_features_to_emotions(analysis_result)
        
        print(f"\nEmotional analysis for: {args.input}")
        print("=" * 50)
        
        # Sort emotions by value
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, value in sorted_emotions:
            bar_length = int(value / 10 * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{emotion:12} {value:5.1f} |{bar}|")
        
        # Show dominant emotion
        dominant = max(emotions.items(), key=lambda x: x[1])
        print(f"\nDominant emotion: {dominant[0]} ({dominant[1]:.1f}/10)")
        
        # Save to file if requested
        if args.output:
            import json
            output_data = {
                'file': args.input,
                'emotions': emotions,
                'dominant_emotion': {'name': dominant[0], 'value': dominant[1]}
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
    except Exception as e:
        logging.error(f"Error in emotional analysis: {str(e)}")

def semantic_stats_command(args):
    """Show semantic search statistics."""
    try:
        semantic_search = SemanticSearch()
        stats = semantic_search.get_stats()
        
        print("\nSemantic Search Statistics")
        print("=" * 30)
        print(f"Indexed tracks: {stats.get('total_tracks', 0)}")
        print(f"Storage directory: {stats.get('storage_dir', 'N/A')}")
        
        if 'embedding_model' in stats:
            print(f"Embedding model: {stats['embedding_model']}")
            
    except Exception as e:
        logging.error(f"Error getting semantic stats: {str(e)}")

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
        default='./public/amen_break.wav',
        help='Input audio file or directory (default: ./public/amen_break.wav)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./public/output',
        help='Output directory for results (default: ./public/output)'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='./configs/default.yaml',
        help='Configuration file path (default: ./configs/default.yaml)'
    )
    
    # Demo mode
    parser.add_argument(
        '--demo',
        choices=['amen', 'full-mix'],
        help='Run analysis on a demo file (amen: amen_break.wav, full-mix: QO_HoofbeatsMusic.wav)'
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
        help='Input audio file or directory (default: ./public/amen_break.wav)'
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
    
    # Demo mode for process command
    process_parser.add_argument(
        '--demo',
        choices=['amen', 'full-mix'],
        help='Run analysis on a demo file (amen: amen_break.wav, full-mix: QO_HoofbeatsMusic.wav)'
    )
    
    process_parser.set_defaults(func=process_command)
    
    # Setup compare command
    compare.setup_parser(subparsers)
    
    # Setup interactive command
    interactive.setup_parser(subparsers)
    
    # Setup Hugging Face commands
    huggingface_commands.setup_parser(subparsers)
    
    # Setup semantic commands
    setup_semantic_commands(subparsers)
    
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
    # Try to load environment variables from .env file
    try:
        from src.utils.env_loader import load_dotenv
        load_dotenv()
    except ImportError:
        # Continue without .env support
        pass
    
    args = parse_args()
    
    # Handle demo mode
    if args.demo:
        if args.demo == 'amen':
            args.input = './public/amen_break.wav'
        elif args.demo == 'full-mix':
            args.input = './public/QO_HoofbeatsMusic.wav'
        logging.info(f"Running in demo mode with input: {args.input}")
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Handle subcommands
    if args.command and hasattr(args, 'func'):
        # Handle demo mode for subcommands
        if hasattr(args, 'demo') and args.demo and not args.input:
            if args.demo == 'amen':
                args.input = './public/amen_break.wav'
            elif args.demo == 'full-mix':
                args.input = './public/QO_HoofbeatsMusic.wav'
            logging.info(f"Running command in demo mode with input: {args.input}")
        
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