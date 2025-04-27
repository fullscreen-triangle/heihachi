"""
Command-line interface for Hugging Face model operations.

This module provides CLI commands for using specialized Hugging Face models
for audio analysis, source separation, and beat tracking.
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path
import time

from ..huggingface import (
    FeatureExtractor, StemSeparator, BeatDetector, DrumAnalyzer, 
    DrumSoundAnalyzer, SimilarityAnalyzer, ZeroShotTagger,
    AudioCaptioner, RealTimeBeatTracker
)

logger = logging.getLogger(__name__)

def extract_features_command(args):
    """CLI command to extract features from audio using specialized models."""
    try:
        # Create feature extractor
        extractor = FeatureExtractor(
            model=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Extract features
        start_time = time.time()
        features = extractor.extract(
            audio_path=args.input,
            return_timestamps=not args.no_timestamps,
            max_length_seconds=args.max_length,
            batch_size_seconds=args.batch_size
        )
        processing_time = time.time() - start_time
        
        # Add metadata
        features["processing_time"] = processing_time
        features["input_file"] = args.input
        features["model"] = args.model
        
        # Save features if output path provided
        if args.output:
            # Convert numpy arrays to lists for JSON serialization
            serializable = {}
            for key, value in features.items():
                if hasattr(value, "tolist"):
                    serializable[key] = value.tolist()
                else:
                    serializable[key] = value
            
            # Save to file
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(serializable, f, indent=2)
            
            logger.info(f"Features saved to {args.output}")
        
        # Print summary
        print(f"\nFeature extraction summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        print(f"  Embedding dimension: {features['embedding_dim']}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        if "timestamps" in features:
            print(f"  Frames: {len(features['timestamps'])}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def separate_stems_command(args):
    """CLI command to separate audio into stems."""
    try:
        # Create stem separator
        separator = StemSeparator(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device,
            num_stems=args.num_stems
        )
        
        # Create output directory if needed
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # Separate stems
        start_time = time.time()
        stems = separator.separate(
            audio_path=args.input,
            output_dir=args.output_dir if args.save_stems else None,
            save_stems=args.save_stems,
            shifts=args.shifts,
            overlap=args.overlap
        )
        processing_time = time.time() - start_time
        
        # Print summary
        print(f"\nStem separation summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        print(f"  Stems: {', '.join(stems.keys())}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        if args.save_stems and args.output_dir:
            print(f"  Stems saved to: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Stem separation error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def detect_beats_command(args):
    """CLI command to detect beats in audio."""
    try:
        # Create beat detector
        detector = BeatDetector(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Create output directory if needed for visualization
        if args.visualize and args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # Detect beats
        start_time = time.time()
        results = detector.detect(
            audio_path=args.input,
            return_downbeats=not args.no_downbeats,
            visualize=args.visualize,
            output_path=args.output if args.visualize else None
        )
        processing_time = time.time() - start_time
        
        # Add metadata
        results["processing_time"] = processing_time
        results["input_file"] = args.input
        results["model"] = args.model
        
        # Save results if output path provided
        if args.output and not args.visualize:
            # Save to file
            with open(args.output, 'w') as f:
                # Convert any numpy values to lists
                serializable = {}
                for key, value in results.items():
                    if hasattr(value, "tolist"):
                        serializable[key] = value.tolist()
                    else:
                        serializable[key] = value
                        
                json.dump(serializable, f, indent=2)
            
            logger.info(f"Beat detection results saved to {args.output}")
        
        # Print summary
        print(f"\nBeat detection summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        print(f"  Tempo: {results['tempo']:.1f} BPM")
        print(f"  Beats: {len(results['beats'])}")
        if "downbeats" in results:
            print(f"  Downbeats: {len(results['downbeats'])}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        if args.visualize and args.output:
            print(f"  Visualization saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Beat detection error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def analyze_drums_command(args):
    """CLI command to analyze drum sounds in audio."""
    try:
        # Create drum analyzer
        analyzer = DrumAnalyzer(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Create output directory if needed for visualization
        if args.visualize:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        
        # Analyze drums
        start_time = time.time()
        results = analyzer.analyze(
            audio_path=args.input,
            detect_onsets=not args.whole_file,
            visualize=args.visualize,
            output_path=args.output if args.visualize else None
        )
        processing_time = time.time() - start_time
        
        # Add metadata
        results["processing_time"] = processing_time
        results["input_file"] = args.input
        results["model"] = args.model
        
        # Save results if output path provided
        if args.output and not args.visualize:
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Drum analysis results saved to {args.output}")
        
        # Print summary
        print(f"\nDrum analysis summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        if "num_hits" in results:
            print(f"  Detected hits: {results['num_hits']}")
        elif "whole_audio" in results:
            print(f"  Main drum type: {results['whole_audio']['instrument']}")
            print(f"  Confidence: {results['whole_audio']['score']:.2f}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        if args.visualize and args.output:
            print(f"  Visualization saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Drum analysis error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def analyze_drum_patterns_command(args):
    """CLI command to analyze drum patterns in audio using specialized drum analyzer."""
    try:
        # Create drum analyzer
        analyzer = DrumSoundAnalyzer(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Analyze drums
        start_time = time.time()
        
        if args.mode == "single":
            # Analyze a single hit
            results = analyzer.analyze_drum_hit(args.input)
        elif args.mode == "detect":
            # Detect and analyze all hits
            results = analyzer.detect_drum_hits(args.input)
        elif args.mode == "pattern":
            # Full pattern analysis
            hits = analyzer.detect_drum_hits(args.input)
            results = analyzer.create_drum_pattern(
                hits=hits,
                quantize=not args.no_quantize,
                tempo=args.tempo or 120.0
            )
            
            # Export pattern if requested
            if args.export_format:
                export_result = analyzer.export_pattern(
                    pattern=results,
                    output_format=args.export_format,
                    file_path=args.export_path
                )
                if isinstance(export_result, str):
                    print(f"Pattern exported to: {export_result}")
        else:
            # Full file analysis
            results = analyzer.analyze_file(args.input)
            
        processing_time = time.time() - start_time
        
        # Add metadata
        if isinstance(results, dict):
            results["processing_time"] = processing_time
            results["input_file"] = args.input
            results["model"] = args.model
        
        # Save results if output path provided
        if args.output:
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Drum pattern analysis results saved to {args.output}")
        
        # Print summary
        print(f"\nDrum pattern analysis summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        print(f"  Mode: {args.mode}")
        if args.mode == "pattern" and "num_hits" in results:
            print(f"  Total hits: {results['num_hits']}")
            print(f"  Instruments: {', '.join(results.get('instruments', []))}")
        elif args.mode == "detect" and isinstance(results, list):
            print(f"  Detected hits: {len(results)}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Drum pattern analysis error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def audio_caption_command(args):
    """CLI command to generate captions for audio."""
    try:
        # Create audio captioner
        captioner = AudioCaptioner(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Generate caption
        start_time = time.time()
        
        if args.sentiment:
            results = captioner.caption_with_sentiment(args.input)
        elif args.mix_notes:
            results = captioner.generate_mix_notes(
                args.input,
                include_timestamps=not args.no_timestamps
            )
        else:
            results = captioner.caption(
                audio_path=args.input,
                segment_audio=not args.whole_file,
                return_all_captions=args.all_segments
            )
            
        processing_time = time.time() - start_time
        
        # Add metadata
        results["processing_time"] = processing_time
        results["input_file"] = args.input
        results["model"] = args.model
        
        # Save results if output path provided
        if args.output:
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Audio caption results saved to {args.output}")
        
        # Print summary
        print(f"\nAudio caption summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        print(f"  Caption: {results.get('caption', '')}")
        if args.sentiment and 'sentiment' in results:
            print(f"  Sentiment: {results['sentiment']['dominant']}")
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Audio caption error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def similarity_analysis_command(args):
    """CLI command to analyze audio-text similarity."""
    try:
        # Create similarity analyzer
        analyzer = SimilarityAnalyzer(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Perform similarity analysis
        start_time = time.time()
        
        if args.mode == "text":
            # Match text queries to audio
            results = analyzer.match_text_to_audio(
                audio_path=args.input,
                text_queries=args.queries
            )
        elif args.mode == "audio":
            # Match audio to reference audio
            results = analyzer.search_with_audio(
                query_audio_path=args.input,
                reference_audios=args.references,
                top_k=args.top_k
            )
        elif args.mode == "timestamps":
            # Find timestamps for text in audio
            results = analyzer.find_timestamps_for_text(
                audio_path=args.input,
                text_query=args.query,
                segment_length=args.segment_length,
                overlap=args.overlap,
                threshold=args.threshold
            )
            
        processing_time = time.time() - start_time
        
        # Add metadata
        results["processing_time"] = processing_time
        results["input_file"] = args.input
        results["model"] = args.model
        
        # Save results if output path provided
        if args.output:
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Similarity analysis results saved to {args.output}")
        
        # Print summary
        print(f"\nSimilarity analysis summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        print(f"  Mode: {args.mode}")
        
        if args.mode == "text" and "matches" in results:
            print("\n  Text matches:")
            for i, match in enumerate(results["matches"][:5]):  # Show top 5
                print(f"    {i+1}. '{match['query']}': {match['score']:.4f}")
        
        elif args.mode == "timestamps" and "matches" in results:
            print(f"\n  Timestamps for query: '{args.query}'")
            for i, match in enumerate(results["matches"][:5]):  # Show top 5
                print(f"    {match['timestamp']:.2f}s - {match['end_time']:.2f}s: {match['score']:.4f}")
        
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Similarity analysis error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def zero_shot_tagging_command(args):
    """CLI command to perform zero-shot tagging of audio."""
    try:
        # Create tagger
        tagger = ZeroShotTagger(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        # Perform tagging
        start_time = time.time()
        
        if args.tags:
            # Use custom tags
            tags = args.tags.split(',')
            results = tagger.tag(
                audio_path=args.input,
                tags=tags,
                top_k=args.top_k,
                threshold=args.threshold
            )
        elif args.categories:
            # Use custom categories
            categories = {}
            for cat in args.categories:
                name, tags_str = cat.split(':')
                categories[name] = tags_str.split(',')
            
            results = tagger.tag_with_custom_categories(
                audio_path=args.input,
                categories=categories,
                threshold=args.threshold
            )
        else:
            # Use default tags
            results = tagger.tag(
                audio_path=args.input,
                top_k=args.top_k,
                threshold=args.threshold
            )
            
        processing_time = time.time() - start_time
        
        # Add metadata
        results["processing_time"] = processing_time
        results["input_file"] = args.input
        results["model"] = args.model
        
        # Save results if output path provided
        if args.output:
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Tagging results saved to {args.output}")
        
        # Print summary
        print(f"\nZero-shot tagging summary:")
        print(f"  Model: {args.model}")
        print(f"  Input: {args.input}")
        
        if "tags" in results:
            print("\n  Top tags:")
            for i, tag_info in enumerate(results["tags"]):
                print(f"    {i+1}. {tag_info['tag']}: {tag_info['score']:.4f}")
        elif "categories" in results:
            print("\n  Categories:")
            for category, tags in results["categories"].items():
                print(f"    {category}:")
                for i, tag_info in enumerate(tags[:5]):  # Show top 5
                    print(f"      {i+1}. {tag_info['tag']}: {tag_info['score']:.4f}")
        
        print(f"  Processing time: {processing_time:.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger.error(f"Zero-shot tagging error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def real_time_beat_tracker_command(args):
    """CLI command to run real-time beat tracking."""
    try:
        # Create beat tracker
        tracker = RealTimeBeatTracker(
            model_name=args.model,
            api_key=args.api_key,
            use_cuda=not args.cpu,
            device=args.device
        )
        
        start_time = time.time()
        
        if args.file:
            # Process a file
            results = tracker.process_file(
                audio_path=args.input,
                simulate_realtime=args.simulate_realtime,
                chunk_size=args.chunk_size
            )
            
            processing_time = time.time() - start_time
            results["processing_time"] = processing_time
            
            # Save results if output path provided
            if args.output:
                # Save to file
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                
                logger.info(f"Beat tracking results saved to {args.output}")
                
            # Export beats to file if requested
            if args.export_beats and args.export_path:
                tracker.save_beats_to_file(
                    beats=results["beats"],
                    output_path=args.export_path,
                    format=args.export_format
                )
                print(f"Beats exported to: {args.export_path}")
            
            # Print summary
            print(f"\nBeat tracking summary:")
            print(f"  Model: {args.model}")
            print(f"  Input: {args.input}")
            print(f"  Beats detected: {len(results['beats'])}")
            print(f"  Estimated tempo: {results['tempo']:.1f} BPM")
            print(f"  Processing time: {processing_time:.2f} seconds")
            
        else:
            # Run real-time demo
            print(f"Running real-time beat tracking demo for {args.duration} seconds...")
            print(f"Please make sure your microphone is connected and unmuted.")
            
            results = tracker.run_real_time_demo(
                duration=args.duration,
                visualize=not args.no_visualize
            )
            
            if "error" in results:
                print(f"Error during real-time demo: {results['error']}")
                return 1
                
            print(f"\nReal-time demo completed:")
            print(f"  Beats detected: {len(results['beats'])}")
            print(f"  Final tempo: {results['tempo']:.1f} BPM")
        
        return 0
        
    except Exception as e:
        logger.error(f"Real-time beat tracking error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

def setup_extract_parser(subparsers):
    """Set up the parser for the feature extraction command."""
    parser = subparsers.add_parser(
        "extract",
        help="Extract features from audio using specialized models"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="microsoft/BEATs-base",
        help="Model name/path (default: microsoft/BEATs-base)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save extracted features JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--max-length",
        type=float,
        default=30.0,
        help="Maximum length of audio to process in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=float,
        default=10.0,
        help="Batch size in seconds for processing long audio (default: 10)"
    )
    
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Don't include timestamps in output"
    )
    
    parser.set_defaults(func=extract_features_command)

def setup_separate_parser(subparsers):
    """Set up the parser for the stem separation command."""
    parser = subparsers.add_parser(
        "separate",
        help="Separate audio into stems"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="facebook/demucs",
        help="Model name/path (default: facebook/demucs)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        help="Directory to save separated stems"
    )
    
    parser.add_argument(
        "--save-stems",
        action="store_true",
        help="Save separated stems to disk"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--num-stems",
        type=int,
        choices=[4, 6],
        default=4,
        help="Number of stems to separate (default: 4)"
    )
    
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for equivariant stabilization (default: 1)"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap between window splits (default: 0.25)"
    )
    
    parser.set_defaults(func=separate_stems_command)

def setup_beats_parser(subparsers):
    """Set up the parser for the beat detection command."""
    parser = subparsers.add_parser(
        "beats",
        help="Detect beats in audio"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="nicolaus625/cmi",
        help="Model name/path (default: nicolaus625/cmi)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save results or visualization"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--no-downbeats",
        action="store_true",
        help="Don't detect downbeats"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of detected beats"
    )
    
    parser.set_defaults(func=detect_beats_command)

def setup_drums_parser(subparsers):
    """Set up the parser for drum analysis command."""
    parser = subparsers.add_parser(
        "analyze-drums",
        help="Analyze drum sounds in audio"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="DunnBC22/wav2vec2-base-Drum_Kit_Sounds",
        help="Model name/path (default: DunnBC22/wav2vec2-base-Drum_Kit_Sounds)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save analysis results JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help="Analyze whole file instead of detecting onsets"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization of drum hits"
    )
    
    parser.set_defaults(func=analyze_drums_command)

def setup_drum_patterns_parser(subparsers):
    """Set up the parser for drum pattern analysis command."""
    parser = subparsers.add_parser(
        "drum-patterns",
        help="Analyze drum patterns using specialized models"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="JackArt/wav2vec2-for-drum-classification",
        help="Model name/path (default: JackArt/wav2vec2-for-drum-classification)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["single", "detect", "pattern", "file"],
        default="file",
        help="Analysis mode (single hit, detect all, create pattern, or full file)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save analysis results JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--tempo",
        type=float,
        help="Tempo for pattern creation (BPM)"
    )
    
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable quantization for pattern creation"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["json", "midi", "csv"],
        help="Format for exporting pattern"
    )
    
    parser.add_argument(
        "--export-path",
        help="Path for exported pattern"
    )
    
    parser.set_defaults(func=analyze_drum_patterns_command)

def setup_caption_parser(subparsers):
    """Set up the parser for audio captioning command."""
    parser = subparsers.add_parser(
        "caption",
        help="Generate captions for audio"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="slseanwu/beats-conformer-bart-audio-captioner",
        help="Model name/path (default: slseanwu/beats-conformer-bart-audio-captioner)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save caption results JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.add_argument(
        "--whole-file",
        action="store_true",
        help="Caption whole file without segmentation"
    )
    
    parser.add_argument(
        "--all-segments",
        action="store_true",
        help="Return captions for all segments"
    )
    
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Include sentiment analysis with caption"
    )
    
    parser.add_argument(
        "--mix-notes",
        action="store_true",
        help="Generate mix notes format"
    )
    
    parser.add_argument(
        "--no-timestamps",
        action="store_true",
        help="Exclude timestamps from mix notes"
    )
    
    parser.set_defaults(func=audio_caption_command)

def setup_similarity_parser(subparsers):
    """Set up the parser for similarity analysis command."""
    parser = subparsers.add_parser(
        "similarity",
        help="Analyze audio-text similarity"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="laion/clap-htsat-fused",
        help="Model name/path (default: laion/clap-htsat-fused)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["text", "audio", "timestamps"],
        default="text",
        help="Analysis mode (text-to-audio, audio-to-audio, or find timestamps)"
    )
    
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Text queries for text-to-audio mode"
    )
    
    parser.add_argument(
        "--query",
        help="Single text query for timestamps mode"
    )
    
    parser.add_argument(
        "--references",
        nargs="+",
        help="Reference audio files for audio-to-audio mode"
    )
    
    parser.add_argument(
        "--segment-length",
        type=float,
        default=3.0,
        help="Segment length in seconds for timestamps mode"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=1.5,
        help="Segment overlap in seconds for timestamps mode"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Similarity threshold"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save similarity results JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.set_defaults(func=similarity_analysis_command)

def setup_tagging_parser(subparsers):
    """Set up the parser for zero-shot tagging command."""
    parser = subparsers.add_parser(
        "tag",
        help="Perform zero-shot tagging of audio"
    )
    
    parser.add_argument(
        "input",
        help="Path to input audio file"
    )
    
    parser.add_argument(
        "--model",
        default="UniMus/OpenJMLA",
        help="Model name/path (default: UniMus/OpenJMLA)"
    )
    
    parser.add_argument(
        "--tags",
        help="Comma-separated list of custom tags"
    )
    
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Custom categories in format 'name:tag1,tag2,tag3'"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top tags to return"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for tags"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save tagging results JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.set_defaults(func=zero_shot_tagging_command)

def setup_realtime_beats_parser(subparsers):
    """Set up the parser for real-time beat tracking command."""
    parser = subparsers.add_parser(
        "realtime-beats",
        help="Run real-time beat tracking"
    )
    
    parser.add_argument(
        "--model",
        default="beast-team/beast-dione",
        help="Model name/path (default: beast-team/beast-dione)"
    )
    
    parser.add_argument(
        "--file",
        action="store_true",
        help="Process a file instead of microphone input"
    )
    
    parser.add_argument(
        "--input",
        help="Path to input audio file (required if --file is used)"
    )
    
    parser.add_argument(
        "--simulate-realtime",
        action="store_true",
        help="Simulate real-time processing when processing a file"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=0.1,
        help="Chunk size in seconds for simulated real-time processing"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Duration of real-time demo in seconds"
    )
    
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visualization for real-time demo"
    )
    
    parser.add_argument(
        "--export-beats",
        action="store_true",
        help="Export detected beats to file"
    )
    
    parser.add_argument(
        "--export-path",
        help="Path for exported beats"
    )
    
    parser.add_argument(
        "--export-format",
        choices=["csv", "txt", "json"],
        default="csv",
        help="Format for exporting beats"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Path to save tracking results JSON"
    )
    
    parser.add_argument(
        "--api-key",
        help="HuggingFace API key"
    )
    
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if GPU is available"
    )
    
    parser.add_argument(
        "--device",
        help="Specific device to use (e.g., 'cuda:0', 'cpu')"
    )
    
    parser.set_defaults(func=real_time_beat_tracker_command)

def setup_parser(subparsers):
    """Set up the parsers for all Hugging Face commands."""
    # Create a subparser for Hugging Face commands
    hf_parser = subparsers.add_parser(
        "huggingface",
        help="Hugging Face model operations"
    )
    
    hf_subparsers = hf_parser.add_subparsers(
        dest="hf_command",
        required=True,
        help="Hugging Face command to execute"
    )
    
    # Set up individual command parsers
    setup_extract_parser(hf_subparsers)
    setup_separate_parser(hf_subparsers)
    setup_beats_parser(hf_subparsers)
    setup_drums_parser(hf_subparsers)
    setup_drum_patterns_parser(hf_subparsers)
    setup_caption_parser(hf_subparsers)
    setup_similarity_parser(hf_subparsers)
    setup_tagging_parser(hf_subparsers)
    setup_realtime_beats_parser(hf_subparsers)
    
    return hf_parser 