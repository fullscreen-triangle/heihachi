import argparse
import os
import json
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import sys

# Add parent directory to path to import core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semantic.vector_store import VectorStore
from semantic.embedding_generator import EmbeddingGenerator
from semantic.query_processor import QueryProcessor
from semantic.chat_interface import ChatInterface

# Import core pipeline if available
try:
    from core.pipeline import Pipeline
except ImportError:
    Pipeline = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("triangle.semantic")

class SemanticSearch:
    """
    Main class for Triangle's semantic search functionality.
    Provides methods for indexing tracks, searching, and running the chat interface.
    """
    
    def __init__(self, storage_dir: str = "data/vector_store", api_key: Optional[str] = None,
                pipeline_config: Optional[str] = None):
        """
        Initialize the semantic search system.
        
        Args:
            storage_dir: Directory to store vector data
            api_key: OpenAI API key (optional, will use environment variable if not provided)
            pipeline_config: Path to pipeline configuration file (optional)
        """
        self.storage_dir = storage_dir
        self.api_key = api_key
        
        # Initialize components
        self.vector_store = VectorStore(storage_dir=storage_dir)
        self.embedding_generator = EmbeddingGenerator(api_key=api_key)
        self.query_processor = QueryProcessor(self.vector_store, api_key=api_key)
        self.chat_interface = ChatInterface(self.vector_store, api_key=api_key)
        
        # Initialize pipeline if config provided
        self.pipeline = None
        if pipeline_config and Pipeline:
            try:
                self.pipeline = Pipeline(config_path=pipeline_config)
                logger.info(f"Initialized Triangle pipeline with config: {pipeline_config}")
            except Exception as e:
                logger.error(f"Failed to initialize Triangle pipeline: {str(e)}")
        
        logger.info(f"Initialized semantic search with {self.vector_store.count()} indexed tracks")
    
    def analyze_and_index_audio(self, audio_path: str, track_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Analyze audio file using Triangle pipeline and index the results.
        
        Args:
            audio_path: Path to audio file
            track_info: Optional track information (will be extracted from filename if not provided)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.pipeline:
            logger.error("Triangle pipeline not initialized. Cannot analyze audio.")
            return False
        
        try:
            # Extract track info from filename if not provided
            if not track_info:
                filename = os.path.basename(audio_path)
                name_parts = os.path.splitext(filename)[0].split(' - ', 1)
                
                if len(name_parts) == 2:
                    artist, title = name_parts
                else:
                    artist = "Unknown Artist"
                    title = name_parts[0]
                
                track_info = {
                    "title": title,
                    "artist": artist,
                    "path": audio_path
                }
            
            # Generate track ID
            track_id = track_info.get("id", f"{track_info.get('artist', 'unknown')}_{track_info.get('title', 'unknown')}")
            
            # Run Triangle pipeline analysis
            logger.info(f"Analyzing audio: {audio_path}")
            analysis_result = self.pipeline.process(audio_path)
            
            # Index the track
            logger.info(f"Indexing track: {track_info.get('title', track_id)}")
            return self.index_track(track_id, track_info, analysis_result)
            
        except Exception as e:
            logger.error(f"Error analyzing and indexing audio {audio_path}: {str(e)}")
            return False
    
    def batch_analyze_and_index(self, audio_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze and index multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            Dictionary with success and failure counts
        """
        if not self.pipeline:
            logger.error("Triangle pipeline not initialized. Cannot analyze audio.")
            return {"success_count": 0, "failure_count": len(audio_paths), "total": len(audio_paths)}
        
        success_count = 0
        failure_count = 0
        
        for audio_path in audio_paths:
            if self.analyze_and_index_audio(audio_path):
                success_count += 1
            else:
                failure_count += 1
        
        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "total": success_count + failure_count
        }
    
    def index_track(self, track_id: str, track_info: Dict[str, Any], 
                   analysis_result: Dict[str, Any]) -> bool:
        """
        Index a track for semantic search.
        
        Args:
            track_id: Unique identifier for the track
            track_info: Basic track information (title, artist, etc.)
            analysis_result: Complete analysis result from Triangle pipeline
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate embedding
            embedding = self.embedding_generator.generate_track_embedding(track_info, analysis_result)
            
            # Extract emotional features for metadata
            emotions = self.embedding_generator.feature_mapper.map_features_to_emotions(analysis_result)
            
            # Create metadata
            metadata = {
                "emotions": emotions,
                "analysis_summary": {
                    "bpm": self.embedding_generator._extract_bpm(analysis_result),
                    "key": self.embedding_generator._extract_key(analysis_result),
                    "has_amen_break": "amen_break" in analysis_result.get("alignment", {})
                }
            }
            
            # Add to vector store
            success = self.vector_store.add_track(track_id, track_info, embedding, metadata)
            
            if success:
                logger.info(f"Successfully indexed track: {track_info.get('title', track_id)}")
            else:
                logger.error(f"Failed to index track: {track_info.get('title', track_id)}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error indexing track {track_id}: {str(e)}")
            return False
    
    def batch_index_tracks(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Index multiple tracks in batch.
        
        Args:
            analysis_results: List of analysis results with track info
            
        Returns:
            Dictionary with success and failure counts
        """
        success_count = 0
        failure_count = 0
        
        for result in analysis_results:
            if "track_info" not in result or "analysis" not in result:
                logger.warning("Skipping invalid analysis result")
                failure_count += 1
                continue
            
            track_info = result["track_info"]
            analysis = result["analysis"]
            
            # Generate track ID if not provided
            track_id = track_info.get("id", f"{track_info.get('artist', 'unknown')}_{track_info.get('title', 'unknown')}")
            
            if self.index_track(track_id, track_info, analysis):
                success_count += 1
            else:
                failure_count += 1
        
        return {
            "success_count": success_count,
            "failure_count": failure_count,
            "total": success_count + failure_count
        }
    
    def search(self, query: str, top_k: int = 5, enhance_query: bool = True) -> Dict[str, Any]:
        """
        Search for tracks matching a query.
        
        Args:
            query: User's natural language query
            top_k: Number of results to return
            enhance_query: Whether to enhance the query with emotional context
            
        Returns:
            Search results
        """
        return self.query_processor.process_query(query, top_k=top_k, enhance_query=enhance_query)
    
    def run_chat_interface(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Run the chat interface server.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        logger.info(f"Starting chat interface on {host}:{port}")
        self.chat_interface.run(host=host, port=port)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the indexed tracks.
        
        Returns:
            Dictionary with statistics
        """
        track_count = self.vector_store.count()
        
        # Get sample of track emotions if available
        emotion_stats = {}
        if track_count > 0:
            # Sample up to 100 tracks
            sample_size = min(track_count, 100)
            sample_ids = self.vector_store.get_all_ids()[:sample_size]
            
            # Collect emotion values
            emotions = {}
            for track_id in sample_ids:
                metadata = self.vector_store.get_metadata(track_id)
                if metadata and "emotions" in metadata:
                    for emotion, value in metadata["emotions"].items():
                        if emotion not in emotions:
                            emotions[emotion] = []
                        emotions[emotion].append(value)
            
            # Calculate stats
            for emotion, values in emotions.items():
                if values:
                    emotion_stats[emotion] = {
                        "mean": np.mean(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "std": np.std(values)
                    }
        
        return {
            "track_count": track_count,
            "emotion_stats": emotion_stats
        }


def main():
    """Command-line interface for Triangle semantic search."""
    parser = argparse.ArgumentParser(description="Triangle Semantic Search")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index tracks for semantic search")
    index_parser.add_argument("--input", "-i", required=True, help="Path to analysis results JSON file")
    index_parser.add_argument("--storage-dir", "-d", default="data/vector_store", help="Storage directory for vector store")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze and index audio files")
    analyze_parser.add_argument("--input", "-i", required=True, help="Path to audio file or directory")
    analyze_parser.add_argument("--storage-dir", "-d", default="data/vector_store", help="Storage directory for vector store")
    analyze_parser.add_argument("--pipeline-config", "-c", default="configs/default.yaml", help="Path to pipeline configuration file")
    analyze_parser.add_argument("--recursive", "-r", action="store_true", help="Recursively process directories")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for tracks")
    search_parser.add_argument("--query", "-q", required=True, help="Search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--no-enhance", action="store_true", help="Disable query enhancement")
    search_parser.add_argument("--storage-dir", "-d", default="data/vector_store", help="Storage directory for vector store")
    
    # Chat interface command
    chat_parser = subparsers.add_parser("chat", help="Run chat interface")
    chat_parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    chat_parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind the server to")
    chat_parser.add_argument("--storage-dir", "-d", default="data/vector_store", help="Storage directory for vector store")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics about indexed tracks")
    stats_parser.add_argument("--storage-dir", "-d", default="data/vector_store", help="Storage directory for vector store")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Execute command
    if args.command == "index":
        # Initialize semantic search
        semantic_search = SemanticSearch(storage_dir=args.storage_dir, api_key=api_key)
        
        # Load analysis results
        with open(args.input, 'r') as f:
            analysis_results = json.load(f)
        
        # Index tracks
        result = semantic_search.batch_index_tracks(analysis_results)
        print(f"Indexed {result['success_count']} tracks successfully, {result['failure_count']} failures")
    
    elif args.command == "analyze":
        # Check if Pipeline is available
        if not Pipeline:
            print("Error: Triangle pipeline not available. Make sure core modules are in the Python path.")
            return
        
        # Initialize semantic search with pipeline
        semantic_search = SemanticSearch(
            storage_dir=args.storage_dir, 
            api_key=api_key,
            pipeline_config=args.pipeline_config
        )
        
        # Process input path
        input_path = args.input
        if os.path.isfile(input_path):
            # Process single file
            if semantic_search.analyze_and_index_audio(input_path):
                print(f"Successfully analyzed and indexed: {input_path}")
            else:
                print(f"Failed to analyze and index: {input_path}")
        elif os.path.isdir(input_path):
            # Process directory
            audio_paths = []
            
            # Collect audio files
            if args.recursive:
                for root, _, files in os.walk(input_path):
                    for file in files:
                        if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                            audio_paths.append(os.path.join(root, file))
            else:
                for file in os.listdir(input_path):
                    if file.endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                        audio_paths.append(os.path.join(input_path, file))
            
            if not audio_paths:
                print(f"No audio files found in {input_path}")
                return
            
            print(f"Found {len(audio_paths)} audio files to process")
            
            # Process files
            result = semantic_search.batch_analyze_and_index(audio_paths)
            print(f"Analyzed and indexed {result['success_count']} tracks successfully, {result['failure_count']} failures")
        else:
            print(f"Error: Input path does not exist: {input_path}")
    
    elif args.command == "search":
        # Initialize semantic search
        semantic_search = SemanticSearch(storage_dir=args.storage_dir, api_key=api_key)
        
        # Search for tracks
        results = semantic_search.search(args.query, top_k=args.top_k, enhance_query=not args.no_enhance)
        
        # Print results
        print(f"Query: {args.query}")
        if args.no_enhance:
            print("Query enhancement disabled")
        else:
            print(f"Enhanced query: {results['enhanced_query']}")
        
        print("\nResults:")
        for i, result in enumerate(results["results"]):
            print(f"{i+1}. {result['info'].get('title', 'Unknown')} by {result['info'].get('artist', 'Unknown Artist')}")
            print(f"   Similarity: {result['similarity']:.4f}")
            if "metadata" in result and "emotions" in result["metadata"]:
                top_emotions = sorted(result["metadata"]["emotions"].items(), key=lambda x: x[1], reverse=True)[:3]
                emotions_str = ", ".join([f"{e}: {v:.1f}" for e, v in top_emotions])
                print(f"   Top emotions: {emotions_str}")
            print()
        
        if results["explanation"]:
            print("Explanation:")
            print(results["explanation"])
    
    elif args.command == "chat":
        # Initialize semantic search
        semantic_search = SemanticSearch(storage_dir=args.storage_dir, api_key=api_key)
        
        # Run chat interface
        semantic_search.run_chat_interface(host=args.host, port=args.port)
    
    elif args.command == "stats":
        # Initialize semantic search
        semantic_search = SemanticSearch(storage_dir=args.storage_dir, api_key=api_key)
        
        # Get statistics
        stats = semantic_search.get_stats()
        
        print(f"Total tracks indexed: {stats['track_count']}")
        
        if "emotion_stats" in stats and stats["emotion_stats"]:
            print("\nEmotion statistics:")
            for emotion, values in stats["emotion_stats"].items():
                print(f"  {emotion}:")
                print(f"    Mean: {values['mean']:.2f}")
                print(f"    Range: {values['min']:.2f} - {values['max']:.2f}")
                print(f"    Std Dev: {values['std']:.2f}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
