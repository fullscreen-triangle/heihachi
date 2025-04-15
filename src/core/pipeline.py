import torch
import numpy as np
import gc
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import logging
import os

# Add the project root to sys.path to ensure modules can be found
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from alignment.alignment import SequenceAligner
from alignment.prior_subspace_analysis import PriorSubspaceAnalysis
from alignment.similarity import SimilarityAnalyzer
from alignment.composite_similarity import CompositeSimilarity

from feature_extraction.baseline_analysis import BasslineAnalyzer
from feature_extraction.bpm_analysis import BPMAnalyzer
from feature_extraction.drum_analysis import DrumAnalyzer
from feature_extraction.groove_analysis import GrooveAnalyzer
from feature_extraction.percussion_analysis import PercussionAnalyzer
from feature_extraction.rhythmic_analysis import RhythmAnalyzer

from core.mix_analyzer import MixAnalyzer
from core.audio_scene import AudioSceneAnalyzer
from core.audio_processing import AudioProcessor

from annotation.peak_detection import PeakDetector
from annotation.segment_clustering import SegmentClusterer
from annotation.transition_detector import TransitionDetector

from utils.visualization import MixVisualizer, AnalysisVisualizer
from utils.storage import AudioCache, FeatureStorage, AnalysisVersion
from utils.logging_utils import setup_logging, start_memory_monitoring
from utils.config import ConfigManager

# Set up logger
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config_path: str = "../configs/default.yaml"):
        """Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to the config file
        """
        self.config = ConfigManager(config_path)
        
        # Set device (CPU or GPU)
        use_gpu = self.config.get('processing', 'use_gpu')
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set number of workers
        self.num_workers = self.config.get('processing', 'n_workers')
        if self.num_workers is None:
            self.num_workers = mp.cpu_count()
        logger.info(f"Using {self.num_workers} workers for parallel processing")

        # Just use simple hardcoded paths
        cache_dir = "../cache"
        results_dir = "../results"
        visualizations_dir = "../visualizations"
        
        # Initialize storage
        self.cache = AudioCache(cache_dir)
        self.storage = FeatureStorage(results_dir)
        
        # Initialize visualizers
        self.mix_visualizer = MixVisualizer(results_dir)
        self.analysis_visualizer = AnalysisVisualizer(visualizations_dir)

        # Initialize components as needed
        self._initialize_components()

    def _initialize_components(self):
        """Initialize analysis components on demand to save memory."""
        logger.info("Initializing pipeline components")
        
        # Initialize audio processor first (needed for all operations)
        self.audio_processor = AudioProcessor()
        
        # Initialize analyzers with delayed loading for others
        self.mix_analyzer = MixAnalyzer(use_gpu=(self.device.type == 'cuda'))
        
        # These will be initialized when needed
        self._sequence_aligner = None
        self._prior_subspace = None
        self._similarity_analyzer = None
        self._composite_similarity = None
        self._baseline_analyzer = None
        self._bpm_analyzer = None
        self._drum_analyzer = None
        self._groove_analyzer = None
        self._percussion_analyzer = None
        self._rhythmic_analyzer = None
        self._scene_analyzer = None
        self._peak_detector = None
        self._segment_clusterer = None
        self._transition_detector = None

    def process(self, audio_path: str) -> Dict:
        """Main pipeline processing with memory optimization.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict: Analysis results
        """
        # Start memory monitoring
        memory_monitor = start_memory_monitoring(interval=2.0)
        
        try:
            # Try loading from cache first
            cached_result = self.cache.get(audio_path)
            if cached_result:
                logger.info(f"Found cached results for {audio_path}")
                memory_monitor.stop()
                return cached_result

            # Get file size
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"Processing file of size: {file_size_mb:.2f} MB")

            # Load and preprocess audio
            logger.info("Loading audio file")
            audio = self.audio_processor.load(audio_path)
            logger.info(f"Audio loaded: {len(audio)} samples")
            
            # Free up memory after loading
            gc.collect()

            # Process with parallel executors
            logger.info("Starting parallel processing")
            results = self._parallel_process(audio)
            
            # Free memory after processing
            del audio
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Generate visualizations
            logger.info("Generating visualizations")
            results['visualizations'] = self._generate_visualizations(results)

            # Generate analysis report
            results['report'] = self.mix_visualizer.create_summary_report(results)

            # Cache results
            logger.info("Caching results")
            self.cache.set(audio_path, results)

            # Save to persistent storage
            logger.info("Saving results to storage")
            self.storage.save_results(
                audio_path,
                results,
                metadata={'version': AnalysisVersion.to_string()}
            )
            
            # Stop memory monitoring
            memory_monitor.stop()

            return results

        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            if memory_monitor:
                memory_monitor.stop()
            raise

    def _generate_visualizations(self, results: Dict) -> Dict:
        """Generate all visualizations for the analysis results."""
        visualizations = {
            'mix_signature': self.mix_visualizer.visualize_mix_signature(
                results['analysis']['mix'].get('graph', {}),
                results['analysis']['mix'].get('metrics', {}),
                'mix_signature.png'
            ),
            'mix_heatmap': self.mix_visualizer.plot_mix_heatmap(
                results['features'],
                results['analysis'].get('timestamps', []),
                'mix_heatmap.png'
            ),
            'mix_radar': self.mix_visualizer.plot_mix_radar(
                results['features'],
                results['analysis'].get('timestamps', []),
                'mix_radar.html'
            ),
            'clusters': self.mix_visualizer.plot_clusters(
                results['annotation']['segments'],
                results['features'].get('spectral', {}),
                'clusters.png'
            ),
            'components': self.analysis_visualizer.plot_component_analysis(
                results['analysis'].get('components', {}),
                'component_analysis.html'
            )
        }
        
        # Add Amen break visualization if segments were found
        # segments have to be found, if none where found, then the process/pipeline failed.
        if results.get('alignment', {}).get('sequence', {}).get('segments'):
            # Create placeholders for the audio data that's no longer in memory
            dummy_audio = np.zeros(1000)  # Just a placeholder
            visualizations['amen_break'] = self.analysis_visualizer.plot_amen_break_analysis(
                dummy_audio,  # We don't need the actual audio for visualization
                results['alignment']['sequence'],
                'amen_break_analysis.html'
            )
        
        return visualizations

    def _parallel_process(self, audio: np.ndarray) -> Dict:
        """Process audio with parallel executors and memory optimization."""
        # Initialize needed components based on file size
        self._initialize_needed_components(len(audio))
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Feature extraction - most critical for memory
            logger.info("Starting feature extraction...")
            feature_futures = {}
            
            # Always extract BPM and baseline features (small memory footprint)
            if self._bpm_analyzer:
                feature_futures['bpm'] = executor.submit(self._bpm_analyzer.analyze, audio)
            if self._baseline_analyzer:
                feature_futures['baseline'] = executor.submit(self._baseline_analyzer.analyze, audio)
            
            # Conditionally add heavier analyses based on remaining memory
            self._add_conditional_analyses(feature_futures, executor, audio)
            
            # Analysis
            logger.info("Starting audio analysis...")
            analysis_futures = {
                'mix': executor.submit(self.mix_analyzer.analyze, audio),
            }
            
            if self._scene_analyzer:
                analysis_futures['scene'] = executor.submit(self._scene_analyzer.analyze, audio)

            # Alignment
            logger.info("Starting alignment analysis...")
            alignment_futures = {}
            
            if self._sequence_aligner:
                # Ensure we have a valid sample rate
                sample_rate = self.config.get('audio', 'sample_rate')
                if sample_rate is None:
                    sample_rate = 44100  # Default to 44.1kHz if not specified in config
                
                alignment_futures['sequence'] = executor.submit(
                    self._sequence_aligner.align_sequences, 
                    audio, 
                    sample_rate
                )
            
            if self._prior_subspace:
                alignment_futures['psa'] = executor.submit(self._prior_subspace.analyze, audio)
                
            if self._similarity_analyzer:
                alignment_futures['similarity'] = executor.submit(self._similarity_analyzer.analyze, audio)

            # Annotation
            logger.info("Starting annotation analysis...")
            annotation_futures = {}
            
            if self._peak_detector:
                annotation_futures['peaks'] = executor.submit(self._peak_detector.detect, audio)
                
            if self._segment_clusterer:
                annotation_futures['segments'] = executor.submit(self._segment_clusterer.cluster, audio)
                
            if self._transition_detector:
                annotation_futures['transitions'] = executor.submit(self._transition_detector.detect, audio)

            # Gather results
            results = {
                'features': {k: v.result() for k, v in feature_futures.items()},
                'analysis': {k: v.result() for k, v in analysis_futures.items()},
                'alignment': {k: v.result() for k, v in alignment_futures.items()},
                'annotation': {k: v.result() for k, v in annotation_futures.items()}
            }

            # Process alignment results with enhanced Amen break analysis
            self._process_alignment_results(results)
            
            # Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return results
            
    def _initialize_needed_components(self, audio_length: int):
        """Initialize only the components needed for the analysis based on file size."""
        # Calculate audio length in seconds (assuming 44.1kHz)
        audio_seconds = audio_length / 44100
        logger.info(f"Audio duration: approximately {audio_seconds:.2f} seconds")
        
        # For very small files, initialize all components
        if audio_seconds < 60:  # Less than a minute
            logger.info("Small file detected, initializing all components")
            self._initialize_all_components()
            return
            
        # For medium files, initialize most components but skip some heavy ones
        if audio_seconds < 300:  # Less than 5 minutes
            logger.info("Medium file detected, initializing most components")
            self._initialize_medium_analysis()
            return
            
        # For large files, initialize only essential components
        logger.info("Large file detected, initializing essential components only")
        self._initialize_minimal_components()
    
    def _initialize_all_components(self):
        """Initialize all analysis components."""
        if not self._sequence_aligner:
            self._sequence_aligner = SequenceAligner()
        if not self._prior_subspace:
            self._prior_subspace = PriorSubspaceAnalysis()
        if not self._similarity_analyzer:
            self._similarity_analyzer = SimilarityAnalyzer()
        if not self._composite_similarity:
            self._composite_similarity = CompositeSimilarity()
        if not self._baseline_analyzer:
            self._baseline_analyzer = BasslineAnalyzer()
        if not self._bpm_analyzer:
            self._bpm_analyzer = BPMAnalyzer()
        if not self._drum_analyzer:
            self._drum_analyzer = DrumAnalyzer()
        if not self._groove_analyzer:
            self._groove_analyzer = GrooveAnalyzer()
        if not self._percussion_analyzer:
            self._percussion_analyzer = PercussionAnalyzer()
        if not self._rhythmic_analyzer:
            self._rhythmic_analyzer = RhythmAnalyzer()
        if not self._scene_analyzer:
            self._scene_analyzer = AudioSceneAnalyzer()
        if not self._peak_detector:
            self._peak_detector = PeakDetector()
        if not self._segment_clusterer:
            self._segment_clusterer = SegmentClusterer()
        if not self._transition_detector:
            self._transition_detector = TransitionDetector()
    
    def _initialize_medium_analysis(self):
        """Initialize components for medium-sized files."""
        if not self._sequence_aligner:
            self._sequence_aligner = SequenceAligner()
        if not self._baseline_analyzer:
            self._baseline_analyzer = BasslineAnalyzer()
        if not self._bpm_analyzer:
            self._bpm_analyzer = BPMAnalyzer()
        if not self._drum_analyzer:
            self._drum_analyzer = DrumAnalyzer()
        if not self._rhythmic_analyzer:
            self._rhythmic_analyzer = RhythmAnalyzer()
        if not self._scene_analyzer:
            self._scene_analyzer = AudioSceneAnalyzer()
        if not self._peak_detector:
            self._peak_detector = PeakDetector()
        if not self._segment_clusterer:
            self._segment_clusterer = SegmentClusterer()
    
    def _initialize_minimal_components(self):
        """Initialize only essential components for large files."""
        if not self._bpm_analyzer:
            self._bpm_analyzer = BPMAnalyzer()
        if not self._baseline_analyzer:
            self._baseline_analyzer = BasslineAnalyzer()
        if not self._peak_detector:
            self._peak_detector = PeakDetector()
    
    def _add_conditional_analyses(self, feature_futures, executor, audio):
        """Add heavier analyses only if memory permits."""
        # Check available memory
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            logger.info(f"Available memory: {available_memory:.2f} MB")
            
            # Add analyses based on available memory
            if available_memory > 2000:  # More than 2GB available
                if self._drum_analyzer:
                    feature_futures['drum'] = executor.submit(self._drum_analyzer.analyze, audio)
                if self._rhythmic_analyzer:
                    feature_futures['rhythmic'] = executor.submit(self._rhythmic_analyzer.analyze, audio)
                    
            if available_memory > 4000:  # More than 4GB available
                if self._groove_analyzer:
                    feature_futures['groove'] = executor.submit(self._groove_analyzer.analyze, audio)
                if self._percussion_analyzer:
                    feature_futures['percussion'] = executor.submit(self._percussion_analyzer.analyze, audio)
                    
        except ImportError:
            # If psutil not available, add minimal analyses
            if self._drum_analyzer:
                feature_futures['drum'] = executor.submit(self._drum_analyzer.analyze, audio)
    
    def _process_alignment_results(self, results):
        """Process alignment results if available."""
        # Only process if sequence alignment was performed and returned results
        if 'alignment' not in results or 'sequence' not in results['alignment']:
            return
            
        alignment_results = results['alignment']['sequence']
        if not alignment_results.get('segments'):
            logger.info("No Amen break patterns detected in the audio")
            return
            
        # Process alignment results
        n_segments = len(alignment_results['segments'])
        logger.info(f"Found {n_segments} potential Amen break segments")
        
        # Only process if sequence aligner is initialized and has template
        if not self._sequence_aligner or not hasattr(self._sequence_aligner, 'amen_template'):
            logger.warning("Sequence aligner not available for detailed analysis")
            return
            
        # Add variation names and characteristics
        try:
            alignment_results['variation_names'] = [
                self._sequence_aligner.amen_template.get_variation_name(v_type)
                for v_type in alignment_results['variation_types']
            ]
            
            alignment_results['variation_characteristics'] = [
                self._sequence_aligner.amen_template.get_variation_characteristics(v_type)
                for v_type in alignment_results['variation_types']
            ]
            
            # Add component analysis
            alignment_results['component_analysis'] = [
                self._sequence_aligner.amen_template.get_component_analysis(v_type)
                for v_type in alignment_results['variation_types']
            ]
            
            # Log detailed information for each segment
            for i, segment in enumerate(alignment_results['segments']):
                variation_name = alignment_results['variation_names'][i]
                # the two variables below are never used
                characteristics = alignment_results['variation_characteristics'][i]
                comp_analysis = alignment_results['component_analysis'][i]
                
                logger.info(
                    f"\nSegment {i+1}:"
                    f"\n  Time: {segment['start']:.2f}s - {segment['end']:.2f}s"
                    f"\n  Variation: {variation_name}"
                    f"\n  Confidence: {alignment_results['confidence'][i]:.2f}"
                    f"\n  Tempo Scale: {alignment_results['tempo_scales'][i]:.2f}x"
                )
                
        except Exception as e:
            logger.error(f"Error processing alignment results: {str(e)}")

    def process_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Process multiple audio files in sequence to manage memory better."""
        results = []
        for path in audio_paths:
            try:
                # Process one at a time to avoid memory issues
                logger.info(f"Processing file: {path}")
                result = self.process(path)
                results.append(result)
                
                # Force garbage collection between files
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"Failed processing {path}: {str(e)}")
                continue
        return results


def main():
    """Main function for processing audio files with Amen break detection.
    
    This function handles the processing of a single audio file and analyzes it
    for Amen break patterns by explicitly providing the Amen break sample file.
    It optimizes memory usage and ensures proper initialization of all components.
    """
    # Configure logging
    setup_logging()
    logger = logging.getLogger("amen_analysis")
    logger.info("Starting Amen break analysis pipeline")
    
    # Use simple hardcoded paths
    
    # Input audio file to analyze
    input_file_path = "../public/QO_HoofbeatsMusic.wav"
    
    # Path to the Amen break sample file (MUST BE PROVIDED)
    amen_break_sample_path = "../public/amen_break.wav"
    
    # Configuration file path
    config_path = "../configs/default.yaml"
    
    # Validate paths
    if not os.path.exists(input_file_path):
        logger.error(f"Input file not found: {input_file_path}")
        return None
        
    if not os.path.exists(amen_break_sample_path):
        logger.error(f"Amen break sample not found: {amen_break_sample_path}")
        return None
    
    # Initialize pipeline
    logger.info(f"Initializing pipeline with config: {config_path}")
    pipeline = Pipeline(config_path=config_path)
    
    # IMPORTANT: Ensure the sequence aligner is properly initialized with the Amen break sample
    from alignment.alignment import SequenceAligner
    
    if not pipeline._sequence_aligner:
        logger.info(f"Initializing sequence aligner with Amen break sample: {amen_break_sample_path}")
        pipeline._sequence_aligner = SequenceAligner()
        
        # Load the Amen break sample explicitly
        amen_break_audio = pipeline.audio_processor.load(amen_break_sample_path)
        
        # Set up template directly if needed
        if hasattr(pipeline._sequence_aligner, 'amen_template') and amen_break_audio is not None:
            # Template already exists by default, so we just need to verify it
            logger.info(f"Amen break template initialized: {len(amen_break_audio)} samples")
        
        # Verify template was loaded
        if not hasattr(pipeline._sequence_aligner, 'amen_template') or pipeline._sequence_aligner.amen_template is None:
            logger.error("Failed to initialize Amen break template - check the sample file")
            return None
    
    # Process the input file
    logger.info(f"Processing input file: {input_file_path}")
    results = pipeline.process(input_file_path)
    
    # Process and display results
    if results and 'alignment' in results and 'sequence' in results['alignment']:
        amen_results = results['alignment']['sequence']
        
        if amen_results.get('segments'):
            logger.info(f"\n{'='*50}")
            logger.info(f"AMEN BREAK ANALYSIS RESULTS")
            logger.info(f"{'='*50}")
            
            logger.info(f"Found {len(amen_results['segments'])} Amen break patterns")
            
            for i, segment in enumerate(amen_results['segments']):
                logger.info(f"\nAmen break segment #{i+1}:")
                logger.info(f"  Time range: {segment['start']:.2f}s - {segment['end']:.2f}s (duration: {segment['end'] - segment['start']:.2f}s)")
                
                if 'variation_names' in amen_results:
                    logger.info(f"  Variation: {amen_results['variation_names'][i]}")
                
                logger.info(f"  Confidence score: {amen_results['confidence'][i]:.2f}")
                logger.info(f"  Tempo scaling: {amen_results['tempo_scales'][i]:.2f}x")
            
            # Show where to find visualization results
            if 'visualizations' in results and 'amen_break' in results['visualizations']:
                logger.info(f"\nAmen break visualization available at:")
                logger.info(f"  {results['visualizations']['amen_break']}")
        else:
            logger.info("No Amen break patterns detected in the audio file")
    else:
        logger.info("Alignment analysis not available - check sequence aligner configuration")
    
    logger.info(f"\nComplete analysis results saved to: {pipeline.config.get('storage', 'results_dir', 'results')}")
    return results


if __name__ == "__main__":
    main()

