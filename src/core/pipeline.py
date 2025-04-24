import torch
import numpy as np
import gc
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import logging
import os
import weakref
import os
import logging
import yaml
from typing import Dict, Any, Optional

# Add the project root to sys.path to ensure modules can be found
import sys

from src.alignment import SequenceAligner, PriorSubspaceAnalysis, SimilarityAnalyzer, CompositeSimilarity
from src.annotation.peak_detection import PeakDetector
from src.annotation.segment_clustering import SegmentClusterer
from src.annotation.transition_detector import TransitionDetector
from src.core import MixAnalyzer, AudioProcessor, AudioSceneAnalyzer
from src.feature_extraction.baseline_analysis import BasslineAnalyzer
from src.feature_extraction.bpm_analysis import BPMAnalyzer
from src.feature_extraction.drum_analysis import DrumAnalyzer
from src.feature_extraction.groove_analysis import GrooveAnalyzer
from src.feature_extraction.percussion_analysis import PercussionAnalyzer
from src.feature_extraction.rhythmic_analysis import RhythmAnalyzer
from src.utils.logging_utils import start_memory_monitoring, setup_logging
from src.utils.storage import AnalysisVersion, AudioCache, FeatureStorage
from src.utils.visualization import MixVisualizer, AnalysisVisualizer
from src.utils.profiling import global_profiler, profile

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set up logger
logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main processing pipeline for Heihachi audio analysis.
    
    This class orchestrates the complete audio analysis process, from loading
    audio files to generating final results.
    """
    
    def __init__(self, config_path: str = "../configs/default.yaml"):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Set up device for GPU acceleration if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set up number of workers for parallel processing - configurable now
        self.config = self._load_config(config_path)
        self.num_workers = self.config.get('processing', {}).get('num_workers', min(mp.cpu_count(), 4))
        logger.info(f"Using {self.num_workers} workers for parallel processing")
        
        logger.debug("Configuration loaded successfully")
        
        # Get paths from config if available
        cache_dir = self.config.get('storage', {}).get('cache_dir', "../cache")
        results_dir = self.config.get('storage', {}).get('results_dir', "../results")
        visualizations_dir = self.config.get('storage', {}).get('visualizations_dir', "../visualizations")
        
        # Configure memory management
        self.memory_limit_mb = self.config.get('processing', {}).get('memory_limit_mb', 1024)
        logger.info(f"Memory usage limit set to {self.memory_limit_mb} MB")
        
        # Initialize storage
        self.cache = AudioCache(cache_dir, compression_level=self.config.get('storage', {}).get('compression_level', 6))
        self.storage = FeatureStorage(results_dir)
        logger.debug(f"Storage initialized with cache dir: {cache_dir}, results dir: {results_dir}")
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(config_path)
        
        # Only initialize essential components here
        self.mix_analyzer = MixAnalyzer(use_gpu=(self.device.type == 'cuda'))
        
        # Hold weak references to other components to be initialized on demand
        self._component_references = {}
        
        # Initialize visualizers
        self._init_visualizers(visualizations_dir)
        
        logger.info("Pipeline initialization complete")
    
    def _init_visualizers(self, visualizations_dir):
        """Initialize visualization components (separated for better memory management)."""
        self.mix_visualizer = MixVisualizer(visualizations_dir)
        self.analysis_visualizer = AnalysisVisualizer(visualizations_dir)
        logger.debug(f"Visualizers initialized with output dir: {visualizations_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            if os.path.exists(config_path):
                logger.debug(f"Loading configuration from: {config_path}")
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file {config_path} not found, using default configuration")
                return self._default_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            logger.warning("Using default configuration instead")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        logger.debug("Creating default configuration dictionary")
        return {
            "general": {
                "sample_rate": 44100,
                "channels": 2
            },
            "feature_extraction": {
                "enable": True
            },
            "analysis": {
                "enable": True
            },
            "visualization": {
                "enable": True
            },
            "processing": {
                "num_workers": min(mp.cpu_count(), 4),
                "memory_limit_mb": 1024,
                "chunk_size": 4096,
                "batch_size": 16
            },
            "storage": {
                "cache_dir": "../cache",
                "results_dir": "../results",
                "visualizations_dir": "../visualizations",
                "compression_level": 6
            }
        }

    def _initialize_needed_components(self, audio_length: int) -> None:
        """Initialize analysis components on demand to save memory.
        
        Args:
            audio_length: Length of audio data for configuring components
        """
        logger.info("Initializing pipeline components")
        
        # Initialize components with configurations optimized for audio length
        self._get_or_initialize("bpm_analyzer", lambda: BPMAnalyzer())
        
        # For longer audio files, use more optimized analyzer instances
        if audio_length > 44100 * 60 * 10:  # More than 10 minutes
            logger.info("Long audio detected, using memory-optimized analyzers")
            self._get_or_initialize("sequence_aligner", lambda: SequenceAligner(use_reduced_memory=True))
            self._get_or_initialize("similarity_analyzer", lambda: SimilarityAnalyzer(use_reduced_memory=True))
        else:
            self._get_or_initialize("sequence_aligner", lambda: SequenceAligner())
            self._get_or_initialize("similarity_analyzer", lambda: SimilarityAnalyzer())
        
        # Initialize other components as needed
        self._get_or_initialize("prior_subspace", lambda: PriorSubspaceAnalysis(use_gpu=(self.device.type == 'cuda')))
        self._get_or_initialize("composite_similarity", lambda: CompositeSimilarity())
        self._get_or_initialize("baseline_analyzer", lambda: BasslineAnalyzer())
        self._get_or_initialize("drum_analyzer", lambda: DrumAnalyzer())
        self._get_or_initialize("groove_analyzer", lambda: GrooveAnalyzer())
        self._get_or_initialize("percussion_analyzer", lambda: PercussionAnalyzer())
        self._get_or_initialize("rhythmic_analyzer", lambda: RhythmAnalyzer())
        self._get_or_initialize("scene_analyzer", lambda: AudioSceneAnalyzer())
        self._get_or_initialize("peak_detector", lambda: PeakDetector())
        self._get_or_initialize("segment_clusterer", lambda: SegmentClusterer())
        self._get_or_initialize("transition_detector", lambda: TransitionDetector())
        
        logger.debug("Pipeline components initialization complete")
    
    def _get_or_initialize(self, component_name: str, initializer) -> Any:
        """Get existing component or initialize it if it doesn't exist.
        
        Args:
            component_name: Name of the component
            initializer: Function to initialize the component if it doesn't exist
            
        Returns:
            Component instance
        """
        # Check if we have a weak reference and if it's still alive
        if component_name in self._component_references:
            component = self._component_references[component_name]()
            if component is not None:
                return component
        
        # Initialize and store weak reference to allow GC
        logger.debug(f"Initializing {component_name}")
        component = initializer()
        self._component_references[component_name] = weakref.ref(component)
        setattr(self, f"_{component_name}", component)
        return component
    
    def _release_components(self, keep: Optional[List[str]] = None) -> None:
        """Release components to free memory.
        
        Args:
            keep: List of component names to keep
        """
        keep = keep or []
        logger.debug(f"Releasing components, keeping {keep}")
        
        for component_name in list(self._component_references.keys()):
            if component_name not in keep:
                if hasattr(self, f"_{component_name}"):
                    logger.debug(f"Releasing {component_name}")
                    delattr(self, f"_{component_name}")
                    self._component_references.pop(component_name, None)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Component release and garbage collection completed")

    @profile("process")
    def process(self, audio_path: str) -> Dict:
        """Main pipeline processing with memory optimization.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dict: Analysis results
        """
        # Start memory monitoring
        logger.info(f"Starting processing for audio file: {audio_path}")
        memory_monitor = start_memory_monitoring(interval=2.0)
        logger.debug("Memory monitoring started")
        
        try:
            # Try loading from cache first
            logger.debug(f"Checking cache for: {audio_path}")
            cached_result = self.cache.get(audio_path)
            if cached_result:
                logger.info(f"Found cached results for {audio_path}")
                memory_monitor.stop()
                logger.debug("Memory monitoring stopped - using cached results")
                return cached_result

            # Get file size
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            logger.info(f"Processing file of size: {file_size_mb:.2f} MB")

            # Load and preprocess audio using stage-by-stage processing
            return self._process_in_stages(audio_path, memory_monitor)

        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}", exc_info=True)
            if memory_monitor:
                memory_monitor.stop()
                logger.debug("Memory monitoring stopped due to error")
            raise
    
    def _process_in_stages(self, audio_path: str, memory_monitor) -> Dict:
        """Process audio file in stages to optimize memory usage.
        
        Args:
            audio_path: Path to the audio file
            memory_monitor: Memory monitoring thread
            
        Returns:
            Dict: Analysis results
        """
        results = {'features': {}, 'analysis': {}, 'annotation': {}, 'visualizations': {}}
        
        # Stage 1: Audio loading
        logger.info("Stage 1: Loading and preprocessing audio file")
        audio = self.audio_processor.load(audio_path)
        logger.debug(f"Audio loaded: {len(audio)} samples, shape: {audio.shape}")
        
        # Free up memory after loading if it's a large file
        if len(audio) > 44100 * 60 * 5:  # More than 5 minutes
            logger.debug("Large audio file, performing initial GC")
            gc.collect()
        
        # Stage 2: Feature extraction and analysis
        logger.info("Stage 2: Performing parallel feature extraction and analysis")
        process_results = self._parallel_process(audio)
        results.update(process_results)
        
        # Free memory after processing (audio data no longer needed)
        del audio
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Audio data released and garbage collection performed")
        
        # Stage 3: Visualization generation
        if self.config.get('visualization', {}).get('enable', True):
            logger.info("Stage 3: Generating visualizations")
            results['visualizations'] = self._generate_visualizations(results)
            logger.debug(f"Generated {len(results['visualizations'])} visualizations")
        
        # Stage 4: Report generation
        logger.info("Stage 4: Creating analysis summary report")
        results['report'] = self.mix_visualizer.create_summary_report(results)
        logger.debug("Summary report created")
        
        # Stage 5: Storage
        logger.info("Stage 5: Storing results")
        
        # Cache results
        self.cache.set(audio_path, results)
        logger.debug("Results cached successfully")
        
        # Save to persistent storage
        self.storage.save_results(
            audio_path,
            results,
            metadata={'version': AnalysisVersion.to_string()}
        )
        logger.debug(f"Results saved to {self.storage.storage_dir}")
        
        # Final cleanup
        memory_monitor.stop()
        logger.debug("Memory monitoring stopped")
        logger.info(f"Processing complete for: {audio_path}")
        
        return results

    @profile("parallel_process")
    def _parallel_process(self, audio: np.ndarray) -> Dict:
        """Process audio in parallel to extract features and perform analysis.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Dict: Process results containing features and analysis data
        """
        logger.debug("Starting parallel processing of audio data")
        
        # Initialize needed components based on audio length
        logger.debug(f"Determining optimal components for audio length: {len(audio)}")
        self._initialize_needed_components(len(audio))
        
        # Prepare results dictionary
        results = {'features': {}, 'analysis': {}, 'annotation': {}}
        
        # Determine batch size based on audio length and available memory
        batch_size = self._calculate_optimal_batch_size(audio)
        
        # Process in parallel with thread pool
        logger.debug(f"Creating thread pool with {self.num_workers} workers")
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Extract basic features first
            logger.info("Extracting basic features")
            feature_futures = []
            
            # Basic BPM analysis - always needed
            bpm_analyzer = self._get_or_initialize("bpm_analyzer", lambda: BPMAnalyzer())
            if bpm_analyzer:
                logger.debug("Submitting BPM analysis")
                feature_futures.append(executor.submit(bpm_analyzer.analyze, audio))
            
            # Basic mix analysis - critical component, submitted first
            logger.debug("Submitting mix analysis")
            mix_future = executor.submit(self.mix_analyzer.analyze, audio)
            
            # Add conditional analyses based on available components and memory
            logger.debug("Adding conditional analyses")
            self._add_conditional_analyses(feature_futures, executor, audio, batch_size)
            
            # Wait for mix analysis to complete first (it's often needed by other components)
            logger.debug("Waiting for mix analysis to complete")
            try:
                results['analysis']['mix'] = mix_future.result()
                logger.debug("Mix analysis complete")
            except Exception as e:
                logger.error(f"Mix analysis failed: {str(e)}", exc_info=True)
                results['analysis']['mix'] = {'error': str(e)}
            
            # Process feature futures as they complete to avoid holding all in memory
            for future_idx, future in enumerate(feature_futures):
                try:
                    future_result = future.result()
                    if future_result:
                        # If result is a dictionary with multiple features, merge them all
                        if isinstance(future_result, dict) and any(isinstance(value, dict) for value in future_result.values()):
                            for key, value in future_result.items():
                                if key == 'similarity' or key == 'composite':
                                    results['features'][key] = value
                                else:
                                    # Handle nested dictionaries
                                    self._merge_results(results, key, value)
                        else:
                            # Add simple result to the appropriate category
                            category = self._determine_result_category(future_idx, future_result)
                            results[category][f"feature_{future_idx}"] = future_result
                    
                    # Release memory after each future if needed
                    if future_idx % 3 == 0:  # Every 3 futures
                        gc.collect()
                        if torch.cuda.is_available() and self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Feature extraction failed for future {future_idx}: {str(e)}", exc_info=True)
                    results['features'][f"error_{future_idx}"] = str(e)
        
        # Process annotation components (they use features from previous steps)
        self._process_annotations(audio, results)
        
        # Process alignment results if available
        if any(key in results.get('features', {}) for key in ['sequence_alignment', 'similarity']):
            alignment_data = self._process_alignment_results(results.get('features', {}))
            if alignment_data:
                results['analysis']['alignment'] = alignment_data
                logger.debug("Alignment analysis complete")
        
        # Release components not needed for later stages
        self._release_components(['bpm_analyzer'])  # Keep only essential analyzers
        
        logger.info("Parallel processing complete")
        return results
    
    def _calculate_optimal_batch_size(self, audio: np.ndarray) -> int:
        """Calculate optimal batch size based on audio length and available memory.
        
        Args:
            audio: Audio data
            
        Returns:
            Optimal batch size
        """
        default_batch_size = self.config.get('processing', {}).get('batch_size', 16)
        
        # For very short audio, process all at once
        if len(audio) < 44100 * 30:  # Less than 30 seconds
            return 1
            
        # For very long audio, use smaller batches
        if len(audio) > 44100 * 60 * 20:  # More than 20 minutes
            return max(1, default_batch_size // 4)
            
        # For long audio, use reduced batch size
        if len(audio) > 44100 * 60 * 5:  # More than 5 minutes
            return max(1, default_batch_size // 2)
            
        return default_batch_size
    
    def _merge_results(self, results: Dict, key: str, value: Any) -> None:
        """Merge results into the appropriate category.
        
        Args:
            results: Results dictionary
            key: Result key
            value: Result value
        """
        # Determine category based on key name
        if any(k in key for k in ['bpm', 'tempo', 'beat', 'rhythm', 'spectral']):
            results['features'][key] = value
        elif any(k in key for k in ['segment', 'peak', 'transition']):
            results['annotation'][key] = value
        else:
            results['analysis'][key] = value
    
    def _determine_result_category(self, future_idx: int, result: Any) -> str:
        """Determine which category a result belongs to.
        
        Args:
            future_idx: Index of the future
            result: Future result
            
        Returns:
            Category name
        """
        # Simple heuristic based on result data - adjust as needed
        if isinstance(result, dict) and any(k in result for k in ['frequency', 'spectrogram', 'mfcc']):
            return 'features'
        elif isinstance(result, dict) and any(k in result for k in ['segments', 'peaks', 'transitions']):
            return 'annotation'
        else:
            return 'analysis'
    
    def _add_conditional_analyses(self, feature_futures, executor, audio, batch_size) -> None:
        """Add conditional analyses based on available components and memory constraints.
        
        Args:
            feature_futures: List of futures
            executor: ThreadPoolExecutor
            audio: Audio data
            batch_size: Batch size for processing
        """
        # Check memory before adding tasks
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / (1024 * 1024)
        
        # If memory usage is already high, limit additional components
        if current_memory > self.memory_limit_mb * 0.7:
            logger.warning(f"Memory usage high ({current_memory:.2f}MB), limiting additional analysis components")
            return
        
        # Add alignment analysis if sequence aligner is available
        sequence_aligner = self._get_or_initialize("sequence_aligner", lambda: None)
        if sequence_aligner:
            feature_futures.append(
                executor.submit(self._process_in_batches, sequence_aligner.align, audio, batch_size)
            )
            
        # Add subspace analysis if prior subspace analyzer is available
        prior_subspace = self._get_or_initialize("prior_subspace", lambda: None)
        if prior_subspace:
            feature_futures.append(
                executor.submit(prior_subspace.analyze, audio)
            )
            
        # Add similarity analysis if both components are available
        similarity_analyzer = self._get_or_initialize("similarity_analyzer", lambda: None)
        composite_similarity = self._get_or_initialize("composite_similarity", lambda: None)
        if similarity_analyzer and composite_similarity:
            feature_futures.append(
                executor.submit(
                    lambda: {
                        'similarity': self._process_in_batches(similarity_analyzer.analyze, audio, batch_size),
                        'composite': composite_similarity.analyze(audio)
                    }
                )
            )
    
    def _process_in_batches(self, func, audio, batch_size):
        """Process audio data in batches to minimize memory usage.
        
        Args:
            func: Function to apply to each batch
            audio: Audio data
            batch_size: Batch size
            
        Returns:
            Combined results
        """
        if batch_size <= 1:
            return func(audio)
            
        # Split audio into batches
        audio_length = len(audio)
        batch_size_samples = audio_length // batch_size
        
        results = []
        for i in range(batch_size):
            start = i * batch_size_samples
            end = start + batch_size_samples if i < batch_size - 1 else audio_length
            
            batch = audio[start:end]
            try:
                batch_result = func(batch)
                results.append(batch_result)
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
                
            # Clean up after each batch
            del batch
            gc.collect()
            
        # Combine results - this is a simple concatenation, but might need
        # more complex logic depending on the function
        return self._combine_batch_results(results)
    
    def _combine_batch_results(self, results):
        """Combine results from batch processing.
        
        Args:
            results: List of batch results
            
        Returns:
            Combined results
        """
        if not results:
            return None
            
        # If results are numpy arrays, concatenate them
        if all(isinstance(r, np.ndarray) for r in results):
            return np.concatenate(results)
            
        # If results are dictionaries, merge them
        if all(isinstance(r, dict) for r in results):
            combined = {}
            for r in results:
                for k, v in r.items():
                    if k not in combined:
                        combined[k] = v
                    else:
                        # If values are arrays, concatenate them
                        if isinstance(v, np.ndarray) and isinstance(combined[k], np.ndarray):
                            combined[k] = np.concatenate([combined[k], v])
                        # If values are lists, extend them
                        elif isinstance(v, list) and isinstance(combined[k], list):
                            combined[k].extend(v)
                        # If values are numbers, average them
                        elif isinstance(v, (int, float)) and isinstance(combined[k], (int, float)):
                            combined[k] = (combined[k] + v) / 2
                        # Otherwise, keep the first value
                        else:
                            pass
            return combined
            
        # If results are lists, flatten them
        if all(isinstance(r, list) for r in results):
            return [item for sublist in results for item in sublist]
            
        # If we can't combine results, return the first one
        return results[0]
    
    def _process_annotations(self, audio, results):
        """Process annotation components using extracted features.
        
        Args:
            audio: Audio data
            results: Results dictionary to update
        """
        try:
            peak_detector = self._get_or_initialize("peak_detector", lambda: None)
            if peak_detector:
                logger.debug("Running peak detection")
                results['annotation']['peaks'] = peak_detector.detect(
                    audio, results.get('features', {}).get('bpm', {})
                )
                
            segment_clusterer = self._get_or_initialize("segment_clusterer", lambda: None)
            if segment_clusterer and 'peaks' in results['annotation']:
                logger.debug("Running segment clustering")
                results['annotation']['segments'] = segment_clusterer.cluster(
                    results['annotation']['peaks'], results.get('features', {})
                )
                
            transition_detector = self._get_or_initialize("transition_detector", lambda: None)
            if transition_detector:
                logger.debug("Running transition detection")
                results['annotation']['transitions'] = transition_detector.detect(
                    audio, results.get('features', {}), results.get('analysis', {})
                )
        except Exception as e:
            logger.error(f"Annotation processing failed: {str(e)}", exc_info=True)
            results['annotation']['error'] = str(e)
    
    def _process_alignment_results(self, results) -> Dict:
        """Process alignment results to extract meaningful features.
        
        Args:
            results: Analysis results
            
        Returns:
            Processed alignment data
        """
        alignment_data = {}
        
        # Extract key metrics from sequence alignment results
        if 'sequence_alignment' in results:
            seq_align = results['sequence_alignment']
            if isinstance(seq_align, dict):
                alignment_data['sequence'] = {
                    'similarity_score': seq_align.get('similarity', 0.0),
                    'aligned_segments': len(seq_align.get('segments', [])),
                    'match_confidence': seq_align.get('confidence', 0.0),
                    'pattern_length': seq_align.get('pattern_length', 0),
                }
                
                # Extract timing information
                timing = seq_align.get('timing', {})
                if timing:
                    alignment_data['timing'] = {
                        'average_deviation': timing.get('avg_deviation', 0.0),
                        'max_deviation': timing.get('max_deviation', 0.0),
                        'stability': timing.get('stability', 0.0),
                        'consistency': timing.get('consistency', 0.0),
                    }
        
        # Extract subspace analysis data
        if 'subspace_analysis' in results:
            subspace = results['subspace_analysis']
            if isinstance(subspace, dict):
                components = subspace.get('components', [])
                if components:
                    alignment_data['subspace'] = {
                        'component_count': len(components),
                        'explained_variance': subspace.get('explained_variance', 0.0),
                        'dominant_component': subspace.get('dominant_idx', 0),
                    }
                    
        return alignment_data
    
    def _generate_visualizations(self, results: Dict) -> Dict[str, str]:
        """Generate visualizations based on analysis results.
        
        Args:
            results: Analysis results
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        visualizations = {}
        
        try:
            # Only generate visualizations if enabled in config
            if not self.config.get('visualization', {}).get('enable', True):
                logger.info("Visualizations disabled in configuration")
                return visualizations
                
            # Generate mix visualizations
            if 'mix' in results.get('analysis', {}):
                logger.debug("Generating mix visualizations")
                mix_plots = self.mix_visualizer.visualize(results['analysis']['mix'])
                visualizations.update(mix_plots)
                
            # Generate feature visualizations if available
            if results.get('features'):
                logger.debug("Generating feature visualizations")
                feature_plots = self.analysis_visualizer.visualize_features(results['features'])
                visualizations.update(feature_plots)
                
            # Generate annotation visualizations if available
            if results.get('annotation'):
                logger.debug("Generating annotation visualizations")
                annotation_plots = self.analysis_visualizer.visualize_annotations(
                    results['annotation'], 
                    results.get('features', {}),
                    results.get('analysis', {})
                )
                visualizations.update(annotation_plots)
                
            # Clean up after visualization generation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"Generated {len(visualizations)} visualizations")
            return visualizations
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}", exc_info=True)
            visualizations['error'] = str(e)
            return visualizations


def main() -> None:
    """
    Entry point for the pipeline when run as a standalone script.
    
    This function initializes the pipeline with default configuration and
    processes a sample audio file to demonstrate the workflow.
    """
    import sys
    import os.path
    
    # Setup logging
    setup_logging()
    
    # If no argument is provided, show usage
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file_path>")
        return
    
    # Get file path from command line
    audio_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    
    # Initialize pipeline
    pipeline = Pipeline()
    
    # Process audio file
    print(f"Processing {audio_path}...")
    results = pipeline.process(audio_path)
    
    # Display success message
    print(f"Analysis complete. Results saved in {pipeline.storage.storage_dir}")
    print(f"Visualizations saved in {pipeline.analysis_visualizer.output_dir}")
    
    return


if __name__ == "__main__":
    main()

