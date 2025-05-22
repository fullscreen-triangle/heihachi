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
import json
from pathlib import Path
import time
import psutil  # Add missing import for psutil

# Add project root to sys.path to ensure modules can be found
import sys

from src.alignment import SequenceAligner, PriorSubspaceAnalysis, SimilarityAnalyzer, CompositeSimilarity
from src.annotation.peak_detection import PeakDetector
from src.annotation.segment_clustering import SegmentClusterer
from src.annotation.transition_detector import TransitionDetector
# Import directly from modules to avoid circular imports
from src.core.mix_analyzer import MixAnalyzer
from src.core.audio_processing import AudioProcessor
from src.core.audio_scene import AudioSceneAnalyzer
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
from src.feature_extraction import feature_extractor
from src.visualization import visualizer
from src.utils.config import ConfigManager
from src.utils.cache import result_cache
from src.huggingface import (
    HuggingFaceAudioAnalyzer, FeatureExtractor, StemSeparator, BeatDetector,
    DrumSoundAnalyzer, ZeroShotTagger, AudioCaptioner, RealTimeBeatTracker
)

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
        config_manager = ConfigManager(config_path)
        self.config = config_manager.config
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
        
        # Initialize HuggingFace analyzer if enabled
        self.huggingface_enabled = self.config.get("huggingface.enabled", False)
        self._init_huggingface()
        
        # Initialize other components as needed - lazy loading
        self._components = {}
        logger.debug("Component dictionary initialized")
        
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
    
    def _init_huggingface(self):
        """Initialize HuggingFace integration."""
        hf_config = self.config.get("huggingface", {})
        api_key = hf_config.get("api_key")
        
        if api_key:
            logger.info("Initializing HuggingFace integration with API key")
        else:
            logger.info("Initializing HuggingFace integration without API key")
        
        # Initialize base analyzer
        self.hf_analyzer = HuggingFaceAudioAnalyzer(
            model_name=hf_config.get("model", "facebook/wav2vec2-base-960h"),
            api_key=api_key,
            genre_model=hf_config.get("genre_model"),
            instrument_model=hf_config.get("instrument_model"),
            config=hf_config
        )
        
        # Initialize specialized components as needed
        self.specialized_models = {}
        
        # Initialize specialized components based on config
        if hf_config.get("use_specialized_models", True):
            self._init_specialized_models(hf_config)
    
    def _init_specialized_models(self, hf_config: Dict):
        """Initialize specialized HuggingFace models for specific tasks.
        
        Args:
            hf_config: HuggingFace configuration section
        """
        api_key = hf_config.get("api_key")
        use_cuda = hf_config.get("use_cuda", True)
        device = hf_config.get("device", None)
        
        # Common initialization parameters
        init_params = {
            "api_key": api_key,
            "use_cuda": use_cuda,
            "device": device,
            "config": hf_config
        }
        
        # Feature extraction
        if hf_config.get("feature_extraction.enabled", True):
            model_name = hf_config.get("feature_extraction.model", "microsoft/BEATs-base")
            logger.info(f"Initializing feature extractor: {model_name}")
            self.specialized_models["feature_extractor"] = FeatureExtractor(
                model=model_name, **init_params
            )
        
        # Stem separation
        if hf_config.get("stem_separation.enabled", False):
            model_name = hf_config.get("stem_separation.model", "htdemucs")
            logger.info(f"Initializing stem separator: {model_name}")
            self.specialized_models["stem_separator"] = StemSeparator(
                model_name=model_name, **init_params
            )
        
        # Beat detection
        if hf_config.get("beat_detection.enabled", True):
            model_name = hf_config.get("beat_detection.model", "amaai/music-tempo-beats")
            logger.info(f"Initializing beat detector: {model_name}")
            self.specialized_models["beat_detector"] = BeatDetector(
                model_name=model_name, **init_params
            )
        
        # Real-time beat tracking
        if hf_config.get("realtime_beats.enabled", False):
            model_name = hf_config.get("realtime_beats.model", "beast-team/beast-dione")
            logger.info(f"Initializing real-time beat tracker: {model_name}")
            self.specialized_models["realtime_beat_tracker"] = RealTimeBeatTracker(
                model_name=model_name, **init_params
            )
        
        # Drum analysis
        if hf_config.get("drum_analysis.enabled", False):
            model_name = hf_config.get("drum_analysis.model", "DunnBC22/wav2vec2-base-Drum_Kit_Sounds")
            logger.info(f"Initializing drum analyzer: {model_name}")
            self.specialized_models["drum_analyzer"] = DrumAnalyzer(
                model_name=model_name, **init_params
            )
        
        # Specialized drum sound analysis
        if hf_config.get("drum_sound_analysis.enabled", False):
            model_name = hf_config.get("drum_sound_analysis.model", "JackArt/wav2vec2-for-drum-classification")
            logger.info(f"Initializing drum sound analyzer: {model_name}")
            self.specialized_models["drum_sound_analyzer"] = DrumSoundAnalyzer(
                model_name=model_name, **init_params
            )
        
        # Audio-text similarity
        if hf_config.get("similarity.enabled", False):
            model_name = hf_config.get("similarity.model", "laion/clap-htsat-fused")
            logger.info(f"Initializing similarity analyzer: {model_name}")
            self.specialized_models["similarity_analyzer"] = SimilarityAnalyzer(
                model_name=model_name, **init_params
            )
        
        # Zero-shot tagging
        if hf_config.get("tagging.enabled", False):
            model_name = hf_config.get("tagging.model", "UniMus/OpenJMLA")
            logger.info(f"Initializing zero-shot tagger: {model_name}")
            self.specialized_models["zero_shot_tagger"] = ZeroShotTagger(
                model_name=model_name, **init_params
            )
        
        # Audio captioning
        if hf_config.get("captioning.enabled", False):
            model_name = hf_config.get("captioning.model", "slseanwu/beats-conformer-bart-audio-captioner")
            logger.info(f"Initializing audio captioner: {model_name}")
            self.specialized_models["audio_captioner"] = AudioCaptioner(
                model_name=model_name, **init_params
            )
    
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
        """Process audio file in memory-optimized stages.
        
        Args:
            audio_path: Path to the audio file
            memory_monitor: Memory monitoring object
            
        Returns:
            Dict: Analysis results
        """
        results = {}
        
        # Stage 1: Load audio
        logger.info("Stage 1: Loading audio")
        audio, sample_rate = self.audio_processor.load(audio_path)
        
        # Add basic file info to results
        results['metadata'] = {
            'filename': os.path.basename(audio_path),
            'duration': len(audio) / sample_rate,
            'sample_rate': sample_rate,
            'channels': 1 if audio.ndim == 1 else audio.shape[1]
        }
        
        # Stage 2: Feature extraction and analysis
        logger.info("Stage 2: Extracting features")
        analysis_results = self._parallel_process(audio)
        results.update(analysis_results)
        
        # Stage 3: HuggingFace analysis if enabled
        if self.huggingface_enabled:
            logger.info("Stage 3: Running HuggingFace audio analysis")
            try:
                hf_results = self._run_huggingface_analysis(audio, sample_rate)
                results['huggingface'] = hf_results
                logger.debug("HuggingFace analysis complete")
            except Exception as e:
                logger.error(f"HuggingFace analysis failed: {str(e)}", exc_info=True)
                results['huggingface'] = {'error': str(e)}
        
        # Stage 4: Visualization
        logger.info("Stage 4: Generating visualizations")
        if self.config.get('visualization', {}).get('enable', True):
            try:
                visualization_dir = self.config.get('storage', {}).get('visualizations_dir', "../visualizations")
                vis_manager = self._get_or_initialize("visualization_manager", lambda: VisualizationManager(visualization_dir))
                
                if vis_manager:
                    # Generate base filename from audio path
                    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
                    results['visualizations'] = vis_manager.generate_all(audio, results, base_filename)
                    logger.debug(f"Visualizations saved to {visualization_dir}")
            except Exception as e:
                logger.error(f"Visualization generation failed: {str(e)}", exc_info=True)
                results['visualizations'] = {'error': str(e)}
        
        # Stage 5: Report generation
        logger.info("Stage 5: Creating analysis summary report")
        results['report'] = self.mix_visualizer.create_summary_report(results)
        logger.debug("Summary report created")
        
        # Stage 6: Storage
        logger.info("Stage 6: Storing results")
        
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
    
    @profile
    def process_with_huggingface(self, file_path: str) -> Dict[str, Any]:
        """Process audio with HuggingFace models.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with HuggingFace analysis results
        """
        try:
            result = {}
            
            # Process with base analyzer if available
            if hasattr(self, 'hf_analyzer') and self.hf_analyzer is not None:
                # Extract features, classify genre, detect instruments
                base_results = self._analyze_with_base_model(file_path)
                if base_results:
                    result.update(base_results)
            
            # Process with specialized components if available
            specialized_results = self._analyze_with_specialized_models(file_path)
            if specialized_results:
                result.update(specialized_results)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in HuggingFace processing: {str(e)}")
            import traceback
            logger.debug(traceback.format_exc())
            return {"error": str(e)}
    
    def _analyze_with_base_model(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio with the base HuggingFace model.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with base model analysis results
        """
        import librosa
        import numpy as np
        
        result = {}
        
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=None)
            
            # Basic analysis
            basic_result = self.hf_analyzer.analyze(audio, sr)
            if 'error' not in basic_result:
                result["basic"] = basic_result
            
            # Genre classification
            if self.config.get("huggingface.classify_genre", True):
                genre_result = self.hf_analyzer.classify_genre(audio, sr)
                result["genre"] = genre_result
            
            # Instrument detection
            if self.config.get("huggingface.detect_instruments", True):
                instrument_result = self.hf_analyzer.detect_instruments(audio, sr)
                result["instruments"] = instrument_result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in base model analysis: {str(e)}")
            return {"base_error": str(e)}
    
    def _analyze_with_specialized_models(self, file_path: str) -> Dict[str, Any]:
        """Analyze audio with specialized HuggingFace models.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with specialized model analysis results
        """
        result = {}
        
        # Feature extraction
        if "feature_extractor" in self.specialized_models and self.config.get("huggingface.feature_extraction.apply", True):
            try:
                extractor = self.specialized_models["feature_extractor"]
                features = extractor.extract(file_path)
                result["extracted_features"] = features
            except Exception as e:
                logger.error(f"Feature extraction error: {str(e)}")
                result["extracted_features"] = {"error": str(e)}
        
        # Beat detection
        if "beat_detector" in self.specialized_models and self.config.get("huggingface.beat_detection.apply", True):
            try:
                detector = self.specialized_models["beat_detector"]
                beats = detector.detect(file_path)
                result["beats"] = beats
            except Exception as e:
                logger.error(f"Beat detection error: {str(e)}")
                result["beats"] = {"error": str(e)}
        
        # Stem separation - only if explicitly enabled due to high resource usage
        if "stem_separator" in self.specialized_models and self.config.get("huggingface.stem_separation.apply", False):
            try:
                separator = self.specialized_models["stem_separator"]
                output_dir = self.config.get("huggingface.stem_separation.output_dir")
                if not output_dir:
                    # Create default output directory based on input file
                    file_name = os.path.basename(file_path)
                    base_name = os.path.splitext(file_name)[0]
                    output_dir = os.path.join(os.path.dirname(file_path), f"{base_name}_stems")
                
                stems = separator.separate(file_path, output_dir=output_dir)
                result["stems"] = {"stems": list(stems.keys()), "output_dir": output_dir}
            except Exception as e:
                logger.error(f"Stem separation error: {str(e)}")
                result["stems"] = {"error": str(e)}
        
        # Drum analysis
        if "drum_analyzer" in self.specialized_models and self.config.get("huggingface.drum_analysis.apply", False):
            try:
                analyzer = self.specialized_models["drum_analyzer"]
                drum_analysis = analyzer.analyze(file_path)
                result["drum_analysis"] = drum_analysis
            except Exception as e:
                logger.error(f"Drum analysis error: {str(e)}")
                result["drum_analysis"] = {"error": str(e)}
        
        # Specialized drum sound analysis
        if "drum_sound_analyzer" in self.specialized_models and self.config.get("huggingface.drum_sound_analysis.apply", False):
            try:
                analyzer = self.specialized_models["drum_sound_analyzer"]
                drum_hits = analyzer.detect_drum_hits(file_path)
                pattern = analyzer.create_drum_pattern(drum_hits)
                
                result["drum_patterns"] = {
                    "hits": len(drum_hits),
                    "pattern": pattern,
                }
            except Exception as e:
                logger.error(f"Drum pattern analysis error: {str(e)}")
                result["drum_patterns"] = {"error": str(e)}
        
        # Audio-text similarity
        if "similarity_analyzer" in self.specialized_models and self.config.get("huggingface.similarity.apply", False):
            try:
                analyzer = self.specialized_models["similarity_analyzer"]
                default_queries = ["electronic music", "ambient", "techno", "drums", "synthesizer", "bass", "melody"]
                
                # Get custom queries from config if available
                queries = self.config.get("huggingface.similarity.queries", default_queries)
                
                similarity = analyzer.match_text_to_audio(audio_path=file_path, text_queries=queries)
                result["similarity"] = similarity
            except Exception as e:
                logger.error(f"Similarity analysis error: {str(e)}")
                result["similarity"] = {"error": str(e)}
        
        # Zero-shot tagging
        if "zero_shot_tagger" in self.specialized_models and self.config.get("huggingface.tagging.apply", False):
            try:
                tagger = self.specialized_models["zero_shot_tagger"]
                tags = tagger.tag(audio_path=file_path)
                result["tags"] = tags
            except Exception as e:
                logger.error(f"Zero-shot tagging error: {str(e)}")
                result["tags"] = {"error": str(e)}
        
        # Audio captioning
        if "audio_captioner" in self.specialized_models and self.config.get("huggingface.captioning.apply", False):
            try:
                captioner = self.specialized_models["audio_captioner"]
                caption = captioner.caption(audio_path=file_path)
                result["caption"] = caption
            except Exception as e:
                logger.error(f"Audio captioning error: {str(e)}")
                result["caption"] = {"error": str(e)}
        
        # Real-time beat tracking (if applied to file)
        if "realtime_beat_tracker" in self.specialized_models and self.config.get("huggingface.realtime_beats.apply", False):
            try:
                tracker = self.specialized_models["realtime_beat_tracker"]
                beats = tracker.process_file(audio_path=file_path)
                result["realtime_beats"] = beats
            except Exception as e:
                logger.error(f"Real-time beat tracking error: {str(e)}")
                result["realtime_beats"] = {"error": str(e)}
        
        return result

    def batch_process(self, input_dir: str, output_dir: Optional[str] = None, 
                     extensions: List[str] = None) -> Dict[str, Dict]:
        """Process all audio files in a directory.
        
        Args:
            input_dir: Input directory containing audio files
            output_dir: Output directory for results (default: None)
            extensions: List of file extensions to process (default: None)
            
        Returns:
            Dictionary mapping file paths to processing results
        """
        # Default extensions if None provided
        if extensions is None:
            extensions = ['wav', 'mp3', 'flac', 'm4a', 'ogg']
            
        # Normalize extensions
        extensions = [ext.lower().strip('.') for ext in extensions]
        
        # Ensure input directory exists
        if not os.path.isdir(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return {"error": f"Input directory not found: {input_dir}"}
        
        # Create output directory if provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all matching audio files
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(f'.{ext}') for ext in extensions):
                    audio_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        # Process each file
        results = {}
        for i, file_path in enumerate(audio_files):
            logger.info(f"Processing file {i+1}/{len(audio_files)}: {file_path}")
            
            # Process file
            result = self.process_file(file_path)
            results[file_path] = result
            
            # Save result to output directory if provided
            if output_dir and result.get("status") == "success":
                rel_path = os.path.relpath(file_path, input_dir)
                output_path = os.path.join(output_dir, f"{rel_path}.json")
                
                # Create parent directories if needed
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save result
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Result saved to: {output_path}")
        
        return results


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

