import torch
import numpy as np
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

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
from utils.logging_utils import setup_logging
from utils.config import ConfigManager

logger = setup_logging({})

class Pipeline:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = ConfigManager(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = self.config.get('num_workers', mp.cpu_count())

        # Initialize storage
        self.cache = AudioCache(self.config.get('cache_dir', 'cache'))
        self.storage = FeatureStorage(self.config.get('storage_dir', 'results'))
        
        # Initialize visualizers
        self.mix_visualizer = MixVisualizer(self.config.get('output_dir', 'results'))
        self.analysis_visualizer = AnalysisVisualizer(self.config.get('output_dir', 'visualizations'))

        # Initialize all components
        self.sequence_aligner = SequenceAligner()
        self.prior_subspace = PriorSubspaceAnalysis()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.composite_similarity = CompositeSimilarity()

        self.baseline_analyzer = BasslineAnalyzer()
        self.bpm_analyzer = BPMAnalyzer()
        self.drum_analyzer = DrumAnalyzer()
        self.groove_analyzer = GrooveAnalyzer()
        self.percussion_analyzer = PercussionAnalyzer()
        self.rhythmic_analyzer = RhythmAnalyzer()

        self.mix_analyzer = MixAnalyzer()
        self.scene_analyzer = AudioSceneAnalyzer()
        self.audio_processor = AudioProcessor()

        self.peak_detector = PeakDetector()
        self.segment_clusterer = SegmentClusterer()
        self.transition_detector = TransitionDetector()

    def process(self, audio_path: str) -> Dict:
        """Main pipeline processing"""
        try:
            # Try loading from cache first
            cached_result = self.cache.get(audio_path)
            if cached_result:
                logger.info(f"Found cached results for {audio_path}")
                return cached_result

            # Load and preprocess audio
            audio = self.audio_processor.load(audio_path)

            # Process with parallel executors...
            results = self._parallel_process(audio)

            # Generate visualizations
            results['visualizations'] = self._generate_visualizations(audio, results)

            # Generate analysis report
            results['report'] = self.mix_visualizer.create_summary_report(results)

            # Cache results
            self.cache.set(audio_path, results)

            # Save to persistent storage
            self.storage.save_results(
                audio_path,
                results,
                metadata={'version': AnalysisVersion.to_string()}
            )

            return results

        except Exception as e:
            logger.error(f"Pipeline processing failed: {str(e)}")
            raise

    def _generate_visualizations(self, audio: np.ndarray, results: Dict) -> Dict:
        """Generate all visualizations for the analysis results."""
        visualizations = {
            'mix_signature': self.mix_visualizer.visualize_mix_signature(
                results['analysis']['mix']['graph'],
                results['analysis']['mix']['metrics'],
                'mix_signature.png'
            ),
            'mix_heatmap': self.mix_visualizer.plot_mix_heatmap(
                results['features'],
                results['analysis']['timestamps'],
                'mix_heatmap.png'
            ),
            'mix_radar': self.mix_visualizer.plot_mix_radar(
                results['features'],
                results['analysis']['timestamps'],
                'mix_radar.html'
            ),
            'clusters': self.mix_visualizer.plot_clusters(
                results['annotation']['segments'],
                results['features']['spectral'],
                'clusters.png'
            ),
            'components': self.analysis_visualizer.plot_component_analysis(
                results['analysis']['components'],
                'component_analysis.html'
            )
        }
        
        # Add Amen break visualization if segments were found
        if results['alignment']['sequence'].get('segments'):
            visualizations['amen_break'] = self.analysis_visualizer.plot_amen_break_analysis(
                audio,
                results['alignment']['sequence'],
                'amen_break_analysis.html'
            )
        
        return visualizations

    def _parallel_process(self, audio: np.ndarray) -> Dict:
        """Process audio with parallel executors."""
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Feature extraction
            logger.info("Starting feature extraction...")
            feature_futures = {
                'baseline': executor.submit(self.baseline_analyzer.analyze, audio),
                'bpm': executor.submit(self.bpm_analyzer.analyze, audio),
                'drum': executor.submit(self.drum_analyzer.analyze, audio),
                'groove': executor.submit(self.groove_analyzer.analyze, audio),
                'percussion': executor.submit(self.percussion_analyzer.analyze, audio),
                'rhythmic': executor.submit(self.rhythmic_analyzer.analyze, audio)
            }

            # Analysis
            logger.info("Starting audio analysis...")
            analysis_futures = {
                'mix': executor.submit(self.mix_analyzer.analyze, audio),
                'scene': executor.submit(self.scene_analyzer.analyze, audio)
            }

            # Alignment
            logger.info("Starting alignment analysis...")
            alignment_futures = {
                'sequence': executor.submit(self.sequence_aligner.align_sequences, audio, self.config.get('sample_rate', 44100)),
                'psa': executor.submit(self.prior_subspace.analyze, audio),
                'similarity': executor.submit(self.similarity_analyzer.analyze, audio)
            }

            # Annotation
            logger.info("Starting annotation analysis...")
            annotation_futures = {
                'peaks': executor.submit(self.peak_detector.detect, audio),
                'segments': executor.submit(self.segment_clusterer.cluster, audio),
                'transitions': executor.submit(self.transition_detector.detect, audio)
            }

            # Gather results
            results = {
                'features': {k: v.result() for k, v in feature_futures.items()},
                'analysis': {k: v.result() for k, v in analysis_futures.items()},
                'alignment': {k: v.result() for k, v in alignment_futures.items()},
                'annotation': {k: v.result() for k, v in annotation_futures.items()}
            }

            # Process alignment results with enhanced Amen break analysis
            alignment_results = results['alignment']['sequence']
            if alignment_results['segments']:
                n_segments = len(alignment_results['segments'])
                logger.info(f"Found {n_segments} potential Amen break segments")
                
                # Add variation names and characteristics
                alignment_results['variation_names'] = [
                    self.sequence_aligner.amen_template.get_variation_name(v_type)
                    for v_type in alignment_results['variation_types']
                ]
                
                alignment_results['variation_characteristics'] = [
                    self.sequence_aligner.amen_template.get_variation_characteristics(v_type)
                    for v_type in alignment_results['variation_types']
                ]
                
                # Add component analysis
                alignment_results['component_analysis'] = [
                    self.sequence_aligner.amen_template.get_component_analysis(v_type)
                    for v_type in alignment_results['variation_types']
                ]
                
                # Log detailed information for each segment
                for i, segment in enumerate(alignment_results['segments']):
                    variation_name = alignment_results['variation_names'][i]
                    characteristics = alignment_results['variation_characteristics'][i]
                    comp_analysis = alignment_results['component_analysis'][i]
                    
                    logger.info(
                        f"\nSegment {i+1}:"
                        f"\n  Time: {segment['start']:.2f}s - {segment['end']:.2f}s"
                        f"\n  Variation: {variation_name}"
                        f"\n  Confidence: {alignment_results['confidence'][i]:.2f}"
                        f"\n  Tempo Scale: {alignment_results['tempo_scales'][i]:.2f}x"
                        f"\n  Characteristics:"
                        f"\n    - Complexity: {characteristics['complexity']:.2f}"
                        f"\n    - Energy: {characteristics['energy']:.2f}"
                        f"\n    - Groove: {characteristics['groove']:.2f}"
                        f"\n  Component Analysis:"
                    )
                    
                    for comp, metrics in comp_analysis.items():
                        logger.info(
                            f"\n    {comp.capitalize()}:"
                            f"\n      - Presence: {metrics['presence']:.2f}"
                            f"\n      - Hits: {metrics['n_hits']}"
                            f"\n      - Avg Velocity: {metrics['avg_velocity']:.2f}"
                            f"\n      - Velocity Variance: {metrics['velocity_variance']:.2f}"
                        )
                
                # Generate enhanced visualization
                results['visualizations']['amen_break'] = self.analysis_visualizer.plot_amen_break_analysis(
                    audio,
                    alignment_results,
                    'amen_break_analysis.html'
                )
            else:
                logger.info("No Amen break patterns detected in the audio")

            return results

    def process_batch(self, audio_paths: List[str]) -> List[Dict]:
        """Process multiple audio files in parallel"""
        results = []
        for path in audio_paths:
            try:
                result = self.process(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed processing {path}: {str(e)}")
                continue
        return results

