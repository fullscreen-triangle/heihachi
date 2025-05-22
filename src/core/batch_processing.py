import os
import time
import logging
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import gc
import json
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.utils.progress import track_progress, update_progress, complete_progress
from src.core.audio_processing import AudioProcessor
from src.utils.config import ConfigManager

logger = get_logger(__name__)

class BatchProcessingError(Exception):
    """Error during batch processing."""
    pass

class BatchProcessor:
    """Process multiple audio files with optimized batch processing."""
    
    def __init__(self, config_path: str = "../configs/performance.yaml"):
        """Initialize batch processor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        
        # Configure workers
        self.num_workers = self.config.get('processing', 'num_workers', 
                                          min(mp.cpu_count(), 4))
        self.batch_size = self.config.get('processing', 'batch_size', 16)
        self.memory_limit_mb = self.config.get('processing', 'memory_limit_mb', 1024)
        
        # Initialize processor
        self.audio_processor = AudioProcessor(config_path)
        
        # Results storage
        self.results_dir = Path(self.config.get('storage', 'results_dir', "../results"))
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine if we should use process pool (safer) or thread pool (faster)
        self.use_processes = self.config.get('advanced', 'feature_extraction', {}).get('parallel', True)
        
        logger.info(f"Batch processor initialized with {self.num_workers} workers")
        logger.info(f"Using {'process' if self.use_processes else 'thread'} pool for parallel execution")
    
    def process_directory(self, input_dir: str, 
                         file_extension: Optional[List[str]] = None,
                         output_dir: Optional[str] = None,
                         pipeline_configs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            file_extension: List of file extensions to process (optional)
            output_dir: Directory to store results (optional)
            pipeline_configs: List of pipeline configurations to apply (optional)
            
        Returns:
            Dictionary of results and statistics
        """
        # Get file list
        if file_extension is None:
            file_extension = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
            
        file_paths = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extension):
                    file_paths.append(os.path.join(root, file))
        
        # Set output directory
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Process files
        logger.info(f"Processing {len(file_paths)} files from {input_dir}")
        return self.process_files(file_paths, output_dir, pipeline_configs)
    
    def process_files(self, file_paths: List[str],
                     output_dir: Optional[str] = None,
                     pipeline_configs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Process a list of audio files.
        
        Args:
            file_paths: List of audio file paths
            output_dir: Directory to store results (optional)
            pipeline_configs: List of pipeline configurations to apply (optional)
            
        Returns:
            Dictionary of results and statistics
        """
        if not file_paths:
            logger.warning("No files to process")
            return {'success': [], 'failed': [], 'stats': {'total': 0, 'success': 0, 'failed': 0}}
        
        # Set output directory
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Use default configuration if none provided
        if pipeline_configs is None:
            pipeline_configs = [{}]  # Empty dict means use defaults
        
        # Determine batch size based on number of files and configurations
        total_tasks = len(file_paths) * len(pipeline_configs)
        effective_batch_size = min(self.batch_size, total_tasks)
        
        # Start tracking progress
        start_time = time.time()
        progress = track_progress("batch_processing", total_tasks, 
                                 f"Processing {len(file_paths)} files with "
                                 f"{len(pipeline_configs)} configurations")
        
        # Process files
        results = {
            'success': [],
            'failed': [],
            'stats': {
                'total': total_tasks,
                'success': 0,
                'failed': 0,
                'total_time': 0
            }
        }
        
        try:
            # Process files in batches for better memory management
            for i in range(0, len(file_paths), effective_batch_size):
                batch_files = file_paths[i:i+effective_batch_size]
                
                # Process this batch
                batch_results = self._process_batch(batch_files, output_dir, pipeline_configs, progress)
                
                # Merge results
                results['success'].extend(batch_results['success'])
                results['failed'].extend(batch_results['failed'])
                results['stats']['success'] += batch_results['stats']['success']
                results['stats']['failed'] += batch_results['stats']['failed']
                
                # Force garbage collection between batches
                if self.config.get('advanced', 'memory_management', {}).get('force_gc', True):
                    gc.collect()
                    if torch.cuda.is_available() and self.config.get('advanced', 'memory_management', {}).get('clear_cuda_cache', True):
                        torch.cuda.empty_cache()
                
                # Save interim results for fault tolerance
                self._save_interim_results(results, output_dir)
            
        except Exception as e:
            logger.error(f"Error during batch processing: {e}")
            logger.error(traceback.format_exc())
            
            # Save what we have so far
            self._save_interim_results(results, output_dir)
            
            # Finalize progress tracking
            complete_progress("batch_processing")
            
            return results
        
        # Finalize progress tracking
        complete_progress("batch_processing")
        
        # Update stats
        results['stats']['total_time'] = time.time() - start_time
        
        # Save final results
        self._save_final_results(results, output_dir)
        
        logger.info(f"Batch processing complete: {results['stats']['success']} successful, "
                   f"{results['stats']['failed']} failed, "
                   f"total time: {results['stats']['total_time']:.2f}s")
        
        return results
    
    def _process_batch(self, file_paths: List[str], 
                      output_dir: Path,
                      pipeline_configs: List[Dict],
                      progress) -> Dict[str, Any]:
        """Process a batch of files.
        
        Args:
            file_paths: List of audio file paths
            output_dir: Directory to store results
            pipeline_configs: List of pipeline configurations to apply
            progress: Progress tracker
            
        Returns:
            Dictionary of batch results
        """
        batch_results = {
            'success': [],
            'failed': [],
            'stats': {
                'success': 0,
                'failed': 0
            }
        }
        
        # Create tasks for all file/config combinations
        tasks = []
        for file_path in file_paths:
            for config_idx, pipeline_config in enumerate(pipeline_configs):
                tasks.append((file_path, pipeline_config, config_idx, output_dir))
        
        # Use appropriate executor for parallel processing
        if self.use_processes and len(tasks) > 1:
            # Use process pool for better isolation
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_single_file, *task) for task in tasks]
                
                for future in futures:
                    try:
                        result = future.result()
                        if result['success']:
                            batch_results['success'].append(result)
                            batch_results['stats']['success'] += 1
                        else:
                            batch_results['failed'].append(result)
                            batch_results['stats']['failed'] += 1
                        
                        update_progress("batch_processing")
                    except Exception as e:
                        logger.error(f"Error processing file: {e}")
                        batch_results['failed'].append({
                            'file_path': "unknown",
                            'success': False,
                            'error': str(e)
                        })
                        batch_results['stats']['failed'] += 1
                        update_progress("batch_processing")
        else:
            # Use sequential processing for smaller batches or if processes are disabled
            for task in tasks:
                try:
                    result = self._process_single_file(*task)
                    if result['success']:
                        batch_results['success'].append(result)
                        batch_results['stats']['success'] += 1
                    else:
                        batch_results['failed'].append(result)
                        batch_results['stats']['failed'] += 1
                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    batch_results['failed'].append({
                        'file_path': task[0] if len(task) > 0 else "unknown",
                        'success': False,
                        'error': str(e)
                    })
                    batch_results['stats']['failed'] += 1
                
                update_progress("batch_processing")
        
        return batch_results
    
    def _process_single_file(self, file_path: str, 
                            pipeline_config: Dict,
                            config_idx: int,
                            output_dir: Path) -> Dict[str, Any]:
        """Process a single file with a specific configuration.
        
        Args:
            file_path: Path to the audio file
            pipeline_config: Pipeline configuration
            config_idx: Configuration index
            output_dir: Directory to store results
            
        Returns:
            Dictionary with processing results
        """
        # Import Pipeline locally to avoid circular imports
        from src.core.pipeline import Pipeline
        
        start_time = time.time()
        file_name = os.path.basename(file_path)
        
        logger.info(f"Processing {file_name} with config {config_idx}")
        
        result = {
            'file_path': file_path,
            'file_name': file_name,
            'config_idx': config_idx,
            'success': False,
            'time': 0
        }
        
        try:
            # Initialize pipeline with configuration
            pipeline = Pipeline(pipeline_config)
            
            # Process file
            process_result = pipeline.process(file_path)
            
            # Save results
            if config_idx > 0:
                # For multiple configs, use config-specific directories
                config_dir = output_dir / f"config_{config_idx}"
                config_dir.mkdir(exist_ok=True)
                result_path = config_dir / f"{file_name}.json"
            else:
                result_path = output_dir / f"{file_name}.json"
            
            # Extract key metrics and save
            metrics = self._extract_metrics(process_result)
            with open(result_path, 'w') as f:
                json.dump({
                    'file_path': file_path,
                    'file_name': file_name,
                    'config_idx': config_idx,
                    'metrics': metrics,
                    'processing_time': time.time() - start_time
                }, f, indent=2)
            
            # Update result
            result['success'] = True
            result['metrics'] = metrics
            result['result_path'] = str(result_path)
            
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")
            logger.error(traceback.format_exc())
            result['error'] = str(e)
        
        # Record processing time
        result['time'] = time.time() - start_time
        
        return result
    
    def _extract_metrics(self, process_result: Dict) -> Dict[str, Any]:
        """Extract key metrics from processing results.
        
        Args:
            process_result: Pipeline processing results
            
        Returns:
            Dictionary of key metrics
        """
        metrics = {}
        
        # Extract mix analysis metrics
        if 'analysis' in process_result and 'mix' in process_result['analysis']:
            mix = process_result['analysis']['mix']
            if isinstance(mix, dict):
                if 'metrics' in mix:
                    metrics['mix'] = mix['metrics']
                else:
                    metrics['mix'] = {k: v for k, v in mix.items() 
                                     if not isinstance(v, (dict, list, np.ndarray))}
        
        # Extract BPM
        if 'features' in process_result and 'bpm' in process_result['features']:
            bpm_data = process_result['features']['bpm']
            if isinstance(bpm_data, dict):
                metrics['bpm'] = bpm_data.get('bpm')
                metrics['tempo'] = bpm_data.get('tempo')
            elif isinstance(bpm_data, (int, float)):
                metrics['bpm'] = bpm_data
        
        # Extract alignment metrics
        if 'analysis' in process_result and 'alignment' in process_result['analysis']:
            alignment = process_result['analysis']['alignment']
            if isinstance(alignment, dict):
                metrics['alignment'] = alignment
        
        # Extract annotation metrics
        if 'annotation' in process_result:
            annotation = process_result['annotation']
            if 'segments' in annotation:
                segments = annotation['segments']
                metrics['segments'] = {
                    'count': len(segments) if isinstance(segments, list) else 0
                }
            
            if 'transitions' in annotation:
                transitions = annotation['transitions']
                metrics['transitions'] = {
                    'count': len(transitions) if isinstance(transitions, list) else 0
                }
        
        return metrics
    
    def _save_interim_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save interim results for fault tolerance.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        try:
            # Save to a temporary file first to avoid corrupting existing file
            temp_path = output_dir / "batch_results_temp.json"
            with open(temp_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Rename to final path
            final_path = output_dir / "batch_results.json"
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(temp_path, final_path)
            
            logger.debug(f"Saved interim results to {final_path}")
        except Exception as e:
            logger.warning(f"Failed to save interim results: {e}")
    
    def _save_final_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Save final results with timestamp.
        
        Args:
            results: Results dictionary
            output_dir: Output directory
        """
        try:
            # Save to a timestamped file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_path = output_dir / f"batch_results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Also save to standard location
            standard_path = output_dir / "batch_results.json"
            with open(standard_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved final results to {results_path}")
        except Exception as e:
            logger.warning(f"Failed to save final results: {e}")
    
    def resume_batch_processing(self, input_dir: str, 
                               output_dir: Optional[str] = None,
                               pipeline_configs: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Resume batch processing from previous interrupted run.
        
        Args:
            input_dir: Directory containing audio files
            output_dir: Directory to store results (optional)
            pipeline_configs: List of pipeline configurations to apply (optional)
            
        Returns:
            Dictionary of results and statistics
        """
        # Set output directory
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load previous results
        previous_results = self._load_previous_results(output_dir)
        if previous_results is None:
            logger.warning("No previous results found, starting from scratch")
            return self.process_directory(input_dir, output_dir=output_dir, pipeline_configs=pipeline_configs)
        
        # Get list of successfully processed files
        processed_files = set()
        for result in previous_results.get('success', []):
            if 'file_path' in result:
                processed_files.add(result['file_path'])
        
        # Get list of all files
        file_extension = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        all_file_paths = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extension):
                    all_file_paths.append(os.path.join(root, file))
        
        # Filter out already processed files
        file_paths = [path for path in all_file_paths if path not in processed_files]
        
        if not file_paths:
            logger.info("All files already processed")
            return previous_results
        
        logger.info(f"Resuming processing: {len(file_paths)} files remaining, "
                   f"{len(processed_files)} already processed")
        
        # Process remaining files
        new_results = self.process_files(file_paths, output_dir, pipeline_configs)
        
        # Merge results
        merged_results = self._merge_results(previous_results, new_results)
        
        # Save merged results
        self._save_final_results(merged_results, output_dir)
        
        return merged_results
    
    def _load_previous_results(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Load previous batch processing results.
        
        Args:
            output_dir: Output directory
            
        Returns:
            Previous results dictionary, or None if not found
        """
        results_path = output_dir / "batch_results.json"
        if not results_path.exists():
            return None
        
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            logger.info(f"Loaded previous results from {results_path}")
            return results
        except Exception as e:
            logger.warning(f"Failed to load previous results: {e}")
            return None
    
    def _merge_results(self, previous_results: Dict[str, Any], 
                      new_results: Dict[str, Any]) -> Dict[str, Any]:
        """Merge previous and new results.
        
        Args:
            previous_results: Previous batch processing results
            new_results: New batch processing results
            
        Returns:
            Merged results dictionary
        """
        merged = {
            'success': previous_results.get('success', []) + new_results.get('success', []),
            'failed': previous_results.get('failed', []) + new_results.get('failed', []),
            'stats': {
                'total': (previous_results.get('stats', {}).get('total', 0) + 
                          new_results.get('stats', {}).get('total', 0)),
                'success': (previous_results.get('stats', {}).get('success', 0) + 
                            new_results.get('stats', {}).get('success', 0)),
                'failed': (previous_results.get('stats', {}).get('failed', 0) + 
                          new_results.get('stats', {}).get('failed', 0)),
                'total_time': (previous_results.get('stats', {}).get('total_time', 0) + 
                              new_results.get('stats', {}).get('total_time', 0))
            }
        }
        
        return merged


def process_directory(input_dir: str, output_dir: Optional[str] = None,
                     config_path: str = "../configs/performance.yaml") -> Dict[str, Any]:
    """
    Process all audio files in a directory.
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to store results (optional)
        config_path: Path to configuration file
        
    Returns:
        Dictionary of results and statistics
    """
    processor = BatchProcessor(config_path)
    return processor.process_directory(input_dir, output_dir=output_dir)


def resume_processing(input_dir: str, output_dir: Optional[str] = None,
                     config_path: str = "../configs/performance.yaml") -> Dict[str, Any]:
    """
    Resume batch processing from a previous interrupted run.
    
    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to store results (optional)
        config_path: Path to configuration file
        
    Returns:
        Dictionary of results and statistics
    """
    processor = BatchProcessor(config_path)
    return processor.resume_batch_processing(input_dir, output_dir=output_dir) 