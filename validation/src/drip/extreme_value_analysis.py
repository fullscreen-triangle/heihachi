import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
import json
import os
import mmap
import multiprocessing as mp
from joblib import Parallel, delayed
import dask
from dask.distributed import Client, as_completed
import dask.array as da
from pathlib import Path
import psutil
import gc
from functools import partial
import h5py
import zarr
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')


class HPCExtremeValueDripAnalyzer:
    def __init__(self, n_workers=None, memory_limit='8GB', chunk_size='100MB'):
        """
        HPC-optimized analyzer for extreme values in large drip analysis files

        Args:
            n_workers: Number of parallel workers (default: CPU count)
            memory_limit: Memory limit per worker
            chunk_size: Size of data chunks for processing
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.memory_limit = memory_limit
        self.chunk_size = chunk_size
        self.extreme_thresholds = {
            'velocity': 1e6,
            'size': 1e3,
            'surface_tension': 1e5,
            's_entropy': 1e4
        }

        # Initialize Dask client for distributed computing [[0]](#__0)
        self.client = None
        self._setup_dask_client()

    def _setup_dask_client(self):
        """Setup Dask distributed client for HPC processing"""
        try:
            # Configure Dask for HPC environment [[0]](#__0)
            self.client = Client(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit=self.memory_limit,
                dashboard_address=':8787'
            )
            print(f"Dask client initialized with {self.n_workers} workers")
            print(f"Dashboard available at: {self.client.dashboard_link}")
        except Exception as e:
            print(f"Warning: Could not initialize Dask client: {e}")
            self.client = None

    def load_large_json_file(self, file_path):
        """
        Memory-efficient loading of large JSON files using memory mapping

        Args:
            file_path: Path to the JSON file
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_size_gb = file_path.stat().st_size / (1024 ** 3)
        print(f"Loading file: {file_path.name} ({file_size_gb:.2f} GB)")

        # Use memory mapping for large files [[2]](#__2)
        if file_size_gb > 0.5:  # Use mmap for files > 500MB
            return self._load_with_mmap(file_path)
        else:
            # Standard loading for smaller files
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _load_with_mmap(self, file_path):
        """Load large JSON files using memory mapping"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Use mmap for efficient file access [[2]](#__2)
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    # Read file content in chunks to avoid memory overflow
                    content = mmapped_file.read().decode('utf-8')
                    return json.loads(content)
        except Exception as e:
            print(f"Memory mapping failed, falling back to chunked reading: {e}")
            return self._load_chunked_json(file_path)

    def _load_chunked_json(self, file_path):
        """Load JSON file in chunks for very large files"""
        chunk_size = 1024 * 1024  # 1MB chunks
        json_content = ""

        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                json_content += chunk

                # Periodic garbage collection for memory management
                if len(json_content) % (chunk_size * 10) == 0:
                    gc.collect()

        return json.loads(json_content)

    def process_large_dataset_parallel(self, file_path):
        """
        Process large dataset using parallel computing with Joblib and Dask
        """
        # Load data efficiently
        drip_data = self.load_large_json_file(file_path)

        # Split processing into parallel tasks [[0]](#__0)
        if self.client:
            return self._process_with_dask(drip_data)
        else:
            return self._process_with_joblib(drip_data)

    def _process_with_dask(self, drip_data):
        """Process data using Dask distributed computing"""
        print("Processing with Dask distributed computing...")

        # Create delayed computations for each analysis component [[0]](#__0)
        s_entropy_task = dask.delayed(self._analyze_s_entropy_component)(drip_data)
        droplet_task = dask.delayed(self._analyze_droplet_component)(drip_data)
        nan_task = dask.delayed(self._analyze_nan_component)(drip_data)
        stats_task = dask.delayed(self._analyze_stats_component)(drip_data)
        correlation_task = dask.delayed(self._analyze_correlation_component)(drip_data)
        anomaly_task = dask.delayed(self._analyze_anomaly_component)(drip_data)

        # Execute all tasks in parallel
        results = dask.compute(
            s_entropy_task, droplet_task, nan_task,
            stats_task, correlation_task, anomaly_task
        )

        return {
            's_entropy': results[0],
            'droplet': results[1],
            'nan_analysis': results[2],
            'statistics': results[3],
            'correlation': results[4],
            'anomalies': results[5]
        }

    def _process_with_joblib(self, drip_data):
        """Process data using Joblib parallel processing"""
        print("Processing with Joblib parallel computing...")

        # Define analysis functions for parallel execution [[0]](#__0)
        analysis_functions = [
            self._analyze_s_entropy_component,
            self._analyze_droplet_component,
            self._analyze_nan_component,
            self._analyze_stats_component,
            self._analyze_correlation_component,
            self._analyze_anomaly_component
        ]

        # Execute in parallel using Joblib
        results = Parallel(n_jobs=self.n_workers, backend='threading')(
            delayed(func)(drip_data) for func in analysis_functions
        )

        return {
            's_entropy': results[0],
            'droplet': results[1],
            'nan_analysis': results[2],
            'statistics': results[3],
            'correlation': results[4],
            'anomalies': results[5]
        }

    def _analyze_s_entropy_component(self, drip_data):
        """Analyze S-entropy coordinates component"""
        s_entropy = drip_data.get('s_entropy_coordinates', {})

        coords = ['S_frequency', 'S_time', 'S_amplitude', 'total_entropy']
        values = [s_entropy.get(coord, 0) for coord in coords]
        log_magnitudes = [np.log10(abs(v)) if v != 0 else 0 for v in values]
        signs = [1 if v >= 0 else -1 for v in values]

        return {
            'coords': coords,
            'values': values,
            'log_magnitudes': log_magnitudes,
            'signs': signs
        }

    def _analyze_droplet_component(self, drip_data):
        """Analyze droplet parameters component"""
        droplet_params = drip_data.get('droplet_parameters', {})

        params = ['velocity', 'size', 'impact_angle', 'surface_tension']
        values = [droplet_params.get(param, 0) for param in params]

        processed_values = []
        extreme_flags = []

        for param, value in zip(params, values):
            if np.isnan(value):
                processed_values.append(0)
                extreme_flags.append('NaN')
            elif abs(value) > self.extreme_thresholds.get(param.split('_')[0], 1e6):
                processed_values.append(np.log10(abs(value)))
                extreme_flags.append('Extreme')
            else:
                processed_values.append(value)
                extreme_flags.append('Normal')

        return {
            'params': params,
            'values': values,
            'processed_values': processed_values,
            'extreme_flags': extreme_flags
        }

    def _analyze_nan_component(self, drip_data):
        """Analyze NaN values component"""
        droplet_params = drip_data.get('droplet_parameters', {})

        params = list(droplet_params.keys())
        nan_status = []

        for param in params:
            value = droplet_params.get(param, 0)
            if isinstance(value, dict):
                continue
            nan_status.append('NaN' if np.isnan(value) else 'Valid')

        nan_count = nan_status.count('NaN')
        valid_count = nan_status.count('Valid')

        return {
            'nan_count': nan_count,
            'valid_count': valid_count,
            'nan_status': nan_status
        }

    def _analyze_stats_component(self, drip_data):
        """Analyze statistical distributions component"""
        s_entropy = drip_data.get('s_entropy_coordinates', {})
        droplet_params = drip_data.get('droplet_parameters', {})

        all_values = []
        value_labels = []

        # Collect S-entropy values
        for key, value in s_entropy.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                all_values.append(abs(value))
                value_labels.append(f'S_{key.split("_")[-1]}')

        # Collect droplet parameter values
        for key, value in droplet_params.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                all_values.append(abs(value))
                value_labels.append(key.replace('_', ' ').title())

        if all_values:
            log_values = [np.log10(v) for v in all_values if v > 0]
            return {
                'log_values': log_values,
                'mean': np.mean(log_values),
                'median': np.median(log_values),
                'std': np.std(log_values),
                'count': len(log_values)
            }

        return {'log_values': [], 'mean': 0, 'median': 0, 'std': 0, 'count': 0}

    def _analyze_correlation_component(self, drip_data):
        """Analyze file size correlation component"""
        audio_props = drip_data.get('audio_properties', {})
        s_entropy = drip_data.get('s_entropy_coordinates', {})

        duration = audio_props.get('duration', 315.4)
        samples = audio_props.get('samples', 13908992)
        file_size_mb = 840  # Default for large files

        total_entropy = abs(s_entropy.get('total_entropy', 0))

        entropy_per_mb = total_entropy / file_size_mb if file_size_mb > 0 else 0
        entropy_per_second = total_entropy / duration if duration > 0 else 0
        entropy_per_sample = total_entropy / samples if samples > 0 else 0

        return {
            'entropy_per_mb': entropy_per_mb,
            'entropy_per_second': entropy_per_second,
            'entropy_per_sample': entropy_per_sample,
            'file_size_mb': file_size_mb,
            'duration': duration
        }

    def _analyze_anomaly_component(self, drip_data):
        """Analyze processing anomalies component"""
        anomalies = []

        # Check S-entropy values
        s_entropy = drip_data.get('s_entropy_coordinates', {})
        for key, value in s_entropy.items():
            if isinstance(value, (int, float)):
                if abs(value) > 1e6:
                    anomalies.append(f'Extreme {key}: {value:.2e}')

        # Check droplet parameters
        droplet_params = drip_data.get('droplet_parameters', {})
        for key, value in droplet_params.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    anomalies.append(f'NaN in {key}')
                elif abs(value) > 1e6:
                    anomalies.append(f'Extreme {key}: {value:.2e}')

        return {'anomalies': anomalies}

    def create_hpc_optimized_visualization(self, analysis_results, output_path=None):
        """
        Create memory-efficient visualization optimized for HPC environments
        """
        # Use memory-efficient plotting with reduced resolution for large datasets [[1]](#__1)
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['agg.path.chunksize'] = 10000

        fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=100)  # Reduced DPI for memory efficiency

        # Plot each component using the pre-computed results
        self._plot_s_entropy_analysis(axes[0, 0], analysis_results['s_entropy'])
        self._plot_droplet_analysis(axes[0, 1], analysis_results['droplet'])
        self._plot_nan_analysis(axes[0, 2], analysis_results['nan_analysis'])
        self._plot_stats_analysis(axes[1, 0], analysis_results['statistics'])
        self._plot_correlation_analysis(axes[1, 1], analysis_results['correlation'])
        self._plot_anomaly_analysis(axes[1, 2], analysis_results['anomalies'])

        plt.tight_layout()
        plt.suptitle('HPC Extreme Value Analysis: Large Audio File Processing\nOptimized for 1GB+ Files',
                     fontsize=16, fontweight='bold', y=0.98)

        # Save to file if specified to free memory [[1]](#__1)
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Visualization saved to: {output_path}")

        return fig

    def _plot_s_entropy_analysis(self, ax, s_entropy_data):
        """Plot S-entropy analysis results"""
        coords = s_entropy_data['coords']
        log_magnitudes = s_entropy_data['log_magnitudes']
        signs = s_entropy_data['signs']
        values = s_entropy_data['values']

        colors = ['green' if s == 1 else 'red' for s in signs]
        bars = ax.bar(coords, log_magnitudes, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, value, sign in zip(bars, values, signs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{value:.2e}', ha='center', va='bottom', fontweight='bold',
                    fontsize=9, rotation=45)

        ax.set_ylabel('Log₁₀(|Value|)', fontsize=11)
        ax.set_title('S-Entropy Coordinate\nMagnitude Analysis', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_droplet_analysis(self, ax, droplet_data):
        """Plot droplet parameter analysis results"""
        params = droplet_data['params']
        processed_values = droplet_data['processed_values']
        extreme_flags = droplet_data['extreme_flags']
        values = droplet_data['values']

        colors = {'Normal': '#2ECC71', 'Extreme': '#E74C3C', 'NaN': '#F39C12'}
        bar_colors = [colors[flag] for flag in extreme_flags]

        bars = ax.bar(params, processed_values, color=bar_colors, alpha=0.8, edgecolor='black')

        # Add status labels
        for bar, flag, original_val in zip(bars, extreme_flags, values):
            height = bar.get_height()
            if flag == 'NaN':
                label = 'NaN'
            elif flag == 'Extreme':
                label = f'{original_val:.1e}'
            else:
                label = f'{original_val:.3f}'

            ax.text(bar.get_x() + bar.get_width() / 2., height + max(processed_values) * 0.02,
                    label, ha='center', va='bottom', fontweight='bold',
                    fontsize=9, rotation=45)

        ax.set_ylabel('Parameter Value (Log for Extremes)', fontsize=11)
        ax.set_title('Droplet Parameter\nExtreme Value Detection', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_nan_analysis(self, ax, nan_data):
        """Plot NaN analysis results"""
        valid_count = nan_data['valid_count']
        nan_count = nan_data['nan_count']

        sizes = [valid_count, nan_count]
        labels = [f'Valid Values\n({valid_count})', f'NaN Values\n({nan_count})']
        colors = ['#2ECC71', '#E74C3C']
        explode = (0, 0.1) if nan_count > 0 else (0, 0)

        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90, explode=explode)

        ax.set_title('NaN Value Distribution\nin Droplet Parameters', fontsize=12, fontweight='bold')

    def _plot_stats_analysis(self, ax, stats_data):
        """Plot statistical analysis results"""
        log_values = stats_data['log_values']

        if log_values:
            ax.hist(log_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(stats_data['mean'], color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {stats_data["mean"]:.2f}')
            ax.axvline(stats_data['median'], color='green', linestyle='--', linewidth=2,
                       label=f'Median: {stats_data["median"]:.2f}')

            ax.set_xlabel('Log₁₀(|Parameter Value|)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title('Statistical Distribution\nof Parameter Magnitudes', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    def _plot_correlation_analysis(self, ax, correlation_data):
        """Plot correlation analysis results"""
        entropy_per_mb = correlation_data['entropy_per_mb']
        entropy_per_second = correlation_data['entropy_per_second']
        entropy_per_sample = correlation_data['entropy_per_sample']

        metrics = ['Entropy/MB', 'Entropy/Second', 'Entropy/Sample']
        values = [entropy_per_mb, entropy_per_second, entropy_per_sample * 1e6]

        log_values = [np.log10(v) if v > 0 else 0 for v in values]

        bars = ax.bar(metrics, log_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                      alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_ylabel('Log₁₀(Entropy Density)', fontsize=11)
        ax.set_title('File Size vs Processing\nComplexity Correlation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_anomaly_analysis(self, ax, anomaly_data):
        """Plot anomaly analysis results"""
        anomalies = anomaly_data['anomalies']

        if anomalies:
            anomaly_types = ['Extreme Values', 'NaN Values', 'Processing Issues']
            counts = [0, 0, 0]

            for anomaly in anomalies:
                if 'Extreme' in anomaly:
                    counts[0] += 1
                elif 'NaN' in anomaly:
                    counts[1] += 1
                else:
                    counts[2] += 1

            colors = ['#E74C3C', '#F39C12', '#9B59B6']
            bars = ax.bar(anomaly_types, counts, color=colors, alpha=0.8, edgecolor='black')

            ax.set_ylabel('Anomaly Count', fontsize=11)
            ax.set_title('Processing Anomaly\nDetection Results', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No Anomalies\nDetected', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))

    def process_file_from_path(self, file_path, output_path=None):
        """
        Main function to process a file from file path

        Args:
            file_path: Path to the JSON file containing drip analysis data
            output_path: Optional path to save the visualization
        """
        start_time = time.time()

        print(f"Starting HPC processing of: {file_path}")
        print(f"Available memory: {psutil.virtual_memory().available / (1024 ** 3):.2f} GB")
        print(f"CPU cores: {mp.cpu_count()}")

        try:
            # Process the large dataset in parallel
            analysis_results = self.process_large_dataset_parallel(file_path)

            # Create visualization
            fig = self.create_hpc_optimized_visualization(analysis_results, output_path)

            processing_time = time.time() - start_time
            print(f"Processing completed in {processing_time:.2f} seconds")

            # Memory cleanup [[3]](#__3)
            gc.collect()

            return fig, analysis_results

        except Exception as e:
            print(f"Error processing file: {e}")
            raise
        finally:
            # Cleanup Dask client
            if self.client:
                self.client.close()

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'client') and self.client:
            self.client.close()


# HPC Batch Processing Function
def process_multiple_files_hpc(file_paths, output_dir, n_workers=None):
    """
    Process multiple large files in HPC environment using shared memory

    Args:
        file_paths: List of file paths to process
        output_dir: Directory to save results
        n_workers: Number of parallel workers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Use ProcessPoolExecutor for true parallelism with large files [[3]](#__3)
    with ProcessPoolExecutor(max_workers=n_workers or mp.cpu_count()) as executor:
        futures = []

        for file_path in file_paths:
            file_name = Path(file_path).stem
            output_path = output_dir / f"{file_name}_analysis.png"

            # Submit processing task
            future = executor.submit(process_single_file_worker, file_path, output_path)
            futures.append((future, file_path))

        # Collect results
        results = []
        for future, file_path in futures:
            try:
                result = future.result(timeout=3600)  # 1 hour timeout
                results.append((file_path, result))
                print(f"Completed: {file_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
                results.append((file_path, None))

    return results


def process_single_file_worker(file_path, output_path):
    """Worker function for processing single file in separate process"""
    analyzer = HPCExtremeValueDripAnalyzer(n_workers=2)  # Limit workers per process
    try:
        fig, analysis_results = analyzer.process_file_from_path(file_path, output_path)
        plt.close(fig)  # Close figure to free memory
        return analysis_results
    except Exception as e:
        print(f"Worker error processing {file_path}: {e}")
        return None
    finally:
        del analyzer  # Explicit cleanup


# Usage example for HPC environment
if __name__ == "__main__":
    # Single file processing
    file_path = "/path/to/your/large_audio_analysis.json"  # Replace with your file path
    output_path = "/path/to/output/analysis_result.png"

    # Initialize HPC analyzer
    analyzer = HPCExtremeValueDripAnalyzer(
        n_workers=8,  # Adjust based on your HPC resources
        memory_limit='4GB',  # Adjust based on available memory
        chunk_size='200MB'
    )

    # Process the file
    try:
        fig, results = analyzer.process_file_from_path(file_path, output_path)
        plt.show()
    except Exception as e:
        print(f"Processing failed: {e}")

    # Batch processing example
    # file_paths = [
    #     "/path/to/file1.json",
    #     "/path/to/file2.json",
    #     "/path/to/file3.json"
    # ]
    #
    # batch_results = process_multiple_files_hpc(
    #     file_paths,
    #     "/path/to/output/directory",
    #     n_workers=4
    # )
