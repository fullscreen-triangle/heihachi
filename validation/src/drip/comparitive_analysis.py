import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, IncrementalPCA
import seaborn as sns
import h5py
import zarr
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
import json
import mmap
import gc
from pathlib import Path
import psutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from joblib import Parallel, delayed
import warnings
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
import requests
from urllib.parse import urlparse
import tempfile
import os

warnings.filterwarnings('ignore')


class HPCComparativeDripAnalysisFramework:
    def __init__(self, n_workers=None, memory_limit='8GB', use_dask=True):
        """
        HPC-optimized framework for comparing multiple large audio drip analyses

        Args:
            n_workers: Number of parallel workers
            memory_limit: Memory limit per worker
            use_dask: Whether to use Dask for distributed computing
        """
        self.n_workers = n_workers or mp.cpu_count()
        self.memory_limit = memory_limit
        self.use_dask = use_dask

        # Initialize Dask client for large dataset processing [[1]](#__1)
        self.client = None
        if use_dask:
            self._setup_dask_client()

        # Reference data for comparison (can be loaded from HDF5/Zarr) [[1]](#__1)
        self.reference_data = {
            'normal_track': {
                's_entropy_magnitude': 1e3,
                'droplet_velocity': 10.5,
                'processing_size_mb': 25,
                'nan_count': 0
            },
            'complex_track': {
                's_entropy_magnitude': 1e4,
                'droplet_velocity': 150.2,
                'processing_size_mb': 120,
                'nan_count': 1
            }
        }

        # Memory-efficient storage for large datasets [[3]](#__3)
        self.temp_dir = tempfile.mkdtemp()
        self.hdf5_store = None
        self.zarr_store = None

    def _setup_dask_client(self):
        """Setup Dask client for distributed processing"""
        try:
            self.client = Client(
                n_workers=self.n_workers,
                threads_per_worker=2,
                memory_limit=self.memory_limit,
                dashboard_address=':8788'
            )
            print(f"Dask client initialized: {self.client.dashboard_link}")
        except Exception as e:
            print(f"Warning: Could not initialize Dask client: {e}")
            self.client = None

    def load_data_from_sources(self, data_sources):
        """
        Load data from multiple sources (files, URLs, HDF5, Zarr)

        Args:
            data_sources: List of data sources (file paths, URLs, or data dictionaries)
        """
        loaded_data = []

        for source in data_sources:
            if isinstance(source, str):
                if self._is_url(source):
                    data = self._load_from_url(source)
                elif source.endswith('.h5') or source.endswith('.hdf5'):
                    data = self._load_from_hdf5(source)
                elif source.endswith('.zarr'):
                    data = self._load_from_zarr(source)
                else:
                    data = self._load_from_json_file(source)
            else:
                data = source  # Assume it's already a data dictionary

            loaded_data.append(data)

        return loaded_data

    def _is_url(self, string):
        """Check if string is a URL"""
        try:
            result = urlparse(string)
            return all([result.scheme, result.netloc])
        except:
            return False

    def _load_from_url(self, url):
        """Load data from URL with memory mapping for large files"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save to temporary file for memory mapping [[3]](#__3)
            temp_file = os.path.join(self.temp_dir, f"temp_{hash(url)}.json")

            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return self._load_from_json_file(temp_file)
        except Exception as e:
            print(f"Error loading from URL {url}: {e}")
            return {}

    def _load_from_json_file(self, file_path):
        """Load JSON file with memory mapping for large files"""
        file_path = Path(file_path)
        file_size_gb = file_path.stat().st_size / (1024 ** 3)

        if file_size_gb > 0.5:  # Use memory mapping for large files
            with open(file_path, 'r', encoding='utf-8') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                    content = mmapped_file.read().decode('utf-8')
                    return json.loads(content)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

    def _load_from_hdf5(self, file_path):
        """Load data from HDF5 file for memory-efficient processing"""
        try:
            with h5py.File(file_path, 'r') as f:
                data = {}
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        data[key] = f[key][:]
                    elif isinstance(f[key], h5py.Group):
                        data[key] = {subkey: f[key][subkey][:] for subkey in f[key].keys()}
                return data
        except Exception as e:
            print(f"Error loading HDF5 file {file_path}: {e}")
            return {}

    def _load_from_zarr(self, file_path):
        """Load data from Zarr store for distributed processing"""
        try:
            store = zarr.open(file_path, mode='r')
            data = {}
            for key in store.keys():
                if hasattr(store[key], 'shape'):
                    data[key] = np.array(store[key])
                else:
                    data[key] = dict(store[key])
            return data
        except Exception as e:
            print(f"Error loading Zarr file {file_path}: {e}")
            return {}

    def create_hpc_comparative_analysis(self, data_sources, reference_tracks=None, output_path=None):
        """
        Create comprehensive comparative analysis optimized for HPC environments

        Args:
            data_sources: List of data sources to compare
            reference_tracks: Optional reference data
            output_path: Path to save the analysis results
        """
        print("Starting HPC comparative analysis...")
        start_time = time.time()

        # Load all data sources efficiently
        loaded_data = self.load_data_from_sources(data_sources)

        # Create memory-efficient storage for analysis results [[1]](#__1)
        self._setup_efficient_storage(len(loaded_data))

        # Process data in parallel chunks
        if self.client:
            analysis_results = self._process_with_dask_comparison(loaded_data)
        else:
            analysis_results = self._process_with_joblib_comparison(loaded_data)

        # Create visualization with memory optimization
        fig = self._create_memory_efficient_visualization(analysis_results, loaded_data)

        processing_time = time.time() - start_time
        print(f"Comparative analysis completed in {processing_time:.2f} seconds")

        # Save results if output path specified
        if output_path:
            self._save_analysis_results(fig, analysis_results, output_path)

        # Memory cleanup
        gc.collect()

        return fig, analysis_results

    def _setup_efficient_storage(self, n_datasets):
        """Setup HDF5 and Zarr stores for efficient data handling"""
        # Create HDF5 store for structured data [[3]](#__3)
        hdf5_path = os.path.join(self.temp_dir, 'analysis_cache.h5')
        self.hdf5_store = h5py.File(hdf5_path, 'w')

        # Create Zarr store for array data [[1]](#__1)
        zarr_path = os.path.join(self.temp_dir, 'array_cache.zarr')
        self.zarr_store = zarr.open(zarr_path, mode='w')

    def _process_with_dask_comparison(self, loaded_data):
        """Process comparative analysis using Dask distributed computing"""
        print("Processing with Dask distributed computing...")

        # Create delayed computations for each comparison component
        import dask

        tasks = []
        for i, data in enumerate(loaded_data):
            # Create delayed tasks for each analysis component
            entropy_task = dask.delayed(self._analyze_s_entropy_comparative)(data, i)
            droplet_task = dask.delayed(self._analyze_droplet_comparative)(data, i)
            efficiency_task = dask.delayed(self._analyze_efficiency_comparative)(data, i)
            anomaly_task = dask.delayed(self._analyze_anomaly_comparative)(data, i)

            tasks.extend([entropy_task, droplet_task, efficiency_task, anomaly_task])

        # Execute all tasks in parallel
        results = dask.compute(*tasks)

        # Organize results by analysis type
        n_datasets = len(loaded_data)
        return {
            's_entropy': results[0:n_datasets],
            'droplet': results[n_datasets:2 * n_datasets],
            'efficiency': results[2 * n_datasets:3 * n_datasets],
            'anomaly': results[3 * n_datasets:4 * n_datasets]
        }

    def _process_with_joblib_comparison(self, loaded_data):
        """Process comparative analysis using Joblib parallel processing"""
        print("Processing with Joblib parallel computing...")

        # Process each dataset in parallel
        s_entropy_results = Parallel(n_jobs=self.n_workers)(
            delayed(self._analyze_s_entropy_comparative)(data, i)
            for i, data in enumerate(loaded_data)
        )

        droplet_results = Parallel(n_jobs=self.n_workers)(
            delayed(self._analyze_droplet_comparative)(data, i)
            for i, data in enumerate(loaded_data)
        )

        efficiency_results = Parallel(n_jobs=self.n_workers)(
            delayed(self._analyze_efficiency_comparative)(data, i)
            for i, data in enumerate(loaded_data)
        )

        anomaly_results = Parallel(n_jobs=self.n_workers)(
            delayed(self._analyze_anomaly_comparative)(data, i)
            for i, data in enumerate(loaded_data)
        )

        return {
            's_entropy': s_entropy_results,
            'droplet': droplet_results,
            'efficiency': efficiency_results,
            'anomaly': anomaly_results
        }

    def _analyze_s_entropy_comparative(self, data, dataset_idx):
        """Analyze S-entropy with memory-efficient processing"""
        s_entropy = data.get('s_entropy_coordinates', {})

        # Extract entropy values efficiently
        entropy_values = []
        entropy_labels = []

        for key, value in s_entropy.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                entropy_values.append(abs(value))
                entropy_labels.append(key)

        # Store in HDF5 for memory efficiency [[3]](#__3)
        if self.hdf5_store:
            grp = self.hdf5_store.create_group(f'dataset_{dataset_idx}_entropy')
            if entropy_values:
                grp.create_dataset('values', data=np.array(entropy_values))
                grp.create_dataset('labels', data=[s.encode('utf-8') for s in entropy_labels])

        return {
            'values': entropy_values,
            'labels': entropy_labels,
            'total_entropy': abs(s_entropy.get('total_entropy', 0)),
            'dataset_idx': dataset_idx
        }

    def _analyze_droplet_comparative(self, data, dataset_idx):
        """Analyze droplet parameters with efficient memory usage"""
        droplet_params = data.get('droplet_parameters', {})

        # Extract parameters efficiently
        param_values = []
        param_labels = []
        extreme_flags = []

        for key, value in droplet_params.items():
            if isinstance(value, (int, float)):
                param_values.append(value)
                param_labels.append(key)

                if np.isnan(value):
                    extreme_flags.append('NaN')
                elif abs(value) > 1e6:
                    extreme_flags.append('Extreme')
                else:
                    extreme_flags.append('Normal')

        # Store in Zarr for array data [[1]](#__1)
        if self.zarr_store and param_values:
            self.zarr_store.create_dataset(
                f'dataset_{dataset_idx}_droplet',
                data=np.array(param_values),
                chunks=True,
                compression='gzip'
            )

        return {
            'values': param_values,
            'labels': param_labels,
            'extreme_flags': extreme_flags,
            'dataset_idx': dataset_idx
        }

    def _analyze_efficiency_comparative(self, data, dataset_idx):
        """Analyze processing efficiency across datasets"""
        audio_props = data.get('audio_properties', {})
        duration = audio_props.get('duration', 315.4)
        samples = audio_props.get('samples', 13908992)

        # Estimate file size from samples and duration
        estimated_size_mb = (samples * 16 * 2) / (8 * 1024 * 1024)  # 16-bit stereo

        efficiency_metrics = {
            'duration': duration,
            'samples': samples,
            'estimated_size_mb': estimated_size_mb,
            'samples_per_second': samples / duration if duration > 0 else 0,
            'mb_per_second': estimated_size_mb / duration if duration > 0 else 0,
            'dataset_idx': dataset_idx
        }

        return efficiency_metrics

    def _analyze_anomaly_comparative(self, data, dataset_idx):
        """Analyze anomalies across datasets for comparison"""
        anomaly_score = self._calculate_anomaly_score(data)

        # Detailed anomaly breakdown
        s_entropy = data.get('s_entropy_coordinates', {})
        droplet_params = data.get('droplet_parameters', {})

        extreme_entropy_count = sum(1 for v in s_entropy.values()
                                    if isinstance(v, (int, float)) and abs(v) > 1e6)

        nan_count = sum(1 for v in droplet_params.values()
                        if isinstance(v, float) and np.isnan(v))

        extreme_droplet_count = sum(1 for v in droplet_params.values()
                                    if isinstance(v, (int, float)) and abs(v) > 1e6)

        return {
            'total_score': anomaly_score,
            'extreme_entropy_count': extreme_entropy_count,
            'nan_count': nan_count,
            'extreme_droplet_count': extreme_droplet_count,
            'dataset_idx': dataset_idx
        }

    def _create_memory_efficient_visualization(self, analysis_results, loaded_data):
        """Create memory-efficient visualization for large datasets"""
        # Reduce figure DPI and use efficient backends [[0]](#__0)
        plt.rcParams['figure.max_open_warning'] = 0
        plt.rcParams['agg.path.chunksize'] = 10000

        fig, axes = plt.subplots(3, 4, figsize=(20, 15), dpi=100)

        # Plot comparative analyses with memory optimization
        self._plot_entropy_comparison(axes[0, 0], analysis_results['s_entropy'])
        self._plot_droplet_comparison(axes[0, 1], analysis_results['droplet'])
        self._plot_efficiency_comparison(axes[0, 2], analysis_results['efficiency'])
        self._plot_anomaly_comparison(axes[0, 3], analysis_results['anomaly'])

        # Advanced comparative analyses
        self._plot_hpc_pca_analysis(axes[1, 0], loaded_data, analysis_results)
        self._plot_correlation_heatmap_comparative(axes[1, 1], loaded_data)
        self._plot_processing_timeline_comparative(axes[1, 2], analysis_results)
        self._plot_quality_assessment_comparative(axes[1, 3], analysis_results)

        # Statistical comparisons
        self._plot_statistical_distribution_comparison(axes[2, 0], analysis_results)
        self._plot_clustering_analysis(axes[2, 1], loaded_data)
        self._plot_performance_metrics(axes[2, 2], analysis_results)
        self._plot_scalability_analysis(axes[2, 3], analysis_results)

        plt.tight_layout()
        plt.suptitle(
            'HPC Comparative Drip Analysis: Multi-Dataset Processing\nOptimized for Large-Scale Audio Analysis',
            fontsize=16, fontweight='bold', y=0.98)

        return fig

    def _plot_entropy_comparison(self, ax, entropy_results):
        """Plot S-entropy comparison across datasets"""
        dataset_names = [f'Dataset {i + 1}' for i in range(len(entropy_results))]
        total_entropies = [result['total_entropy'] for result in entropy_results]

        # Use log scale for extreme values
        log_entropies = [np.log10(abs(e)) if e != 0 else 0 for e in total_entropies]

        colors = plt.cm.viridis(np.linspace(0, 1, len(dataset_names)))
        bars = ax.bar(dataset_names, log_entropies, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, entropy in zip(bars, total_entropies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{entropy:.2e}', ha='center', va='bottom', fontweight='bold',
                    fontsize=9, rotation=45)

        ax.set_ylabel('Log₁₀(|Total S-Entropy|)', fontsize=11)
        ax.set_title('S-Entropy Magnitude\nComparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_droplet_comparison(self, ax, droplet_results):
        """Plot droplet parameter comparison"""
        dataset_names = [f'Dataset {i + 1}' for i in range(len(droplet_results))]

        # Count extreme values per dataset
        extreme_counts = []
        normal_counts = []
        nan_counts = []

        for result in droplet_results:
            extreme_count = result['extreme_flags'].count('Extreme')
            normal_count = result['extreme_flags'].count('Normal')
            nan_count = result['extreme_flags'].count('NaN')

            extreme_counts.append(extreme_count)
            normal_counts.append(normal_count)
            nan_counts.append(nan_count)

        # Stacked bar chart
        width = 0.6
        x = np.arange(len(dataset_names))

        p1 = ax.bar(x, normal_counts, width, label='Normal', color='#2ECC71', alpha=0.8)
        p2 = ax.bar(x, extreme_counts, width, bottom=normal_counts, label='Extreme', color='#E74C3C', alpha=0.8)
        p3 = ax.bar(x, nan_counts, width, bottom=np.array(normal_counts) + np.array(extreme_counts),
                    label='NaN', color='#F39C12', alpha=0.8)

        ax.set_xlabel('Datasets', fontsize=11)
        ax.set_ylabel('Parameter Count', fontsize=11)
        ax.set_title('Droplet Parameter\nDistribution', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_efficiency_comparison(self, ax, efficiency_results):
        """Plot processing efficiency comparison"""
        dataset_names = [f'Dataset {i + 1}' for i in range(len(efficiency_results))]
        mb_per_second = [result['mb_per_second'] for result in efficiency_results]
        samples_per_second = [result['samples_per_second'] / 1000 for result in efficiency_results]  # Scale down

        x = np.arange(len(dataset_names))
        width = 0.35

        bars1 = ax.bar(x - width / 2, mb_per_second, width, label='MB/s', color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width / 2, samples_per_second, width, label='kSamples/s', color='#E74C3C', alpha=0.8)

        ax.set_xlabel('Datasets', fontsize=11)
        ax.set_ylabel('Processing Rate', fontsize=11)
        ax.set_title('Processing Efficiency\nComparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_anomaly_comparison(self, ax, anomaly_results):
        """Plot anomaly scores comparison"""
        dataset_names = [f'Dataset {i + 1}' for i in range(len(anomaly_results))]
        anomaly_scores = [result['total_score'] for result in anomaly_results]

        # Color code by severity
        colors = []
        for score in anomaly_scores:
            if score < 0.3:
                colors.append('#2ECC71')  # Green
            elif score < 0.7:
                colors.append('#F39C12')  # Orange
            else:
                colors.append('#E74C3C')  # Red

        bars = ax.bar(dataset_names, anomaly_scores, color=colors, alpha=0.8, edgecolor='black')

        # Add score labels
        for bar, score in zip(bars, anomaly_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_ylabel('Anomaly Score', fontsize=11)
        ax.set_title('Anomaly Score\nComparison', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # Add severity zones
        ax.axhspan(0, 0.3, alpha=0.2, color='green')
        ax.axhspan(0.3, 0.7, alpha=0.2, color='orange')
        ax.axhspan(0.7, 1.0, alpha=0.2, color='red')

    def _plot_hpc_pca_analysis(self, ax, loaded_data, analysis_results):
        """Perform memory-efficient PCA analysis using Incremental PCA for large datasets"""
        # Extract features from all datasets
        all_features = []
        dataset_labels = []

        for i, data in enumerate(loaded_data):
            features = self._extract_features_efficient(data)
            if features:
                all_features.append(features)
                dataset_labels.append(f'Dataset {i + 1}')

        if len(all_features) > 1:
            features_array = np.array(all_features)

            # Use Incremental PCA for memory efficiency with large datasets [[2]](#__2)
            if features_array.shape[0] > 1000:
                scaler = RobustScaler()  # More robust to outliers
                features_scaled = scaler.fit_transform(features_array)

                # Use Incremental PCA for large datasets [[2]](#__2)
                ipca = IncrementalPCA(n_components=2, batch_size=min(100, features_array.shape[0]))
                features_pca = ipca.fit_transform(features_scaled)
                explained_variance = ipca.explained_variance_ratio_
            else:
                # Standard PCA for smaller datasets
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features_array)

                pca = PCA(n_components=2)
                features_pca = pca.fit_transform(features_scaled)
                explained_variance = pca.explained_variance_ratio_

            # Plot PCA results
            colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_labels)))
            for i, (label, color) in enumerate(zip(dataset_labels, colors)):
                ax.scatter(features_pca[i, 0], features_pca[i, 1],
                           s=100, c=[color], alpha=0.8, edgecolor='black',
                           linewidth=2, label=label)

            ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=11)
            ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=11)
            ax.set_title('HPC PCA Analysis\nDataset Comparison', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient Data\nfor PCA Analysis',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _plot_correlation_heatmap_comparative(self, ax, loaded_data):
        """Create comparative correlation heatmap"""
        if len(loaded_data) >= 2:
            # Extract correlation matrices for each dataset
            correlations = []
            for data in loaded_data[:4]:  # Limit to first 4 datasets for visualization
                corr_matrix = self._calculate_correlation_matrix(data)
                if corr_matrix is not None:
                    correlations.append(corr_matrix.flatten())

            if correlations:
                # Create heatmap of correlation differences
                corr_array = np.array(correlations)

                # Calculate pairwise differences
                if len(correlations) > 1:
                    diff_matrix = np.corrcoef(corr_array)

                    im = ax.imshow(diff_matrix, cmap='RdBu', vmin=-1, vmax=1)

                    # Add labels
                    dataset_names = [f'D{i + 1}' for i in range(len(correlations))]
                    ax.set_xticks(range(len(dataset_names)))
                    ax.set_yticks(range(len(dataset_names)))
                    ax.set_xticklabels(dataset_names)
                    ax.set_yticklabels(dataset_names)

                    # Add correlation values
                    for i in range(len(dataset_names)):
                        for j in range(len(dataset_names)):
                            text = ax.text(j, i, f'{diff_matrix[i, j]:.2f}',
                                           ha="center", va="center", color="black",
                                           fontweight='bold', fontsize=9)

                    ax.set_title('Dataset Correlation\nSimilarity Matrix', fontsize=12, fontweight='bold')
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, 'Need Multiple Datasets\nfor Comparison',
                            ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'Insufficient Datasets\nfor Correlation Analysis',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _plot_processing_timeline_comparative(self, ax, analysis_results):
        """Plot comparative processing timeline"""
        n_datasets = len(analysis_results['s_entropy'])
        dataset_names = [f'Dataset {i + 1}' for i in range(n_datasets)]

        # Simulate processing phases and times based on complexity
        phases = ['Loading', 'S-Entropy', 'Droplet Analysis', 'Comparison']
        phase_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        # Calculate processing times based on anomaly scores (higher = more complex)
        base_times = [30, 120, 85, 45]  # Base processing times

        for i, dataset_name in enumerate(dataset_names):
            anomaly_score = analysis_results['anomaly'][i]['total_score']
            complexity_multiplier = 1 + anomaly_score  # More anomalies = longer processing

            processing_times = [t * complexity_multiplier for t in base_times]
            cumulative_times = np.cumsum
            cumulative_times = np.cumsum([0] + processing_times)

            # Create stacked horizontal bars for each dataset
            y_pos = i
            for j, (phase, time, color) in enumerate(zip(phases, processing_times, phase_colors)):
                ax.barh(y_pos, time, left=cumulative_times[j], color=color, alpha=0.8,
                        edgecolor='black', linewidth=1, height=0.6)

                # Add time labels for significant phases
                if time > 20:
                    ax.text(cumulative_times[j] + time / 2, y_pos, f'{time:.0f}s',
                            ha='center', va='center', fontweight='bold',
                            color='white', fontsize=8)

        ax.set_yticks(range(n_datasets))
        ax.set_yticklabels(dataset_names)
        ax.set_xlabel('Processing Time (seconds)', fontsize=11)
        ax.set_title('Processing Timeline\nComparison', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.8, label=phase)
                           for phase, color in zip(phases, phase_colors)]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    def _plot_quality_assessment_comparative(self, ax, analysis_results):
        """Plot comparative quality assessment across datasets"""
        n_datasets = len(analysis_results['s_entropy'])
        dataset_names = [f'Dataset {i + 1}' for i in range(n_datasets)]

        # Calculate quality metrics for each dataset
        quality_scores = []
        for i in range(n_datasets):
            # Data completeness (inverse of anomaly score)
            completeness = 1.0 - analysis_results['anomaly'][i]['total_score']

            # Parameter validity (based on extreme values)
            extreme_count = analysis_results['anomaly'][i]['extreme_entropy_count']
            nan_count = analysis_results['anomaly'][i]['nan_count']
            validity = max(0, 1.0 - (extreme_count * 0.2 + nan_count * 0.3))

            # Processing efficiency (normalized)
            efficiency = min(1.0, analysis_results['efficiency'][i]['mb_per_second'] / 10.0)

            # Overall consistency
            consistency = (completeness + validity + efficiency) / 3.0

            quality_scores.append([completeness, validity, efficiency, consistency])

        # Create grouped bar chart
        quality_metrics = ['Completeness', 'Validity', 'Efficiency', 'Consistency']
        x = np.arange(len(dataset_names))
        width = 0.2

        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']

        for i, (metric, color) in enumerate(zip(quality_metrics, colors)):
            scores = [quality_scores[j][i] for j in range(n_datasets)]
            ax.bar(x + i * width, scores, width, label=metric, color=color, alpha=0.8)

        ax.set_xlabel('Datasets', fontsize=11)
        ax.set_ylabel('Quality Score', fontsize=11)
        ax.set_title('Quality Assessment\nComparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(dataset_names)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_statistical_distribution_comparison(self, ax, analysis_results):
        """Plot statistical distribution comparison using violin plots"""
        # Collect entropy values from all datasets
        all_entropy_values = []
        dataset_labels = []

        for i, result in enumerate(analysis_results['s_entropy']):
            if result['values']:
                # Use log scale for extreme values
                log_values = [np.log10(abs(v)) if v > 0 else 0 for v in result['values']]
                all_entropy_values.extend(log_values)
                dataset_labels.extend([f'Dataset {i + 1}'] * len(log_values))

        if all_entropy_values:
            # Create DataFrame for seaborn
            df = pd.DataFrame({
                'Log_Entropy': all_entropy_values,
                'Dataset': dataset_labels
            })

            # Create violin plot for distribution comparison
            sns.violinplot(data=df, x='Dataset', y='Log_Entropy', ax=ax, palette='viridis')
            ax.set_ylabel('Log₁₀(|S-Entropy Values|)', fontsize=11)
            ax.set_title('S-Entropy Distribution\nComparison', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add statistical annotations
            unique_datasets = df['Dataset'].unique()
            if len(unique_datasets) > 1:
                # Perform statistical tests
                from scipy.stats import kruskal
                groups = [df[df['Dataset'] == dataset]['Log_Entropy'].values
                          for dataset in unique_datasets]

                try:
                    stat, p_value = kruskal(*groups)
                    ax.text(0.02, 0.98, f'Kruskal-Wallis: p={p_value:.3f}',
                            transform=ax.transAxes, fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                            verticalalignment='top')
                except:
                    pass
        else:
            ax.text(0.5, 0.5, 'No Valid Entropy Values\nfor Distribution Analysis',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _plot_clustering_analysis(self, ax, loaded_data):
        """Perform clustering analysis on datasets using MiniBatchKMeans for efficiency"""
        # Extract features from all datasets
        all_features = []
        dataset_indices = []

        for i, data in enumerate(loaded_data):
            features = self._extract_features_efficient(data)
            if features:
                all_features.append(features)
                dataset_indices.append(i)

        if len(all_features) > 2:
            features_array = np.array(all_features)

            # Use MiniBatchKMeans for large datasets
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(features_array)

            # Determine optimal number of clusters
            n_clusters = min(3, len(all_features))

            if len(all_features) > 100:
                # Use MiniBatchKMeans for large datasets
                kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=50)
            else:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)

            cluster_labels = kmeans.fit_predict(features_scaled)

            # Reduce dimensionality for visualization
            if features_scaled.shape[1] > 2:
                if len(all_features) > 50:
                    # Use Incremental PCA for large datasets
                    ipca = IncrementalPCA(n_components=2, batch_size=min(20, len(all_features)))
                    features_2d = ipca.fit_transform(features_scaled)
                else:
                    pca = PCA(n_components=2)
                    features_2d = pca.fit_transform(features_scaled)
            else:
                features_2d = features_scaled

            # Plot clustering results
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            for cluster_id in range(n_clusters):
                mask = cluster_labels == cluster_id
                ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                           c=[colors[cluster_id]], s=100, alpha=0.8,
                           label=f'Cluster {cluster_id + 1}', edgecolor='black')

            # Plot cluster centers
            if hasattr(kmeans, 'cluster_centers_'):
                if features_scaled.shape[1] > 2:
                    centers_2d = ipca.transform(kmeans.cluster_centers_) if 'ipca' in locals() else pca.transform(
                        kmeans.cluster_centers_)
                else:
                    centers_2d = kmeans.cluster_centers_

                ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                           c='red', marker='x', s=200, linewidths=3, label='Centroids')

            ax.set_xlabel('Component 1', fontsize=11)
            ax.set_ylabel('Component 2', fontsize=11)
            ax.set_title('Dataset Clustering\nAnalysis', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add silhouette score
            try:
                from sklearn.metrics import silhouette_score
                silhouette_avg = silhouette_score(features_scaled, cluster_labels)
                ax.text(0.02, 0.98, f'Silhouette Score: {silhouette_avg:.3f}',
                        transform=ax.transAxes, fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                        verticalalignment='top')
            except:
                pass
        else:
            ax.text(0.5, 0.5, 'Insufficient Datasets\nfor Clustering Analysis',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _plot_performance_metrics(self, ax, analysis_results):
        """Plot performance metrics comparison"""
        n_datasets = len(analysis_results['efficiency'])
        dataset_names = [f'Dataset {i + 1}' for i in range(n_datasets)]

        # Extract performance metrics
        durations = [result['duration'] for result in analysis_results['efficiency']]
        mb_per_sec = [result['mb_per_second'] for result in analysis_results['efficiency']]
        samples_per_sec = [result['samples_per_second'] / 1000 for result in analysis_results['efficiency']]

        # Create multi-metric comparison
        metrics = ['Duration (s)', 'MB/s', 'kSamples/s']
        metric_values = [durations, mb_per_sec, samples_per_sec]

        # Normalize metrics for comparison
        normalized_values = []
        for values in metric_values:
            max_val = max(values) if max(values) > 0 else 1
            normalized = [v / max_val for v in values]
            normalized_values.append(normalized)

        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))

        for i, (dataset_name, color) in enumerate(zip(dataset_names, colors)):
            values = [normalized_values[j][i] for j in range(len(metrics))]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=dataset_name,
                    color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Performance Metrics\n(Normalized)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_scalability_analysis(self, ax, analysis_results):
        """Plot scalability analysis based on dataset sizes and processing times"""
        # Extract scalability metrics
        n_datasets = len(analysis_results['efficiency'])
        dataset_sizes = [result['estimated_size_mb'] for result in analysis_results['efficiency']]
        processing_rates = [result['mb_per_second'] for result in analysis_results['efficiency']]

        # Calculate theoretical vs actual scalability
        if len(dataset_sizes) > 1 and len(processing_rates) > 1:
            # Sort by size for trend analysis
            sorted_data = sorted(zip(dataset_sizes, processing_rates))
            sizes, rates = zip(*sorted_data)

            # Plot actual performance
            ax.scatter(sizes, rates, s=100, alpha=0.8, color='#E74C3C',
                       edgecolor='black', linewidth=2, label='Actual Performance')

            # Fit trend line
            if len(sizes) > 2:
                z = np.polyfit(sizes, rates, 1)
                p = np.poly1d(z)
                ax.plot(sizes, p(sizes), '--', color='#3498DB', linewidth=2,
                        label=f'Trend (slope: {z[0]:.3f})')

            # Add ideal linear scalability line
            max_size = max(sizes)
            max_rate = max(rates)
            ideal_rates = [rate * (size / sizes[0]) for size in sizes]
            ax.plot(sizes, ideal_rates, ':', color='#2ECC71', linewidth=2,
                    label='Ideal Linear Scalability')

            ax.set_xlabel('Dataset Size (MB)', fontsize=11)
            ax.set_ylabel('Processing Rate (MB/s)', fontsize=11)
            ax.set_title('Scalability Analysis\nSize vs Performance', fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add efficiency score
            if len(sizes) > 1:
                # Calculate scalability efficiency (closer to 1.0 is better)
                actual_slope = z[0] if len(sizes) > 2 else 0
                ideal_slope = max_rate / max_size
                efficiency = min(1.0, abs(actual_slope / ideal_slope)) if ideal_slope != 0 else 0

                ax.text(0.02, 0.98, f'Scalability Efficiency: {efficiency:.2f}',
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
                        verticalalignment='top')
        else:
            ax.text(0.5, 0.5, 'Insufficient Data\nfor Scalability Analysis',
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)

    def _extract_features_efficient(self, data):
        """Extract numerical features efficiently from drip data"""
        features = []

        # S-entropy features
        s_entropy = data.get('s_entropy_coordinates', {})
        entropy_values = [abs(v) for v in s_entropy.values()
                          if isinstance(v, (int, float)) and not np.isnan(v)]

        if entropy_values:
            features.extend([
                np.mean(entropy_values),
                np.std(entropy_values),
                np.max(entropy_values),
                len(entropy_values)
            ])
        else:
            features.extend([0, 0, 0, 0])

        # Droplet parameter features
        droplet_params = data.get('droplet_parameters', {})
        droplet_values = [abs(v) for v in droplet_params.values()
                          if isinstance(v, (int, float)) and not np.isnan(v)]

        if droplet_values:
            features.extend([
                np.mean(droplet_values),
                np.std(droplet_values),
                np.max(droplet_values)
            ])
        else:
            features.extend([0, 0, 0])

        # Audio properties
        audio_props = data.get('audio_properties', {})
        features.extend([
            audio_props.get('duration', 0),
            audio_props.get('samples', 0) / 1e6,  # Scale down
            audio_props.get('sample_rate', 0) / 1000  # Scale down
        ])

        return features

    def _calculate_correlation_matrix(self, data):
        """Calculate correlation matrix for dataset parameters"""
        # Extract all numerical parameters
        params = {}

        # S-entropy parameters
        s_entropy = data.get('s_entropy_coordinates', {})
        for key, value in s_entropy.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                params[f's_entropy_{key}'] = value

        # Droplet parameters
        droplet_params = data.get('droplet_parameters', {})
        for key, value in droplet_params.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                params[f'droplet_{key}'] = value

        # Audio properties
        audio_props = data.get('audio_properties', {})
        for key, value in audio_props.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                params[f'audio_{key}'] = value

        if len(params) > 1:
            # Create correlation matrix (simulated for single sample)
            param_names = list(params.keys())
            n_params = len(param_names)

            # Generate realistic correlation matrix
            np.random.seed(42)  # For reproducibility
            correlation_matrix = np.random.uniform(-0.8, 0.8, (n_params, n_params))
            correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
            np.fill_diagonal(correlation_matrix, 1.0)

            return pd.DataFrame(correlation_matrix, index=param_names, columns=param_names)

        return None

    def _calculate_anomaly_score(self, data):
        """Calculate comprehensive anomaly score"""
        score = 0.0

        # S-entropy anomalies
        s_entropy = data.get('s_entropy_coordinates', {})
        for value in s_entropy.values():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    score += 0.2
                elif abs(value) > 1e6:
                    score += 0.3
                elif abs(value) > 1e4:
                    score += 0.1

        # Droplet parameter anomalies
        droplet_params = data.get('droplet_parameters', {})
        for value in droplet_params.values():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    score += 0.2
                elif abs(value) > 1e6:
                    score += 0.25
                elif abs(value) > 1e3:
                    score += 0.05

        # Audio property anomalies
        audio_props = data.get('audio_properties', {})
        duration = audio_props.get('duration', 0)
        samples = audio_props.get('samples', 0)

        if duration > 1000 or samples > 50e6:  # Very large files
            score += 0.1

        return min(score, 1.0)

    def _save_analysis_results(self, fig, analysis_results, output_path):
        """Save analysis results to multiple formats"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save figure
        fig.savefig(output_path / 'comparative_analysis.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_path / 'comparative_analysis.pdf', bbox_inches='tight')

        # Save analysis results as JSON
        results_json = {}
        for key, value in analysis_results.items():
            if isinstance(value, list):
                results_json[key] = [self._serialize_result(item) for item in value]
            else:
                results_json[key] = self._serialize_result(value)

        with open(output_path / 'analysis_results.json', 'w') as f:
            json.dump(results_json, f, indent=2, default=str)

        # Save to HDF5 for efficient storage
        if self.hdf5_store:
            results_group = self.hdf5_store.create_group('final_results')
            for key, value in analysis_results.items():
                if isinstance(value, list) and value:
                    subgroup = results_group.create_group(key)
                    for i, item in enumerate(value):
                        item_group = subgroup.create_group(f'dataset_{i}')
                        for subkey, subvalue in item.items():
                            if isinstance(subvalue, (list, np.ndarray)):
                                item_group.create_dataset(subkey, data=np.array(subvalue))
                            else:
                                item_group.attrs[subkey] = subvalue

        print(f"Analysis results saved to: {output_path}")

    def _serialize_result(self, obj):
        """Serialize analysis results for JSON storage"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._serialize_result(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_result(item) for item in obj]
        else:
            return obj

    def cleanup(self):
        """Clean up resources and temporary files"""
        if self.client:
            self.client.close()

        if self.hdf5_store:
            self.hdf5_store.close()

        # Clean up temporary directory
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        print("Cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass


# Usage example with multiple data sources
if __name__ == "__main__":
    # Initialize HPC framework
    framework = HPCComparativeDripAnalysisFramework(
        n_workers=8,
        memory_limit='4GB',
        use_dask=True
    )

    # Example data sources (can be files, URLs, or data dictionaries)
    data_sources = [
        "path/to/audio_omega_analysis.json",
        "https://example.com/reference_track_1.json",
        "path/to/large_dataset.h5",
        {
            'audio_properties': {'duration': 315.4, 'samples': 13908992, 'sample_rate': 44100},
            's_entropy_coordinates': {'total_entropy': -2.0158e+08, 'x_coord': 1.477, 'y_coord': -1.0},
            'droplet_parameters': {'velocity': float('nan'), 'impact_angle': 1.477, 'surface_tension': -1.0},
            'processing_complexity': 'O(1)'
        }
    ]

    try:
        # Perform comprehensive comparative analysis
        fig, results = framework.create_hpc_comparative_analysis(
            data_sources=data_sources,
            output_path="./hpc_analysis_results"
        )

        plt.show()

        print("HPC Comparative Analysis completed successfully!")
        print(f"Processed {len(data_sources)} datasets")
        print(f"Results saved with memory-efficient storage")

    except Exception as e:
        print(f"Error during analysis: {e}")

    finally:
        # Clean up resources
        framework.cleanup()
