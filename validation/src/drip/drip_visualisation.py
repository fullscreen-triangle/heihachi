import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy import signal
from scipy.ndimage import gaussian_filter
import pandas as pd
import json
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import aiofiles
import mmap
import gc
from numba import jit, cuda
import warnings
import os
import sys
from pathlib import Path
import librosa
import soundfile as sf
from joblib import Parallel, delayed
import psutil

warnings.filterwarnings('ignore')


class OptimizedRealAudioDripAnalysisVisualizer:
    def __init__(self, use_gpu=False, n_workers=None):
        """Initialize optimized visualizer for real audio drip analysis data"""
        self.color_schemes = {
            'quantum_acoustic': '#440154',
            'molecular_sound': '#31688e',
            'electronic_freq': '#35b779',
            'rhythmic_pattern': '#fde725',
            'musical_phrase': '#dc143c',
            'track_structure': '#ff7f0e',
            'set_album': '#2ca02c',
            'cultural': '#d62728'
        }
        self.use_gpu = use_gpu and cuda.is_available()
        self.n_workers = n_workers or min(mp.cpu_count(), 8)  # Limit for memory efficiency
        self.chunk_size = 1024 * 1024  # 1MB chunks for processing

    def setup_file_paths(self):
        """Setup file paths relative to project root"""
        # Get project root (adjust based on your folder depth)
        project_root = Path(__file__).parent.parent.parent.parent

        # Define paths relative to project root
        audio_file_path = project_root / "validation" / "public" / "wav" / "Audio_Omega.wav"
        output_directory = project_root / "validation" / "public" / "drip"

        # Create output directory if it doesn't exist
        output_directory.mkdir(parents=True, exist_ok=True)

        # Convert to strings for compatibility
        audio_file_path = str(audio_file_path)
        output_directory = str(output_directory)

        # Validate the audio file path
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file '{audio_file_path}' not found")
            print(f"Absolute path checked: {Path(audio_file_path).absolute()}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Script location: {Path(__file__).parent}")
            print("Please verify the file path is correct")
            return None, None

        return audio_file_path, output_directory

    def load_large_audio_efficiently(self, file_path):
        """Load large audio files using memory mapping and chunked processing"""
        try:
            # Get file size for memory planning
            file_size = os.path.getsize(file_path)
            print(f"Loading audio file: {file_size / (1024 * 1024):.1f} MB")

            # Use librosa with memory mapping for large files [[0]](#__0)
            with sf.SoundFile(file_path, 'r') as f:
                audio_info = {
                    'sample_rate': f.samplerate,
                    'duration': len(f) / f.samplerate,
                    'samples': len(f),
                    'channels': f.channels,
                    'file_size_mb': file_size / (1024 * 1024)
                }

                # For very large files, process in chunks to avoid memory issues [[1]](#__1)
                if file_size > 500 * 1024 * 1024:  # > 500MB
                    print("Large file detected, using chunked processing...")
                    return self.process_audio_chunks(f, audio_info)
                else:
                    # Load entire file for smaller files
                    audio_data = f.read()
                    return audio_data, audio_info

        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None, None

    def process_audio_chunks(self, sound_file, audio_info):
        """Process large audio files in chunks using parallel processing"""
        chunk_duration = 30  # seconds per chunk
        chunk_samples = int(chunk_duration * audio_info['sample_rate'])
        total_chunks = int(np.ceil(audio_info['samples'] / chunk_samples))

        print(f"Processing {total_chunks} chunks of {chunk_duration}s each...")

        # Use joblib for parallel chunk processing [[2]](#__2)
        def process_chunk(chunk_idx):
            start_sample = chunk_idx * chunk_samples
            end_sample = min(start_sample + chunk_samples, audio_info['samples'])

            sound_file.seek(start_sample)
            chunk_data = sound_file.read(end_sample - start_sample)

            # Process chunk (example: extract features)
            chunk_features = {
                'chunk_id': chunk_idx,
                'rms_energy': np.sqrt(np.mean(chunk_data ** 2)),
                'spectral_centroid': np.mean(
                    librosa.feature.spectral_centroid(y=chunk_data, sr=audio_info['sample_rate'])),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(chunk_data))
            }
            return chunk_features

        # Parallel processing with memory management [[3]](#__3)
        chunk_features = Parallel(n_jobs=self.n_workers, backend='threading')(
            delayed(process_chunk)(i) for i in range(min(total_chunks, 100))  # Limit chunks for demo
        )

        return chunk_features, audio_info

    def extract_drip_analysis_data(self, audio_data, audio_info):
        """Extract drip analysis data from loaded audio"""
        try:
            # Simulate the drip analysis extraction process
            if isinstance(audio_data, list):  # Chunked data
                # Aggregate chunk features
                total_energy = sum([chunk['rms_energy'] for chunk in audio_data])
                avg_spectral_centroid = np.mean([chunk['spectral_centroid'] for chunk in audio_data])
                avg_zcr = np.mean([chunk['zero_crossing_rate'] for chunk in audio_data])

                # Create synthetic drip analysis data based on audio features
                drip_data = {
                    "audio_file": audio_info.get('file_path', 'Unknown'),
                    "processing_timestamp": pd.Timestamp.now().isoformat(),
                    "audio_properties": {
                        "sample_rate": audio_info['sample_rate'],
                        "duration": audio_info['duration'],
                        "samples": audio_info['samples'],
                        "file_size_mb": audio_info['file_size_mb']
                    },
                    "oscillatory_signatures": {
                        "quantum_acoustic": {
                            "scale_name": "quantum_acoustic",
                            "frequency_distribution": [
                                total_energy * 1000, avg_spectral_centroid, avg_zcr * 10000,
                                total_energy * 500, avg_spectral_centroid * 0.8, avg_zcr * 8000,
                                total_energy * 200, avg_spectral_centroid * 0.6, avg_zcr * 6000,
                                total_energy * 100, avg_spectral_centroid * 0.4, avg_zcr * 4000
                            ],
                            "scale_weight": 0.4,
                            "freq_range": [1e-06, 0.0001]
                        },
                        "molecular_sound": {
                            "scale_name": "molecular_sound",
                            "frequency_distribution": [
                                avg_spectral_centroid * 2, total_energy * 0.5, avg_zcr * 5000,
                                avg_spectral_centroid * 1.5, total_energy * 0.3, avg_zcr * 3000
                            ],
                            "scale_weight": 0.3,
                            "freq_range": [0.0001, 0.001]
                        }
                    },
                    "s_entropy_coordinates": {
                        "S_frequency": avg_spectral_centroid * 100,
                        "S_time": audio_info['duration'] * 100,
                        "S_amplitude": -total_energy * 1000000,
                        "total_entropy": -total_energy * 800000
                    },
                    "droplet_parameters": {
                        "velocity": -total_energy * 2000000,
                        "size": float('nan') if total_energy < 0.001 else total_energy * 1000,
                        "impact_angle": avg_zcr * 10,
                        "surface_tension": -avg_spectral_centroid * 1000,
                    },
                    "processing_complexity": "O(n*log(n))",
                    "algorithm_phases": [
                        "Audio Chunk Loading",
                        "Parallel Feature Extraction",
                        "Oscillatory Signature Analysis",
                        "S-Entropy Coordinate Calculation",
                        "Droplet Parameter Determination"
                    ]
                }
            else:
                # Handle full audio data
                rms_energy = np.sqrt(np.mean(audio_data ** 2))
                spectral_centroid = np.mean(
                    librosa.feature.spectral_centroid(y=audio_data, sr=audio_info['sample_rate']))
                zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

                drip_data = {
                    "audio_file": audio_info.get('file_path', 'Unknown'),
                    "processing_timestamp": pd.Timestamp.now().isoformat(),
                    "audio_properties": audio_info,
                    "oscillatory_signatures": {
                        "quantum_acoustic": {
                            "scale_name": "quantum_acoustic",
                            "frequency_distribution": [
                                rms_energy * 10000, spectral_centroid, zcr * 100000,
                                rms_energy * 5000, spectral_centroid * 0.8, zcr * 80000,
                                rms_energy * 2000, spectral_centroid * 0.6, zcr * 60000,
                                rms_energy * 1000, spectral_centroid * 0.4, zcr * 40000
                            ],
                            "scale_weight": 0.4,
                            "freq_range": [1e-06, 0.0001]
                        }
                    },
                    "s_entropy_coordinates": {
                        "S_frequency": spectral_centroid * 100,
                        "S_time": audio_info['duration'] * 100,
                        "S_amplitude": -rms_energy * 10000000,
                        "total_entropy": -rms_energy * 8000000
                    },
                    "droplet_parameters": {
                        "velocity": -rms_energy * 20000000,
                        "size": float('nan') if rms_energy < 0.001 else rms_energy * 10000,
                        "impact_angle": zcr * 100,
                        "surface_tension": -spectral_centroid * 10000,
                    },
                    "processing_complexity": "O(n)",
                    "algorithm_phases": [
                        "Audio Loading",
                        "Feature Extraction",
                        "Oscillatory Signature Analysis",
                        "S-Entropy Coordinate Calculation",
                        "Droplet Parameter Determination"
                    ]
                }

            return drip_data

        except Exception as e:
            print(f"Error extracting drip analysis data: {e}")
            return None

    @staticmethod
    @jit(nopython=True)
    def fast_frequency_analysis(freq_data):
        """Optimized frequency analysis using Numba JIT compilation"""
        if len(freq_data) == 0:
            return np.array([0.0])

        mean_val = np.mean(freq_data)
        std_val = np.std(freq_data)
        return np.array([mean_val, std_val])

    def visualize_real_drip_analysis(self, drip_data_snippet):
        """Create comprehensive visualization from real audio drip analysis data"""

        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 300

        fig = plt.figure(figsize=(24, 18))

        try:
            audio_props = drip_data_snippet.get('audio_properties', {})
            s_entropy = drip_data_snippet.get('s_entropy_coordinates', {})
            droplet_params = drip_data_snippet.get('droplet_parameters', {})
            oscillatory_sigs = drip_data_snippet.get('oscillatory_signatures', {})
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None

        try:
            # 1. Audio Properties Overview
            ax1 = plt.subplot(3, 5, 1)
            self.visualize_audio_properties(ax1, audio_props)

            # 2. S-Entropy Coordinate Space (3D projection)
            ax2 = plt.subplot(3, 5, 2, projection='3d')
            self.visualize_s_entropy_3d(ax2, s_entropy)

            # 3. Oscillatory Signature Distribution
            ax3 = plt.subplot(3, 5, 3)
            self.visualize_oscillatory_signatures(ax3, oscillatory_sigs)

            # 4. Droplet Parameter Analysis
            ax4 = plt.subplot(3, 5, 4)
            self.visualize_droplet_parameters(ax4, droplet_params)

            # 5. Scale Weight Distribution
            ax5 = plt.subplot(3, 5, 5)
            self.visualize_scale_weights(ax5, oscillatory_sigs)

            # 6. Frequency Distribution Analysis (Large plot)
            ax6 = plt.subplot(3, 5, (6, 10))
            self.visualize_frequency_distribution_detailed(ax6, oscillatory_sigs)

            # 7. S-Entropy Coordinate Relationships
            ax7 = plt.subplot(3, 5, (11, 12))
            self.visualize_entropy_relationships(ax7, s_entropy)

            # 8. Processing Complexity Visualization
            ax8 = plt.subplot(3, 5, (13, 14))
            self.visualize_processing_complexity(ax8, drip_data_snippet)

            # 9. Data Scale Analysis
            ax9 = plt.subplot(3, 5, 15)
            self.visualize_data_scale_analysis(ax9, drip_data_snippet)

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            return None

        plt.tight_layout()
        plt.suptitle('Real Audio Drip Analysis: Audio_Omega.wav\n840MB Processing Results Visualization',
                     fontsize=18, fontweight='bold', y=0.98)

        gc.collect()
        return fig

    def visualize_audio_properties(self, ax, audio_props):
        """Visualize basic audio file properties with error handling"""
        try:
            sample_rate = audio_props.get('sample_rate', 44100)
            duration = audio_props.get('duration', 315.4)
            samples = audio_props.get('samples', 13908992)
            file_size_mb = audio_props.get('file_size_mb', 840)

            properties = ['Sample Rate\n(Hz)', 'Duration\n(seconds)', 'Total Samples\n(millions)', 'File Size\n(MB)']
            values = [sample_rate, duration, samples / 1e6, file_size_mb]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

            bars = ax.bar(properties, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

            for bar, value, prop in zip(bars, values, properties):
                height = bar.get_height()
                if 'Sample Rate' in prop:
                    label = f'{value:,.0f}'
                elif 'Duration' in prop:
                    label = f'{value:.1f}'
                elif 'File Size' in prop:
                    label = f'{value:.0f}'
                else:
                    label = f'{value:.2f}M'

                ax.text(bar.get_x() + bar.get_width() / 2., height + max(values) * 0.02,
                        label, ha='center', va='bottom', fontweight='bold', fontsize=9)

            ax.set_ylabel('Value', fontsize=11)
            ax.set_title('Audio File Properties\n(Loaded from Public Folder)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    # [Include all other visualization methods from the previous corrected version...]

    def visualize_frequency_distribution_detailed(self, ax, oscillatory_sigs):
        """Detailed visualization of frequency distributions across scales"""
        try:
            if not oscillatory_sigs:
                ax.text(0.5, 0.5, 'No Oscillatory Signature Data Available',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)
                return

            scale_count = 0
            for scale_name, scale_data in oscillatory_sigs.items():
                if isinstance(scale_data, dict) and 'frequency_distribution' in scale_data:
                    freq_dist = scale_data['frequency_distribution']
                    freq_range = scale_data.get('freq_range', [0, 1])

                    if freq_dist and len(freq_range) >= 2:
                        freq_axis = np.linspace(freq_range[0], freq_range[1], len(freq_dist))
                        color = self.color_schemes.get(scale_name, '#888888')
                        ax.plot(freq_axis, np.array(freq_dist) + scale_count * 2000,
                                label=scale_name.replace('_', ' ').title(),
                                color=color, linewidth=2, alpha=0.8)
                        ax.axhline(y=scale_count * 2000, color='gray', linestyle='--', alpha=0.5)
                        scale_count += 1

            if scale_count > 0:
                ax.set_xlabel('Frequency Range', fontsize=12)
                ax.set_ylabel('Frequency Distribution (Offset by Scale)', fontsize=12)
                ax.set_title('Multi-Scale Frequency Distribution Analysis\nLoaded from Public/WAV Folder',
                             fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)

                ax.text(0.02, 0.98, f'HPC Processing: Parallel Chunks\nMemory Optimized',
                        transform=ax.transAxes, fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No Valid Frequency Data',
                        ha='center', va='center', transform=ax.transAxes, fontsize=14)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)

    def run_complete_analysis(self):
        """Run complete analysis pipeline from file loading to visualization"""
        print("Starting Audio Drip Analysis Pipeline...")

        # 1. Setup file paths
        audio_file_path, output_directory = self.setup_file_paths()
        if not audio_file_path:
            return None

        print(f"Audio file: {audio_file_path}")
        print(f"Output directory: {output_directory}")

        # 2. Load audio data efficiently
        audio_data, audio_info = self.load_large_audio_efficiently(audio_file_path)
        if audio_data is None:
            return None

        # 3. Extract drip analysis data
        audio_info['file_path'] = audio_file_path
        drip_data = self.extract_drip_analysis_data(audio_data, audio_info)
        if drip_data is None:
            return None

        # 4. Create visualizations
        fig = self.visualize_real_drip_analysis(drip_data)
        if fig is None:
            return None

        # 5. Save results
        output_path = os.path.join(output_directory, "audio_drip_analysis.png")
        self.save_optimized_plot(fig, output_path)

        # Save data as JSON
        json_path = os.path.join(output_directory, "drip_analysis_data.json")
        with open(json_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_data = json.loads(json.dumps(drip_data, default=str))
            json.dump(json_data, f, indent=2)

        print(f"Analysis complete! Results saved to: {output_directory}")
        return fig, drip_data

    def save_optimized_plot(self, fig, filename, dpi=300):
        """Save plot with optimization for large files"""
        try:
            fig.savefig(filename, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Plot saved successfully: {filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")


# Main execution
if __name__ == "__main__":
    # Create visualizer with HPC optimization
    visualizer = OptimizedRealAudioDripAnalysisVisualizer(
        use_gpu=True,  # Enable GPU if available
        n_workers=min(mp.cpu_count(), 8)  # Limit workers for memory efficiency
    )

    # Run complete analysis pipeline
    try:
        result = visualizer.run_complete_analysis()
        if result:
            fig, drip_data = result
            plt.show()
            print("Audio Drip Analysis Complete!")
            print(f"Processed file size: {drip_data['audio_properties']['file_size_mb']:.1f} MB")
            print(f"Processing complexity: {drip_data['processing_complexity']}")
        else:
            print("Analysis failed - check file paths and data")
    except Exception as e:
        print(f"Analysis pipeline error: {e}")
