import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from pathlib import Path
import os
import sys

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def load_data(filepath):
    """Load JSON data from file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filepath}': {e}")
        return None


def plot_molecular_features(data, output_file='molecular_analysis.png'):
    """
    Visualize frequency and temporal molecular features
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Audio Molecular Features Analysis', fontsize=14, fontweight='bold', y=0.995)

    # Extract data
    freq_density = np.array(data['molecular_features']['frequency_molecules']['density'])
    temp_energy = np.array(data['molecular_features']['temporal_molecules']['energy'])

    # 1. Frequency Molecule Density Distribution
    ax1 = axes[0, 0]
    freq_bins = np.arange(len(freq_density))
    ax1.plot(freq_bins, freq_density, linewidth=1.5, color='#2E86AB', alpha=0.8)
    ax1.fill_between(freq_bins, freq_density, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Frequency Bin', fontweight='bold')
    ax1.set_ylabel('Density', fontweight='bold')
    ax1.set_title('Frequency Molecule Density Distribution', fontweight='bold')
    ax1.grid(alpha=0.3, linestyle='--')

    # Add statistics
    stats_text = f'Mean: {np.mean(freq_density):.6f}\nStd: {np.std(freq_density):.6f}\nMax: {np.max(freq_density):.6f}'
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=8)

    # 2. Frequency Density Spectrogram
    ax2 = axes[0, 1]
    # Reshape for spectrogram-like visualization
    n_segments = 50
    segment_size = len(freq_density) // n_segments
    spectrogram_data = freq_density[:segment_size * n_segments].reshape(n_segments, segment_size).T

    im = ax2.imshow(spectrogram_data, aspect='auto', cmap='viridis', origin='lower',
                    interpolation='bilinear')
    ax2.set_xlabel('Time Segment', fontweight='bold')
    ax2.set_ylabel('Frequency Bin', fontweight='bold')
    ax2.set_title('Frequency Density Spectrogram', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Density', fontweight='bold')

    # 3. Temporal Energy Evolution
    ax3 = axes[1, 0]
    time_bins = np.arange(len(temp_energy))
    ax3.plot(time_bins, temp_energy, linewidth=1, color='#A23B72', alpha=0.7)

    # Add smoothed trend
    window_size = max(51, len(temp_energy) // 100)
    if window_size % 2 == 0:
        window_size += 1
    smoothed = signal.savgol_filter(temp_energy, window_size, 3)
    ax3.plot(time_bins, smoothed, linewidth=2, color='#F18F01', label='Smoothed Trend')

    ax3.set_xlabel('Temporal Frame', fontweight='bold')
    ax3.set_ylabel('Energy', fontweight='bold')
    ax3.set_title('Temporal Energy Evolution', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(alpha=0.3, linestyle='--')
    ax3.set_yscale('log')

    # 4. Energy Distribution Histogram
    ax4 = axes[1, 1]
    ax4.hist(temp_energy, bins=100, color='#6A994E', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Energy Level', fontweight='bold')
    ax4.set_ylabel('Frequency Count', fontweight='bold')
    ax4.set_title('Temporal Energy Distribution', fontweight='bold')
    ax4.set_yscale('log')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistical annotations
    median_energy = np.median(temp_energy)
    mean_energy = np.mean(temp_energy)
    ax4.axvline(median_energy, color='red', linestyle='--', linewidth=2, label=f'Median: {median_energy:.2e}')
    ax4.axvline(mean_energy, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_energy:.2e}')
    ax4.legend()

    # Add metadata
    fig.text(0.02, 0.01, f"Audio File: {data['audio_file'].split('/')[-1]}\n"
                         f"Duration: {data['molecular_features']['duration']:.2f}s\n"
                         f"Total Frequency Molecules: {data['molecular_features']['frequency_molecules']['count']}\n"
                         f"Total Frequency Energy: {data['molecular_features']['frequency_molecules']['total_energy']:.2f}",
             fontsize=7, style='italic', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION


    audio_name = "Angel_Techno"  # Change this for different audio files
    audio_folder = "techno"  # Change this for different genre folders
    # Change this for different genre folders

    # Construct paths
    json_file_path = project_root / "validation" / "public" / audio_folder / "gas_information_analysis.json"
    output_directory = project_root / "validation" / "public" / audio_folder

    # Convert to strings for compatibility
    json_file_path = str(json_file_path)
    output_directory = str(output_directory)

    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found")
        print(f"Make sure you've run the gas information analysis first!")
        sys.exit(1)

    try:
        # Load data
        print(f"Loading data from: {json_file_path}")
        data = load_data(json_file_path)
        
        if data is None:
            sys.exit(1)

        # Create visualization
        print("Creating molecular features visualization...")
        output_file = os.path.join(output_directory, f'{audio_name}_molecular_analysis.png')
        plot_molecular_features(data, output_file)

        print("\n" + "=" * 60)
        print("MOLECULAR FEATURES VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"JSON file: {json_file_path}")
        print(f"Output file: {output_file}")
        print(f"Results saved to: {output_directory}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
