import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import os
import sys
from scipy import signal
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib.patches import Rectangle

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")


def create_molecular_analysis_visualization(json_file_path, output_directory):
    """Create publication-ready visualization of molecular frequency analysis"""

    # Convert paths to Path objects for easier handling
    json_path = Path(json_file_path)
    output_dir = Path(output_directory)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded data from {json_path}")
    except FileNotFoundError:
        print(f"Error: File '{json_path}' not found")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{json_path}': {e}")
        return None

    # Extract molecular features
    density = np.array(data['molecular_features']['frequency_molecules']['density'])
    duration = data['molecular_features']['duration']

    # Create frequency bins (assuming linear spacing)
    n_bins = len(density)
    frequency_bins = np.linspace(0, 22050, n_bins)  # Assuming Nyquist frequency

    # Set up the figure with academic proportions
    fig = plt.figure(figsize=(18, 12))

    # Create a 3x3 grid with specific spacing
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3,
                          left=0.06, right=0.96, top=0.92, bottom=0.08)

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. Full Molecular Density Spectrum (top, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(frequency_bins, density, color='#2E86AB', linewidth=1.2, alpha=0.8)
    ax1.fill_between(frequency_bins, density, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('Molecular Density', fontsize=11)
    ax1.set_title('(a) Molecular Frequency Density Distribution', fontsize=12, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(frequency_bins))

    # Add peak annotations
    peaks, _ = signal.find_peaks(density, height=np.mean(density) + 2 * np.std(density))
    if len(peaks) > 0:
        peak_freqs = frequency_bins[peaks]
        peak_densities = density[peaks]
        ax1.scatter(peak_freqs[:5], peak_densities[:5], color='red', s=50, zorder=5)
        for i, (freq, dens) in enumerate(zip(peak_freqs[:3], peak_densities[:3])):
            ax1.annotate(f'{freq:.0f} Hz', (freq, dens), xytext=(10, 10),
                         textcoords='offset points', fontsize=9,
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    # 2. Log-Scale Density (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.semilogy(frequency_bins, density + 1e-6, color='#A23B72', linewidth=1.2)
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Log Molecular Density', fontsize=11)
    ax2.set_title('(b) Log-Scale Distribution', fontsize=12, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)

    # 3. Frequency Band Analysis (middle-left)
    ax3 = fig.add_subplot(gs[1, 0])

    # Define frequency bands for molecular analysis
    bands = {
        'Sub-Bass': (20, 60),
        'Bass': (60, 250),
        'Low-Mid': (250, 500),
        'Mid': (500, 2000),
        'High-Mid': (2000, 4000),
        'High': (4000, 8000),
        'Ultra-High': (8000, 20000)
    }

    band_energies = []
    band_names = []

    for band_name, (low, high) in bands.items():
        # Find indices for this frequency range
        mask = (frequency_bins >= low) & (frequency_bins <= high)
        band_energy = np.sum(density[mask])
        band_energies.append(band_energy)
        band_names.append(band_name)

    bars = ax3.bar(range(len(band_names)), band_energies, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Frequency Bands', fontsize=11)
    ax3.set_ylabel('Integrated Density', fontsize=11)
    ax3.set_title('(c) Band Energy Distribution', fontsize=12, fontweight='bold', pad=15)
    ax3.set_xticks(range(len(band_names)))
    ax3.set_xticklabels(band_names, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, energy in zip(bars, band_energies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{energy:.3f}', ha='center', va='bottom', fontsize=8)

    # 4. Statistical Analysis (middle-center)
    ax4 = fig.add_subplot(gs[1, 1])

    # Calculate statistical measures
    stats = {
        'Mean': np.mean(density),
        'Std Dev': np.std(density),
        'Skewness': float(np.mean(((density - np.mean(density)) / np.std(density)) ** 3)),
        'Kurtosis': float(np.mean(((density - np.mean(density)) / np.std(density)) ** 4)) - 3,
        'Max': np.max(density),
        'Min': np.min(density)
    }

    stat_names = list(stats.keys())
    stat_values = list(stats.values())

    bars4 = ax4.bar(range(len(stat_names)), stat_values, color='#F39C12', alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Statistical Measures', fontsize=11)
    ax4.set_ylabel('Values', fontsize=11)
    ax4.set_title('(d) Statistical Properties', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xticks(range(len(stat_names)))
    ax4.set_xticklabels(stat_names, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Density Histogram (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    n_hist, bins_hist, patches = ax5.hist(density, bins=50, color='#8E44AD', alpha=0.7,
                                          edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Molecular Density', fontsize=11)
    ax5.set_ylabel('Frequency Count', fontsize=11)
    ax5.set_title('(e) Density Distribution', fontsize=12, fontweight='bold', pad=15)
    ax5.grid(True, alpha=0.3)

    # Add statistical overlay
    ax5.axvline(np.mean(density), color='red', linestyle='--', linewidth=2, label='Mean')
    ax5.axvline(np.median(density), color='orange', linestyle='--', linewidth=2, label='Median')
    ax5.legend(fontsize=9)

    # 6. Spectral Centroid Analysis (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])

    # Calculate spectral centroid over time (using sliding window)
    window_size = max(1, len(density) // 20)  # 20 time segments, minimum 1
    centroids = []
    time_points = []

    for i in range(0, len(density) - window_size, window_size):
        window_density = density[i:i + window_size]
        window_freqs = frequency_bins[i:i + window_size]
        if np.sum(window_density) > 0:  # Avoid division by zero
            centroid = np.sum(window_freqs * window_density) / np.sum(window_density)
            centroids.append(centroid)
            time_points.append(i * duration / len(density))

    if centroids:  # Only plot if we have data
        ax6.plot(time_points, centroids, color='#E74C3C', linewidth=2, marker='o', markersize=4)
    ax6.set_xlabel('Time (seconds)', fontsize=11)
    ax6.set_ylabel('Spectral Centroid (Hz)', fontsize=11)
    ax6.set_title('(f) Temporal Centroid Evolution', fontsize=12, fontweight='bold', pad=15)
    ax6.grid(True, alpha=0.3)

    # 7. Peak Analysis (bottom-center)
    ax7 = fig.add_subplot(gs[2, 1])

    # Find and analyze peaks
    peaks, properties = signal.find_peaks(density, height=np.mean(density), distance=10)
    peak_heights = density[peaks]
    peak_frequencies = frequency_bins[peaks]

    # Plot top 10 peaks
    if len(peaks) > 0:
        top_peaks_idx = np.argsort(peak_heights)[-10:]
        top_peak_freqs = peak_frequencies[top_peaks_idx]
        top_peak_heights = peak_heights[top_peaks_idx]

        bars7 = ax7.bar(range(len(top_peak_freqs)), top_peak_heights,
                        color='#27AE60', alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars7, top_peak_freqs)):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{freq:.0f}Hz', ha='center', va='bottom', fontsize=7, rotation=45)

    ax7.set_xlabel('Peak Index', fontsize=11)
    ax7.set_ylabel('Peak Density', fontsize=11)
    ax7.set_title('(g) Top 10 Molecular Peaks', fontsize=12, fontweight='bold', pad=15)
    ax7.grid(True, alpha=0.3)

    # 8. Summary Information (bottom-right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    # Calculate spectral centroid safely
    total_density = np.sum(density)
    if total_density > 0:
        spectral_centroid = np.sum(frequency_bins * density) / total_density
    else:
        spectral_centroid = 0

    # Create summary information
    summary_info = f"""
MOLECULAR ANALYSIS SUMMARY

Audio Duration: {duration:.1f} seconds
Total Frequency Bins: {len(density):,}
Frequency Resolution: {frequency_bins[1] - frequency_bins[0]:.1f} Hz

DENSITY STATISTICS:
Mean Density: {np.mean(density):.6f}
Max Density: {np.max(density):.6f}
Min Density: {np.min(density):.6f}
Standard Deviation: {np.std(density):.6f}

SPECTRAL FEATURES:
Number of Peaks: {len(peaks)}
Dominant Frequency: {frequency_bins[np.argmax(density)]:.1f} Hz
Energy Centroid: {spectral_centroid:.1f} Hz

DISTRIBUTION SHAPE:
Skewness: {stats['Skewness']:.3f}
Kurtosis: {stats['Kurtosis']:.3f}
"""

    ax8.text(0.05, 0.95, summary_info, transform=ax8.transAxes,
             verticalalignment='top', fontfamily='monospace', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    ax8.set_title('(h) Analysis Summary', fontsize=12, fontweight='bold', pad=15)

    # Add main title
    audio_name = Path(data['molecular_features']['audio_file']).stem
    fig.suptitle(f'Molecular Frequency Analysis: {audio_name}',
                 fontsize=16, fontweight='bold', y=0.97)

    # Save with high DPI for publication
    output_file = output_dir / f'{audio_name}_molecular_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"✓ Molecular analysis visualization saved as '{output_file}'")

    # Return some key statistics for further analysis
    return {
        'peak_frequencies': peak_frequencies if len(peaks) > 0 else [],
        'peak_densities': peak_heights if len(peaks) > 0 else [],
        'band_energies': dict(zip(band_names, band_energies)),
        'spectral_centroid': spectral_centroid,
        'total_energy': total_density,
        'output_file': str(output_file)
    }


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    audio_name = "Angel_Techno"  # Change this for different audio files
    audio_folder = "drip"  # Change this for different genre folders

    # Construct paths
    json_file_path = project_root / "validation" / "public" / audio_folder / "gas_information_analysis.json"
    output_directory = project_root / "validation" / "public" / audio_folder

    # Convert to strings for compatibility
    json_file_path = str(json_file_path)
    output_directory = str(output_directory)

    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found")
        print(f"Make sure you've run the gas molecular analysis first!")
        sys.exit(1)

    try:
        # Create visualization
        print("Creating molecular analysis visualization...")
        stats = create_molecular_analysis_visualization(json_file_path, output_directory)

        if stats:
            print("\n" + "=" * 60)
            print("MOLECULAR VISUALIZATION COMPLETE")
            print("=" * 60)
            print(f"JSON file: {json_file_path}")
            print(f"Output file: {stats['output_file']}")
            print(f"Spectral Centroid: {stats['spectral_centroid']:.1f} Hz")
            print(f"Total Molecular Energy: {stats['total_energy']:.6f}")
            print(f"Number of Significant Peaks: {len(stats['peak_frequencies'])}")
            print(f"Results saved to: {output_directory}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
