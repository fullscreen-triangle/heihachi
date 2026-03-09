import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pathlib import Path
import os
import sys

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10


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


def plot_frequency_band_analysis(data, output_file='frequency_analysis.png'):
    """
    Create comprehensive frequency band analysis visualization
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # Extract band data
    bands = data['electronic_band_analysis']
    band_names = list(bands.keys())

    # Color palette
    colors = sns.color_palette("viridis", len(band_names))

    # 1. Energy Distribution (Bar Chart)
    ax1 = fig.add_subplot(gs[0, :])
    energies = [bands[band]['energy'] for band in band_names]
    bars = ax1.bar(range(len(band_names)), energies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Frequency Band', fontweight='bold')
    ax1.set_ylabel('Energy (arbitrary units)', fontweight='bold')
    ax1.set_title('Energy Distribution Across Frequency Bands', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(band_names)))
    ax1.set_xticklabels([name.replace('_', ' ').title() for name in band_names], rotation=15, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2e}',
                 ha='center', va='bottom', fontsize=8)

    # 2. Mean Magnitude Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    magnitudes = [bands[band]['mean_magnitude'] for band in band_names]
    ax2.barh(range(len(band_names)), magnitudes, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Frequency Band', fontweight='bold')
    ax2.set_xlabel('Mean Magnitude', fontweight='bold')
    ax2.set_title('Mean Magnitude by Band', fontsize=11, fontweight='bold')
    ax2.set_yticks(range(len(band_names)))
    ax2.set_yticklabels([name.replace('_', ' ').title() for name in band_names])
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    # 3. Energy Ratio (Pie Chart)
    ax3 = fig.add_subplot(gs[1, 1])
    energy_ratios = [bands[band]['energy_ratio'] for band in band_names]
    wedges, texts, autotexts = ax3.pie(energy_ratios, labels=[name.replace('_', ' ').title() for name in band_names],
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax3.set_title('Energy Ratio Distribution', fontsize=11, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

    # 4. Frequency Range Visualization
    ax4 = fig.add_subplot(gs[2, :])
    for i, band in enumerate(band_names):
        freq_range = bands[band]['frequency_range']
        peak_freq = bands[band]['peak_frequency']

        # Draw frequency range as horizontal bars
        ax4.barh(i, freq_range[1] - freq_range[0], left=freq_range[0],
                 height=0.6, color=colors[i], alpha=0.6, edgecolor='black')

        # Mark peak frequency
        ax4.scatter(peak_freq, i, color='red', s=100, zorder=5, marker='D',
                    edgecolor='darkred', linewidth=1.5, label='Peak' if i == 0 else '')

    ax4.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax4.set_ylabel('Frequency Band', fontweight='bold')
    ax4.set_title('Frequency Ranges and Peak Frequencies', fontsize=12, fontweight='bold')
    ax4.set_yticks(range(len(band_names)))
    ax4.set_yticklabels([name.replace('_', ' ').title() for name in band_names])
    ax4.set_xscale('log')
    ax4.grid(axis='x', alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right')

    # Add metadata
    fig.text(0.02, 0.02, f"Audio File: {data['audio_file'].split('/')[-1]}\n"
                         f"Duration: {data['duration']:.2f}s | Sample Rate: {data['sample_rate']} Hz\n"
                         f"Processed: {data['processing_timestamp']}",
             fontsize=7, style='italic', alpha=0.7)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    audio_name = "Angel_Techno"  # Change this for different audio files
    audio_folder = "techno"  # Change this for different genre folders

    # Construct paths
    json_file_path = project_root / "validation" / "public" / audio_folder / "electronic_frequency_analysis.json"
    output_directory = project_root / "validation" / "public" / audio_folder

    # Convert to strings for compatibility
    json_file_path = str(json_file_path)
    output_directory = str(output_directory)

    # Check if JSON file exists
    if not os.path.exists(json_file_path):
        print(f"Error: JSON file '{json_file_path}' not found")
        print(f"Make sure you've run the electronic frequency analysis first!")
        sys.exit(1)

    try:
        # Load data
        print(f"Loading data from: {json_file_path}")
        data = load_data(json_file_path)
        
        if data is None:
            sys.exit(1)

        # Create visualization
        print("Creating frequency dynamics visualization...")
        output_file = os.path.join(output_directory, f'{audio_name}_frequency_dynamics.png')
        plot_frequency_band_analysis(data, output_file)

        print("\n" + "=" * 60)
        print("FREQUENCY DYNAMICS VISUALIZATION COMPLETE")
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
