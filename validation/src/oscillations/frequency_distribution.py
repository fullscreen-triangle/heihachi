import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import os
import sys
from matplotlib.patches import Rectangle
import seaborn as sns

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def create_academic_visualization(json_file_path, output_directory):
    """Create publication-ready visualization of electronic frequency analysis"""

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

    # Set up the figure with academic proportions
    fig = plt.figure(figsize=(16, 10))

    # Create a 3x3 grid with specific spacing
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Extract data
    bands = list(data['electronic_band_analysis'].keys())
    band_labels = [band.replace('_', ' ').title() for band in bands]
    energy_ratios = [data['electronic_band_analysis'][band]['energy_ratio'] for band in bands]
    energies = [data['electronic_band_analysis'][band]['energy'] for band in bands]
    peak_freqs = [data['electronic_band_analysis'][band]['peak_frequency'] for band in bands]
    freq_ranges = [(data['electronic_band_analysis'][band]['frequency_range'][0],
                    data['electronic_band_analysis'][band]['frequency_range'][1]) for band in bands]

    # Color scheme for consistency
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#577590']

    # 1. Energy Distribution (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    wedges, texts, autotexts = ax1.pie(energy_ratios, labels=band_labels, autopct='%1.1f%%',
                                       startangle=90, colors=colors, textprops={'fontsize': 9})
    ax1.set_title('(a) Energy Distribution by Frequency Band', fontsize=12, fontweight='bold', pad=20)

    # 2. Band Energy Levels (top-center)
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(len(bands)), np.array(energies) / 1e6, color=colors, alpha=0.8, edgecolor='black',
                   linewidth=0.5)
    ax2.set_xlabel('Frequency Bands', fontsize=10)
    ax2.set_ylabel('Energy (×10⁶)', fontsize=10)
    ax2.set_title('(b) Energy Magnitude by Band', fontsize=12, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(bands)))
    ax2.set_xticklabels(band_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, energy) in enumerate(zip(bars, energies)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{energy / 1e6:.0f}M', ha='center', va='bottom', fontsize=8)

    # 3. Spectral Characteristics (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    spectral_features = ['Spectral\nCentroid', 'Spectral\nBandwidth', 'Spectral\nRolloff']
    spectral_values = [
        data['electronic_signature']['spectral_centroid_mean'],
        data['electronic_signature']['spectral_bandwidth_mean'],
        data['electronic_signature']['spectral_rolloff_mean']
    ]
    bars3 = ax3.bar(spectral_features, np.array(spectral_values) / 1000,
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_ylabel('Frequency (kHz)', fontsize=10)
    ax3.set_title('(c) Spectral Characteristics', fontsize=12, fontweight='bold', pad=20)
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, value in zip(bars3, spectral_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{value / 1000:.1f}k', ha='center', va='bottom', fontsize=8)

    # 4. Frequency Band Ranges (middle-left, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])

    # Create frequency range visualization
    y_positions = np.arange(len(bands))
    for i, (band, (low, high), color) in enumerate(zip(band_labels, freq_ranges, colors)):
        ax4.barh(i, high - low, left=low, height=0.6, color=color, alpha=0.7,
                 edgecolor='black', linewidth=0.5)
        # Add frequency range labels
        ax4.text(high + (high - low) * 0.05, i, f'{low:.0f}-{high:.0f} Hz',
                 va='center', fontsize=9)

    ax4.set_xlabel('Frequency (Hz)', fontsize=10)
    ax4.set_ylabel('Frequency Bands', fontsize=10)
    ax4.set_title('(d) Frequency Band Coverage', fontsize=12, fontweight='bold', pad=20)
    ax4.set_yticks(y_positions)
    ax4.set_yticklabels(band_labels, fontsize=9)
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(20, 25000)

    # 5. Character Analysis (middle-right)
    ax5 = fig.add_subplot(gs[1, 2])
    character_types = ['Bass', 'Lead', 'Kick']
    character_values = [
        data['electronic_signature']['energy_distribution']['bass_ratio'],
        data['electronic_signature']['energy_distribution']['lead_ratio'],
        data['electronic_signature']['energy_distribution']['kick_ratio']
    ]
    bars5 = ax5.bar(character_types, character_values,
                    color=['#8E44AD', '#E74C3C', '#F39C12'], alpha=0.8,
                    edgecolor='black', linewidth=0.5)
    ax5.set_ylabel('Energy Ratio', fontsize=10)
    ax5.set_title('(e) Character Distribution', fontsize=12, fontweight='bold', pad=20)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, max(character_values) * 1.1)

    # Add value labels
    for bar, value in zip(bars5, character_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=8)

    # 6. Peak Frequency Analysis (bottom-left)
    ax6 = fig.add_subplot(gs[2, 0])
    scatter = ax6.scatter(peak_freqs, energy_ratios, c=colors, s=100, alpha=0.8,
                          edgecolors='black', linewidth=1)

    # Add labels for each point
    for i, (freq, ratio, label) in enumerate(zip(peak_freqs, energy_ratios, band_labels)):
        ax6.annotate(label, (freq, ratio), xytext=(5, 5), textcoords='offset points',
                     fontsize=8, ha='left')

    ax6.set_xlabel('Peak Frequency (Hz)', fontsize=10)
    ax6.set_ylabel('Energy Ratio', fontsize=10)
    ax6.set_title('(f) Peak Frequency vs Energy', fontsize=12, fontweight='bold', pad=20)
    ax6.set_xscale('log')
    ax6.grid(True, alpha=0.3)

    # 7. Summary Statistics (bottom-center and right)
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')

    # Create summary table
    summary_data = [
        ['Audio Duration', f"{data['duration']:.1f} seconds"],
        ['Sample Rate', f"{data['sample_rate']:,} Hz"],
        ['Harmonic Ratio', f"{data['electronic_signature']['harmonic_ratio']:.3f}"],
        ['Dominant Character', data['electronic_signature']['dominant_character'].replace('_', ' ').title()],
        ['Total Energy', f"{sum(energies) / 1e9:.2f} × 10⁹"],
        ['Spectral Centroid', f"{data['electronic_signature']['spectral_centroid_mean']:.0f} Hz"],
    ]

    # Create table
    table_data = []
    for i, (param, value) in enumerate(summary_data):
        table_data.append([param, value])

    table = ax7.table(cellText=table_data,
                      colLabels=['Parameter', 'Value'],
                      cellLoc='left',
                      loc='center',
                      colWidths=[0.4, 0.4])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ECF0F1' if i % 2 == 0 else 'white')
            cell.set_edgecolor('black')
            cell.set_linewidth(0.5)

    ax7.set_title('(g) Analysis Summary', fontsize=12, fontweight='bold', pad=20)

    # Add main title
    audio_name = Path(data['audio_file']).stem
    fig.suptitle(f'Electronic Frequency Analysis: {audio_name}',
                 fontsize=16, fontweight='bold', y=0.97)

    # Save with high DPI for publication
    output_file = output_dir / f'{audio_name}_electronic_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"✓ Publication-ready visualization saved as '{output_file}'")

    # Return analysis results
    return {
        'total_energy': sum(energies),
        'dominant_character': data['electronic_signature']['dominant_character'],
        'spectral_centroid': data['electronic_signature']['spectral_centroid_mean'],
        'harmonic_ratio': data['electronic_signature']['harmonic_ratio'],
        'band_energies': dict(zip(bands, energies)),
        'output_file': str(output_file)
    }


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    audio_name = "Angel_Techno"  # Change this for different audio files
    audio_folder = "heavyweight"  # Change this for different genre folders

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
        # Create visualization
        print("Creating electronic frequency analysis visualization...")
        results = create_academic_visualization(json_file_path, output_directory)

        if results:
            print("\n" + "=" * 60)
            print("ELECTRONIC FREQUENCY VISUALIZATION COMPLETE")
            print("=" * 60)
            print(f"JSON file: {json_file_path}")
            print(f"Output file: {results['output_file']}")
            print(f"Total Energy: {results['total_energy'] / 1e9:.2f} × 10⁹")
            print(f"Dominant Character: {results['dominant_character'].replace('_', ' ').title()}")
            print(f"Spectral Centroid: {results['spectral_centroid']:.0f} Hz")
            print(f"Harmonic Ratio: {results['harmonic_ratio']:.3f}")
            print(f"Results saved to: {output_directory}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
