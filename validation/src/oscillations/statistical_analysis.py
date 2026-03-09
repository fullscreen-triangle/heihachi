import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import os
import sys

plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300


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


def plot_statistical_summary(freq_data, mol_data, output_file='statistical_summary.png'):
    """
    Create comprehensive statistical summary
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Extract data
    bands = freq_data['electronic_band_analysis']
    freq_density = np.array(mol_data['molecular_features']['frequency_molecules']['density'])
    temp_energy = np.array(mol_data['molecular_features']['temporal_molecules']['energy'])

    # 1. Band Energy Comparison with Error Bars
    ax1 = fig.add_subplot(gs[0, :])
    band_names = list(bands.keys())
    energies = [bands[band]['energy'] for band in band_names]
    magnitudes = [bands[band]['mean_magnitude'] for band in band_names]

    x = np.arange(len(band_names))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, energies, width, label='Energy', alpha=0.8, color='#3A86FF')
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width / 2, magnitudes, width, label='Mean Magnitude', alpha=0.8, color='#FB5607')

    ax1.set_xlabel('Frequency Band', fontweight='bold')
    ax1.set_ylabel('Energy', fontweight='bold', color='#3A86FF')
    ax1_twin.set_ylabel('Mean Magnitude', fontweight='bold', color='#FB5607')
    ax1.set_title('Energy vs. Mean Magnitude Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.replace('_', ' ').title() for name in band_names], rotation=15, ha='right')
    ax1.tick_params(axis='y', labelcolor='#3A86FF')
    ax1_twin.tick_params(axis='y', labelcolor='#FB5607')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(alpha=0.3, linestyle='--')

    # 2. Frequency Density Statistics
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.boxplot([freq_density], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Density Value', fontweight='bold')
    ax2.set_title('Frequency Density\nDistribution', fontweight='bold')
    ax2.set_xticklabels(['Freq. Density'])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text = f'Q1: {np.percentile(freq_density, 25):.6f}\n' \
                 f'Median: {np.median(freq_density):.6f}\n' \
                 f'Q3: {np.percentile(freq_density, 75):.6f}\n' \
                 f'IQR: {stats.iqr(freq_density):.6f}'
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=7)

    # 3. Temporal Energy Statistics
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.boxplot([temp_energy], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', alpha=0.7),
                medianprops=dict(color='darkred', linewidth=2))
    ax3.set_ylabel('Energy Value', fontweight='bold')
    ax3.set_title('Temporal Energy\nDistribution', fontweight='bold')
    ax3.set_xticklabels(['Temp. Energy'])
    ax3.set_yscale('log')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')

    # 4. Correlation Matrix
    ax4 = fig.add_subplot(gs[1, 2])
    correlation_data = {
        'Energy': energies,
        'Magnitude': magnitudes,
        'Energy Ratio': [bands[band]['energy_ratio'] for band in band_names],
        'Peak Freq': [bands[band]['peak_frequency'] for band in band_names]
    }
    corr_matrix = np.corrcoef(list(correlation_data.values()))

    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(correlation_data)))
    ax4.set_yticks(range(len(correlation_data)))
    ax4.set_xticklabels(correlation_data.keys(), rotation=45, ha='right', fontsize=8)
    ax4.set_yticklabels(correlation_data.keys(), fontsize=8)
    ax4.set_title('Feature Correlation Matrix', fontweight='bold')

    # Add correlation values
    for i in range(len(correlation_data)):
        for j in range(len(correlation_data)):
            text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)

    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Correlation', fontweight='bold')

    # 5. Frequency Density Histogram with KDE
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.hist(freq_density, bins=100, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Histogram')

    # Add KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(freq_density)
    x_range = np.linspace(freq_density.min(), freq_density.max(), 1000)
    ax5.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    ax5.set_xlabel('Frequency Density', fontweight='bold')
    ax5.set_ylabel('Probability Density', fontweight='bold')
    ax5.set_title('Frequency Density Distribution with KDE', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3, linestyle='--')

    # 6. Summary Statistics Table
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.axis('off')

    summary_stats = [
        ['Metric', 'Freq. Density', 'Temp. Energy'],
        ['Mean', f'{np.mean(freq_density):.6f}', f'{np.mean(temp_energy):.2e}'],
        ['Std Dev', f'{np.std(freq_density):.6f}', f'{np.std(temp_energy):.2e}'],
        ['Min', f'{np.min(freq_density):.6f}', f'{np.min(temp_energy):.2e}'],
        ['Max', f'{np.max(freq_density):.6f}', f'{np.max(temp_energy):.2e}'],
        ['Skewness', f'{stats.skew(freq_density):.4f}', f'{stats.skew(temp_energy):.4f}'],
        ['Kurtosis', f'{stats.kurtosis(freq_density):.4f}', f'{stats.kurtosis(temp_energy):.4f}']
    ]

    table = ax6.table(cellText=summary_stats, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax6.set_title('Summary Statistics', fontweight='bold', pad=20)

    # Add metadata
    fig.text(0.02, 0.01, f"Comparative Analysis\n"
                         f"Audio: {freq_data['audio_file'].split('/')[-1]} | "
                         f"Duration: {freq_data['duration']:.2f}s | "
                         f"Sample Rate: {freq_data['sample_rate']} Hz",
             fontsize=7, style='italic', alpha=0.7)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.show()


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    audio_name = "Angel_Techno"  # Change this for different audio files
    audio_folder = "techno"  # Change this for different genre folders

    # Construct paths
    freq_json_path = project_root / "validation" / "public" / audio_folder / "electronic_frequency_analysis.json"
    mol_json_path = project_root / "validation" / "public" / audio_folder / "gas_information_analysis.json"
    output_directory = project_root / "validation" / "public" / audio_folder

    # Convert to strings for compatibility
    freq_json_path = str(freq_json_path)
    mol_json_path = str(mol_json_path)
    output_directory = str(output_directory)

    # Check if JSON files exist
    if not os.path.exists(freq_json_path):
        print(f"Error: JSON file '{freq_json_path}' not found")
        print(f"Make sure you've run the electronic frequency analysis first!")
        sys.exit(1)
    
    if not os.path.exists(mol_json_path):
        print(f"Error: JSON file '{mol_json_path}' not found")
        print(f"Make sure you've run the gas information analysis first!")
        sys.exit(1)

    try:
        # Load data
        print(f"Loading frequency data from: {freq_json_path}")
        freq_data = load_data(freq_json_path)
        
        print(f"Loading molecular data from: {mol_json_path}")
        mol_data = load_data(mol_json_path)
        
        if freq_data is None or mol_data is None:
            sys.exit(1)

        # Create visualization
        print("Creating statistical summary visualization...")
        output_file = os.path.join(output_directory, f'{audio_name}_statistical_summary.png')
        plot_statistical_summary(freq_data, mol_data, output_file)

        print("\n" + "=" * 60)
        print("STATISTICAL SUMMARY VISUALIZATION COMPLETE")
        print("=" * 60)
        print(f"Frequency JSON: {freq_json_path}")
        print(f"Molecular JSON: {mol_json_path}")
        print(f"Output file: {output_file}")
        print(f"Results saved to: {output_directory}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
