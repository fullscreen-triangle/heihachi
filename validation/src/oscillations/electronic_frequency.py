#!/usr/bin/env python3
"""
Electronic Frequency Analysis - Primary Audio Content

This script analyzes the electronic frequency content (primary audio content) as part of the
eight-scale oscillatory analysis framework.

Usage:
    python electronic_frequency.py <audio_file.wav>
"""

import os
import sys
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ElectronicFrequencyAnalyzer:
    """Analyze primary electronic frequency content of audio signals"""

    def __init__(self):
        # Electronic music frequency bands
        self.electronic_bands = {
            'kick_fundamental': (40, 80),
            'bass_synth': (80, 200),
            'lead_synth': (200, 2000),
            'pad_sweep': (500, 4000),
            'hi_freq_elements': (4000, 12000),
            'air_band': (12000, 20000)
        }

    def extract_electronic_features(self, audio_file):
        """Extract electronic frequency features"""

        print(f"Loading audio: {audio_file}")
        audio_data, sr = librosa.load(audio_file, sr=None)

        # STFT analysis
        stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
        magnitude = np.abs(stft)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)

        # Basic spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]

        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio_data)
        harmonic_ratio = np.sum(harmonic ** 2) / (np.sum(harmonic ** 2) + np.sum(percussive ** 2))

        return {
            'audio_file': str(audio_file),
            'duration': len(audio_data) / sr,
            'sample_rate': sr,
            'magnitude_spectrum': magnitude,
            'freq_bins': freq_bins,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'harmonic_ratio': float(harmonic_ratio)
        }

    def analyze_electronic_bands(self, features):
        """Analyze energy in electronic music frequency bands"""

        magnitude = features['magnitude_spectrum']
        freq_bins = features['freq_bins']

        band_analysis = {}
        total_energy = 0

        for band_name, (low_freq, high_freq) in self.electronic_bands.items():
            # Find frequency bin indices
            low_bin = np.argmin(np.abs(freq_bins - low_freq))
            high_bin = np.argmin(np.abs(freq_bins - high_freq))

            # Calculate band energy
            band_magnitude = magnitude[low_bin:high_bin + 1, :]
            band_energy = np.sum(band_magnitude ** 2)
            total_energy += band_energy

            band_analysis[band_name] = {
                'frequency_range': [float(low_freq), float(high_freq)],
                'energy': float(band_energy),
                'mean_magnitude': float(np.mean(band_magnitude)),
                'peak_frequency': float(freq_bins[low_bin + np.argmax(np.mean(band_magnitude, axis=1))])
            }

        # Calculate energy ratios
        for band_name in band_analysis:
            band_analysis[band_name]['energy_ratio'] = band_analysis[band_name]['energy'] / (total_energy + 1e-8)

        return band_analysis

    def calculate_electronic_signature(self, features, band_analysis):
        """Calculate electronic signature"""

        # Key spectral features
        centroid_mean = float(np.mean(features['spectral_centroid']))
        bandwidth_mean = float(np.mean(features['spectral_bandwidth']))
        rolloff_mean = float(np.mean(features['spectral_rolloff']))

        # Electronic character classification
        bass_energy = band_analysis['bass_synth']['energy_ratio']
        lead_energy = band_analysis['lead_synth']['energy_ratio']
        kick_energy = band_analysis['kick_fundamental']['energy_ratio']

        # Determine dominant character
        if bass_energy > 0.3:
            dominant_character = 'bass_heavy'
        elif lead_energy > 0.3:
            dominant_character = 'lead_focused'
        elif kick_energy > 0.2:
            dominant_character = 'percussion_driven'
        else:
            dominant_character = 'balanced'

        return {
            'spectral_centroid_mean': centroid_mean,
            'spectral_bandwidth_mean': bandwidth_mean,
            'spectral_rolloff_mean': rolloff_mean,
            'harmonic_ratio': features['harmonic_ratio'],
            'dominant_character': dominant_character,
            'energy_distribution': {
                'bass_ratio': float(bass_energy),
                'lead_ratio': float(lead_energy),
                'kick_ratio': float(kick_energy)
            }
        }

    def create_visualizations(self, features, band_analysis, signature, output_dir):
        """Create visualizations"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # 1. Electronic band energies
        band_names = list(band_analysis.keys())
        band_energies = [band_analysis[name]['energy'] for name in band_names]

        fig1 = go.Figure(data=[
            go.Bar(x=band_names, y=band_energies, marker_color='orange')
        ])
        fig1.update_layout(
            title='Electronic Frequency Band Analysis',
            xaxis_title='Frequency Band',
            yaxis_title='Energy',
            xaxis_tickangle=-45
        )
        fig1.write_html(output_path / "electronic_bands.html")

        # 2. Comprehensive plot
        plt.figure(figsize=(15, 10))

        # Spectrogram (focused on electronic range)
        plt.subplot(2, 3, 1)
        magnitude = features['magnitude_spectrum']
        plt.imshow(20 * np.log10(magnitude[:200, :] + 1e-8),
                   aspect='auto', origin='lower', cmap='viridis')
        plt.title('Spectrogram (0-4kHz)')
        plt.xlabel('Time Frames')
        plt.ylabel('Frequency Bins')
        plt.colorbar()

        # Spectral centroid evolution
        plt.subplot(2, 3, 2)
        time_frames = np.arange(len(features['spectral_centroid'])) * 512 / features['sample_rate']
        plt.plot(time_frames, features['spectral_centroid'])
        plt.title('Spectral Centroid Evolution')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')

        # Electronic band energies
        plt.subplot(2, 3, 3)
        plt.bar(band_names, band_energies)
        plt.title('Electronic Band Energies')
        plt.xticks(rotation=45)
        plt.ylabel('Energy')

        # Energy ratios
        plt.subplot(2, 3, 4)
        energy_ratios = [band_analysis[name]['energy_ratio'] for name in band_names]
        plt.pie(energy_ratios, labels=band_names, autopct='%1.1f%%')
        plt.title('Energy Distribution')

        # Spectral features
        plt.subplot(2, 3, 5)
        spectral_features = ['Centroid', 'Bandwidth', 'Rolloff']
        spectral_values = [signature['spectral_centroid_mean'],
                           signature['spectral_bandwidth_mean'],
                           signature['spectral_rolloff_mean']]
        # Normalize for display
        normalized_values = [v / max(spectral_values) for v in spectral_values]
        plt.bar(spectral_features, normalized_values)
        plt.title('Spectral Characteristics (Normalized)')
        plt.ylabel('Normalized Value')

        # Character analysis
        plt.subplot(2, 3, 6)
        char_names = ['Bass', 'Lead', 'Kick']
        char_values = [signature['energy_distribution']['bass_ratio'],
                       signature['energy_distribution']['lead_ratio'],
                       signature['energy_distribution']['kick_ratio']]
        plt.bar(char_names, char_values, color='purple')
        plt.title(f'Electronic Character: {signature["dominant_character"]}')
        plt.ylabel('Energy Ratio')

        plt.tight_layout()
        plt.savefig(output_path / "electronic_frequency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Visualizations saved to {output_path}/")


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    # Define paths relative to project root
    audio_file_path = project_root / "validation" / "public" / "wav" / "djs_fresh_heavyweight.wav"
    output_directory = project_root / "validation" / "public" / "heavyweight"

    # Convert to strings for compatibility
    audio_file_path = str(audio_file_path)
    output_directory = str(output_directory)

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file '{audio_file_path}' not found")
        sys.exit(1)

    try:
        # Initialize analyzer
        analyzer = ElectronicFrequencyAnalyzer()

        # Extract features
        print("Extracting electronic frequency features...")
        features = analyzer.extract_electronic_features(audio_file_path)

        # Analyze electronic bands
        print("Analyzing electronic bands...")
        band_analysis = analyzer.analyze_electronic_bands(features)

        # Calculate signature
        print("Calculating electronic signature...")
        signature = analyzer.calculate_electronic_signature(features, band_analysis)

        # Compile analysis
        analysis = {
            'audio_file': str(audio_file_path),
            'processing_timestamp': datetime.now().isoformat(),
            'duration': features['duration'],
            'sample_rate': features['sample_rate'],
            'electronic_band_analysis': band_analysis,
            'electronic_signature': signature,
            'analysis_scale': 'electronic_frequency'
        }

        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)

        # Save results with custom encoder
        json_file = output_dir / "electronic_frequency_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
        print(f"✓ Analysis saved to {json_file}")

        # Create visualizations
        analyzer.create_visualizations(features, band_analysis, signature, output_directory)

        # Print summary
        print("\n" + "=" * 50)
        print("ELECTRONIC FREQUENCY ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Audio file: {audio_file_path}")
        print(f"Duration: {features['duration']:.2f}s")
        print(f"Spectral Centroid: {signature['spectral_centroid_mean']:.1f} Hz")
        print(f"Harmonic Ratio: {signature['harmonic_ratio']:.3f}")
        print(f"Dominant Character: {signature['dominant_character']}")
        print(f"Bass Energy Ratio: {signature['energy_distribution']['bass_ratio']:.3f}")
        print(f"Results saved to: {output_dir}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
