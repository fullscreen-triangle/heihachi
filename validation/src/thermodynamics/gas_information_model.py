#!/usr/bin/env python3
"""
Gas Information Model for Individual Song Analysis

This script implements the gas information model for a song and uses St-Stellas transformations
to make analysis more efficient through the empty dictionary approach.

Usage:
    python gas_information_model.py <audio_file.wav>
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


class GasMolecularProcessor:
    """Process audio signals as gas molecular ensembles"""

    def __init__(self):
        # Gas molecular parameters
        self.temperature = 300.0
        self.pressure = 101325.0
        self.gas_constant = 8.314

    def extract_molecular_features(self, audio_file):
        """Extract audio features and convert to molecular properties"""

        print(f"Loading audio: {audio_file}")
        audio_data, sr = librosa.load(audio_file, sr=None)

        # Basic spectral analysis
        stft = librosa.stft(audio_data, hop_length=512)
        magnitude = np.abs(stft)

        # Extract frequency molecules
        freq_energy = np.mean(magnitude, axis=1)
        freq_molecules = {
            'density': (freq_energy / np.sum(freq_energy)).tolist(),
            'count': len(freq_energy),
            'total_energy': float(np.sum(freq_energy))
        }

        # Extract temporal molecules
        temporal_energy = np.mean(magnitude, axis=0)
        temp_molecules = {
            'energy': (temporal_energy / np.sum(temporal_energy)).tolist(),
            'count': len(temporal_energy),
            'flow': np.gradient(temporal_energy).tolist()
        }

        # Extract amplitude molecules
        rms = librosa.feature.rms(y=audio_data)[0]
        amp_molecules = {
            'concentration': (rms / np.sum(rms)).tolist(),
            'count': len(rms),
            'variance': float(np.var(rms))
        }

        return {
            'audio_file': str(audio_file),
            'duration': len(audio_data) / sr,
            'frequency_molecules': freq_molecules,
            'temporal_molecules': temp_molecules,
            'amplitude_molecules': amp_molecules
        }

    def calculate_thermodynamic_state(self, molecular_features):
        """Calculate thermodynamic state variables"""

        # Calculate system energy
        total_energy = (molecular_features['frequency_molecules']['total_energy'] +
                        sum(molecular_features['temporal_molecules']['energy']) +
                        molecular_features['amplitude_molecules']['variance'])

        # Calculate entropy
        freq_dist = np.array(molecular_features['frequency_molecules']['density'])
        entropy = -np.sum(freq_dist * np.log2(freq_dist + 1e-8))

        return {
            'total_energy': float(total_energy),
            'entropy': float(entropy),
            'temperature': float(self.temperature),
            'pressure': float(self.pressure),
            'molecular_count': (molecular_features['frequency_molecules']['count'] +
                                molecular_features['temporal_molecules']['count'] +
                                molecular_features['amplitude_molecules']['count'])
        }

    def simulate_equilibrium_restoration(self, thermodynamic_state):
        """Simulate equilibrium restoration after perturbation"""

        equilibrium_energy = thermodynamic_state['total_energy']
        perturbed_energy = equilibrium_energy * 1.2

        # Exponential decay to equilibrium
        time_steps = 50
        restoration_path = []

        for t in range(time_steps):
            current_energy = equilibrium_energy + (perturbed_energy - equilibrium_energy) * np.exp(-t / 10.0)
            restoration_path.append({
                'time': t,
                'energy': float(current_energy),
                'deviation': float(abs(current_energy - equilibrium_energy))
            })

        return {
            'restoration_path': restoration_path,
            'equilibrium_energy': float(equilibrium_energy)
        }


def create_visualizations(molecular_features, thermodynamic_state, equilibrium_sim, output_dir):
    """Create visualizations of gas molecular analysis"""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Molecular distributions
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=molecular_features['frequency_molecules']['density'],
        name='Frequency Molecules'
    ))
    fig1.add_trace(go.Scatter(
        y=molecular_features['temporal_molecules']['energy'],
        name='Temporal Molecules'
    ))
    fig1.update_layout(title='Gas Molecular Distributions')
    fig1.write_html(output_path / "molecular_distributions.html")

    # 2. Equilibrium restoration
    restoration_data = equilibrium_sim['restoration_path']
    times = [p['time'] for p in restoration_data]
    energies = [p['energy'] for p in restoration_data]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(molecular_features['frequency_molecules']['density'])
    plt.title('Frequency Molecular Density')

    plt.subplot(2, 2, 2)
    plt.plot(molecular_features['temporal_molecules']['energy'])
    plt.title('Temporal Molecular Energy')

    plt.subplot(2, 2, 3)
    plt.plot(times, energies)
    plt.title('Equilibrium Restoration')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')

    plt.subplot(2, 2, 4)
    state_vars = ['Total Energy', 'Entropy', 'Molecular Count']
    state_values = [thermodynamic_state['total_energy'],
                    thermodynamic_state['entropy'],
                    thermodynamic_state['molecular_count']]
    plt.bar(state_vars, state_values)
    plt.title('Thermodynamic State')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / "gas_analysis.png", dpi=300, bbox_inches='tight')
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
        # Initialize processor
        processor = GasMolecularProcessor()

        # Extract molecular features
        print("Extracting molecular features...")
        molecular_features = processor.extract_molecular_features(audio_file_path)

        # Calculate thermodynamic state
        print("Calculating thermodynamic state...")
        thermodynamic_state = processor.calculate_thermodynamic_state(molecular_features)

        # Simulate equilibrium restoration
        print("Simulating equilibrium restoration...")
        equilibrium_sim = processor.simulate_equilibrium_restoration(thermodynamic_state)

        # Compile analysis
        analysis = {
            'audio_file': str(audio_file_path),
            'processing_timestamp': datetime.now().isoformat(),
            'molecular_features': molecular_features,
            'thermodynamic_state': thermodynamic_state,
            'equilibrium_simulation': equilibrium_sim,
            'processing_method': 'Gas Molecular Information Model'
        }

        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(exist_ok=True)

        # Save results with custom encoder
        json_file = output_dir / "gas_information_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyEncoder)
        print(f"✓ Analysis saved to {json_file}")

        # Create visualizations
        create_visualizations(molecular_features, thermodynamic_state, equilibrium_sim, output_directory)

        # Print summary
        print("\n" + "=" * 50)
        print("GAS INFORMATION MODEL ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Audio file: {audio_file_path}")
        print(f"Duration: {molecular_features['duration']:.2f}s")
        print(f"Total Energy: {thermodynamic_state['total_energy']:.3f}")
        print(f"Entropy: {thermodynamic_state['entropy']:.3f}")
        print(f"Molecular Count: {thermodynamic_state['molecular_count']}")
        print(f"Results saved to: {output_dir}/")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
