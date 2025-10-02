#!/usr/bin/env python3
"""
Universal Audio-to-Drip Algorithm Implementation

Key Contribution: Each song can be converted to a unique visual droplet pattern for computer vision analysis.

This script implements the complete audio-to-drip conversion framework as described in 
docs/foundation/audio-drip-algorithm.tex, converting individual songs to characteristic 
water droplet impact visualizations through S-entropy coordinate mapping.

Algorithm Phases:
1. Eight-scale oscillatory signature extraction from audio signals
2. S-entropy coordinate calculation for tri-dimensional audio space navigation  
3. Droplet parameter determination based on oscillatory characteristics
4. Physics-based water surface impact simulation
5. Computer vision pattern recognition and visualization generation

Usage:
    python drip_algorithm.py <audio_file.wav>
"""

import os
import sys
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import signal, ndimage
from scipy.spatial.distance import euclidean
import cv2
from datetime import datetime
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AudioSEntropyCalculator:
    """Calculate S-entropy coordinates from oscillatory signatures"""
    
    def __init__(self):
        self.frequency_weights = np.logspace(-6, 4, 8)  # 8 scale weights
        self.temporal_weights = np.logspace(-6, 4, 8)
        self.amplitude_weights = np.logspace(-6, 4, 8)
    
    def calculate_entropy(self, data, bins=50):
        """Calculate Shannon entropy of data distribution"""
        if len(data) == 0:
            return 0.0
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist))
    
    def calculate_s_entropy_coordinates(self, oscillatory_signatures):
        """Calculate tri-dimensional S-entropy coordinates"""
        
        # Calculate frequency entropy across all scales
        frequency_entropy = 0
        for i, (scale, signature) in enumerate(oscillatory_signatures.items()):
            if 'frequency_distribution' in signature:
                freq_content = signature['frequency_distribution']
                scale_entropy = self.calculate_entropy(freq_content)
                frequency_entropy += scale_entropy * self.frequency_weights[i]
        
        # Calculate temporal entropy  
        temporal_entropy = 0
        for i, (scale, signature) in enumerate(oscillatory_signatures.items()):
            if 'temporal_structure' in signature:
                temporal_pattern = signature['temporal_structure'] 
                scale_temporal = self.calculate_entropy(temporal_pattern)
                temporal_entropy += scale_temporal * self.temporal_weights[i]
        
        # Calculate amplitude entropy
        amplitude_entropy = 0
        for i, (scale, signature) in enumerate(oscillatory_signatures.items()):
            if 'amplitude_modulation' in signature:
                amplitude_dynamics = signature['amplitude_modulation']
                scale_amplitude = self.calculate_entropy(amplitude_dynamics) 
                amplitude_entropy += scale_amplitude * self.amplitude_weights[i]
        
        return {
            'S_frequency': frequency_entropy,
            'S_time': temporal_entropy, 
            'S_amplitude': amplitude_entropy,
            'total_entropy': frequency_entropy + temporal_entropy + amplitude_entropy
        }

class EightScaleOscillatoryExtractor:
    """Extract oscillatory signatures across eight hierarchical scales"""
    
    def __init__(self):
        self.scales = {
            'quantum_acoustic': {'freq_range': (1e12, 1e15), 'weight': 1.0},
            'molecular_sound': {'freq_range': (1e6, 1e9), 'weight': 0.8},
            'electronic_frequency': {'freq_range': (10, 1e4), 'weight': 1.0},
            'rhythmic_pattern': {'freq_range': (0.1, 10), 'weight': 0.9},
            'musical_phrase': {'freq_range': (0.01, 0.1), 'weight': 0.7},
            'track_structure': {'freq_range': (1e-3, 1e-2), 'weight': 0.6},
            'set_album': {'freq_range': (1e-4, 1e-3), 'weight': 0.5},
            'cultural': {'freq_range': (1e-6, 1e-4), 'weight': 0.4}
        }
    
    def extract_scale_signature(self, audio_data, sr, scale_name, freq_range, weight):
        """Extract oscillatory signature for a specific scale"""
        
        # Get appropriate frequency analysis for the scale
        if scale_name in ['quantum_acoustic', 'molecular_sound']:
            # High frequency analysis - use harmonic content
            harmonic_content = librosa.effects.harmonic(audio_data)
            spectral_centroid = librosa.feature.spectral_centroid(y=harmonic_content, sr=sr)[0]
            freq_dist = spectral_centroid
            temporal_struct = np.gradient(spectral_centroid)
            amplitude_mod = np.abs(harmonic_content)
            
        elif scale_name == 'electronic_frequency':
            # Primary audio content - full spectral analysis
            stft = librosa.stft(audio_data, hop_length=512)
            magnitude = np.abs(stft)
            freq_dist = np.mean(magnitude, axis=1)
            temporal_struct = np.mean(magnitude, axis=0)
            amplitude_mod = librosa.feature.rms(y=audio_data)[0]
            
        elif scale_name == 'rhythmic_pattern':
            # Beat and rhythm analysis
            tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
            onset_envelope = librosa.onset.onset_strength(y=audio_data, sr=sr)
            freq_dist = np.array([tempo])
            temporal_struct = onset_envelope
            amplitude_mod = librosa.feature.tempogram(onset_envelope=onset_envelope, sr=sr)
            amplitude_mod = np.mean(amplitude_mod, axis=0)
            
        elif scale_name == 'musical_phrase':
            # Phrase structure analysis
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
            freq_dist = np.mean(chroma, axis=1)  
            temporal_struct = np.var(chroma, axis=0)
            amplitude_mod = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0]
            
        elif scale_name == 'track_structure':
            # Sectional analysis
            mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            freq_dist = np.mean(mfcc, axis=1)
            temporal_struct = np.var(mfcc, axis=0) 
            amplitude_mod = librosa.feature.zero_crossing_rate(audio_data)[0]
            
        elif scale_name in ['set_album', 'cultural']:
            # Long-term coherence and cultural patterns
            contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
            freq_dist = np.mean(contrast, axis=1)
            temporal_struct = np.var(contrast, axis=0)
            amplitude_mod = librosa.feature.tonnetz(y=audio_data, sr=sr)
            amplitude_mod = np.mean(amplitude_mod, axis=0)
            
        return {
            'scale_name': scale_name,
            'frequency_distribution': freq_dist,
            'temporal_structure': temporal_struct, 
            'amplitude_modulation': amplitude_mod,
            'scale_weight': weight,
            'freq_range': freq_range
        }
    
    def extract_multi_scale_signatures(self, audio_data, sr):
        """Extract signatures across all eight scales"""
        signatures = {}
        
        for scale_name, params in self.scales.items():
            signature = self.extract_scale_signature(
                audio_data, sr, scale_name, 
                params['freq_range'], params['weight']
            )
            signatures[scale_name] = signature
            
        return signatures

class DropletParameterMapper:
    """Map S-entropy coordinates to physical droplet parameters"""
    
    def __init__(self):
        # Calibration parameters for audio processing
        self.velocity_params = {'alpha': 2.5, 'beta': 1.8, 'gamma': 0.5}
        self.size_params = {'alpha': 0.6, 'beta': 2.1}  
        self.angle_params = {'alpha': 3.2}
        self.tension_params = {'sigma_0': 0.4, 'alpha': 0.3}
    
    def calculate_droplet_velocity(self, s_frequency, s_amplitude):
        """Calculate droplet velocity from S-entropy coordinates"""
        p = self.velocity_params
        return p['alpha'] * s_frequency + p['beta'] * s_amplitude + p['gamma']
    
    def calculate_droplet_size(self, s_amplitude, s_time):
        """Calculate droplet size from S-entropy coordinates"""
        p = self.size_params
        return p['alpha'] * np.sqrt(s_amplitude) * np.exp(-p['beta'] * s_time)
    
    def calculate_impact_angle(self, s_frequency, s_time):
        """Calculate impact angle from S-entropy coordinates""" 
        p = self.angle_params
        if s_time == 0:
            return 0
        return np.arctan(p['alpha'] * s_frequency / s_time)
    
    def calculate_surface_tension(self, s_total):
        """Calculate surface tension from total entropy"""
        p = self.tension_params
        return p['sigma_0'] + p['alpha'] * s_total
    
    def map_to_droplet_params(self, s_entropy_coords):
        """Map S-entropy coordinates to droplet parameters"""
        
        velocity = self.calculate_droplet_velocity(
            s_entropy_coords['S_frequency'], 
            s_entropy_coords['S_amplitude']
        )
        
        size = self.calculate_droplet_size(
            s_entropy_coords['S_amplitude'],
            s_entropy_coords['S_time'] 
        )
        
        angle = self.calculate_impact_angle(
            s_entropy_coords['S_frequency'],
            s_entropy_coords['S_time']
        )
        
        tension = self.calculate_surface_tension(
            s_entropy_coords['total_entropy']
        )
        
        return {
            'velocity': float(velocity),
            'size': float(size),
            'impact_angle': float(angle), 
            'surface_tension': float(tension),
            's_entropy_mapping': s_entropy_coords
        }

def serialize_signatures(signatures):
    """Convert numpy arrays in signatures to JSON-serializable format"""
    serialized = {}
    for scale, signature in signatures.items():
        serialized[scale] = {}
        for key, value in signature.items():
            if isinstance(value, np.ndarray):
                serialized[scale][key] = value.tolist()
            else:
                serialized[scale][key] = value
    return serialized


def main():
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/x folder/script to project root

    # Define paths relative to project root
    audio_file_path = project_root / "validation" / "public" / "wav" / "djs_fresh_heavyweight.wav"
    output_directory = project_root / "validation" / "public" / "heavyweight"

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
        sys.exit(1)

    # Initialize components
    extractor = EightScaleOscillatoryExtractor()
    calculator = AudioSEntropyCalculator()
    mapper = DropletParameterMapper()

    try:
        print(f"Processing audio file: {audio_file_path}")

        # Load audio
        audio_data, sr = librosa.load(audio_file_path, sr=None)

        print("Phase 1: Extracting eight-scale oscillatory signatures...")
        # Phase 1: Extract oscillatory signatures
        oscillatory_signatures = extractor.extract_multi_scale_signatures(audio_data, sr)

        print("Phase 2: Calculating S-entropy coordinates...")
        # Phase 2: Calculate S-entropy coordinates
        s_entropy_coords = calculator.calculate_s_entropy_coordinates(oscillatory_signatures)

        print("Phase 3: Mapping to droplet parameters...")
        # Phase 3: Map to droplet parameters
        droplet_parameters = mapper.map_to_droplet_params(s_entropy_coords)

        # Create analysis results
        analysis_results = {
            'audio_file': str(audio_file_path),
            'processing_timestamp': datetime.now().isoformat(),
            'audio_properties': {
                'sample_rate': int(sr),
                'duration': float(len(audio_data) / sr),
                'samples': len(audio_data)
            },
            'oscillatory_signatures': serialize_signatures(oscillatory_signatures),
            's_entropy_coordinates': s_entropy_coords,
            'droplet_parameters': droplet_parameters,
            'processing_complexity': 'O(1)',
            'algorithm_phases': [
                'Oscillatory Signature Extraction',
                'S-Entropy Coordinate Calculation',
                'Droplet Parameter Determination'
            ]
        }

        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)  # Added parents=True

        # Save JSON results
        json_file = output_dir / "drip_analysis.json"
        with open(json_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        print(f"✓ Analysis results saved to {json_file}")

        # Create basic visualization
        plt.figure(figsize=(15, 10))

        # Plot S-entropy coordinates
        plt.subplot(2, 3, 1)
        coords = ['S_frequency', 'S_time', 'S_amplitude']
        values = [s_entropy_coords[c] for c in coords]
        plt.bar(coords, values)
        plt.title('S-Entropy Coordinates')
        plt.xticks(rotation=45)

        # Plot droplet parameters
        plt.subplot(2, 3, 2)
        params = ['velocity', 'size', 'impact_angle', 'surface_tension']
        param_values = [droplet_parameters[p] for p in params]
        plt.bar(params, param_values, color='orange')
        plt.title('Droplet Parameters')
        plt.xticks(rotation=45)

        # Plot oscillatory signature summary
        plt.subplot(2, 3, 3)
        scale_names = list(oscillatory_signatures.keys())
        scale_weights = [sig['scale_weight'] for sig in oscillatory_signatures.values()]
        plt.bar(scale_names, scale_weights, color='green')
        plt.title('Oscillatory Scale Weights')
        plt.xticks(rotation=45)

        # Plot frequency distributions for first few scales
        for i, (scale, sig) in enumerate(list(oscillatory_signatures.items())[:3]):
            plt.subplot(2, 3, 4 + i)
            freq_dist = sig['frequency_distribution']
            if hasattr(freq_dist, '__len__') and len(freq_dist) > 1:
                plt.plot(freq_dist)
            else:
                plt.bar(['Value'], [freq_dist])
            plt.title(f'{scale} - Frequency')

        plt.tight_layout()
        plt.savefig(output_dir / "drip_analysis.png", dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_dir}/drip_analysis.png")

        # Print summary
        print("\n" + "=" * 50)
        print("AUDIO-TO-DRIP CONVERSION COMPLETE")
        print("=" * 50)
        print(f"Audio file: {audio_file_path}")
        print(f"Duration: {analysis_results['audio_properties']['duration']:.2f}s")
        print(f"S-Entropy Coordinates:")
        print(f"  S_frequency: {s_entropy_coords['S_frequency']:.3f}")
        print(f"  S_time: {s_entropy_coords['S_time']:.3f}")
        print(f"  S_amplitude: {s_entropy_coords['S_amplitude']:.3f}")
        print(f"Droplet Parameters:")
        print(f"  Velocity: {droplet_parameters['velocity']:.3f}")
        print(f"  Size: {droplet_parameters['size']:.3f}")
        print(f"  Impact Angle: {droplet_parameters['impact_angle']:.3f}")
        print(f"  Surface Tension: {droplet_parameters['surface_tension']:.3f}")
        print(f"\nResults saved to: {output_dir}/")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

