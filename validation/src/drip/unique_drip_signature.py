#!/usr/bin/env python3
"""
Unique Song Drip Signatures Generator

Unique Song Drip Signatures: Every song produces a distinctive water droplet impact pattern based on its S-entropy coordinates

Usage:
    python unique_drip_signature.py <audio_file.wav>
"""

import os
import sys
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import argparse
from pathlib import Path
import hashlib

class UniqueDripSignatureGenerator:
    """Generate unique drip signatures based on S-entropy coordinates"""
    
    def __init__(self):
        self.signature_database = {}
        
    def calculate_entropy(self, data, bins=50):
        """Calculate Shannon entropy"""
        if len(data) == 0:
            return 0.0
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist))
    
    def calculate_s_entropy_coordinates(self, audio_file):
        """Calculate S-entropy coordinates from audio"""
        
        # Load audio
        audio_data, sr = librosa.load(audio_file, sr=None)
        
        # Spectral analysis
        stft = librosa.stft(audio_data, hop_length=512)
        magnitude = np.abs(stft)
        
        # Frequency entropy
        freq_content = np.mean(magnitude, axis=1)
        s_frequency = self.calculate_entropy(freq_content)
        
        # Temporal entropy
        temporal_content = np.mean(magnitude, axis=0)
        s_time = self.calculate_entropy(temporal_content)
        
        # Amplitude entropy
        rms = librosa.feature.rms(y=audio_data)[0]
        s_amplitude = self.calculate_entropy(rms)
        
        return {
            'S_frequency': s_frequency,
            'S_time': s_time,
            'S_amplitude': s_amplitude,
            'total_entropy': s_frequency + s_time + s_amplitude
        }
    
    def generate_droplet_signature(self, s_entropy_coords, audio_file):
        """Generate unique droplet signature"""
        
        # Calculate droplet parameters
        velocity = 2.5 * s_entropy_coords['S_frequency'] + 1.8 * s_entropy_coords['S_amplitude'] + 0.5
        size = 0.6 * np.sqrt(s_entropy_coords['S_amplitude']) * np.exp(-2.1 * s_entropy_coords['S_time'])
        
        if s_entropy_coords['S_time'] == 0:
            impact_angle = 0
        else:
            impact_angle = np.arctan(3.2 * s_entropy_coords['S_frequency'] / s_entropy_coords['S_time'])
        
        surface_tension = 0.4 + 0.3 * s_entropy_coords['total_entropy']
        
        # Create unique hash
        signature_string = f"{velocity:.6f}_{size:.6f}_{impact_angle:.6f}_{surface_tension:.6f}"
        signature_hash = hashlib.md5(signature_string.encode()).hexdigest()[:16]
        
        return {
            'signature_hash': signature_hash,
            'velocity': float(velocity),
            'size': float(size),
            'impact_angle': float(impact_angle),
            'surface_tension': float(surface_tension),
            's_entropy_coords': s_entropy_coords
        }
    
    def create_signature_fingerprint(self, signature):
        """Create visual fingerprint"""
        
        # Create 16x16 fingerprint based on signature parameters
        fingerprint = np.zeros((16, 16))
        
        # Map parameters to spatial patterns
        v_pattern = np.sin(signature['velocity'] * np.linspace(0, 2*np.pi, 16))
        s_pattern = signature['size'] * np.eye(16)
        a_pattern = np.cos(signature['impact_angle'] * np.linspace(0, 4*np.pi, 16))
        t_pattern = signature['surface_tension'] * np.ones(16)
        
        # Fill fingerprint
        for i in range(16):
            fingerprint[i] = v_pattern * s_pattern[i] + a_pattern * t_pattern[i]
        
        return fingerprint
    
    def analyze_signature(self, audio_file):
        """Complete signature analysis"""
        
        print(f"Analyzing signature for: {audio_file}")
        
        # Calculate S-entropy coordinates
        s_entropy_coords = self.calculate_s_entropy_coordinates(audio_file)
        
        # Generate signature
        signature = self.generate_droplet_signature(s_entropy_coords, audio_file)
        
        # Create fingerprint
        fingerprint = self.create_signature_fingerprint(signature)
        
        # Compile results
        analysis = {
            'audio_file': str(audio_file),
            'processing_timestamp': datetime.now().isoformat(),
            'signature_hash': signature['signature_hash'],
            's_entropy_coordinates': s_entropy_coords,
            'droplet_signature': signature,
            'fingerprint_shape': fingerprint.shape
        }
        
        return analysis, fingerprint
    
    def create_visualizations(self, analysis, fingerprint, output_dir):
        """Create visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. S-entropy 3D plot
        fig1 = go.Figure(data=[go.Scatter3d(
            x=[analysis['s_entropy_coordinates']['S_frequency']],
            y=[analysis['s_entropy_coordinates']['S_time']],
            z=[analysis['s_entropy_coordinates']['S_amplitude']],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Song Signature'
        )])
        
        fig1.update_layout(
            title=f'S-Entropy Signature: {analysis["signature_hash"]}',
            scene=dict(
                xaxis_title='S_frequency',
                yaxis_title='S_time',
                zaxis_title='S_amplitude'
            )
        )
        fig1.write_html(output_path / "s_entropy_signature.html")
        
        # 2. Droplet parameters
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        coords = ['S_frequency', 'S_time', 'S_amplitude']
        values = [analysis['s_entropy_coordinates'][c] for c in coords]
        plt.bar(coords, values)
        plt.title('S-Entropy Coordinates')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        params = ['velocity', 'size', 'impact_angle', 'surface_tension']
        param_values = [analysis['droplet_signature'][p] for p in params]
        plt.bar(params, param_values, color='orange')
        plt.title('Droplet Parameters')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        plt.imshow(fingerprint, cmap='viridis')
        plt.title('Signature Fingerprint')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        plt.plot(fingerprint.flatten())
        plt.title('Fingerprint Signal')
        plt.xlabel('Position')
        plt.ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(output_path / "signature_analysis.png", dpi=300, bbox_inches='tight')
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
        print(f"Absolute path checked: {Path(audio_file_path).absolute()}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script location: {Path(__file__).parent}")
        print("Please verify the file path is correct")
        sys.exit(1)

    try:
        # Initialize generator
        generator = UniqueDripSignatureGenerator()

        # Analyze signature
        analysis, fingerprint = generator.analyze_signature(audio_file_path)

        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        json_file = output_dir / "unique_signature.json"
        with open(json_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Results saved to {json_file}")

        # Save fingerprint
        np.save(output_dir / "signature_fingerprint.npy", fingerprint)

        # Create visualizations
        generator.create_visualizations(analysis, fingerprint, output_dir)

        # Print summary
        print("\n" + "=" * 50)
        print("UNIQUE SIGNATURE ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"Audio file: {audio_file_path}")
        print(f"Signature Hash: {analysis['signature_hash']}")
        print(f"S-Entropy Coordinates:")
        print(f"  S_frequency: {analysis['s_entropy_coordinates']['S_frequency']:.3f}")
        print(f"  S_time: {analysis['s_entropy_coordinates']['S_time']:.3f}")
        print(f"  S_amplitude: {analysis['s_entropy_coordinates']['S_amplitude']:.3f}")
        print(f"Results saved to: {output_dir}/")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
