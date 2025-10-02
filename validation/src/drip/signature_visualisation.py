import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Circle, Wedge
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import json
from math import pi, cos, sin
import colorsys
from matplotlib.collections import LineCollection
import warnings

warnings.filterwarnings('ignore')


class AudioSignatureVisualizer:
    def __init__(self):
        """Initialize audio signature visualizer with multiple chart types"""
        self.signature_colors = {
            'S_frequency': '#FF6B6B',
            'S_time': '#4ECDC4',
            'S_amplitude': '#45B7D1',
            'total_entropy': '#96CEB4',
            'velocity': '#FFEAA7',
            'size': '#DDA0DD',
            'impact_angle': '#98D8C8',
            'surface_tension': '#F7DC6F'
        }

    def load_signature_data(self, json_data):
        """Load and parse signature data from JSON"""
        if isinstance(json_data, str):
            with open(json_data, 'r') as f:
                data = json.load(f)
        else:
            data = json_data

        return data

    def create_radar_signature_chart(self, signature_data):
        """Create radar chart for audio signature visualization using Plotly"""
        # Extract signature parameters [[0]](#__0)
        s_entropy = signature_data.get('s_entropy_coordinates', {})
        droplet_sig = signature_data.get('droplet_signature', {})

        # Prepare data for radar chart
        categories = []
        values = []
        colors = []

        # S-Entropy coordinates
        for key, value in s_entropy.items():
            if not pd.isna(value) and np.isfinite(value):
                categories.append(key.replace('_', ' ').title())
                # Normalize values for better visualization
                if 'frequency' in key.lower():
                    values.append(abs(value) * 10)  # Scale frequency
                elif 'time' in key.lower():
                    values.append(abs(value))
                else:
                    values.append(abs(value) / 10)  # Scale large negative values
                colors.append(self.signature_colors.get(key, '#888888'))

        # Droplet signature parameters
        droplet_params = ['velocity', 'impact_angle', 'surface_tension']
        for param in droplet_params:
            value = droplet_sig.get(param)
            if value is not None and not pd.isna(value) and np.isfinite(value):
                categories.append(param.replace('_', ' ').title())
                values.append(abs(value) / 10 if abs(value) > 10 else abs(value))
                colors.append(self.signature_colors.get(param, '#888888'))

        # Create Plotly radar chart [[0]](#__0)
        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Audio Signature',
            line=dict(color='rgba(255, 107, 107, 0.8)', width=3),
            fillcolor='rgba(255, 107, 107, 0.2)',
            marker=dict(size=8, color=colors)
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.1] if values else [0, 10],
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    gridwidth=1,
                ),
                angularaxis=dict(
                    gridcolor='rgba(128, 128, 128, 0.3)',
                    gridwidth=1,
                )
            ),
            showlegend=True,
            title={
                'text': f"Audio Signature Radar Chart<br>Hash: {signature_data.get('signature_hash', 'Unknown')}",
                'x': 0.5,
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            width=600,
            height=600,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        return fig

    def create_circular_amplitude_plot(self, signature_data):
        """Create circular plot for amplitude and frequency visualization"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        s_entropy = signature_data.get('s_entropy_coordinates', {})
        droplet_sig = signature_data.get('droplet_signature', {})

        # Create circular visualization with multiple rings [[1]](#__1)
        angles = np.linspace(0, 2 * pi, 8, endpoint=False)

        # Ring 1: S-Entropy coordinates
        entropy_values = []
        entropy_labels = []
        for key, value in s_entropy.items():
            if not pd.isna(value) and np.isfinite(value):
                entropy_values.append(abs(value))
                entropy_labels.append(key.replace('_', ' ').title())

        if entropy_values:
            # Normalize entropy values
            max_entropy = max(entropy_values)
            normalized_entropy = [v / max_entropy for v in entropy_values]

            # Plot entropy ring
            theta_entropy = angles[:len(normalized_entropy)]
            ax.plot(theta_entropy, normalized_entropy, 'o-', linewidth=3,
                    markersize=10, label='S-Entropy Coords', color='#FF6B6B', alpha=0.8)
            ax.fill(theta_entropy, normalized_entropy, alpha=0.2, color='#FF6B6B')

        # Ring 2: Droplet parameters
        droplet_values = []
        droplet_labels = []
        droplet_params = ['velocity', 'impact_angle', 'surface_tension']

        for param in droplet_params:
            value = droplet_sig.get(param)
            if value is not None and not pd.isna(value) and np.isfinite(value):
                droplet_values.append(abs(value))
                droplet_labels.append(param.replace('_', ' ').title())

        if droplet_values:
            # Normalize droplet values
            max_droplet = max(droplet_values)
            normalized_droplet = [v / max_droplet * 0.7 for v in droplet_values]  # Scale to 70% for inner ring

            # Plot droplet ring
            theta_droplet = angles[:len(normalized_droplet)]
            ax.plot(theta_droplet, normalized_droplet, 's-', linewidth=3,
                    markersize=8, label='Droplet Params', color='#4ECDC4', alpha=0.8)
            ax.fill(theta_droplet, normalized_droplet, alpha=0.2, color='#4ECDC4')

        # Customize the plot [[1]](#__1)
        ax.set_ylim(0, 1.2)
        ax.set_title(f'Circular Audio Signature Visualization\nHash: {signature_data.get("signature_hash", "Unknown")}',
                     pad=30, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True, alpha=0.3)

        # Add parameter labels
        all_labels = entropy_labels + droplet_labels
        all_angles = list(theta_entropy) + list(theta_droplet) if entropy_values and droplet_values else []

        for angle, label in zip(all_angles, all_labels):
            ax.text(angle, 1.1, label,
                    rotation=np.degrees(angle) - 90 if angle > pi / 2 and angle < 3 * pi / 2 else np.degrees(angle),
                    ha='center', va='center', fontsize=10, fontweight='bold')

        return fig

    def create_signature_heatmap(self, signature_data):
        """Create heatmap visualization of signature fingerprint"""
        fingerprint_shape = signature_data.get('fingerprint_shape', [16, 16])

        # Generate synthetic fingerprint data based on signature parameters [[2]](#__2)
        s_entropy = signature_data.get('s_entropy_coordinates', {})

        # Create fingerprint matrix
        rows, cols = fingerprint_shape
        fingerprint_matrix = np.zeros((rows, cols))

        # Fill matrix based on signature parameters
        if s_entropy:
            freq_val = s_entropy.get('S_frequency', 1.0)
            time_val = s_entropy.get('S_time', 1.0)
            amp_val = abs(s_entropy.get('S_amplitude', -1.0))
            entropy_val = abs(s_entropy.get('total_entropy', -1.0))

            # Create pattern based on parameters
            for i in range(rows):
                for j in range(cols):
                    # Create complex pattern based on signature values
                    pattern_val = (
                            np.sin(i * freq_val / 10) * np.cos(j * time_val / 10) * amp_val / 100 +
                            np.cos(i * entropy_val / 50) * np.sin(j * freq_val / 5) * 0.5
                    )
                    fingerprint_matrix[i, j] = abs(pattern_val)

        # Create heatmap using Seaborn [[2]](#__2)
        plt.figure(figsize=(10, 8))
        sns.heatmap(fingerprint_matrix,
                    cmap='viridis',
                    cbar_kws={'label': 'Signature Intensity'},
                    square=True,
                    linewidths=0.1,
                    linecolor='white')

        plt.title(
            f'Audio Signature Fingerprint Heatmap\nHash: {signature_data.get("signature_hash", "Unknown")}\nShape: {fingerprint_shape}',
            fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Frequency Dimension', fontsize=12)
        plt.ylabel('Time Dimension', fontsize=12)

        return plt.gcf()

    def create_3d_signature_space(self, signature_data):
        """Create 3D visualization of signature in S-entropy space"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        s_entropy = signature_data.get('s_entropy_coordinates', {})
        droplet_sig = signature_data.get('droplet_signature', {})

        # Extract coordinates for 3D plot [[3]](#__3)
        x = s_entropy.get('S_frequency', 0)
        y = s_entropy.get('S_time', 0)
        z = s_entropy.get('S_amplitude', 0)

        # Plot main signature point
        ax.scatter(x, y, z, s=200, c='red', marker='o', alpha=0.8, label='Main Signature')

        # Add droplet parameters as additional points
        if droplet_sig:
            velocity = droplet_sig.get('velocity', 0)
            impact_angle = droplet_sig.get('impact_angle', 0)
            surface_tension = droplet_sig.get('surface_tension', 0)

            # Create related points in 3D space
            related_points = [
                (x * 0.8, y * 0.8, velocity / 10),
                (impact_angle * 10, y * 0.6, z * 0.8),
                (x * 0.6, surface_tension / 10, z * 0.6)
            ]

            colors = ['blue', 'green', 'orange']
            labels = ['Velocity', 'Impact Angle', 'Surface Tension']

            for point, color, label in zip(related_points, colors, labels):
                if all(np.isfinite(coord) for coord in point):
                    ax.scatter(*point, s=100, c=color, marker='^', alpha=0.6, label=label)
                    # Draw connection line
                    ax.plot([x, point[0]], [y, point[1]], [z, point[2]],
                            color=color, alpha=0.3, linewidth=2)

        # Customize 3D plot [[3]](#__3)
        ax.set_xlabel('S-Frequency', fontsize=12)
        ax.set_ylabel('S-Time', fontsize=12)
        ax.set_zlabel('S-Amplitude', fontsize=12)
        ax.set_title(f'3D Audio Signature Space\nHash: {signature_data.get("signature_hash", "Unknown")}',
                     fontsize=14, fontweight='bold', pad=20)
        ax.legend()

        # Add grid and styling
        ax.grid(True, alpha=0.3)

        return fig

    def create_comprehensive_signature_dashboard(self, signature_data):
        """Create comprehensive dashboard with all visualization types"""
        # Create subplot layout
        fig = plt.figure(figsize=(20, 16))

        # 1. Circular amplitude plot
        ax1 = plt.subplot(2, 3, 1, projection='polar')
        self.create_circular_amplitude_subplot(ax1, signature_data)

        # 2. Signature heatmap
        ax2 = plt.subplot(2, 3, 2)
        self.create_signature_heatmap_subplot(ax2, signature_data)

        # 3. 3D signature space
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        self.create_3d_signature_subplot(ax3, signature_data)

        # 4. Parameter comparison bar chart
        ax4 = plt.subplot(2, 3, 4)
        self.create_parameter_comparison(ax4, signature_data)

        # 5. Entropy evolution plot
        ax5 = plt.subplot(2, 3, 5)
        self.create_entropy_evolution(ax5, signature_data)

        # 6. Signature statistics
        ax6 = plt.subplot(2, 3, 6)
        self.create_signature_statistics(ax6, signature_data)

        plt.tight_layout()
        plt.suptitle(
            f'Audio Signature Comprehensive Dashboard\nFile: {signature_data.get("audio_file", "Unknown")}\nHash: {signature_data.get("signature_hash", "Unknown")}',
            fontsize=16, fontweight='bold', y=0.98)

        return fig

    def create_circular_amplitude_subplot(self, ax, signature_data):
        """Create circular plot as subplot"""
        s_entropy = signature_data.get('s_entropy_coordinates', {})

        # Create circular visualization
        angles = np.linspace(0, 2 * pi, len(s_entropy), endpoint=False)
        values = [abs(v) if not pd.isna(v) and np.isfinite(v) else 0 for v in s_entropy.values()]

        if values:
            max_val = max(values) if max(values) > 0 else 1
            normalized_values = [v / max_val for v in values]

            ax.plot(angles, normalized_values, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
            ax.fill(angles, normalized_values, alpha=0.3, color='#FF6B6B')

        ax.set_title('Circular Signature', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def create_signature_heatmap_subplot(self, ax, signature_data):
        """Create heatmap as subplot"""
        fingerprint_shape = signature_data.get('fingerprint_shape', [8, 8])  # Smaller for subplot
        s_entropy = signature_data.get('s_entropy_coordinates', {})

        rows, cols = fingerprint_shape
        matrix = np.random.rand(rows, cols)  # Simplified for subplot

        if s_entropy:
            freq_val = abs(s_entropy.get('S_frequency', 1.0))
            for i in range(rows):
                for j in range(cols):
                    matrix[i, j] = abs(np.sin(i * freq_val) * np.cos(j * freq_val))

        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title('Signature Fingerprint', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    def create_3d_signature_subplot(self, ax, signature_data):
        """Create 3D plot as subplot"""
        s_entropy = signature_data.get('s_entropy_coordinates', {})

        x = s_entropy.get('S_frequency', 0)
        y = s_entropy.get('S_time', 0)
        z = s_entropy.get('S_amplitude', 0)

        ax.scatter(x, y, z, s=100, c='red', marker='o')
        ax.set_xlabel('S-Freq')
        ax.set_ylabel('S-Time')
        ax.set_zlabel('S-Amp')
        ax.set_title('3D Signature Space', fontsize=12, fontweight='bold')

    def create_parameter_comparison(self, ax, signature_data):
        """Create parameter comparison bar chart"""
        s_entropy = signature_data.get('s_entropy_coordinates', {})
        droplet_sig = signature_data.get('droplet_signature', {})

        params = []
        values = []
        colors = []

        for key, value in s_entropy.items():
            if not pd.isna(value) and np.isfinite(value):
                params.append(key.replace('_', ' ').title())
                values.append(abs(value))
                colors.append(self.signature_colors.get(key, '#888888'))

        if params and values:
            bars = ax.bar(params, values, color=colors, alpha=0.7)
            ax.set_title('Parameter Magnitudes', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9)

    def create_entropy_evolution(self, ax, signature_data):
        """Create entropy evolution visualization"""
        s_entropy = signature_data.get('s_entropy_coordinates', {})

        entropy_keys = ['S_frequency', 'S_time', 'S_amplitude', 'total_entropy']
        entropy_values = [s_entropy.get(key, 0) for key in entropy_keys]

        # Create evolution plot
        x = range(len(entropy_keys))
        ax.plot(x, entropy_values, 'o-', linewidth=2, markersize=8, color='#4ECDC4')
        ax.fill_between(x, entropy_values, alpha=0.3, color='#4ECDC4')

        ax.set_xticks(x)
        ax.set_xticklabels([key.replace('_', '\n') for key in entropy_keys], fontsize=9)
        ax.set_title('Entropy Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def create_signature_statistics(self, ax, signature_data):
        """Create signature statistics summary"""
        s_entropy = signature_data.get('s_entropy_coordinates', {})
        droplet_sig = signature_data.get('droplet_signature', {})

        # Calculate statistics
        entropy_values = [v for v in s_entropy.values() if not pd.isna(v) and np.isfinite(v)]

        if entropy_values:
            stats = {
                'Mean': np.mean(entropy_values),
                'Std': np.std(entropy_values),
                'Min': np.min(entropy_values),
                'Max': np.max(entropy_values),
                'Range': np.max(entropy_values) - np.min(entropy_values)
            }

            y_pos = range(len(stats))
            values = list(stats.values())

            bars = ax.barh(y_pos, values, color='#96CEB4', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(list(stats.keys()))
            ax.set_title('Signature Statistics', fontsize=12, fontweight='bold')

            # Add value labels
            for bar, value in zip(bars, values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2.,
                        f'{value:.2f}', ha='left', va='center', fontsize=9)

        ax.text(0.5, 0.1, f'Hash: {signature_data.get("signature_hash", "Unknown")}',
                transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))


# Example usage with your signature data
def visualize_audio_signature(signature_json):
    """Main function to create all signature visualizations"""

    # Sample data structure matching your format
    sample_signature_data = {
        "audio_file": "DJ Fresh - Heavyweight",
        "processing_timestamp": "2025-10-01T11:15:09.504715",
        "signature_hash": "f1c6f10e008e1458",
        "s_entropy_coordinates": {
            "S_frequency": 1.4762894648972744,
            "S_time": 15.870666547783808,
            "S_amplitude": -49.76767174333248,
            "total_entropy": -32.420715730651395
        },
        "droplet_signature": {
            "signature_hash": "f1c6f10e008e1458",
            "velocity": -85.39108547575528,
            "size": float('nan'),
            "impact_angle": 0.28931231132355384,
            "surface_tension": -9.326214719195418,
            "s_entropy_coords": {
                "S_frequency": 1.4762894648972744,
                "S_time": 15.870666547783808,
                "S_amplitude": -49.76767174333248,
                "total_entropy": -32.420715730651395
            }
        },
        "fingerprint_shape": [
            16,
            16
        ]
    }

    # Use provided data or sample data
    if isinstance(signature_json, str):
        with open(signature_json, 'r') as f:
            signature_data = json.load(f)
    elif isinstance(signature_json, dict):
        signature_data = signature_json
    else:
        signature_data = sample_signature_data

    # Create visualizer
    visualizer = AudioSignatureVisualizer()

    # Create different types of visualizations
    print("Creating radar chart...")
    radar_fig = visualizer.create_radar_signature_chart(signature_data)

    print("Creating circular amplitude plot...")
    circular_fig = visualizer.create_circular_amplitude_plot(signature_data)

    print("Creating signature heatmap...")
    heatmap_fig = visualizer.create_signature_heatmap(signature_data)

    print("Creating 3D signature space...")
    space_3d_fig = visualizer.create_3d_signature_space(signature_data)

    print("Creating comprehensive dashboard...")
    dashboard_fig = visualizer.create_comprehensive_signature_dashboard(signature_data)

    return {
        'radar': radar_fig,
        'circular': circular_fig,
        'heatmap': heatmap_fig,
        '3d_space': space_3d_fig,
        'dashboard': dashboard_fig
    }


# Main execution
if __name__ == "__main__":
    # Your signature data
    signature_data = {
        "audio_file": "DJ Fresh - Heavyweight",
        "processing_timestamp": "2025-10-01T11:15:09.504715",
        "signature_hash": "f1c6f10e008e1458",
        "s_entropy_coordinates": {
            "S_frequency": 1.4762894648972744,
            "S_time": 15.870666547783808,
            "S_amplitude": -49.76767174333248,
            "total_entropy": -32.420715730651395
        },
        "droplet_signature": {
            "signature_hash": "f1c6f10e008e1458",
            "velocity": -85.39108547575528,
            "size": float('nan'),
            "impact_angle": 0.28931231132355384,
            "surface_tension": -9.326214719195418,
            "s_entropy_coords": {
                "S_frequency": 1.4762894648972744,
                "S_time": 15.870666547783808,
                "S_amplitude": -49.76767174333248,
                "total_entropy": -32.420715730651395
            }
        },
        "fingerprint_shape": [
            16,
            16
        ]
    }

    # Create all visualizations
    visualizations = visualize_audio_signature(signature_data)

    # Display Plotly radar chart
    if 'radar' in visualizations:
        visualizations['radar'].show()

    # Display matplotlib plots
    for name, fig in visualizations.items():
        if name != 'radar' and fig is not None:
            plt.figure(fig.number)
            plt.show()

    print("All signature visualizations created successfully!")
