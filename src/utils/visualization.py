import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import librosa
import librosa.display
import plotly.express as px
from plotly.subplots import make_subplots
import os


class MixVisualizer:
    def __init__(self, output_dir: str = "../results/"):
        """Initialize the mix visualizer.
        
        Args:
            output_dir: Path to the output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"MixVisualizer initialized with output directory: {self.output_dir}")

    def save_results_as_json(self, results: Dict, filename: str):
        """Save analysis results as JSON."""
        output_path = self.output_dir / f"{filename}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)

    def visualize_mix_signature(self, graph, metrics, output_path):
        """Visualize mix signature with transitions and VIP sections."""
        pos = nx.spring_layout(graph, seed=42)

        # Node colors based on VIP likelihood
        node_colors = [
            "red" if metrics[node].get("vip_likelihood", False) else "blue"
            for node in graph.nodes
        ]

        # Edge colors based on transition strength
        edge_colors = [
            graph[u][v].get("transition_strength", 0.5)
            for u, v in graph.edges
        ]

        plt.figure(figsize=(12, 8))
        nx.draw(
            graph,
            pos,
            with_labels=True,
            node_size=700,
            node_color=node_colors,
            edge_color=edge_colors,
            edge_cmap=plt.cm.YlOrRd,
            font_size=10,
            width=2
        )

        plt.savefig(self.output_dir / output_path)
        plt.close()

    def plot_mix_heatmap(self, metrics: Dict, times: List[float],
                         output_path: str = "../results/mix_heatmap.png"):
        """Plot enhanced heatmap with annotations."""
        metric_names = list(metrics.keys())
        data = np.array([metrics[metric] for metric in metric_names])

        # Create heatmap with annotations
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            data,
            xticklabels=[f"{t:.2f}s" for t in times],
            yticklabels=metric_names,
            cmap="viridis",
            annot=True,
            fmt=".2f",
            cbar_kws={'label': 'Metric Value'}
        )

        plt.xlabel("Time")
        plt.ylabel("Metrics")
        plt.title("Mix Analysis Heatmap")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(self.output_dir / output_path)
        plt.close()

    def plot_mix_radar(self, metrics: Dict, timestamps: List[float],
                       output_path: str = "../results/mix_radar.html"):
        """Create interactive radar chart showing mix progression."""
        # Normalize metrics
        scaler = MinMaxScaler()
        metrics_normalized = {}
        for key, values in metrics.items():
            metrics_normalized[key] = scaler.fit_transform(
                np.array(values).reshape(-1, 1)
            ).flatten()

        # Create radar chart
        fig = go.Figure()

        for i, timestamp in enumerate(timestamps):
            fig.add_trace(go.Scatterpolar(
                r=[metrics_normalized[key][i] for key in metrics.keys()],
                theta=list(metrics.keys()),
                name=f"t={timestamp:.2f}s",
                fill='toself'
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Mix Evolution Radar Chart"
        )

        fig.write_html(self.output_dir / output_path)

    def plot_clusters(self, clusters: Dict, features: np.ndarray,
                      output_path: str = "../results/clusters.png"):
        """Visualize audio clusters with their characteristics."""
        # Reduce dimensionality for visualization if needed
        if features.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
        else:
            features_2d = features

        plt.figure(figsize=(10, 8))

        # Plot clusters
        for cluster_id, indices in clusters.items():
            cluster_points = features_2d[indices]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                label=f"Cluster {cluster_id}",
                alpha=0.6
            )

            # Plot cluster center
            center = np.mean(cluster_points, axis=0)
            plt.scatter(
                center[0],
                center[1],
                marker='x',
                s=200,
                linewidths=3,
                color='black'
            )

        plt.title("Audio Segments Clustering")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(self.output_dir / output_path)
        plt.close()

    def create_summary_report(self, results: Dict, output_path: str = "../results/summary.html"):
        """Create an HTML summary report with all visualizations and metrics."""
        html_content = f"""
        <html>
        <head>
            <title>Mix Analysis Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 30px; }}
                img {{ max-width: 100%; }}
            </style>
        </head>
        <body>
            <h1>Mix Analysis Summary</h1>

            <div class="section">
                <h2>Mix Structure</h2>
                <img src="mix_heatmap.png">
            </div>

            <div class="section">
                <h2>Mix Evolution</h2>
                <iframe src="mix_radar.html" width="100%" height="600px"></iframe>
            </div>

            <div class="section">
                <h2>Segment Clusters</h2>
                <img src="clusters.png">
            </div>

            <div class="section">
                <h2>Key Metrics</h2>
                <pre>{json.dumps(results.get('summary_metrics', {}), indent=2)}</pre>
            </div>
        </body>
        </html>
        """

        with open(self.output_dir / output_path, 'w') as f:
            f.write(html_content)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class AnalysisVisualizer:
    def __init__(self, output_dir: str = "../visualizations/"):
        """Initialize the analysis visualizer.
        
        Args:
            output_dir: Path to the output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"AnalysisVisualizer initialized with output directory: {self.output_dir}")
        
        # Set style defaults
        try:
            # Try the new style name format first (matplotlib >= 3.6)
            plt.style.use('seaborn-v0_8-darkgrid')
        except (OSError, ValueError):
            try:
                # Fall back to old style name (matplotlib < 3.6)
                plt.style.use('seaborn-darkgrid')
            except (OSError, ValueError):
                # If all else fails, use a built-in style
                plt.style.use('ggplot')
        
        self.color_palette = sns.color_palette("husl", 8)
        
        # Frequency bands for visualization
        self.freq_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, 8000)
        }

    # all the functions below have the wrong return variable Expected type 'None', got 'Figure' instead
    def plot_component_analysis(self, components: Dict, 
                              save_path: Optional[str] = None) -> go.Figure:
        """Visualize component analysis results."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Component Energy Distribution',
                'Spectral Profiles',
                'Temporal Patterns',
                'Transformation Types',
                'Component Relationships',
                'Confidence Scores'
            )
        )

        # Energy distribution
        energies = {c['type']: c['energy'] for c in components['components']}
        fig.add_trace(
            go.Bar(x=list(energies.keys()), y=list(energies.values())),
            row=1, col=1
        )

        # Spectral profiles
        for i, comp in enumerate(components['components']):
            fig.add_trace(
                go.Scatter(
                    y=comp['spectral'],
                    name=comp['type'],
                    line=dict(color=px.colors.qualitative.Set1[i])
                ),
                row=1, col=2
            )

        # Save if path provided
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig

    def plot_groove_analysis(self, groove_data: Dict,
                           save_path: Optional[str] = None) -> go.Figure:
        """Visualize groove analysis results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Microtiming Deviations',
                'Swing Ratio',
                'Syncopation Pattern',
                'Ghost Note Distribution'
            )
        )

        # Microtiming plot
        if 'microtiming' in groove_data:
            fig.add_trace(
                go.Scatter(
                    y=groove_data['microtiming'],
                    mode='lines+markers',
                    name='Microtiming'
                ),
                row=1, col=1
            )

        # Save if path provided
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig

    def plot_similarity_analysis(self, similarity_data: Dict,
                               save_path: Optional[str] = None) -> go.Figure:
        """Visualize similarity analysis results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Component-wise Similarity',
                'Transformation Detection',
                'Band-wise Analysis',
                'Confidence Distribution'
            )
        )

        # Component similarity scores
        scores = similarity_data['component_scores']
        fig.add_trace(
            go.Bar(
                x=list(scores.keys()),
                y=[s if isinstance(s, float) else np.mean(list(s.values())) 
                   for s in scores.values()]
            ),
            row=1, col=1
        )

        # Save if path provided
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig

    def plot_amen_break_analysis(self, audio: np.ndarray, results: Dict, 
                               save_path: Optional[str] = None) -> go.Figure:
        """Create detailed visualization of Amen break analysis results."""
        if not results.get('segments'):
            return None
            
        # Create subplots for each detected segment
        n_segments = len(results['segments'])
        
        # Generate subplot titles - 3 titles for each segment
        all_titles = []
        for i in range(n_segments):
            all_titles.extend([
                f"Segment {i+1} Waveform",
                f"Segment {i+1} Component Analysis",
                f"Segment {i+1} Characteristics"
            ])
        
        # Create specs for the subplot layout
        specs = []
        for _ in range(n_segments):
            specs.append([{"type": "xy"}, {"type": "xy"}, {"type": "polar"}])
        
        # Create the subplots
        fig = make_subplots(
            rows=n_segments, 
            cols=3,
            subplot_titles=all_titles,
            specs=specs,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        for i, (segment, confidence, variation_type, tempo_scale) in enumerate(zip(
            results['segments'],
            results['confidence'],
            results['variation_types'],
            results['tempo_scales']
        )):
            # Get segment audio
            start_idx = int(segment['start'] * 44100)  # Assuming sr=44100
            end_idx = int(segment['end'] * 44100)
            segment_audio = audio[start_idx:end_idx]
            
            # 1. Plot waveform with alignment points
            times = np.linspace(segment['start'], segment['end'], len(segment_audio))
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=segment_audio,
                    name=f"Waveform",
                    line=dict(color='blue', width=1)
                ),
                row=i+1, col=1
            )
            
            # Add alignment points
            alignment_points = results['alignments'][i]
            fig.add_trace(
                go.Scatter(
                    x=[times[p[0]] for p in alignment_points],
                    y=[segment_audio[p[0]] for p in alignment_points],
                    mode='markers',
                    name='Alignment Points',
                    marker=dict(
                        color='red',
                        size=8,
                        symbol='x'
                    )
                ),
                row=i+1, col=1
            )
            
            # 2. Plot component analysis
            if 'component_analysis' in results:
                comp_analysis = results['component_analysis'][i]
                components = ['kick', 'snare', 'hihat']
                metrics = ['presence', 'n_hits', 'avg_velocity', 'velocity_variance']
                
                for j, comp in enumerate(components):
                    comp_data = comp_analysis[comp]
                    fig.add_trace(
                        go.Bar(
                            name=comp.capitalize(),
                            x=metrics,
                            y=[comp_data[m] for m in metrics],
                            # here a dict type was expected.
                            marker_color=['rgb(158,202,225)', 'rgb(94,158,217)', 
                                        'rgb(32,102,148)', 'rgb(8,48,107)'][j],
                            showlegend=False
                        ),
                        row=i+1, col=2
                    )
            
            # 3. Plot characteristics radar
            characteristics = results.get('variation_characteristics', [])[i]
            if characteristics:
                fig.add_trace(
                    go.Scatterpolar(
                        r=[
                            characteristics['complexity'],
                            characteristics['energy'],
                            characteristics['groove'],
                            confidence,
                            tempo_scale
                        ],
                        theta=['Complexity', 'Energy', 'Groove', 
                               'Confidence', 'Tempo Scale'],
                        fill='toself',
                        name=f"Characteristics"
                    ),
                    row=i+1, col=3
                )
            
            # Add variation type annotation
            fig.add_annotation(
                text=f"Variation: {results.get('variation_names', [])[i]}",
                xref="paper", yref="paper",
                x=1.0, y=1.0 - (i/n_segments),
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
            
            # Update axes
            fig.update_xaxes(title_text="Time (s)", row=i+1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=i+1, col=1)
            fig.update_xaxes(title_text="Metric", row=i+1, col=2)
            fig.update_yaxes(title_text="Value", row=i+1, col=2)
            
            # Update polar axis
            fig.update_polars(
                radialaxis=dict(range=[0, 2], showticklabels=True),
                row=i+1, col=3
            )
        
        # Update layout
        fig.update_layout(
            height=400 * n_segments,
            width=1500,
            showlegend=False,
            title_text="Amen Break Analysis Results"
        )
        
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig

    def plot_spectral_analysis(self, audio: np.ndarray, sr: int,
                             save_path: Optional[str] = None) -> go.Figure:
        """Create detailed spectral visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Spectrogram',
                'Mel Spectrogram',
                'Chromagram',
                'Onset Strength'
            )
        )

        # Compute STFT
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

        # Spectrogram
        fig.add_trace(
            go.Heatmap(
                z=S_db,
                colorscale='Viridis'
            ),
            row=1, col=1
        )

        # Save if path provided
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig

    def plot_transition_analysis(self, transitions: List[Dict],
                               save_path: Optional[str] = None) -> go.Figure:
        """Visualize transition analysis results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Transition Points',
                'Transition Types',
                'Confidence Distribution',
                'Duration Analysis'
            )
        )

        # Transition points timeline
        points = [(t['start_time'], t['end_time']) for t in transitions]
        for start, end in points:
            fig.add_trace(
                go.Scatter(
                    x=[start, end],
                    y=[1, 1],
                    mode='lines',
                    showlegend=False
                ),
                row=1, col=1
            )

        # Save if path provided
        if save_path:
            fig.write_html(self.output_dir / save_path)
        
        return fig

    def create_analysis_dashboard(self, analysis_results: Dict,
                                save_path: str) -> go.Figure:
        """Create comprehensive analysis dashboard."""
        # Create main figure with subplots for each analysis type
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Component Analysis',
                'Groove Analysis',
                'Similarity Analysis',
                'Amen Break Analysis',
                'Spectral Analysis',
                'Transition Analysis'
            ),
            vertical_spacing=0.1
        )

        # Add component analysis
        if 'components' in analysis_results:
            self._add_component_summary(fig, analysis_results['components'], 1, 1)

        # Save dashboard
        fig.write_html(self.output_dir / save_path)
        
        return fig

    def _add_component_summary(self, fig: go.Figure, components: Dict,
                             row: int, col: int) -> None:
        """Add component analysis summary to dashboard."""
        energies = {c['type']: c['energy'] for c in components['components']}
        fig.add_trace(
            go.Bar(
                x=list(energies.keys()),
                y=list(energies.values()),
                name='Component Energy'
            ),
            row=row, col=col
        )

    def export_analysis_report(self, analysis_results: Dict,
                             output_path: str) -> None:
        """Export analysis results as an HTML report."""
        # Create visualizations
        figs = []
        
        if 'components' in analysis_results:
            figs.append(self.plot_component_analysis(analysis_results['components']))
            
        if 'groove' in analysis_results:
            figs.append(self.plot_groove_analysis(analysis_results['groove']))
            
        if 'similarity' in analysis_results:
            figs.append(self.plot_similarity_analysis(analysis_results['similarity']))

        # Combine into HTML report
        html_content = """
        <html>
        <head>
            <title>Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .figure { margin: 20px 0; }
            </style>
        </head>
        <body>
        """
        
        for fig in figs:
            html_content += f"<div class='figure'>{fig.to_html()}</div>"
            
        html_content += "</body></html>"
        
        # Save report
        with open(self.output_dir / output_path, 'w') as f:
            f.write(html_content)
