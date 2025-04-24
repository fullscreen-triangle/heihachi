#!/usr/bin/env python3
"""
Web interface for visualizing Heihachi audio analysis results.

This module provides a web-based user interface for exploring and visualizing
the results of audio analysis performed with Heihachi.
"""

import os
import json
import sys
import time
import threading
import webbrowser
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Flask and web dependencies
from flask import Flask, render_template, request, jsonify, send_from_directory
import plotly
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logging_utils import get_logger
from src.utils.export import export_results

logger = get_logger(__name__)

# Create Flask app
app = Flask(
    __name__, 
    template_folder=os.path.join(project_root, 'src', 'templates'),
    static_folder=os.path.join(project_root, 'src', 'static')
)

# Global variables
RESULTS_DATA = {}
CURRENT_RESULT_FILE = None

def load_results_file(file_path: str) -> Dict[str, Any]:
    """Load results from a JSON file.
    
    Args:
        file_path: Path to the results file
        
    Returns:
        Loaded results data or empty dict if loading fails
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded results from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load results from {file_path}: {e}")
        return {}

def create_waveform_plot(data: Dict[str, Any], title: str = "Waveform") -> Dict[str, Any]:
    """Create a waveform visualization using Plotly.
    
    Args:
        data: Analysis result data
        title: Plot title
        
    Returns:
        JSON representation of the Plotly figure
    """
    # Extract waveform data
    waveform = data.get("waveform", [])
    if not waveform:
        return {}
        
    # Create time axis
    sample_rate = data.get("sample_rate", 44100)
    duration = len(waveform) / sample_rate
    time = np.linspace(0, duration, len(waveform))
    
    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=waveform,
        mode="lines",
        name="Amplitude",
        line=dict(color="#1f77b4", width=1)
    ))
    
    # Add layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"
    )
    
    # Return JSON representation
    return json.loads(plotly.io.to_json(fig))

def create_spectrogram_plot(data: Dict[str, Any], title: str = "Spectrogram") -> Dict[str, Any]:
    """Create a spectrogram visualization using Plotly.
    
    Args:
        data: Analysis result data
        title: Plot title
        
    Returns:
        JSON representation of the Plotly figure
    """
    # Extract spectrogram data
    spectrogram = data.get("spectrogram", [])
    if not spectrogram:
        return {}
        
    # Convert to numpy array if needed
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = np.array(spectrogram)
    
    # Get dimensions
    sample_rate = data.get("sample_rate", 44100)
    hop_length = data.get("hop_length", 512)
    
    # Create time and frequency axes
    time_steps = spectrogram.shape[1]
    time = np.arange(time_steps) * hop_length / sample_rate
    
    freq_bins = spectrogram.shape[0]
    freqs = np.linspace(0, sample_rate / 2, freq_bins)
    
    # Create figure
    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        x=time,
        y=freqs,
        colorscale="Viridis",
        colorbar=dict(title="Intensity")
    ))
    
    # Add layout
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white"
    )
    
    # Return JSON representation
    return json.loads(plotly.io.to_json(fig))

def create_feature_plot(feature_name: str, feature_data: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a visualization for a specific feature.
    
    Args:
        feature_name: Name of the feature
        feature_data: Feature data
        data: Complete analysis result data
        
    Returns:
        JSON representation of the Plotly figure
    """
    # Handle different feature types
    if isinstance(feature_data, list) or isinstance(feature_data, np.ndarray):
        # Time series feature
        sample_rate = data.get("sample_rate", 44100)
        hop_length = data.get("hop_length", 512)
        
        if len(feature_data) > 0:
            feature_data = np.array(feature_data)
            if len(feature_data.shape) == 1:
                # 1D feature (e.g. tempo, energy)
                time_steps = len(feature_data)
                time = np.arange(time_steps) * hop_length / sample_rate
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time,
                    y=feature_data,
                    mode="lines",
                    name=feature_name,
                    line=dict(color="#1f77b4", width=2)
                ))
                
                fig.update_layout(
                    title=f"Feature: {feature_name}",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Value",
                    margin=dict(l=50, r=50, t=50, b=50),
                    template="plotly_white"
                )
                
                return json.loads(plotly.io.to_json(fig))
                
            elif len(feature_data.shape) == 2:
                # 2D feature (e.g. MFCC)
                time_steps = feature_data.shape[1]
                time = np.arange(time_steps) * hop_length / sample_rate
                
                freq_bins = feature_data.shape[0]
                bins = np.arange(freq_bins)
                
                fig = go.Figure(data=go.Heatmap(
                    z=feature_data,
                    x=time,
                    y=bins,
                    colorscale="Viridis",
                    colorbar=dict(title="Value")
                ))
                
                fig.update_layout(
                    title=f"Feature: {feature_name}",
                    xaxis_title="Time (seconds)",
                    yaxis_title="Bin",
                    margin=dict(l=50, r=50, t=50, b=50),
                    template="plotly_white"
                )
                
                return json.loads(plotly.io.to_json(fig))
    
    # For scalar features or other types
    return {}

def create_metrics_plot(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a visualization of audio metrics.
    
    Args:
        data: Analysis result data
        
    Returns:
        JSON representation of the Plotly figure
    """
    # Extract metrics
    metrics = {}
    
    # Quality metrics
    if "quality" in data:
        quality = data.get("quality", {})
        for key, value in quality.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
    
    # Tempo and rhythm metrics
    if "tempo" in data:
        tempo = data.get("tempo", {})
        if "bpm" in tempo and isinstance(tempo["bpm"], (int, float)):
            metrics["BPM"] = tempo["bpm"]
    
    # If we have enough metrics, create a radar chart
    if len(metrics) >= 3:
        # Normalize values between 0 and 1
        keys = list(metrics.keys())
        values = list(metrics.values())
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val > min_val:
            normalized = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized = [0.5] * len(values)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized,
            theta=keys,
            fill='toself',
            name='Audio Metrics'
        ))
        
        fig.update_layout(
            title="Audio Quality Metrics",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_white"
        )
        
        return json.loads(plotly.io.to_json(fig))
    
    # If not enough metrics for a radar chart, create a bar chart
    elif metrics:
        keys = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=keys,
            y=values,
            name='Metrics',
            marker_color='rgb(55, 83, 109)'
        ))
        
        fig.update_layout(
            title="Audio Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            margin=dict(l=50, r=50, t=50, b=50),
            template="plotly_white"
        )
        
        return json.loads(plotly.io.to_json(fig))
    
    return {}

def get_file_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract file information from analysis results.
    
    Args:
        data: Analysis result data
        
    Returns:
        Dictionary with file information
    """
    info = {}
    
    # Basic audio properties
    info["filename"] = data.get("filename", "Unknown")
    info["format"] = data.get("format", "Unknown")
    info["duration"] = data.get("duration", 0)
    info["sample_rate"] = data.get("sample_rate", 0)
    info["channels"] = data.get("channels", 0)
    
    # Try to get additional metadata
    if "metadata" in data:
        metadata = data.get("metadata", {})
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                info[key] = value
    
    return info

def get_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from analysis results.
    
    Args:
        data: Analysis result data
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Tempo metrics
    if "tempo" in data:
        tempo = data.get("tempo", {})
        metrics["BPM"] = tempo.get("bpm", "N/A")
        metrics["Tempo"] = tempo.get("label", "N/A")
        
    # Key and scale
    if "key" in data:
        key = data.get("key", {})
        metrics["Key"] = key.get("key", "N/A")
        metrics["Scale"] = key.get("scale", "N/A")
        metrics["Key Confidence"] = key.get("confidence", "N/A")
    
    # Quality metrics
    if "quality" in data:
        quality = data.get("quality", {})
        for key, value in quality.items():
            metrics[f"Quality: {key}"] = value
    
    return metrics

def get_all_features(data: Dict[str, Any]) -> Dict[str, str]:
    """Get all available features in the analysis results.
    
    Args:
        data: Analysis result data
        
    Returns:
        Dictionary mapping feature paths to human-readable names
    """
    features = {}
    
    def explore_dict(d, path=""):
        for key, value in d.items():
            # Skip metadata and file info
            if key in ["metadata", "file_info"]:
                continue
                
            current_path = f"{path}/{key}" if path else key
            
            if isinstance(value, dict):
                explore_dict(value, current_path)
            else:
                features[current_path] = key.replace("_", " ").title()
    
    explore_dict(data)
    return features

def get_feature_by_path(data: Dict[str, Any], path: str) -> Any:
    """Get a feature by its path in the results data.
    
    Args:
        data: Analysis result data
        path: Path to the feature
        
    Returns:
        Feature data or None if not found
    """
    parts = path.split("/")
    
    current = data
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    
    return current

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    global RESULTS_DATA, CURRENT_RESULT_FILE
    
    # Check if we have results loaded
    if not RESULTS_DATA:
        return render_template('loading.html')
    
    # Get file info
    file_info = get_file_info(RESULTS_DATA)
    
    # Get metrics
    metrics = get_metrics(RESULTS_DATA)
    
    # Get available features
    features = get_all_features(RESULTS_DATA)
    
    return render_template(
        'index.html',
        file_info=file_info,
        metrics=metrics,
        features=features,
        filename=CURRENT_RESULT_FILE
    )

@app.route('/api/waveform')
def api_waveform():
    """API endpoint for waveform visualization."""
    global RESULTS_DATA
    
    if not RESULTS_DATA:
        return jsonify({"error": "No data loaded"})
    
    plot = create_waveform_plot(RESULTS_DATA)
    return jsonify(plot)

@app.route('/api/spectrogram')
def api_spectrogram():
    """API endpoint for spectrogram visualization."""
    global RESULTS_DATA
    
    if not RESULTS_DATA:
        return jsonify({"error": "No data loaded"})
    
    plot = create_spectrogram_plot(RESULTS_DATA)
    return jsonify(plot)

@app.route('/api/metrics')
def api_metrics():
    """API endpoint for metrics visualization."""
    global RESULTS_DATA
    
    if not RESULTS_DATA:
        return jsonify({"error": "No data loaded"})
    
    plot = create_metrics_plot(RESULTS_DATA)
    return jsonify(plot)

@app.route('/api/feature')
def api_feature():
    """API endpoint for feature visualization."""
    global RESULTS_DATA
    
    if not RESULTS_DATA:
        return jsonify({"error": "No data loaded"})
    
    # Get feature path from query parameter
    feature_path = request.args.get('path', '')
    if not feature_path:
        return jsonify({"error": "No feature path specified"})
    
    # Get feature data
    feature_data = get_feature_by_path(RESULTS_DATA, feature_path)
    if feature_data is None:
        return jsonify({"error": f"Feature not found: {feature_path}"})
    
    # Create plot
    feature_name = feature_path.split('/')[-1].replace('_', ' ').title()
    plot = create_feature_plot(feature_name, feature_data, RESULTS_DATA)
    
    return jsonify(plot)

@app.route('/api/export')
def api_export():
    """API endpoint for exporting results."""
    global RESULTS_DATA
    
    if not RESULTS_DATA:
        return jsonify({"error": "No data loaded"})
    
    # Get export format from query parameter
    export_format = request.args.get('format', 'json')
    
    # Get feature path from query parameter (optional)
    feature_path = request.args.get('path', '')
    
    # Determine what to export
    if feature_path:
        # Export specific feature
        data = get_feature_by_path(RESULTS_DATA, feature_path)
        if data is None:
            return jsonify({"error": f"Feature not found: {feature_path}"})
            
        filename = feature_path.replace('/', '_')
    else:
        # Export all results
        data = RESULTS_DATA
        filename = "results"
    
    # Export to temporary file
    try:
        exports_dir = os.path.join(project_root, 'exports')
        os.makedirs(exports_dir, exist_ok=True)
        
        export_file = os.path.join(exports_dir, f"{filename}.{export_format}")
        export_results(data, export_file, export_format)
        
        return jsonify({"success": True, "file": export_file})
    except Exception as e:
        logger.error(f"Export error: {e}")
        return jsonify({"error": f"Export failed: {str(e)}"})

@app.route('/exports/<path:filename>')
def download_export(filename):
    """Download an exported file."""
    exports_dir = os.path.join(project_root, 'exports')
    return send_from_directory(exports_dir, filename, as_attachment=True)

class WebUI:
    """Web UI handler for Heihachi."""
    
    def __init__(self):
        """Initialize the web UI."""
        self.server_thread = None
        self.host = "localhost"
        self.port = 5000
        self.running = False
    
    def start(self, result_file: str, host: str = "localhost", port: int = 5000, 
              open_browser: bool = True) -> bool:
        """Start the web UI server.
        
        Args:
            result_file: Path to the results file
            host: Host to bind to
            port: Port to listen on
            open_browser: Whether to open a browser window
            
        Returns:
            True if the server started successfully, False otherwise
        """
        global RESULTS_DATA, CURRENT_RESULT_FILE
        
        # Load results
        data = load_results_file(result_file)
        if not data:
            logger.error(f"Failed to load results from {result_file}")
            return False
        
        RESULTS_DATA = data
        CURRENT_RESULT_FILE = os.path.basename(result_file)
        
        # Set up server
        self.host = host
        self.port = port
        
        # Start the server in a separate thread
        self.server_thread = threading.Thread(
            target=self._run_server,
            args=(host, port),
            daemon=True
        )
        self.server_thread.start()
        
        # Wait for the server to start
        time.sleep(1)
        
        # Open browser if requested
        if open_browser:
            webbrowser.open(f"http://{host}:{port}/")
        
        self.running = True
        return True
    
    def _run_server(self, host: str, port: int) -> None:
        """Run the Flask server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
        """
        app.run(host=host, port=port, debug=False, use_reloader=False)
    
    def stop(self) -> None:
        """Stop the web UI server."""
        self.running = False
        # Flask servers can't be easily stopped programmatically
        # The server will be terminated when the main thread exits

# Ensure templates and static directories exist
def ensure_web_dirs():
    """Create required directories for the web UI."""
    templates_dir = os.path.join(project_root, 'src', 'templates')
    static_dir = os.path.join(project_root, 'src', 'static')
    
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # Create basic HTML templates if they don't exist
    index_template = os.path.join(templates_dir, 'index.html')
    if not os.path.exists(index_template):
        with open(index_template, 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Heihachi Audio Analysis Results</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <header>
        <h1>Heihachi Audio Analysis</h1>
        <p>File: {{ filename }}</p>
    </header>
    
    <div class="container">
        <div class="sidebar">
            <div class="file-info">
                <h2>File Information</h2>
                <table>
                    {% for key, value in file_info.items() %}
                    <tr>
                        <td><strong>{{ key }}</strong></td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="metrics">
                <h2>Audio Metrics</h2>
                <table>
                    {% for key, value in metrics.items() %}
                    <tr>
                        <td><strong>{{ key }}</strong></td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <div class="features">
                <h2>Features</h2>
                <ul>
                    {% for path, name in features.items() %}
                    <li><a href="#" class="feature-link" data-path="{{ path }}">{{ name }}</a></li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="export">
                <h2>Export</h2>
                <select id="export-format">
                    <option value="json">JSON</option>
                    <option value="csv">CSV</option>
                    <option value="yaml">YAML</option>
                </select>
                <button id="export-button">Export</button>
                <div id="export-status"></div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="tabs">
                <button class="tab-button active" data-tab="waveform">Waveform</button>
                <button class="tab-button" data-tab="spectrogram">Spectrogram</button>
                <button class="tab-button" data-tab="metrics">Metrics</button>
                <button class="tab-button" data-tab="feature">Feature</button>
            </div>
            
            <div class="tab-content" id="waveform-tab">
                <div id="waveform-plot" class="plot"></div>
            </div>
            
            <div class="tab-content" id="spectrogram-tab" style="display: none;">
                <div id="spectrogram-plot" class="plot"></div>
            </div>
            
            <div class="tab-content" id="metrics-tab" style="display: none;">
                <div id="metrics-plot" class="plot"></div>
            </div>
            
            <div class="tab-content" id="feature-tab" style="display: none;">
                <h3 id="feature-name">Select a feature from the sidebar</h3>
                <div id="feature-plot" class="plot"></div>
            </div>
        </div>
    </div>
    
    <script>
        // Load initial plots
        fetch('/api/waveform')
            .then(response => response.json())
            .then(data => {
                Plotly.newPlot('waveform-plot', data.data, data.layout);
            });
            
        // Tab switching
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', () => {
                // Update active tab button
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                button.classList.add('active');
                
                // Hide all tab content
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.style.display = 'none';
                });
                
                // Show selected tab content
                const tabId = button.getAttribute('data-tab');
                document.getElementById(`${tabId}-tab`).style.display = 'block';
                
                // Load plot data if needed
                if (tabId === 'spectrogram' && !document.getElementById('spectrogram-plot').innerHTML) {
                    fetch('/api/spectrogram')
                        .then(response => response.json())
                        .then(data => {
                            Plotly.newPlot('spectrogram-plot', data.data, data.layout);
                        });
                } else if (tabId === 'metrics' && !document.getElementById('metrics-plot').innerHTML) {
                    fetch('/api/metrics')
                        .then(response => response.json())
                        .then(data => {
                            if (data.data) {
                                Plotly.newPlot('metrics-plot', data.data, data.layout);
                            } else {
                                document.getElementById('metrics-plot').innerHTML = 'No metrics available for visualization.';
                            }
                        });
                }
            });
        });
        
        // Feature selection
        document.querySelectorAll('.feature-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Get feature path
                const path = link.getAttribute('data-path');
                
                // Switch to feature tab
                document.querySelectorAll('.tab-button').forEach(btn => {
                    btn.classList.remove('active');
                });
                document.querySelector('.tab-button[data-tab="feature"]').classList.add('active');
                
                document.querySelectorAll('.tab-content').forEach(content => {
                    content.style.display = 'none';
                });
                document.getElementById('feature-tab').style.display = 'block';
                
                // Update feature name
                document.getElementById('feature-name').textContent = link.textContent;
                
                // Load feature plot
                fetch(`/api/feature?path=${path}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.data) {
                            Plotly.newPlot('feature-plot', data.data, data.layout);
                        } else {
                            document.getElementById('feature-plot').innerHTML = 'This feature cannot be visualized.';
                        }
                    });
            });
        });
        
        // Export button
        document.getElementById('export-button').addEventListener('click', () => {
            const format = document.getElementById('export-format').value;
            const statusDiv = document.getElementById('export-status');
            
            statusDiv.textContent = 'Exporting...';
            
            fetch(`/api/export?format=${format}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        statusDiv.textContent = `Error: ${data.error}`;
                    } else {
                        statusDiv.innerHTML = `<a href="/exports/${data.file.split('/').pop()}" download>Download export</a>`;
                    }
                });
        });
    </script>
</body>
</html>''')
    
    loading_template = os.path.join(templates_dir, 'loading.html')
    if not os.path.exists(loading_template):
        with open(loading_template, 'w') as f:
            f.write('''<!DOCTYPE html>
<html>
<head>
    <title>Loading Heihachi Audio Analysis Results</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f0f0f0;
        }
        
        .loader {
            text-align: center;
        }
        
        .spinner {
            border: 8px solid #f3f3f3;
            border-top: 8px solid #3498db;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 2s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="loader">
        <div class="spinner"></div>
        <h2>Loading Analysis Results</h2>
        <p>Please wait...</p>
    </div>
</body>
</html>''')
    
    # Create CSS file if it doesn't exist
    css_file = os.path.join(static_dir, 'style.css')
    if not os.path.exists(css_file):
        with open(css_file, 'w') as f:
            f.write('''/* Heihachi Web UI Styles */

body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f0f0f0;
}

header {
    background-color: #333;
    color: white;
    padding: 1rem;
    text-align: center;
}

.container {
    display: flex;
    margin: 1rem;
}

.sidebar {
    width: 300px;
    background-color: #fff;
    border-radius: 5px;
    padding: 1rem;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    margin-right: 1rem;
}

.main-content {
    flex: 1;
    background-color: #fff;
    border-radius: 5px;
    padding: 1rem;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
}

.file-info, .metrics, .features, .export {
    margin-bottom: 1.5rem;
}

h2 {
    margin-top: 0;
    font-size: 1.2rem;
    border-bottom: 1px solid #ddd;
    padding-bottom: 0.5rem;
}

table {
    width: 100%;
    border-collapse: collapse;
}

table td {
    padding: 0.3rem 0;
}

ul {
    list-style-type: none;
    padding: 0;
    margin: 0;
}

li {
    padding: 0.3rem 0;
}

.feature-link {
    text-decoration: none;
    color: #3498db;
}

.feature-link:hover {
    text-decoration: underline;
}

.tabs {
    display: flex;
    border-bottom: 1px solid #ddd;
    margin-bottom: 1rem;
}

.tab-button {
    background-color: transparent;
    border: none;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: background-color 0.3s;
}

.tab-button:hover {
    background-color: #f0f0f0;
}

.tab-button.active {
    border-bottom: 2px solid #3498db;
    font-weight: bold;
}

.plot {
    height: 500px;
}

select, button {
    padding: 0.5rem;
    margin-top: 0.5rem;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s;
}

button:hover {
    background-color: #2980b9;
}

#export-status {
    margin-top: 0.5rem;
}''')

# Create web UI instance
web_ui = WebUI()

def start_web_ui(result_file: str, host: str = "localhost", port: int = 5000, open_browser: bool = True) -> bool:
    """Start the web UI for visualization.
    
    Args:
        result_file: Path to the results file
        host: Host to bind to
        port: Port to listen on
        open_browser: Whether to open a browser window
        
    Returns:
        True if the UI started successfully, False otherwise
    """
    ensure_web_dirs()
    return web_ui.start(result_file, host, port, open_browser)

def is_running() -> bool:
    """Check if the web UI is running.
    
    Returns:
        True if the web UI is running, False otherwise
    """
    return web_ui.running

if __name__ == "__main__":
    # For standalone testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Heihachi Web UI for visualization")
    parser.add_argument("result_file", help="Path to the results file")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser window")
    
    args = parser.parse_args()
    
    if start_web_ui(args.result_file, args.host, args.port, not args.no_browser):
        print(f"Web UI started at http://{args.host}:{args.port}/")
        
        try:
            # Keep the server running until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Web UI stopped") 