"""
Interactive web application for exploring audio analysis results.
Uses Dash and Plotly for visualization and interaction.
"""

import os
import json
import logging
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Dict, List, Any, Optional, Union, Tuple

import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from ..utils.visualization_optimization import VisualizationOptimizer


# Initialize the Dash app with Bootstrap CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Global variables
results_directory = None
results_data = {}
comparison_data = None


def load_results(directory: str) -> Dict[str, Any]:
    """Load all JSON result files from a directory.
    
    Args:
        directory: Path to directory containing result files
    
    Returns:
        Dictionary mapping filenames to result data
    """
    result_files = {}
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logging.error(f"Results directory {directory} does not exist")
        return {}
    
    # Load all JSON files
    for file_path in dir_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                result_files[file_path.stem] = data
                logging.info(f"Loaded {file_path.name}")
        except Exception as e:
            logging.warning(f"Error loading {file_path.name}: {e}")
    
    return result_files


def create_layout() -> html.Div:
    """Create the main layout for the Dash app.
    
    Returns:
        Dash layout
    """
    # Navbar with app title
    navbar = dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/logo.png", height="30px"), width="auto"),
                    dbc.Col(dbc.NavbarBrand("Heihachi - Audio Analysis Explorer", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"},
            )
        ]),
        color="dark",
        dark=True,
        className="mb-4",
    )
    
    # File selector
    file_selector = dbc.Card([
        dbc.CardHeader("Select Audio File"),
        dbc.CardBody([
            dcc.Dropdown(
                id="file-dropdown",
                options=[],
                placeholder="Select a file to explore",
            ),
        ]),
    ], className="mb-3")
    
    # View selector
    view_selector = dbc.Card([
        dbc.CardHeader("Select View"),
        dbc.CardBody([
            dbc.Tabs([
                dbc.Tab(label="Waveform", tab_id="waveform"),
                dbc.Tab(label="Spectrogram", tab_id="spectrogram"),
                dbc.Tab(label="Features", tab_id="features"),
                dbc.Tab(label="Metrics", tab_id="metrics"),
                dbc.Tab(label="Compare", tab_id="compare"),
            ], id="view-tabs", active_tab="waveform"),
        ]),
    ], className="mb-3")
    
    # Main content area
    content = dbc.Card([
        dbc.CardHeader(html.Div(id="content-header")),
        dbc.CardBody([
            html.Div(id="content-body"),
        ]),
    ], className="mb-3")
    
    # Metadata and details
    details = dbc.Card([
        dbc.CardHeader("Details"),
        dbc.CardBody([
            html.Div(id="details-content"),
        ]),
    ])
    
    # Export options
    export_options = dbc.Card([
        dbc.CardHeader("Export Options"),
        dbc.CardBody([
            dbc.Button("Export Current View", id="export-button", color="primary", className="me-2"),
            dcc.Download(id="download-data"),
            dbc.FormText("Export the current view as PNG, CSV, or JSON"),
        ]),
    ], className="mb-3")
    
    # Layout with sidebar and content
    return html.Div([
        navbar,
        dbc.Container([
            dbc.Row([
                # Sidebar
                dbc.Col([
                    file_selector,
                    view_selector,
                    export_options,
                ], width=3),
                
                # Main content
                dbc.Col([
                    content,
                    details,
                ], width=9),
            ]),
        ]),
    ])


@callback(
    Output("file-dropdown", "options"),
    Input("file-dropdown", "search_value"),
)
def update_file_options(search_value):
    """Update file dropdown options based on search value.
    
    Args:
        search_value: Search string
    
    Returns:
        List of dropdown options
    """
    if not results_data:
        return []
    
    # Filter files based on search value
    if search_value:
        filtered_files = [f for f in results_data.keys() if search_value.lower() in f.lower()]
    else:
        filtered_files = list(results_data.keys())
    
    return [{"label": f, "value": f} for f in filtered_files]


@callback(
    [Output("content-header", "children"),
     Output("content-body", "children"),
     Output("details-content", "children")],
    [Input("file-dropdown", "value"),
     Input("view-tabs", "active_tab")],
)
def update_content(file_name, active_tab):
    """Update content based on selected file and view.
    
    Args:
        file_name: Selected file name
        active_tab: Active tab ID
    
    Returns:
        Tuple of (header, body, details) content
    """
    if not file_name or file_name not in results_data:
        return "No file selected", "Please select a file to view", ""
    
    data = results_data[file_name]
    
    # Header with file name
    header = html.H5(f"File: {file_name}")
    
    # Details panel content
    details = create_details_panel(data)
    
    # Main content based on active tab
    if active_tab == "waveform":
        body = create_waveform_view(data, file_name)
    elif active_tab == "spectrogram":
        body = create_spectrogram_view(data, file_name)
    elif active_tab == "features":
        body = create_features_view(data, file_name)
    elif active_tab == "metrics":
        body = create_metrics_view(data, file_name)
    elif active_tab == "compare":
        body = create_compare_view(file_name)
    else:
        body = html.Div("Select a view to display")
    
    return header, body, details


def create_waveform_view(data: Dict[str, Any], file_name: str) -> html.Div:
    """Create waveform visualization view.
    
    Args:
        data: Analysis result data
        file_name: Name of the file
    
    Returns:
        Dash component with waveform visualization
    """
    # Check if waveform data is available
    if "waveform" not in data or not data["waveform"]:
        return html.Div("Waveform data not available for this file")
    
    # Get waveform data
    waveform = data.get("waveform", [])
    
    # If waveform is too large, downsample it
    if len(waveform) > 10000:
        step = len(waveform) // 10000
        waveform = waveform[::step]
    
    # Create time axis
    sample_rate = data.get("sample_rate", 44100)
    time = np.arange(len(waveform)) / sample_rate
    
    # Create waveform figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=waveform,
        mode="lines",
        name="Waveform",
        line=dict(color="rgba(0, 123, 255, 0.8)", width=1),
    ))
    
    # Add markers for detected events if available
    if "events" in data and data["events"]:
        events = data["events"]
        event_times = [e["time"] for e in events]
        event_labels = [e["label"] for e in events]
        
        fig.add_trace(go.Scatter(
            x=event_times,
            y=[0] * len(event_times),
            mode="markers+text",
            text=event_labels,
            textposition="top center",
            marker=dict(size=10, color="red"),
            name="Events"
        ))
    
    # Layout
    fig.update_layout(
        title=f"Waveform: {file_name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    return html.Div([
        dcc.Graph(id="waveform-graph", figure=fig),
        html.Hr(),
        html.H6("Playback Controls"),
        html.Audio(
            id="audio-player",
            controls=True,
            src=f"/audio/{file_name}",
            style={"width": "100%"}
        ),
    ])


def create_spectrogram_view(data: Dict[str, Any], file_name: str) -> html.Div:
    """Create spectrogram visualization view.
    
    Args:
        data: Analysis result data
        file_name: Name of the file
    
    Returns:
        Dash component with spectrogram visualization
    """
    # Check if spectrogram data is available
    if "spectrogram" not in data or not data["spectrogram"]:
        return html.Div("Spectrogram data not available for this file")
    
    # Get spectrogram data
    spectrogram = np.array(data.get("spectrogram", []))
    
    if len(spectrogram.shape) != 2:
        return html.Div("Invalid spectrogram data format")
    
    # Create time and frequency axes
    sample_rate = data.get("sample_rate", 44100)
    hop_length = data.get("hop_length", 512)
    n_fft = data.get("n_fft", 2048)
    
    time_steps = spectrogram.shape[1]
    time = np.arange(time_steps) * hop_length / sample_rate
    
    freq_bins = spectrogram.shape[0]
    freqs = np.arange(freq_bins) * sample_rate / (2 * freq_bins)
    
    # Create spectrogram figure
    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        x=time,
        y=freqs,
        colorscale="Viridis",
    ))
    
    # Layout
    fig.update_layout(
        title=f"Spectrogram: {file_name}",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    
    # Controls for spectrogram settings
    controls = html.Div([
        html.Hr(),
        html.H6("Spectrogram Settings"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Color Scale"),
                dcc.Dropdown(
                    id="colorscale-dropdown",
                    options=[
                        {"label": "Viridis", "value": "Viridis"},
                        {"label": "Plasma", "value": "Plasma"},
                        {"label": "Inferno", "value": "Inferno"},
                        {"label": "Magma", "value": "Magma"},
                        {"label": "Cividis", "value": "Cividis"},
                    ],
                    value="Viridis",
                ),
            ], width=4),
            
            dbc.Col([
                dbc.Label("Log Scale"),
                dbc.Switch(
                    id="log-scale-switch",
                    value=True,
                ),
            ], width=4),
        ]),
    ])
    
    return html.Div([
        dcc.Graph(id="spectrogram-graph", figure=fig),
        controls,
    ])


def create_features_view(data: Dict[str, Any], file_name: str) -> html.Div:
    """Create features visualization view.
    
    Args:
        data: Analysis result data
        file_name: Name of the file
    
    Returns:
        Dash component with features visualization
    """
    # Check if features data is available
    if "features" not in data or not data["features"]:
        return html.Div("Features data not available for this file")
    
    features = data.get("features", {})
    
    # Convert features to DataFrame
    feature_data = []
    for feature_name, values in features.items():
        if isinstance(values, list):
            # Time series feature
            time_steps = len(values)
            sample_rate = data.get("sample_rate", 44100)
            hop_length = data.get("hop_length", 512)
            
            time = np.arange(time_steps) * hop_length / sample_rate
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time,
                y=values,
                mode="lines",
                name=feature_name,
            ))
            
            fig.update_layout(
                title=f"Feature: {feature_name}",
                xaxis_title="Time (seconds)",
                yaxis_title="Value",
                template="plotly_white",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20),
            )
            
            feature_data.append(html.Div([
                dcc.Graph(figure=fig),
                html.Hr(),
            ]))
        else:
            # Scalar feature
            feature_data.append(dbc.Row([
                dbc.Col(html.Strong(f"{feature_name}:"), width=4),
                dbc.Col(html.Span(f"{values}"), width=8),
                html.Hr(),
            ]))
    
    return html.Div([
        html.H6("Audio Features"),
        html.Div(feature_data),
    ])


def create_metrics_view(data: Dict[str, Any], file_name: str) -> html.Div:
    """Create metrics visualization view.
    
    Args:
        data: Analysis result data
        file_name: Name of the file
    
    Returns:
        Dash component with metrics visualization
    """
    # Extract metrics from data
    metrics = {}
    
    # BPM and tempo
    if "tempo" in data:
        metrics["BPM"] = data["tempo"].get("bpm", "N/A")
        metrics["Tempo"] = data["tempo"].get("label", "N/A")
    
    # Key and scale
    if "key" in data:
        metrics["Key"] = data["key"].get("key", "N/A")
        metrics["Scale"] = data["key"].get("scale", "N/A")
        metrics["Key Confidence"] = f"{data['key'].get('confidence', 0):.2f}"
    
    # Audio quality metrics
    if "quality" in data:
        quality = data["quality"]
        for metric_name, value in quality.items():
            if isinstance(value, (int, float)):
                metrics[f"Quality: {metric_name}"] = f"{value:.2f}"
            else:
                metrics[f"Quality: {metric_name}"] = str(value)
    
    # Create metrics table
    metrics_table = dbc.Table.from_dataframe(
        pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Value": list(metrics.values()),
        }),
        striped=True,
        bordered=True,
        hover=True,
    )
    
    # Create radar chart for quality metrics
    quality_metrics = {k: v for k, v in metrics.items() if k.startswith("Quality:") and isinstance(v, str) and v not in ("N/A", "Unknown")}
    
    if quality_metrics:
        # Extract numeric values
        quality_values = []
        quality_names = []
        
        for name, value_str in quality_metrics.items():
            try:
                value = float(value_str)
                quality_values.append(value)
                quality_names.append(name.replace("Quality: ", ""))
            except (ValueError, TypeError):
                pass
        
        if quality_values:
            # Normalize values between 0 and 1
            min_val = min(quality_values)
            max_val = max(quality_values)
            
            if max_val > min_val:
                normalized_values = [(v - min_val) / (max_val - min_val) for v in quality_values]
            else:
                normalized_values = [0.5] * len(quality_values)
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=quality_names,
                fill='toself',
                name='Quality Metrics'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=False,
                title="Quality Metrics",
                height=400,
            )
            
            radar_chart = dcc.Graph(figure=fig)
        else:
            radar_chart = html.Div("Not enough quality metrics for visualization")
    else:
        radar_chart = html.Div("No quality metrics available")
    
    return html.Div([
        html.H6("Audio Metrics"),
        metrics_table,
        html.Hr(),
        radar_chart,
    ])


def create_compare_view(file_name: str) -> html.Div:
    """Create comparison view for multiple files.
    
    Args:
        file_name: Currently selected file name
    
    Returns:
        Dash component with comparison visualization
    """
    if not results_data or len(results_data) <= 1:
        return html.Div("Need at least two files to compare")
    
    # File selection for comparison
    file_selector = html.Div([
        html.H6("Select Files to Compare"),
        dcc.Checklist(
            id="compare-files-checklist",
            options=[{"label": f, "value": f} for f in results_data.keys()],
            value=[file_name] if file_name else [],
            labelStyle={"display": "block"},
        ),
        html.Hr(),
    ])
    
    # Metric selection for comparison
    metrics = set()
    for data in results_data.values():
        metrics.update(extract_metrics(data).keys())
    
    metric_selector = html.Div([
        html.H6("Select Metrics to Compare"),
        dcc.Checklist(
            id="compare-metrics-checklist",
            options=[{"label": m, "value": m} for m in sorted(metrics)],
            value=[next(iter(metrics))] if metrics else [],
            labelStyle={"display": "block"},
        ),
        html.Hr(),
    ])
    
    # Comparison result placeholder
    comparison_result = html.Div(id="comparison-result")
    
    return html.Div([
        file_selector,
        metric_selector,
        comparison_result,
    ])


@callback(
    Output("comparison-result", "children"),
    [Input("compare-files-checklist", "value"),
     Input("compare-metrics-checklist", "value")],
)
def update_comparison(selected_files, selected_metrics):
    """Update comparison visualization based on selected files and metrics.
    
    Args:
        selected_files: List of selected file names
        selected_metrics: List of selected metric names
    
    Returns:
        Dash component with comparison visualization
    """
    if not selected_files or len(selected_files) < 2 or not selected_metrics:
        return html.Div("Select at least two files and one metric to compare")
    
    # Extract selected metrics from each file
    comparison_data = {}
    
    for file_name in selected_files:
        if file_name in results_data:
            metrics = extract_metrics(results_data[file_name])
            comparison_data[file_name] = {m: metrics.get(m, "N/A") for m in selected_metrics}
    
    # Convert to DataFrame for visualization
    df = pd.DataFrame(comparison_data).T
    
    # Create comparison figures
    figures = []
    
    for metric in selected_metrics:
        # Extract numeric values
        numeric_data = {}
        
        for file_name, values in comparison_data.items():
            try:
                value = float(values.get(metric, "N/A"))
                numeric_data[file_name] = value
            except (ValueError, TypeError):
                pass
        
        if numeric_data:
            # Create bar chart
            fig = px.bar(
                x=list(numeric_data.keys()),
                y=list(numeric_data.values()),
                labels={"x": "File", "y": metric},
                title=f"Comparison of {metric}",
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,
                margin=dict(l=20, r=20, t=50, b=100),
                height=400,
            )
            
            figures.append(dcc.Graph(figure=fig))
        else:
            figures.append(html.Div(f"No numeric data available for {metric}"))
    
    # Create comparison table
    table = dbc.Table.from_dataframe(
        df,
        striped=True,
        bordered=True,
        hover=True,
        index=True,
    )
    
    return html.Div([
        html.H6("Comparison Results"),
        table,
        html.Hr(),
        html.Div(figures),
    ])


def create_details_panel(data: Dict[str, Any]) -> html.Div:
    """Create details panel with file metadata.
    
    Args:
        data: Analysis result data
    
    Returns:
        Dash component with file details
    """
    # Extract metadata
    metadata = {}
    
    # Audio properties
    metadata["Sample Rate"] = f"{data.get('sample_rate', 'N/A')} Hz"
    metadata["Channels"] = data.get("channels", "N/A")
    metadata["Duration"] = f"{data.get('duration', 'N/A'):.2f} seconds"
    metadata["Format"] = data.get("format", "N/A")
    
    # Analysis metadata
    metadata["Analysis Time"] = data.get("analysis_time", "N/A")
    metadata["Pipeline"] = data.get("pipeline", "N/A")
    metadata["Version"] = data.get("version", "N/A")
    
    # Create metadata table
    metadata_items = []
    for key, value in metadata.items():
        metadata_items.append(dbc.Row([
            dbc.Col(html.Strong(f"{key}:"), width=4),
            dbc.Col(html.Span(f"{value}"), width=8),
        ]))
    
    return html.Div([
        html.H6("File Metadata"),
        html.Div(metadata_items),
    ])


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metrics from result data.
    
    Args:
        data: Analysis result data
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # BPM and tempo
    if "tempo" in data:
        metrics["BPM"] = data["tempo"].get("bpm", "N/A")
        metrics["Tempo"] = data["tempo"].get("label", "N/A")
    
    # Key and scale
    if "key" in data:
        metrics["Key"] = data["key"].get("key", "N/A")
        metrics["Scale"] = data["key"].get("scale", "N/A")
        metrics["Key Confidence"] = data["key"].get("confidence", "N/A")
    
    # Audio quality metrics
    if "quality" in data:
        quality = data["quality"]
        for metric_name, value in quality.items():
            metrics[f"Quality_{metric_name}"] = value
    
    # Features
    if "features" in data:
        features = data["features"]
        for feature_name, value in features.items():
            if not isinstance(value, list):
                metrics[f"Feature_{feature_name}"] = value
    
    return metrics


@callback(
    Output("download-data", "data"),
    Input("export-button", "n_clicks"),
    [State("file-dropdown", "value"),
     State("view-tabs", "active_tab")],
    prevent_initial_call=True,
)
def export_data(n_clicks, file_name, active_tab):
    """Export current view data.
    
    Args:
        n_clicks: Button click count
        file_name: Selected file name
        active_tab: Active tab ID
    
    Returns:
        Download data
    """
    if not file_name or file_name not in results_data:
        return None
    
    data = results_data[file_name]
    
    if active_tab == "waveform":
        # Export waveform as CSV
        waveform = data.get("waveform", [])
        sample_rate = data.get("sample_rate", 44100)
        time = np.arange(len(waveform)) / sample_rate
        
        df = pd.DataFrame({
            "time": time,
            "amplitude": waveform,
        })
        
        return dcc.send_data_frame(df.to_csv, f"{file_name}_waveform.csv")
        
    elif active_tab == "spectrogram":
        # Export spectrogram data as CSV
        spectrogram = np.array(data.get("spectrogram", []))
        
        df = pd.DataFrame(spectrogram)
        
        return dcc.send_data_frame(df.to_csv, f"{file_name}_spectrogram.csv")
        
    elif active_tab == "features" or active_tab == "metrics":
        # Export features/metrics as JSON
        return dict(
            content=json.dumps(data, indent=2),
            filename=f"{file_name}_data.json",
            type="application/json",
        )
        
    elif active_tab == "compare":
        # Export comparison data as CSV
        return dcc.send_data_frame(pd.DataFrame(comparison_data).to_csv, "comparison.csv")
    
    return None


def serve_audio_file(filename):
    """Serve audio file for playback.
    
    Args:
        filename: Audio file name
    
    Returns:
        Audio file data
    """
    # This would need to be implemented in a real Flask/Dash app
    # For now, we'll just return a placeholder
    return "Audio file data"


def start_interactive_app(results_dir: str, host: str = "localhost", port: int = 8050, open_browser: bool = True) -> int:
    """Start the interactive web application.
    
    Args:
        results_dir: Directory containing analysis results
        host: Host to bind to
        port: Port to listen on
        open_browser: Whether to open a browser window automatically
    
    Returns:
        Exit code
    """
    global results_directory, results_data
    
    # Load results data
    results_directory = results_dir
    results_data = load_results(results_dir)
    
    if not results_data:
        logging.error(f"No result files found in {results_dir}")
        return 1
    
    # Create app layout
    app.layout = create_layout()
    
    # Open browser if requested
    if open_browser:
        Timer(1, lambda: webbrowser.open_new(f"http://{host}:{port}")).start()
    
    # Run the app
    try:
        app.run_server(host=host, port=port, debug=False)
        return 0
    except Exception as e:
        logging.error(f"Error starting interactive app: {e}")
        return 1


def start_interactive_comparison(comparison_results: Dict[str, Any], host: str = "localhost", port: int = 8050, open_browser: bool = True) -> int:
    """Start the interactive web application with comparison results.
    
    Args:
        comparison_results: Comparison results data
        host: Host to bind to
        port: Port to listen on
        open_browser: Whether to open a browser window automatically
    
    Returns:
        Exit code
    """
    global comparison_data
    
    # Set comparison data
    comparison_data = comparison_results
    
    # Create app layout
    app.layout = create_layout()
    
    # Open browser if requested
    if open_browser:
        Timer(1, lambda: webbrowser.open_new(f"http://{host}:{port}")).start()
    
    # Run the app
    try:
        app.run_server(host=host, port=port, debug=False)
        return 0
    except Exception as e:
        logging.error(f"Error starting interactive app: {e}")
        return 1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python app.py RESULTS_DIRECTORY [--host HOST] [--port PORT] [--no-browser]")
        sys.exit(1)
    
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Heihachi Interactive Results Explorer")
    parser.add_argument("results_dir", help="Directory containing analysis results")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8050, help="Port to listen on")
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser window automatically")
    
    args = parser.parse_args()
    
    # Start the app
    sys.exit(start_interactive_app(
        results_dir=args.results_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser
    )) 