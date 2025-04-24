#!/usr/bin/env python3
"""
Visualization utilities for audio analysis results.

This module provides functions for visualizing various aspects of audio analysis,
including waveforms, spectrograms, and feature comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Any, Union, List, Optional, Tuple

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def plot_waveform(waveform: np.ndarray, 
                  sample_rate: int, 
                  title: str = "Audio Waveform",
                  figsize: Tuple[int, int] = (10, 4)) -> Figure:
    """Plot audio waveform.
    
    Args:
        waveform: Audio waveform data as numpy array
        sample_rate: Sample rate of the audio
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert waveform to numpy array if it's not already
    if not isinstance(waveform, np.ndarray):
        waveform = np.array(waveform)
    
    # Calculate time axis
    if waveform.ndim == 1:
        # Mono audio
        duration = len(waveform) / sample_rate
        time = np.linspace(0, duration, len(waveform))
        ax.plot(time, waveform, color='blue', alpha=0.7)
    else:
        # Multi-channel audio
        duration = waveform.shape[1] / sample_rate
        time = np.linspace(0, duration, waveform.shape[1])
        for i, channel in enumerate(waveform):
            ax.plot(time, channel, label=f'Channel {i+1}', alpha=0.7)
        ax.legend()
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig


def plot_spectrogram(spectrogram: np.ndarray, 
                     sample_rate: int,
                     title: str = "Spectrogram",
                     figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """Plot spectrogram of audio.
    
    Args:
        spectrogram: Spectrogram data as numpy array
        sample_rate: Sample rate of the audio
        title: Plot title
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert to numpy array if it's not already
    if not isinstance(spectrogram, np.ndarray):
        spectrogram = np.array(spectrogram)
    
    # Ensure spectrogram is 2D
    if spectrogram.ndim > 2:
        # If it's a multi-channel spectrogram, use the first channel
        spectrogram = spectrogram[0]
    
    # Plot the spectrogram
    # Determine time and frequency axis
    if hasattr(spectrogram, 'shape'):
        duration = spectrogram.shape[1] * 4 / sample_rate  # Assuming hop length is sample_rate / 4
        extent = [0, duration, 0, sample_rate / 2]  # Time x Frequency
        
        im = ax.imshow(
            spectrogram, 
            aspect='auto', 
            origin='lower',
            extent=extent,
            cmap='viridis'
        )
        
        # Colorbar for magnitude
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)')
        
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_ylim(0, sample_rate / 2)  # Nyquist frequency
    else:
        ax.text(0.5, 0.5, "Invalid spectrogram data", 
                horizontalalignment='center', verticalalignment='center')
    
    fig.tight_layout()
    
    return fig


def plot_feature_comparison(feature_name: str, 
                           data: Dict[str, Dict[str, Any]],
                           title: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 6)) -> Figure:
    """Plot comparison of a feature across multiple audio files.
    
    Args:
        feature_name: Name of the feature to plot
        data: Dictionary mapping file names to dictionaries with "values",
              "sample_rate", and "duration" keys
        title: Plot title (defaults to feature name)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if not title:
        title = f"Comparison of {feature_name}"
    
    if not data:
        ax.text(0.5, 0.5, "No data to plot", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Plot each file's feature values
    for filename, file_data in data.items():
        values = file_data.get("values", [])
        sample_rate = file_data.get("sample_rate", 44100)
        duration = file_data.get("duration", len(values) / sample_rate)
        
        # Convert to numpy array if needed
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        
        # Create time axis
        time = np.linspace(0, duration, len(values))
        
        # Display filename without extension for legend
        display_name = filename.split(".")[0]
        
        # Plot line
        ax.plot(time, values, label=display_name, alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(feature_name)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig


def plot_feature_distribution(feature_name: str,
                             data: Dict[str, Dict[str, Any]],
                             title: Optional[str] = None,
                             figsize: Tuple[int, int] = (10, 6)) -> Figure:
    """Plot distribution of a feature across multiple audio files.
    
    Args:
        feature_name: Name of the feature to plot
        data: Dictionary mapping file names to dictionaries with "values"
        title: Plot title (defaults to feature name)
        figsize: Figure size (width, height) in inches
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if not title:
        title = f"Distribution of {feature_name}"
    
    if not data:
        ax.text(0.5, 0.5, "No data to plot", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Determine number of bins based on data
    all_values = []
    for file_data in data.values():
        values = file_data.get("values", [])
        if isinstance(values, (list, np.ndarray)):
            all_values.extend(values)
    
    n_bins = min(50, int(np.sqrt(len(all_values))))
    
    # Plot histograms for each file
    for filename, file_data in data.items():
        values = file_data.get("values", [])
        
        # Convert to numpy array if needed
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        
        # Display filename without extension for legend
        display_name = filename.split(".")[0]
        
        # Plot histogram
        ax.hist(values, bins=n_bins, alpha=0.5, label=display_name)
    
    ax.set_title(title)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    return fig


def create_summary_plots(result_data: Dict[str, Any], 
                         output_path: Optional[str] = None) -> List[Figure]:
    """Create a set of summary plots for an audio analysis result.
    
    Args:
        result_data: Dictionary containing analysis results
        output_path: Optional path to save plots
        
    Returns:
        List of matplotlib figures
    """
    figures = []
    
    # Basic file info
    file_path = result_data.get("file_path", "Unknown")
    filename = file_path.split("/")[-1] if isinstance(file_path, str) else "Unknown"
    sample_rate = result_data.get("sample_rate", 44100)
    
    # Plot waveform if available
    features = result_data.get("features", {})
    if "waveform" in features:
        waveform = features["waveform"]
        fig = plot_waveform(
            waveform, 
            sample_rate, 
            title=f"Waveform - {filename}"
        )
        figures.append(fig)
        
        if output_path:
            fig.savefig(f"{output_path}/waveform.png", dpi=300)
    
    # Plot spectrogram if available
    if "spectrogram" in features:
        spectrogram = features["spectrogram"]
        fig = plot_spectrogram(
            spectrogram, 
            sample_rate, 
            title=f"Spectrogram - {filename}"
        )
        figures.append(fig)
        
        if output_path:
            fig.savefig(f"{output_path}/spectrogram.png", dpi=300)
    
    # Plot other time-series features
    for feature_name, values in features.items():
        # Skip waveform and spectrogram, already plotted
        if feature_name in ["waveform", "spectrogram"]:
            continue
        
        # Skip if not a time series
        if not isinstance(values, (list, np.ndarray)) or len(values) <= 1:
            continue
        
        try:
            # Create time axis
            duration = result_data.get("duration", len(values) / sample_rate)
            time = np.linspace(0, duration, len(values))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time, values, color='blue', alpha=0.7)
            ax.set_title(f"{feature_name.capitalize()} - {filename}")
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(feature_name)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            figures.append(fig)
            
            if output_path:
                fig.savefig(f"{output_path}/{feature_name}.png", dpi=300)
        except Exception as e:
            logger.error(f"Error plotting feature {feature_name}: {e}")
    
    return figures 