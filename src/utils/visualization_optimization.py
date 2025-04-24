import numpy as np
import matplotlib
# Use Agg backend for better performance and to avoid GUI dependencies
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import os
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import gc
from pathlib import Path

from src.utils.logging_utils import get_logger
from src.utils.cache import result_cache, cached_result
from src.utils.progress import track_progress, update_progress, complete_progress

logger = get_logger(__name__)

class VisualizationOptimizer:
    """Utility for optimizing visualization generation for large audio files."""
    
    def __init__(
        self, 
        output_dir: str = "./visualizations",
        dpi: int = 100,
        max_points: int = 10000,
        figsize: Tuple[int, int] = (10, 6),
        cache_enabled: bool = True
    ):
        """Initialize visualization optimizer.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: DPI for output images
            max_points: Maximum number of points for time series visualizations
            figsize: Default figure size (width, height) in inches
            cache_enabled: Whether to cache visualization results
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.max_points = max_points
        self.figsize = figsize
        self.cache_enabled = cache_enabled
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Cache for results
        self._cache = {}
        
        logging.info(f"VisualizationOptimizer initialized with output directory: {output_dir}")
    
    def downsample_data(self, data: List[float], max_points: Optional[int] = None) -> List[float]:
        """Downsample data to a maximum number of points.
        
        Args:
            data: Data to downsample
            max_points: Maximum number of points (if None, use self.max_points)
            
        Returns:
            Downsampled data
        """
        if max_points is None:
            max_points = self.max_points
            
        if len(data) <= max_points:
            return data
        
        # Linear interpolation for downsampling
        indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
        return [data[i] for i in indices]
    
    def optimize_spectrogram(
        self, 
        spectrogram: np.ndarray, 
        max_resolution: Tuple[int, int] = (1000, 500)
    ) -> np.ndarray:
        """Optimize spectrogram data for visualization.
        
        Args:
            spectrogram: Spectrogram data
            max_resolution: Maximum resolution (width, height)
            
        Returns:
            Optimized spectrogram data
        """
        if spectrogram.shape[0] <= max_resolution[0] and spectrogram.shape[1] <= max_resolution[1]:
            return spectrogram
        
        # Determine scaling factors
        scale_y = max_resolution[0] / spectrogram.shape[0] if spectrogram.shape[0] > max_resolution[0] else 1
        scale_x = max_resolution[1] / spectrogram.shape[1] if spectrogram.shape[1] > max_resolution[1] else 1
        scale = min(scale_x, scale_y)
        
        # Calculate new dimensions
        new_shape = (
            int(spectrogram.shape[0] * scale),
            int(spectrogram.shape[1] * scale)
        )
        
        # Resize spectrogram
        from scipy.ndimage import zoom
        return zoom(spectrogram, (new_shape[0] / spectrogram.shape[0], new_shape[1] / spectrogram.shape[1]))
    
    def create_figure(
        self, 
        figsize: Optional[Tuple[int, int]] = None
    ) -> Figure:
        """Create a new figure with the specified size.
        
        Args:
            figsize: Figure size (width, height) in inches
            
        Returns:
            Matplotlib figure
        """
        if figsize is None:
            figsize = self.figsize
            
        return plt.figure(figsize=figsize)
    
    def save_figure(
        self, 
        fig: Figure, 
        filename: str, 
        dpi: Optional[int] = None,
        close_fig: bool = True
    ) -> str:
        """Save a figure to a file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: DPI for output image
            close_fig: Whether to close the figure after saving
            
        Returns:
            Path to the saved file
        """
        if dpi is None:
            dpi = self.dpi
            
        # Ensure the filename has an extension
        if not any(filename.endswith(ext) for ext in ['.png', '.jpg', '.pdf', '.svg']):
            filename += '.png'
            
        # Create full path
        output_path = os.path.join(self.output_dir, filename)
        
        # Save figure
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        
        # Close figure if requested
        if close_fig:
            plt.close(fig)
            
        return output_path
    
    def plot_waveform(
        self, 
        audio_data: List[float], 
        title: str = "Waveform",
        sample_rate: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save: bool = False,
        show: bool = False
    ) -> Optional[str]:
        """Plot the waveform of an audio file.
        
        Args:
            audio_data: Audio data
            title: Plot title
            sample_rate: Sample rate of the audio
            figsize: Figure size
            save: Whether to save the figure
            show: Whether to show the figure
            
        Returns:
            Path to the saved file if save=True, None otherwise
        """
        # Check cache
        cache_key = f"waveform_{title}"
        if self.cache_enabled and cache_key in self._cache:
            if save:
                return self._cache[cache_key]
            return None
            
        # Downsample data
        data = self.downsample_data(audio_data)
        
        # Create figure
        fig = self.create_figure(figsize)
        
        # Plot waveform
        if sample_rate:
            time_axis = np.linspace(0, len(audio_data) / sample_rate, len(data))
            plt.plot(time_axis, data)
            plt.xlabel("Time (s)")
        else:
            plt.plot(data)
            plt.xlabel("Sample")
            
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        # Save or show figure
        output_path = None
        if save:
            output_path = self.save_figure(
                fig, 
                f"waveform_{title.replace(' ', '_')}",
                close_fig=not show
            )
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = output_path
        
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
            
        return output_path
    
    def plot_spectrogram(
        self, 
        spectrogram: np.ndarray, 
        title: str = "Spectrogram",
        figsize: Optional[Tuple[int, int]] = None,
        save: bool = False,
        show: bool = False
    ) -> Optional[str]:
        """Plot a spectrogram.
        
        Args:
            spectrogram: Spectrogram data
            title: Plot title
            figsize: Figure size
            save: Whether to save the figure
            show: Whether to show the figure
            
        Returns:
            Path to the saved file if save=True, None otherwise
        """
        # Check cache
        cache_key = f"spectrogram_{title}"
        if self.cache_enabled and cache_key in self._cache:
            if save:
                return self._cache[cache_key]
            return None
            
        # Optimize spectrogram
        spec_data = self.optimize_spectrogram(spectrogram)
        
        # Create figure
        fig = self.create_figure(figsize)
        
        # Plot spectrogram
        plt.imshow(spec_data, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.xlabel("Time Frame")
        plt.ylabel("Frequency Bin")
        plt.tight_layout()
        
        # Save or show figure
        output_path = None
        if save:
            output_path = self.save_figure(
                fig, 
                f"spectrogram_{title.replace(' ', '_')}",
                close_fig=not show
            )
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = output_path
        
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
            
        return output_path
    
    def plot_feature(
        self, 
        feature_data: List[float], 
        feature_name: str,
        title: str = "Feature",
        figsize: Optional[Tuple[int, int]] = None,
        save: bool = False,
        show: bool = False
    ) -> Optional[str]:
        """Plot a feature.
        
        Args:
            feature_data: Feature data
            feature_name: Name of the feature
            title: Plot title
            figsize: Figure size
            save: Whether to save the figure
            show: Whether to show the figure
            
        Returns:
            Path to the saved file if save=True, None otherwise
        """
        # Check cache
        cache_key = f"feature_{feature_name}_{title}"
        if self.cache_enabled and cache_key in self._cache:
            if save:
                return self._cache[cache_key]
            return None
            
        # Downsample data
        data = self.downsample_data(feature_data)
        
        # Create figure
        fig = self.create_figure(figsize)
        
        # Plot feature
        plt.plot(data)
        plt.xlabel("Frame")
        plt.ylabel(feature_name)
        plt.title(f"{feature_name} - {title}")
        plt.grid(True, alpha=0.3)
        
        # Save or show figure
        output_path = None
        if save:
            output_path = self.save_figure(
                fig, 
                f"{feature_name}_{title.replace(' ', '_')}",
                close_fig=not show
            )
            
            # Cache result
            if self.cache_enabled:
                self._cache[cache_key] = output_path
        
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
            
        return output_path
    
    def compare_features(
        self, 
        feature_data_dict: Dict[str, List[float]], 
        feature_name: str,
        normalize: bool = True,
        figsize: Optional[Tuple[int, int]] = None,
        save: bool = False,
        show: bool = False
    ) -> Optional[str]:
        """Compare the same feature across multiple audio files.
        
        Args:
            feature_data_dict: Dictionary of {filename: feature_data}
            feature_name: Name of the feature
            normalize: Whether to normalize the data
            figsize: Figure size
            save: Whether to save the figure
            show: Whether to show the figure
            
        Returns:
            Path to the saved file if save=True, None otherwise
        """
        # Create figure
        fig = self.create_figure(figsize or (12, 6))
        
        # Plot each feature
        for filename, data in feature_data_dict.items():
            # Downsample data
            downsampled = self.downsample_data(data)
            
            # Normalize if requested
            if normalize and all(isinstance(x, (int, float)) for x in downsampled):
                data_min = min(downsampled)
                data_max = max(downsampled)
                if data_max > data_min:
                    downsampled = [(x - data_min) / (data_max - data_min) for x in downsampled]
            
            # Plot the data
            plt.plot(downsampled, label=os.path.basename(filename))
        
        plt.xlabel("Frame")
        plt.ylabel(feature_name + (" (Normalized)" if normalize else ""))
        plt.title(f"Comparison of {feature_name}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Save or show figure
        output_path = None
        if save:
            suffix = "_normalized" if normalize else ""
            output_path = self.save_figure(
                fig, 
                f"compare_{feature_name}{suffix}",
                close_fig=not show
            )
        
        if show:
            plt.show()
        elif not save:
            plt.close(fig)
            
        return output_path
    
    def generate_batch_visualizations(
        self, 
        audio_items: List[Dict[str, Any]],
        visualizations: List[str] = ['waveform', 'spectrogram'],
        save: bool = True,
        show: bool = False
    ) -> Dict[str, List[str]]:
        """Generate visualizations for multiple audio items.
        
        Args:
            audio_items: List of dictionaries with audio data
            visualizations: List of visualizations to generate
            save: Whether to save the figures
            show: Whether to show the figures
            
        Returns:
            Dictionary of {visualization_type: [paths]}
        """
        results = {viz_type: [] for viz_type in visualizations}
        
        total_items = len(audio_items)
        logging.info(f"Generating {len(visualizations)} visualization types for {total_items} audio items")
        
        # Process each audio item
        for i, item in enumerate(audio_items):
            filename = item.get('filename', f"audio_{i}")
            logging.debug(f"Processing {filename} ({i+1}/{total_items})")
            
            try:
                # Generate visualizations
                if 'waveform' in visualizations and 'audio_data' in item:
                    path = self.plot_waveform(
                        item['audio_data'],
                        title=filename,
                        sample_rate=item.get('sample_rate'),
                        save=save,
                        show=show
                    )
                    if path:
                        results['waveform'].append(path)
                
                if 'spectrogram' in visualizations and 'spectrogram' in item:
                    path = self.plot_spectrogram(
                        item['spectrogram'],
                        title=filename,
                        save=save,
                        show=show
                    )
                    if path:
                        results['spectrogram'].append(path)
                
                # Process features
                if 'features' in item and isinstance(item['features'], dict):
                    for feature_name, feature_data in item['features'].items():
                        if isinstance(feature_data, list) and len(feature_data) > 0:
                            if feature_name in visualizations:
                                path = self.plot_feature(
                                    feature_data,
                                    feature_name,
                                    title=filename,
                                    save=save,
                                    show=show
                                )
                                if path:
                                    if feature_name not in results:
                                        results[feature_name] = []
                                    results[feature_name].append(path)
            
            except Exception as e:
                logging.error(f"Error generating visualizations for {filename}: {e}")
        
        # Log summary
        for viz_type, paths in results.items():
            logging.info(f"Generated {len(paths)} {viz_type} visualizations")
        
        return results
    
    def clear_cache(self):
        """Clear the visualization cache."""
        self._cache.clear()
        logging.debug("Visualization cache cleared")


# For direct testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test data
    test_audio = np.sin(np.linspace(0, 10 * np.pi, 1000))
    test_spectrogram = np.random.random((100, 100))
    
    # Create optimizer
    viz = VisualizationOptimizer(output_dir="./test_visualizations")
    
    # Test waveform
    viz.plot_waveform(test_audio, title="Test Waveform", save=True)
    
    # Test spectrogram
    viz.plot_spectrogram(test_spectrogram, title="Test Spectrogram", save=True)
    
    # Test feature
    viz.plot_feature(test_audio, "Amplitude", title="Test Feature", save=True)
    
    logging.info("Test visualizations generated in ./test_visualizations")


# Helper functions for easy access

def optimize_audio_visualization(audio: np.ndarray, sr: int, 
                               output_filename: str, resolution: str = "medium") -> str:
    """Generate optimized waveform visualization for audio data.
    
    Args:
        audio: Audio data
        sr: Sample rate
        output_filename: Output filename
        resolution: Desired resolution (low, medium, high)
        
    Returns:
        Path to visualization file
    """
    optimizer = VisualizationOptimizer()
    return optimizer.plot_waveform(audio, sr, output_filename, resolution=resolution)

def optimize_spectrogram_visualization(spectrogram: np.ndarray, sr: int, 
                                     hop_length: int, output_filename: str, 
                                     resolution: str = "medium") -> str:
    """Generate optimized spectrogram visualization.
    
    Args:
        spectrogram: Spectrogram data
        sr: Sample rate
        hop_length: Hop length
        output_filename: Output filename
        resolution: Desired resolution (low, medium, high)
        
    Returns:
        Path to visualization file
    """
    optimizer = VisualizationOptimizer()
    return optimizer.plot_spectrogram(spectrogram, sr, hop_length, 
                                     output_filename, resolution=resolution) 