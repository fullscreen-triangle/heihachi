"""
Utility module for comparing analysis results from multiple audio files.
Provides tools for generating comparison reports, identifying trends, and visualizing differences.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy import stats
import seaborn as sns
from sklearn.cluster import KMeans

from ..utils.visualization_optimization import VisualizationOptimizer


class ResultComparison:
    """Compare analysis results from multiple audio files."""

    def __init__(self, 
                 output_dir: Optional[str] = None,
                 visualization_config: Optional[Dict[str, Any]] = None):
        """Initialize the result comparison utility.
        
        Args:
            output_dir: Directory to save comparison results
            visualization_config: Configuration for visualizations
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        self.visualization_config = visualization_config or {}
        self.viz_optimizer = VisualizationOptimizer(
            output_dir=output_dir,
            dpi=self.visualization_config.get('dpi', 100),
            max_points=self.visualization_config.get('max_points', 10000)
        )
        
        # Results data
        self.results_data = {}
        self.summary_data = None
        self.comparison_metrics = None
        
    def load_results(self, results_dir: str) -> Dict[str, Any]:
        """Load all JSON result files from a directory.
        
        Args:
            results_dir: Path to directory containing result files
            
        Returns:
            Dictionary of results data
        """
        dir_path = Path(results_dir)
        if not dir_path.exists():
            logging.error(f"Results directory {results_dir} does not exist")
            return {}
        
        # Load all JSON files
        file_count = 0
        for file_path in dir_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.results_data[file_path.stem] = data
                    file_count += 1
                    logging.info(f"Loaded {file_path.name}")
            except Exception as e:
                logging.warning(f"Error loading {file_path.name}: {e}")
                
        logging.info(f"Loaded {file_count} result files from {results_dir}")
        return self.results_data
    
    def load_multiple_directories(self, directories: List[str]) -> Dict[str, Any]:
        """Load results from multiple directories.
        
        Args:
            directories: List of directories containing result files
            
        Returns:
            Dictionary of results data
        """
        for directory in directories:
            self.load_results(directory)
        
        return self.results_data
    
    def load_specific_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Load specific result files.
        
        Args:
            file_paths: List of paths to result files
            
        Returns:
            Dictionary of results data
        """
        for file_path in file_paths:
            path = Path(file_path)
            if not path.exists():
                logging.warning(f"File not found: {file_path}")
                continue
                
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    self.results_data[path.stem] = data
                    logging.info(f"Loaded {path.name}")
            except Exception as e:
                logging.warning(f"Error loading {path.name}: {e}")
                
        return self.results_data
    
    def extract_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant metrics from result data.
        
        Args:
            data: Analysis result data
            
        Returns:
            Dictionary of extracted metrics
        """
        metrics = {}
        
        # Basic audio properties
        metrics["duration"] = data.get("duration", 0)
        metrics["sample_rate"] = data.get("sample_rate", 0)
        metrics["channels"] = data.get("channels", 0)
        
        # BPM and tempo
        if "tempo" in data:
            metrics["bpm"] = data["tempo"].get("bpm", 0)
            metrics["tempo_label"] = data["tempo"].get("label", "")
        
        # Key and scale
        if "key" in data:
            metrics["key"] = data["key"].get("key", "")
            metrics["scale"] = data["key"].get("scale", "")
            metrics["key_confidence"] = data["key"].get("confidence", 0)
        
        # Audio quality metrics
        if "quality" in data:
            quality = data["quality"]
            for metric_name, value in quality.items():
                if isinstance(value, (int, float)):
                    metrics[f"quality_{metric_name}"] = value
        
        # Features
        if "features" in data:
            features = data["features"]
            for feature_name, value in features.items():
                if not isinstance(value, list):
                    metrics[f"feature_{feature_name}"] = value
        
        # Processing metadata
        metrics["analysis_time"] = data.get("analysis_time", 0)
        metrics["memory_usage"] = data.get("memory_usage", 0)
        
        return metrics
    
    def generate_summary(self) -> pd.DataFrame:
        """Generate a summary of results for all files.
        
        Returns:
            DataFrame with summary information
        """
        if not self.results_data:
            logging.warning("No results data available for summary")
            return pd.DataFrame()
        
        # Extract metrics for all files
        summary_data = {}
        for file_name, data in self.results_data.items():
            summary_data[file_name] = self.extract_metrics(data)
        
        # Convert to DataFrame
        self.summary_data = pd.DataFrame.from_dict(summary_data, orient='index')
        
        # Add file names as a column
        self.summary_data['file_name'] = self.summary_data.index
        
        return self.summary_data
    
    def compare_metrics(self, 
                        metrics: Optional[List[str]] = None,
                        group_by: Optional[str] = None) -> pd.DataFrame:
        """Compare specific metrics across all files.
        
        Args:
            metrics: List of metric names to compare (if None, compare all numeric metrics)
            group_by: Column to group results by
            
        Returns:
            DataFrame with comparison results
        """
        if self.summary_data is None:
            self.generate_summary()
            
        if self.summary_data.empty:
            return pd.DataFrame()
        
        # Filter numeric columns if no specific metrics are provided
        if metrics is None:
            numeric_cols = self.summary_data.select_dtypes(include=[np.number]).columns
            metrics = list(numeric_cols)
        else:
            # Filter metrics that exist in the data
            metrics = [m for m in metrics if m in self.summary_data.columns]
            
        if not metrics:
            logging.warning("No valid metrics found for comparison")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        comparison_data = self.summary_data[metrics].copy()
        
        # Add basic statistics
        comparison_stats = pd.DataFrame({
            'mean': comparison_data.mean(),
            'median': comparison_data.median(),
            'std': comparison_data.std(),
            'min': comparison_data.min(),
            'max': comparison_data.max(),
            'range': comparison_data.max() - comparison_data.min(),
            'count': comparison_data.count()
        })
        
        # Apply grouping if specified
        if group_by and group_by in self.summary_data.columns:
            grouped = self.summary_data.groupby(group_by)[metrics].agg(['mean', 'median', 'std', 'min', 'max', 'count'])
            grouped['range'] = grouped['max'] - grouped['min']
            comparison_stats = grouped
        
        self.comparison_metrics = comparison_stats
        return comparison_stats
    
    def find_outliers(self, 
                      metrics: Optional[List[str]] = None, 
                      threshold: float = 2.0) -> Dict[str, List[str]]:
        """Find outliers in the results data.
        
        Args:
            metrics: List of metric names to check for outliers
            threshold: Z-score threshold for identifying outliers
            
        Returns:
            Dictionary mapping metrics to lists of outlier file names
        """
        if self.summary_data is None:
            self.generate_summary()
            
        if self.summary_data.empty:
            return {}
        
        # Filter numeric columns if no specific metrics are provided
        if metrics is None:
            numeric_cols = self.summary_data.select_dtypes(include=[np.number]).columns
            metrics = list(numeric_cols)
        else:
            # Filter metrics that exist in the data
            metrics = [m for m in metrics if m in self.summary_data.columns]
            
        if not metrics:
            logging.warning("No valid metrics found for outlier detection")
            return {}
        
        # Find outliers for each metric
        outliers = {}
        for metric in metrics:
            # Calculate z-scores
            z_scores = stats.zscore(self.summary_data[metric].dropna())
            
            # Find outlier indices
            outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
            
            if len(outlier_indices) > 0:
                # Get file names for outliers
                files_with_metric = self.summary_data[~self.summary_data[metric].isna()]
                outlier_files = files_with_metric.iloc[outlier_indices].index.tolist()
                outliers[metric] = outlier_files
        
        return outliers
    
    def identify_clusters(self, 
                         metrics: List[str], 
                         n_clusters: int = 2) -> Dict[str, List[str]]:
        """Identify clusters in the results data.
        
        Args:
            metrics: List of metric names to use for clustering
            n_clusters: Number of clusters to identify
            
        Returns:
            Dictionary mapping cluster IDs to lists of file names
        """
        if self.summary_data is None:
            self.generate_summary()
            
        if self.summary_data.empty:
            return {}
        
        # Filter metrics that exist in the data
        metrics = [m for m in metrics if m in self.summary_data.columns]
        if not metrics:
            logging.warning("No valid metrics found for clustering")
            return {}
        
        # Import here to avoid dependency if not used
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Prepare data for clustering
        cluster_data = self.summary_data[metrics].dropna()
        if len(cluster_data) < n_clusters:
            logging.warning(f"Too few valid data points ({len(cluster_data)}) for {n_clusters} clusters")
            return {}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # Map files to clusters
        clusters = defaultdict(list)
        for i, file_name in enumerate(cluster_data.index):
            cluster_id = f"Cluster {cluster_labels[i]}"
            clusters[cluster_id].append(file_name)
        
        return dict(clusters)
    
    def plot_metric_distribution(self, 
                                metric: str, 
                                plot_type: str = 'boxplot',
                                group_by: Optional[str] = None) -> Optional[Figure]:
        """Plot the distribution of a metric across all files.
        
        Args:
            metric: Metric name to plot
            plot_type: Type of plot ('boxplot', 'histogram', or 'violin')
            group_by: Column to group results by
            
        Returns:
            Matplotlib figure
        """
        if self.summary_data is None:
            self.generate_summary()
            
        if self.summary_data.empty or metric not in self.summary_data.columns:
            logging.warning(f"Metric {metric} not found in data")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Apply grouping if specified
        if group_by and group_by in self.summary_data.columns:
            data = self.summary_data[[metric, group_by]].copy()
            
            if plot_type == 'boxplot':
                ax = data.boxplot(column=metric, by=group_by, ax=ax)
                ax.set_title(f'Distribution of {metric} by {group_by}')
            elif plot_type == 'violin':
                ax = sns.violinplot(x=group_by, y=metric, data=data, ax=ax)
                ax.set_title(f'Distribution of {metric} by {group_by}')
            else:  # histogram
                groups = data.groupby(group_by)
                for name, group in groups:
                    group[metric].plot.hist(alpha=0.5, label=name, ax=ax)
                ax.legend()
                ax.set_title(f'Histogram of {metric} by {group_by}')
        else:
            # No grouping
            if plot_type == 'boxplot':
                ax = self.summary_data[metric].plot.box(ax=ax)
                ax.set_title(f'Distribution of {metric}')
            elif plot_type == 'violin':
                ax = sns.violinplot(y=self.summary_data[metric], ax=ax)
                ax.set_title(f'Distribution of {metric}')
            else:  # histogram
                ax = self.summary_data[metric].plot.hist(ax=ax)
                ax.set_title(f'Histogram of {metric}')
        
        # Set labels
        ax.set_xlabel('Value' if plot_type == 'histogram' else '')
        ax.set_ylabel(metric)
        
        # Save figure if output directory is specified
        if self.output_dir:
            filename = f"{metric}_distribution_{plot_type}.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.visualization_config.get('dpi', 100))
            logging.info(f"Saved distribution plot to {filepath}")
        
        return fig
    
    def plot_correlation_matrix(self, 
                               metrics: Optional[List[str]] = None) -> Optional[Figure]:
        """Plot correlation matrix between metrics.
        
        Args:
            metrics: List of metric names to include in correlation matrix
            
        Returns:
            Matplotlib figure
        """
        if self.summary_data is None:
            self.generate_summary()
            
        if self.summary_data.empty:
            return None
        
        # Filter numeric columns if no specific metrics are provided
        if metrics is None:
            numeric_cols = self.summary_data.select_dtypes(include=[np.number]).columns
            metrics = list(numeric_cols)
        else:
            # Filter metrics that exist in the data
            metrics = [m for m in metrics if m in self.summary_data.columns]
            
        if not metrics or len(metrics) < 2:
            logging.warning("Not enough valid metrics found for correlation matrix")
            return None
        
        # Calculate correlation matrix
        corr_matrix = self.summary_data[metrics].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            annot=True,
            fmt=".2f",
            ax=ax
        )
        
        ax.set_title('Correlation Matrix of Metrics')
        
        # Save figure if output directory is specified
        if self.output_dir:
            filepath = os.path.join(self.output_dir, "correlation_matrix.png")
            fig.savefig(filepath, dpi=self.visualization_config.get('dpi', 100))
            logging.info(f"Saved correlation matrix to {filepath}")
        
        return fig
    
    def compare_time_series(self, 
                           feature_name: str, 
                           file_names: Optional[List[str]] = None,
                           normalize: bool = True) -> Optional[Figure]:
        """Compare time series features across multiple files.
        
        Args:
            feature_name: Name of the time series feature to compare
            file_names: List of file names to include (if None, use all files)
            normalize: Whether to normalize time series for comparison
            
        Returns:
            Matplotlib figure
        """
        if not self.results_data:
            logging.warning("No results data available for comparison")
            return None
        
        # Filter files if specified
        if file_names is None:
            file_names = list(self.results_data.keys())
        else:
            file_names = [f for f in file_names if f in self.results_data]
            
        if not file_names:
            logging.warning("No valid files found for comparison")
            return None
        
        # Extract time series data
        time_series_data = {}
        for file_name in file_names:
            data = self.results_data[file_name]
            
            # Check if feature exists in features
            if "features" in data and isinstance(data["features"], dict):
                features = data["features"]
                if feature_name in features and isinstance(features[feature_name], list):
                    time_series_data[file_name] = features[feature_name]
            
            # Also check for direct time series data
            if feature_name in data and isinstance(data[feature_name], list):
                time_series_data[file_name] = data[feature_name]
        
        if not time_series_data:
            logging.warning(f"No time series data found for feature {feature_name}")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot time series
        for file_name, values in time_series_data.items():
            # Normalize values if requested
            if normalize and values:
                min_val = min(values)
                max_val = max(values)
                if max_val > min_val:
                    values = [(v - min_val) / (max_val - min_val) for v in values]
            
            # Use optimized downsampling
            if len(values) > self.viz_optimizer.max_points:
                time_axis = np.linspace(0, 1, len(values))
                values = self.viz_optimizer.downsample_data(values)
                time_axis = np.linspace(0, 1, len(values))
            else:
                time_axis = np.linspace(0, 1, len(values))
            
            ax.plot(time_axis, values, label=file_name)
        
        # Set labels and title
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel(f'{"Normalized " if normalize else ""}Value')
        ax.set_title(f'Comparison of {feature_name} across files')
        ax.legend()
        
        # Save figure if output directory is specified
        if self.output_dir:
            filename = f"{feature_name}_comparison.png"
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=self.visualization_config.get('dpi', 100))
            logging.info(f"Saved time series comparison to {filepath}")
        
        return fig
    
    def generate_comparison_report(self, 
                                  output_format: str = 'json',
                                  include_visualizations: bool = True) -> Dict[str, Any]:
        """Generate a comprehensive comparison report.
        
        Args:
            output_format: Format for report output ('json', 'csv', or 'html')
            include_visualizations: Whether to include visualizations in the report
            
        Returns:
            Dictionary with report data
        """
        if not self.results_data:
            logging.warning("No results data available for report")
            return {}
        
        # Generate summary
        if self.summary_data is None:
            self.generate_summary()
            
        # Compare metrics
        if self.comparison_metrics is None:
            self.compare_metrics()
            
        # Find outliers
        outliers = self.find_outliers()
        
        # Generate report data
        report = {
            'summary': self.summary_data.to_dict() if self.summary_data is not None else {},
            'comparison': self.comparison_metrics.to_dict() if self.comparison_metrics is not None else {},
            'outliers': outliers,
            'file_count': len(self.results_data),
            'metrics_analyzed': list(self.summary_data.columns) if self.summary_data is not None else []
        }
        
        # Generate visualizations if requested
        if include_visualizations and self.output_dir:
            visualization_files = []
            
            # Generate correlation matrix
            corr_fig = self.plot_correlation_matrix()
            if corr_fig:
                visualization_files.append("correlation_matrix.png")
            
            # Generate distribution plots for key metrics
            key_metrics = []
            
            # Add BPM if available
            if 'bpm' in self.summary_data:
                key_metrics.append('bpm')
                bpm_fig = self.plot_metric_distribution('bpm')
                if bpm_fig:
                    visualization_files.append("bpm_distribution_boxplot.png")
            
            # Add key confidence if available
            if 'key_confidence' in self.summary_data:
                key_metrics.append('key_confidence')
                key_fig = self.plot_metric_distribution('key_confidence')
                if key_fig:
                    visualization_files.append("key_confidence_distribution_boxplot.png")
            
            # Add quality metrics if available
            quality_metrics = [col for col in self.summary_data.columns if col.startswith('quality_')]
            for metric in quality_metrics[:3]:  # Limit to first 3 quality metrics
                key_metrics.append(metric)
                qual_fig = self.plot_metric_distribution(metric)
                if qual_fig:
                    visualization_files.append(f"{metric}_distribution_boxplot.png")
            
            # Add performance metrics if available
            perf_metrics = ['analysis_time', 'memory_usage']
            for metric in perf_metrics:
                if metric in self.summary_data:
                    key_metrics.append(metric)
                    perf_fig = self.plot_metric_distribution(metric)
                    if perf_fig:
                        visualization_files.append(f"{metric}_distribution_boxplot.png")
            
            report['visualizations'] = visualization_files
            report['key_metrics'] = key_metrics
        
        # Save report to file
        if self.output_dir:
            if output_format == 'json':
                report_path = os.path.join(self.output_dir, "comparison_report.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
            elif output_format == 'csv':
                # Save summary to CSV
                summary_path = os.path.join(self.output_dir, "summary.csv")
                if self.summary_data is not None:
                    self.summary_data.to_csv(summary_path)
                
                # Save comparison metrics to CSV
                comparison_path = os.path.join(self.output_dir, "comparison_metrics.csv")
                if self.comparison_metrics is not None:
                    self.comparison_metrics.to_csv(comparison_path)
                
                # Save outliers to CSV
                outliers_path = os.path.join(self.output_dir, "outliers.csv")
                outliers_df = pd.DataFrame({
                    'metric': [k for k, v in outliers.items() for _ in v],
                    'file_name': [f for k, v in outliers.items() for f in v]
                })
                outliers_df.to_csv(outliers_path, index=False)
                
                report['csv_files'] = [summary_path, comparison_path, outliers_path]
            elif output_format == 'html':
                # Create HTML report
                try:
                    import jinja2
                    
                    # Load template
                    template_str = """
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Audio Analysis Comparison Report</title>
                        <style>
                            body { font-family: Arial, sans-serif; margin: 20px; }
                            h1, h2, h3 { color: #333; }
                            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                            th { background-color: #f2f2f2; }
                            tr:nth-child(even) { background-color: #f9f9f9; }
                            .visualization { margin: 20px 0; text-align: center; }
                            .visualization img { max-width: 100%; height: auto; }
                        </style>
                    </head>
                    <body>
                        <h1>Audio Analysis Comparison Report</h1>
                        
                        <h2>Summary</h2>
                        <p>Total files analyzed: {{ file_count }}</p>
                        
                        <h2>Key Metrics Comparison</h2>
                        {% for metric in key_metrics %}
                        <h3>{{ metric }}</h3>
                        <table>
                            <tr>
                                <th>Statistic</th>
                                <th>Value</th>
                            </tr>
                            {% for stat, value in comparison[metric].items() %}
                            <tr>
                                <td>{{ stat }}</td>
                                <td>{{ value }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                        {% endfor %}
                        
                        <h2>Outliers</h2>
                        {% if outliers %}
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Outlier Files</th>
                            </tr>
                            {% for metric, files in outliers.items() %}
                            <tr>
                                <td>{{ metric }}</td>
                                <td>{{ files|join(', ') }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                        {% else %}
                        <p>No outliers detected.</p>
                        {% endif %}
                        
                        {% if visualizations %}
                        <h2>Visualizations</h2>
                        {% for viz in visualizations %}
                        <div class="visualization">
                            <img src="{{ viz }}" alt="{{ viz }}">
                            <p>{{ viz }}</p>
                        </div>
                        {% endfor %}
                        {% endif %}
                    </body>
                    </html>
                    """
                    
                    # Render template
                    template = jinja2.Template(template_str)
                    html_content = template.render(
                        file_count=report['file_count'],
                        key_metrics=report.get('key_metrics', []),
                        comparison={k: v for k, v in report.get('comparison', {}).items() if k in report.get('key_metrics', [])},
                        outliers=report['outliers'],
                        visualizations=report.get('visualizations', [])
                    )
                    
                    # Save HTML file
                    html_path = os.path.join(self.output_dir, "comparison_report.html")
                    with open(html_path, 'w') as f:
                        f.write(html_content)
                    
                    report['html_file'] = html_path
                    
                except ImportError:
                    logging.warning("jinja2 package not found, skipping HTML report generation")
        
        return report 