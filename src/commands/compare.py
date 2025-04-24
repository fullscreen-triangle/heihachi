"""
Command-line module for comparing analysis results between multiple audio files.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..utils.comparison import ResultComparison


def setup_parser(subparsers) -> None:
    """Set up the command-line parser for the compare command.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    parser = subparsers.add_parser(
        'compare',
        help='Compare analysis results between multiple audio files',
        description='Generate comparison reports and visualizations from audio analysis results'
    )
    
    # Input sources
    input_group = parser.add_argument_group('Input options')
    input_group.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Directory containing JSON result files to compare'
    )
    input_group.add_argument(
        '--files', '-f',
        type=str,
        nargs='+',
        help='Specific JSON result files to compare'
    )
    input_group.add_argument(
        '--directories', '-d',
        type=str,
        nargs='+',
        help='Multiple directories containing JSON result files to compare'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output options')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./comparison_results',
        help='Directory to save comparison results (default: ./comparison_results)'
    )
    output_group.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'html'],
        default='json',
        help='Output format for comparison report (default: json)'
    )
    output_group.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Disable generation of visualizations'
    )
    
    # Visualization options
    viz_group = parser.add_argument_group('Visualization options')
    viz_group.add_argument(
        '--dpi',
        type=int,
        default=100,
        help='DPI for output visualizations (default: 100)'
    )
    viz_group.add_argument(
        '--max-points',
        type=int,
        default=10000,
        help='Maximum number of data points for time series visualizations (default: 10000)'
    )
    
    # Analysis options
    analysis_group = parser.add_argument_group('Analysis options')
    analysis_group.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='Specific metrics to compare (default: all numeric metrics)'
    )
    analysis_group.add_argument(
        '--group-by',
        type=str,
        help='Column to group results by (e.g., "key", "bpm", etc.)'
    )
    analysis_group.add_argument(
        '--outlier-threshold',
        type=float,
        default=2.0,
        help='Z-score threshold for identifying outliers (default: 2.0)'
    )
    analysis_group.add_argument(
        '--cluster',
        action='store_true',
        help='Perform clustering analysis on results'
    )
    analysis_group.add_argument(
        '--n-clusters',
        type=int,
        default=3,
        help='Number of clusters for clustering analysis (default: 3)'
    )
    
    parser.set_defaults(func=run_compare)


def run_compare(args: argparse.Namespace) -> None:
    """Run the compare command with the given arguments.
    
    Args:
        args: Command-line arguments
    """
    # Validate input sources
    if not any([args.input_dir, args.files, args.directories]):
        logging.error("No input source specified. Please provide --input-dir, --files, or --directories")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize comparison utility
    visualization_config = {
        'dpi': args.dpi,
        'max_points': args.max_points
    }
    
    comparer = ResultComparison(
        output_dir=args.output_dir,
        visualization_config=visualization_config
    )
    
    # Load results
    if args.input_dir:
        logging.info(f"Loading results from directory: {args.input_dir}")
        comparer.load_results(args.input_dir)
        
    if args.files:
        logging.info(f"Loading {len(args.files)} specific result files")
        comparer.load_specific_files(args.files)
        
    if args.directories:
        logging.info(f"Loading results from {len(args.directories)} directories")
        comparer.load_multiple_directories(args.directories)
    
    # Generate summary
    logging.info("Generating summary statistics")
    summary_df = comparer.generate_summary()
    if summary_df.empty:
        logging.error("No valid results data found")
        return
    
    logging.info(f"Loaded {len(summary_df)} files with {len(summary_df.columns)} metrics")
    
    # Compare metrics
    logging.info("Comparing metrics across files")
    comparison_df = comparer.compare_metrics(metrics=args.metrics, group_by=args.group_by)
    
    # Find outliers
    logging.info(f"Identifying outliers (threshold: {args.outlier_threshold})")
    outliers = comparer.find_outliers(metrics=args.metrics, threshold=args.outlier_threshold)
    
    if outliers:
        logging.info(f"Found outliers in {len(outliers)} metrics")
        for metric, files in outliers.items():
            logging.info(f"  {metric}: {len(files)} outliers")
    else:
        logging.info("No outliers found")
    
    # Perform clustering if requested
    if args.cluster and args.metrics:
        logging.info(f"Performing clustering analysis with {args.n_clusters} clusters")
        clusters = comparer.identify_clusters(metrics=args.metrics, n_clusters=args.n_clusters)
        
        if clusters:
            logging.info(f"Identified {len(clusters)} clusters")
            for cluster_id, files in clusters.items():
                logging.info(f"  {cluster_id}: {len(files)} files")
        else:
            logging.info("Clustering analysis failed or produced no results")
    
    # Generate visualizations
    if not args.no_visualizations:
        logging.info("Generating visualizations")
        
        # Correlation matrix
        comparer.plot_correlation_matrix()
        
        # Distribution plots for key metrics
        if args.metrics:
            for metric in args.metrics:
                comparer.plot_metric_distribution(metric)
        else:
            # Auto-select some interesting metrics
            numeric_cols = summary_df.select_dtypes(include=['number']).columns
            for metric in list(numeric_cols)[:5]:  # First 5 numeric metrics
                comparer.plot_metric_distribution(metric)
        
        # Time series comparisons
        # Look for time series data in the first file
        for file_name, data in comparer.results_data.items():
            if "features" in data and isinstance(data["features"], dict):
                for feature_name, value in data["features"].items():
                    if isinstance(value, list) and len(value) > 10:
                        logging.info(f"Comparing time series for feature: {feature_name}")
                        comparer.compare_time_series(feature_name, normalize=True)
                        break
            break
    
    # Generate report
    logging.info(f"Generating comparison report in {args.format} format")
    report = comparer.generate_comparison_report(
        output_format=args.format,
        include_visualizations=not args.no_visualizations
    )
    
    # Summary of report
    if args.format == 'json':
        logging.info(f"Report saved to {os.path.join(args.output_dir, 'comparison_report.json')}")
    elif args.format == 'csv':
        for csv_file in report.get('csv_files', []):
            logging.info(f"Data saved to {csv_file}")
    elif args.format == 'html':
        if 'html_file' in report:
            logging.info(f"HTML report saved to {report['html_file']}")
    
    logging.info("Comparison completed successfully")


if __name__ == "__main__":
    # For testing the command directly
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    setup_parser(subparsers)
    
    test_args = parser.parse_args([
        'compare',
        '--input-dir', './results',
        '--output-dir', './comparison_results',
        '--format', 'html'
    ])
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if hasattr(test_args, 'func'):
        test_args.func(test_args)
    else:
        parser.print_help() 