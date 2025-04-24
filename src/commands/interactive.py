"""
Interactive mode for the Heihachi audio analysis framework.
Provides a command-line interface for real-time audio analysis and visualization.
"""

import os
import sys
import argparse
import logging
import time
import cmd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..core.pipeline import Pipeline
from ..utils.visualization_optimization import VisualizationOptimizer
from ..utils.comparison import ResultComparison


class HeihachiBatchShell(cmd.Cmd):
    """Interactive shell for the Heihachi audio analysis framework."""
    
    intro = """
    Heihachi Interactive Shell
    Type 'help' or '?' to list commands.
    Type 'exit' or 'quit' to exit.
    """
    prompt = "heihachi> "
    
    def __init__(
        self, 
        config_path: Optional[str] = None,
        output_dir: str = "./output",
        debug: bool = False
    ):
        """Initialize the interactive shell.
        
        Args:
            config_path: Path to the configuration file
            output_dir: Directory to save output files
            debug: Enable debug mode
        """
        super().__init__()
        self.config_path = config_path
        self.output_dir = output_dir
        self.debug = debug
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipeline
        self.pipeline = Pipeline(config_path=config_path)
        
        # Initialize visualization optimizer
        self.visualizer = VisualizationOptimizer(
            output_dir=os.path.join(output_dir, "visualizations"),
            dpi=100,
            max_points=10000
        )
        
        # Initialize result comparison
        self.comparer = ResultComparison(
            output_dir=os.path.join(output_dir, "comparisons")
        )
        
        # Keep track of processed files
        self.processed_files = {}
        self.current_file = None
        self.results = {}
        
        logging.info("Interactive shell initialized")
    
    def do_analyze(self, arg):
        """Analyze an audio file or directory.
        
        Usage: analyze <file_path> [--no-viz]
        """
        args = arg.split()
        if not args:
            print("Error: Please provide a file path")
            return
        
        file_path = args[0]
        generate_viz = "--no-viz" not in args
        
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File or directory '{file_path}' not found")
            return
        
        if path.is_file():
            self._analyze_file(str(path), generate_viz)
        elif path.is_dir():
            print(f"Analyzing all audio files in directory: {file_path}")
            for audio_file in path.glob("*.wav"):
                self._analyze_file(str(audio_file), generate_viz)
    
    def _analyze_file(self, file_path: str, generate_viz: bool = True):
        """Analyze a single audio file."""
        try:
            print(f"Analyzing file: {file_path}")
            start_time = time.time()
            
            # Process the file
            result = self.pipeline.process_file(file_path)
            
            # Store the result
            self.results[file_path] = result
            self.processed_files[file_path] = {
                "file_name": Path(file_path).name,
                "file_path": file_path,
                "processed_at": time.time(),
                "processing_time": time.time() - start_time
            }
            self.current_file = file_path
            
            # Save the result
            output_file = os.path.join(
                self.output_dir, 
                f"{Path(file_path).stem}_result.json"
            )
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Generate visualizations
            if generate_viz:
                self.do_visualize(file_path)
            
            print(f"Analysis completed in {time.time() - start_time:.2f} seconds")
            print(f"Results saved to {output_file}")
            
        except Exception as e:
            print(f"Error analyzing file: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def do_visualize(self, arg):
        """Generate visualizations for an analyzed file.
        
        Usage: visualize [file_path]
        If no file path is provided, visualizes the most recently analyzed file.
        """
        file_path = arg.strip() if arg.strip() else self.current_file
        
        if not file_path:
            print("Error: No file specified and no recent file available")
            return
        
        if file_path not in self.results:
            print(f"Error: File '{file_path}' has not been analyzed yet")
            return
        
        try:
            print(f"Generating visualizations for: {file_path}")
            result = self.results[file_path]
            
            # Generate visualizations
            self.visualizer.plot_waveform(
                result.get("audio_data", []), 
                os.path.basename(file_path),
                save=True
            )
            
            if "features" in result:
                features = result["features"]
                for feature_name, feature_data in features.items():
                    if isinstance(feature_data, list) and len(feature_data) > 0:
                        self.visualizer.plot_feature(
                            feature_data, 
                            feature_name, 
                            os.path.basename(file_path),
                            save=True
                        )
            
            if "spectrogram" in result:
                self.visualizer.plot_spectrogram(
                    result["spectrogram"],
                    os.path.basename(file_path),
                    save=True
                )
            
            print(f"Visualizations saved to {self.visualizer.output_dir}")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def do_compare(self, arg):
        """Compare analyzed files.
        
        Usage: compare [file1] [file2] ... [--metrics=metric1,metric2] [--format=json|csv|html]
        If no files are specified, compares all analyzed files.
        """
        args = arg.split()
        
        # Parse metrics
        metrics = None
        for arg in args:
            if arg.startswith("--metrics="):
                metrics_str = arg.split("=")[1]
                metrics = metrics_str.split(",")
                args.remove(arg)
                break
        
        # Parse format
        output_format = "json"
        for arg in args:
            if arg.startswith("--format="):
                output_format = arg.split("=")[1]
                args.remove(arg)
                break
        
        # Determine files to compare
        files_to_compare = args if args else list(self.results.keys())
        
        if len(files_to_compare) < 2:
            print("Error: Need at least 2 files to compare")
            return
        
        # Check if all files have been analyzed
        for file_path in files_to_compare:
            if file_path not in self.results:
                print(f"Error: File '{file_path}' has not been analyzed yet")
                return
        
        try:
            print(f"Comparing {len(files_to_compare)} files")
            
            # Load results into comparer
            for file_path in files_to_compare:
                self.comparer.add_result(file_path, self.results[file_path])
            
            # Generate summary
            summary = self.comparer.generate_summary()
            print("\nSummary Statistics:")
            print(summary)
            
            # Compare metrics
            comparison = self.comparer.compare_metrics(metrics=metrics)
            print("\nMetric Comparison:")
            print(comparison)
            
            # Find outliers
            outliers = self.comparer.find_outliers(metrics=metrics)
            if outliers:
                print("\nOutliers:")
                for metric, files in outliers.items():
                    print(f"  {metric}: {len(files)} outliers")
            
            # Generate report
            print(f"\nGenerating comparison report in {output_format} format")
            report = self.comparer.generate_comparison_report(
                output_format=output_format,
                include_visualizations=True
            )
            
            # Print report location
            if output_format == 'json':
                print(f"Report saved to {os.path.join(self.comparer.output_dir, 'comparison_report.json')}")
            elif output_format == 'csv':
                for csv_file in report.get('csv_files', []):
                    print(f"Data saved to {csv_file}")
            elif output_format == 'html':
                if 'html_file' in report:
                    print(f"HTML report saved to {report['html_file']}")
            
        except Exception as e:
            print(f"Error comparing files: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def do_list(self, arg):
        """List processed files.
        
        Usage: list
        """
        if not self.processed_files:
            print("No files have been processed yet")
            return
        
        print("\nProcessed Files:")
        print("-" * 80)
        print(f"{'File Name':<30} {'Processing Time (s)':<20} {'Processed At':<30}")
        print("-" * 80)
        
        for info in self.processed_files.values():
            file_name = info["file_name"]
            processing_time = f"{info['processing_time']:.2f}"
            processed_at = time.strftime(
                "%Y-%m-%d %H:%M:%S", 
                time.localtime(info["processed_at"])
            )
            print(f"{file_name:<30} {processing_time:<20} {processed_at:<30}")
    
    def do_info(self, arg):
        """Show information about a processed file.
        
        Usage: info [file_path]
        If no file path is provided, shows information about the most recently analyzed file.
        """
        file_path = arg.strip() if arg.strip() else self.current_file
        
        if not file_path:
            print("Error: No file specified and no recent file available")
            return
        
        if file_path not in self.results:
            print(f"Error: File '{file_path}' has not been analyzed yet")
            return
        
        result = self.results[file_path]
        info = self.processed_files[file_path]
        
        print("\nFile Information:")
        print(f"File: {file_path}")
        print(f"Processed at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(info['processed_at']))}")
        print(f"Processing time: {info['processing_time']:.2f} seconds")
        
        print("\nAnalysis Results:")
        # Print basic audio info
        if "sample_rate" in result:
            print(f"Sample rate: {result['sample_rate']} Hz")
        if "duration" in result:
            print(f"Duration: {result['duration']:.2f} seconds")
        if "channels" in result:
            print(f"Channels: {result['channels']}")
        
        # Print key analysis results
        if "bpm" in result:
            print(f"BPM: {result['bpm']}")
        if "key" in result:
            print(f"Key: {result['key']}")
        if "scale" in result:
            print(f"Scale: {result['scale']}")
        
        # Print feature summary
        if "features" in result:
            print("\nFeatures:")
            for feature_name, feature_data in result["features"].items():
                if isinstance(feature_data, list):
                    if len(feature_data) > 0:
                        print(f"  {feature_name}: {len(feature_data)} values")
                else:
                    print(f"  {feature_name}: {feature_data}")
    
    def do_exit(self, arg):
        """Exit the interactive shell."""
        print("Exiting Heihachi Interactive Shell")
        return True
    
    def do_quit(self, arg):
        """Exit the interactive shell."""
        return self.do_exit(arg)
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass


def setup_parser(subparsers) -> None:
    """Set up the command-line parser for the interactive command.
    
    Args:
        subparsers: Subparsers object from the main parser
    """
    parser = subparsers.add_parser(
        'interactive',
        help='Start interactive mode',
        description='Launch an interactive shell for real-time audio analysis'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Directory to save output files (default: ./output)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.set_defaults(func=run_interactive)


def run_interactive(args: argparse.Namespace) -> None:
    """Run the interactive command with the given arguments.
    
    Args:
        args: Command-line arguments
    """
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start interactive shell
    shell = HeihachiBatchShell(
        config_path=args.config,
        output_dir=args.output_dir,
        debug=args.debug
    )
    
    try:
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting Heihachi Interactive Shell")
        sys.exit(0)


if __name__ == "__main__":
    # For testing the command directly
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    setup_parser(subparsers)
    
    test_args = parser.parse_args([
        'interactive',
        '--output-dir', './interactive_output',
        '--debug'
    ])
    
    if hasattr(test_args, 'func'):
        test_args.func(test_args)
    else:
        parser.print_help() 