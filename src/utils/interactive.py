#!/usr/bin/env python3
"""
Interactive mode for exploring and comparing audio analysis results.

This module provides a command-line interface for interactively exploring
audio analysis results, comparing multiple files, and visualizing key metrics.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from src.utils.logging_utils import get_logger
from src.utils.export import load_results
from src.visualization.plots import plot_feature_comparison, plot_waveform, plot_spectrogram

logger = get_logger(__name__)

# Define command styles
style = Style.from_dict({
    'prompt': '#00aa00 bold',
    'command': '#ffffff',
})

class InteractiveExplorer:
    """Interactive explorer for audio analysis results."""
    
    def __init__(self, results_dir: Union[str, Path]):
        """Initialize the interactive explorer.
        
        Args:
            results_dir: Directory containing analysis results
        """
        self.results_dir = Path(results_dir) if isinstance(results_dir, str) else results_dir
        self.results: Dict[str, Any] = {}
        self.current_file: Optional[str] = None
        self.comparison_files: List[str] = []
        self.running = True
        
        # Load all results
        self._load_results()
        
        # Setup prompt session with command completion
        self.commands = [
            'help', 'list', 'show', 'compare', 'plot', 'exit', 'quit',
            'feature', 'summary', 'waveform', 'spectrogram', 'export',
            'clear', 'load', 'select', 'add', 'remove'
        ]
        self.command_completer = WordCompleter(self.commands)
        self.session = PromptSession(
            completer=self.command_completer,
            history=InMemoryHistory(),
            style=style
        )
    
    def _load_results(self) -> None:
        """Load all result files from the results directory."""
        if not self.results_dir.exists():
            logger.error(f"Results directory not found: {self.results_dir}")
            return
        
        # Look for JSON, CSV, and YAML files
        result_files = list(self.results_dir.glob("*.json"))
        result_files.extend(self.results_dir.glob("*.csv"))
        result_files.extend(self.results_dir.glob("*.yaml"))
        
        for file_path in result_files:
            try:
                # Skip batch summary files
                if file_path.name.startswith("batch_summary"):
                    continue
                
                # Load the results
                data = load_results(file_path)
                if data:
                    self.results[file_path.name] = data
            except Exception as e:
                logger.error(f"Error loading result file {file_path}: {e}")
        
        logger.info(f"Loaded {len(self.results)} result files from {self.results_dir}")
    
    def start(self) -> None:
        """Start the interactive explorer."""
        self._print_welcome()
        
        while self.running:
            try:
                user_input = self.session.prompt("buru-sukirin> ", style=style).strip()
                if user_input:
                    self._process_command(user_input)
            except KeyboardInterrupt:
                print("\nUse 'exit' or 'quit' to close the explorer")
            except Exception as e:
                logger.error(f"Error processing command: {e}")
                print(f"Error: {e}")
    
    def _print_welcome(self) -> None:
        """Print welcome message and basic instructions."""
        print("\n=== Buru Sukirin Interactive Explorer ===")
        print(f"Loaded {len(self.results)} result files from {self.results_dir}")
        print("\nType 'help' for a list of commands\n")
    
    def _process_command(self, command: str) -> None:
        """Process a user command.
        
        Args:
            command: User command string
        """
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd in ('exit', 'quit'):
            self.running = False
        elif cmd == 'help':
            self._show_help()
        elif cmd == 'list':
            self._list_results()
        elif cmd == 'show':
            self._show_result(args)
        elif cmd == 'select':
            self._select_file(args)
        elif cmd == 'compare':
            self._compare_results(args)
        elif cmd == 'add':
            self._add_comparison_file(args)
        elif cmd == 'remove':
            self._remove_comparison_file(args)
        elif cmd == 'clear':
            self._clear_comparison()
        elif cmd == 'plot':
            self._plot_feature(args)
        elif cmd == 'feature':
            self._show_feature(args)
        elif cmd == 'summary':
            self._show_summary(args)
        elif cmd == 'waveform':
            self._plot_waveform(args)
        elif cmd == 'spectrogram':
            self._plot_spectrogram(args)
        elif cmd == 'export':
            self._export_comparison(args)
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for a list of commands")
    
    def _show_help(self) -> None:
        """Show help information."""
        commands = [
            ["help", "Show this help message"],
            ["list", "List available result files"],
            ["show [file]", "Show details of a result file"],
            ["select <file>", "Select a file as the current file"],
            ["add <file>", "Add a file to the comparison list"],
            ["remove <file>", "Remove a file from the comparison list"],
            ["clear", "Clear the comparison list"],
            ["compare", "Compare selected files"],
            ["feature <name> [files...]", "Show values for a specific feature"],
            ["summary [files...]", "Show summary statistics for files"],
            ["plot <feature> [files...]", "Plot a feature from selected files"],
            ["waveform [file]", "Plot waveform of a file"],
            ["spectrogram [file]", "Plot spectrogram of a file"],
            ["export <format>", "Export comparison to CSV/JSON"],
            ["exit/quit", "Exit the explorer"],
        ]
        
        print("\nAvailable commands:")
        print(tabulate(commands, headers=["Command", "Description"]))
        print()
    
    def _list_results(self) -> None:
        """List all available result files."""
        if not self.results:
            print("No result files loaded.")
            return
        
        files = []
        for idx, (filename, data) in enumerate(self.results.items(), 1):
            # Get file type and features if available
            file_type = data.get("file_type", "unknown")
            features = list(data.get("features", {}).keys())
            feature_str = ", ".join(features[:3])
            if len(features) > 3:
                feature_str += f"... ({len(features)} total)"
            
            # Check if this file is in comparison list
            status = ""
            if filename == self.current_file:
                status = "* "
            elif filename in self.comparison_files:
                status = "+ "
            
            files.append([idx, status + filename, file_type, feature_str])
        
        print("\nAvailable result files:")
        print(tabulate(files, headers=["#", "Filename", "Type", "Features"]))
        print("\n* = current file, + = in comparison list\n")
    
    def _show_result(self, args: List[str]) -> None:
        """Show details of a specific result file.
        
        Args:
            args: Command arguments, should contain the file name or index
        """
        # If no argument is provided, use the current file
        if not args and self.current_file:
            filename = self.current_file
        elif not args:
            print("No file specified. Use 'show <filename>' or 'select <filename>' first.")
            return
        else:
            # Try to get the file by index or name
            try:
                if args[0].isdigit():
                    idx = int(args[0]) - 1
                    if idx < 0 or idx >= len(self.results):
                        print(f"Invalid index: {args[0]}")
                        return
                    filename = list(self.results.keys())[idx]
                else:
                    filename = args[0]
                    if filename not in self.results:
                        print(f"File not found: {filename}")
                        return
            except Exception as e:
                print(f"Error: {e}")
                return
        
        # Show the result data
        data = self.results.get(filename)
        if not data:
            print(f"File not found: {filename}")
            return
        
        print(f"\nFile: {filename}")
        print("-" * (len(filename) + 7))
        
        # Basic file info
        file_info = [
            ["File path", data.get("file_path", "Unknown")],
            ["Sample rate", data.get("sample_rate", "Unknown")],
            ["Duration", f"{data.get('duration', 0):.2f} seconds"],
            ["Channels", data.get("channels", "Unknown")],
        ]
        print(tabulate(file_info, tablefmt="plain"))
        
        # Show available features
        features = data.get("features", {})
        if features:
            print("\nAvailable features:")
            feature_list = []
            for name, value in features.items():
                feature_type = "Time series" if isinstance(value, list) else "Scalar"
                feature_list.append([name, feature_type])
            
            print(tabulate(feature_list, headers=["Feature", "Type"]))
            print("\nUse 'feature <name>' to see values of a specific feature")
        
        print()
    
    def _select_file(self, args: List[str]) -> None:
        """Select a file as the current file.
        
        Args:
            args: Command arguments, should contain the file name or index
        """
        if not args:
            print("Please specify a file to select.")
            return
        
        # Try to get the file by index or name
        try:
            if args[0].isdigit():
                idx = int(args[0]) - 1
                if idx < 0 or idx >= len(self.results):
                    print(f"Invalid index: {args[0]}")
                    return
                filename = list(self.results.keys())[idx]
            else:
                filename = args[0]
                if filename not in self.results:
                    print(f"File not found: {filename}")
                    return
        except Exception as e:
            print(f"Error: {e}")
            return
        
        self.current_file = filename
        print(f"Selected file: {filename}")
        
        # Add to comparison list if not already there
        if filename not in self.comparison_files:
            self.comparison_files.append(filename)
    
    def _add_comparison_file(self, args: List[str]) -> None:
        """Add a file to the comparison list.
        
        Args:
            args: Command arguments, should contain the file name or index
        """
        if not args:
            print("Please specify a file to add.")
            return
        
        # Try to get the file by index or name
        try:
            if args[0].isdigit():
                idx = int(args[0]) - 1
                if idx < 0 or idx >= len(self.results):
                    print(f"Invalid index: {args[0]}")
                    return
                filename = list(self.results.keys())[idx]
            else:
                filename = args[0]
                if filename not in self.results:
                    print(f"File not found: {filename}")
                    return
        except Exception as e:
            print(f"Error: {e}")
            return
        
        # Add to comparison list if not already there
        if filename not in self.comparison_files:
            self.comparison_files.append(filename)
            print(f"Added {filename} to comparison list")
        else:
            print(f"{filename} is already in the comparison list")
    
    def _remove_comparison_file(self, args: List[str]) -> None:
        """Remove a file from the comparison list.
        
        Args:
            args: Command arguments, should contain the file name or index
        """
        if not args:
            print("Please specify a file to remove.")
            return
        
        # Try to get the file by index or name
        try:
            if args[0].isdigit():
                idx = int(args[0]) - 1
                if idx < 0 or idx >= len(self.results):
                    print(f"Invalid index: {args[0]}")
                    return
                filename = list(self.results.keys())[idx]
            else:
                filename = args[0]
        except Exception as e:
            print(f"Error: {e}")
            return
        
        # Remove from comparison list
        if filename in self.comparison_files:
            self.comparison_files.remove(filename)
            print(f"Removed {filename} from comparison list")
            
            # If it was the current file, reset current file
            if filename == self.current_file:
                self.current_file = self.comparison_files[0] if self.comparison_files else None
        else:
            print(f"{filename} is not in the comparison list")
    
    def _clear_comparison(self) -> None:
        """Clear the comparison list."""
        self.comparison_files = []
        self.current_file = None
        print("Cleared comparison list")
    
    def _compare_results(self, args: List[str]) -> None:
        """Compare selected files.
        
        Args:
            args: Command arguments, optional file names to compare
        """
        # If specific files are provided, use those
        files_to_compare = []
        if args:
            for arg in args:
                if arg in self.results:
                    files_to_compare.append(arg)
                elif arg.isdigit() and 0 <= int(arg) - 1 < len(self.results):
                    idx = int(arg) - 1
                    files_to_compare.append(list(self.results.keys())[idx])
        else:
            # Otherwise use the comparison list
            files_to_compare = self.comparison_files
        
        if not files_to_compare:
            print("No files selected for comparison. Use 'add <file>' to add files.")
            return
        
        # Get common features across files
        common_features = set()
        for i, filename in enumerate(files_to_compare):
            data = self.results.get(filename, {})
            features = data.get("features", {}).keys()
            
            if i == 0:
                common_features = set(features)
            else:
                common_features = common_features.intersection(features)
        
        if not common_features:
            print("No common features found across the selected files.")
            return
        
        # Create comparison table for scalar features
        scalar_features = []
        for feature in sorted(common_features):
            # Check if the feature is a scalar (not a list/array) in all files
            is_scalar = True
            for filename in files_to_compare:
                value = self.results[filename].get("features", {}).get(feature)
                if isinstance(value, (list, np.ndarray)) and len(value) > 1:
                    is_scalar = False
                    break
            
            if is_scalar:
                scalar_features.append(feature)
        
        if scalar_features:
            # Create comparison table
            headers = ["Feature"] + [f.split("_")[0] for f in files_to_compare]
            rows = []
            
            for feature in scalar_features:
                row = [feature]
                for filename in files_to_compare:
                    value = self.results[filename].get("features", {}).get(feature)
                    if isinstance(value, (int, float)):
                        row.append(f"{value:.4f}")
                    else:
                        row.append(str(value))
                rows.append(row)
            
            print("\nFeature Comparison:")
            print(tabulate(rows, headers=headers))
        
        # List time series features that can be plotted
        time_series_features = list(common_features - set(scalar_features))
        if time_series_features:
            print("\nPlottable time series features:")
            for feature in sorted(time_series_features):
                print(f" - {feature}")
            print("\nUse 'plot <feature>' to visualize these features")
        
        print()
    
    def _show_feature(self, args: List[str]) -> None:
        """Show values for a specific feature.
        
        Args:
            args: Command arguments [feature_name, file1, file2, ...]
        """
        if not args:
            print("Please specify a feature name.")
            return
        
        feature_name = args[0]
        
        # Determine which files to use
        files_to_use = args[1:] if len(args) > 1 else self.comparison_files
        if not files_to_use and self.current_file:
            files_to_use = [self.current_file]
            
        if not files_to_use:
            print("No files selected. Use 'select <file>' or 'add <file>' first.")
            return
        
        # Show feature values for each file
        for filename in files_to_use:
            if filename not in self.results:
                print(f"File not found: {filename}")
                continue
                
            data = self.results[filename]
            features = data.get("features", {})
            
            if feature_name not in features:
                print(f"Feature '{feature_name}' not found in {filename}")
                continue
            
            value = features[feature_name]
            print(f"\n{filename} - {feature_name}:")
            
            if isinstance(value, (list, np.ndarray)) and len(value) > 10:
                # For time series, show statistics and first/last values
                arr = np.array(value)
                stats = [
                    ["Count", len(arr)],
                    ["Mean", np.mean(arr)],
                    ["Std Dev", np.std(arr)],
                    ["Min", np.min(arr)],
                    ["25%", np.percentile(arr, 25)],
                    ["Median", np.median(arr)],
                    ["75%", np.percentile(arr, 75)],
                    ["Max", np.max(arr)]
                ]
                print(tabulate(stats, tablefmt="plain"))
                
                print("\nFirst 5 values:")
                print(arr[:5])
                print("\nLast 5 values:")
                print(arr[-5:])
            else:
                # For scalar or short lists, show the full value
                print(value)
        
        print()
    
    def _show_summary(self, args: List[str]) -> None:
        """Show summary statistics for files.
        
        Args:
            args: Command arguments, optional file names
        """
        # Determine which files to use
        files_to_use = args if args else self.comparison_files
        if not files_to_use and self.current_file:
            files_to_use = [self.current_file]
            
        if not files_to_use:
            print("No files selected. Use 'select <file>' or 'add <file>' first.")
            return
        
        # Prepare summary data
        summary_data = []
        
        for filename in files_to_use:
            if filename not in self.results:
                print(f"File not found: {filename}")
                continue
                
            data = self.results[filename]
            
            # Extract basic information
            file_path = data.get("file_path", "Unknown")
            sample_rate = data.get("sample_rate", "Unknown")
            duration = data.get("duration", 0)
            channels = data.get("channels", "Unknown")
            num_features = len(data.get("features", {}))
            
            summary_data.append([
                filename, 
                os.path.basename(file_path), 
                f"{duration:.2f}s",
                sample_rate, 
                channels,
                num_features
            ])
        
        print("\nSummary for selected files:")
        headers = ["Result File", "Audio File", "Duration", "Sample Rate", "Channels", "Features"]
        print(tabulate(summary_data, headers=headers))
        print()
    
    def _plot_feature(self, args: List[str]) -> None:
        """Plot a feature from selected files.
        
        Args:
            args: Command arguments [feature_name, file1, file2, ...]
        """
        if not args:
            print("Please specify a feature name to plot.")
            return
        
        feature_name = args[0]
        
        # Determine which files to use
        files_to_use = args[1:] if len(args) > 1 else self.comparison_files
        if not files_to_use and self.current_file:
            files_to_use = [self.current_file]
            
        if not files_to_use:
            print("No files selected. Use 'select <file>' or 'add <file>' first.")
            return
        
        # Gather data for plotting
        plot_data = {}
        for filename in files_to_use:
            if filename not in self.results:
                print(f"File not found: {filename}")
                continue
                
            data = self.results[filename]
            features = data.get("features", {})
            
            if feature_name not in features:
                print(f"Feature '{feature_name}' not found in {filename}")
                continue
            
            value = features[feature_name]
            if not isinstance(value, (list, np.ndarray)) or len(value) <= 1:
                print(f"Feature '{feature_name}' in {filename} is not a time series.")
                continue
                
            # Add to plot data
            plot_data[filename] = {
                "values": value,
                "sample_rate": data.get("sample_rate", 44100),
                "duration": data.get("duration", len(value) / 44100)
            }
        
        if not plot_data:
            print("No plottable data found for the selected feature and files.")
            return
        
        # Plot the data
        try:
            plot_feature_comparison(feature_name, plot_data)
            plt.show()
        except Exception as e:
            print(f"Error plotting feature: {e}")
    
    def _plot_waveform(self, args: List[str]) -> None:
        """Plot waveform of a file.
        
        Args:
            args: Command arguments, optional file name
        """
        # Determine which file to use
        if args:
            filename = args[0]
            if filename not in self.results:
                print(f"File not found: {filename}")
                return
        elif self.current_file:
            filename = self.current_file
        else:
            print("No file selected. Use 'select <file>' first.")
            return
        
        data = self.results[filename]
        
        # Check if waveform data is available
        if "waveform" not in data.get("features", {}):
            print(f"Waveform data not available for {filename}")
            return
        
        # Plot the waveform
        try:
            waveform = data["features"]["waveform"]
            sample_rate = data.get("sample_rate", 44100)
            plot_waveform(waveform, sample_rate, title=f"Waveform - {os.path.basename(data.get('file_path', filename))}")
            plt.show()
        except Exception as e:
            print(f"Error plotting waveform: {e}")
    
    def _plot_spectrogram(self, args: List[str]) -> None:
        """Plot spectrogram of a file.
        
        Args:
            args: Command arguments, optional file name
        """
        # Determine which file to use
        if args:
            filename = args[0]
            if filename not in self.results:
                print(f"File not found: {filename}")
                return
        elif self.current_file:
            filename = self.current_file
        else:
            print("No file selected. Use 'select <file>' first.")
            return
        
        data = self.results[filename]
        
        # Check if spectrogram data is available
        if "spectrogram" not in data.get("features", {}):
            print(f"Spectrogram data not available for {filename}")
            return
        
        # Plot the spectrogram
        try:
            spectrogram = data["features"]["spectrogram"]
            sample_rate = data.get("sample_rate", 44100)
            plot_spectrogram(spectrogram, sample_rate, title=f"Spectrogram - {os.path.basename(data.get('file_path', filename))}")
            plt.show()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")
    
    def _export_comparison(self, args: List[str]) -> None:
        """Export comparison data to a file.
        
        Args:
            args: Command arguments [format, output_file]
        """
        if not args:
            print("Please specify an export format (csv or json).")
            return
        
        export_format = args[0].lower()
        if export_format not in ["csv", "json"]:
            print("Invalid export format. Use 'csv' or 'json'.")
            return
        
        # Get output file name
        output_file = args[1] if len(args) > 1 else f"comparison_export.{export_format}"
        
        # Check if we have files to compare
        if not self.comparison_files:
            print("No files selected for comparison. Use 'add <file>' to add files.")
            return
        
        # Get common scalar features across files
        common_features = set()
        for i, filename in enumerate(self.comparison_files):
            data = self.results.get(filename, {})
            features = data.get("features", {})
            
            # Only include scalar features
            scalar_features = {
                name for name, value in features.items()
                if not isinstance(value, (list, np.ndarray)) or len(value) <= 1
            }
            
            if i == 0:
                common_features = scalar_features
            else:
                common_features = common_features.intersection(scalar_features)
        
        if not common_features:
            print("No common scalar features found across the selected files.")
            return
        
        # Create comparison data
        comparison_data = {}
        
        for filename in self.comparison_files:
            data = self.results.get(filename, {})
            file_data = {
                "file_path": data.get("file_path", "Unknown"),
                "sample_rate": data.get("sample_rate", "Unknown"),
                "duration": data.get("duration", 0),
                "channels": data.get("channels", "Unknown"),
            }
            
            # Add feature values
            for feature in sorted(common_features):
                file_data[feature] = data.get("features", {}).get(feature, None)
            
            comparison_data[filename] = file_data
        
        # Export the data
        try:
            if export_format == "csv":
                # Convert to DataFrame for CSV export
                df_data = []
                for filename, file_data in comparison_data.items():
                    row = {"filename": filename}
                    row.update(file_data)
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_file, index=False)
            else:  # JSON
                with open(output_file, 'w') as f:
                    json.dump(comparison_data, f, indent=2)
            
            print(f"Comparison data exported to {output_file}")
        except Exception as e:
            print(f"Error exporting comparison data: {e}")
            

def explore_results(results_dir: str = "results") -> None:
    """Start the interactive explorer for result files.
    
    Args:
        results_dir: Directory containing result files
    """
    explorer = InteractiveExplorer(results_dir)
    explorer.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive explorer for audio analysis results")
    parser.add_argument("--results-dir", "-r", default="results", 
                        help="Directory containing result files")
    
    args = parser.parse_args()
    
    try:
        explore_results(args.results_dir)
    except Exception as e:
        logger.error(f"Explorer error: {e}")
        print(f"Error: {e}") 