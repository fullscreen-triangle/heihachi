#!/usr/bin/env python3
"""
Interactive explorer for audio analysis results.

This module provides an interactive command-line interface for exploring 
and visualizing audio analysis results.
"""

import os
import json
import cmd
import glob
import shutil
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from src.utils.logging_utils import get_logger
from src.visualization.plots import (
    plot_waveform, 
    plot_spectrogram, 
    plot_feature_comparison,
    plot_feature_distribution
)
from src.utils.file_utils import load_json, ensure_directory

logger = get_logger(__name__)
console = Console()


class AnalysisExplorer(cmd.Cmd):
    """Interactive command-line interface for exploring audio analysis results."""
    
    intro = """
    Welcome to the Audio Analysis Explorer!
    
    Type 'help' or '?' to list available commands.
    Type 'exit' or 'quit' to exit the explorer.
    """
    prompt = "[analysis] > "
    
    def __init__(self, results_dir: str = "results", 
                visualization_dir: str = "visualizations"):
        """Initialize the explorer.
        
        Args:
            results_dir: Directory containing analysis results
            visualization_dir: Directory to save visualizations
        """
        super().__init__()
        self.results_dir = os.path.abspath(results_dir)
        self.visualization_dir = os.path.abspath(visualization_dir)
        self.current_file = None
        self.current_data = None
        self.available_features = []
        
        # Ensure visualization directory exists
        ensure_directory(self.visualization_dir)
        
        # Load available results
        self.refresh_results()
    
    def refresh_results(self) -> None:
        """Refresh the list of available results files."""
        self.results_files = glob.glob(os.path.join(self.results_dir, "**/*.json"), recursive=True)
        self.results_files.sort()
    
    def do_list(self, arg: str) -> None:
        """List available analysis result files.
        
        Usage: list [pattern]
        """
        pattern = arg.strip() if arg else "*"
        matching_files = glob.glob(os.path.join(self.results_dir, f"**/{pattern}.json"), recursive=True)
        matching_files.sort()
        
        if not matching_files:
            console.print("[yellow]No matching result files found.[/yellow]")
            return
        
        table = Table(title="Available Analysis Results")
        table.add_column("Index", style="cyan")
        table.add_column("Filename", style="green")
        table.add_column("Last Modified", style="magenta")
        table.add_column("Size", style="blue", justify="right")
        
        for i, filepath in enumerate(matching_files):
            rel_path = os.path.relpath(filepath, self.results_dir)
            stat = os.stat(filepath)
            last_modified = pd.Timestamp(stat.st_mtime, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            size = f"{stat.st_size / 1024:.1f} KB"
            
            table.add_row(str(i+1), rel_path, last_modified, size)
        
        console.print(table)
    
    def do_open(self, arg: str) -> None:
        """Open an analysis result file.
        
        Usage: open INDEX or open FILENAME
        """
        if not arg:
            console.print("[yellow]Please specify an index or filename.[/yellow]")
            return
        
        try:
            # Check if argument is an index
            idx = int(arg) - 1
            if 0 <= idx < len(self.results_files):
                file_path = self.results_files[idx]
            else:
                console.print(f"[red]Index {arg} is out of range.[/red]")
                return
        except ValueError:
            # Argument is a filename
            file_path = arg
            if not os.path.isabs(file_path):
                file_path = os.path.join(self.results_dir, file_path)
            
            if not os.path.exists(file_path):
                file_path_with_ext = file_path
                if not file_path.endswith('.json'):
                    file_path_with_ext = file_path + '.json'
                
                if os.path.exists(file_path_with_ext):
                    file_path = file_path_with_ext
                else:
                    console.print(f"[red]File not found: {file_path}[/red]")
                    return
        
        try:
            self.current_data = load_json(file_path)
            self.current_file = file_path
            
            # Extract available features
            self.available_features = []
            if "features" in self.current_data:
                self.available_features = list(self.current_data["features"].keys())
            
            rel_path = os.path.relpath(file_path, self.results_dir)
            console.print(f"[green]Opened analysis result: {rel_path}[/green]")
            
            # Show summary of the file
            self.do_summary("")
        except Exception as e:
            console.print(f"[red]Error opening file: {e}[/red]")
    
    def do_summary(self, arg: str) -> None:
        """Show summary of the current analysis result.
        
        Usage: summary
        """
        if not self.current_data:
            console.print("[yellow]No analysis result is currently open. Use 'open' to load a file.[/yellow]")
            return
        
        # Create a table with summary information
        table = Table(title=f"Summary for {os.path.basename(self.current_file)}")
        
        # Basic audio information
        if "audio_info" in self.current_data:
            audio_info = self.current_data["audio_info"]
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in audio_info.items():
                table.add_row(key, str(value))
        
        console.print(table)
        
        # Feature summary if available
        if "features" in self.current_data and self.available_features:
            feature_table = Table(title="Available Features")
            feature_table.add_column("Feature", style="cyan")
            feature_table.add_column("Type", style="magenta")
            feature_table.add_column("Shape/Size", style="blue")
            
            for feature_name in self.available_features:
                feature_data = self.current_data["features"][feature_name]
                
                # Determine type and shape
                if isinstance(feature_data, dict):
                    feat_type = "Object"
                    shape = f"{len(feature_data)} properties"
                elif isinstance(feature_data, list):
                    feat_type = "Array"
                    shape = f"{len(feature_data)} elements"
                elif isinstance(feature_data, (int, float, str, bool)):
                    feat_type = type(feature_data).__name__
                    shape = "Scalar"
                else:
                    feat_type = "Unknown"
                    shape = "Unknown"
                
                feature_table.add_row(feature_name, feat_type, shape)
            
            console.print(feature_table)
    
    def do_info(self, arg: str) -> None:
        """Show detailed information about a specific feature.
        
        Usage: info FEATURE_NAME
        """
        if not self.current_data:
            console.print("[yellow]No analysis result is currently open. Use 'open' to load a file.[/yellow]")
            return
        
        if not arg:
            console.print("[yellow]Please specify a feature name.[/yellow]")
            console.print(f"[blue]Available features: {', '.join(self.available_features)}[/blue]")
            return
        
        feature_name = arg.strip()
        
        if feature_name not in self.available_features and "features" in self.current_data:
            console.print(f"[yellow]Feature '{feature_name}' not found.[/yellow]")
            console.print(f"[blue]Available features: {', '.join(self.available_features)}[/blue]")
            return
        
        feature_data = self.current_data["features"][feature_name]
        
        # Determine how to display the feature based on its type
        if isinstance(feature_data, dict):
            # Display as table
            table = Table(title=f"Feature: {feature_name}")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in feature_data.items():
                value_str = str(value)
                if len(value_str) > 100:
                    value_str = value_str[:97] + "..."
                table.add_row(key, value_str)
            
            console.print(table)
        elif isinstance(feature_data, list):
            # Show stats for numeric arrays
            try:
                arr = np.array(feature_data)
                if np.issubdtype(arr.dtype, np.number):
                    stats = {
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "mean": float(np.mean(arr)),
                        "median": float(np.median(arr)),
                        "std": float(np.std(arr)),
                        "shape": arr.shape
                    }
                    
                    table = Table(title=f"Feature: {feature_name} (Statistics)")
                    table.add_column("Statistic", style="cyan")
                    table.add_column("Value", style="green")
                    
                    for key, value in stats.items():
                        table.add_row(key, str(value))
                    
                    console.print(table)
                else:
                    # For non-numeric arrays, show preview
                    console.print(Panel(
                        str(feature_data[:10]) + ("..." if len(feature_data) > 10 else ""),
                        title=f"Feature: {feature_name}"
                    ))
            except Exception as e:
                # Fallback to simple preview
                console.print(Panel(
                    str(feature_data[:10]) + ("..." if len(feature_data) > 10 else ""),
                    title=f"Feature: {feature_name}"
                ))
        else:
            # Simple scalar value
            console.print(Panel(
                str(feature_data),
                title=f"Feature: {feature_name}"
            ))
    
    def do_plot(self, arg: str) -> None:
        """Plot a feature or visualization from the current analysis.
        
        Usage: plot TYPE [FEATURE_NAME] [--save FILENAME]
        
        Types:
          waveform - Plot audio waveform
          spectrogram - Plot audio spectrogram
          feature FEATURE_NAME - Plot a specific feature
        """
        if not self.current_data:
            console.print("[yellow]No analysis result is currently open. Use 'open' to load a file.[/yellow]")
            return
        
        args = arg.split()
        if not args:
            console.print("[yellow]Please specify a plot type. Use 'help plot' for more information.[/yellow]")
            return
        
        plot_type = args[0].lower()
        save_file = None
        
        # Check for --save flag
        if "--save" in args:
            save_index = args.index("--save")
            if save_index + 1 < len(args):
                save_file = args[save_index + 1]
                # Remove the save arguments from args list
                args = args[:save_index] + args[save_index+2:]
            else:
                console.print("[yellow]Missing filename after --save flag.[/yellow]")
                return
        
        figure = None
        
        try:
            # Different plot types
            if plot_type == "waveform":
                if "waveform" in self.current_data:
                    waveform = self.current_data["waveform"]
                    sample_rate = self.current_data["audio_info"]["sample_rate"]
                    figure = plot_waveform(waveform, sample_rate, 
                                        title=f"Waveform - {os.path.basename(self.current_file)}")
                else:
                    console.print("[yellow]Waveform data not found in the current file.[/yellow]")
                    return
                
            elif plot_type == "spectrogram":
                if "spectrogram" in self.current_data:
                    spectrogram = self.current_data["spectrogram"]
                    sample_rate = self.current_data["audio_info"]["sample_rate"]
                    figure = plot_spectrogram(spectrogram, sample_rate,
                                           title=f"Spectrogram - {os.path.basename(self.current_file)}")
                else:
                    console.print("[yellow]Spectrogram data not found in the current file.[/yellow]")
                    return
                
            elif plot_type == "feature":
                if len(args) < 2:
                    console.print("[yellow]Please specify a feature name.[/yellow]")
                    console.print(f"[blue]Available features: {', '.join(self.available_features)}[/blue]")
                    return
                
                feature_name = args[1]
                if feature_name not in self.available_features:
                    console.print(f"[yellow]Feature '{feature_name}' not found.[/yellow]")
                    console.print(f"[blue]Available features: {', '.join(self.available_features)}[/blue]")
                    return
                
                feature_data = self.current_data["features"][feature_name]
                
                # Determine how to plot based on feature type
                if isinstance(feature_data, list):
                    try:
                        # Try to plot as a line graph
                        arr = np.array(feature_data)
                        plt.figure(figsize=(10, 6))
                        plt.plot(arr)
                        plt.title(f"Feature: {feature_name}")
                        plt.xlabel("Index")
                        plt.ylabel("Value")
                        plt.tight_layout()
                        figure = plt.gcf()
                    except Exception as e:
                        console.print(f"[red]Error plotting feature: {e}[/red]")
                        return
                elif isinstance(feature_data, dict):
                    # Plot as bar chart
                    plt.figure(figsize=(12, 6))
                    plt.bar(list(feature_data.keys()), list(feature_data.values()))
                    plt.title(f"Feature: {feature_name}")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()
                    figure = plt.gcf()
                else:
                    console.print(f"[yellow]Cannot plot scalar feature: {feature_name}[/yellow]")
                    return
            else:
                console.print(f"[yellow]Unknown plot type: {plot_type}[/yellow]")
                console.print("[blue]Available types: waveform, spectrogram, feature[/blue]")
                return
            
            # Save or display the figure
            if save_file:
                if not save_file.endswith(('.png', '.jpg', '.pdf', '.svg')):
                    save_file += '.png'
                
                save_path = os.path.join(self.visualization_dir, save_file)
                figure.savefig(save_path, dpi=300, bbox_inches='tight')
                console.print(f"[green]Plot saved to: {save_path}[/green]")
            else:
                plt.show()
                
        except Exception as e:
            console.print(f"[red]Error creating plot: {e}[/red]")
    
    def do_compare(self, arg: str) -> None:
        """Compare a feature across multiple analysis results.
        
        Usage: compare FEATURE_NAME [result1.json result2.json ...]
        
        If no result files are specified, all currently opened results will be used.
        """
        args = arg.split()
        if not args:
            console.print("[yellow]Please specify a feature name.[/yellow]")
            return
        
        feature_name = args[0]
        result_files = args[1:] if len(args) > 1 else []
        
        # If no files specified but we have a current file
        if not result_files and self.current_file:
            result_files = [self.current_file]
        
        if not result_files:
            console.print("[yellow]No result files specified. Please open a file first or specify files.[/yellow]")
            return
        
        # Validate files and load data
        file_data = {}
        for file_spec in result_files:
            try:
                # Check if it's an index
                try:
                    idx = int(file_spec) - 1
                    if 0 <= idx < len(self.results_files):
                        file_path = self.results_files[idx]
                    else:
                        console.print(f"[yellow]Index {file_spec} is out of range, skipping.[/yellow]")
                        continue
                except ValueError:
                    # It's a filename
                    file_path = file_spec
                    if not os.path.isabs(file_path):
                        file_path = os.path.join(self.results_dir, file_path)
                
                # Check if file exists
                if not os.path.exists(file_path):
                    if not file_path.endswith('.json'):
                        file_path += '.json'
                    if not os.path.exists(file_path):
                        console.print(f"[yellow]File not found: {file_spec}, skipping.[/yellow]")
                        continue
                
                # Load data
                data = load_json(file_path)
                if "features" not in data or feature_name not in data["features"]:
                    console.print(f"[yellow]Feature '{feature_name}' not found in {os.path.basename(file_path)}, skipping.[/yellow]")
                    continue
                
                # Add to our collection
                file_data[os.path.basename(file_path)] = data["features"][feature_name]
                
            except Exception as e:
                console.print(f"[red]Error processing {file_spec}: {e}[/red]")
        
        if not file_data:
            console.print("[yellow]No valid data found for comparison.[/yellow]")
            return
        
        # Now create comparison visualization based on data type
        try:
            # Determine data type from first file
            first_file = next(iter(file_data.values()))
            
            if all(isinstance(v, (int, float)) for v in file_data.values()):
                # Scalar values - bar chart
                plt.figure(figsize=(10, 6))
                plt.bar(list(file_data.keys()), list(file_data.values()))
                plt.title(f"Comparison of {feature_name}")
                plt.xticks(rotation=45, ha="right")
                plt.ylabel(feature_name)
                plt.tight_layout()
            
            elif all(isinstance(v, list) for v in file_data.values()):
                # List values - line chart overlay
                plt.figure(figsize=(12, 6))
                for filename, values in file_data.items():
                    plt.plot(values, label=filename)
                plt.title(f"Comparison of {feature_name}")
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.legend()
                plt.tight_layout()
            
            elif all(isinstance(v, dict) for v in file_data.values()):
                # Use feature comparison function from visualization module
                figure = plot_feature_comparison(file_data, feature_name)
            
            else:
                console.print("[yellow]Mixed data types not supported for comparison.[/yellow]")
                return
            
            # Ask if user wants to save the comparison
            save_file = None
            if Confirm.ask("Save this comparison?"):
                save_file = Prompt.ask("Enter filename to save as", 
                                      default=f"comparison_{feature_name}.png")
                
                if not save_file.endswith(('.png', '.jpg', '.pdf', '.svg')):
                    save_file += '.png'
                
                save_path = os.path.join(self.visualization_dir, save_file)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                console.print(f"[green]Comparison saved to: {save_path}[/green]")
            
            plt.show()
            
        except Exception as e:
            console.print(f"[red]Error creating comparison: {e}[/red]")
    
    def do_distribution(self, arg: str) -> None:
        """Plot distribution of a feature across multiple analysis results.
        
        Usage: distribution FEATURE_NAME [result1.json result2.json ...]
        
        If no result files are specified, all available results will be used.
        """
        args = arg.split()
        if not args:
            console.print("[yellow]Please specify a feature name.[/yellow]")
            return
        
        feature_name = args[0]
        result_files = args[1:] if len(args) > 1 else []
        
        # If no files specified, use all available results
        if not result_files:
            result_files = self.results_files
        
        if not result_files:
            console.print("[yellow]No result files available.[/yellow]")
            return
        
        # Collect feature values from all files
        feature_values = {}
        
        for file_path in result_files:
            try:
                # Handle index or filename
                if not os.path.exists(file_path):
                    try:
                        idx = int(file_path) - 1
                        if 0 <= idx < len(self.results_files):
                            file_path = self.results_files[idx]
                        else:
                            continue
                    except ValueError:
                        # Try as a path relative to results dir
                        if not os.path.isabs(file_path):
                            file_path = os.path.join(self.results_dir, file_path)
                        if not os.path.exists(file_path):
                            if not file_path.endswith('.json'):
                                file_path += '.json'
                            if not os.path.exists(file_path):
                                continue
                
                # Load data
                data = load_json(file_path)
                if "features" in data and feature_name in data["features"]:
                    feature_value = data["features"][feature_name]
                    if isinstance(feature_value, (int, float)):
                        feature_values[os.path.basename(file_path)] = feature_value
            
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
        
        if not feature_values:
            console.print(f"[yellow]No valid data found for feature '{feature_name}'.[/yellow]")
            return
        
        # Create distribution plot
        try:
            figure = plot_feature_distribution(feature_values, feature_name)
            
            # Ask if user wants to save the distribution
            save_file = None
            if Confirm.ask("Save this distribution?"):
                save_file = Prompt.ask("Enter filename to save as", 
                                      default=f"distribution_{feature_name}.png")
                
                if not save_file.endswith(('.png', '.jpg', '.pdf', '.svg')):
                    save_file += '.png'
                
                save_path = os.path.join(self.visualization_dir, save_file)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                console.print(f"[green]Distribution saved to: {save_path}[/green]")
            
            plt.show()
            
        except Exception as e:
            console.print(f"[red]Error creating distribution: {e}[/red]")
    
    def do_export(self, arg: str) -> None:
        """Export the current analysis result to another format.
        
        Usage: export FORMAT [FILENAME]
        
        Formats:
          csv - Export features to CSV
          json - Pretty format the JSON
          markdown - Export summary as Markdown
        """
        if not self.current_data:
            console.print("[yellow]No analysis result is currently open. Use 'open' to load a file.[/yellow]")
            return
        
        args = arg.split()
        if not args:
            console.print("[yellow]Please specify an export format (csv, json, markdown).[/yellow]")
            return
        
        export_format = args[0].lower()
        
        # Determine filename
        if len(args) > 1:
            filename = args[1]
        else:
            base_name = os.path.splitext(os.path.basename(self.current_file))[0]
            if export_format == "csv":
                filename = f"{base_name}.csv"
            elif export_format == "json":
                filename = f"{base_name}_pretty.json"
            elif export_format == "markdown":
                filename = f"{base_name}.md"
            else:
                filename = f"{base_name}.{export_format}"
        
        # Create export directory if it doesn't exist
        export_dir = os.path.join(self.results_dir, "exports")
        ensure_directory(export_dir)
        export_path = os.path.join(export_dir, filename)
        
        try:
            if export_format == "csv":
                # Extract features to DataFrame
                if "features" not in self.current_data:
                    console.print("[yellow]No features found to export to CSV.[/yellow]")
                    return
                
                # Flatten features dictionary to handle nested structures
                flat_features = {}
                for feature_name, feature_value in self.current_data["features"].items():
                    if isinstance(feature_value, (int, float, str, bool)):
                        flat_features[feature_name] = feature_value
                    elif isinstance(feature_value, list) and all(isinstance(x, (int, float)) for x in feature_value):
                        # For numeric lists, store statistics
                        arr = np.array(feature_value)
                        flat_features[f"{feature_name}_min"] = np.min(arr)
                        flat_features[f"{feature_name}_max"] = np.max(arr)
                        flat_features[f"{feature_name}_mean"] = np.mean(arr)
                        flat_features[f"{feature_name}_median"] = np.median(arr)
                        flat_features[f"{feature_name}_std"] = np.std(arr)
                    elif isinstance(feature_value, dict):
                        # Flatten one level of dictionary
                        for key, val in feature_value.items():
                            if isinstance(val, (int, float, str, bool)):
                                flat_features[f"{feature_name}_{key}"] = val
                
                # Create DataFrame and save to CSV
                df = pd.DataFrame([flat_features])
                df.to_csv(export_path, index=False)
                console.print(f"[green]Exported to CSV: {export_path}[/green]")
                
            elif export_format == "json":
                # Pretty-print JSON
                with open(export_path, "w") as f:
                    json.dump(self.current_data, f, indent=2)
                console.print(f"[green]Exported to formatted JSON: {export_path}[/green]")
                
            elif export_format == "markdown":
                # Generate Markdown report
                with open(export_path, "w") as f:
                    f.write(f"# Analysis Results: {os.path.basename(self.current_file)}\n\n")
                    
                    # Audio information
                    f.write("## Audio Information\n\n")
                    f.write("| Property | Value |\n")
                    f.write("|----------|-------|\n")
                    if "audio_info" in self.current_data:
                        for key, value in self.current_data["audio_info"].items():
                            f.write(f"| {key} | {value} |\n")
                    
                    # Features
                    if "features" in self.current_data and self.available_features:
                        f.write("\n## Features\n\n")
                        for feature_name in self.available_features:
                            feature_data = self.current_data["features"][feature_name]
                            f.write(f"### {feature_name}\n\n")
                            
                            if isinstance(feature_data, dict):
                                f.write("| Property | Value |\n")
                                f.write("|----------|-------|\n")
                                for key, value in feature_data.items():
                                    f.write(f"| {key} | {value} |\n")
                            elif isinstance(feature_data, list):
                                try:
                                    arr = np.array(feature_data)
                                    if np.issubdtype(arr.dtype, np.number):
                                        f.write("| Statistic | Value |\n")
                                        f.write("|-----------|-------|\n")
                                        f.write(f"| Min | {np.min(arr)} |\n")
                                        f.write(f"| Max | {np.max(arr)} |\n")
                                        f.write(f"| Mean | {np.mean(arr)} |\n")
                                        f.write(f"| Median | {np.median(arr)} |\n")
                                        f.write(f"| Std Dev | {np.std(arr)} |\n")
                                    else:
                                        f.write(f"Array with {len(feature_data)} elements\n")
                                except:
                                    f.write(f"Array with {len(feature_data)} elements\n")
                            else:
                                f.write(f"Value: {feature_data}\n")
                            
                            f.write("\n")
                
                console.print(f"[green]Exported to Markdown: {export_path}[/green]")
                
            else:
                console.print(f"[yellow]Unsupported export format: {export_format}[/yellow]")
                console.print("[blue]Supported formats: csv, json, markdown[/blue]")
                
        except Exception as e:
            console.print(f"[red]Error exporting data: {e}[/red]")
    
    def do_refresh(self, arg: str) -> None:
        """Refresh the list of available result files.
        
        Usage: refresh
        """
        self.refresh_results()
        console.print(f"[green]Refreshed result files. {len(self.results_files)} files found.[/green]")
    
    def do_exit(self, arg: str) -> bool:
        """Exit the explorer.
        
        Usage: exit
        """
        return True
    
    def do_quit(self, arg: str) -> bool:
        """Exit the explorer.
        
        Usage: quit
        """
        return True
    
    # Command shortcuts
    do_ls = do_list
    do_l = do_list
    do_o = do_open
    do_s = do_summary
    do_i = do_info
    do_p = do_plot
    do_e = do_export
    do_q = do_quit


def start_explorer(results_dir: str = "results", visualization_dir: str = "visualizations") -> None:
    """Start the interactive explorer.
    
    Args:
        results_dir: Directory containing analysis results
        visualization_dir: Directory to save visualizations
    """
    console.print(Panel.fit(
        "[bold blue]Audio Analysis Explorer[/bold blue]\n\n"
        "An interactive tool for exploring audio analysis results",
        title="Welcome",
        subtitle="Press Ctrl+C to exit"
    ))
    
    try:
        explorer = AnalysisExplorer(results_dir, visualization_dir)
        explorer.cmdloop()
    except KeyboardInterrupt:
        console.print("\n[yellow]Explorer terminated by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error in explorer: {e}[/red]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive audio analysis results explorer")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing analysis results")
    parser.add_argument("--vis-dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    
    args = parser.parse_args()
    start_explorer(args.results_dir, args.vis_dir) 