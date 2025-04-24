#!/usr/bin/env python3
"""
Interactive Result Viewer for Heihachi analysis results.

This module provides a terminal-based interactive interface for exploring
audio analysis results, visualizing data, and comparing multiple results.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import textwrap

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    # Optional dependencies for the interactive viewer
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import curses
    import inquirer
    import tabulate
except ImportError as e:
    logger.warning(f"Missing dependency for interactive viewer: {e}")
    logger.warning("Run: pip install heihachi[interactive] to install dependencies")
    missing_deps = True
else:
    missing_deps = False

class ResultViewer:
    """Interactive terminal-based viewer for analysis results."""
    
    def __init__(self, results: Dict[str, Any], result_file: Optional[Path] = None):
        """Initialize the result viewer.
        
        Args:
            results: Analysis results to explore
            result_file: Optional path to result file (for loading additional data)
        """
        self.results = results
        self.result_file = result_file
        self.stdscr = None
        self.width = 80
        self.height = 24
        self.current_section = None
        self.section_data = None
        
        # Check for required dependencies
        if missing_deps:
            raise ImportError("Missing dependencies for interactive viewer. "
                            "Run: pip install heihachi[interactive]")
    
    def start(self) -> None:
        """Start the interactive viewer."""
        try:
            # Initialize curses
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.start_color()
            curses.curs_set(0)  # Hide cursor
            self.stdscr.keypad(True)
            
            # Initialize color pairs
            curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header
            curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Success
            curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)    # Error
            curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Warning
            
            # Get terminal dimensions
            self.height, self.width = self.stdscr.getmaxyx()
            
            # Start main menu
            self._main_menu()
            
        except Exception as e:
            logger.error(f"Error in interactive viewer: {str(e)}")
        finally:
            # Clean up curses
            if self.stdscr:
                self.stdscr.keypad(False)
                curses.echo()
                curses.nocbreak()
                curses.endwin()
    
    def _main_menu(self) -> None:
        """Display the main menu of the interactive viewer."""
        # Temporarily exit curses mode to use inquirer
        curses.endwin()
        
        # Determine available sections based on results structure
        available_sections = self._get_available_sections()
        
        # Build menu options
        menu_options = [
            "View Analysis Summary",
            "Explore Analysis Details",
        ]
        
        # Add conditional options
        if "metadata" in available_sections:
            menu_options.append("View File Metadata")
        if "metrics" in available_sections:
            menu_options.append("View Analysis Metrics")
        if "segments" in available_sections:
            menu_options.append("Explore Track Segments")
        if "visualizations" in available_sections:
            menu_options.append("View Visualizations")
        if "batch_results" in available_sections:
            menu_options.append("Compare Batch Results")
        
        # Always include export and exit
        menu_options.extend([
            "Export Results",
            "Exit"
        ])
        
        # Create menu
        questions = [
            inquirer.List('option',
                          message="Heihachi Analysis Explorer",
                          choices=menu_options,
                          ),
        ]
        
        # Get user selection
        answer = inquirer.prompt(questions)
        
        if answer:
            selected = answer['option']
            
            if selected == "View Analysis Summary":
                self._view_summary()
            elif selected == "Explore Analysis Details":
                self._explore_details()
            elif selected == "View File Metadata":
                self._view_metadata()
            elif selected == "View Analysis Metrics":
                self._view_metrics()
            elif selected == "Explore Track Segments":
                self._explore_segments()
            elif selected == "View Visualizations":
                self._view_visualizations()
            elif selected == "Compare Batch Results":
                self._compare_batch_results()
            elif selected == "Export Results":
                self._export_results()
            elif selected == "Exit":
                return
        
        # Reinitialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        
        # Return to main menu unless exiting
        if answer and answer['option'] != "Exit":
            self._main_menu()
    
    def _get_available_sections(self) -> List[str]:
        """Determine which sections are available in the results.
        
        Returns:
            List of available section names
        """
        available = []
        
        # Check for metadata
        if "metadata" in self.results:
            available.append("metadata")
        
        # Check for metrics
        if "analysis" in self.results and "mix" in self.results["analysis"]:
            mix = self.results["analysis"]["mix"]
            if isinstance(mix, dict) and "metrics" in mix:
                available.append("metrics")
        
        # Check for segments
        if "segments" in self.results:
            available.append("segments")
        
        # Check for visualizations
        if "visualizations" in self.results:
            available.append("visualizations")
        
        # Check for batch results
        if "stats" in self.results and "results" in self.results:
            available.append("batch_results")
        elif "multi_config_results" in self.results:
            available.append("batch_results")
        
        return available
    
    def _view_summary(self) -> None:
        """Display a summary of the analysis results."""
        # Clear screen
        self.stdscr.clear()
        
        # Prepare summary data
        if "metadata" in self.results:
            # Single file result
            file_path = self.results["metadata"].get("file_path", "Unknown file")
            processing_time = self.results["metadata"].get("processing_time", 0)
            
            # Display header
            self.stdscr.addstr(0, 0, f"Analysis Summary for {os.path.basename(file_path)}", 
                              curses.color_pair(1) | curses.A_BOLD)
            self.stdscr.addstr(1, 0, f"Processing time: {processing_time:.2f}s")
            
            # Display metrics summary if available
            if "analysis" in self.results and "mix" in self.results["analysis"]:
                mix = self.results["analysis"]["mix"]
                if isinstance(mix, dict) and "metrics" in mix:
                    metrics = mix["metrics"]
                    
                    self.stdscr.addstr(3, 0, "Key Metrics:", curses.A_BOLD)
                    row = 4
                    for key, value in list(metrics.items())[:5]:  # Show top 5 metrics
                        if isinstance(value, (int, float)):
                            value_str = f"{value:.4f}"
                        else:
                            value_str = str(value)
                        self.stdscr.addstr(row, 2, f"{key}: {value_str}")
                        row += 1
            
            # Display segments summary if available
            if "segments" in self.results:
                segments = self.results["segments"]
                self.stdscr.addstr(row + 1, 0, "Track Segments:", curses.A_BOLD)
                self.stdscr.addstr(row + 2, 2, f"Total segments: {len(segments)}")
                
                # Show segment types distribution
                segment_types = {}
                for segment in segments:
                    label = segment.get("label", "Unknown")
                    segment_types[label] = segment_types.get(label, 0) + 1
                
                row += 3
                for label, count in segment_types.items():
                    self.stdscr.addstr(row, 2, f"{label}: {count} segments")
                    row += 1
        
        elif "stats" in self.results:
            # Batch result
            stats = self.results["stats"]
            
            # Display header
            self.stdscr.addstr(0, 0, "Batch Processing Summary", 
                              curses.color_pair(1) | curses.A_BOLD)
            
            # Display stats
            self.stdscr.addstr(2, 0, "Statistics:", curses.A_BOLD)
            self.stdscr.addstr(3, 2, f"Total files: {stats.get('total', 0)}")
            self.stdscr.addstr(4, 2, f"Successful: {stats.get('success', 0)}", 
                              curses.color_pair(3))
            self.stdscr.addstr(5, 2, f"Failed: {stats.get('failed', 0)}", 
                              curses.color_pair(4) if stats.get('failed', 0) > 0 else 0)
            self.stdscr.addstr(6, 2, f"Total time: {stats.get('total_time', 0):.2f}s")
        
        else:
            # Generic result
            self.stdscr.addstr(0, 0, "Analysis Results Summary", 
                              curses.color_pair(1) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Result sections:", curses.A_BOLD)
            
            row = 3
            for section in self.results.keys():
                self.stdscr.addstr(row, 2, section)
                row += 1
        
        # Footer
        self.stdscr.addstr(self.height - 2, 0, "Press any key to return to main menu")
        self.stdscr.refresh()
        
        # Wait for keypress
        self.stdscr.getch()
    
    def _explore_details(self) -> None:
        """Explore detailed analysis results by navigating through sections."""
        # Exit curses temporarily to use inquirer
        curses.endwin()
        
        # Build available sections
        sections = list(self.results.keys())
        if not sections:
            print("No result sections available to explore.")
            time.sleep(2)
            return
        
        # Create section selection menu
        questions = [
            inquirer.List('section',
                         message="Select a section to explore",
                         choices=sections + ["Back to Main Menu"],
                         ),
        ]
        
        # Get user selection
        answer = inquirer.prompt(questions)
        
        if answer and answer['section'] != "Back to Main Menu":
            selected_section = answer['section']
            self.current_section = selected_section
            self.section_data = self.results[selected_section]
            
            # Display the selected section
            self._display_section(selected_section, self.section_data)
        
        # Reinitialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
    
    def _display_section(self, section_name: str, data: Any) -> None:
        """Display the content of a result section.
        
        Args:
            section_name: Name of the section
            data: Section data to display
        """
        # Format data for display
        formatted_data = json.dumps(data, indent=2)
        
        # Split into lines
        lines = formatted_data.split('\n')
        
        # Setup pager display
        current_line = 0
        page_size = self.height - 4  # Leave room for header and footer
        
        # Display pager
        while True:
            self.stdscr.clear()
            
            # Display header
            self.stdscr.addstr(0, 0, f"Section: {section_name}", 
                              curses.color_pair(1) | curses.A_BOLD)
            
            # Display content page
            line_num = 1
            for i in range(current_line, min(current_line + page_size, len(lines))):
                if i < len(lines):
                    # Truncate long lines
                    line = lines[i]
                    if len(line) > self.width - 2:
                        line = line[:self.width - 5] + "..."
                    
                    # Display line
                    self.stdscr.addstr(line_num, 1, line)
                    line_num += 1
            
            # Display footer/navigation
            footer = f"Line {current_line + 1}-{min(current_line + page_size, len(lines))} of {len(lines)}"
            nav_help = "↑/↓: Scroll | q: Back to menu"
            self.stdscr.addstr(self.height - 2, 0, footer)
            self.stdscr.addstr(self.height - 1, 0, nav_help)
            
            self.stdscr.refresh()
            
            # Handle navigation
            key = self.stdscr.getch()
            
            if key == curses.KEY_UP and current_line > 0:
                current_line -= 1
            elif key == curses.KEY_DOWN and current_line < len(lines) - page_size:
                current_line += 1
            elif key == ord('q') or key == 27:  # q or ESC
                break
            elif key == curses.KEY_NPAGE:  # Page Down
                current_line = min(current_line + page_size, len(lines) - page_size)
            elif key == curses.KEY_PPAGE:  # Page Up
                current_line = max(0, current_line - page_size)
    
    def _view_metadata(self) -> None:
        """View file metadata information."""
        if "metadata" not in self.results:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "Metadata Not Available", 
                              curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        
        metadata = self.results["metadata"]
        
        # Display metadata in a tabular format
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "File Metadata", curses.color_pair(1) | curses.A_BOLD)
        
        row = 2
        for key, value in metadata.items():
            if key == "processing_time" and isinstance(value, (int, float)):
                value = f"{value:.2f}s"
            
            # Format value display
            if isinstance(value, str) and len(value) > self.width - 20:
                value_display = value[:self.width - 23] + "..."
            else:
                value_display = str(value)
            
            self.stdscr.addstr(row, 2, key)
            self.stdscr.addstr(row, 20, value_display)
            row += 1
            
            if row >= self.height - 3:
                break
        
        self.stdscr.addstr(self.height - 2, 0, "Press any key to return")
        self.stdscr.refresh()
        self.stdscr.getch()
    
    def _view_metrics(self) -> None:
        """View analysis metrics in a structured format."""
        # Extract metrics
        metrics = {}
        if "analysis" in self.results and "mix" in self.results["analysis"]:
            mix = self.results["analysis"]["mix"]
            if isinstance(mix, dict) and "metrics" in mix:
                metrics = mix["metrics"]
        
        if not metrics:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "Metrics Not Available", 
                              curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        
        # Display metrics in a tabular format
        self.stdscr.clear()
        self.stdscr.addstr(0, 0, "Analysis Metrics", curses.color_pair(1) | curses.A_BOLD)
        
        # Sort metrics by name
        sorted_metrics = sorted(metrics.items())
        
        row = 2
        for key, value in sorted_metrics:
            if isinstance(value, (int, float)):
                value_display = f"{value:.4f}"
            else:
                value_display = str(value)
            
            self.stdscr.addstr(row, 2, key)
            self.stdscr.addstr(row, 25, value_display)
            row += 1
            
            if row >= self.height - 3:
                break
        
        self.stdscr.addstr(self.height - 2, 0, "Press any key to return")
        self.stdscr.refresh()
        self.stdscr.getch()
    
    def _explore_segments(self) -> None:
        """Explore track segments with detailed information."""
        if "segments" not in self.results or not self.results["segments"]:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "Segments Not Available", 
                              curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        
        segments = self.results["segments"]
        
        # Exit curses temporarily to use inquirer
        curses.endwin()
        
        # Create segment selection menu
        segment_options = [
            f"Segment {i+1}: {seg.get('label', 'Unknown')} ({seg.get('start_time', 0):.2f}s - {seg.get('end_time', 0):.2f}s)"
            for i, seg in enumerate(segments)
        ]
        
        questions = [
            inquirer.List('segment',
                         message="Select a segment to view details",
                         choices=segment_options + ["Back to Main Menu"],
                         ),
        ]
        
        # Get user selection
        answer = inquirer.prompt(questions)
        
        # Reinitialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        
        if answer and answer['segment'] != "Back to Main Menu":
            # Extract segment index
            index = int(answer['segment'].split(':')[0].split()[1]) - 1
            selected_segment = segments[index]
            
            # Display segment details
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, f"Segment {index + 1} Details", 
                              curses.color_pair(1) | curses.A_BOLD)
            
            row = 2
            for key, value in selected_segment.items():
                if isinstance(value, (int, float)):
                    if key in ('start_time', 'end_time'):
                        value_display = f"{value:.2f}s"
                    else:
                        value_display = f"{value:.4f}"
                else:
                    value_display = str(value)
                
                self.stdscr.addstr(row, 2, key)
                self.stdscr.addstr(row, 20, value_display)
                row += 1
            
            self.stdscr.addstr(self.height - 2, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            
            # Return to segment selection
            self._explore_segments()
    
    def _view_visualizations(self) -> None:
        """View and navigate available visualizations."""
        # Check if matplotlib is available
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "Visualization Not Available", 
                              curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Matplotlib is required for visualizations")
            self.stdscr.addstr(3, 0, "Install with: pip install matplotlib")
            self.stdscr.addstr(5, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        
        # Check for visualization data
        if "visualizations" in self.results:
            viz_data = self.results["visualizations"]
        else:
            # Try to extract data for visualization
            viz_data = self._extract_visualization_data()
        
        if not viz_data:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "No Visualization Data Available", 
                              curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        
        # Exit curses temporarily
        curses.endwin()
        
        # Show visualizations based on the data
        if "waveform" in viz_data:
            self._plot_waveform(viz_data["waveform"])
        
        if "spectrogram" in viz_data:
            self._plot_spectrogram(viz_data["spectrogram"])
        
        # Additional visualizations as available
        
        # Reinitialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
    
    def _extract_visualization_data(self) -> Dict[str, Any]:
        """Extract data that can be used for visualization.
        
        Returns:
            Dictionary of visualization data
        """
        viz_data = {}
        
        # Try to find audio data
        if "audio" in self.results:
            viz_data["waveform"] = {
                "data": self.results["audio"],
                "sr": self.results.get("sample_rate", 44100)
            }
        
        # Try to find spectrogram data
        if "analysis" in self.results:
            analysis = self.results["analysis"]
            if "spectrogram" in analysis:
                viz_data["spectrogram"] = {
                    "data": analysis["spectrogram"],
                    "sr": self.results.get("sample_rate", 44100)
                }
        
        return viz_data
    
    def _plot_waveform(self, waveform_data: Dict[str, Any]) -> None:
        """Plot audio waveform.
        
        Args:
            waveform_data: Dictionary with audio data and sample rate
        """
        plt.figure(figsize=(10, 4))
        
        audio = waveform_data["data"]
        sr = waveform_data["sr"]
        
        # Create time axis
        if len(audio) > 0:
            time = np.linspace(0, len(audio) / sr, len(audio))
            plt.plot(time, audio)
            plt.title("Audio Waveform")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
    
    def _plot_spectrogram(self, spectrogram_data: Dict[str, Any]) -> None:
        """Plot spectrogram.
        
        Args:
            spectrogram_data: Dictionary with spectrogram data and parameters
        """
        plt.figure(figsize=(10, 6))
        
        spec = spectrogram_data["data"]
        sr = spectrogram_data["sr"]
        
        plt.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title("Audio Spectrogram")
        plt.xlabel("Time")
        plt.ylabel("Frequency (Hz)")
        
        # Add frequency ticks if available
        if sr is not None:
            plt.yticks(
                np.linspace(0, spec.shape[0], 6),
                [f"{f:.0f}" for f in np.linspace(0, sr/2, 6)]
            )
        
        plt.tight_layout()
        plt.show()
    
    def _compare_batch_results(self) -> None:
        """Compare results from batch processing."""
        # Check for batch results
        batch_data = None
        if "stats" in self.results and "results" in self.results:
            batch_data = self.results
        elif "multi_config_results" in self.results:
            batch_data = self.results["multi_config_results"]
        
        if not batch_data:
            self.stdscr.clear()
            self.stdscr.addstr(0, 0, "Batch Results Not Available", 
                              curses.color_pair(4) | curses.A_BOLD)
            self.stdscr.addstr(2, 0, "Press any key to return")
            self.stdscr.refresh()
            self.stdscr.getch()
            return
        
        # Exit curses temporarily
        curses.endwin()
        
        # Extract comparative data
        comparative_data = self._extract_comparative_data(batch_data)
        
        if not comparative_data:
            print("No comparative data available")
            time.sleep(2)
            
            # Reinitialize curses
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            return
        
        # Display tabular data
        headers = list(comparative_data[0].keys())
        table_data = [[row.get(h, "") for h in headers] for row in comparative_data]
        
        print("\nComparative Results\n")
        print(tabulate.tabulate(table_data, headers=headers, tablefmt="grid"))
        print("\nPress Enter to return to the main menu...")
        input()
        
        # Reinitialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
    
    def _extract_comparative_data(self, batch_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract comparative data from batch results.
        
        Args:
            batch_data: Batch processing results
            
        Returns:
            List of dictionaries with comparative data
        """
        comparative_data = []
        
        # Handle different batch data structures
        if isinstance(batch_data, list):
            # Multi-config results
            for result_item in batch_data:
                if isinstance(result_item, dict):
                    config = result_item.get("config", "Unknown")
                    result = result_item.get("result", {})
                    
                    if "error" in result:
                        # Failed result
                        comparative_data.append({
                            "Config": config,
                            "Status": "Failed",
                            "Error": result["error"].get("type", "Unknown error")
                        })
                    elif "metadata" in result:
                        # Extract key metrics
                        metrics = {}
                        if "analysis" in result and "mix" in result["analysis"]:
                            mix = result["analysis"]["mix"]
                            if isinstance(mix, dict) and "metrics" in mix:
                                metrics = mix["metrics"]
                        
                        data_row = {
                            "Config": config,
                            "Status": "Success",
                            "Processing Time": f"{result['metadata'].get('processing_time', 0):.2f}s"
                        }
                        
                        # Add top metrics
                        for k, v in list(metrics.items())[:5]:
                            if isinstance(v, (int, float)):
                                data_row[k] = f"{v:.4f}"
                            else:
                                data_row[k] = str(v)
                        
                        comparative_data.append(data_row)
        
        elif "stats" in batch_data and "results" in batch_data:
            # Standard batch results
            if "files" in batch_data:
                for file_data in batch_data["files"]:
                    data_row = {
                        "File": file_data.get("file", "Unknown"),
                        "Status": file_data.get("status", "Unknown"),
                        "Processing Time": f"{file_data.get('processing_time', 0):.2f}s"
                    }
                    
                    # Add metrics if available
                    if "metrics" in file_data:
                        for k, v in list(file_data["metrics"].items())[:5]:
                            if isinstance(v, (int, float)):
                                data_row[k] = f"{v:.4f}"
                            else:
                                data_row[k] = str(v)
                    
                    comparative_data.append(data_row)
            
            # If no direct file results but we have a list of individual results
            elif "results" in batch_data and isinstance(batch_data["results"], list):
                for result_item in batch_data["results"]:
                    if isinstance(result_item, dict) and "results" in result_item:
                        config = result_item.get("config", "default")
                        results_data = result_item["results"]
                        
                        if "files" in results_data:
                            for file_data in results_data["files"]:
                                data_row = {
                                    "Config": config,
                                    "File": file_data.get("file", "Unknown"),
                                    "Status": file_data.get("status", "Unknown"),
                                    "Processing Time": f"{file_data.get('processing_time', 0):.2f}s"
                                }
                                
                                # Add metrics if available
                                if "metrics" in file_data:
                                    for k, v in list(file_data["metrics"].items())[:3]:
                                        if isinstance(v, (int, float)):
                                            data_row[k] = f"{v:.4f}"
                                        else:
                                            data_row[k] = str(v)
                                
                                comparative_data.append(data_row)
        
        return comparative_data
    
    def _export_results(self) -> None:
        """Export results to various formats."""
        # Exit curses temporarily
        curses.endwin()
        
        # Import export module
        try:
            from src.utils.export import export_results
        except ImportError:
            print("Export module not available")
            time.sleep(2)
            
            # Reinitialize curses
            self.stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            self.stdscr.keypad(True)
            return
        
        # Get available formats
        format_choices = [
            "JSON",
            "CSV",
            "YAML",
            "Markdown",
            "HTML",
            "XML",
            "Cancel"
        ]
        
        # Create format selection menu
        questions = [
            inquirer.List('format',
                         message="Select export format",
                         choices=format_choices,
                         ),
            inquirer.Text('output_dir',
                         message="Output directory",
                         default="./exports"),
        ]
        
        # Get user selection
        answers = inquirer.prompt(questions)
        
        if answers and answers['format'] != "Cancel":
            selected_format = answers['format'].lower()
            output_dir = answers['output_dir']
            
            try:
                # Create output directory
                Path(output_dir).mkdir(exist_ok=True, parents=True)
                
                # Export results
                export_path = export_results(
                    self.results, 
                    selected_format, 
                    output_dir,
                    "interactive_export"
                )
                
                print(f"Results exported to: {export_path}")
                print("Press Enter to continue...")
                input()
                
            except Exception as e:
                print(f"Error exporting results: {str(e)}")
                print("Press Enter to continue...")
                input()
        
        # Reinitialize curses
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)


def main():
    """Run the interactive result viewer as a standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Heihachi Interactive Result Viewer")
    parser.add_argument("result_file", help="Path to result JSON file")
    args = parser.parse_args()
    
    try:
        # Load results from file
        with open(args.result_file, 'r') as f:
            results = json.load(f)
        
        # Start viewer
        viewer = ResultViewer(results, Path(args.result_file))
        viewer.start()
        
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("Please install required dependencies:")
        print("pip install heihachi[interactive]")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 