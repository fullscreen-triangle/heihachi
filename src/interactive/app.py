#!/usr/bin/env python3
"""
Interactive mode for exploring Heihachi analysis results.

This module provides an interactive interface for exploring and visualizing
analysis results from the Heihachi audio framework.
"""

import os
import sys
import cmd
import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import readline
import shutil
import textwrap
from colorama import Fore, Back, Style, init as colorama_init

# Initialize colorama for cross-platform colored terminal output
colorama_init()

# Add project root to path if needed
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.logging_utils import get_logger
from src.utils.export import export_results
from src.utils.visualization import visualize_feature, visualize_comparison

logger = get_logger(__name__)

class HeihachiInteractive(cmd.Cmd):
    """Interactive command processor for exploring Heihachi results."""
    
    intro = f"""
{Fore.CYAN}{Style.BRIGHT}======================================{Style.RESET_ALL}
{Fore.CYAN}{Style.BRIGHT}Heihachi Interactive Analysis Explorer{Style.RESET_ALL}
{Fore.CYAN}{Style.BRIGHT}======================================{Style.RESET_ALL}

Type {Fore.GREEN}'help'{Style.RESET_ALL} or {Fore.GREEN}'?'{Style.RESET_ALL} to list commands.
Type {Fore.GREEN}'help <command>'{Style.RESET_ALL} for information on a specific command.
Type {Fore.GREEN}'exit'{Style.RESET_ALL} to quit.
"""
    prompt = f"{Fore.BLUE}{Style.BRIGHT}heihachi>{Style.RESET_ALL} "
    
    def __init__(self, results: Dict[str, Any], result_file: Optional[Path] = None):
        """Initialize the interactive shell with analysis results.
        
        Args:
            results: Analysis results dictionary
            result_file: Optional path to the results file
        """
        super().__init__()
        self.results = results
        self.result_file = result_file
        self.current_path = []  # For navigating nested dictionaries
        self.visualizations = []  # Keep track of visualization figures
        self.export_formats = ['json', 'csv', 'yaml', 'markdown', 'html']
        
        # Check terminal size for display formatting
        self.term_width, self.term_height = shutil.get_terminal_size()
        
        # Load history if available
        self.history_file = os.path.expanduser('~/.heihachi_history')
        try:
            readline.read_history_file(self.history_file)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        
        logger.info("Interactive mode initialized")
    
    def cmdloop(self, intro=None):
        """Override cmdloop to handle keyboard interrupts gracefully."""
        while True:
            try:
                super().cmdloop(intro=intro)
                break
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}KeyboardInterrupt: Press Ctrl+D or type 'exit' to quit{Style.RESET_ALL}")
                intro = ""
    
    def preloop(self):
        """Actions to perform before command loop starts."""
        self._show_summary()
    
    def postloop(self):
        """Actions to perform before exiting."""
        # Save command history
        try:
            readline.write_history_file(self.history_file)
        except Exception as e:
            logger.warning(f"Failed to save command history: {e}")
        
        # Close any open visualizations
        plt.close('all')
        
        print(f"\n{Fore.GREEN}Thank you for using Heihachi Interactive Explorer!{Style.RESET_ALL}")
    
    def emptyline(self):
        """Do nothing on empty line."""
        pass
    
    def default(self, line):
        """Handle unknown commands."""
        print(f"{Fore.RED}Unknown command: {line}{Style.RESET_ALL}")
        print(f"Type {Fore.GREEN}'help'{Style.RESET_ALL} for a list of commands.")
    
    def completedefault(self, text, line, begidx, endidx):
        """Default completion handler."""
        return self._complete_path(text)
    
    def _complete_path(self, text):
        """Complete a path in the results structure."""
        current = self._get_current_dict()
        if not text:
            return list(current.keys())
        return [k for k in current.keys() if k.startswith(text)]
    
    def _show_summary(self):
        """Show a summary of the analysis results."""
        if not self.results:
            print(f"{Fore.YELLOW}No results available.{Style.RESET_ALL}")
            return
            
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Analysis Results Summary:{Style.RESET_ALL}\n")
        
        # File info if available
        if 'file_info' in self.results:
            file_info = self.results['file_info']
            print(f"{Fore.CYAN}File:{Style.RESET_ALL} {file_info.get('filename', 'Unknown')}")
            print(f"{Fore.CYAN}Format:{Style.RESET_ALL} {file_info.get('format', 'Unknown')}")
            print(f"{Fore.CYAN}Duration:{Style.RESET_ALL} {file_info.get('duration', 0):.2f} seconds")
            print(f"{Fore.CYAN}Sample Rate:{Style.RESET_ALL} {file_info.get('sample_rate', 0)} Hz")
            print()
        
        # Top-level sections
        print(f"{Fore.CYAN}Available sections:{Style.RESET_ALL}")
        for key in self.results.keys():
            if key != 'file_info':
                print(f"  - {key}")
        
        print(f"\nUse {Fore.GREEN}'ls <section>'{Style.RESET_ALL} to explore specific sections.")
        print(f"Use {Fore.GREEN}'visualize <feature>'{Style.RESET_ALL} to visualize features.")
        print()
    
    def _get_current_dict(self):
        """Get the current dictionary based on the path."""
        current = self.results
        for key in self.current_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return {}
        return current if isinstance(current, dict) else {}
    
    def _format_value(self, value, indent=0):
        """Format a value for display, handling different types."""
        if isinstance(value, dict):
            if not value:
                return f"{Fore.BLUE}{{}}{Style.RESET_ALL}"
            preview = ', '.join(f"{k}: ..." for k in list(value.keys())[:3])
            if len(value) > 3:
                preview += f", ... ({len(value)} keys total)"
            return f"{Fore.BLUE}{{{preview}}}{Style.RESET_ALL}"
        elif isinstance(value, list):
            if not value:
                return f"{Fore.BLUE}[]{Style.RESET_ALL}"
            if len(value) > 3:
                return f"{Fore.BLUE}[...] ({len(value)} items){Style.RESET_ALL}"
            return f"{Fore.BLUE}{str(value)}{Style.RESET_ALL}"
        elif isinstance(value, (int, float)):
            return f"{Fore.MAGENTA}{value}{Style.RESET_ALL}"
        elif isinstance(value, str):
            if len(value) > 50:
                return f"{Fore.GREEN}'{value[:47]}...'{Style.RESET_ALL}"
            return f"{Fore.GREEN}'{value}'{Style.RESET_ALL}"
        elif value is None:
            return f"{Fore.RED}None{Style.RESET_ALL}"
        elif isinstance(value, np.ndarray):
            shape_str = f"ndarray{value.shape}"
            return f"{Fore.CYAN}{shape_str}{Style.RESET_ALL}"
        else:
            return f"{Fore.YELLOW}{str(value)}{Style.RESET_ALL}"
    
    def _print_path(self):
        """Print the current path."""
        path_str = '/'.join(self.current_path) if self.current_path else '/'
        print(f"{Fore.YELLOW}Current path: {path_str}{Style.RESET_ALL}")
    
    # Command implementations
    def do_ls(self, arg):
        """List contents of the current path or specified path.
        
        Usage: ls [path]
        """
        # Save current path
        saved_path = self.current_path.copy()
        
        # Change to the specified path if provided
        if arg:
            try:
                self._change_path(arg)
            except ValueError as e:
                print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
                return
        
        current = self._get_current_dict()
        path_str = '/'.join(self.current_path) if self.current_path else '/'
        
        print(f"\n{Fore.YELLOW}Contents of {path_str}:{Style.RESET_ALL}\n")
        
        # Display dictionary items
        if isinstance(current, dict):
            # Get max key length for alignment
            max_key_len = max([len(str(k)) for k in current.keys()]) if current else 0
            
            for key, value in current.items():
                key_str = str(key).ljust(max_key_len)
                print(f"  {Fore.CYAN}{key_str}{Style.RESET_ALL} : {self._format_value(value)}")
        else:
            print(f"{self._format_value(current)}")
        
        print()
        
        # Restore path if we changed it temporarily
        if arg:
            self.current_path = saved_path
    
    def do_cd(self, arg):
        """Change current path.
        
        Usage: cd [path]
        
        Examples:
          cd                 # Return to root
          cd ..              # Go up one level
          cd features        # Go to features
          cd features/tempo  # Go to features/tempo
        """
        if not arg:
            # Return to root
            self.current_path = []
            print(f"{Fore.GREEN}Changed to root path{Style.RESET_ALL}")
            return
            
        try:
            self._change_path(arg)
            path_str = '/'.join(self.current_path) if self.current_path else '/'
            print(f"{Fore.GREEN}Changed to path: {path_str}{Style.RESET_ALL}")
        except ValueError as e:
            print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
    
    def _change_path(self, path):
        """Change the current path.
        
        Args:
            path: Path to change to
            
        Raises:
            ValueError: If the path is invalid
        """
        parts = path.split('/')
        
        # Handle relative paths
        if parts[0] == '..':
            if not self.current_path:
                raise ValueError("Already at root path")
            self.current_path.pop()
            if len(parts) > 1:
                self._change_path('/'.join(parts[1:]))
            return
        
        # Construct new path and validate
        new_path = self.current_path.copy()
        
        for part in parts:
            if not part or part == '.':
                continue
                
            if part == '..':
                if not new_path:
                    raise ValueError("Cannot go up from root path")
                new_path.pop()
                continue
                
            # Validate key exists
            current = self.results
            for key in new_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    raise ValueError(f"Invalid path: {'/'.join(new_path)}/{part}")
            
            if not isinstance(current, dict) or part not in current:
                raise ValueError(f"Path not found: {'/'.join(new_path)}/{part}")
                
            new_path.append(part)
        
        self.current_path = new_path
    
    def complete_cd(self, text, line, begidx, endidx):
        """Tab completion for cd command."""
        # Split the path by slashes
        parts = line.split()[1:] if len(line.split()) > 1 else ['']
        
        if len(parts) > 1:
            # We're in a path with slashes
            base_path = '/'.join(parts[:-1])
            if base_path.endswith('/'):
                base_path = base_path[:-1]
            
            # Save current path
            saved_path = self.current_path.copy()
            
            try:
                self._change_path(base_path)
                completions = self._complete_path(parts[-1])
                self.current_path = saved_path  # Restore path
                return completions
            except ValueError:
                self.current_path = saved_path  # Restore path
                return []
        else:
            # Simple completion
            return self._complete_path(text)
    
    def do_cat(self, arg):
        """Display the contents of a specific path.
        
        Usage: cat <path>
        
        Example: cat features/tempo
        """
        if not arg:
            print(f"{Fore.RED}Error: Path required{Style.RESET_ALL}")
            print(f"Usage: cat <path>")
            return
            
        # Save current path
        saved_path = self.current_path.copy()
        
        try:
            # Extract parent path and key
            if '/' in arg:
                parent_path, key = arg.rsplit('/', 1)
                self._change_path(parent_path)
                current = self._get_current_dict()
                
                if key not in current:
                    print(f"{Fore.RED}Key not found: {key}{Style.RESET_ALL}")
                    self.current_path = saved_path
                    return
                    
                value = current[key]
            else:
                current = self._get_current_dict()
                
                if arg not in current:
                    print(f"{Fore.RED}Key not found: {arg}{Style.RESET_ALL}")
                    return
                    
                value = current[arg]
                
            # Special handling for different types
            if isinstance(value, (dict, list)):
                json_str = json.dumps(value, indent=2, default=str)
                print(f"\n{json_str}\n")
            elif isinstance(value, np.ndarray):
                print(f"\nArray shape: {value.shape}")
                print(f"Array type: {value.dtype}")
                print(f"Min: {value.min()}, Max: {value.max()}, Mean: {value.mean()}")
                if value.size <= 100:  # Only show full array if it's small
                    print(f"\n{value}\n")
                else:
                    print(f"\nFirst 10 elements: {value.flatten()[:10]}\n")
            else:
                print(f"\n{value}\n")
                
        except ValueError as e:
            print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
        finally:
            # Restore path
            self.current_path = saved_path
    
    def complete_cat(self, text, line, begidx, endidx):
        """Tab completion for cat command."""
        return self.complete_cd(text, line, begidx, endidx)
    
    def do_pwd(self, arg):
        """Print current path.
        
        Usage: pwd
        """
        self._print_path()
    
    def do_visualize(self, arg):
        """Visualize a feature from the analysis.
        
        Usage: visualize <feature_path> [<options>]
        
        Examples:
          visualize features/spectrum
          visualize features/mfcc --cmap viridis
        """
        if not arg:
            print(f"{Fore.RED}Error: Feature path required{Style.RESET_ALL}")
            print(f"Usage: visualize <feature_path> [<options>]")
            print(f"\nAvailable features for visualization:")
            print(f"  features/spectrum - Audio spectrum")
            print(f"  features/mfcc - Mel-frequency cepstral coefficients")
            print(f"  features/tempo - Tempo analysis")
            print(f"  features/onsets - Onset detection")
            return
            
        # Split args
        args = arg.split()
        feature_path = args[0]
        
        # Parse options
        options = {}
        i = 1
        while i < len(args):
            if args[i].startswith('--'):
                option = args[i][2:]
                if i + 1 < len(args) and not args[i+1].startswith('--'):
                    options[option] = args[i+1]
                    i += 2
                else:
                    options[option] = True
                    i += 1
            else:
                i += 1
        
        # Save current path
        saved_path = self.current_path.copy()
        
        try:
            # Extract data to visualize
            if '/' in feature_path:
                parent_path, key = feature_path.rsplit('/', 1)
                self._change_path(parent_path)
                current = self._get_current_dict()
                
                if key not in current:
                    print(f"{Fore.RED}Feature not found: {key}{Style.RESET_ALL}")
                    self.current_path = saved_path
                    return
                    
                data = current[key]
            else:
                current = self._get_current_dict()
                
                if feature_path not in current:
                    print(f"{Fore.RED}Feature not found: {feature_path}{Style.RESET_ALL}")
                    return
                    
                data = current[feature_path]
            
            # Visualize
            fig = visualize_feature(data, feature_path, **options)
            
            if fig:
                plt.show(block=False)
                self.visualizations.append(fig)
                print(f"{Fore.GREEN}Visualization created and displayed{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Unable to visualize: {feature_path}{Style.RESET_ALL}")
                print(f"Feature may not be compatible with visualization.")
                
        except Exception as e:
            print(f"{Fore.RED}Error in visualization: {str(e)}{Style.RESET_ALL}")
            logger.exception(f"Visualization error for {feature_path}")
        finally:
            # Restore path
            self.current_path = saved_path
    
    def complete_visualize(self, text, line, begidx, endidx):
        """Tab completion for visualize command."""
        if line.count(' ') == 1 or (line.count(' ') > 1 and not text):
            # Complete the feature path
            return self.complete_cd(text, "cd " + line.split(' ', 1)[1], begidx, endidx)
        elif line.count(' ') > 1 and text.startswith('--'):
            # Complete option names
            options = ['--cmap', '--title', '--figsize', '--dpi', '--style']
            return [opt for opt in options if opt.startswith(text)]
        elif line.count(' ') > 1 and line.split()[-2] == '--cmap':
            # Complete colormap names
            cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                    'Greys', 'Blues', 'Reds', 'jet', 'rainbow']
            return [cmap for cmap in cmaps if cmap.startswith(text)]
        elif line.count(' ') > 1 and line.split()[-2] == '--style':
            # Complete style names
            styles = ['default', 'classic', 'bmh', 'dark_background', 'ggplot',
                     'seaborn', 'seaborn-darkgrid', 'seaborn-whitegrid']
            return [style for style in styles if style.startswith(text)]
        return []
    
    def do_export(self, arg):
        """Export results to a file.
        
        Usage: export <format> [<path> [<output_file>]]
        
        Formats: json, csv, yaml, markdown, html
        
        Examples:
          export json                     # Export all results to JSON
          export csv features/spectrum    # Export spectrum data to CSV
          export yaml features output.yaml # Export features to YAML file
        """
        if not arg:
            print(f"{Fore.RED}Error: Export format required{Style.RESET_ALL}")
            print(f"Usage: export <format> [<path> [<output_file>]]")
            print(f"\nAvailable formats: {', '.join(self.export_formats)}")
            return
            
        # Parse arguments
        args = arg.split()
        format_name = args[0].lower()
        
        if format_name not in self.export_formats:
            print(f"{Fore.RED}Invalid format: {format_name}{Style.RESET_ALL}")
            print(f"Available formats: {', '.join(self.export_formats)}")
            return
        
        # Get path to export
        if len(args) > 1:
            path = args[1]
        else:
            path = ''
            
        # Get output filename
        if len(args) > 2:
            output_file = args[2]
        else:
            # Generate default filename
            base_name = 'results'
            if path:
                base_name = path.replace('/', '_')
            output_file = f"{base_name}.{format_name}"
            
        # Save current path
        saved_path = self.current_path.copy()
        
        try:
            # Get data to export
            if path:
                try:
                    self._change_path(path)
                    data = self._get_current_dict()
                except ValueError:
                    # Try as a direct path
                    if '/' in path:
                        parent_path, key = path.rsplit('/', 1)
                        try:
                            self._change_path(parent_path)
                            current = self._get_current_dict()
                            
                            if key not in current:
                                print(f"{Fore.RED}Path not found: {path}{Style.RESET_ALL}")
                                self.current_path = saved_path
                                return
                                
                            data = {key: current[key]}
                        except ValueError:
                            print(f"{Fore.RED}Path not found: {path}{Style.RESET_ALL}")
                            self.current_path = saved_path
                            return
                    else:
                        # Try as a key in current path
                        current = self._get_current_dict()
                        if path not in current:
                            print(f"{Fore.RED}Path not found: {path}{Style.RESET_ALL}")
                            return
                        data = {path: current[path]}
            else:
                data = self.results
                
            # Export data
            export_results(data, output_file, format_name)
            print(f"{Fore.GREEN}Results exported to {output_file}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Export error: {str(e)}{Style.RESET_ALL}")
            logger.exception(f"Export error for {path} to {format_name}")
        finally:
            # Restore path
            self.current_path = saved_path
    
    def complete_export(self, text, line, begidx, endidx):
        """Tab completion for export command."""
        args = line.split()
        
        if len(args) == 1 or (len(args) == 2 and not text):
            # Complete format
            return [fmt for fmt in self.export_formats if fmt.startswith(text)]
        elif len(args) == 2 or (len(args) == 3 and not text):
            # Complete path
            return self.complete_cd(text, "cd " + (args[2] if len(args) > 2 else ""), begidx, endidx)
        return []
    
    def do_find(self, arg):
        """Find keys in the results.
        
        Usage: find <search_term>
        """
        if not arg:
            print(f"{Fore.RED}Error: Search term required{Style.RESET_ALL}")
            print(f"Usage: find <search_term>")
            return
            
        search_term = arg.lower()
        results = []
        
        def search_dict(d, path=''):
            """Recursively search a dictionary for keys."""
            for key, value in d.items():
                current_path = f"{path}/{key}" if path else key
                
                if str(key).lower().find(search_term) >= 0:
                    results.append((current_path, value))
                    
                if isinstance(value, dict):
                    search_dict(value, current_path)
        
        search_dict(self.results)
        
        if results:
            print(f"\n{Fore.GREEN}Found {len(results)} matches:{Style.RESET_ALL}\n")
            
            for path, value in results:
                print(f"  {Fore.CYAN}{path}{Style.RESET_ALL} : {self._format_value(value)}")
            print()
        else:
            print(f"\n{Fore.YELLOW}No matches found for '{search_term}'{Style.RESET_ALL}\n")
    
    def do_compare(self, arg):
        """Compare features from different parts of the results.
        
        Usage: compare <feature1> <feature2> [<options>]
        
        Examples:
          compare features/mfcc features/spectrum
        """
        if not arg:
            print(f"{Fore.RED}Error: At least two features required{Style.RESET_ALL}")
            print(f"Usage: compare <feature1> <feature2> [<options>]")
            return
            
        # Parse arguments
        args = arg.split()
        if len(args) < 2:
            print(f"{Fore.RED}Error: At least two features required{Style.RESET_ALL}")
            print(f"Usage: compare <feature1> <feature2> [<options>]")
            return
            
        feature1 = args[0]
        feature2 = args[1]
        
        # Parse options
        options = {}
        i = 2
        while i < len(args):
            if args[i].startswith('--'):
                option = args[i][2:]
                if i + 1 < len(args) and not args[i+1].startswith('--'):
                    options[option] = args[i+1]
                    i += 2
                else:
                    options[option] = True
                    i += 1
            else:
                i += 1
        
        # Save current path
        saved_path = self.current_path.copy()
        
        try:
            # Extract data for feature 1
            data1 = self._get_feature_data(feature1)
            if data1 is None:
                self.current_path = saved_path
                return
                
            # Extract data for feature 2
            data2 = self._get_feature_data(feature2)
            if data2 is None:
                self.current_path = saved_path
                return
                
            # Compare and visualize
            fig = visualize_comparison(data1, data2, feature1, feature2, **options)
            
            if fig:
                plt.show(block=False)
                self.visualizations.append(fig)
                print(f"{Fore.GREEN}Comparison visualization created{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Unable to compare features{Style.RESET_ALL}")
                print(f"Features may not be compatible for comparison.")
                
        except Exception as e:
            print(f"{Fore.RED}Comparison error: {str(e)}{Style.RESET_ALL}")
            logger.exception(f"Comparison error for {feature1} and {feature2}")
        finally:
            # Restore path
            self.current_path = saved_path
    
    def _get_feature_data(self, feature_path):
        """Extract data for a feature path."""
        try:
            # Extract data
            if '/' in feature_path:
                parent_path, key = feature_path.rsplit('/', 1)
                self._change_path(parent_path)
                current = self._get_current_dict()
                
                if key not in current:
                    print(f"{Fore.RED}Feature not found: {key}{Style.RESET_ALL}")
                    return None
                    
                return current[key]
            else:
                current = self._get_current_dict()
                
                if feature_path not in current:
                    print(f"{Fore.RED}Feature not found: {feature_path}{Style.RESET_ALL}")
                    return None
                    
                return current[feature_path]
        except ValueError as e:
            print(f"{Fore.RED}{str(e)}{Style.RESET_ALL}")
            return None
    
    def complete_compare(self, text, line, begidx, endidx):
        """Tab completion for compare command."""
        args = line.split()
        
        if len(args) == 1 or (len(args) == 2 and not text):
            # Complete first feature
            return self.complete_cd(text, "cd " + (args[1] if len(args) > 1 else ""), begidx, endidx)
        elif len(args) == 2 or (len(args) == 3 and not text):
            # Complete second feature
            return self.complete_cd(text, "cd " + (args[2] if len(args) > 2 else ""), begidx, endidx)
        elif len(args) > 2 and text.startswith('--'):
            # Complete option names
            options = ['--title', '--figsize', '--dpi', '--style', '--mode']
            return [opt for opt in options if opt.startswith(text)]
        elif len(args) > 2 and args[-2] == '--mode':
            # Complete mode names
            modes = ['overlay', 'side-by-side', 'difference', 'correlation']
            return [mode for mode in modes if mode.startswith(text)]
        return []
    
    def do_help(self, arg):
        """List available commands with help text."""
        if arg:
            # Show help for a specific command
            super().do_help(arg)
            return
            
        # Custom help display
        print(f"\n{Fore.GREEN}{Style.BRIGHT}Available Commands:{Style.RESET_ALL}\n")
        
        # Group commands
        navigation = ['ls', 'cd', 'pwd', 'find']
        visualization = ['visualize', 'compare']
        data = ['cat', 'export']
        other = ['help', 'exit']
        
        print(f"{Fore.CYAN}{Style.BRIGHT}Navigation:{Style.RESET_ALL}")
        for cmd in navigation:
            doc = getattr(self, f'do_{cmd}').__doc__
            if doc:
                doc = doc.split('\n')[0]
            else:
                doc = "No help available"
            print(f"  {Fore.YELLOW}{cmd.ljust(10)}{Style.RESET_ALL} {doc}")
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Visualization:{Style.RESET_ALL}")
        for cmd in visualization:
            doc = getattr(self, f'do_{cmd}').__doc__
            if doc:
                doc = doc.split('\n')[0]
            else:
                doc = "No help available"
            print(f"  {Fore.YELLOW}{cmd.ljust(10)}{Style.RESET_ALL} {doc}")
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Data Handling:{Style.RESET_ALL}")
        for cmd in data:
            doc = getattr(self, f'do_{cmd}').__doc__
            if doc:
                doc = doc.split('\n')[0]
            else:
                doc = "No help available"
            print(f"  {Fore.YELLOW}{cmd.ljust(10)}{Style.RESET_ALL} {doc}")
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Other:{Style.RESET_ALL}")
        for cmd in other:
            doc = getattr(self, f'do_{cmd}').__doc__
            if doc:
                doc = doc.split('\n')[0]
            else:
                doc = "No help available"
            print(f"  {Fore.YELLOW}{cmd.ljust(10)}{Style.RESET_ALL} {doc}")
        
        print(f"\nType {Fore.GREEN}'help <command>'{Style.RESET_ALL} for detailed help on a specific command.")
        print(f"Press {Fore.GREEN}Tab{Style.RESET_ALL} to use command completion.\n")
    
    def do_exit(self, arg):
        """Exit the interactive shell."""
        return True
    
    def do_quit(self, arg):
        """Alias for exit."""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl+D."""
        print()  # Add a newline
        return True


def start_interactive_mode(results, result_file=None):
    """Start the interactive shell with the analysis results.
    
    Args:
        results: Analysis results dictionary
        result_file: Optional path to the results file
    """
    interactive = HeihachiInteractive(results, result_file)
    interactive.cmdloop()
    return interactive


if __name__ == "__main__":
    # For testing standalone
    import sys
    
    if len(sys.argv) > 1:
        result_file = Path(sys.argv[1])
        if result_file.exists():
            with open(result_file, 'r') as f:
                try:
                    results = json.load(f)
                    start_interactive_mode(results, result_file)
                except json.JSONDecodeError:
                    print(f"{Fore.RED}Error: Invalid JSON file{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Error: File not found: {result_file}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Error: Result file required{Style.RESET_ALL}")
        print(f"Usage: {sys.argv[0]} <result_file>") 