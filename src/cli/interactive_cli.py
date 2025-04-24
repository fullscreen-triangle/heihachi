#!/usr/bin/env python3
"""
Command-line interface for launching the interactive audio analysis explorer.

This module provides command-line entry points for launching different
interactive exploration tools for audio analysis results.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from rich.console import Console
from rich.panel import Panel

from src.utils.logging_utils import get_logger, configure_logging
from src.interactive.explorer import start_explorer
from src.utils.web_ui import start_web_ui, is_running
from src.utils.file_utils import ensure_directory, load_json

logger = get_logger(__name__)
console = Console()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the interactive CLI.
    
    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Interactive audio analysis explorer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start the CLI explorer with default results directory
  python -m src.cli.interactive_cli

  # Start the CLI explorer with a specific results directory
  python -m src.cli.interactive_cli --results-dir /path/to/results

  # Start the web UI with a specific result file
  python -m src.cli.interactive_cli --web --result-file results/analysis.json

  # Start the web UI on a specific port
  python -m src.cli.interactive_cli --web --port 8080
"""
    )
    
    parser.add_argument("--results-dir", type=str, default="results",
                      help="Directory containing analysis results (default: results)")
    
    parser.add_argument("--vis-dir", type=str, default="visualizations",
                      help="Directory to save visualizations (default: visualizations)")
    
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      default="INFO", help="Set logging level (default: INFO)")
    
    # Web UI arguments
    web_group = parser.add_argument_group("Web UI Options")
    web_group.add_argument("--web", action="store_true",
                         help="Start web UI instead of CLI explorer")
    
    web_group.add_argument("--host", type=str, default="localhost",
                         help="Host to bind the web server to (default: localhost)")
    
    web_group.add_argument("--port", type=int, default=5000,
                         help="Port to bind the web server to (default: 5000)")
    
    web_group.add_argument("--result-file", type=str,
                         help="Specific result file to load in web UI")
    
    web_group.add_argument("--no-browser", action="store_true",
                         help="Don't automatically open browser")
    
    return parser


def launch_cli_explorer(args: argparse.Namespace) -> int:
    """Launch the command-line interactive explorer.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        # Check if results directory exists
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            console.print(f"[yellow]Results directory not found: {results_dir}[/yellow]")
            if console.input("Create directory? [y/N]: ").lower() == "y":
                ensure_directory(results_dir)
            else:
                return 1
        
        # Ensure visualization directory exists
        vis_dir = Path(args.vis_dir)
        ensure_directory(vis_dir)
        
        # Start explorer
        console.print(Panel.fit(
            f"Starting interactive explorer\n\n"
            f"Results directory: [blue]{results_dir}[/blue]\n"
            f"Visualization directory: [blue]{vis_dir}[/blue]",
            title="Heihachi Interactive Explorer",
            border_style="green"
        ))
        
        start_explorer(results_dir=str(results_dir), visualization_dir=str(vis_dir))
        return 0
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Explorer terminated by user.[/yellow]")
        return 0
    except Exception as e:
        logger.error(f"Error launching CLI explorer: {e}")
        console.print(f"[red]Error launching explorer: {e}[/red]")
        return 1


def launch_web_ui(args: argparse.Namespace) -> int:
    """Launch the web UI.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Exit code (0 for success)
    """
    try:
        # If result file is specified, check if it exists
        result_file = args.result_file
        if result_file:
            result_path = Path(result_file)
            if not result_path.exists():
                console.print(f"[red]Result file not found: {result_path}[/red]")
                return 1
            
            # Test if the file is a valid JSON file with expected structure
            try:
                data = load_json(result_path)
                if not isinstance(data, dict):
                    console.print(f"[red]Invalid result file format. Expected a JSON object.[/red]")
                    return 1
            except Exception as e:
                console.print(f"[red]Failed to parse result file: {e}[/red]")
                return 1
        
        # Check if web UI is already running
        if is_running():
            console.print("[yellow]Web UI appears to be already running.[/yellow]")
            if console.input("Force start new instance? [y/N]: ").lower() != "y":
                console.print("Aborting launch. Use the existing Web UI instance.")
                return 0
        
        # Start web UI
        host = args.host
        port = args.port
        open_browser = not args.no_browser
        
        console.print(Panel.fit(
            f"Starting web UI\n\n"
            f"Host: [blue]{host}[/blue]\n"
            f"Port: [blue]{port}[/blue]\n"
            f"Result file: [blue]{result_file or 'None'}[/blue]",
            title="Heihachi Web UI",
            border_style="green"
        ))
        
        success = start_web_ui(
            result_file=result_file,
            host=host,
            port=port,
            open_browser=open_browser
        )
        
        if not success:
            console.print("[red]Failed to start Web UI.[/red]")
            return 1
        
        return 0
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Web UI launch terminated by user.[/yellow]")
        return 0
    except Exception as e:
        logger.error(f"Error launching Web UI: {e}")
        console.print(f"[red]Error launching Web UI: {e}[/red]")
        return 1


def main() -> int:
    """Main entry point for the interactive CLI.
    
    Returns:
        Exit code (0 for success)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=args.log_level)
    
    # Launch appropriate interface
    if args.web:
        return launch_web_ui(args)
    else:
        return launch_cli_explorer(args)


if __name__ == "__main__":
    sys.exit(main()) 