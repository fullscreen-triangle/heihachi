#!/usr/bin/env python3
"""
Interactive demo script for Heihachi audio analysis.

This script demonstrates how to use the interactive explorer and progress utilities
for audio analysis. It guides users through a demonstration of the framework's capabilities.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import subprocess
from typing import List, Dict, Any, Optional, Union

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from src.utils.progress import ProgressManager, progress_context, SimpleProgress
from src.utils.file_utils import ensure_directory, find_files
from src.utils.config import load_default_config
from src.core.pipeline import Pipeline


console = Console()


def check_environment() -> bool:
    """Check if the environment is properly set up.
    
    Returns:
        True if environment is ready, False otherwise
    """
    # Check for sample audio files
    sample_dir = Path(project_root) / "samples"
    if not sample_dir.exists() or not list(sample_dir.glob("*.wav")):
        console.print("[yellow]No sample audio files found in 'samples' directory.[/yellow]")
        console.print("Would you like to create sample directory and download example files?")
        
        if Confirm.ask("Download sample files?"):
            ensure_directory(sample_dir)
            
            try:
                with progress_context("Downloading samples", total=100) as update_progress:
                    update_progress(10, "Preparing download")
                    
                    # Here you would implement actual download logic
                    # For demo purposes, we're just simulating with sleep
                    time.sleep(1)
                    update_progress(30, "Downloading files")
                    time.sleep(1)
                    update_progress(60, "Extracting samples")
                    time.sleep(1)
                    update_progress(100, "Complete")
                
                # Create dummy sample if we don't actually download
                if not list(sample_dir.glob("*.wav")):
                    console.print("[yellow]Creating dummy sample file for demonstration...[/yellow]")
                    sample_file = sample_dir / "test_sample.wav"
                    with open(sample_file, "wb") as f:
                        # Write a minimal valid WAV header
                        f.write(b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x44\xac\x00\x00\x88\x58\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
            except Exception as e:
                console.print(f"[red]Error setting up samples: {e}[/red]")
                return False
                
        else:
            console.print("[yellow]Sample files are needed to run the demo.[/yellow]")
            return False
    
    # Check for results directory
    results_dir = Path(project_root) / "results"
    ensure_directory(results_dir)
    
    # Check for visualizations directory
    vis_dir = Path(project_root) / "visualizations"
    ensure_directory(vis_dir)
    
    return True


def run_demo_processing() -> bool:
    """Run a demo processing on sample files to generate results.
    
    Returns:
        True if processing was successful, False otherwise
    """
    sample_dir = Path(project_root) / "samples"
    results_dir = Path(project_root) / "results"
    
    sample_files = find_files(sample_dir, "*.wav")
    if not sample_files:
        console.print("[red]No sample WAV files found.[/red]")
        return False
    
    console.print(Panel.fit(
        f"Found {len(sample_files)} sample files to process.\n"
        f"Will generate analysis results for demonstration.",
        title="Demo Processing",
        border_style="green"
    ))
    
    # Process files with progress tracking
    try:
        config = load_default_config()
        
        with ProgressManager() as progress:
            for i, file_path in enumerate(sample_files):
                operation_id = f"process_{i}"
                filename = Path(file_path).name
                
                progress.start_operation(
                    operation_id=operation_id,
                    description=f"Processing {filename}",
                    total=100
                )
                
                # Create pipeline
                pipeline = Pipeline(config)
                
                try:
                    # Process file
                    progress.update(operation_id, 10, "Loading audio")
                    time.sleep(0.5)  # Simulate processing
                    
                    progress.update(operation_id, 20, "Analyzing audio")
                    result = pipeline.process(file_path)
                    time.sleep(0.5)  # Simulate processing
                    
                    progress.update(operation_id, 70, "Generating visualizations")
                    time.sleep(0.5)  # Simulate processing
                    
                    # Save result
                    output_file = results_dir / f"{Path(file_path).stem}_analysis.json"
                    progress.update(operation_id, 90, "Saving results")
                    
                    with open(output_file, "w") as f:
                        import json
                        json.dump(result, f, indent=2)
                    
                    progress.complete(operation_id, f"Completed {filename}")
                    
                except Exception as e:
                    progress.error(operation_id, f"Error: {e}")
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error in demo processing: {e}[/red]")
        return False


def launch_interactive_explorer() -> None:
    """Launch the interactive explorer."""
    console.print("\n[green]Launching interactive explorer...[/green]")
    
    cmd = [
        sys.executable,
        "-m",
        "src.cli.interactive_cli",
        "--results-dir", "results",
        "--vis-dir", "visualizations"
    ]
    
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        console.print(f"[red]Error launching explorer: {e}[/red]")


def show_demonstration() -> None:
    """Show various progress bar demonstrations."""
    console.print(Panel.fit(
        "This demo will showcase different progress indicators available in Heihachi.",
        title="Progress Utilities Demo",
        border_style="blue"
    ))
    
    # Demonstrate simple progress bar
    console.print("\n[bold]Simple Progress Bar[/bold]")
    simple_progress = SimpleProgress(total=100, desc="Simple progress")
    for i in range(101):
        simple_progress.update(1)
        time.sleep(0.01)
    
    # Demonstrate progress context
    console.print("\n[bold]Progress Context (Rich)[/bold]")
    with progress_context("Processing stages", total=100) as update_progress:
        for i in range(5):
            stage_name = f"Stage {i+1}/5"
            update_progress(5, stage_name)
            time.sleep(0.5)
            
            # Sub-progress
            for j in range(15):
                update_progress(1, f"{stage_name} - Step {j+1}")
                time.sleep(0.05)
    
    # Demonstrate progress manager for multiple operations
    console.print("\n[bold]Progress Manager (Multiple Operations)[/bold]")
    with ProgressManager() as progress:
        # Start multiple operations
        progress.start_operation("op1", "Operation 1", 100)
        progress.start_operation("op2", "Operation 2", 100)
        progress.start_operation("op3", "Operation 3", 100)
        
        # Update operations with different speeds
        for i in range(100):
            if i < 100:
                progress.update("op1", 1, f"Step {i+1}")
            if i % 2 == 0 and i < 100:
                progress.update("op2", 2, f"Step {i//2+1}")
            if i % 4 == 0 and i < 100:
                progress.update("op3", 4, f"Step {i//4+1}")
            time.sleep(0.05)
        
        # Complete operations
        progress.complete("op1", "Operation 1 completed")
        progress.complete("op2", "Operation 2 completed")
        progress.complete("op3", "Operation 3 completed")
    
    console.print("\n[green]Progress demonstration completed![/green]")


def main() -> int:
    """Main entry point for the demo script.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(description="Interactive demo for Heihachi audio analysis")
    parser.add_argument("--skip-processing", action="store_true", 
                      help="Skip sample processing and use existing results")
    parser.add_argument("--progress-demo", action="store_true",
                      help="Show progress indicators demonstration")
    
    args = parser.parse_args()
    
    try:
        # Show welcome message
        console.print(Panel.fit(
            "[bold blue]Welcome to the Heihachi Interactive Demo![/bold blue]\n\n"
            "This script will guide you through a demonstration of the framework's\n"
            "interactive exploration capabilities and progress utilities.",
            border_style="green"
        ))
        
        # Check environment
        if not check_environment():
            console.print("[red]Environment check failed. Please fix the issues and try again.[/red]")
            return 1
        
        # If requested, show progress demo
        if args.progress_demo:
            show_demonstration()
            return 0
        
        # Run processing unless skipped
        if not args.skip_processing:
            if not run_demo_processing():
                console.print("[red]Demo processing failed. Cannot continue.[/red]")
                return 1
        
        # Launch interactive explorer
        launch_choice = Prompt.ask(
            "Choose interface to launch",
            choices=["cli", "web", "skip"],
            default="cli"
        )
        
        if launch_choice == "cli":
            launch_interactive_explorer()
        elif launch_choice == "web":
            console.print("\n[green]Launching web UI...[/green]")
            
            cmd = [
                sys.executable,
                "-m",
                "src.cli.interactive_cli",
                "--web",
                "--results-dir", "results"
            ]
            
            try:
                subprocess.run(cmd, cwd=project_root)
            except KeyboardInterrupt:
                pass
            except Exception as e:
                console.print(f"[red]Error launching web UI: {e}[/red]")
        
        console.print("\n[bold green]Demo completed successfully![/bold green]")
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error in demo: {e}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 