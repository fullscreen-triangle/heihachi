#!/usr/bin/env python3
"""
Progress bar utilities for long-running operations.

This module provides functions and classes for displaying progress bars
in both CLI and interactive contexts.
"""

import sys
import time
import uuid
from typing import Optional, Union, Any, Iterator, Iterable, Dict, List, Callable
from contextlib import contextmanager

from tqdm import tqdm
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, SpinnerColumn

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def cli_progress_bar(iterable: Optional[Iterable] = None, 
                    total: Optional[int] = None,
                    desc: str = "",
                    unit: str = "it",
                    leave: bool = True,
                    **kwargs) -> tqdm:
    """Create a simple progress bar for CLI operations using tqdm.
    
    Args:
        iterable: Optional iterable to wrap
        total: Total number of iterations
        desc: Description text
        unit: Unit name
        leave: Whether to leave the progress bar after completion
        **kwargs: Additional arguments for tqdm
        
    Returns:
        tqdm progress bar
    """
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        **kwargs
    )


@contextmanager
def progress_context(description: str = "Processing",
                    total: int = 100,
                    show_eta: bool = True) -> Iterator[Callable[[int, Optional[str]], None]]:
    """Context manager for progress reporting.
    
    Args:
        description: Operation description
        total: Total steps
        show_eta: Whether to show estimated time
        
    Yields:
        Update function that accepts current step and optional status message
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn() if show_eta else None,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        
        def update_progress(step: int, status: Optional[str] = None):
            if status:
                progress.update(task_id, advance=step, description=f"[bold blue]{description}: {status}")
            else:
                progress.update(task_id, advance=step)
        
        try:
            yield update_progress
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            raise


class ProgressManager:
    """Manager for handling multiple progress bars and operation statuses."""
    
    def __init__(self, enable: bool = True, quiet: bool = False):
        """Initialize progress manager.
        
        Args:
            enable: Whether to enable progress reporting
            quiet: If True, suppresses all output
        """
        self.enable = enable
        self.quiet = quiet
        self.bars = {}
        self.progress = None
        
    def start_operation(self, operation_id: str, description: str, total: int = 100) -> None:
        """Start tracking a new operation.
        
        Args:
            operation_id: Unique identifier for the operation
            description: Description of the operation
            total: Total steps in the operation
        """
        if not self.enable or self.quiet:
            return
            
        if not self.progress:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            self.progress.start()
            
        task_id = self.progress.add_task(description, total=total)
        self.bars[operation_id] = task_id
    
    def update(self, operation_id: str, advance: int = 1, status: Optional[str] = None) -> None:
        """Update progress for an operation.
        
        Args:
            operation_id: Operation identifier
            advance: How many steps to advance
            status: Optional status message to display
        """
        if not self.enable or self.quiet or not self.progress:
            return
            
        if operation_id in self.bars:
            task_id = self.bars[operation_id]
            if status:
                # Get the current description and update it
                current_desc = self.progress.tasks[task_id].description
                base_desc = current_desc.split(':')[0] if ':' in current_desc else current_desc
                self.progress.update(task_id, advance=advance, description=f"{base_desc}: {status}")
            else:
                self.progress.update(task_id, advance=advance)
    
    def complete(self, operation_id: str, message: Optional[str] = None) -> None:
        """Mark an operation as complete.
        
        Args:
            operation_id: Operation identifier
            message: Optional completion message
        """
        if not self.enable or self.quiet or not self.progress:
            return
            
        if operation_id in self.bars:
            task_id = self.bars[operation_id]
            if message:
                self.progress.update(task_id, completed=100, description=f"✓ {message}")
            else:
                self.progress.update(task_id, completed=100)
    
    def error(self, operation_id: str, message: str) -> None:
        """Mark an operation as failed.
        
        Args:
            operation_id: Operation identifier
            message: Error message
        """
        if not self.enable or self.quiet or not self.progress:
            return
            
        if operation_id in self.bars:
            task_id = self.bars[operation_id]
            self.progress.update(task_id, description=f"✗ Error: {message}")
    
    def stop_all(self) -> None:
        """Stop all progress reporting."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.bars = {}
    
    def __enter__(self) -> 'ProgressManager':
        """Context manager enter method."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit method."""
        self.stop_all()


# Simple progress indicator for cases when tqdm or rich aren't suitable
class SimpleProgress:
    """Simple progress indicator for basic console output."""
    
    def __init__(self, total: int = 100, width: int = 40, 
                desc: str = "Progress", output=sys.stdout,
                show_percentage: bool = True):
        """Initialize simple progress bar.
        
        Args:
            total: Total number of steps
            width: Bar width in characters
            desc: Description to display
            output: Output stream
            show_percentage: Whether to display percentage
        """
        self.total = max(1, total)  # Ensure total is at least 1
        self.width = width
        self.desc = desc
        self.output = output
        self.show_percentage = show_percentage
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, n: int = 1) -> None:
        """Update progress by n steps.
        
        Args:
            n: Number of steps to advance
        """
        self.current = min(self.total, self.current + n)
        
        # Throttle updates (max 10 per second)
        current_time = time.time()
        if current_time - self.last_update < 0.1 and self.current < self.total:
            return
            
        self.last_update = current_time
        
        # Calculate percentage and bar
        percentage = 100.0 * self.current / self.total
        filled_width = int(self.width * self.current / self.total)
        bar = '=' * filled_width + '>' + ' ' * (self.width - filled_width - 1)
        
        # Calculate elapsed time
        elapsed = current_time - self.start_time
        
        # Format output
        if self.show_percentage:
            output_text = f"\r{self.desc}: [{bar}] {percentage:.1f}% ({self.current}/{self.total}) [{elapsed:.1f}s]"
        else:
            output_text = f"\r{self.desc}: [{bar}] ({self.current}/{self.total}) [{elapsed:.1f}s]"
        
        # Write to output
        self.output.write(output_text)
        self.output.flush()
        
        # Print new line when complete
        if self.current >= self.total:
            self.output.write("\n")
    
    def finish(self) -> None:
        """Mark progress as complete."""
        if self.current < self.total:
            self.current = self.total
            self.update(0)  # Force display update 


# Global ProgressTracker for the simplified progress tracking API
class ProgressTracker:
    """Simplified progress tracking for operations across function calls.
    
    This class provides a simpler interface for tracking progress compared to
    ProgressManager, with a focus on ease of use across different functions.
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        self.operations = {}
        self.progress = None
        self.enable = True
    
    def start(self, description: str, total: int = 100) -> str:
        """Start tracking a new operation.
        
        Args:
            description: Description of the operation
            total: Total steps in the operation
            
        Returns:
            Operation ID that can be used for updating progress
        """
        if not self.enable:
            return ""
            
        # Create a unique ID for this operation
        operation_id = str(uuid.uuid4())
        
        # Initialize progress display if needed
        if not self.progress:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
            )
            self.progress.start()
        
        # Create a new task
        task_id = self.progress.add_task(description, total=total)
        self.operations[operation_id] = {
            "task_id": task_id,
            "description": description,
            "total": total
        }
        
        return operation_id
    
    def update(self, operation_id: str, advance: int = 1, status: Optional[str] = None) -> None:
        """Update progress for an operation.
        
        Args:
            operation_id: Operation identifier
            advance: How many steps to advance
            status: Optional status message to display
        """
        if not self.enable or not self.progress or not operation_id or operation_id not in self.operations:
            return
            
        task_id = self.operations[operation_id]["task_id"]
        if status:
            base_desc = self.operations[operation_id]["description"]
            self.progress.update(task_id, advance=advance, description=f"[bold blue]{base_desc}: {status}")
        else:
            self.progress.update(task_id, advance=advance)
    
    def complete(self, operation_id: str, message: Optional[str] = None) -> None:
        """Mark an operation as complete.
        
        Args:
            operation_id: Operation identifier
            message: Optional completion message
        """
        if not self.enable or not self.progress or not operation_id or operation_id not in self.operations:
            return
            
        task_id = self.operations[operation_id]["task_id"]
        if message:
            self.progress.update(task_id, completed=100, description=f"✓ {message}")
        else:
            self.progress.update(task_id, completed=100)
            
        # Remove from tracked operations
        del self.operations[operation_id]
    
    def error(self, operation_id: str, message: str) -> None:
        """Mark an operation as failed.
        
        Args:
            operation_id: Operation identifier
            message: Error message
        """
        if not self.enable or not self.progress or not operation_id or operation_id not in self.operations:
            return
            
        task_id = self.operations[operation_id]["task_id"]
        self.progress.update(task_id, description=f"✗ Error: {message}")
        
        # Remove from tracked operations
        del self.operations[operation_id]
    
    def disable(self) -> None:
        """Disable progress reporting."""
        self.enable = False
    
    def enable_tracking(self) -> None:
        """Enable progress reporting."""
        self.enable = True
    
    def stop_all(self) -> None:
        """Stop all progress reporting."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.operations = {}


# Create a global instance for use with the functional API
_tracker = ProgressTracker()

def track_progress(description: str, total: int = 100) -> str:
    """Start tracking progress for a new operation.
    
    This is a simplified interface for tracking progress across functions.
    
    Args:
        description: Description of the operation
        total: Total steps in the operation
        
    Returns:
        Operation ID that can be used with update_progress and complete_progress
    """
    return _tracker.start(description, total)

def update_progress(operation_id: str, advance: int = 1, status: Optional[str] = None) -> None:
    """Update progress for an operation.
    
    Args:
        operation_id: Operation identifier from track_progress
        advance: How many steps to advance
        status: Optional status message to display
    """
    _tracker.update(operation_id, advance, status)

def complete_progress(operation_id: str, message: Optional[str] = None) -> None:
    """Mark an operation as complete.
    
    Args:
        operation_id: Operation identifier from track_progress
        message: Optional completion message
    """
    _tracker.complete(operation_id, message) 