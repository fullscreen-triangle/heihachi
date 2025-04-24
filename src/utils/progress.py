import time
import sys
import threading
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timedelta

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ProgressTracker:
    """Track progress of long-running operations with ETA calculation."""
    
    def __init__(self, total: int, description: str = "", log_interval: float = 1.0, log_to_console: bool = True):
        """Initialize progress tracker with total items.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
            log_interval: Minimum interval between progress logs in seconds
            log_to_console: Whether to print progress to console
        """
        self.total = total
        self.description = description
        self.completed = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.log_interval = log_interval
        self.log_to_console = log_to_console
        self.active = False
        self._lock = threading.Lock()
        
        logger.info(f"Progress tracker initialized: {description} (total: {total})")
    
    def start(self) -> None:
        """Start or restart the progress tracker."""
        with self._lock:
            self.start_time = time.time()
            self.last_update_time = 0
            self.completed = 0
            self.active = True
        
        self._log_progress()
        logger.info(f"Progress tracker started: {self.description}")
    
    def update(self, increment: int = 1) -> None:
        """Update progress count.
        
        Args:
            increment: Number of items completed
        """
        with self._lock:
            if not self.active:
                return
                
            self.completed += increment
            current_time = time.time()
            
            # Log progress at specified intervals
            if current_time - self.last_update_time >= self.log_interval:
                self.last_update_time = current_time
                self._log_progress()
    
    def complete(self) -> None:
        """Mark the task as completed."""
        with self._lock:
            self.completed = self.total
            self.active = False
            
        self._log_progress(force=True)
        
        # Final update
        elapsed = time.time() - self.start_time
        logger.info(f"Task completed: {self.description} in {self._format_time(elapsed)}")
    
    def _log_progress(self, force: bool = False) -> None:
        """Log progress with estimated time remaining.
        
        Args:
            force: Force logging regardless of interval
        """
        if not self.active and not force:
            return
            
        # Calculate progress percentage
        if self.total > 0:
            percentage = min(100, (self.completed / self.total) * 100)
        else:
            percentage = 100
            
        # Calculate elapsed time and ETA
        elapsed = time.time() - self.start_time
        if self.completed > 0 and self.completed < self.total:
            eta = (elapsed / self.completed) * (self.total - self.completed)
        else:
            eta = 0
            
        # Log progress
        status = f"{self.description}: {percentage:.1f}% ({self.completed}/{self.total})"
        status += f" | Elapsed: {self._format_time(elapsed)}"
        
        if eta > 0:
            status += f" | ETA: {self._format_time(eta)}"
            
        logger.info(status)
        
        # Print to console if enabled
        if self.log_to_console:
            print(f"\r{status}", end="", flush=True)
            if force or self.completed >= self.total:
                print()  # Add newline at completion
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            seconds = seconds % 60
            return f"{minutes}m {int(seconds)}s"
        else:
            hours = int(seconds // 3600)
            seconds %= 3600
            minutes = int(seconds // 60)
            return f"{hours}h {minutes}m"


class ProgressReporter:
    """Progress reporter for tracking and reporting progress of multiple operations."""
    
    def __init__(self):
        """Initialize progress reporter with empty trackers."""
        self.trackers: Dict[str, ProgressTracker] = {}
        self._lock = threading.Lock()
    
    def create_tracker(self, name: str, total: int, description: str = "", 
                      log_interval: float = 1.0, log_to_console: bool = True) -> ProgressTracker:
        """Create a new progress tracker.
        
        Args:
            name: Unique name for the tracker
            total: Total number of items to process
            description: Description of the operation
            log_interval: Minimum interval between progress logs in seconds
            log_to_console: Whether to print progress to console
            
        Returns:
            The created tracker
        """
        with self._lock:
            # If a tracker with this name already exists, stop it first
            if name in self.trackers:
                self.trackers[name].active = False
                
            # Create new tracker
            tracker = ProgressTracker(total, description, log_interval, log_to_console)
            self.trackers[name] = tracker
            
        return tracker
    
    def update(self, name: str, increment: int = 1) -> None:
        """Update progress for a tracker.
        
        Args:
            name: Name of the tracker
            increment: Number of items completed
        """
        with self._lock:
            if name in self.trackers:
                self.trackers[name].update(increment)
    
    def complete(self, name: str) -> None:
        """Mark a tracker as completed.
        
        Args:
            name: Name of the tracker
        """
        with self._lock:
            if name in self.trackers:
                self.trackers[name].complete()
                
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all trackers.
        
        Returns:
            Dictionary mapping tracker names to status information
        """
        status = {}
        with self._lock:
            for name, tracker in self.trackers.items():
                if tracker.active:
                    # Calculate progress
                    percentage = (tracker.completed / tracker.total) * 100 if tracker.total > 0 else 100
                    elapsed = time.time() - tracker.start_time
                    
                    # Calculate ETA
                    if tracker.completed > 0 and tracker.completed < tracker.total:
                        eta = (elapsed / tracker.completed) * (tracker.total - tracker.completed)
                    else:
                        eta = 0
                        
                    status[name] = {
                        'description': tracker.description,
                        'completed': tracker.completed,
                        'total': tracker.total,
                        'percentage': percentage,
                        'elapsed': elapsed,
                        'eta': eta,
                        'active': tracker.active
                    }
        
        return status


# Global instance for easy access
global_progress = ProgressReporter()

def track_progress(name: str, total: int, description: str = "", 
                  log_interval: float = 1.0, log_to_console: bool = True) -> ProgressTracker:
    """Create and start a progress tracker using the global progress reporter.
    
    Args:
        name: Unique name for the tracker
        total: Total number of items to process
        description: Description of the operation
        log_interval: Minimum interval between progress logs in seconds
        log_to_console: Whether to print progress to console
        
    Returns:
        The created and started tracker
    """
    tracker = global_progress.create_tracker(name, total, description, log_interval, log_to_console)
    tracker.start()
    return tracker

def update_progress(name: str, increment: int = 1) -> None:
    """Update progress for a tracker using the global progress reporter.
    
    Args:
        name: Name of the tracker
        increment: Number of items completed
    """
    global_progress.update(name, increment)

def complete_progress(name: str) -> None:
    """Mark a tracker as completed using the global progress reporter.
    
    Args:
        name: Name of the tracker
    """
    global_progress.complete(name)

def get_all_progress() -> Dict[str, Dict[str, Any]]:
    """Get status of all trackers using the global progress reporter.
    
    Returns:
        Dictionary mapping tracker names to status information
    """
    return global_progress.get_status() 