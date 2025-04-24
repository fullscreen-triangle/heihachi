import cProfile
import pstats
import io
import time
import functools
import os
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import psutil
import torch
import numpy as np
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class Profiler:
    """Performance profiler for identifying bottlenecks in the application."""
    
    def __init__(self, output_dir: str = "../profiling"):
        """Initialize profiler with output directory.
        
        Args:
            output_dir: Directory to store profiling results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.active_profilers: Dict[str, cProfile.Profile] = {}
        self.execution_times: Dict[str, List[float]] = {}
        self.memory_usage: Dict[str, List[float]] = {}
        
        logger.info(f"Profiler initialized with output directory: {self.output_dir}")

    def start(self, name: str) -> None:
        """Start profiling a section of code.
        
        Args:
            name: Name of the profiled section
        """
        if name in self.active_profilers:
            logger.warning(f"Profiler '{name}' already running")
            return
            
        profiler = cProfile.Profile()
        profiler.enable()
        self.active_profilers[name] = profiler
        
        # Initialize time and memory tracking if needed
        if name not in self.execution_times:
            self.execution_times[name] = []
        if name not in self.memory_usage:
            self.memory_usage[name] = []
            
        # Store initial memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        self.memory_usage[name].append(memory_info.rss / (1024 * 1024))
        
        logger.debug(f"Started profiling '{name}'")

    def stop(self, name: str) -> None:
        """Stop profiling a section of code and save results.
        
        Args:
            name: Name of the profiled section
        """
        if name not in self.active_profilers:
            logger.warning(f"No active profiler named '{name}'")
            return
            
        profiler = self.active_profilers.pop(name)
        profiler.disable()
        
        # Store final memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        final_memory = memory_info.rss / (1024 * 1024)
        self.memory_usage[name].append(final_memory)
        
        # Calculate memory difference
        memory_diff = self.memory_usage[name][-1] - self.memory_usage[name][0]
        
        # Save profiling results
        output_file = self.output_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.prof"
        profiler.dump_stats(str(output_file))
        
        # Generate a readable report
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        stats.print_stats(20)  # Print top 20 functions
        
        report_file = self.output_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w') as f:
            f.write(s.getvalue())
            f.write(f"\n\nMemory usage: {memory_diff:.2f} MB\n")
        
        logger.info(f"Profiling for '{name}' complete. Results saved to {output_file}")
        logger.info(f"Memory change during '{name}': {memory_diff:.2f} MB")

    def profile_function(self, name: Optional[str] = None):
        """Decorator to profile a function.
        
        Args:
            name: Optional name for the profile, defaults to function name
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                profile_name = name or func.__name__
                start_time = time.time()
                self.start(profile_name)
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.stop(profile_name)
                    end_time = time.time()
                    duration = end_time - start_time
                    self.execution_times[profile_name].append(duration)
                    logger.info(f"Function '{profile_name}' took {duration:.4f} seconds to execute")
            return wrapper
        return decorator

    def generate_summary(self, output_file: str = "profile_summary.txt") -> None:
        """Generate summary of all profiling results.
        
        Args:
            output_file: Name of the summary file
        """
        output_path = self.output_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("=== Heihachi Performance Profile Summary ===\n\n")
            
            f.write("== Execution Times ==\n")
            for name, times in self.execution_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    f.write(f"{name}:\n")
                    f.write(f"  Calls: {len(times)}\n")
                    f.write(f"  Average: {avg_time:.4f}s\n")
                    f.write(f"  Min: {min_time:.4f}s\n")
                    f.write(f"  Max: {max_time:.4f}s\n\n")
            
            f.write("== Memory Usage ==\n")
            for name, usage in self.memory_usage.items():
                if len(usage) >= 2:
                    memory_diff = usage[-1] - usage[0]
                    f.write(f"{name}: {memory_diff:.2f} MB\n")
            
        logger.info(f"Profiling summary saved to {output_path}")


# Global instance for easy access
global_profiler = Profiler()

def profile(name: Optional[str] = None):
    """Decorator to profile a function using the global profiler.
    
    Args:
        name: Optional name for the profile, defaults to function name
        
    Returns:
        Decorated function
    """
    return global_profiler.profile_function(name) 