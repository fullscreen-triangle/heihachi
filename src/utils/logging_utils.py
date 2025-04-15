import logging
import os
import sys
import time
from typing import Dict, Optional
import psutil
import threading
from datetime import datetime

# Create a global logger
logger = logging.getLogger("heihachi")

class MemoryMonitor(threading.Thread):
    """Thread that monitors memory usage during processing."""
    def __init__(self, interval=1.0, log_to_file=True):
        super().__init__()
        self.daemon = True  # Thread will exit when main thread exits
        self.interval = interval
        self.running = False
        self.log_to_file = log_to_file
        self.peak_memory = 0
        self._start_time = None
        self._logger = logging.getLogger("heihachi.memory")
        
    def run(self):
        self._start_time = time.time()
        self.running = True
        while self.running:
            self._log_memory_usage()
            time.sleep(self.interval)
    
    def stop(self):
        self.running = False
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self._logger.info(f"Memory monitoring stopped. Peak usage: {self.peak_memory:.2f} MB over {elapsed:.1f} seconds")
    
    def _log_memory_usage(self):
        # Get current process memory usage
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        
        # Update peak memory
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        # Log every 5 seconds or if there's a significant change
        elapsed = time.time() - self._start_time
        if elapsed % 5 < self.interval:
            self._logger.debug(f"Memory usage: {memory_mb:.2f} MB (Peak: {self.peak_memory:.2f} MB)")

class LogFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m',  # Red
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        # Check if stdout is a terminal
        is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
        
        log_fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        if is_tty:
            # Add colors if output is a terminal
            log_fmt = f'%(asctime)s [{self.COLORS[record.levelname]}%(levelname)s{self.COLORS["RESET"]}] %(name)s: %(message)s'
            
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logging(config: Optional[Dict] = None) -> logging.Logger:
    """Set up logging for the application.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        The configured logger
    """
    if config is None:
        config = {}
    
    # Get log level from config or default to INFO
    log_level = config.get('log_level', 'INFO').upper()
    
    # Use simple relative path for logs
    log_dir = "../logs"
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'heihachi_{timestamp}.log')
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(LogFormatter())
    console_handler.setLevel(getattr(logging, log_level))
    root_logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    root_logger.addHandler(file_handler)
    
    # Configure heihachi logger
    logger.setLevel(getattr(logging, log_level))
    logger.info(f"Logging initialized (level: {log_level}, file: {log_file})")
    
    return logger
    
def start_memory_monitoring(interval: float = 1.0) -> MemoryMonitor:
    """Start monitoring memory usage in a background thread.
    
    Args:
        interval: Monitoring interval in seconds
        
    Returns:
        The memory monitor thread
    """
    monitor = MemoryMonitor(interval=interval)
    monitor.start()
    logger.debug("Memory monitoring started")
    return monitor

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger for the specified module.
    
    Args:
        name: Module name for the logger
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)