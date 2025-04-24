#!/usr/bin/env python3
"""
Error handler for Heihachi with improved error messages and actionable suggestions.

This module provides utility functions and classes for handling errors in a user-friendly
way, offering context-specific suggestions to resolve common issues.
"""

import os
import sys
import traceback
import logging
from typing import Any, Dict, List, Optional, Union, Type, Callable
import importlib.util

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class ErrorHandler:
    """Error handler with improved error messages and suggestions."""
    
    def __init__(self):
        """Initialize the error handler with error type mappings."""
        # Map of error types to handler functions
        self.error_handlers = {
            # File errors
            FileNotFoundError: self._handle_file_not_found,
            PermissionError: self._handle_permission_error,
            IsADirectoryError: self._handle_is_directory,
            NotADirectoryError: self._handle_not_directory,
            
            # I/O and data errors
            IOError: self._handle_io_error,
            EOFError: self._handle_eof_error,
            UnicodeError: self._handle_unicode_error,
            json.JSONDecodeError: self._handle_json_decode,
            
            # Value and type errors
            ValueError: self._handle_value_error,
            TypeError: self._handle_type_error,
            IndexError: self._handle_index_error,
            KeyError: self._handle_key_error,
            AttributeError: self._handle_attribute_error,
            
            # Import and dependency errors
            ImportError: self._handle_import_error,
            ModuleNotFoundError: self._handle_module_not_found,
            
            # Memory errors
            MemoryError: self._handle_memory_error,
            
            # Runtime and OS errors
            RuntimeError: self._handle_runtime_error,
            OSError: self._handle_os_error,
            TimeoutError: self._handle_timeout_error,
            
            # Domain-specific errors
            "AudioLoadError": self._handle_audio_load_error,
            "ConfigurationError": self._handle_configuration_error,
            "ProcessingError": self._handle_processing_error,
            "VisualizationError": self._handle_visualization_error,
            "ParallelProcessingError": self._handle_parallel_processing_error
        }
        
        # Error context for additional information
        self.error_context = {}
    
    def set_context(self, key: str, value: Any) -> None:
        """Set context information to enhance error handling.
        
        Args:
            key: Context key
            value: Context value
        """
        self.error_context[key] = value
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle an error and provide actionable suggestions.
        
        Args:
            error: The exception to handle
            context: Additional context information
            
        Returns:
            Dictionary with error details and suggestion
        """
        error_type = type(error)
        error_str = str(error)
        error_name = error_type.__name__
        
        logger.debug(f"Handling error: {error_name}: {error_str}")
        
        # Update context with provided information
        if context:
            for key, value in context.items():
                self.error_context[key] = value
        
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Find the appropriate handler
        handler = None
        if error_type in self.error_handlers:
            handler = self.error_handlers[error_type]
        elif error_name in self.error_handlers:
            handler = self.error_handlers[error_name]
        else:
            # Try to find a parent class handler
            for err_type, err_handler in self.error_handlers.items():
                if isinstance(err_type, type) and isinstance(error, err_type):
                    handler = err_handler
                    break
        
        # If handler found, get suggestion
        suggestion = "No specific suggestion available. Check logs for more details."
        if handler:
            handler_suggestion = handler(error)
            if handler_suggestion:
                suggestion = handler_suggestion
        
        # Build error details
        error_details = {
            'type': error_name,
            'message': error_str,
            'suggestion': suggestion,
            'trace': stack_trace
        }
        
        # Log the error
        logger.error(f"{error_name}: {error_str}")
        logger.error(f"Suggestion: {suggestion}")
        
        # Clear context
        self.error_context = {}
        
        return error_details
    
    # File error handlers
    
    def _handle_file_not_found(self, error: FileNotFoundError) -> str:
        """Handle file not found errors.
        
        Args:
            error: The FileNotFoundError
            
        Returns:
            Suggestion for resolving the error
        """
        file_path = str(error).split("'")[1] if "'" in str(error) else "unknown file"
        
        # Check if it's an audio file
        audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.ogg']
        if any(file_path.lower().endswith(ext) for ext in audio_extensions):
            return (f"The audio file '{os.path.basename(file_path)}' could not be found. "
                   f"Ensure the file exists and the path is correct. "
                   f"If using a relative path, check your current working directory.")
        
        # Check if it's a configuration file
        if file_path.lower().endswith(('.yaml', '.yml', '.json')):
            return (f"The configuration file '{os.path.basename(file_path)}' could not be found. "
                   f"Ensure the configuration file exists. Default configurations are in the "
                   f"'configs' directory.")
        
        # Generic file not found
        return (f"File '{os.path.basename(file_path)}' could not be found. "
               f"Check that the file exists and the path is correct. "
               f"If using a relative path, try using an absolute path instead.")
    
    def _handle_permission_error(self, error: PermissionError) -> str:
        """Handle permission errors.
        
        Args:
            error: The PermissionError
            
        Returns:
            Suggestion for resolving the error
        """
        file_path = str(error).split("'")[1] if "'" in str(error) else "unknown file"
        
        return (f"Permission denied when accessing '{os.path.basename(file_path)}'. "
               f"Check that you have the necessary permissions to read or write to this file. "
               f"Try running with elevated privileges if appropriate, or change the file permissions.")
    
    def _handle_is_directory(self, error: IsADirectoryError) -> str:
        """Handle is-a-directory errors.
        
        Args:
            error: The IsADirectoryError
            
        Returns:
            Suggestion for resolving the error
        """
        path = str(error).split("'")[1] if "'" in str(error) else "unknown path"
        
        return (f"'{os.path.basename(path)}' is a directory, but a file was expected. "
               f"Please specify a file path instead. "
               f"If you want to process all files in this directory, use the --batch flag.")
    
    def _handle_not_directory(self, error: NotADirectoryError) -> str:
        """Handle not-a-directory errors.
        
        Args:
            error: The NotADirectoryError
            
        Returns:
            Suggestion for resolving the error
        """
        path = str(error).split("'")[1] if "'" in str(error) else "unknown path"
        
        return (f"'{os.path.basename(path)}' is not a directory. "
               f"Please specify a directory path for batch processing. "
               f"If you want to process a single file, remove the --batch flag.")
    
    # I/O and data error handlers
    
    def _handle_io_error(self, error: IOError) -> str:
        """Handle I/O errors.
        
        Args:
            error: The IOError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "no space" in error_str:
            return "Not enough disk space. Free up some disk space and try again."
        elif "too many open" in error_str:
            return "Too many open files. Close some files or increase the open file limit."
        elif "broken pipe" in error_str:
            return "Broken pipe error. The connection to a subprocess or pipe was broken."
        
        return (f"I/O error occurred: {str(error)}. "
               f"Check disk space, file permissions, and ensure the file is not in use by another process.")
    
    def _handle_eof_error(self, error: EOFError) -> str:
        """Handle end-of-file errors.
        
        Args:
            error: The EOFError
            
        Returns:
            Suggestion for resolving the error
        """
        return "Unexpected end of file. The input file may be corrupted or truncated."
    
    def _handle_unicode_error(self, error: UnicodeError) -> str:
        """Handle Unicode errors.
        
        Args:
            error: The UnicodeError
            
        Returns:
            Suggestion for resolving the error
        """
        return (f"Unicode encoding/decoding error: {str(error)}. "
               f"The file may contain characters that cannot be properly decoded. "
               f"Try specifying the correct encoding or using binary mode.")
    
    def _handle_json_decode(self, error: Exception) -> str:
        """Handle JSON decode errors.
        
        Args:
            error: The JSONDecodeError
            
        Returns:
            Suggestion for resolving the error
        """
        return (f"Invalid JSON: {str(error)}. "
               f"Check that the JSON file is properly formatted and complete. "
               f"You can validate it using a JSON linter.")
    
    # Value and type error handlers
    
    def _handle_value_error(self, error: ValueError) -> str:
        """Handle value errors.
        
        Args:
            error: The ValueError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "sample rate" in error_str:
            return ("Invalid sample rate. Ensure the audio file has a valid sample rate. "
                   "Try resampling the file to 44100 Hz.")
        elif "shape" in error_str and "dimension" in error_str:
            return ("Array shape mismatch. This may be due to an invalid audio file format "
                   "or corrupted audio data.")
        elif "channel" in error_str:
            return ("Channel configuration error. The audio file may have an unsupported "
                   "number of channels. Try converting to mono or stereo.")
        elif "bit depth" in error_str or "bit rate" in error_str:
            return ("Unsupported bit depth or bit rate. Try converting the audio file "
                   "to a standard format like 16-bit WAV.")
        
        return (f"Invalid value: {str(error)}. "
               f"Check the input parameters and ensure they are within the expected ranges.")
    
    def _handle_type_error(self, error: TypeError) -> str:
        """Handle type errors.
        
        Args:
            error: The TypeError
            
        Returns:
            Suggestion for resolving the error
        """
        return (f"Type error: {str(error)}. "
               f"Check that you're providing the correct data types for function parameters.")
    
    def _handle_index_error(self, error: IndexError) -> str:
        """Handle index errors.
        
        Args:
            error: The IndexError
            
        Returns:
            Suggestion for resolving the error
        """
        return (f"Index error: {str(error)}. "
               f"An array index is out of bounds. Check array sizes and indices.")
    
    def _handle_key_error(self, error: KeyError) -> str:
        """Handle key errors.
        
        Args:
            error: The KeyError
            
        Returns:
            Suggestion for resolving the error
        """
        key = str(error).strip("'")
        
        if key in ['metrics', 'analysis', 'segments', 'metadata']:
            return (f"The result doesn't contain the expected '{key}' key. "
                   f"This may indicate incomplete processing or an unsupported file format.")
        
        return (f"Key error: '{key}' not found. "
               f"Check dictionary keys or configuration parameters.")
    
    def _handle_attribute_error(self, error: AttributeError) -> str:
        """Handle attribute errors.
        
        Args:
            error: The AttributeError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error)
        attribute = error_str.split("'")[1] if "'" in error_str else "unknown"
        
        return (f"Attribute error: {error_str}. "
               f"Check that the object has the attribute '{attribute}' "
               f"or that you're using the correct object type.")
    
    # Import and dependency error handlers
    
    def _handle_import_error(self, error: ImportError) -> str:
        """Handle import errors.
        
        Args:
            error: The ImportError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error)
        module = error_str.split("'")[1] if "'" in error_str else "unknown module"
        
        if module in ['numpy', 'scipy', 'librosa', 'matplotlib']:
            return (f"Could not import '{module}'. This is a required dependency. "
                   f"Install it with: pip install {module}")
        elif module in ['torch', 'tensorflow', 'numba']:
            return (f"Could not import '{module}'. This is an optional dependency for GPU acceleration. "
                   f"Install it with: pip install {module}")
        elif module in ['inquirer', 'tabulate', 'pyyaml']:
            return (f"Could not import '{module}'. This is required for the feature you're using. "
                   f"Install it with: pip install {module}")
        
        return (f"Import error: {error_str}. "
               f"Check that the required dependency is installed: pip install {module}")
    
    def _handle_module_not_found(self, error: ModuleNotFoundError) -> str:
        """Handle module not found errors.
        
        Args:
            error: The ModuleNotFoundError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error)
        module = error_str.split("'")[1] if "'" in error_str else "unknown module"
        
        # Check if it's one of our own modules
        if module.startswith('src.'):
            return (f"Could not find module '{module}'. "
                   f"This is an internal module. Make sure you're running from the correct directory "
                   f"and the project structure is intact.")
        
        # Suggest installation based on module name
        return (f"Module '{module}' not found. "
               f"Install it with: pip install {module}")
    
    # Memory error handlers
    
    def _handle_memory_error(self, error: MemoryError) -> str:
        """Handle memory errors.
        
        Args:
            error: The MemoryError
            
        Returns:
            Suggestion for resolving the error
        """
        return ("Memory error: Not enough memory to complete the operation. "
               "Try processing smaller chunks, reducing batch size, or using the --memory-limit "
               "option to set a lower memory limit. Close other applications to free up memory.")
    
    # Runtime and OS error handlers
    
    def _handle_runtime_error(self, error: RuntimeError) -> str:
        """Handle runtime errors.
        
        Args:
            error: The RuntimeError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "cuda" in error_str or "gpu" in error_str:
            return ("GPU error. Try disabling GPU acceleration with --no-gpu, "
                   "or check your GPU drivers and CUDA installation.")
        elif "thread" in error_str or "concurrent" in error_str:
            return ("Threading error. Try reducing the number of worker threads "
                   "with --workers option or disable parallel processing.")
        elif "librosa" in error_str and "load" in error_str:
            return ("Error loading audio with librosa. The file may be corrupted or in an "
                   "unsupported format. Try converting it to a WAV file.")
        
        return (f"Runtime error: {str(error)}. "
               f"This is an unexpected error during execution. Check logs for more details.")
    
    def _handle_os_error(self, error: OSError) -> str:
        """Handle OS errors.
        
        Args:
            error: The OSError
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "no such file" in error_str:
            return self._handle_file_not_found(FileNotFoundError(str(error)))
        elif "permission" in error_str:
            return self._handle_permission_error(PermissionError(str(error)))
        elif "disk" in error_str and "space" in error_str:
            return "Not enough disk space. Free up some disk space and try again."
        elif "too many open" in error_str:
            return "Too many open files. Close some files or increase the open file limit."
        
        return (f"OS error: {str(error)}. "
               f"This is an operating system-related error. Check permissions, disk space, "
               f"and system resources.")
    
    def _handle_timeout_error(self, error: TimeoutError) -> str:
        """Handle timeout errors.
        
        Args:
            error: The TimeoutError
            
        Returns:
            Suggestion for resolving the error
        """
        return ("Operation timed out. The process may be taking too long due to file size "
               "or complexity. Try increasing the timeout or processing a smaller file.")
    
    # Domain-specific error handlers
    
    def _handle_audio_load_error(self, error: Exception) -> str:
        """Handle audio loading errors.
        
        Args:
            error: The error
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "format" in error_str:
            return ("Unsupported or invalid audio format. Try converting the file to a "
                   "common format like WAV, MP3, or FLAC.")
        elif "corrupt" in error_str:
            return "The audio file appears to be corrupted. Try re-downloading or converting it."
        elif "duration" in error_str:
            return ("The audio file is too short or has an invalid duration. "
                   "Ensure the file is complete and properly encoded.")
        
        return ("Error loading audio file. Check that the file is a valid audio file "
               "in a supported format (WAV, MP3, FLAC, AIFF, OGG). If the file is "
               "valid, try converting it to WAV format.")
    
    def _handle_configuration_error(self, error: Exception) -> str:
        """Handle configuration errors.
        
        Args:
            error: The error
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "yaml" in error_str:
            return ("Invalid YAML configuration. Check the syntax of your configuration file. "
                   "Ensure indentation is consistent and all values are properly formatted.")
        elif "missing" in error_str:
            return ("Missing configuration value. Check that your configuration file "
                   "contains all required parameters.")
        
        return ("Configuration error. Check that your configuration file is valid and "
               "contains all required parameters. You can start with the default "
               "configuration in 'configs/default.yaml'.")
    
    def _handle_processing_error(self, error: Exception) -> str:
        """Handle processing errors.
        
        Args:
            error: The error
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "memory" in error_str:
            return ("Not enough memory for processing. Try using a smaller chunk size "
                   "or enable memory-mapped processing with --optimize.")
        elif "timeout" in error_str:
            return "Processing timed out. The file may be too large or complex."
        
        return ("Error during audio processing. This may be due to an invalid or "
               "unsupported audio file. Try processing a different file or check "
               "the logs for more details.")
    
    def _handle_visualization_error(self, error: Exception) -> str:
        """Handle visualization errors.
        
        Args:
            error: The error
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "matplotlib" in error_str:
            return ("Matplotlib error. This may be due to missing dependencies or "
                   "an invalid display configuration. Try installing matplotlib: "
                   "pip install matplotlib")
        
        return ("Visualization error. Check that you have the necessary dependencies "
               "installed (matplotlib, numpy) and that the data is valid for visualization.")
    
    def _handle_parallel_processing_error(self, error: Exception) -> str:
        """Handle parallel processing errors.
        
        Args:
            error: The error
            
        Returns:
            Suggestion for resolving the error
        """
        error_str = str(error).lower()
        
        if "deadlock" in error_str:
            return ("Deadlock detected in parallel processing. Try reducing the number "
                   "of worker processes or disable parallel processing.")
        elif "worker" in error_str and "crash" in error_str:
            return ("Worker process crashed. This may be due to insufficient memory "
                   "or an unhandled exception. Try reducing the number of workers.")
        
        return ("Error in parallel processing. Try reducing the number of worker "
               "processes with --workers or disable parallel processing.")


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle an error using the global error handler.
    
    Args:
        error: The exception to handle
        context: Additional context information
        
    Returns:
        Dictionary with error details and suggestion
    """
    return error_handler.handle_error(error, context)


def suggest_solution(error: Exception) -> str:
    """Get a suggested solution for an error.
    
    Args:
        error: The exception
        
    Returns:
        Suggestion string
    """
    error_details = handle_error(error)
    return error_details.get('suggestion', '')


def is_dependency_available(module_name: str) -> bool:
    """Check if a dependency is available.
    
    Args:
        module_name: Name of the module to check
        
    Returns:
        Whether the module is available
    """
    return importlib.util.find_spec(module_name) is not None


def format_error_message(error: Exception, include_suggestion: bool = True) -> str:
    """Format an error message for display.
    
    Args:
        error: The exception
        include_suggestion: Whether to include a suggestion
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    error_message = str(error)
    
    message = f"Error: {error_type}: {error_message}"
    
    if include_suggestion:
        suggestion = suggest_solution(error)
        if suggestion:
            message += f"\n\nSuggestion: {suggestion}"
    
    return message


# Custom exception classes for domain-specific errors

class AudioLoadError(Exception):
    """Error when loading audio files."""
    pass


class ConfigurationError(Exception):
    """Error in configuration files or parameters."""
    pass


class ProcessingError(Exception):
    """Error during audio processing."""
    pass


class VisualizationError(Exception):
    """Error during visualization generation."""
    pass


class ParallelProcessingError(Exception):
    """Error in parallel processing."""
    pass 