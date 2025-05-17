"""
Environment variable loader for Heihachi.

This module provides utilities for loading environment variables from a .env file,
which is useful for storing sensitive information like API keys.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def load_dotenv(dotenv_path: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.
    
    Args:
        dotenv_path: Path to .env file. If None, looks for .env in current
                    directory and parent directories.
                    
    Returns:
        bool: True if .env file was loaded successfully, False otherwise
    """
    # Try to import python-dotenv (if installed)
    try:
        from dotenv import load_dotenv as dotenv_loader
        
        # If no path specified, try to find .env file
        if not dotenv_path:
            # Try current directory
            if os.path.exists(".env"):
                dotenv_path = ".env"
            # Try project root (parent of src directory)
            elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")):
                dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        
        # Load from specified or found path
        if dotenv_path and os.path.exists(dotenv_path):
            logger.info(f"Loading environment variables from {dotenv_path}")
            return dotenv_loader(dotenv_path=dotenv_path)
        else:
            logger.warning("No .env file found")
            return False
            
    except ImportError:
        # Fall back to manual implementation if python-dotenv is not installed
        logger.warning("python-dotenv not installed, using basic .env loader")
        return _load_dotenv_manual(dotenv_path)
        
def _load_dotenv_manual(dotenv_path: Optional[str] = None) -> bool:
    """
    Basic implementation of .env file loading without dependencies.
    
    Args:
        dotenv_path: Path to .env file
        
    Returns:
        bool: True if .env file was loaded successfully, False otherwise
    """
    # If no path specified, try to find .env file
    if not dotenv_path:
        # Try current directory
        if os.path.exists(".env"):
            dotenv_path = ".env"
        # Try project root (parent of src directory)
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")):
            dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    
    # Check if file exists
    if not dotenv_path or not os.path.exists(dotenv_path):
        logger.warning("No .env file found")
        return False
    
    try:
        # Read .env file and set environment variables
        with open(dotenv_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key-value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value and value[0] == value[-1] == '"':
                        value = value[1:-1]
                    elif value and value[0] == value[-1] == "'":
                        value = value[1:-1]
                    
                    # Set environment variable
                    os.environ[key] = value
        
        logger.info(f"Loaded environment variables from {dotenv_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error loading .env file: {str(e)}")
        return False

def get_api_key(key_name: str = "HF_API_TOKEN", 
                fallback_names: Optional[list] = None,
                default: Optional[str] = None) -> Optional[str]:
    """
    Get API key from environment variables with fallbacks.
    
    Args:
        key_name: Primary environment variable name to check
        fallback_names: List of fallback environment variable names
        default: Default value if no environment variable is found
        
    Returns:
        str or None: API key if found, else default value
    """
    # Try loading from .env file first
    load_dotenv()
    
    # Check primary key name
    api_key = os.environ.get(key_name)
    if api_key:
        return api_key
    
    # Try fallback names
    if fallback_names:
        for name in fallback_names:
            api_key = os.environ.get(name)
            if api_key:
                return api_key
    
    # Return default if no key found
    return default 