#!/usr/bin/env python3
"""
Export utilities for Heihachi analysis results.

This module provides functions for exporting analysis results
to various formats like JSON, CSV, YAML, etc.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def export_to_json(data: Any, file_path: str) -> bool:
    """Export data to JSON format.
    
    Args:
        data: Data to export
        file_path: Path to save the JSON file
        
    Returns:
        True if export was successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        logger.info(f"Data exported to JSON: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to export to JSON: {e}")
        return False

def export_to_csv(data: Any, file_path: str) -> bool:
    """Export data to CSV format.
    
    Args:
        data: Data to export (must be convertible to DataFrame)
        file_path: Path to save the CSV file
        
    Returns:
        True if export was successful, False otherwise
    """
    try:
        # Handle different data types
        if isinstance(data, dict):
            # Try to convert dict to dataframe
            # For nested dicts, flatten to single level
            flat_data = {}
            
            def flatten_dict(d, parent_key=''):
                for key, value in d.items():
                    new_key = f"{parent_key}_{key}" if parent_key else key
                    
                    if isinstance(value, dict):
                        flatten_dict(value, new_key)
                    elif isinstance(value, (list, np.ndarray)) and len(value) > 0:
                        # For arrays, serialize to JSON string to preserve in CSV
                        if isinstance(value, np.ndarray):
                            value = value.tolist()
                        flat_data[new_key] = json.dumps(value)
                    elif not isinstance(value, (dict, list, np.ndarray)):
                        # Only include scalar values
                        flat_data[new_key] = value
            
            flatten_dict(data)
            
            # Convert to DataFrame
            df = pd.DataFrame([flat_data])
        
        elif isinstance(data, (list, np.ndarray)):
            # For 1D arrays, create a single column dataframe
            if isinstance(data, np.ndarray):
                if len(data.shape) == 1:
                    df = pd.DataFrame(data, columns=['value'])
                elif len(data.shape) == 2:
                    # For 2D arrays, create a multi-column dataframe
                    if data.shape[1] < 100:  # Reasonable number of columns
                        df = pd.DataFrame(data)
                    else:
                        # Too many columns, transpose if needed
                        if data.shape[0] < data.shape[1]:
                            data = data.T
                        df = pd.DataFrame(data)
                else:
                    # For higher dimensional arrays, flatten to JSON
                    df = pd.DataFrame([{'data': json.dumps(data.tolist())}])
            else:
                # Regular list
                if all(isinstance(item, (int, float, str, bool)) for item in data):
                    df = pd.DataFrame(data, columns=['value'])
                else:
                    # List of complex objects, convert to JSON strings
                    df = pd.DataFrame([{'data': json.dumps(data)}])
        
        else:
            # Try to convert to DataFrame directly
            df = pd.DataFrame([data])
        
        # Write to CSV
        df.to_csv(file_path, index=False)
        logger.info(f"Data exported to CSV: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export to CSV: {e}")
        return False

def export_to_yaml(data: Any, file_path: str) -> bool:
    """Export data to YAML format.
    
    Args:
        data: Data to export
        file_path: Path to save the YAML file
        
    Returns:
        True if export was successful, False otherwise
    """
    if not YAML_AVAILABLE:
        logger.error("PyYAML is not installed. Cannot export to YAML.")
        return False
    
    try:
        # Convert NumPy arrays to lists
        if isinstance(data, dict):
            # Process dictionary recursively
            processed_data = {}
            
            def process_dict(d):
                result = {}
                for key, value in d.items():
                    if isinstance(value, dict):
                        result[key] = process_dict(value)
                    elif isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                    elif isinstance(value, (list, tuple)):
                        result[key] = process_list(value)
                    else:
                        result[key] = value
                return result
            
            def process_list(lst):
                result = []
                for item in lst:
                    if isinstance(item, dict):
                        result.append(process_dict(item))
                    elif isinstance(item, np.ndarray):
                        result.append(item.tolist())
                    elif isinstance(item, (list, tuple)):
                        result.append(process_list(item))
                    else:
                        result.append(item)
                return result
            
            processed_data = process_dict(data)
            
            # Write to YAML
            with open(file_path, 'w') as f:
                yaml.dump(processed_data, f, default_flow_style=False)
            
            logger.info(f"Data exported to YAML: {file_path}")
            return True
            
    except Exception as e:
        logger.error(f"Failed to export to YAML: {e}")
        return False

def export_results(data: Any, file_path: str, format: str = "json") -> bool:
    """Export analysis results to the specified format.
    
    Args:
        data: Analysis result data
        file_path: Path to save the exported file
        format: Export format (json, csv, yaml)
        
    Returns:
        True if export was successful, False otherwise
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Export based on format
    format = format.lower()
    
    if format == "json":
        return export_to_json(data, file_path)
    elif format == "csv":
        return export_to_csv(data, file_path)
    elif format == "yaml":
        return export_to_yaml(data, file_path)
    else:
        logger.error(f"Unsupported export format: {format}")
        return False

if __name__ == "__main__":
    # For testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Heihachi analysis results")
    parser.add_argument("input_file", help="Input JSON result file")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument("--format", choices=["json", "csv", "yaml"], default="json",
                        help="Export format")
    
    args = parser.parse_args()
    
    # Load input data
    try:
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        # Export to specified format
        if export_results(data, args.output_file, args.format):
            print(f"Exported to {args.output_file}")
        else:
            print("Export failed")
            
    except Exception as e:
        print(f"Error: {e}") 