#!/usr/bin/env python3
"""
Export utilities for Heihachi analysis results.

This module provides functions to export analysis results in various formats,
including JSON, CSV, YAML, Markdown, and HTML.
"""

import os
import json
import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

def export_results(results: Dict[str, Any], 
                 format_type: str, 
                 output_dir: Union[str, Path],
                 filename_prefix: str = "results") -> str:
    """Export results in the specified format.
    
    Args:
        results: Analysis results to export
        format_type: Format to export (json, csv, yaml, md, markdown, html, xml)
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        
    Returns:
        Path to the exported file
    """
    # Convert output_dir to Path if it's a string
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate timestamp for filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Call appropriate export function based on format
    format_type = format_type.lower()
    
    if format_type == 'json':
        return export_json(results, output_dir, filename_prefix, timestamp)
    elif format_type == 'csv':
        return export_csv(results, output_dir, filename_prefix, timestamp)
    elif format_type in ('yaml', 'yml'):
        return export_yaml(results, output_dir, filename_prefix, timestamp)
    elif format_type in ('md', 'markdown'):
        return export_markdown(results, output_dir, filename_prefix, timestamp)
    elif format_type == 'html':
        return export_html(results, output_dir, filename_prefix, timestamp)
    elif format_type == 'xml':
        return export_xml(results, output_dir, filename_prefix, timestamp)
    else:
        raise ValueError(f"Unsupported export format: {format_type}")


def export_json(results: Dict[str, Any], 
               output_dir: Path,
               filename_prefix: str,
               timestamp: str) -> str:
    """Export results as JSON.
    
    Args:
        results: Analysis results to export
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        timestamp: Timestamp string for the filename
        
    Returns:
        Path to the exported file
    """
    filepath = output_dir / f"{filename_prefix}_{timestamp}.json"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported results as JSON to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error exporting to JSON: {str(e)}")
        raise


def export_csv(results: Dict[str, Any], 
              output_dir: Path,
              filename_prefix: str,
              timestamp: str) -> str:
    """Export results as CSV.
    
    Args:
        results: Analysis results to export
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        timestamp: Timestamp string for the filename
        
    Returns:
        Path to the exported file
    """
    filepath = output_dir / f"{filename_prefix}_{timestamp}.csv"
    
    try:
        # Extract metrics and metadata to flatten
        flattened_data = flatten_dict(results)
        
        # Write to CSV
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            if flattened_data:
                fieldnames = flattened_data[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            else:
                writer = csv.writer(f)
                writer.writerow(["No data to export"])
        
        logger.info(f"Exported results as CSV to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error exporting to CSV: {str(e)}")
        raise


def export_yaml(results: Dict[str, Any], 
               output_dir: Path,
               filename_prefix: str,
               timestamp: str) -> str:
    """Export results as YAML.
    
    Args:
        results: Analysis results to export
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        timestamp: Timestamp string for the filename
        
    Returns:
        Path to the exported file
    """
    filepath = output_dir / f"{filename_prefix}_{timestamp}.yaml"
    
    try:
        # Try to import YAML library
        try:
            import yaml
        except ImportError:
            logger.error("PyYAML library not found. Install with: pip install pyyaml")
            raise ImportError("PyYAML library required for YAML export")
        
        # Export to YAML
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Exported results as YAML to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error exporting to YAML: {str(e)}")
        raise


def export_markdown(results: Dict[str, Any], 
                  output_dir: Path,
                  filename_prefix: str,
                  timestamp: str) -> str:
    """Export results as Markdown.
    
    Args:
        results: Analysis results to export
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        timestamp: Timestamp string for the filename
        
    Returns:
        Path to the exported file
    """
    filepath = output_dir / f"{filename_prefix}_{timestamp}.md"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Write title
            if 'metadata' in results and 'file_path' in results['metadata']:
                title = f"Analysis Results for {os.path.basename(results['metadata']['file_path'])}"
            else:
                title = "Heihachi Analysis Results"
            
            f.write(f"# {title}\n\n")
            
            # Write timestamp
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write metadata section
            if 'metadata' in results:
                f.write("## Metadata\n\n")
                f.write("| Property | Value |\n")
                f.write("|----------|-------|\n")
                
                for key, value in results['metadata'].items():
                    if key == 'processing_time' and isinstance(value, (int, float)):
                        value = f"{value:.2f}s"
                    f.write(f"| {key} | {value} |\n")
                
                f.write("\n")
            
            # Write metrics section
            metrics = {}
            if 'analysis' in results and 'mix' in results['analysis']:
                mix = results['analysis']['mix']
                if isinstance(mix, dict) and 'metrics' in mix:
                    metrics = mix['metrics']
            
            if metrics:
                f.write("## Analysis Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}"
                    f.write(f"| {key} | {value} |\n")
                
                f.write("\n")
            
            # Write segments section
            if 'segments' in results:
                f.write("## Track Segments\n\n")
                f.write("| Start Time | End Time | Label | Confidence |\n")
                f.write("|------------|----------|-------|------------|\n")
                
                for segment in results['segments']:
                    start = segment.get('start_time', 0)
                    end = segment.get('end_time', 0)
                    label = segment.get('label', 'Unknown')
                    confidence = segment.get('confidence', 0)
                    
                    f.write(f"| {start:.2f}s | {end:.2f}s | {label} | {confidence:.2f} |\n")
                
                f.write("\n")
            
            # Write additional sections based on result content
            for section, content in results.items():
                if section not in ('metadata', 'analysis', 'segments'):
                    f.write(f"## {section.capitalize()}\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(content, indent=2))
                    f.write("\n```\n\n")
        
        logger.info(f"Exported results as Markdown to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error exporting to Markdown: {str(e)}")
        raise


def export_html(results: Dict[str, Any], 
               output_dir: Path,
               filename_prefix: str,
               timestamp: str) -> str:
    """Export results as HTML.
    
    Args:
        results: Analysis results to export
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        timestamp: Timestamp string for the filename
        
    Returns:
        Path to the exported file
    """
    filepath = output_dir / f"{filename_prefix}_{timestamp}.html"
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Get file name for title
            title = "Heihachi Analysis Results"
            if 'metadata' in results and 'file_path' in results['metadata']:
                title = f"Analysis Results for {os.path.basename(results['metadata']['file_path'])}"
            
            # Write HTML header
            f.write(f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metadata {{ background-color: #f5f5f5; padding: 10px; border-radius: 4px; }}
        pre {{ background-color: #f0f0f0; padding: 10px; border-radius: 4px; overflow: auto; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
""")
            
            # Write metadata section
            if 'metadata' in results:
                f.write('<h2>Metadata</h2>\n<div class="metadata">\n<table>\n')
                f.write('<tr><th>Property</th><th>Value</th></tr>\n')
                
                for key, value in results['metadata'].items():
                    if key == 'processing_time' and isinstance(value, (int, float)):
                        value = f"{value:.2f}s"
                    f.write(f'<tr><td>{key}</td><td>{value}</td></tr>\n')
                
                f.write('</table>\n</div>\n')
            
            # Write metrics section
            metrics = {}
            if 'analysis' in results and 'mix' in results['analysis']:
                mix = results['analysis']['mix']
                if isinstance(mix, dict) and 'metrics' in mix:
                    metrics = mix['metrics']
            
            if metrics:
                f.write('<h2>Analysis Metrics</h2>\n<table>\n')
                f.write('<tr><th>Metric</th><th>Value</th></tr>\n')
                
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        value = f"{value:.4f}"
                    f.write(f'<tr><td>{key}</td><td>{value}</td></tr>\n')
                
                f.write('</table>\n')
            
            # Write segments section
            if 'segments' in results:
                f.write('<h2>Track Segments</h2>\n<table>\n')
                f.write('<tr><th>Start Time</th><th>End Time</th><th>Label</th><th>Confidence</th></tr>\n')
                
                for segment in results['segments']:
                    start = segment.get('start_time', 0)
                    end = segment.get('end_time', 0)
                    label = segment.get('label', 'Unknown')
                    confidence = segment.get('confidence', 0)
                    
                    f.write(f'<tr><td>{start:.2f}s</td><td>{end:.2f}s</td><td>{label}</td><td>{confidence:.2f}</td></tr>\n')
                
                f.write('</table>\n')
            
            # Write additional sections based on result content
            for section, content in results.items():
                if section not in ('metadata', 'analysis', 'segments'):
                    f.write(f'<h2>{section.capitalize()}</h2>\n')
                    f.write('<pre>\n')
                    f.write(json.dumps(content, indent=2))
                    f.write('\n</pre>\n')
            
            # Write HTML footer
            f.write("""
</body>
</html>
""")
        
        logger.info(f"Exported results as HTML to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error exporting to HTML: {str(e)}")
        raise


def export_xml(results: Dict[str, Any], 
              output_dir: Path,
              filename_prefix: str,
              timestamp: str) -> str:
    """Export results as XML.
    
    Args:
        results: Analysis results to export
        output_dir: Directory to save exported files
        filename_prefix: Prefix for output filename
        timestamp: Timestamp string for the filename
        
    Returns:
        Path to the exported file
    """
    filepath = output_dir / f"{filename_prefix}_{timestamp}.xml"
    
    try:
        # Try to import XML library
        try:
            import xml.dom.minidom as minidom
            from xml.etree import ElementTree as ET
        except ImportError:
            logger.error("XML libraries not found")
            raise ImportError("XML libraries required for XML export")
        
        # Create root element
        root = ET.Element("HeihachResults")
        
        # Add generation timestamp
        timestamp_elem = ET.SubElement(root, "GeneratedTimestamp")
        timestamp_elem.text = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Convert dictionary to XML recursively
        for key, value in results.items():
            _dict_to_xml(root, key, value)
        
        # Pretty print
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        logger.info(f"Exported results as XML to {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Error exporting to XML: {str(e)}")
        raise


def _dict_to_xml(parent, tag, data):
    """
    Recursively convert dictionary to XML.
    
    Args:
        parent: Parent XML element
        tag: Tag name for the element
        data: Data to convert
    """
    if isinstance(data, dict):
        elem = ET.SubElement(parent, tag)
        for key, value in data.items():
            _dict_to_xml(elem, key, value)
    elif isinstance(data, list):
        elem = ET.SubElement(parent, tag)
        for i, value in enumerate(data):
            item_tag = "item"
            if all(isinstance(x, dict) for x in data):
                item_tag = tag[:-1] if tag.endswith('s') else f"{tag}_item"
            _dict_to_xml(elem, item_tag, value)
    else:
        elem = ET.SubElement(parent, tag)
        elem.text = str(data)


def flatten_dict(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten a nested dictionary for CSV export.
    
    This function attempts to extract important metrics and statistics
    from the results for tabular export.
    
    Args:
        data: Results dictionary
        
    Returns:
        List of flattened dictionaries
    """
    flattened = []
    
    # Check if this is a batch result
    if 'stats' in data and 'results' in data:
        # This is a batch result, extract stats from each file
        file_results = []
        
        if 'files' in data:
            # Direct file results available
            for file_data in data['files']:
                file_dict = {
                    'filename': file_data.get('file', 'unknown'),
                    'status': file_data.get('status', 'unknown'),
                    'processing_time': file_data.get('processing_time', 0),
                }
                
                # Extract metrics if available
                if 'metrics' in file_data:
                    for k, v in file_data['metrics'].items():
                        file_dict[f'metric_{k}'] = v
                
                file_results.append(file_dict)
        
        # If no direct file results but we have a list of individual results
        elif 'results' in data and isinstance(data['results'], list):
            for result_item in data['results']:
                if isinstance(result_item, dict) and 'results' in result_item:
                    config = result_item.get('config', 'default')
                    
                    if 'files' in result_item['results']:
                        for file_data in result_item['results']['files']:
                            file_dict = {
                                'config': config,
                                'filename': file_data.get('file', 'unknown'),
                                'status': file_data.get('status', 'unknown'),
                                'processing_time': file_data.get('processing_time', 0),
                            }
                            
                            # Extract metrics if available
                            if 'metrics' in file_data:
                                for k, v in file_data['metrics'].items():
                                    file_dict[f'metric_{k}'] = v
                            
                            file_results.append(file_dict)
        
        if file_results:
            flattened = file_results
        else:
            # Couldn't extract detailed file results, just return overall stats
            flattened = [{
                'total_files': data['stats'].get('total', 0),
                'successful': data['stats'].get('success', 0),
                'failed': data['stats'].get('failed', 0),
                'total_time': data['stats'].get('total_time', 0),
            }]
    
    # Check if this is a single file result
    elif 'metadata' in data and 'file_path' in data['metadata']:
        file_dict = {
            'filename': os.path.basename(data['metadata']['file_path']),
            'processing_time': data['metadata'].get('processing_time', 0),
        }
        
        # Extract metrics from analysis results if available
        if 'analysis' in data and 'mix' in data['analysis']:
            mix = data['analysis']['mix']
            if isinstance(mix, dict) and 'metrics' in mix:
                for k, v in mix['metrics'].items():
                    file_dict[f'metric_{k}'] = v
        
        flattened = [file_dict]
    
    # If we couldn't extract structured data, create a generic representation
    if not flattened:
        flat_dict = {}
        
        def _flatten_recursive(d, prefix=''):
            for k, v in d.items():
                key = f"{prefix}_{k}" if prefix else k
                if isinstance(v, dict):
                    _flatten_recursive(v, key)
                elif isinstance(v, (int, float, str, bool)) or v is None:
                    flat_dict[key] = v
        
        _flatten_recursive(data)
        flattened = [flat_dict] if flat_dict else []
    
    return flattened 