#!/usr/bin/env python3
"""
Script to showcase Heihachi Audio Analysis Framework capabilities.

This script creates visualizations and summaries from pre-analyzed data
to showcase the framework's capabilities.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add the project root to the path so we can import the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Hardcoded paths for simplicity
RESULTS_DIR = "./public/results/MachineCodeAudioCommunications"
OUTPUT_DIR = "./visualizations/MachineCodeAudioCommunications3_20250327_121818"

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file safely.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data, or empty dict if file doesn't exist
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return {}

def sample_large_json(file_path: str, max_size: int = 10000) -> Dict[str, Any]:
    """Sample a large JSON file without loading it entirely.
    
    Args:
        file_path: Path to the JSON file
        max_size: Maximum number of items to load per array
        
    Returns:
        Dictionary with sampled data
    """
    try:
        with open(file_path, 'r') as f:
            # Read first character to determine structure
            char = f.read(1)
            if char != '{':
                logger.error(f"File {file_path} does not start with a JSON object")
                return {}
            
            # Reset file position
            f.seek(0)
            
            # Load just the structure (assuming manageable size)
            data = json.load(f)
            
            # Sample large arrays
            for key, value in data.items():
                if isinstance(value, list) and len(value) > max_size:
                    # Take samples from beginning, middle, and end
                    step = len(value) // max_size
                    indices = list(range(0, len(value), step))[:max_size]
                    data[key] = [value[i] for i in indices]
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, list) and len(subvalue) > max_size:
                            step = len(subvalue) // max_size
                            indices = list(range(0, len(subvalue), step))[:max_size]
                            data[key][subkey] = [subvalue[i] for i in indices]
            
            return data
    except Exception as e:
        logger.error(f"Error sampling {file_path}: {e}")
        return {}

def create_framework_overview() -> None:
    """Create a visual overview of the framework capabilities."""
    logger.info(f"Creating framework overview from {RESULTS_DIR}")
    
    # Load metadata
    metadata = load_json_file(os.path.join(RESULTS_DIR, "metadata.json"))
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create a summary report
    report = {
        "framework": "Heihachi Audio Analysis Framework",
        "version": "1.0.0",  # Replace with actual version if available
        "analysis_id": metadata.get("analysis_id", "unknown"),
        "timestamp": metadata.get("timestamp", "unknown"),
        "audio_file": metadata.get("audio_path", "unknown"),
        "duration": metadata.get("duration", 0),
        "sample_rate": metadata.get("sample_rate", 0),
        "capabilities": {
            "feature_extraction": [],
            "visualization": [],
            "annotation": [],
            "alignment": []
        }
    }
    
    # Load other result files
    features = sample_large_json(os.path.join(RESULTS_DIR, "features.json"))
    annotation = load_json_file(os.path.join(RESULTS_DIR, "annotation.json"))
    alignment = load_json_file(os.path.join(RESULTS_DIR, "alignment.json"))
    
    # Extract capabilities
    if features:
        report["capabilities"]["feature_extraction"] = list(features.keys())
    
    if annotation:
        for key in annotation.keys():
            report["capabilities"]["annotation"].append(key)
    
    if alignment:
        for key in alignment.keys():
            report["capabilities"]["alignment"].append(key)
    
    # Save the report
    with open(os.path.join(OUTPUT_DIR, "framework_overview.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a visualization of framework capabilities
    create_capabilities_visualization(report)
    
    logger.info(f"Framework overview saved to {OUTPUT_DIR}")

def create_capabilities_visualization(report: Dict[str, Any]) -> None:
    """Create a visualization of the framework capabilities.
    
    Args:
        report: Dictionary containing the framework report
    """
    # Create a capabilities diagram
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up colors
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Create categories
    categories = list(report["capabilities"].keys())
    
    # Create a nested pie chart
    total_caps = sum(len(caps) for caps in report["capabilities"].values())
    
    # Outer ring - categories
    category_sizes = [len(report["capabilities"][cat]) for cat in categories]
    category_percentages = [size / total_caps * 100 for size in category_sizes]
    
    ax.pie(
        category_percentages, 
        labels=categories,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(categories)],
        wedgeprops=dict(width=0.4, edgecolor='w')
    )
    
    # Add title and metadata
    plt.title(f"Heihachi Framework Capabilities\n{report['analysis_id']}", fontsize=14)
    
    # Add a summary table
    summary_text = (
        f"Audio: {os.path.basename(report['audio_file'])}\n"
        f"Duration: {report['duration']:.2f} seconds\n"
        f"Sample Rate: {report['sample_rate']} Hz\n"
        f"Analysis Time: {report['timestamp']}\n\n"
        f"Total Capabilities: {total_caps}"
    )
    
    plt.figtext(0.5, 0.01, summary_text, wrap=True, horizontalalignment='center', fontsize=12)
    
    # Save the figure
    plt.savefig(os.path.join(OUTPUT_DIR, "framework_capabilities.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create a detailed capabilities breakdown
    fig, axs = plt.subplots(len(categories), 1, figsize=(10, 3*len(categories)))
    
    if len(categories) == 1:
        axs = [axs]
    
    for i, category in enumerate(categories):
        capabilities = report["capabilities"][category]
        if capabilities:
            y_pos = np.arange(len(capabilities))
            axs[i].barh(y_pos, [1] * len(capabilities), color=colors[i])
            axs[i].set_yticks(y_pos)
            axs[i].set_yticklabels(capabilities)
            axs[i].set_xlabel('Available')
            axs[i].set_title(f"{category.capitalize()} Capabilities")
            axs[i].grid(axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "capabilities_breakdown.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_html_report() -> None:
    """Generate an HTML report for the analysis."""
    # Load metadata
    metadata = load_json_file(os.path.join(RESULTS_DIR, "metadata.json"))
    
    # Load the framework overview
    overview = load_json_file(os.path.join(OUTPUT_DIR, "framework_overview.json"))
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Heihachi Framework Showcase - {metadata.get("analysis_id", "Unknown")}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            .header {{
                background-color: #34495e;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
                border-radius: 5px;
            }}
            .section {{
                margin-bottom: 30px;
                padding: 20px;
                background-color: #f9f9f9;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
            .metadata {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
            }}
            .metadata div {{
                padding: 10px;
                background-color: #ecf0f1;
                border-radius: 3px;
            }}
            .capabilities {{
                display: flex;
                flex-wrap: wrap;
                gap: 15px;
                margin-top: 20px;
            }}
            .capability-group {{
                flex: 1;
                min-width: 250px;
                background-color: #ecf0f1;
                padding: 15px;
                border-radius: 5px;
            }}
            .capability-list {{
                list-style-type: none;
                padding-left: 0;
            }}
            .capability-list li {{
                padding: 5px 0;
                border-bottom: 1px solid #ddd;
            }}
            .visualizations {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .visualization {{
                text-align: center;
            }}
            .visualization img {{
                max-width: 100%;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }}
            footer {{
                text-align: center;
                margin-top: 50px;
                color: #7f8c8d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Heihachi Audio Analysis Framework</h1>
            <p>Showcase of framework capabilities and analysis results</p>
        </div>

        <div class="section">
            <h2>Analysis Overview</h2>
            <div class="metadata">
                <div><strong>Analysis ID:</strong> {metadata.get("analysis_id", "Unknown")}</div>
                <div><strong>Timestamp:</strong> {metadata.get("timestamp", "Unknown")}</div>
                <div><strong>Audio File:</strong> {metadata.get("audio_path", "Unknown")}</div>
                <div><strong>Duration:</strong> {metadata.get("duration", 0):.2f} seconds</div>
                <div><strong>Sample Rate:</strong> {metadata.get("sample_rate", 0)} Hz</div>
            </div>
        </div>

        <div class="section">
            <h2>Framework Capabilities</h2>
            <div class="visualizations">
                <div class="visualization">
                    <img src="framework_capabilities.png" alt="Framework Capabilities">
                    <p>Framework capabilities breakdown by category</p>
                </div>
                <div class="visualization">
                    <img src="capabilities_breakdown.png" alt="Capabilities Breakdown">
                    <p>Detailed capabilities by category</p>
                </div>
            </div>
            
            <div class="capabilities">
    """
    
    # Add capability groups
    for category, capabilities in overview.get("capabilities", {}).items():
        if capabilities:
            html_content += f"""
                <div class="capability-group">
                    <h3>{category.replace('_', ' ').title()}</h3>
                    <ul class="capability-list">
            """
            
            for capability in capabilities:
                html_content += f"<li>{capability}</li>\n"
            
            html_content += """
                    </ul>
                </div>
            """
    
    # Complete the HTML
    html_content += """
            </div>
        </div>

        <footer>
            <p>Generated with Heihachi Audio Analysis Framework</p>
        </footer>
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(os.path.join(OUTPUT_DIR, "index.html"), 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated at {os.path.join(OUTPUT_DIR, 'index.html')}")

def main():
    """Main function."""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load metadata
    metadata = load_json_file(os.path.join(RESULTS_DIR, "metadata.json"))
    
    # Create a summary report
    report = {
        "framework": "Heihachi Audio Analysis Framework",
        "version": "1.0.0",
        "analysis_id": metadata.get("analysis_id", "unknown"),
        "timestamp": metadata.get("timestamp", "unknown"),
        "audio_file": metadata.get("audio_path", "unknown"),
        "duration": metadata.get("duration", 0),
        "sample_rate": metadata.get("sample_rate", 0),
        "capabilities": {
            "feature_extraction": [],
            "visualization": [],
            "annotation": [],
            "alignment": []
        }
    }
    
    # Load result files
    features = sample_large_json(os.path.join(RESULTS_DIR, "features.json"))
    annotation = load_json_file(os.path.join(RESULTS_DIR, "annotation.json"))
    alignment = load_json_file(os.path.join(RESULTS_DIR, "alignment.json"))
    
    # Extract capabilities
    if features:
        report["capabilities"]["feature_extraction"] = list(features.keys())
    
    if annotation:
        for key in annotation.keys():
            report["capabilities"]["annotation"].append(key)
    
    if alignment:
        for key in alignment.keys():
            report["capabilities"]["alignment"].append(key)
    
    # Save the report
    with open(os.path.join(OUTPUT_DIR, "framework_overview.json"), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create visualizations 
    create_capabilities_visualization(report)
    
    # Generate HTML report
    generate_html_report()
    
    logger.info(f"Showcase completed. Results saved to {OUTPUT_DIR}")
    print(f"\nShowcase completed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Open {OUTPUT_DIR}/index.html to view the report.")
    print(f"Framework capability images saved to:")
    print(f"  - {OUTPUT_DIR}/framework_capabilities.png")
    print(f"  - {OUTPUT_DIR}/capabilities_breakdown.png")

if __name__ == "__main__":
    main() 