#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_features(file_path):
    """Load features and extract useful information about them."""
    print(f"Loading features from {file_path}...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("Features loaded successfully")
    
    # The features.json file is a dictionary with several keys
    print(f"Top-level keys: {list(data.keys())}")
    
    # Get percussion events which contain the drum hit data
    percussion_data = None
    if 'percussion' in data and 'events' in data['percussion']:
        percussion_data = data['percussion']['events']
        print(f"Found {len(percussion_data)} percussion events")
    else:
        # Try to find drum hits in other locations
        if 'drum' in data:
            drum_data = data['drum']
            print(f"Drum data keys: {list(drum_data.keys())}")
            
        # Look for the data shown in the file snippet
        for key in data.keys():
            if isinstance(data[key], dict) and 'events' in data[key]:
                percussion_data = data[key]['events']
                print(f"Found events in '{key}' with {len(percussion_data)} items")
                break
    
    return data, percussion_data

def analyze_drum_hits(percussion_data):
    """Analyze the drum hit data and prepare visualization-ready datasets."""
    if percussion_data is None or len(percussion_data) == 0:
        print("No percussion events data found")
        return None
    
    print(f"Analyzing {len(percussion_data)} percussion events")
    
    # Convert to dataframe for easier analysis
    df = pd.DataFrame(percussion_data)
    
    # Check what columns we have
    print(f"Columns in percussion data: {list(df.columns)}")
    
    # Make sure we have the required columns
    required_columns = ['type', 'time']
    if not all(col in df.columns for col in required_columns):
        print(f"Missing required columns. Available columns: {list(df.columns)}")
        return None
    
    # Count hits by type
    hits_by_type = df['type'].value_counts().to_dict()
    
    # Group statistics by type
    confidence_by_type = {}
    velocity_by_type = {}
    
    if 'confidence' in df.columns:
        confidence_by_type = df.groupby('type')['confidence'].mean().to_dict()
    
    if 'velocity' in df.columns:
        velocity_by_type = df.groupby('type')['velocity'].mean().to_dict()
    
    results = {
        'hit_count': len(percussion_data),
        'hits_by_type': hits_by_type,
        'confidence_by_type': confidence_by_type,
        'velocity_by_type': velocity_by_type,
        'hits_dataframe': df,
    }
    
    print(f"Analyzed {results['hit_count']} drum hits of {len(results['hits_by_type'])} types")
    return results

def create_visualizations(analysis_results, output_dir, full_data=None):
    """Create visualizations from the analysis results."""
    os.makedirs(output_dir, exist_ok=True)
    df = analysis_results['hits_dataframe']
    
    # 1. Distribution of drum hit types
    plt.figure(figsize=(12, 8))
    hit_counts = analysis_results['hits_by_type']
    labels = list(hit_counts.keys())
    values = list(hit_counts.values())
    
    # Sort by count
    sorted_indices = np.argsort(values)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]
    
    # Calculate percentages
    total = sum(sorted_values)
    percentages = [f"{100*v/total:.1f}%" for v in sorted_values]
    
    # Create pie chart with percentages
    plt.pie(sorted_values, labels=[f"{l} ({p})" for l, p in zip(sorted_labels, percentages)], 
            autopct='%1.1f%%', startangle=90, shadow=True)
    plt.axis('equal')
    plt.title('Distribution of Drum Hit Types')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drum_hit_types_pie.png'), dpi=300)
    plt.close()
    print("Created distribution pie chart")
    
    # 2. Drum hit type counts
    plt.figure(figsize=(14, 8))
    plt.bar(sorted_labels, sorted_values, color=sns.color_palette("viridis", len(sorted_labels)))
    plt.xlabel('Drum Type')
    plt.ylabel('Count')
    plt.title('Number of Hits by Drum Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drum_hit_types_bar.png'), dpi=300)
    plt.close()
    print("Created hit count bar chart")
    
    # 3. Confidence vs. Velocity scatter plot by type (if both exist)
    if 'confidence' in df.columns and 'velocity' in df.columns:
        plt.figure(figsize=(14, 10))
        for drum_type, group in df.groupby('type'):
            plt.scatter(group['confidence'], group['velocity'], 
                       alpha=0.6, label=drum_type)
        plt.xlabel('Confidence')
        plt.ylabel('Velocity')
        plt.title('Confidence vs. Velocity for Different Drum Types')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_velocity_scatter.png'), dpi=300)
        plt.close()
        print("Created confidence vs velocity scatter plot")
    
    # 4. Timeline of drum hits
    # For larger datasets, sample or bin the data to avoid performance issues
    plt.figure(figsize=(20, 10))
    
    # Create a color map for drum types
    drum_types = sorted(df['type'].unique())
    colors = sns.color_palette("husl", len(drum_types))
    color_map = {drum_type: colors[i] for i, drum_type in enumerate(drum_types)}
    
    # For large datasets, we'll plot with some transparency and smaller points
    for drum_type, group in df.groupby('type'):
        plt.scatter(group['time'], [drum_type] * len(group), 
                   alpha=0.5, s=5, color=color_map[drum_type], label=drum_type)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Drum Type')
    plt.title('Timeline of Drum Hits')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drum_hits_timeline.png'), dpi=300)
    plt.close()
    print("Created drum hits timeline")
    
    # 5. Pattern visualization - look at hits over time windows
    # First check the total duration to determine appropriate time windows
    max_time = df['time'].max()
    min_time = df['time'].min()
    duration = max_time - min_time
    
    # Determine an appropriate window size
    window_size = max(5, duration / 100)  # Either 5 seconds or 1/100th of total duration
    bins = np.arange(min_time, max_time + window_size, window_size)
    
    plt.figure(figsize=(20, 10))
    
    # Count hits in each time window for each drum type
    for drum_type, group in df.groupby('type'):
        # Use histogram to count hits in time windows
        hist, _ = np.histogram(group['time'], bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        plt.plot(bin_centers, hist, label=drum_type, linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Drum Hits per {window_size:.1f}-second Window')
    plt.title('Drum Hit Density Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drum_density.png'), dpi=300)
    plt.close()
    print("Created drum density plot")
    
    # 6. Heatmap of hits over time by type
    # Determine appropriate number of time bins
    n_bins = min(50, int(duration / window_size))
    time_bins = np.linspace(min_time, max_time, n_bins)
    
    # Create matrix for heatmap (rows = types, columns = time bins)
    heatmap_data = np.zeros((len(drum_types), len(time_bins)-1))
    
    # Fill heatmap data
    for i, drum_type in enumerate(drum_types):
        type_df = df[df['type'] == drum_type]
        hist, _ = np.histogram(type_df['time'], bins=time_bins)
        heatmap_data[i, :] = hist
    
    # Normalize per row for better visualization
    row_maxes = heatmap_data.max(axis=1, keepdims=True)
    normalized_data = heatmap_data / np.where(row_maxes > 0, row_maxes, 1)
    
    plt.figure(figsize=(20, 10))
    plt.imshow(normalized_data, aspect='auto', cmap='viridis')
    plt.colorbar(label='Normalized Hit Density')
    plt.yticks(np.arange(len(drum_types)), drum_types)
    plt.xlabel('Time (binned)')
    plt.title('Drum Hit Patterns Over Time')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drum_pattern_heatmap.png'), dpi=300)
    plt.close()
    print("Created pattern heatmap")
    
    # 7. Interactive timeline with Plotly
    # For large datasets, we might need to sample the data
    sample_size = min(10000, len(df))  # Limit to at most 10,000 points for performance
    if len(df) > sample_size:
        print(f"Sampling {sample_size} points for interactive visualization")
        sampled_df = df.sample(sample_size)
    else:
        sampled_df = df
    
    fig = go.Figure()
    
    # Add a trace for each drum type
    for drum_type, group in sampled_df.groupby('type'):
        # Prepare hover data
        hover_template = '<b>%{y}</b><br>Time: %{x:.2f}s'
        
        if 'confidence' in group.columns and 'velocity' in group.columns:
            # Use both confidence and velocity
            fig.add_trace(go.Scatter(
                x=group['time'],
                y=[drum_type] * len(group),
                mode='markers',
                name=drum_type,
                marker=dict(
                    size=group['velocity'] * 2,  # Scale down for performance
                    opacity=0.7,
                    line=dict(width=1)
                ),
                customdata=np.stack((group['confidence'], group['velocity']), axis=-1),
                hovertemplate=hover_template + '<br>Confidence: %{customdata[0]:.2f}<br>Velocity: %{customdata[1]:.2f}<extra></extra>'
            ))
        else:
            # Basic marker with no customization
            fig.add_trace(go.Scatter(
                x=group['time'],
                y=[drum_type] * len(group),
                mode='markers',
                name=drum_type,
                marker=dict(size=8, opacity=0.7),
                hovertemplate=hover_template + '<extra></extra>'
            ))
    
    fig.update_layout(
        title='Interactive Drum Hit Timeline',
        xaxis_title='Time (seconds)',
        yaxis_title='Drum Type',
        legend_title='Drum Type',
        height=800,
        width=1200
    )
    
    fig.write_html(os.path.join(output_dir, 'interactive_timeline.html'))
    print("Created interactive timeline")
    
    # 8. Average confidence and velocity by drum type (if available)
    if 'confidence' in df.columns or 'velocity' in df.columns:
        plt.figure(figsize=(14, 8))
        
        # Create bar data
        x = np.arange(len(drum_types))
        width = 0.35
        
        # Plot available data
        if 'confidence' in df.columns and 'velocity' in df.columns:
            # Plot both
            confidence_values = [analysis_results['confidence_by_type'][dt] for dt in drum_types]
            velocity_values = [analysis_results['velocity_by_type'][dt] for dt in drum_types]
            
            plt.bar(x - width/2, confidence_values, width, label='Confidence')
            plt.bar(x + width/2, velocity_values, width, label='Velocity')
            title = 'Average Confidence and Velocity by Drum Type'
        elif 'confidence' in df.columns:
            # Plot just confidence
            confidence_values = [analysis_results['confidence_by_type'][dt] for dt in drum_types]
            plt.bar(x, confidence_values, width, label='Confidence')
            title = 'Average Confidence by Drum Type'
        else:
            # Plot just velocity
            velocity_values = [analysis_results['velocity_by_type'][dt] for dt in drum_types]
            plt.bar(x, velocity_values, width, label='Velocity')
            title = 'Average Velocity by Drum Type'
        
        plt.xlabel('Drum Type')
        plt.ylabel('Value')
        plt.title(title)
        plt.xticks(x, drum_types, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'avg_confidence_velocity.png'), dpi=300)
        plt.close()
        print("Created average stats chart")
    
    # 9. Create a summary text file with key statistics
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("DRUM HIT ANALYSIS SUMMARY\n")
        f.write("=========================\n\n")
        
        f.write(f"Total drum hits analyzed: {analysis_results['hit_count']}\n")
        f.write(f"Time range: {min_time:.2f}s to {max_time:.2f}s (Duration: {duration:.2f}s)\n\n")
        
        f.write("Hit counts by drum type:\n")
        for drum_type, count in sorted(analysis_results['hits_by_type'].items(), 
                                       key=lambda x: x[1], reverse=True):
            percentage = 100 * count / analysis_results['hit_count']
            f.write(f"  {drum_type}: {count} hits ({percentage:.1f}%)\n")
        
        if analysis_results['confidence_by_type']:
            f.write("\nAverage confidence by drum type:\n")
            for drum_type, conf in sorted(analysis_results['confidence_by_type'].items(),
                                         key=lambda x: x[1], reverse=True):
                f.write(f"  {drum_type}: {conf:.3f}\n")
        
        if analysis_results['velocity_by_type']:
            f.write("\nAverage velocity by drum type:\n")
            for drum_type, vel in sorted(analysis_results['velocity_by_type'].items(),
                                        key=lambda x: x[1], reverse=True):
                f.write(f"  {drum_type}: {vel:.3f}\n")
        
        # Add BPM information if available
        if full_data and 'bpm' in full_data and 'bpm' in full_data['bpm']:
            f.write(f"\nBPM: {full_data['bpm']['bpm']}")
            if 'confidence' in full_data['bpm']:
                f.write(f" (confidence: {full_data['bpm']['confidence']:.3f})")
    
    print(f"Created analysis summary at {summary_path}")
    print(f"Created all visualizations in {output_dir}")
    
def main():
    features_path = 'public/results/MachineCodeAudioCommunications/features.json'
    output_dir = 'visualizations/drum_feature_analysis'
    
    data, percussion_data = load_features(features_path)
    
    if percussion_data:
        analysis_results = analyze_drum_hits(percussion_data)
        if analysis_results:
            create_visualizations(analysis_results, output_dir, full_data=data)
            print("Analysis and visualization complete")
        else:
            print("Failed to analyze percussion data")
    else:
        print("No percussion data found in the features file")

if __name__ == "__main__":
    main() 