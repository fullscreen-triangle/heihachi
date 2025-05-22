#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ijson  # For streaming large JSON files
from tqdm import tqdm
import time

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir

def process_results_file(file_path, chunk_size=10000, max_items=None):
    """Process the large results file using streaming to handle its size.
    
    Args:
        file_path: Path to the results.json file
        chunk_size: How many items to process before reporting progress
        max_items: Maximum number of items to process (for testing)
        
    Returns:
        DataFrame containing the processed results
    """
    print(f"Processing {file_path} with streaming...")
    
    # Time tracking
    start_time = time.time()
    
    # Process in chunks
    records = []
    count = 0
    
    try:
        # Use ijson to stream the file as an array of objects
        with open(file_path, 'rb') as f:
            for item in tqdm(ijson.items(f, 'item'), desc="Processing results"):
                # Add the item to our records
                records.append(item)
                count += 1
                
                # Report progress
                if count % chunk_size == 0:
                    print(f"Processed {count} items...")
                
                # Optionally limit the number of items for testing
                if max_items and count >= max_items:
                    print(f"Reached limit of {max_items} items")
                    break
                    
        print(f"Successfully processed {count} items")
        
    except Exception as e:
        print(f"Error processing file with ijson: {e}")
        
        # Try with direct line-by-line reading for more resilience
        try:
            print("Attempting alternative processing...")
            count = 0
            records = []
            
            with open(file_path, 'r') as f:
                # Check first character to see if it's an array
                first_char = f.read(1)
                f.seek(0)  # Reset to beginning
                
                if first_char == '[':
                    print("File appears to be a JSON array")
                    # Skip the opening bracket
                    line = f.readline().strip()
                    if line != '[':
                        f.seek(0)  # Reset if not just a bracket
                    
                    # Process each line as a potential JSON object
                    line_num = 1
                    for line in tqdm(f, desc="Processing line by line"):
                        line_num += 1
                        line = line.strip()
                        
                        # Skip empty lines, array brackets, and handle trailing commas
                        if not line or line in ['[', ']']:
                            continue
                        if line.endswith(','):
                            line = line[:-1]
                        if not line or line in ['[', ']']:
                            continue
                            
                        try:
                            item = json.loads(line)
                            records.append(item)
                            count += 1
                            
                            if count % chunk_size == 0:
                                print(f"Processed {count} items...")
                                
                            if max_items and count >= max_items:
                                print(f"Reached limit of {max_items} items")
                                break
                                
                        except json.JSONDecodeError as je:
                            # Skip lines that aren't valid JSON
                            continue
                else:
                    print("File does not appear to be a JSON array")
            
            print(f"Alternative processing completed, processed {count} items")
            
        except Exception as e2:
            print(f"Alternative processing also failed: {e2}")
            if count == 0:
                print("No items could be processed from the file")
                return None
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    print(f"Total entries processed: {count}")
    
    # Convert records to DataFrame for analysis
    if records:
        df = pd.DataFrame(records)
        print(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        print(f"Column names: {list(df.columns)}")
        return df
    else:
        print("No records were processed successfully")
        return None

def analyze_results(df):
    """Analyze the results DataFrame and generate statistics."""
    if df is None or len(df) == 0:
        print("No data to analyze")
        return None
        
    stats = {}
    
    # Basic information
    stats['total_records'] = len(df)
    stats['columns'] = list(df.columns)
    
    # Check for common fields
    if 'type' in df.columns:
        stats['types'] = df['type'].value_counts().to_dict()
        stats['type_percentages'] = ((df['type'].value_counts() / len(df)) * 100).to_dict()
        
    if 'time' in df.columns:
        stats['min_time'] = df['time'].min()
        stats['max_time'] = df['time'].max()
        stats['duration'] = stats['max_time'] - stats['min_time']
        
    if 'confidence' in df.columns:
        stats['avg_confidence'] = df['confidence'].mean()
        if 'type' in df.columns:
            stats['confidence_by_type'] = df.groupby('type')['confidence'].mean().to_dict()
            
    if 'velocity' in df.columns:
        stats['avg_velocity'] = df['velocity'].mean()
        if 'type' in df.columns:
            stats['velocity_by_type'] = df.groupby('type')['velocity'].mean().to_dict()
    
    # Return stats and the DataFrame for visualization
    return {
        'stats': stats,
        'df': df
    }

def create_visualizations(analysis, output_dir):
    """Create visualizations from the analysis results."""
    print("Creating visualizations...")
    stats = analysis['stats']
    df = analysis['df']
    
    # 1. Distribution of result types (pie chart)
    if 'types' in stats:
        plt.figure(figsize=(12, 8))
        labels = list(stats['types'].keys())
        values = list(stats['types'].values())
        
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
        plt.title('Distribution of Result Types')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'result_types_pie.png'), dpi=300)
        plt.close()
        print("Created result types pie chart")
    
    # 2. Bar chart of result type counts
    if 'types' in stats:
        plt.figure(figsize=(14, 8))
        sorted_items = sorted(stats['types'].items(), key=lambda x: x[1], reverse=True)
        labels = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        plt.bar(labels, values, color=sns.color_palette("viridis", len(labels)))
        plt.xlabel('Result Type')
        plt.ylabel('Count')
        plt.title('Number of Results by Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'result_types_bar.png'), dpi=300)
        plt.close()
        print("Created result types bar chart")
    
    # 3. Confidence vs. Velocity scatter plot by type (if both exist)
    if 'confidence' in df.columns and 'velocity' in df.columns and 'type' in df.columns:
        plt.figure(figsize=(14, 10))
        
        # For performance, sample large datasets
        sample_size = min(20000, len(df))
        if len(df) > sample_size:
            plot_df = df.sample(sample_size)
            title_suffix = f" (Sampled {sample_size} points)"
        else:
            plot_df = df
            title_suffix = ""
            
        for drum_type, group in plot_df.groupby('type'):
            plt.scatter(group['confidence'], group['velocity'], 
                       alpha=0.6, label=drum_type)
                       
        plt.xlabel('Confidence')
        plt.ylabel('Velocity')
        plt.title(f'Confidence vs. Velocity for Different Types{title_suffix}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_velocity_scatter.png'), dpi=300)
        plt.close()
        print("Created confidence vs velocity scatter plot")
    
    # 4. Timeline visualization (if time exists)
    if 'time' in df.columns and 'type' in df.columns:
        min_time = stats['min_time']
        max_time = stats['max_time']
        duration = stats['duration']
        
        # Determine appropriate sampling rate for visualization
        sample_size = min(20000, len(df))
        if len(df) > sample_size:
            sampled_df = df.sample(sample_size)
            print(f"Sampling {sample_size} points for timeline visualization")
            title_suffix = f" (Sampled {sample_size} points)"
        else:
            sampled_df = df
            title_suffix = ""
            
        # Create a color map
        drum_types = sorted(df['type'].unique())
        colors = sns.color_palette("husl", len(drum_types))
        color_map = {drum_type: colors[i] for i, drum_type in enumerate(drum_types)}
        
        plt.figure(figsize=(20, 10))
        
        # Plot scatter
        for drum_type, group in sampled_df.groupby('type'):
            plt.scatter(group['time'], [drum_type] * len(group), 
                       alpha=0.5, s=5, 
                       color=color_map[drum_type], 
                       label=drum_type)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Type')
        plt.title(f'Timeline of Results{title_suffix}')
        
        # If too many types, limit legend
        if len(drum_types) > 10:
            handles, labels = plt.gca().get_legend_handles_labels()
            counts = df['type'].value_counts()
            top_types = counts.nlargest(10).index.tolist()
            
            handles_dict = {label: handle for handle, label in zip(handles, labels)}
            top_handles = [handles_dict[t] for t in top_types if t in handles_dict]
            top_labels = [t for t in top_types if t in handles_dict]
            
            plt.legend(top_handles, top_labels, title="Top 10 Types", loc='upper right')
        else:
            plt.legend(loc='upper right')
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_timeline.png'), dpi=300)
        plt.close()
        print("Created results timeline")
        
        # 5. Density over time visualization
        plt.figure(figsize=(20, 10))
        
        # Determine appropriate bin size
        bin_size = max(0.5, duration / 200)  # Either 0.5 seconds or divide total duration into ~200 bins
        bins = np.arange(min_time, max_time + bin_size, bin_size)
        
        # Plot density for top types
        top_types = df['type'].value_counts().nlargest(10).index.tolist()
        
        for drum_type in top_types:
            type_df = df[df['type'] == drum_type]
            hist, _ = np.histogram(type_df['time'], bins=bins)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            plt.plot(bin_centers, hist, label=drum_type, linewidth=2)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel(f'Count per {bin_size:.1f}-second Window')
        plt.title('Density of Top 10 Types Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'results_density.png'), dpi=300)
        plt.close()
        print("Created density plot")
    
    # 6. Interactive timeline with Plotly (if time exists)
    if 'time' in df.columns and 'type' in df.columns:
        # Sample for performance
        sample_size = min(10000, len(df))
        if len(df) > sample_size:
            sampled_df = df.sample(sample_size)
            print(f"Sampling {sample_size} points for interactive visualization")
            title_suffix = f" (Sampled {sample_size}/{len(df)} points)"
        else:
            sampled_df = df
            title_suffix = ""
            
        fig = go.Figure()
        
        # Add traces for each type
        for drum_type, group in sampled_df.groupby('type'):
            # Build hover information based on available columns
            hover_data = []
            for _, row in group.iterrows():
                hover_info = f"<b>{drum_type}</b><br>Time: {row['time']:.2f}s"
                
                if 'confidence' in row:
                    hover_info += f"<br>Confidence: {row['confidence']:.2f}"
                    
                if 'velocity' in row:
                    hover_info += f"<br>Velocity: {row['velocity']:.2f}"
                    
                hover_data.append(hover_info)
            
            # Add the trace
            marker_size = group['velocity'] * 2 if 'velocity' in group else 8
            marker_opacity = group['confidence'] if 'confidence' in group else 0.7
            
            fig.add_trace(go.Scatter(
                x=group['time'],
                y=[drum_type] * len(group),
                mode='markers',
                name=drum_type,
                marker=dict(
                    size=marker_size,
                    opacity=marker_opacity,
                    line=dict(width=1)
                ),
                hovertext=hover_data,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'Interactive Results Timeline{title_suffix}',
            xaxis_title='Time (seconds)',
            yaxis_title='Type',
            legend_title='Type',
            height=800,
            width=1200
        )
        
        fig.write_html(os.path.join(output_dir, 'interactive_results_timeline.html'))
        print("Created interactive timeline")
    
    # 7. Create a summary text file with key statistics
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("RESULTS ANALYSIS SUMMARY\n")
        f.write("=======================\n\n")
        
        f.write(f"Total records analyzed: {stats['total_records']}\n")
        if 'min_time' in stats and 'max_time' in stats:
            f.write(f"Time range: {stats['min_time']:.2f}s to {stats['max_time']:.2f}s (Duration: {stats['duration']:.2f}s)\n\n")
        
        if 'types' in stats:
            f.write("Record counts by type:\n")
            for type_name, count in sorted(stats['types'].items(), key=lambda x: x[1], reverse=True):
                percentage = stats['type_percentages'][type_name]
                f.write(f"  {type_name}: {count} records ({percentage:.1f}%)\n")
        
        if 'confidence_by_type' in stats:
            f.write("\nAverage confidence by type:\n")
            for type_name, conf in sorted(stats['confidence_by_type'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {type_name}: {conf:.3f}\n")
        
        if 'velocity_by_type' in stats:
            f.write("\nAverage velocity by type:\n")
            for type_name, vel in sorted(stats['velocity_by_type'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {type_name}: {vel:.3f}\n")
                
        if 'columns' in stats:
            f.write("\nAvailable data fields:\n")
            for col in sorted(stats['columns']):
                f.write(f"  - {col}\n")
    
    print(f"Created analysis summary at {summary_path}")
    print(f"Created all visualizations in {output_dir}")

def main():
    # File paths and directories
    results_path = 'public/results/MachineCodeAudioCommunications/results.json'
    output_dir = 'visualizations/results_analysis'
    
    # Create output directory
    create_output_directory(output_dir)
    
    # Process the results file - for testing, limit to 100,000 items
    df = process_results_file(results_path, max_items=None)
    
    if df is not None:
        # Analyze the results
        analysis_results = analyze_results(df)
        
        if analysis_results:
            # Create visualizations
            create_visualizations(analysis_results, output_dir)
            print("Analysis and visualization complete")
        else:
            print("Analysis failed")
    else:
        print("Processing failed, no data available for analysis")

if __name__ == "__main__":
    main() 