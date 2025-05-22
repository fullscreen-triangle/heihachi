#!/usr/bin/env python3
import json
import ijson
import os
import sys
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict

def create_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    return output_dir

def extract_time_slice(file_path, start_time, end_time, output_dir, max_items=None):
    """
    Extract a slice of data from a specific time range from the large results.json file.
    
    Args:
        file_path: Path to the results.json file
        start_time: Start time in seconds
        end_time: End time in seconds
        output_dir: Directory to save extracted data and visualizations
        max_items: Maximum number of items to extract (for testing)
    """
    print(f"Extracting time slice from {start_time}s to {end_time}s")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Track extracted data
    extracted_data = []
    item_count = 0
    start_time_found = False
    
    # Track timing
    extraction_start = time.time()
    
    try:
        with open(file_path, 'rb') as f:
            # Check if it's an array
            first_char = f.read(1).decode('utf-8').strip()
            f.seek(0)  # Reset position
            
            if first_char == '[':
                print("File is a JSON array, processing items...")
                
                # Process array items
                parser = ijson.items(f, 'item')
                for item in parser:
                    # Check if item has time
                    if 'time' in item:
                        item_time = float(item['time'])
                        
                        # Check if we're in the time range
                        if start_time <= item_time <= end_time:
                            extracted_data.append(item)
                            item_count += 1
                            
                            if not start_time_found:
                                start_time_found = True
                                print(f"Found first item at time {item_time}s")
                        
                        # Once we pass the end time, we can stop
                        elif item_time > end_time and start_time_found:
                            print(f"Reached end time after extracting {item_count} items")
                            break
                    
                    # Optional limit for testing
                    if max_items and item_count >= max_items:
                        print(f"Reached item limit of {max_items}")
                        break
                    
                    # Print progress
                    if item_count > 0 and item_count % 1000 == 0:
                        elapsed = time.time() - extraction_start
                        print(f"Extracted {item_count} items in {elapsed:.1f} seconds")
            else:
                print("File is not a JSON array, unable to process")
    
    except Exception as e:
        print(f"Error extracting data: {e}")
    
    # Save extracted data
    if extracted_data:
        output_file = os.path.join(output_dir, f"slice_{start_time}_{end_time}.json")
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        print(f"Saved {len(extracted_data)} items to {output_file}")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(extracted_data)
        
        # Create visualizations
        visualize_extracted_data(df, output_dir, start_time, end_time)
        
        return df
    else:
        print("No data found in the specified time range")
        return None

def extract_by_type(file_path, event_type, output_dir, limit=1000):
    """
    Extract items of a specific type from the large results.json file.
    
    Args:
        file_path: Path to the results.json file
        event_type: Type of events to extract
        output_dir: Directory to save extracted data and visualizations
        limit: Maximum number of items to extract
    """
    print(f"Extracting items of type '{event_type}'")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Track extracted data
    extracted_data = []
    item_count = 0
    
    # Track timing
    extraction_start = time.time()
    
    try:
        with open(file_path, 'rb') as f:
            # Check if it's an array
            first_char = f.read(1).decode('utf-8').strip()
            f.seek(0)  # Reset position
            
            if first_char == '[':
                print("File is a JSON array, processing items...")
                
                # Process array items
                parser = ijson.items(f, 'item')
                for item in parser:
                    # Check if item matches the type
                    if 'type' in item and item['type'] == event_type:
                        extracted_data.append(item)
                        item_count += 1
                        
                        # Print progress
                        if item_count % 100 == 0:
                            elapsed = time.time() - extraction_start
                            print(f"Extracted {item_count} items in {elapsed:.1f} seconds")
                        
                        # Stop if we reach the limit
                        if item_count >= limit:
                            print(f"Reached limit of {limit} items")
                            break
            else:
                print("File is not a JSON array, unable to process")
    
    except Exception as e:
        print(f"Error extracting data: {e}")
    
    # Save extracted data
    if extracted_data:
        output_file = os.path.join(output_dir, f"type_{event_type}.json")
        with open(output_file, 'w') as f:
            json.dump(extracted_data, f, indent=2)
        
        print(f"Saved {len(extracted_data)} items to {output_file}")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(extracted_data)
        
        # Create visualizations
        visualize_type_data(df, event_type, output_dir)
        
        return df
    else:
        print(f"No items found with type '{event_type}'")
        return None

def count_events_by_type(file_path, output_dir):
    """
    Count the number of events by type in the large results.json file.
    
    Args:
        file_path: Path to the results.json file
        output_dir: Directory to save visualizations
    """
    print("Counting events by type...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Track counts
    type_counts = defaultdict(int)
    total_items = 0
    
    # Track timing
    start_time = time.time()
    
    try:
        with open(file_path, 'rb') as f:
            # Check if it's an array
            first_char = f.read(1).decode('utf-8').strip()
            f.seek(0)  # Reset position
            
            if first_char == '[':
                print("File is a JSON array, processing items...")
                
                # Process array items
                parser = ijson.items(f, 'item')
                for item in parser:
                    # Check if item has a type
                    if 'type' in item:
                        type_counts[item['type']] += 1
                    
                    total_items += 1
                    
                    # Print progress
                    if total_items % 10000 == 0:
                        elapsed = time.time() - start_time
                        print(f"Processed {total_items} items in {elapsed:.1f} seconds")
            else:
                print("File is not a JSON array, unable to process")
    
    except Exception as e:
        print(f"Error counting events: {e}")
    
    # Create visualizations
    if type_counts:
        # Sort by count
        sorted_counts = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Create dataframe
        df = pd.DataFrame(sorted_counts, columns=['type', 'count'])
        
        # Save to CSV
        csv_file = os.path.join(output_dir, "event_type_counts.csv")
        df.to_csv(csv_file, index=False)
        
        # Create bar chart
        plt.figure(figsize=(14, 8))
        plt.bar(df['type'], df['count'], color=sns.color_palette("viridis", len(df)))
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.title('Event Counts by Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "event_type_counts.png"), dpi=300)
        plt.close()
        
        # Create pie chart (for top types)
        top_df = df.head(10)  # Top 10 types
        
        # Add "Other" category for the rest
        if len(df) > 10:
            other_count = df.iloc[10:]['count'].sum()
            top_df = pd.concat([top_df, pd.DataFrame([{'type': 'Other', 'count': other_count}])])
        
        plt.figure(figsize=(12, 8))
        plt.pie(top_df['count'], labels=top_df['type'], autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Distribution of Event Types')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "event_type_pie.png"), dpi=300)
        plt.close()
        
        print(f"Found {len(type_counts)} different event types")
        print(f"Total items processed: {total_items}")
        print(f"Results saved to {output_dir}")
        
        return df
    else:
        print("No event types found")
        return None

def visualize_extracted_data(df, output_dir, start_time, end_time):
    """Create visualizations from the extracted data."""
    print("Creating visualizations...")
    
    # 1. Distribution of types in this time slice
    if 'type' in df.columns:
        plt.figure(figsize=(14, 8))
        type_counts = df['type'].value_counts()
        
        plt.bar(type_counts.index, type_counts.values, color=sns.color_palette("viridis", len(type_counts)))
        plt.xlabel('Event Type')
        plt.ylabel('Count')
        plt.title(f'Event Counts by Type ({start_time}s to {end_time}s)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"slice_{start_time}_{end_time}_types.png"), dpi=300)
        plt.close()
        print("Created type distribution chart")
    
    # 2. Timeline of events
    if 'time' in df.columns and 'type' in df.columns:
        plt.figure(figsize=(20, 10))
        
        # Create a color map for event types
        event_types = sorted(df['type'].unique())
        colors = sns.color_palette("husl", len(event_types))
        color_map = {event_type: colors[i] for i, event_type in enumerate(event_types)}
        
        # Plot events on timeline
        for event_type, group in df.groupby('type'):
            plt.scatter(group['time'], [event_type] * len(group), 
                       alpha=0.5, s=10, color=color_map[event_type], label=event_type)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Event Type')
        plt.title(f'Timeline of Events ({start_time}s to {end_time}s)')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"slice_{start_time}_{end_time}_timeline.png"), dpi=300)
        plt.close()
        print("Created event timeline")
    
    # 3. Confidence distribution (if available)
    if 'confidence' in df.columns and 'type' in df.columns:
        plt.figure(figsize=(14, 8))
        
        for event_type, group in df.groupby('type'):
            if len(group) > 5:  # Only plot if we have enough data
                sns.kdeplot(group['confidence'], label=event_type)
        
        plt.xlabel('Confidence')
        plt.ylabel('Density')
        plt.title(f'Confidence Distribution by Type ({start_time}s to {end_time}s)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"slice_{start_time}_{end_time}_confidence.png"), dpi=300)
        plt.close()
        print("Created confidence distribution")

def visualize_type_data(df, event_type, output_dir):
    """Create visualizations for a specific event type."""
    print(f"Creating visualizations for event type '{event_type}'...")
    
    # 1. Timeline of events
    if 'time' in df.columns:
        plt.figure(figsize=(20, 6))
        
        # Plot events on timeline
        plt.scatter(df['time'], [1] * len(df), alpha=0.5, s=10)
        
        plt.xlabel('Time (seconds)')
        plt.yticks([])  # Hide y-axis ticks
        plt.title(f'Timeline of {event_type} Events')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"type_{event_type}_timeline.png"), dpi=300)
        plt.close()
        print("Created event timeline")
    
    # 2. Confidence vs. velocity (if available)
    if 'confidence' in df.columns and 'velocity' in df.columns:
        plt.figure(figsize=(10, 8))
        
        plt.scatter(df['confidence'], df['velocity'], alpha=0.5)
        
        plt.xlabel('Confidence')
        plt.ylabel('Velocity')
        plt.title(f'Confidence vs. Velocity for {event_type} Events')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"type_{event_type}_conf_vel.png"), dpi=300)
        plt.close()
        print("Created confidence vs. velocity plot")
    
    # 3. Histogram of event distribution over time
    if 'time' in df.columns:
        plt.figure(figsize=(14, 6))
        
        # Create histogram
        min_time = df['time'].min()
        max_time = df['time'].max()
        duration = max_time - min_time
        
        # Use appropriate bin size
        bin_size = max(1, duration / 100)
        plt.hist(df['time'], bins=np.arange(min_time, max_time + bin_size, bin_size), alpha=0.7)
        
        plt.xlabel('Time (seconds)')
        plt.ylabel('Event Count')
        plt.title(f'Distribution of {event_type} Events Over Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"type_{event_type}_distribution.png"), dpi=300)
        plt.close()
        print("Created time distribution histogram")
    
    # 4. Create a summary text file
    summary_file = os.path.join(output_dir, f"type_{event_type}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"SUMMARY OF {event_type.upper()} EVENTS\n")
        f.write("=" * (len(event_type) + 18) + "\n\n")
        
        f.write(f"Total events: {len(df)}\n")
        
        if 'time' in df.columns:
            min_time = df['time'].min()
            max_time = df['time'].max()
            duration = max_time - min_time
            f.write(f"Time range: {min_time:.2f}s to {max_time:.2f}s (Duration: {duration:.2f}s)\n")
            
            # Calculate intervals between events
            if len(df) > 1:
                times = sorted(df['time'])
                intervals = np.diff(times)
                f.write(f"Average interval between events: {np.mean(intervals):.2f}s\n")
                f.write(f"Median interval: {np.median(intervals):.2f}s\n")
                f.write(f"Min interval: {np.min(intervals):.4f}s\n")
                f.write(f"Max interval: {np.max(intervals):.2f}s\n")
        
        if 'confidence' in df.columns:
            f.write(f"\nConfidence statistics:\n")
            f.write(f"  Mean: {df['confidence'].mean():.4f}\n")
            f.write(f"  Median: {df['confidence'].median():.4f}\n")
            f.write(f"  Min: {df['confidence'].min():.4f}\n")
            f.write(f"  Max: {df['confidence'].max():.4f}\n")
        
        if 'velocity' in df.columns:
            f.write(f"\nVelocity statistics:\n")
            f.write(f"  Mean: {df['velocity'].mean():.4f}\n")
            f.write(f"  Median: {df['velocity'].median():.4f}\n")
            f.write(f"  Min: {df['velocity'].min():.4f}\n")
            f.write(f"  Max: {df['velocity'].max():.4f}\n")
        
        # List columns found in the data
        f.write(f"\nColumns in the data:\n")
        for col in df.columns:
            f.write(f"  {col}\n")
    
    print(f"Created summary file at {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract and visualize data from a large JSON file')
    
    # Add subparsers for different operations
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Subparser for time slice extraction
    time_parser = subparsers.add_parser('time', help='Extract a time slice')
    time_parser.add_argument('--start', type=float, required=True, help='Start time in seconds')
    time_parser.add_argument('--end', type=float, required=True, help='End time in seconds')
    time_parser.add_argument('--limit', type=int, default=None, help='Maximum items to extract')
    
    # Subparser for event type extraction
    type_parser = subparsers.add_parser('type', help='Extract events by type')
    type_parser.add_argument('--type', type=str, required=True, help='Event type to extract')
    type_parser.add_argument('--limit', type=int, default=1000, help='Maximum items to extract')
    
    # Subparser for counting events
    count_parser = subparsers.add_parser('count', help='Count events by type')
    
    # Add common arguments
    for subparser in [time_parser, type_parser, count_parser]:
        subparser.add_argument('--file', type=str, default='public/results/MachineCodeAudioCommunications/results.json', 
                             help='Path to the results.json file')
        subparser.add_argument('--output', type=str, default='visualizations/results_analysis',
                             help='Output directory for extracted data and visualizations')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default values if no command was provided
    if not hasattr(args, 'command') or args.command is None:
        args.command = 'count'
    
    # Set default output directory
    if not hasattr(args, 'output'):
        args.output = 'visualizations/results_analysis'
    
    # Set default file path
    if not hasattr(args, 'file'):
        args.file = 'public/results/MachineCodeAudioCommunications/results.json'
    
    # Create output directory
    create_output_directory(args.output)
    
    # Run the appropriate command
    if args.command == 'time':
        extract_time_slice(args.file, args.start, args.end, args.output, args.limit)
    elif args.command == 'type':
        extract_by_type(args.file, args.type, args.output, args.limit)
    elif args.command == 'count':
        count_events_by_type(args.file, args.output)
    else:
        # Default to counting if no command specified
        count_events_by_type(args.file, args.output)

if __name__ == "__main__":
    main() 