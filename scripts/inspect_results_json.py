#!/usr/bin/env python3
import json
import ijson
import os
import sys
from collections import Counter, defaultdict
import time

def inspect_json_structure(file_path, sample_size=100):
    """
    Inspect the structure of a large JSON file using streaming.
    
    Args:
        file_path: Path to the JSON file
        sample_size: Number of items to sample
    """
    print(f"Inspecting file: {file_path}")
    file_size_gb = os.path.getsize(file_path) / (1024 * 1024 * 1024)
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Track structure information
    structure_info = {
        'root_type': None,
        'sample_items': [],
        'keys_found': Counter(),
        'types_found': Counter(),
        'value_examples': defaultdict(list),
    }
    
    # Track timing
    start_time = time.time()
    
    try:
        # First try to determine the root structure
        with open(file_path, 'rb') as f:
            # Read just the first few bytes to check if it's an array or object
            first_chars = f.read(10).decode('utf-8').strip()
            if first_chars.startswith('['):
                structure_info['root_type'] = 'array'
                print("Root structure: JSON array")
            elif first_chars.startswith('{'):
                structure_info['root_type'] = 'object'
                print("Root structure: JSON object")
            else:
                print(f"Unknown root structure starting with: {first_chars}")
                
        # Now try to sample items
        with open(file_path, 'rb') as f:
            # If it's an array, sample items directly
            if structure_info['root_type'] == 'array':
                parser = ijson.items(f, 'item')
                for i, item in enumerate(parser):
                    if i >= sample_size:
                        break
                    
                    # Add to sample
                    if len(structure_info['sample_items']) < 5:  # Store just a few full items
                        structure_info['sample_items'].append(item)
                    
                    # Track keys and types
                    if isinstance(item, dict):
                        for key, value in item.items():
                            structure_info['keys_found'][key] += 1
                            structure_info['types_found'][key + ': ' + type(value).__name__] += 1
                            
                            # Store examples of values (up to 3 per key)
                            if len(structure_info['value_examples'][key]) < 3:
                                if isinstance(value, (str, int, float, bool, type(None))):
                                    structure_info['value_examples'][key].append(value)
                                elif isinstance(value, (list, dict)):
                                    structure_info['value_examples'][key].append(f"{type(value).__name__} of size {len(value)}")
                                else:
                                    structure_info['value_examples'][key].append(f"Complex type: {type(value).__name__}")
                    
                    # Print progress
                    if (i + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"Processed {i + 1} items in {elapsed:.1f} seconds")
                
                print(f"Sampled {min(i + 1, sample_size)} items from the array")
            
            # If it's an object, check its top-level keys
            elif structure_info['root_type'] == 'object':
                try:
                    # Try to read top-level keys
                    top_level_keys = []
                    for prefix, event, value in ijson.parse(f):
                        if event == 'map_key':
                            top_level_keys.append(value)
                            
                            # Track findings
                            structure_info['keys_found'][value] += 1
                        
                        # Once we have some keys, stop
                        if len(top_level_keys) >= 20:
                            break
                    
                    print(f"Found top-level keys: {', '.join(top_level_keys)}")
                    
                    # Try to sample values from some keys
                    f.seek(0)  # Reset to start of file
                    
                    # Sample from the first few keys
                    for key in top_level_keys[:5]:
                        try:
                            # Read just this key's data
                            sample_parser = ijson.items(f, key)
                            for value in sample_parser:
                                value_type = type(value).__name__
                                structure_info['types_found'][key + ': ' + value_type] += 1
                                
                                # Store example
                                if isinstance(value, (str, int, float, bool, type(None))):
                                    structure_info['value_examples'][key].append(value)
                                elif isinstance(value, (list, dict)):
                                    size = len(value)
                                    structure_info['value_examples'][key].append(f"{value_type} of size {size}")
                                    
                                    # If it's a list, try to get first item
                                    if isinstance(value, list) and len(value) > 0:
                                        first_item = value[0]
                                        if isinstance(first_item, dict):
                                            structure_info['value_examples'][key].append(f"First item keys: {list(first_item.keys())}")
                                        else:
                                            structure_info['value_examples'][key].append(f"First item type: {type(first_item).__name__}")
                                break  # Only need first value
                        except Exception as e:
                            print(f"Error sampling key {key}: {e}")
                except Exception as e:
                    print(f"Error reading object structure: {e}")
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
    
    # Now print summary information
    print("\nSTRUCTURE SUMMARY")
    print("================")
    
    # Print sample items (truncated for display)
    print("\nSample items:")
    for i, item in enumerate(structure_info['sample_items']):
        item_json = json.dumps(item, indent=2)
        if len(item_json) > 1000:
            item_json = item_json[:1000] + "... (truncated)"
        print(f"Item {i + 1}:\n{item_json}\n")
    
    # Print most common keys
    print("\nMost common keys:")
    for key, count in structure_info['keys_found'].most_common(20):
        print(f"  {key}: {count} occurrences")
    
    # Print value types
    print("\nField data types:")
    for key_type, count in structure_info['types_found'].most_common(20):
        print(f"  {key_type}: {count} occurrences")
    
    # Print example values
    print("\nExample values for common fields:")
    for key, examples in structure_info['value_examples'].items():
        print(f"  {key}: {examples}")
    
    elapsed_time = time.time() - start_time
    print(f"\nInspection completed in {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'public/results/MachineCodeAudioCommunications/results.json'
    
    # Can specify sample size with second argument
    sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    inspect_json_structure(file_path, sample_size) 