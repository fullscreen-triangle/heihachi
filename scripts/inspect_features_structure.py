#!/usr/bin/env python3
import json
import sys

def inspect_json_structure(file_path):
    """Inspect the structure of a JSON file and print useful information."""
    print(f"Inspecting file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        print(f"Data type: {type(data)}")
        
        if isinstance(data, list):
            print(f"Found a list with {len(data)} items")
            
            if len(data) > 0:
                print("\nFirst item structure:")
                first_item = data[0]
                print(f"Item type: {type(first_item)}")
                
                if isinstance(first_item, dict):
                    print("Keys in first item:")
                    for key, value in first_item.items():
                        print(f"  {key}: {type(value)} - {value}")
                else:
                    print(f"Value: {first_item}")
                
        elif isinstance(data, dict):
            print(f"Found a dictionary with {len(data)} keys")
            print("\nTop-level keys:")
            for key in data.keys():
                value = data[key]
                if isinstance(value, (list, dict)):
                    if isinstance(value, list) and len(value) > 0:
                        print(f"  {key}: list with {len(value)} items")
                        if isinstance(value[0], dict):
                            print(f"    First item keys: {list(value[0].keys())}")
                        else:
                            print(f"    First item: {value[0]}")
                    elif isinstance(value, dict):
                        print(f"  {key}: dict with {len(value)} keys")
                        print(f"    Keys: {list(value.keys())}")
                else:
                    print(f"  {key}: {type(value)} - {value}")
        else:
            print(f"Unexpected data type: {type(data)}")
            
    except Exception as e:
        print(f"Error inspecting file: {e}")
        
if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'public/results/MachineCodeAudioCommunications/features.json'
    inspect_json_structure(file_path) 