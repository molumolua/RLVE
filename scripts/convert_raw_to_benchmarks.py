import os
import json
import pandas as pd
import numpy as np

def extract_user_prompt(prompt_field):
    """
    Extracts the user prompt from the prompt field.
    Handles list of dicts (chat format) or other formats.
    """
    if hasattr(prompt_field, 'tolist'):
        prompt_field = prompt_field.tolist()
        
    if isinstance(prompt_field, list):
        # Look for the last message with role 'user'
        for message in reversed(prompt_field):
            if isinstance(message, dict) and message.get('role') == 'user':
                return message.get('content')
        # If no user role found, return the last string if it's a list of strings
        if len(prompt_field) > 0 and isinstance(prompt_field[-1], str):
            return prompt_field[-1]
    
    # If it's already a string or something else, return as is
    return str(prompt_field)

def convert_parquet_to_json(source_dir, target_dir):
    """
    Converts all parquet files in source_dir to json files in target_dir.
    Extracts user_prompt from prompt list and keeps ground_truth.
    """
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created target directory: {target_dir}")

    files = [f for f in os.listdir(source_dir) if f.endswith('.parquet')]
    
    if not files:
        print(f"No parquet files found in {source_dir}")
        return

    print(f"Found {len(files)} parquet files to convert.")

    for filename in files:
        source_path = os.path.join(source_dir, filename)
        target_filename = os.path.splitext(filename)[0] + '.json'
        target_path = os.path.join(target_dir, target_filename)
        
        print(f"Processing {filename}...")
        
        try:
            # Read parquet file
            df = pd.read_parquet(source_path)
            
            # Check required columns
            if 'prompt' not in df.columns:
                print(f"Skipping {filename}: 'prompt' column not found.")
                continue
            
            # Transform data
            records = []
            for _, row in df.iterrows():
                record = {}
                
                # Extract user_prompt
                prompt_val = row['prompt']
                record['user_prompt'] = extract_user_prompt(prompt_val)
                
                # Keep ground_truth if it exists
                if 'ground_truth' in row:
                    record['ground_truth'] = row['ground_truth']
                
                records.append(record)
            
            # Write to JSON
            with open(target_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4, ensure_ascii=False)
            
            print(f"Saved {target_filename}")
            
        except Exception as e:
            print(f"Error converting {filename}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    SOURCE_DIR = "/Users/molu/Downloads/RLVE/raw_data"
    TARGET_DIR = "/Users/molu/Downloads/RLVE/data/BENCHMARKS"
    
    convert_parquet_to_json(SOURCE_DIR, TARGET_DIR)
