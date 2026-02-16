
import pandas as pd
import sys

try:
    file_path = '/Users/molu/Downloads/RLVE/raw_data/think_aime24_aime24_test.parquet'
    df = pd.read_parquet(file_path)
    print(f"Columns: {df.columns.tolist()}")
    if 'prompt' in df.columns:
        sample_prompt = df['prompt'].iloc[0]
        print(f"Sample prompt: {sample_prompt}")
        print(f"Type of prompt: {type(sample_prompt)}")
    else:
        print("'prompt' column not found")
        
    if 'ground_truth' in df.columns:
        print(f"Sample ground_truth: {df['ground_truth'].iloc[0]}")

except Exception as e:
    print(f"Error: {e}")
