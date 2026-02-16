import pandas as pd
import glob
import os

# Find the first parquet file in raw_data
files = glob.glob('/Users/molu/Downloads/RLVE/raw_data/*.parquet')
if not files:
    print("No parquet files found")
    exit()

f = files[0]
print(f"Reading {f}")

try:
    df = pd.read_parquet(f)
    print("Columns:", df.columns.tolist())
    print("\nFirst row:")
    print(df.iloc[0])
    
    # Check the 'prompt' column specifically if it exists
    if 'prompt' in df.columns:
        print("\n'prompt' column first value type:", type(df.iloc[0]['prompt']))
        print("'prompt' column first value content:", df.iloc[0]['prompt'])
        
except Exception as e:
    print(f"Error reading parquet: {e}")
