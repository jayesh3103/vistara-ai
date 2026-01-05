import pandas as pd
import glob
import os

base_path = "datasets"
files = glob.glob(os.path.join(base_path, "*", "*.csv"))

print(f"Scanning {len(files)} files...")

for f in files:
    try:
        df = pd.read_csv(f, nrows=5, dtype={'pincode': str})
        if 'state' in df.columns:
            # Check if state looks numeric
            states = df['state'].dropna().astype(str)
            if states.str.match(r'^\d+$').any():
                print(f"SUSPECT FILE (Numeric State): {f}")
                print(df.head())
        else:
            print(f"SUSPECT FILE (Missing 'state' header): {f}")
            print(df.head())
    except Exception as e:
        print(f"Error reading {f}: {e}")
