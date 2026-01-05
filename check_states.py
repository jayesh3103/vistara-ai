import pandas as pd
from data_processor import load_and_merge_data

print("Loading data to list unique states...")
# We use the existing loader which already does some basic stripping/upper casing
df = load_and_merge_data()
states = sorted(df['state'].unique())
print("\n--- Unique States Found ---")
for s in states:
    print(s)
print("---------------------------")
