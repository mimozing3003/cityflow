import pandas as pd
import numpy as np
from pathlib import Path

datasets = {
    "TGSIM_I294_L2": Path("data/raw/TGSIM_I294_L2/I294_L2_final.csv"),
    "TGSIM_I90_I94_MOVING": Path("data/raw/TGSIM_I90_I94_MOVING/I90_94_moving_final.csv"),
    "TGSIM_FOGGY_BOTTOM": Path("data/raw/TGSIM_FOGGY_BOTTOM/TGSIM-Foggy-Bottom-Data.csv")
}

for name, path in datasets.items():
    print("=" * 80)
    print(f"AUDITING DATASET: {name}")
    print(f"Path: {path} | Size: {path.stat().st_size / (1024**2):.2f} MB")
    print("=" * 80)
    
    # Read sample
    df_sample = pd.read_csv(path, nrows=5000)
    print(f"Columns ({len(df_sample.columns)}):")
    for col in df_sample.columns:
        dtype = str(df_sample[col].dtype)
        null_count = df_sample[col].isnull().sum()
        sample_vals = list(df_sample[col].dropna().head(3))
        print(f"  - {col:25s} | {dtype:10s} | nulls: {null_count:4d} | sample: {sample_vals}")
    print()
