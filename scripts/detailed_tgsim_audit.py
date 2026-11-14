import pandas as pd
import numpy as np
from pathlib import Path
import json

datasets = [
    {
        "name": "TGSIM_I294_L2",
        "path": Path("data/raw/TGSIM_I294_L2/I294_L2_final.csv"),
        "id_col": "id",
        "time_col": "time",
        "x_col": "xloc_kf",
        "y_col": "yloc_kf",
        "speed_col": "speed_kf",
        "accel_col": "acceleration_kf",
        "lane_col": "lane_kf",
        "run_col": "run_index"
    },
    {
        "name": "TGSIM_I90_I94_MOVING",
        "path": Path("data/raw/TGSIM_I90_I94_MOVING/I90_94_moving_final.csv"),
        "id_col": "ID",
        "time_col": "time",
        "x_col": "xloc_kf",
        "y_col": "yloc_kf",
        "speed_col": "speed_kf",
        "accel_col": "acceleration_kf",
        "lane_col": "lane_kf",
        "run_col": "run_index"
    },
    {
        "name": "TGSIM_FOGGY_BOTTOM",
        "path": Path("data/raw/TGSIM_FOGGY_BOTTOM/TGSIM-Foggy-Bottom-Data.csv"),
        "id_col": "id",
        "time_col": "time",
        "x_col": "xloc_kf",
        "y_col": "yloc_kf",
        "speed_col": "speed_kf_x", # has x and y
        "accel_col": "acceleration_kf_x",
        "lane_col": "lane_kf",
        "run_col": None
    }
]

inventory_rows = []

for d in datasets:
    name = d["name"]
    path = d["path"]
    print(f"Auditing {name}...")
    
    # Read columns needed or sample if large
    if name == "TGSIM_FOGGY_BOTTOM":
        # Foggy bottom is 323MB, read chunks or needed cols
        usecols = [d["id_col"], d["time_col"], d["x_col"], d["y_col"], "speed_kf_x", "speed_kf_y", d["lane_col"]]
        df = pd.read_csv(path, usecols=usecols)
    else:
        df = pd.read_csv(path)
        
    num_rows = len(df)
    cols = list(df.columns)
    num_cols = len(cols)
    unique_ids = df[d["id_col"]].nunique()
    
    # Sampling interval
    # group by vehicle and compute dt
    sample_ids = df[d["id_col"]].unique()[:50]
    sample_sub = df[df[d["id_col"]].isin(sample_ids)]
    dt_vals = sample_sub.groupby(d["id_col"])[d["time_col"]].diff().dropna()
    dt_valid = dt_vals[dt_vals > 0]
    median_dt = float(dt_valid.median()) if len(dt_valid) > 0 else 0.1
    p95_dt = float(np.percentile(dt_valid, 95)) if len(dt_valid) > 0 else 0.1
    
    # Trajectory durations
    durations = df.groupby(d["id_col"])[d["time_col"]].agg(lambda s: s.max() - s.min())
    mean_duration = float(durations.mean())
    median_duration = float(durations.median())
    min_duration = float(durations.min())
    max_duration = float(durations.max())
    
    # Trajectories >= 6.0 seconds (needed for 3s past + 3s future)
    viable_trajs = int((durations >= 6.0).sum())
    
    # Coordinate system bounds
    x_min, x_max = float(df[d["x_col"]].min()), float(df[d["x_col"]].max())
    y_min, y_max = float(df[d["y_col"]].min()), float(df[d["y_col"]].max())
    
    # Missingness
    total_nulls = int(df.isnull().sum().sum())
    
    inv_item = {
        "dataset_name": name,
        "file_name": path.name,
        "file_size_mb": round(path.stat().st_size / (1024**2), 2),
        "total_rows": num_rows,
        "column_count": num_cols,
        "vehicle_id_col": d["id_col"],
        "timestamp_col": d["time_col"],
        "x_col": d["x_col"],
        "y_col": d["y_col"],
        "speed_col": d["speed_col"],
        "accel_col": d["accel_col"],
        "lane_col": d["lane_col"],
        "unique_vehicles": unique_ids,
        "viable_trajectories_ge_6s": viable_trajs,
        "median_dt_s": median_dt,
        "nominal_frequency_hz": round(1.0 / median_dt if median_dt > 0 else 0, 1),
        "mean_duration_s": round(mean_duration, 2),
        "median_duration_s": round(median_duration, 2),
        "max_duration_s": round(max_duration, 2),
        "x_range_m": f"[{x_min:.1f}, {x_max:.1f}]",
        "y_range_m": f"[{y_min:.1f}, {y_max:.1f}]",
        "coordinate_system": "Cartesian Local Projected (Meters)",
        "total_null_values": total_nulls
    }
    inventory_rows.append(inv_item)

inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv("data/manifests/tgsim_dataset_inventory.csv", index=False)
print("Saved data/manifests/tgsim_dataset_inventory.csv")
print(inv_df[["dataset_name", "total_rows", "unique_vehicles", "viable_trajectories_ge_6s", "median_dt_s", "nominal_frequency_hz"]].to_string())
