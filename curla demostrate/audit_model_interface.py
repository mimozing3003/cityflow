from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import torch

ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow")
DEMO = ROOT / "curla demostrate"

CHECKPOINT = ROOT / "data" / "models" / "gru_baseline" / "best_gru_baseline.pt"
NORMALIZER = (
    ROOT / "data" / "processed" / "i24_sequences"
    / "gru_inputs" / "train_normalization_statistics.csv"
)
FEATURES = (
    ROOT / "data" / "processed" / "i24_sequences"
    / "feature_names.npy"
)
ADAPTER = ROOT / "carla_eval" / "carla_adapter.py"
TRAJECTORY = DEMO / "trajectories_valid_following.csv"

EXPECTED_FEATURES = [
    "vx_causal",
    "vy_causal",
    "speed_derived",
    "acceleration_native",
    "acceleration_causal_0p5s",
    "turn_rate_valid",
    "curvature_valid",
    "jerk_causal",
    "SpaceGap",
    "RelativeVelocity",
    "LateralDistance",
    "time_headway",
    "L_Approach",
    "R_Approach",
    "L_Adjacent",
    "R_Adjacent",
]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

print("=" * 72)
print("TRAJPILOT CITYFLOW | FROZEN MODEL INTERFACE AUDIT")
print("=" * 72)

for path in [CHECKPOINT, NORMALIZER, FEATURES, ADAPTER, TRAJECTORY]:
    print(f"{'EXISTS' if path.exists() else 'MISSING'}: {path}")

if not all(path.exists() for path in [CHECKPOINT, NORMALIZER, FEATURES, ADAPTER, TRAJECTORY]):
    raise FileNotFoundError("One or more required audit files are missing.")

print("\n[1] FROZEN ARTIFACT HASHES")
print("Checkpoint SHA-256:", sha256(CHECKPOINT))
print("Normalizer SHA-256:", sha256(NORMALIZER))
print("Feature file SHA-256:", sha256(FEATURES))

print("\n[2] FEATURE ORDER")
features = np.load(FEATURES, allow_pickle=True).tolist()
features = [str(feature) for feature in features]

print("Feature count:", len(features))
for index, feature in enumerate(features):
    print(f"{index:02d}: {feature}")

print("Matches expected 16-feature order:", features == EXPECTED_FEATURES)

print("\n[3] NORMALIZATION STATISTICS")
stats = pd.read_csv(NORMALIZER)
print("Shape:", stats.shape)
print("Columns:", stats.columns.tolist())
print(stats.head(25).to_string(index=False))

print("\n[4] CHECKPOINT METADATA")
checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("Checkpoint object type:", type(checkpoint).__name__)

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys()))

    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"{key}: mapping with {len(value)} entries")
            tensor_items = [
                (name, tuple(item.shape))
                for name, item in value.items()
                if hasattr(item, "shape")
            ]
            for name, shape in tensor_items[:25]:
                print(f"  {name}: {shape}")
        elif hasattr(value, "shape"):
            print(f"{key}: tensor shape={tuple(value.shape)}")
        elif isinstance(value, (str, int, float, bool, list, tuple)):
            print(f"{key}: {str(value)[:300]}")

print("\n[5] ADAPTER SOURCE")
with ADAPTER.open("r", encoding="utf-8-sig") as file:
    source_lines = file.readlines()

keywords = (
    "FEATURES_16",
    "normal",
    "mean",
    "std",
    "nan",
    "SpaceGap",
    "RelativeVelocity",
    "LateralDistance",
    "time_headway",
    "state_dict",
    "load_state_dict",
    "predict",
    "forward",
)

selected = set()

for index, line in enumerate(source_lines):
    if any(keyword.lower() in line.lower() for keyword in keywords):
        selected.update(range(max(0, index - 2), min(len(source_lines), index + 4)))

for index in sorted(selected):
    print(f"{index + 1:04d}: {source_lines[index].rstrip()}")

print("\n[6] VALID TRAJECTORY LENGTH")
trajectory = pd.read_csv(TRAJECTORY)
print("Shape:", trajectory.shape)
print("Columns:", trajectory.columns.tolist())

if "step" in trajectory.columns:
    print("First step:", trajectory["step"].iloc[0])
    print("Last step:", trajectory["step"].iloc[-1])

print("\nAudit complete. No artifacts were modified.")
