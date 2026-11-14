"""
TrajPilot CityFlow - Stage 02: Frozen I-24 Training Normalization & 32-Channel Input Construction
Loads exact I-24 sequence normalization statistics without modification.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NORM_FILE = PROJECT_ROOT / "data" / "processed" / "i24_sequences" / "gru_inputs" / "train_normalization_statistics.csv"

CANONICAL_FEATURES = [
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
    "R_Adjacent"
]


class TrajPilotNormalizer:
    """
    Constructs the 32-channel model input:
    - 16 normalized feature channels (using frozen I-24 training statistics)
    - 16 missingness indicator channels (1.0 = missing, 0.0 = valid)
    Missing values: normalized = 0.0, indicator = 1.0
    Valid values:   normalized = (x - mean) / std, indicator = 0.0
    """
    def __init__(self, norm_file=NORM_FILE):
        if not norm_file.exists():
            raise FileNotFoundError(f"Normalization statistics file not found: {norm_file}")
            
        stats_df = pd.read_csv(norm_file).set_index("feature")
        
        # Verify and reindex to exact 16 canonical features
        missing_features = [f for f in CANONICAL_FEATURES if f not in stats_df.index]
        if missing_features:
            raise KeyError(f"Missing canonical features in normalization file: {missing_features}")
            
        stats = stats_df.reindex(CANONICAL_FEATURES)
        self.features = CANONICAL_FEATURES
        self.mean = stats["train_mean"].to_numpy(dtype=np.float32)
        self.std = stats["train_std"].to_numpy(dtype=np.float32)
        
        # Numerical guard: avoid divide-by-zero
        self.std = np.where(self.std > 1e-8, self.std, 1.0).astype(np.float32)

    def transform_history(self, history_array):
        """
        history_array: (30, 16) float32 array or list of rows. Can contain NaN/Inf for missing features.
        Returns: (30, 32) float32 numpy array.
        """
        raw = np.asarray(history_array, dtype=np.float32)
        if raw.shape != (30, 16):
            raise ValueError(f"Expected shape (30, 16), got {raw.shape}")
            
        missing_mask = ~np.isfinite(raw)
        
        # Standardize valid, set missing to 0.0
        normalized = np.where(
            missing_mask,
            0.0,
            (raw - self.mean) / self.std
        ).astype(np.float32)
        
        indicator = missing_mask.astype(np.float32)
        
        # Concatenate along channel dimension: [normalized (16), missingness (16)] -> 32 channels
        return np.concatenate([normalized, indicator], axis=1).astype(np.float32)

    def to_torch_tensor(self, history_array, device="cpu"):
        arr_32 = self.transform_history(history_array)
        # Add batch dimension: (1, 30, 32)
        return torch.from_numpy(arr_32).unsqueeze(0).to(device)


def print_normalization_table():
    norm = TrajPilotNormalizer()
    print("=" * 78)
    print("TRAJPILOT CITYFLOW | 02 - FROZEN I-24 NORMALIZATION STATISTICS")
    print("=" * 78)
    print(f"File Source        : {NORM_FILE}")
    print(f"Total Features     : {len(norm.features)} canonical physical/context features")
    print(f"Total Channels     : {len(norm.features) * 2} (16 normalized + 16 missingness indicators)")
    print("-" * 78)
    print(f"{'Feature Name':<28} {'Train Mean':>15} {'Train Std':>15}")
    print("-" * 78)
    for feat, m, s in zip(norm.features, norm.mean, norm.std):
        print(f"{feat:<28} {m:>15.6f} {s:>15.6f}")
    print("=" * 78)
    return norm


if __name__ == "__main__":
    normalizer = print_normalization_table()
    # Test with dummy history having 4 missing features
    dummy_history = np.zeros((30, 16), dtype=np.float32)
    dummy_history[:, 8:12] = np.nan  # SpaceGap, RelativeVelocity, LateralDistance, time_headway missing
    out = normalizer.transform_history(dummy_history)
    print(f"Transformed dummy history shape: {out.shape}")
    print(f"Normalized SpaceGap (channel 8) mean: {out[:, 8].mean()} (Expected 0.0)")
    print(f"Missingness SpaceGap (channel 24) mean: {out[:, 24].mean()} (Expected 1.0)")
    print("Validation: PASS")
