
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ['vx_causal', 'vy_causal', 'speed_derived', 'acceleration_native', 'acceleration_causal_0p5s', 'turn_rate_valid', 'curvature_valid', 'jerk_causal', 'SpaceGap', 'RelativeVelocity', 'LateralDistance', 'time_headway', 'L_Approach', 'R_Approach', 'L_Adjacent', 'R_Adjacent']

SEQ_DIR = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\data\processed\i24_sequences")
NORMALIZATION_PATH = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\data\processed\i24_sequences\gru_inputs\train_normalization_statistics.csv")

normalization_df = pd.read_csv(
    NORMALIZATION_PATH
)

feature_column = (
    "feature"
    if "feature" in normalization_df.columns
    else normalization_df.columns[0]
)

mean_column = next(
    c for c in normalization_df.columns
    if c.lower() in {"mean", "train_mean", "mu"}
)

std_column = next(
    c for c in normalization_df.columns
    if c.lower() in {"std", "train_std", "sigma"}
)

normalization_df = (
    normalization_df
    .set_index(feature_column)
    .reindex(FEATURES)
)

TRAIN_MEANS = normalization_df[
    mean_column
].to_numpy(dtype=np.float64)

TRAIN_STDS = normalization_df[
    std_column
].to_numpy(dtype=np.float64)

TRAIN_STDS = np.where(
    np.isfinite(TRAIN_STDS) &
    (TRAIN_STDS > 0),
    TRAIN_STDS,
    1.0
)

def exact_16_to_32(x16):

    x16 = np.asarray(
        x16,
        dtype=np.float32
    )

    if x16.ndim == 2:
        x16 = x16[None, ...]

    finite_mask = np.isfinite(x16)

    normalized = np.zeros_like(
        x16,
        dtype=np.float32
    )

    normalized[finite_mask] = (
        (
            x16[finite_mask]
            -
            np.broadcast_to(
                TRAIN_MEANS.astype(np.float32),
                x16.shape
            )[finite_mask]
        )
        /
        np.broadcast_to(
            TRAIN_STDS.astype(np.float32),
            x16.shape
        )[finite_mask]
    )

    missing_mask = (
        ~finite_mask
    ).astype(np.float32)

    return np.concatenate(
        [normalized, missing_mask],
        axis=-1
    )
