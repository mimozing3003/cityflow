"""Lineage-preserving serialization helpers for held-out trajectory forecasts."""
from __future__ import annotations

import math
from typing import Mapping
import numpy as np


def build_prediction_rows(
    dataset: str,
    model_predictions,
    ground_truth,
    window_metadata: list[dict],
) -> list[dict]:
    predictions = np.asarray(model_predictions, dtype=float)
    truth = np.asarray(ground_truth, dtype=float)
    if predictions.ndim != 3 or predictions.shape[1:] != (30, 2):
        raise ValueError(f"predictions must have shape (N,30,2), got {predictions.shape}")
    if truth.shape != predictions.shape:
        raise ValueError(f"ground_truth shape {truth.shape} differs from predictions {predictions.shape}")
    if len(window_metadata) != len(predictions):
        raise ValueError("window_metadata length must match prediction window count")
    if not np.isfinite(predictions).all() or not np.isfinite(truth).all():
        raise ValueError("predictions and ground truth must be finite")

    rows = []
    for i, metadata in enumerate(window_metadata):
        required = [
            "window_id", "trajectory_id", "recording_id", "source_file",
            "source_file_sha256", "window_start_index", "history_start_time_s",
            "prediction_origin_time_s", "future_timestamps_s",
        ]
        missing = [key for key in required if key not in metadata]
        if missing:
            raise ValueError(f"window metadata row {i} missing fields: {missing}")
        future_times = list(metadata["future_timestamps_s"])
        if len(future_times) != 30:
            raise ValueError(f"window {i} must contain 30 future timestamps")
        if not all(math.isfinite(float(t)) for t in future_times):
            raise ValueError(f"window {i} has non-finite future timestamp")

        for horizon_index in range(30):
            dx_pred, dy_pred = map(float, predictions[i, horizon_index])
            dx_true, dy_true = map(float, truth[i, horizon_index])
            rows.append({
                "dataset": dataset,
                "model": str(metadata.get("model_name", "")),
                "split": "test",
                "window_id": metadata["window_id"],
                "window_index": i,
                "trajectory_id": metadata["trajectory_id"],
                "recording_id": metadata["recording_id"],
                "source_file": metadata["source_file"],
                "source_file_sha256": metadata["source_file_sha256"],
                "window_start_index": int(metadata["window_start_index"]),
                "history_start_time_s": float(metadata["history_start_time_s"]),
                "prediction_origin_time_s": float(metadata["prediction_origin_time_s"]),
                "horizon_step": horizon_index + 1,
                "horizon_seconds": float(future_times[horizon_index]) - float(metadata["prediction_origin_time_s"]),
                "target_time_s": float(future_times[horizon_index]),
                "predicted_dx_ego_m": dx_pred,
                "predicted_dy_ego_m": dy_pred,
                "ground_truth_dx_ego_m": dx_true,
                "ground_truth_dy_ego_m": dy_true,
                "displacement_error_m": math.hypot(dx_pred - dx_true, dy_pred - dy_true),
            })
    return rows


def build_window_metadata_rows(dataset: str, split: str, metadata_rows: list[dict]) -> list[dict]:
    output = []
    for index, metadata in enumerate(metadata_rows):
        output.append({
            "dataset": dataset,
            "split": split,
            "window_index": index,
            "window_id": metadata["window_id"],
            "trajectory_id": metadata["trajectory_id"],
            "recording_id": metadata["recording_id"],
            "vehicle_id": metadata.get("vehicle_id", ""),
            "source_file": metadata["source_file"],
            "source_file_sha256": metadata["source_file_sha256"],
            "window_start_index": int(metadata["window_start_index"]),
            "segment_start_index": int(metadata["segment_start_index"]),
            "segment_stop_index": int(metadata["segment_stop_index"]),
            "history_start_time_s": float(metadata["history_start_time_s"]),
            "prediction_origin_time_s": float(metadata["prediction_origin_time_s"]),
            "target_start_time_s": float(metadata["future_timestamps_s"][0]),
            "target_end_time_s": float(metadata["future_timestamps_s"][-1]),
        })
    return output
