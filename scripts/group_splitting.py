"""Deterministic trajectory- or recording-group disjoint train/val/test splits."""
from __future__ import annotations

import random
from collections.abc import Hashable, Sequence


def split_group_keys(
    group_keys: Sequence[Hashable],
    group_to_recording: dict[Hashable, str],
    split_unit: str = "trajectory",
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[dict[str, set[Hashable]], dict]:
    """Assign trajectory keys while keeping the selected split unit disjoint.

    For split_unit='recording', every trajectory from one recording/run is
    assigned to exactly one partition. At least three distinct recording IDs
    are required so train, validation, and test can all contain runs.
    """
    if split_unit not in {"trajectory", "recording"}:
        raise ValueError("split_unit must be 'trajectory' or 'recording'")
    if not group_keys:
        raise ValueError("group_keys must not be empty")
    if any(value < 0 for value in (train_fraction, val_fraction, test_fraction)):
        raise ValueError("split fractions must be non-negative")
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test fractions must sum to one")

    rng = random.Random(seed)
    if split_unit == "trajectory":
        shuffled = list(group_keys)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(train_fraction * n)
        n_val = int(val_fraction * n)
        split = {
            "train": set(shuffled[:n_train]),
            "val": set(shuffled[n_train:n_train+n_val]),
            "test": set(shuffled[n_train+n_val:]),
        }
        audit = {
            "split_unit": "trajectory",
            "trajectory_count": n,
            "recording_count": len({group_to_recording.get(k, "") for k in shuffled if group_to_recording.get(k, "")}),
            "recording_split_overlap_count": None,
            "recording_disjoint": False,
            "recording_disjoint_reason": "Trajectory-level split selected; recordings may appear in multiple partitions.",
        }
        return split, audit

    missing = [key for key in group_keys if not str(group_to_recording.get(key, "")).strip()]
    if missing:
        raise ValueError(
            f"Recording-disjoint split requested but {len(missing)} trajectory groups lack a recording/run ID."
        )

    recording_ids = sorted({str(group_to_recording[key]).strip() for key in group_keys})
    if len(recording_ids) < 3:
        raise ValueError(
            f"Recording-disjoint train/val/test requires at least 3 recording IDs; found {len(recording_ids)}."
        )

    rng.shuffle(recording_ids)
    n_recordings = len(recording_ids)
    n_val = max(1, int(round(val_fraction * n_recordings)))
    n_test = max(1, int(round(test_fraction * n_recordings)))
    if n_val + n_test >= n_recordings:
        n_val, n_test = 1, 1
    n_train = n_recordings - n_val - n_test
    if n_train < 1:
        raise ValueError("Not enough recording IDs to populate train, validation and test.")

    train_recordings = set(recording_ids[:n_train])
    val_recordings = set(recording_ids[n_train:n_train+n_val])
    test_recordings = set(recording_ids[n_train+n_val:])

    split = {
        "train": {key for key in group_keys if str(group_to_recording[key]).strip() in train_recordings},
        "val": {key for key in group_keys if str(group_to_recording[key]).strip() in val_recordings},
        "test": {key for key in group_keys if str(group_to_recording[key]).strip() in test_recordings},
    }
    recording_sets = {
        "train": train_recordings,
        "val": val_recordings,
        "test": test_recordings,
    }
    overlap = (
        (recording_sets["train"] & recording_sets["val"])
        | (recording_sets["train"] & recording_sets["test"])
        | (recording_sets["val"] & recording_sets["test"])
    )
    if overlap:
        raise RuntimeError(f"Internal recording leakage detected: {sorted(overlap)}")
    if any(not split[name] for name in ("train", "val", "test")):
        raise RuntimeError("A recording split produced an empty trajectory partition.")

    return split, {
        "split_unit": "recording",
        "trajectory_count": len(group_keys),
        "recording_count": n_recordings,
        "recording_split_overlap_count": len(overlap),
        "recording_disjoint": True,
        "train_recording_ids": sorted(train_recordings),
        "val_recording_ids": sorted(val_recordings),
        "test_recording_ids": sorted(test_recordings),
        "recording_disjoint_reason": "All trajectories from each source recording/run are assigned to one partition.",
    }
