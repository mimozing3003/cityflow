"""Offline audit of checked-in TrajPilot demonstration CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "raw": "curla demostrate/trajectories_raw.csv",
    "aligned": "curla demostrate/trajectories_aligned.csv",
    "valid_following": "curla demostrate/trajectories_valid_following.csv",
    "predictions": "curla demostrate/step14_live_predictions.csv",
}


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows


def numeric(rows, key):
    result = []
    for index, row in enumerate(rows):
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric {key!r} at CSV row {index + 2}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Non-finite {key!r} at CSV row {index + 2}")
        result.append(value)
    return result


def describe(values):
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": values[0],
        "median": values[(len(values) - 1) // 2],
        "mean": mean(values),
        "max": values[-1],
    }


def unique_rounded(values):
    return sorted({round(float(value), 6) for value in values})


def heading_difference(a, b):
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def contiguous_runs(flags, steps):
    runs, start = [], None
    for index, valid in enumerate(flags):
        if valid and start is None:
            start = index
        if start is not None and (not valid or index == len(flags) - 1):
            end = index if valid and index == len(flags) - 1 else index - 1
            count = end - start + 1
            runs.append({
                "start_step": steps[start],
                "end_step": steps[end],
                "rows": count,
                "duration_seconds": round(count * 0.1, 6),
            })
            start = None
    return runs


def require_columns(rows, columns, label):
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise ValueError(f"{label} CSV missing columns: {missing}")


def audit_demo_data(root=ROOT):
    root = Path(root)
    paths = {name: root / relative for name, relative in FILES.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} CSV: {path}")

    raw = read_csv(paths["raw"])
    aligned = read_csv(paths["aligned"])
    selected = read_csv(paths["valid_following"])
    predictions = read_csv(paths["predictions"])

    require_columns(raw, ["step", "sim_time", "ego_x", "ego_y", "lead_x", "lead_y"], "raw")
    aligned_cols = [
        "step", "frame", "sim_time_s", "elapsed_from_start_s", "road_id", "lane_id",
        "ego_yaw_deg", "lead_yaw_deg", "ego_speed_mps", "lead_speed_mps",
        "longitudinal_gap_m", "lateral_offset_m", "relative_longitudinal_velocity_mps",
    ]
    require_columns(aligned, aligned_cols, "aligned")
    require_columns(selected, list(aligned[0]), "valid_following")
    require_columns(predictions, [
        "horizon_step", "horizon_seconds", "pred_lon_m", "pred_lat_m",
        "gt_lon_m", "gt_lat_m", "displacement_error_m",
    ], "predictions")

    raw_time = numeric(raw, "sim_time")
    raw_steps = [int(v) for v in numeric(raw, "step")]
    raw_dt = [b - a for a, b in zip(raw_time, raw_time[1:])]

    elapsed = numeric(aligned, "elapsed_from_start_s")
    sim_time = numeric(aligned, "sim_time_s")
    frames = [int(v) for v in numeric(aligned, "frame")]
    steps = [int(v) for v in numeric(aligned, "step")]
    aligned_dt = [b - a for a, b in zip(elapsed, elapsed[1:])]
    frame_deltas = [b - a for a, b in zip(frames, frames[1:])]

    lateral = numeric(aligned, "lateral_offset_m")
    ego_yaw = numeric(aligned, "ego_yaw_deg")
    lead_yaw = numeric(aligned, "lead_yaw_deg")
    gaps = numeric(aligned, "longitudinal_gap_m")
    heading = [heading_difference(a, b) for a, b in zip(ego_yaw, lead_yaw)]
    lateral_limit, heading_limit = 2.5, 15.0
    flags = [
        gaps[i] > 0 and abs(lateral[i]) <= lateral_limit and heading[i] <= heading_limit
        for i in range(len(aligned))
    ]
    runs = contiguous_runs(flags, steps)

    selected_is_prefix = (
        len(selected) <= len(aligned)
        and all(
            all(selected[i].get(key) == aligned[i].get(key) for key in aligned[0])
            for i in range(len(selected))
        )
    )
    lane_pairs = sorted({f"{r['road_id']}/{r['lane_id']}" for r in aligned})

    pred_steps = [int(v) for v in numeric(predictions, "horizon_step")]
    horizon_seconds = numeric(predictions, "horizon_seconds")
    pred_lon = numeric(predictions, "pred_lon_m")
    pred_lat = numeric(predictions, "pred_lat_m")
    gt_lon = numeric(predictions, "gt_lon_m")
    gt_lat = numeric(predictions, "gt_lat_m")
    stored = numeric(predictions, "displacement_error_m")
    recomputed = [math.hypot(px - gx, py - gy) for px, py, gx, gy in zip(pred_lon, pred_lat, gt_lon, gt_lat)]
    residual = [abs(a - b) for a, b in zip(stored, recomputed)]

    def endpoint(step):
        indices = [i for i, value in enumerate(pred_steps) if value == step]
        if len(indices) != 1:
            return None
        i = indices[0]
        return {
            "step": pred_steps[i],
            "seconds": horizon_seconds[i],
            "stored_error_m": stored[i],
            "recomputed_error_m": recomputed[i],
        }

    return {
        "scope": "checked-in demonstration logs only; not the I-24/TGSIM training dataset",
        "files": {name: {"path": FILES[name], "rows": len(data)} for name, data in [
            ("raw", raw), ("aligned", aligned), ("valid_following", selected), ("predictions", predictions)
        ]},
        "raw_log": {
            "rows": len(raw),
            "step_deltas": unique_rounded([b - a for a, b in zip(raw_steps, raw_steps[1:])]),
            "simulation_time_deltas_seconds": unique_rounded(raw_dt),
            "start_simulation_time_seconds": raw_time[0],
            "end_simulation_time_seconds": raw_time[-1],
        },
        "aligned_log": {
            "rows": len(aligned),
            "frame_deltas": sorted(set(frame_deltas)),
            "elapsed_time_deltas_seconds": unique_rounded(aligned_dt),
            "start_simulation_time_seconds": sim_time[0],
            "end_simulation_time_seconds": sim_time[-1],
            "mapped_road_lane_pairs": lane_pairs,
            "mapped_road_lane_pair_count": len(lane_pairs),
            "lateral_offset_m": describe(lateral),
            "absolute_heading_difference_deg": describe(heading),
            "longitudinal_gap_m": describe(gaps),
        },
        "valid_following_subset": {
            "rows": len(selected),
            "is_exact_prefix_of_aligned_log": selected_is_prefix,
            "fraction_of_aligned_rows": len(selected) / len(aligned),
        },
        "heuristic_following_screen": {
            "criteria": {
                "positive_longitudinal_gap": True,
                "absolute_lateral_offset_max_m": lateral_limit,
                "absolute_heading_difference_max_deg": heading_limit,
            },
            "screened_rows": len(flags),
            "passing_rows": sum(flags),
            "passing_fraction": sum(flags) / len(flags),
            "contiguous_passing_runs": runs,
            "scope_warning": "Heuristic only; not ground-truth leader labeling.",
        },
        "prediction_demo": {
            "rows": len(predictions),
            "horizon_steps": pred_steps,
            "horizon_seconds_min_max": [min(horizon_seconds), max(horizon_seconds)],
            "stored_error_m": describe(stored),
            "recomputed_error_m": describe(recomputed),
            "max_stored_vs_recomputed_error_delta_m": max(residual),
            "fde_1s": endpoint(10),
            "fde_2s": endpoint(20),
            "fde_3s": endpoint(30),
            "predicted_longitudinal_travel_3s_m": pred_lon[-1],
            "observed_longitudinal_travel_3s_m": gt_lon[-1],
            "longitudinal_travel_overprediction_ratio": pred_lon[-1] / gt_lon[-1] if gt_lon[-1] else None,
        },
        "lineage_checks": {
            "raw_and_aligned_have_same_column_schema": set(raw[0]) == set(aligned[0]),
            "raw_and_aligned_simulation_time_ranges_overlap": not (
                raw_time[-1] < sim_time[0] or sim_time[-1] < raw_time[0]
            ),
            "raw_to_aligned_transform_proven": False,
            "step14_prediction_to_aligned_log_join_proven": False,
        },
        "interpretation": [
            "Raw, aligned, valid-following, and prediction files are distinct demonstration artifacts.",
            "The valid-following file is an exact prefix of the aligned log; its separate selection lineage is not established.",
            "Matching mapped road/lane IDs alone does not prove that the selected leader remained in the ego vehicle's travel corridor.",
            "The Step 14 prediction log is a separate single-run demo and must not be combined with the aligned 15-second trajectory log.",
            "These demonstration files do not establish training, validation, or held-out test sample lineage.",
        ],
    }


def write_report(report, json_path, markdown_path):
    json_path, markdown_path = Path(json_path), Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    aligned, subset = report["aligned_log"], report["valid_following_subset"]
    screen, pred = report["heuristic_following_screen"], report["prediction_demo"]
    lines = [
        "# TrajPilot CityFlow — Demonstration Trajectory Lineage Audit",
        "",
        "## Scope",
        "",
        "This report audits checked-in demonstration CSVs only. It does not treat them as the I-24/TGSIM model training dataset.",
        "",
        "## Findings",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Raw trajectory rows | {report['raw_log']['rows']} |",
        f"| Aligned trajectory rows | {aligned['rows']} |",
        f"| Aligned frame increments | {aligned['frame_deltas']} |",
        f"| Aligned elapsed-time increments (s) | {aligned['elapsed_time_deltas_seconds']} |",
        f"| Valid-following rows | {subset['rows']} |",
        f"| Valid-following file equals aligned prefix | {subset['is_exact_prefix_of_aligned_log']} |",
        f"| Heuristic following screen | {screen['passing_rows']} / {screen['screened_rows']} |",
        f"| 3-second single-run FDE | {pred['fde_3s']['stored_error_m']:.4f} m |",
        "",
        "## Interpretation",
        "",
        f"The aligned log's lateral-offset range is {aligned['lateral_offset_m']['min']:.3f} m to {aligned['lateral_offset_m']['max']:.3f} m. A mapped road/lane identifier alone is not sufficient evidence of a valid leader association.",
        "",
        f"The valid-following file is an exact prefix of the first {subset['rows']} aligned rows. The file contents do not establish why the rest were omitted. A separate heuristic screen requiring positive gap, lateral offset no greater than 2.5 m, and heading difference no greater than 15 degrees passes {screen['passing_rows']} rows. This is not ground-truth labeling.",
        "",
        f"The Step 14 single-run demo predicts {pred['predicted_longitudinal_travel_3s_m']:.3f} m of longitudinal travel against {pred['observed_longitudinal_travel_3s_m']:.3f} m observed, with an overprediction ratio of {pred['longitudinal_travel_overprediction_ratio']:.3f}. Its endpoint error is {pred['fde_3s']['stored_error_m']:.4f} m.",
        "",
        "## Research consequences",
        "",
        "1. Keep the raw, aligned, valid-following, and prediction files distinct until a run/recording ID lineage is supplied.",
        "2. Do not count individual rows in one contiguous log as independent samples.",
        "3. Document the leader association rule, including lane topology, bumper geometry, gap definition, velocity sign, and junction handling.",
        "4. Generate windows only after target semantics and recording-level splits are audited.",
        "5. Do not use successful demo execution status as an accuracy claim.",
        "",
        "## Reproduction",
        "",
        "    python scripts/audit_demo_trajectory_lineage.py",
        "    python -m unittest discover -s tests -v",
        "",
        "The JSON output is reports/demo_trajectory_lineage_audit.json. This audit does not certify I-24/TGSIM data quality, live CARLA behavior, or paper readiness.",
    ]
    markdown_path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit_demo_data(root)
    write_report(
        report,
        root / "reports" / "demo_trajectory_lineage_audit.json",
        root / "reports" / "DEMO_TRAJECTORY_LINEAGE_AUDIT.md",
    )
    print(json.dumps({
        "raw_rows": report["raw_log"]["rows"],
        "aligned_rows": report["aligned_log"]["rows"],
        "valid_following_rows": report["valid_following_subset"]["rows"],
        "valid_following_is_prefix": report["valid_following_subset"]["is_exact_prefix_of_aligned_log"],
        "heuristic_passing_rows": report["heuristic_following_screen"]["passing_rows"],
        "step14_fde_3s_m": report["prediction_demo"]["fde_3s"]["stored_error_m"],
        "json_output": "reports/demo_trajectory_lineage_audit.json",
        "markdown_output": "reports/DEMO_TRAJECTORY_LINEAGE_AUDIT.md",
    }, indent=2))


if __name__ == "__main__":
    main()
