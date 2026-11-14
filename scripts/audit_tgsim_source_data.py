"""Audit an actual local TGSIM source CSV without guessing its schema.

The script uses explicit aliases from the published FHWA dictionary and common
normalized spellings. It does not transform coordinates or certify I-24 parity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import median


SCHEMAS = {
    "TGSIM_I294_L2": {
        "fields": {
            "track_id": ["ID", "id"],
            "time_s": ["time"],
            "x_m": ["xloc-kf", "xloc_kf"],
            "y_m": ["yloc-kf", "yloc_kf"],
            "lane_id": ["lane-kf", "lane_kf"],
            "speed_mps": ["speed-kf", "speed_kf"],
            "acceleration_mps2": ["acceleration-kf", "acceleration_kf"],
            "length_m": ["length-smoothed", "length_smoothed"],
            "width_m": ["width-smoothed", "width_smoothed"],
            "road_user_type": ["type-most-common", "type_most_common"],
            "acc_flag": ["ACC", "acc"],
            "recording_id": ["run_index", "run-index", "run_id", "recording_id"],
        },
        "expected_vectors": [],
        "notes": [
            "I-294 L2 is a distinct product; do not merge with other TGSIM products by name alone.",
            "Published lane IDs are direction-specific: southbound -2..-5 and northbound 2..5.",
            "The documented overpass interpolation mask below assumes run_index has the same values as the FHWA report.",
        ],
    },
    "TGSIM_I90_I94_MOVING": {
        "fields": {
            "track_id": ["ID", "id"],
            "time_s": ["time"],
            "x_m": ["xloc-kf", "xloc_kf"],
            "y_m": ["yloc-kf", "yloc_kf"],
            "lane_id": ["lane-kf", "lane_kf"],
            "speed_mps": ["speed-kf", "speed_kf"],
            "acceleration_mps2": ["acceleration-kf", "acceleration_kf"],
            "length_m": ["length-smoothed", "length_smoothed"],
            "width_m": ["width-smoothed", "width_smoothed"],
            "road_user_type": ["type-most-common", "type_most_common"],
            "acc_flag": ["ACC", "acc"],
            "recording_id": ["run_index", "run-index", "run_id", "recording_id"],
        },
        "expected_vectors": [],
        "notes": [
            "This schema is for the moving-helicopter product, not the separate stationary product.",
            "The moving product is northbound; retain the source run identifier.",
        ],
    },
    "TGSIM_FOGGY_BOTTOM": {
        "fields": {
            "track_id": ["ID", "id"],
            "time_s": ["time"],
            "x_m": ["xloc-kf", "xloc_kf"],
            "y_m": ["yloc-kf", "yloc_kf"],
            "lane_id": ["lane-kf", "lane_kf"],
            "speed_x_mps": ["speed-kf-x", "speed_kf_x"],
            "speed_y_mps": ["speed-kf-y", "speed_kf_y"],
            "acceleration_x_mps2": ["acceleration-kf-x", "acceleration_kf_x"],
            "acceleration_y_mps2": ["acceleration-kf-y", "acceleration_kf_y"],
            "length_m": ["length-smoothed", "length_smoothed"],
            "width_m": ["width-smoothed", "width_smoothed"],
            "road_user_type": ["type-most-common", "type_most_common"],
            "recording_id": ["camera_id", "camera-id", "recording_id", "run_index", "run_id", "session_id"],
        },
        "expected_vectors": [
            ["speed_x_mps", "speed_y_mps"],
            ["acceleration_x_mps2", "acceleration_y_mps2"],
        ],
        "notes": [
            "Foggy Bottom is a grid-road, multi-road-user dataset; lane-kf is a road-segment/region code, not a highway lane number.",
            "Keep x/y velocity and acceleration components separate until a road-direction transform is specified.",
            "Apply an explicit road-user-type filter for vehicle-only experiments.",
        ],
    },
}

REQUIRED = ["track_id", "time_s", "x_m", "y_m"]
DT_SECONDS = 0.1
DT_TOLERANCE_SECONDS = 0.025
WINDOW_SPAN_SECONDS = 5.9

I294_INTERPOLATION_RANGES = {
    1: [(2130.0, 2170.0), (3330.0, 3370.0), (3730.0, 3770.0)],
    3: [(1530.0, 1570.0), (2730.0, 2770.0), (3130.0, 3170.0)],
    7: [(1530.0, 1570.0), (2730.0, 2770.0), (3130.0, 3170.0)],
    9: [(1530.0, 1570.0), (2730.0, 2770.0), (3130.0, 3170.0)],
    11: [(1530.0, 1570.0), (2730.0, 2770.0), (3130.0, 3170.0)],
    19: [(1530.0, 1570.0), (2730.0, 2770.0), (3130.0, 3170.0)],
    21: [(1530.0, 1570.0), (2730.0, 2770.0), (3130.0, 3170.0)],
    8: [(1760.0, 1800.0), (2160.0, 2200.0), (3360.0, 3400.0)],
    18: [(1760.0, 1800.0), (2160.0, 2200.0), (3360.0, 3400.0)],
    20: [(1760.0, 1800.0), (2160.0, 2200.0), (3360.0, 3400.0)],
}


class SchemaError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_fields(headers, schema):
    resolved, missing, ambiguous = {}, [], {}
    header_set = set(headers)
    for semantic, aliases in schema["fields"].items():
        matches = [alias for alias in aliases if alias in header_set]
        if len(matches) > 1:
            ambiguous[semantic] = matches
        elif len(matches) == 1:
            resolved[semantic] = matches[0]
        else:
            missing.append(semantic)
    if missing:
        required_missing = [name for name in REQUIRED if name in missing]
        vector_missing = []
        for vector in schema.get("expected_vectors", []):
            absent = [name for name in vector if name not in resolved]
            if absent:
                vector_missing.extend(absent)
        if required_missing or vector_missing:
            raise SchemaError(
                f"Required semantics unresolved. Missing required={required_missing}; "
                f"missing vector components={vector_missing}; all missing={missing}; "
                f"headers={headers}"
            )
    if ambiguous:
        raise SchemaError(f"Ambiguous semantic aliases (select one explicitly): {ambiguous}")
    return resolved, missing


def _finite_float(value, semantic, row_number, allow_blank=True):
    if value is None or value.strip() == "":
        if allow_blank:
            return None
        raise ValueError(f"blank required value: {semantic}")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"non-numeric value: {semantic}={value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {semantic}={value!r}")
    return result


def _recording_number(value):
    if value is None:
        return None
    try:
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
    except (TypeError, ValueError):
        return None
    return None


def _is_interpolated(dataset, recording_number, x_value):
    if dataset != "TGSIM_I294_L2" or recording_number is None:
        return False
    intervals = I294_INTERPOLATION_RANGES.get(recording_number, [])
    return any(low <= x_value <= high for low, high in intervals)


def audit_csv(dataset: str, csv_path: Path, duplicate_check=True) -> dict:
    if dataset not in SCHEMAS:
        raise ValueError(f"Unsupported dataset key: {dataset}. Choose from {sorted(SCHEMAS)}")
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    schema = SCHEMAS[dataset]
    file_hash = sha256_file(csv_path)
    byte_size = csv_path.stat().st_size

    counts = {
        "rows": 0,
        "blank_required_values": 0,
        "invalid_numeric_rows": 0,
        "duplicate_track_timestamp_keys": 0,
        "non_increasing_track_timestamps": 0,
        "irregular_intervals": 0,
        "road_user_type_counts": {},
        "interpolated_overpass_rows": 0,
    }
    track_stats = {}
    duplicate_keys = set() if duplicate_check else None
    dt_values_sample = []
    x_min = x_max = y_min = y_max = None
    recording_ids = set()
    first_time = last_time = None
    headers = []
    resolved = {}
    missing_optional = []
    invalid_examples = []

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SchemaError("CSV has no header row")
        headers = list(reader.fieldnames)
        resolved, missing_optional = resolve_fields(headers, schema)

        for row_number, row in enumerate(reader, start=2):
            counts["rows"] += 1
            try:
                track_id = row[resolved["track_id"]].strip()
                if not track_id:
                    raise ValueError("blank track_id")
                timestamp = _finite_float(row[resolved["time_s"]], "time_s", row_number, False)
                x_value = _finite_float(row[resolved["x_m"]], "x_m", row_number, False)
                y_value = _finite_float(row[resolved["y_m"]], "y_m", row_number, False)
            except (ValueError, KeyError) as exc:
                counts["invalid_numeric_rows"] += 1
                if len(invalid_examples) < 20:
                    invalid_examples.append({"row": row_number, "error": str(exc)})
                continue

            record_value = row.get(resolved.get("recording_id", ""), "").strip() if "recording_id" in resolved else ""
            recording_key = record_value or "__recording_id_missing__"
            if record_value:
                recording_ids.add(record_value)
            composite_track = (recording_key, track_id)
            stats = track_stats.setdefault(
                composite_track,
                {"first_time": timestamp, "last_seen_time": timestamp,
                 "min_time": timestamp, "max_time": timestamp, "rows": 0,
                 "non_increasing": 0, "irregular": 0}
            )
            stats["rows"] += 1
            stats["min_time"] = min(stats["min_time"], timestamp)
            stats["max_time"] = max(stats["max_time"], timestamp)
            dt = timestamp - stats["last_seen_time"]
            if stats["rows"] > 1:
                if dt <= 0:
                    stats["non_increasing"] += 1
                    counts["non_increasing_track_timestamps"] += 1
                else:
                    if len(dt_values_sample) < 50000:
                        dt_values_sample.append(dt)
                    if abs(dt - DT_SECONDS) > DT_TOLERANCE_SECONDS:
                        stats["irregular"] += 1
                        counts["irregular_intervals"] += 1
            stats["last_seen_time"] = timestamp

            if duplicate_keys is not None:
                key = (recording_key, track_id, timestamp)
                if key in duplicate_keys:
                    counts["duplicate_track_timestamp_keys"] += 1
                else:
                    duplicate_keys.add(key)

            x_min = x_value if x_min is None else min(x_min, x_value)
            x_max = x_value if x_max is None else max(x_max, x_value)
            y_min = y_value if y_min is None else min(y_min, y_value)
            y_max = y_value if y_max is None else max(y_max, y_value)

            type_field = resolved.get("road_user_type")
            if type_field:
                type_value = row.get(type_field, "").strip() or "__missing__"
                counts["road_user_type_counts"][type_value] = counts["road_user_type_counts"].get(type_value, 0) + 1

            if dataset == "TGSIM_I294_L2" and "recording_id" in resolved:
                run_number = _recording_number(record_value)
                if run_number is not None and _is_interpolated(dataset, run_number, x_value):
                    counts["interpolated_overpass_rows"] += 1

            if first_time is None:
                first_time = timestamp
            last_time = timestamp

    track_durations = [s["max_time"] - s["min_time"] for s in track_stats.values()]
    viable_tracks = sum(duration >= WINDOW_SPAN_SECONDS for duration in track_durations)
    recording_id_available = "recording_id" in resolved and bool(recording_ids)
    dt_sorted = sorted(dt_values_sample)
    dt_summary = {
        "sample_count": len(dt_sorted),
        "median_seconds": median(dt_sorted) if dt_sorted else None,
        "min_seconds": dt_sorted[0] if dt_sorted else None,
        "max_seconds": dt_sorted[-1] if dt_sorted else None,
        "sample_cap": 50000,
    }
    schema_status = "PASS" if not missing_optional else "PARTIAL_OPTIONAL_FIELDS_MISSING"
    if counts["invalid_numeric_rows"] or counts["duplicate_track_timestamp_keys"] or counts["non_increasing_track_timestamps"]:
        structural_status = "FAIL"
    else:
        structural_status = "PASS_WITH_WARNINGS" if counts["irregular_intervals"] else "PASS"

    return {
        "dataset": dataset,
        "source_file": {
            "path": str(csv_path.resolve()),
            "bytes": byte_size,
            "sha256": file_hash,
        },
        "schema": {
            "headers": headers,
            "resolved_semantic_to_source_fields": resolved,
            "unresolved_optional_semantics": missing_optional,
            "status": schema_status,
        },
        "audit": {
            **counts,
            "distinct_track_groups": len(track_stats),
            "distinct_track_ids_only": len({track_id for _, track_id in track_stats}),
            "distinct_recording_ids": len(recording_ids),
            "recording_id_column_available": recording_id_available,
            "track_duration_seconds": {
                "min": min(track_durations) if track_durations else None,
                "median": median(track_durations) if track_durations else None,
                "max": max(track_durations) if track_durations else None,
                "tracks_with_at_least_5_9s": viable_tracks,
            },
            "observed_x_range_m": [x_min, x_max],
            "observed_y_range_m": [y_min, y_max],
            "time_interval_sample": dt_summary,
            "first_row_time_seconds": first_time,
            "last_row_time_seconds": last_time,
            "invalid_row_examples": invalid_examples,
        },
        "lineage_gate": {
            "status": "PASS_RECORDING_ID_PRESENT" if recording_id_available else "BLOCKED_MISSING_RECORDING_ID",
            "reason": (
                "A recording/run identifier is available; confirm it represents the source collection session before splitting."
                if recording_id_available else
                "No recording/run ID column was resolved. Track-ID grouping alone is not sufficient proof of session-disjoint splitting."
            ),
        },
        "scientific_warnings": schema["notes"] + [
            "xloc-kf/yloc-kf are source Kalman-filtered reference-image coordinates; this script does not rotate or translate them.",
            "speed-kf and acceleration-kf are filtered trajectory estimates, not raw IMU channels.",
            "Rows passing numeric/time checks are not automatically eligible training windows; occlusion, interpolation, road-user type, lane continuity and coordinate transforms remain separate gates.",
        ],
        "structural_status": structural_status,
        "scope": "Source-file schema and temporal audit only; not feature parity, split certification, or model evaluation.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(SCHEMAS))
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/tgsim_source_audit.json"))
    parser.add_argument("--skip-duplicate-check", action="store_true")
    args = parser.parse_args()
    report = audit_csv(args.dataset, args.csv, duplicate_check=not args.skip_duplicate_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "dataset": report["dataset"],
        "rows": report["audit"]["rows"],
        "distinct_track_groups": report["audit"]["distinct_track_groups"],
        "distinct_recording_ids": report["audit"]["distinct_recording_ids"],
        "recording_lineage_status": report["lineage_gate"]["status"],
        "structural_status": report["structural_status"],
        "sha256": report["source_file"]["sha256"],
        "output": str(args.output),
    }, indent=2))
    if report["structural_status"] == "FAIL":
        raise SystemExit(2)
    if report["lineage_gate"]["status"] != "PASS_RECORDING_ID_PRESENT":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
