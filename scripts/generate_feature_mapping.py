"""Generate a conservative, source-aware mapping for canonical TrajPilot features.

The mapping describes candidate sources and validation gates; it does not claim
that the source features are mathematically equivalent to the I-24 or CARLA inputs.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASETS = {
    "TGSIM_I294_L2": {
        "coordinate_fields": "xloc-kf;yloc-kf",
        "time_field": "time",
        "track_id_field": "ID",
        "lane_field": "lane-kf",
        "speed_fields": "speed-kf",
        "acceleration_fields": "acceleration-kf",
        "dimension_fields": "length-smoothed;width-smoothed",
        "road_user_type_fields": "type-most-common",
        "automation_flag_fields": "ACC",
        "coordinate_note": "Per-run reference-image coordinates in metres after Kalman filtering; lane IDs are direction-specific.",
        "special_note": "I-294 overpass trajectories were interpolated across documented blind-spot coordinate intervals; mark or exclude those samples before model-window construction.",
    },
    "TGSIM_I90_I94_MOVING": {
        "coordinate_fields": "xloc-kf;yloc-kf",
        "time_field": "time",
        "track_id_field": "ID",
        "lane_field": "lane-kf",
        "speed_fields": "speed-kf",
        "acceleration_fields": "acceleration-kf",
        "dimension_fields": "length-smoothed;width-smoothed",
        "road_user_type_fields": "type-most-common",
        "automation_flag_fields": "ACC",
        "coordinate_note": "Per-run reference-image coordinates in metres after Kalman filtering; moving-helicopter dataset is northbound only, lane IDs 1-6.",
        "special_note": "Do not combine with the separately published stationary I-90/I-94 product; retain run ID.",
    },
    "TGSIM_FOGGY_BOTTOM": {
        "coordinate_fields": "xloc-kf;yloc-kf",
        "time_field": "time",
        "track_id_field": "ID",
        "lane_field": "lane-kf",
        "speed_fields": "speed-kf-x;speed-kf-y",
        "acceleration_fields": "acceleration-kf-x;acceleration-kf-y",
        "dimension_fields": "length-smoothed;width-smoothed",
        "road_user_type_fields": "type-most-common",
        "automation_flag_fields": "",
        "coordinate_note": "Reference-image grid-road coordinates in metres after Kalman filtering; x/y speed and acceleration remain separate components.",
        "special_note": "Filter road-user types explicitly; lane-kf includes 49 road segments/intersections/crosswalk regions, not highway lane IDs.",
    },
}

FEATURES = [
    {
        "canonical_feature": "vx_causal",
        "source_fields": "xloc-kf;yloc-kf;time",
        "candidate_method": "Causal displacement velocity, then rotate into the explicitly defined ego-course frame",
        "status": "DERIVED_REQUIRES_COORDINATE_VALIDATION",
        "semantic_caveat": "Coordinates are reference-image axes after Kalman filtering, not automatically ego-body longitudinal velocity. Verify image-axis direction, run transform, and y-axis sign.",
        "validation_gate": "Compare against source speed and direction; test rotation invariance, causal cutoff, and units.",
    },
    {
        "canonical_feature": "vy_causal",
        "source_fields": "xloc-kf;yloc-kf;time",
        "candidate_method": "Causal displacement velocity, then rotate into the explicitly defined ego-course frame",
        "status": "DERIVED_REQUIRES_COORDINATE_VALIDATION",
        "semantic_caveat": "Reference-image lateral component requires verified axis orientation and a documented ego-course transform.",
        "validation_gate": "Test sign convention on straight and turning trajectories; compare vector magnitude with source speed.",
    },
    {
        "canonical_feature": "speed_derived",
        "source_fields": "speed-kf (highway);speed-kf-x;speed-kf-y (Foggy Bottom)",
        "candidate_method": "Use published scalar speed as an independent source measurement; for Foggy Bottom compute hypot(speed-kf-x, speed-kf-y) only after verifying axis units",
        "status": "SOURCE_KF_FILTERED_CROSSCHECK",
        "semantic_caveat": "Published speed fields are Kalman-filtered. They are not interchangeable by assumption with a finite-difference speed derived from positions.",
        "validation_gate": "Quantify bias, lag, and missingness versus position-derived speed before selecting one canonical definition.",
    },
    {
        "canonical_feature": "acceleration_native",
        "source_fields": "acceleration-kf (highway);acceleration-kf-x;acceleration-kf-y (Foggy Bottom)",
        "candidate_method": "Treat as source Kalman-filtered acceleration; project the Foggy Bottom vector only after frame validation",
        "status": "SOURCE_KF_FILTERED_NOT_NATIVE_IMU",
        "semantic_caveat": "The field name acceleration_native in the frozen model must not be interpreted as raw IMU acceleration for TGSIM. The published field is Kalman-filtered trajectory acceleration.",
        "validation_gate": "Confirm source units, sign, filtering, smoothing, and transformation; compare with derivative of source speed.",
    },
    {
        "canonical_feature": "acceleration_causal_0p5s",
        "source_fields": "speed-kf (highway);speed-kf-x;speed-kf-y (Foggy Bottom);time",
        "candidate_method": "Backward 0.5-second finite difference of the chosen canonical speed series",
        "status": "DERIVED_REQUIRES_FILTER_PARITY",
        "semantic_caveat": "Do not mix the filtered source speed with a differently filtered position-derived series without documenting the change.",
        "validation_gate": "Verify 10 Hz continuity, startup policy, missing frames, and matched speed definition.",
    },
    {
        "canonical_feature": "turn_rate_valid",
        "source_fields": "xloc-kf;yloc-kf;time",
        "candidate_method": "Backward difference of the unwrapped course angle from causal displacement velocity",
        "status": "DERIVED_REQUIRES_COORDINATE_VALIDATION",
        "semantic_caveat": "No direct yaw-rate field is defined in the published I-294/I-90/I-94 dictionary. Course angle is not vehicle body heading.",
        "validation_gate": "Verify image y-axis sign, angle unwrap, low-speed mask, and temporal gaps.",
    },
    {
        "canonical_feature": "curvature_valid",
        "source_fields": "xloc-kf;yloc-kf;time",
        "candidate_method": "Course-rate divided by the verified planar speed, with explicit low-speed mask",
        "status": "DERIVED_REQUIRES_COORDINATE_VALIDATION",
        "semantic_caveat": "Depends on the turn-rate and speed definitions; do not reuse a CARLA threshold without sensitivity analysis.",
        "validation_gate": "Check units 1/m, low-speed policy, straight-line and constant-radius synthetic tests.",
    },
    {
        "canonical_feature": "jerk_causal",
        "source_fields": "acceleration-kf (highway);acceleration-kf-x;acceleration-kf-y (Foggy Bottom);time",
        "candidate_method": "Backward derivative of the chosen acceleration series",
        "status": "DERIVED_REQUIRES_FILTER_PARITY",
        "semantic_caveat": "Differentiating already filtered acceleration may amplify residual noise; component-to-longitudinal projection is needed for Foggy Bottom.",
        "validation_gate": "Check timestamp gaps, filtering effects, sign, and outlier sensitivity.",
    },
    {
        "canonical_feature": "SpaceGap",
        "source_fields": "ID;time;xloc-kf;yloc-kf;lane-kf;length-smoothed;width-smoothed",
        "candidate_method": "Derive a leader candidate from synchronized tracks, lane/road geometry, heading, and vehicle extents",
        "status": "DERIVED_REQUIRES_LEADER_ASSOCIATION",
        "semantic_caveat": "No direct space-gap column appears in the published base dictionary. Center distance is not bumper gap; lane IDs alone do not define a leader.",
        "validation_gate": "Validate leader pairing, run boundaries, lane topology, vehicle dimensions, occlusion/interpolation flags, and bumper-gap geometry.",
    },
    {
        "canonical_feature": "RelativeVelocity",
        "source_fields": "ID;time;xloc-kf;yloc-kf;speed-kf or component speeds;lane-kf",
        "candidate_method": "Derive ego-minus-leader longitudinal speed after a validated leader association",
        "status": "DERIVED_REQUIRES_LEADER_ASSOCIATION",
        "semantic_caveat": "The sign convention must be fixed explicitly; per-track scalar speed alone does not establish relative longitudinal velocity on a curved or changing heading.",
        "validation_gate": "Use a hand-calculated paired-track fixture and validate direction changes.",
    },
    {
        "canonical_feature": "LateralDistance",
        "source_fields": "xloc-kf;yloc-kf;lane-kf;per-run lane-centreline geometry",
        "candidate_method": "Project vehicle/leader positions onto a validated lane or road centreline",
        "status": "DERIVED_REQUIRES_ROAD_GEOMETRY",
        "semantic_caveat": "A discrete lane label is not a continuous lateral offset. Geometry is run- and product-specific.",
        "validation_gate": "Test centreline projection across all supported runs, directions, merges, and junctions.",
    },
    {
        "canonical_feature": "time_headway",
        "source_fields": "derived SpaceGap;verified ego longitudinal speed",
        "candidate_method": "SpaceGap divided by positive ego longitudinal speed, with a low-speed invalidity rule",
        "status": "DERIVED_REQUIRES_LEADER_ASSOCIATION",
        "semantic_caveat": "Cannot be created from an unrelated nearest vehicle or Euclidean distance without a validated leader and gap definition.",
        "validation_gate": "Test units, zero/near-zero speed, sign, and leader identity stability.",
    },
    {
        "canonical_feature": "L_Approach",
        "source_fields": "synchronized track table;lane-centreline/road-segment geometry",
        "candidate_method": "Not specified; requires original feature definition before implementation",
        "status": "UNRESOLVED_SOURCE_SEMANTICS",
        "semantic_caveat": "No corresponding direct column or unambiguous physical definition is established by the published data dictionary.",
        "validation_gate": "Recover exact source formula and temporal meaning; otherwise keep masked.",
    },
    {
        "canonical_feature": "R_Approach",
        "source_fields": "synchronized track table;lane-centreline/road-segment geometry",
        "candidate_method": "Not specified; requires original feature definition before implementation",
        "status": "UNRESOLVED_SOURCE_SEMANTICS",
        "semantic_caveat": "No corresponding direct column or unambiguous physical definition is established by the published data dictionary.",
        "validation_gate": "Recover exact source formula and temporal meaning; otherwise keep masked.",
    },
    {
        "canonical_feature": "L_Adjacent",
        "source_fields": "synchronized track table;lane-centreline/road-segment geometry",
        "candidate_method": "Not specified; requires original feature definition before implementation",
        "status": "UNRESOLVED_SOURCE_SEMANTICS",
        "semantic_caveat": "Lane adjacency may be represented by lane topology, but the frozen feature's exact value and coding are not established.",
        "validation_gate": "Recover exact source coding and treatment of merge/diverge/junction segments; otherwise keep masked.",
    },
    {
        "canonical_feature": "R_Adjacent",
        "source_fields": "synchronized track table;lane-centreline/road-segment geometry",
        "candidate_method": "Not specified; requires original feature definition before implementation",
        "status": "UNRESOLVED_SOURCE_SEMANTICS",
        "semantic_caveat": "Lane adjacency may be represented by lane topology, but the frozen feature's exact value and coding are not established.",
        "validation_gate": "Recover exact source coding and treatment of merge/diverge/junction segments; otherwise keep masked.",
    },
]


def build_mapping_rows():
    rows = []
    for dataset_name, dataset in DATASETS.items():
        for feature in FEATURES:
            rows.append({
                "target_dataset": dataset_name,
                "canonical_feature": feature["canonical_feature"],
                "published_source_fields": feature["source_fields"],
                "candidate_method": feature["candidate_method"],
                "equivalence_status": feature["status"],
                "semantic_caveat": feature["semantic_caveat"],
                "validation_gate": feature["validation_gate"],
                "track_id_field": dataset["track_id_field"],
                "time_field": dataset["time_field"],
                "coordinate_fields": dataset["coordinate_fields"],
                "lane_field": dataset["lane_field"],
                "speed_fields": dataset["speed_fields"],
                "acceleration_fields": dataset["acceleration_fields"],
                "dimension_fields": dataset["dimension_fields"],
                "road_user_type_fields": dataset["road_user_type_fields"],
                "automation_flag_fields": dataset["automation_flag_fields"],
                "coordinate_note": dataset["coordinate_note"],
                "dataset_specific_note": dataset["special_note"],
            })
    return rows


def write_mapping(output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_mapping_rows()
    fieldnames = list(rows[0])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/tgsim_feature_mapping.csv"),
    )
    args = parser.parse_args()
    rows = write_mapping(args.output)
    print(f"Saved {len(rows)} source-aware feature mappings to {args.output}")
    print("No exact I-24/TGSIM/CARLA feature parity is claimed by this file.")


if __name__ == "__main__":
    main()
