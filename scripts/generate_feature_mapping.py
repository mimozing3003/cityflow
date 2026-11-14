import pandas as pd

features_list = [
    ("vx_causal", "Ego Kinematics", "Causal backward velocity in x", "m/s", "DERIVED", "(xloc_kf[t] - xloc_kf[t-1])/0.1s (or speed_kf_x)"),
    ("vy_causal", "Ego Kinematics", "Causal backward velocity in y", "m/s", "DERIVED", "(yloc_kf[t] - yloc_kf[t-1])/0.1s (or speed_kf_y)"),
    ("speed_derived", "Ego Kinematics", "Total planar speed magnitude", "m/s", "EXACT", "speed_kf or sqrt(vx^2 + vy^2)"),
    ("acceleration_native", "Ego Kinematics", "Native Kalman-filtered acceleration", "m/s^2", "EXACT", "acceleration_kf (or acceleration_kf_x)"),
    ("acceleration_causal_0p5s", "Ego Kinematics", "0.5-second smoothed causal acceleration", "m/s^2", "DERIVED", "(speed[t] - speed[t-5])/0.5s"),
    ("turn_rate_valid", "Ego Geometry", "Heading rate of change dpsi/dt", "rad/s", "DERIVED", "arctan2(vy, vx) diff / dt"),
    ("curvature_valid", "Ego Geometry", "Spatial path curvature kappa = (dpsi/dt)/v", "1/m", "DERIVED", "turn_rate / max(speed, 1.0)"),
    ("jerk_causal", "Ego Kinematics", "Causal derivative of acceleration", "m/s^3", "DERIVED", "(accel[t] - accel[t-1])/0.1s"),
    ("SpaceGap", "Lead Vehicle Context", "Distance to preceding vehicle", "m", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables"),
    ("RelativeVelocity", "Lead Vehicle Context", "Relative approach velocity to lead vehicle", "m/s", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables"),
    ("LateralDistance", "Boundary Context", "Continuous lateral offset to lane marking", "m", "UNAVAILABLE", "Only discrete integer lane_kf provided"),
    ("time_headway", "Lead Vehicle Context", "Time gap to lead vehicle SpaceGap/speed", "s", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables"),
    ("L_Approach", "Neighboring Interaction", "Left approaching vehicle closing rate", "m/s", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables"),
    ("R_Approach", "Neighboring Interaction", "Right approaching vehicle closing rate", "m/s", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables"),
    ("L_Adjacent", "Neighboring Interaction", "Left adjacent vehicle distance", "m", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables"),
    ("R_Adjacent", "Neighboring Interaction", "Right adjacent vehicle distance", "m", "UNAVAILABLE", "Not recorded in raw TGSIM vehicle tables")
]

records = []
for dname in ["TGSIM_I294_L2", "TGSIM_I90_I94_MOVING", "TGSIM_FOGGY_BOTTOM"]:
    for fname, category, desc, units, status, formula in features_list:
        records.append({
            "target_dataset": dname,
            "canonical_feature": fname,
            "category": category,
            "physical_units": units,
            "equivalence_status": status,
            "mapping_rule": formula,
            "scientific_rationale": "Causal ego-motion derivable from coordinates" if status in ["EXACT", "DERIVED"] else "No surrounding vehicle interaction graph in raw CSV; fabrication rejected"
        })

df_map = pd.DataFrame(records)
df_map.to_csv("data/manifests/tgsim_feature_mapping.csv", index=False)
print("Saved data/manifests/tgsim_feature_mapping.csv")
print(df_map[df_map['target_dataset'] == 'TGSIM_I294_L2'][['canonical_feature', 'category', 'equivalence_status', 'mapping_rule']].to_string())
