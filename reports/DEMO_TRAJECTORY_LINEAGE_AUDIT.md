# TrajPilot CityFlow — Demonstration Trajectory Lineage Audit

**Audit scope:** checked-in CARLA demonstration logs only. These files are not the I-24/TGSIM model training dataset.

## Findings

| Check | Result |
|---|---:|
| Raw trajectory rows | 150 |
| Raw simulation timestamp increment | 0.1 s |
| Aligned trajectory rows | 150 |
| Aligned frame increments | 1 |
| Aligned elapsed-time increments | 0.1 s |
| Valid-following rows | 60 |
| Valid-following file is exact prefix of aligned file | Yes |
| Heuristic following screen | 114 / 150 rows (76.0%) |
| Step 14 single-run ADE at 3 s | 7.4798 m |
| Step 14 single-run FDE at 1 s | 4.3229 m |
| Step 14 single-run FDE at 2 s | 9.8318 m |
| Step 14 single-run FDE at 3 s | 14.3650 m |

## 1. Valid-following file lineage

The 60-row `trajectories_valid_following.csv` file is an exact prefix of rows 0–59 from the 150-row `trajectories_aligned.csv` file. It is therefore 40% of the aligned episode, but the file contents do not prove why the remaining rows were omitted. It is not evidence of 60 independent tracks or 60 independent samples.

A separate transparent screening rule was applied to all aligned rows:
- positive longitudinal gap;
- absolute lateral offset no greater than 2.5 m;
- wrapped ego-to-leader heading difference no greater than 15 degrees.

That heuristic passes 114 contiguous rows (steps 0–113, 11.4 seconds). This is a diagnostic filter only, not ground-truth leader association.

## 2. Same-road/lane caveat

All 150 aligned rows report the same mapped road/lane pair (12/1), but the recorded lateral offset spans approximately -19.150 m to +0.058 m and heading difference reaches approximately 89.128 degrees. A waypoint lookup that independently projects two vehicles onto a mapped road and lane is not, by itself, a sufficient same-lane leader association test.

## 3. Step 14 forecast

The separate 30-step Step 14 prediction artifact records:
- predicted longitudinal travel: 24.8820 m;
- observed longitudinal travel: 10.5176 m;
- longitudinal travel overprediction ratio: approximately 2.365;
- 3-second endpoint error: 14.3650 m.

The stored displacement errors are consistent with recomputing Euclidean error from the logged predicted and ground-truth longitudinal/lateral displacements (maximum rounding difference below 0.002 m). This verifies arithmetic consistency, not model correctness or generalization.

## 4. Run lineage is unresolved

The raw log starts at simulation time 286.9318 s; the aligned log starts at 852.7081 s. The Step 14 prediction artifact starts at 212.3357 s. The files do not contain a shared run identifier that establishes how they relate. They must remain separate rather than being combined into one trajectory record.

## Research consequences

1. Do not describe the valid-following CSV as a separately collected dataset without its selection procedure and recording lineage.
2. Do not count frames from this single contiguous episode as independent training or test samples.
3. Replace fixed lateral-corridor / approximate centre-gap leader association with an explicitly tested lane-topology and bounding-box-aware method, and preserve rejected/ambiguous samples as such.
4. Keep Step 14 results in a single-run demonstration category, not in the generalization leaderboard.
5. The final manuscript still requires the actual I-24/TGSIM split manifests, target writer, feature construction provenance, frozen checkpoints, and independent held-out predictions.

## Reproduction

Run from repository root:

    python scripts/audit_demo_trajectory_lineage.py
    python -m unittest discover -s tests -v

The command writes machine-readable output to `reports/demo_trajectory_lineage_audit.json` and this report to `reports/DEMO_TRAJECTORY_LINEAGE_AUDIT.md`.

This audit does not certify I-24/TGSIM data quality, live CARLA behavior, feature parity, or paper readiness.
