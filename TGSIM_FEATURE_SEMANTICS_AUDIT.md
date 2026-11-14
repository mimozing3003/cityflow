# TGSIM Field Semantics Audit — TrajPilot CityFlow

## Scope

This document reconciles the repository's TGSIM mapping code with the FHWA TGSIM 2024 report (FHWA-JPO-24-133), especially Table 5 (I-294 and I-90/I-94), Table 7 (Foggy Bottom), and the I-294 lane/interpolation notes.

Source report:
https://rosap.ntl.bts.gov/view/dot/74647/dot_74647_DS1.pdf

## 1. Published highway fields are already filtered trajectory estimates

For I-294 and I-90/I-94, the FHWA report defines:

| Published field | Meaning | Consequence |
|---|---|---|
| ID | Unique detected vehicle ID | Must be grouped with run/product context |
| time | Seconds relative to start of run | Use this, not frame number, as source time |
| xloc-kf / yloc-kf | Reference-image coordinates after Kalman filtering, in metres | Not raw pixel detections and not a global geodetic frame |
| lane-kf | Lane ID on the reference image after Kalman filtering | Run/direction-specific categorical lane reference |
| speed-kf | Measured speed after Kalman filtering, m/s | Scalar filtered measurement, not necessarily equal to finite-difference speed |
| acceleration-kf | Measured acceleration after Kalman filtering, m/s² | Filtered trajectory-derived quantity, not native IMU acceleration |
| length-smoothed / width-smoothed | Smoothed bounding-box dimensions, proxies | Do not treat as exact physical vehicle dimensions |
| type-most-common | Most common detected vehicle type | Must preserve/filter road-user/vehicle classes |
| ACC | Yes/no ACC-vehicle flag | Do not conflate with generic automated vehicle indicator |

**Critical correction:** the old feature mapping called TGSIM acceleration “native” and marked it “EXACT”. That is not supported. The FHWA field is explicitly Kalman-filtered. The frozen feature name may remain for model-input compatibility, but manuscripts and mapping artifacts must identify the source as Kalman-filtered trajectory acceleration, not raw IMU acceleration.

## 2. I-294 overpass interpolation

FHWA states that I-294 includes three overpasses that occlude the helicopter camera; trajectories were interpolated through the loss-of-view intervals. The report provides run-index-specific xloc-kf ranges:

| Run indices | Overpass 1 | Overpass 2 | Overpass 3 |
|---|---:|---:|---:|
| 1 | 2130–2170 m | 3330–3370 m | 3730–3770 m |
| 3, 7, 9, 11, 19, 21 | 1530–1570 m | 2730–2770 m | 3130–3170 m |
| 8, 18, 20 | 1760–1800 m | 2160–2200 m | 3360–3400 m |

The exact local run ID and coordinate mapping must be reconciled before these ranges can be applied. Do not treat interpolated samples as ordinary camera observations. At minimum, create an interpolation mask and report metrics both with and without affected windows, or document a defensible exclusion policy.

## 3. Lane coding is product- and direction-specific

- I-294: southbound lanes are -2, -3, -4, -5; northbound lanes are 2, 3, 4, 5. Lane -5 is leftmost and lane 5 is rightmost from the reference-image perspective.
- I-90/I-94 moving-helicopter product: all vehicles are northbound, lanes 1–6, lane 1 innermost; express lanes excluded.
- I-90/I-94 stationary-helicopter diverging-area product: odd-index runs use lane IDs 1–6; even-index runs use 10–15. This is a separate data product and must not be mixed with the moving-helicopter file.

A canonical lane feature cannot be built from lane-kf without product, run, and direction metadata.

## 4. Foggy Bottom is not a highway kinematic table

FHWA describes Foggy Bottom as a grid road network. Its published kinematics include:

- xloc-kf and yloc-kf: filtered reference-image coordinates, metres;
- lane-kf: region IDs 1–49 spanning road segments, intersections and crosswalk regions;
- speed-kf-x and speed-kf-y: separate filtered x/y velocity components;
- acceleration-kf-x and acceleration-kf-y: separate filtered x/y acceleration components;
- type-most-common: 0 persons, 1 bicycles, 2 scooters, 3 passenger cars, 4 automated vehicles, 5 motorcycles, 6 buses, 7 trucks.

The x/y components must be transformed through a local road direction before being interpreted as longitudinal/lateral velocity or acceleration. A single scalar speed or acceleration cannot be read directly from only the x component. Vehicle-only analysis must explicitly filter road-user type values and document its treatment of automated vehicles. The source reports that only a small portion of Foggy Bottom data covers SAE Level 3 ADS-equipped vehicles.

## 5. Corrections made to the repository mapping generator

The revised scripts/generate_feature_mapping.py now:
- records the published field names by dataset product;
- marks the highway speed/acceleration fields as Kalman-filtered source measurements;
- explicitly refuses to call TGSIM acceleration native IMU acceleration or exact feature parity;
- marks ego-aligned components, acceleration windows, turn rate, curvature and jerk as derived features requiring coordinate/filter validation;
- treats interaction features as derivable only after synchronized-track and leader-association logic;
- leaves the four lane-context semantics unresolved until the original formulas/codings are recovered;
- records dataset-specific run, lane and geometry caveats.

The generated mapping is written to data/manifests/tgsim_feature_mapping.csv. The data/ directory is intentionally ignored by Git; the generator source and tests are versioned.

## 6. Research decisions required before feature parity

1. Retrieve the exact local source file headers and compare them with the published hyphenated names. Any snake_case aliases must be written into an explicit raw-to-canonical column mapping.
2. Confirm which I-90/I-94 product is used: moving or stationary.
3. Identify run index for every row, including if it is stored in an external per-file manifest rather than the CSV.
4. Identify and mask/exclude the I-294 overpass interpolation ranges.
5. Verify that the Kalman-filtered speed/acceleration definitions match the model's training definitions before using them.
6. Derive course/yaw from positions only after coordinate axes, y-axis orientation, road direction and timestamp discontinuities are handled.
7. Reconstruct leader/follower interactions only after time synchronization, lane topology, and vehicle dimensions are validated.
8. Keep Foggy Bottom separate from highway domains until the x/y vector fields and grid-road geometry are transformed to a defensible common frame.

Passing the tests means the mapping metadata remains conservative. It does not prove feature equivalence or model validity.
