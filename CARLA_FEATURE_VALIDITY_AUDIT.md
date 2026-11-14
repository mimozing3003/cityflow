# CARLA Feature Validity Audit — TrajPilot CityFlow

## Scope

This audit checks the repository's current CARLA feature pipeline and records code-level implementation risks. It is not a CARLA live-run report, an I-24 parity certificate, or a model-accuracy benchmark.

## Change made on this branch

In `carla_demo/05_feature_engine.py`, the four unresolved lane-context values are now emitted as `NaN` instead of `0.0`:

- `L_Approach`
- `R_Approach`
- `L_Adjacent`
- `R_Adjacent`

The feature metadata already marked these features unavailable and unresolved. The previous zero values contradicted that metadata and were treated as valid observations by the documented 16-value + 16-missingness input representation. The normalizer in `carla_demo/02_load_normalization.py` explicitly marks non-finite feature values as missing and sets their normalized values to zero. Using `NaN` therefore keeps the feature value neutral while setting its missingness channel to one.

## Findings requiring attention

1. **Multiple feature implementations are not interchangeable.** The primary demo engine, `carla_eval/carla_adapter.py`, `carla_eval/carla_runtime_adapter.py`, and `curla demostrate/trajpilot_city_roaming.py` compute overlapping features using different sources and formulas. For example, acceleration is a forward projection of the actor acceleration vector in the primary demo engine, but a 3-D acceleration magnitude in the two `carla_eval` adapters and the city-roaming script.

2. **The feature label "native" does not prove source-domain parity.** The code uses CARLA actor state and yaw-rate telemetry. I-24 source features need their exact original construction definitions, coordinate conventions, filtering, sign conventions, and units verified before calling these equal.

3. **The 0.5-second acceleration startup policy differs by implementation.** The primary demo engine computes a shorter backward difference during warm-up. The older evaluation adapters mark the first five slots as missing. That changes the early temporal input distribution and missingness masks.

4. **Interaction variables are approximate.** The current leader detector uses a fixed lateral corridor and subtracts a constant 4.5 m from longitudinal centre separation as an approximate bumper gap. This is not a source-verified I-24 bumper-gap formula.

5. **Single-run demonstration performance is weak.** The checked-in Step 14 summary records 3-second ADE 7.4798 m and FDE 14.3650 m. It reports predicted longitudinal travel 24.8820 m versus observed 10.5176 m. That is evidence of one unsuccessful/poor forecast, not evidence of generalization. The status string `PASS_SINGLE_RUN_DEMO` means the demonstration completed; it must not be read as an accuracy pass.

6. **Coordinate-frame contract remains unresolved.** The live inference projector rotates predicted forward/lateral offsets using CARLA body yaw. The exact relationship between the I-24 target frame and the CARLA actor frame must be confirmed from the target writer and original dataset. Do not change projection based on a guess.

## Automated checks added

`tests/test_carla_feature_validity.py` exercises:
- canonical 16-feature ordering and output shape;
- unresolved lane features are explicitly unavailable and represented by NaN;
- absent leaders produce missing interaction features;
- leader-relative velocity sign and headway arithmetic;
- unresolved lane metadata remains unavailable.

The GitHub Actions workflow runs these synthetic contract tests on pull requests and branch pushes. It does not require CARLA or claim to test a live simulator session.

## Acceptance boundary

A passing workflow means the tested engineering contracts hold for the fake-actor fixtures. Before making cross-domain claims, the project still needs:
- exact I-24 source feature formula and coordinate-frame provenance;
- a single canonical CARLA feature implementation used by every inference/evaluation path;
- live CARLA collection with episode metadata and saved raw trajectories;
- independent per-episode feature parity and target-frame tests;
- recording/session-level evaluation on untouched held-out episodes.
