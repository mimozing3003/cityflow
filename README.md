# CityFlow: TrajPilot Autonomous Trajectory Planning

CityFlow TrajPilot is an end-to-end framework for evaluating vehicle trajectory prediction models and closed-loop autonomous driving policies within the CARLA simulator.

## Highlights
- **CARLA Integration**: Automated scenario setup, vehicle spawning, and camera telemetry capture in CARLA 0.9.16.
- **Trajectory Modeling**: Evaluates GRU and motion prediction baselines against realistic traffic flows.
- **Diagnostics & Benchmarking**: Tools for real-time latency auditing, feature alignment, and validation checks.

## Project Structure
- `carla_demo/`: Modular scripts for environment verification, model loading, and CARLA scene execution.
- `curla demostrate/`: Live demonstration harnesses and trajectory alignment audits.
- `scripts/`: Utility scripts for environment setup and data handling.
- `configs/`: Experiment and simulation parameter definitions.

## Getting Started

### 1. Environment Setup
Install dependencies with Conda or Python:
```bash
conda env create -f environment.yml
conda activate cityflow
```

### 2. Launch CARLA Simulation
Ensure CARLA 0.9.16 is running, then execute the demo script:
```bash
.\RUN_TRAJPILOT_CARLA.bat
```

## License
This project is made available under the MIT License. See [LICENSE](LICENSE) for details.


## Research Reproducibility & Paper Readiness

The current final-paper evidence gate is documented in [PAPER_READINESS_AUDIT.md](PAPER_READINESS_AUDIT.md). The machine-readable release metadata template is [paper_artifact_manifest.template.json](paper_artifact_manifest.template.json).

The research report contains historical benchmark claims that must be regenerated from versioned data, model, prediction, and evaluation artifacts before final submission. A polished manuscript is not considered paper-ready until all evidence gates pass.


Dataset provenance and non-equivalence notes for I-24 MOTION and the individual TGSIM products are in [DATASET_PROVENANCE_NOTES.md](DATASET_PROVENANCE_NOTES.md). Exact local source-file hashes and extraction counts are still required.


TGSIM's published source-field semantics and the conservative canonical feature mapping are documented in [TGSIM_FEATURE_SEMANTICS_AUDIT.md](TGSIM_FEATURE_SEMANTICS_AUDIT.md). Regenerate the local mapping with `python scripts/generate_feature_mapping.py`; the output is intentionally written under the ignored data/manifests directory and is not itself evidence of source-file validation.


### Auditing a locally downloaded TGSIM source file

The source auditor is intentionally local-file based; it does not download restricted or large datasets automatically. Example:

```bash
python scripts/audit_tgsim_source_data.py --dataset TGSIM_I294_L2 --csv data/raw/TGSIM_I294_L2/I294_L2_final.csv --output reports/i294_source_audit.json
python scripts/audit_tgsim_source_data.py --dataset TGSIM_I90_I94_MOVING --csv data/raw/TGSIM_I90_I94_MOVING/I90_94_moving_final.csv --output reports/i90_moving_source_audit.json
python scripts/audit_tgsim_source_data.py --dataset TGSIM_FOGGY_BOTTOM --csv "data/raw/TGSIM_FOGGY_BOTTOM/TGSIM-Foggy-Bottom-Data.csv" --output reports/foggy_bottom_source_audit.json
```

It accepts only explicit published hyphenated or documented normalized underscore aliases, rejects ambiguous headers, hashes the exact input file, and blocks recording/session-level lineage when no recording/run identifier is present. It does not prove feature parity or transform image/reference coordinates.


### Exact-cadence trajectory windows

The TGSIM training/benchmark extractors now split each track at duplicate, out-of-order, non-finite, or non-10-Hz timestamp intervals (nominal 0.1 s, tolerance ±0.025 s). Windows are generated only within continuous segments; a history/target pair cannot cross a time gap. This deliberately rejects the prior permissive rule that allowed gaps up to 0.25 s while still differentiating at 0.1 s.


The statistical uncertainty protocol and the limitation of trajectory- versus session-level inference are documented in [CLUSTERED_STATISTICAL_PROTOCOL.md](CLUSTERED_STATISTICAL_PROTOCOL.md). Historical bootstrap confidence intervals in the research report are provisional until regenerated from the corrected script on the exact source datasets.
