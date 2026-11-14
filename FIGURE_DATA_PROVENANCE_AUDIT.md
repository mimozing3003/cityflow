# Supplementary Figure Provenance Audit

## Finding

The previous `scripts/generate_supplementary_figures.py` mixed three unsupported sources of numbers into figures presented as empirical:

1. Fixed dataset mean-speed values and trajectory counts written directly into the source.
2. Feature histograms generated from seeded `numpy.random.normal` distributions rather than measured samples.
3. A manually entered feature-correlation matrix rather than a correlation computed from observations.

The old pipeline schematic also displayed unverified numerical/performance claims (e.g., physical violation rate, feature-equivalence count and inference FPS).

These figures are not empirical evidence and must not be submitted as measured research results.

## Correction

The replacement script has two modes:

- `--diagram-only`: generates a method schematic with no performance claims, clearly marked as a schematic rather than a validation certificate.
- Empirical mode: requires both a dataset summary CSV and a canonical feature-sample CSV. It verifies:
  - dataset summary has unique dataset names and `audit_status=PASS`;
  - each source file is linked by SHA-256;
  - positive audited source-row and track-group counts are present;
  - each feature row has a recording ID, a source hash linked to an audited source, and finite values for all eight canonical features;
  - the feature sample contains at least two recording IDs and is from exactly one dataset.

The feature-distribution and correlation figures are generated from those measured rows. Cross-domain feature pooling is deliberately prohibited until equivalence of the feature definitions is demonstrated.

## Input contract

Dataset summary columns:
- `dataset`
- `source_file`
- `source_sha256`
- `audit_status` (must be PASS)
- `source_row_count`
- `distinct_track_groups`

Feature sample columns:
- `dataset`
- `recording_id`
- `source_file_sha256`
- `vx_causal`, `vy_causal`, `speed_derived`, `acceleration_native`,
  `acceleration_causal_0p5s`, `turn_rate_valid`, `curvature_valid`, `jerk_causal`

Feature samples should be exported by the validated, canonical feature builder, not manually assembled or synthetically generated. The feature file must link to the exact source CSV SHA-256 recorded in the dataset summary.

## Reproduction

Schematic only:

    python scripts/generate_supplementary_figures.py --diagram-only

Empirical figures after real source audits and canonical feature export:

    python scripts/generate_supplementary_figures.py --dataset-summary reports/dataset_summary.csv --feature-samples reports/feature_samples.csv

The generator writes `figure_provenance_manifest.json` beside the figures. If data are missing, malformed, unaudited or mixed across domains, empirical generation fails rather than silently substituting synthetic values.

## Paper restriction

Do not use figures from the previous generator. Historical figures must be deleted or replaced by figures regenerated from this provenance-gated script. The new empirical mode has not yet been run because the source data and feature-sample artifact are not present in the tracked repository.
