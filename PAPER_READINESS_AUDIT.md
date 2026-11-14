# TrajPilot CityFlow — Final Paper Readiness Gate

**Current verdict: NOT PAPER-READY**

This is a repository-evidence audit, not a statement that files cannot exist on the author's local workstation or external storage. It asks whether a reviewer can start from a clean clone and independently verify the dataset, feature pipeline, trained models, held-out predictions, and numerical claims. At the current tracked-repository snapshot, the answer is no.

## 1. Evidence audit

| Paper dependency | Repository observation | Status |
|---|---|---|
| I-24 MOTION raw records and source metadata | No tracked raw I-24 trajectory table or source data dictionary is present | BLOCKED |
| TGSIM I-294 / I-90-I-94 / Foggy Bottom source records | No tracked source trajectory tables or data dictionaries are present | BLOCKED |
| Recording/session and vehicle IDs | No canonical raw-data manifest or split manifest is tracked | BLOCKED |
| Preprocessing and target writer | Target conventions are described, but a complete canonical source-to-window pipeline is not released | BLOCKED |
| Frozen I-24 GRU checkpoint | No tracked .pt/.pth checkpoint is present | BLOCKED |
| Frozen training normalization statistics | No tracked train-normalization CSV is present | BLOCKED |
| Window tensors and target arrays | No tracked .npy/.npz/Parquet arrays for reproducible train/validation/test windows are present | BLOCKED |
| CARLA scenario benchmark source table | The report cites outputs/tables/final_audit/CARLA_RESULTS.csv, but this file is absent from the tracked tree | BLOCKED |
| Per-episode held-out predictions | No complete predictions/targets/recording-ID table is present in the tracked tree | BLOCKED |
| Statistical analysis inputs | No cluster-level metric table or bootstrap configuration/resamples are tracked | BLOCKED |
| Ridge/XGBoost serialization | Ridge, scaler and XGBoost artifacts exist in the nested TRAJPILOT CITYFLOW/models directory, but full training-data lineage and benchmark generation are not bundled | PARTIAL |
| CARLA runtime demonstration | Source and single-run demo outputs exist; they are not an independent scenario benchmark | PARTIAL |
| Manuscript and bibliography | A research report exists, but its numerical claims are not all reproducible from tracked evidence | PARTIAL |

## 2. Why a clean clone cannot reproduce the current report

The repository .gitignore excludes data/, outputs/, results/, *.npy, *.npz, *.pkl and *.pt. This may be appropriate for large or licensed datasets, but it requires a controlled artifact release mechanism. At present, the report references outputs/tables/final_audit tables that are not tracked, and the frozen GRU report references a checkpoint and normalization file under data/ that are likewise not tracked.

The report's labels such as "CERTIFIED", "Final Research Freeze", and "Audit-Certified" should therefore be treated as historical project assertions, not independently reproducible certification from the current Git snapshot.

## 3. Required release package

Before the paper is declared final, create a versioned research release containing:

1. A data manifest with dataset name/version, provider citation, license/usage terms, retrieval instructions, file hashes, coordinate system, units, sampling frequency, recording/session IDs and expected counts.
2. A reproducible preprocessing pipeline with exact source column mapping, filtering, coordinate transformations, timestamp handling, duplicate handling and feature formulas.
3. A target builder that writes history/future windows, with tests for the 3 s history / 3 s future convention and ego-aligned target origin.
4. A recording/session-disjoint split manifest. No overlapping windows from the same recording may leak across train/validation/test.
5. Frozen checkpoints, train-only normalization statistics, environment lockfile, seeds, hyperparameters, code commit SHA and checkpoint SHA-256.
6. Per-episode predictions for every reported model and domain, including episode/recording ID, target timestamps, predictions, ground truth and validity flags.
7. One evaluation script that regenerates ADE/FDE at 1 s, 2 s and 3 s, failure rates, physical-constraint violations and latency from prediction tables.
8. Cluster-bootstrap 95% confidence intervals resampled by independent recording/session, paired model comparisons and a multiple-comparison correction plan.
9. CARLA scenario manifests and raw episode logs for each claimed scenario result, with route, map, weather, seed handling, ego identity, traffic context and simulator version.
10. Manuscript tables and figures generated directly from the released evaluation outputs rather than manually copied numbers.

Licensed or large datasets do not need to be committed to Git. Their manifests, hashes, data dictionaries, retrieval instructions and deterministic preparation scripts do need to be available.

## 4. Claim restrictions until the gates pass

Do not present the following as established findings until the evidence package is independently reproduced:

- I-24 / TGSIM dataset certification or exact trajectory counts.
- Session-disjoint or vehicle-disjoint split purity.
- KCL reducing physical violations to exactly 0.00%.
- The reported highway leaderboard and bootstrap confidence intervals.
- CARLA scenario winner counts and metrics attributed to CARLA_RESULTS.csv.
- Exact CARLA-to-I-24 feature equivalence.
- General cross-domain generalization based on a single CARLA demo.

The Step 14 CARLA result (3 s FDE 14.365 m) is a single-run engineering demonstration and must remain separate from held-out benchmark results.

## 5. Explicit final-paper acceptance gates

Mark a gate PASS only when the artifact is present and an independent command/audit reproduces it.

- [ ] Dataset provenance and retrieval manifest verified.
- [ ] Raw data schema and coordinate/unit conventions verified.
- [ ] Canonical causal feature builder frozen and tested against source definitions.
- [ ] Target writer and ego-aligned target frame independently verified.
- [ ] Recording/session-disjoint splits regenerated and leakage-audited.
- [ ] Every reported checkpoint and normalizer hash recorded.
- [ ] Baselines and ML predictions regenerated on untouched held-out episodes.
- [ ] ADE/FDE at 1/2/3 seconds regenerated from per-episode predictions.
- [ ] Cluster-bootstrap confidence intervals and paired comparisons regenerated.
- [ ] Physical violations and latency regenerated with defined measurement boundaries.
- [ ] CARLA benchmark regenerated from scenario-level raw logs and manifests.
- [ ] Manuscript tables/figures regenerated from evaluation outputs.
- [ ] References, limitations, reproducibility statement and data availability statement checked.
- [ ] Clean-clone reproduction run completed and its log archived.

## 6. When can the paper be called final?

Not when the prose is polished. It is final only after every acceptance gate above is PASS, the results are reproducible from released artifacts, and the manuscript contains no unsupported claims.

No calendar completion estimate is provided: the blockers depend on recovering and verifying research artifacts, not on writing speed.

## 7. Newly identified statistical validity blocker

The previous highway ablation pipeline resampled overlapping windows individually for its nominal "trajectory-level" bootstrap and ran paired Wilcoxon/t-tests on window-level errors. The windows are correlated within each vehicle track. Those existing confidence intervals and p-values are not acceptable as final-paper inference. The code has been corrected to use trajectory-cluster bootstrap and paired tests on per-trajectory mean errors, with Holm adjustment across six FDE comparisons; see [CLUSTERED_STATISTICAL_PROTOCOL.md](CLUSTERED_STATISTICAL_PROTOCOL.md). The corrected code has not yet been run against the actual TGSIM source files, and it still does not establish session-level independence because the benchmark split is vehicle/trajectory-disjoint rather than recording/session-disjoint.
