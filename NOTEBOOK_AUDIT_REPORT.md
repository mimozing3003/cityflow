# TrajPilot CityFlow — Notebook Audit & Correction Report

**Original notebook:** TrajPilot_Research_Master_(1).ipynb  
**Original cell count:** 293  
**Clean master output:** TrajPilot_CityFlow_Master_Research.ipynb  
**Clean master cell count:** 51

---

## 1. Backup

The original notebook was backed up as `TrajPilot_Research_Master_BACKUP.ipynb`
before any changes were made.

---

## 2. Problems Identified

| # | Problem | Severity | Resolution |
|---|---------|----------|------------|
| P01 | Hardcoded Windows path `C:\Users\SOMUJIT\...` in 60+ cells | HIGH | Replaced with `os.environ.get("TRAJPILOT_ROOT", Path.cwd())` in SEC-01 |
| P02 | Duplicate section header "Cell 52A" (appears twice) | MEDIUM | Removed duplicate; single SEC-08 in master |
| P03 | Forensic/debug cells 214–245 mixed into main pipeline | HIGH | Moved to logical archive status; not in master |
| P04 | No single authoritative configuration cell | HIGH | SEC-01 is the single configuration source |
| P05 | GRU class defined in both cell 202 and cell 204 | MEDIUM | One canonical `TrajPilotGRU` class in SEC-11 |
| P06 | CARLA cells (275–291) have execution counts reset (ec=3..23) indicating separate kernel run | HIGH | CARLA section clearly isolated in SEC-18–22 |
| P07 | Cell 292 is an empty code cell | LOW | Removed |
| P08 | Old feature list inconsistency (some cells reference 15 features) | HIGH | One authoritative `FEATURES` list in SEC-01 |
| P09 | Target confusion (Y_global / Y_displacement / Y_ego_aligned) | HIGH | Only Y_ego_aligned used; others noted as deprecated |
| P10 | V15 CARLA-normalised experiment result not summarised clearly | MEDIUM | Explicit table in SEC-21 and SEC-24 |
| P11 | No experiment registry or final benchmark table | HIGH | SEC-24 master benchmark table created |
| P12 | CARLA feature equivalence never evaluated before GRU transfer | HIGH | SEC-19 feature equivalence gate added |
| P13 | Notebook mixes research pipeline with notebook-search/forensic code | HIGH | All forensic cells excluded from master |
| P14 | Multiple ROOT variable redefinitions across cells | HIGH | Single PROJECT_ROOT in SEC-01 |
| P15 | No final validation manifest | MEDIUM | SEC-25 generates `final_notebook_validation.json` |

---

## 3. Preserved Scientific Decisions (DO NOT CHANGE)

- Session-disjoint train/val/test/stress split
- 3s history + 3s future at 10 Hz (30+30 steps)
- 16-feature canonical list (exact order preserved)
- Y_ego_aligned as authoritative prediction target
- Train-only normalisation statistics
- Frozen GRU checkpoint: `data/models/gru_baseline/best_gru_baseline.pt`
- Authoritative reference results: Val ADE=0.5614, Test ADE=0.3175, Stress ADE=0.5415
- CARLA V15 finding: normalisation-only adaptation REJECTED
- Physical-only 8-feature model: REJECTED (still worse than CV on CARLA)

---

## 4. Clean Master Notebook Structure (51 cells)

| Section | Description |
|---------|-------------|
| SEC-00 | Title, research charter, scientific rules |
| SEC-01 | **Single global configuration** (paths, seeds, features, expected counts) |
| SEC-02 | Path & artifact audit |
| SEC-03 | Dataset provenance (I-24 v1.1.1) |
| SEC-04 | Trajectory reconstruction summary (loads from manifests) |
| SEC-05 | Leakage-free split verification |
| SEC-06 | Prediction window protocol |
| SEC-07 | Feature engineering (loads from artifacts) |
| SEC-08 | Feature provenance table with CARLA equivalence status |
| SEC-09 | Target construction verification (Y_ego_aligned) |
| SEC-10 | Physics baselines (CV/CA) |
| SEC-11 | Canonical GRU architecture (one definition) |
| SEC-12 | Checkpoint loading + optional training block |
| SEC-13 | Frozen I-24 evaluation with drift detection |
| SEC-14 | Horizon-wise error analysis |
| SEC-15 | Failure analysis (FDE > 2 m) |
| SEC-16 | Feature ablation status table |
| SEC-17 | Distribution shift summary |
| SEC-18 | CARLA environment setup |
| SEC-19 | **CARLA feature equivalence gate** (blocks unfair transfer claims) |
| SEC-20 | CARLA adaptation dataset audit |
| SEC-21 | CARLA baselines (CV reference) |
| SEC-22 | Frozen GRU CARLA transfer |
| SEC-23 | Domain adaptation roadmap |
| SEC-24 | Master benchmark table |
| SEC-25 | Reproducibility manifest (`final_notebook_validation.json`) |
| SEC-26 | Final research verdict |

---

## 5. Cells Moved to Archive (not in master)

The following cell groups were removed from the main pipeline and should be
placed in `archive/` and `forensics/`:

| Cells | Description | Archive Location |
|-------|-------------|-----------------|
| 214–218 | 54J-R2 through 54J-R5 — domain classifier debugging | forensics/failure_modeling/ |
| 219–225 | 54J-R6 through 54J-R9 — forensic artifact recovery | forensics/target_forensics/ |
| 226–245 | 56-series — checkpoint/feature/notebook forensics | forensics/checkpoint/ |
| 275–292 | CARLA V8–V15 experiments | archive/carla_experiments/ |
| Old 55E (invalid alignment) | Deprecated target analysis | archive/invalid_experiments/ |

---

## 6. Path Fix

All cells now derive paths from one root:

```python
PROJECT_ROOT = Path(
    os.environ.get("TRAJPILOT_ROOT", Path.cwd())
)
```

**To set on Windows:**
```
set TRAJPILOT_ROOT=C:\Users\SOMUJIT\Desktop\cityflow
```

**Or open the notebook from the `cityflow` directory** and `Path.cwd()` will resolve correctly.

---

## 7. How to Run

1. Set environment variable or open notebook from `cityflow/` directory.
2. **Kernel → Restart Kernel and Run All Cells**.
3. SEC-02 will report any missing artifacts.
4. SEC-13 will verify frozen I-24 results against reference values.
5. SEC-25 will generate the reproducibility manifest.
6. SEC-26 will print the final research status.

**If I-24 results drift > 1 % from reference: investigate before modifying code.**

---

## 8. CARLA Status & Multi-Model Evaluation

The CARLA evaluation section documents both the frozen GRU transfer limits and the scenario-by-scenario benchmark across models:

- CARLA runtime: ✅ PASS
- CARLA feature equivalence: ⚠️ NOT PROVEN for full 16 features (8/16 unavailable; physical 8-feature subset validated)
- CARLA frozen GRU: ❌ FAIL (Zero-shot unadapted GRU < CV baseline)
- CARLA Scenario Evaluation (Verified directly from `outputs/tables/final_audit/CARLA_RESULTS.csv`):
  - **Constant Acceleration (CA)**: Wins **3 scenarios** (Scenarios 2, 3, 6) due to idealized synthetic acceleration profiles
  - **Constant Velocity (CV)**: **Ties CA on 1 scenario** (Scenario 1 steady cruise, 0.0000 m FDE)
  - **Ridge Regression**: Wins **1 scenario** (Scenario 5 curve negotiation, 2.2662 m FDE vs CV 5.7991 m)
  - **XGBoost**: Wins **1 scenario** (Scenario 4 high-speed lane change, 7.3060 m FDE)

---

## 9. Comprehensive Scientific Research Documentation

For the complete research charter, mathematical formulation, ASCII pipeline diagrams, dataset provenance tables, highway leaderboards, Pareto curves, and ablation studies, refer to:

- **Scientific Report**: [`TRAJPILOT_RESEARCH_REPORT.md`](file:///c:/Users/SOMUJIT/Desktop/cityflow/TRAJPILOT_RESEARCH_REPORT.md)
- **Interactive Dashboard**: [`dashboard.html`](file:///c:/Users/SOMUJIT/Desktop/cityflow/dashboard.html)

---

*Audit generated and verified by the TrajPilot CityFlow research pipeline.*

