# TRAJPILOT CITYFLOW
### Causal, Kinematically Constrained Vehicle Trajectory Prediction & Live CARLA Simulation Transfer
**Academic Research Project Handover & Evaluation Repository**

---

## Executive Summary & Research Abstract

**TrajPilot CityFlow** is an end-to-end autonomous vehicle trajectory forecasting system developed to address a fundamental challenge in intelligent transportation: **can a causal trajectory forecasting model trained on real-world highway sensor data generalize across unseen highway corridors, congested urban arterials, and closed-loop 3D autonomous driving simulations without physical violations?**

Using high-resolution real-world trajectory data from the **I-24 MOTION highway testbed** (16 canonical features, 30-frame history, 30-frame future horizon at 10 Hz / 3.0 seconds), this research rigorously evaluates and compares:
1. **Classical Physics Baselines**: Constant Velocity (CV) and Constant Acceleration (CA).
2. **Deep Recurrent Sequence Architectures**: 2-layer, 128-hidden Gated Recurrent Unit (**TrajPilot GRU**, 227,132 parameters).
3. **Kinematically Constrained Machine Learning Models**: Constrained Ridge Regression and Constrained Gradient Boosted Decision Trees (XGBoost).
4. **Closed-Loop CARLA 0.9.16 Simulator Deployment**: Real-time 10 Hz synchronous closed-loop vehicle control, causal feature extraction, and HUD visualization.

### Key Scientific Findings:
- **The Kinematic Constraint Breakthrough**: Unconstrained learned models suffer catastrophic forward drift when vehicles stop (e.g. at red lights on urban arterials). By implementing an authoritative **Kinematic Constraint Decoder** that clamps predicted displacement based on causality and physical plausibility, our constrained models achieve a **0.00% Physical Violation Rate** across all test domains.
- **Highway Generalization**: On unseen interstate testing (TGSIM I-294 @ 61 mph), **Constrained XGBoost** achieves Rank #1 Final Displacement Error (**FDE = 1.6912 m** vs. 2.6742 m for Constant Velocity, a **36.8% error reduction**).
- **Sub-Millisecond Real-Time Efficiency**: **Constrained Ridge Regression** achieves an ultra-low inference latency of **0.0011 ms (909,000 predictions/second)**, outperforming deep neural networks by over $2,000\times$ in computational efficiency.
- **CARLA 6-Scenario Transfer**: Evaluated across 6 standardized highway scenarios in CARLA 0.9.16, the system outperforms straight-line physics by **60.9% on highway curves** (FDE = 2.27 m vs. 5.80 m).

---

## Handover Package Organization

```
TRAJPILOT CITYFLOW/
│
├── README.md                                                  # Master Project Handover & Evaluation Guide (This Document)
│
├── TrajPilot_Research_20261005.ipynb                           # PRIMARY RESEARCH NOTEBOOK (85 Steps, Complete Research Pipeline)
├── TrajPilot_Complete_AllOperations.ipynb                     # EXTENDED NOTEBOOK (85 Steps + Live CARLA 0.9.16 Operational Cells)
├── TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx  # COMPREHENSIVE 85-STEP WORD REPORT (Detailed Academic Explanation)
│
├── RUN_TRAJPILOT_CARLA.bat                                    # One-Click Windows Launcher for Live CARLA Simulation
├── diagnostics.py                                             # System, Hardware, GPU & Asset Health Auditor
├── environment.yml                                            # Conda Environment Specification File
├── requirements.txt                                           # Pip Package Requirements File
│
├── reports/                                                   # Forensic Reports & Academic Thesis Documentation
│   ├── TRAJPILOT_RESEARCH_REPORT.md                           # Comprehensive Research Thesis & Leaderboard Report
│   ├── CARLA_SETUP_REPORT.md                                  # CARLA 0.9.16 Installation & Real-Time Benchmark Audit
│   ├── NOTEBOOK_AUDIT_REPORT.md                               # Systematic Notebook Architecture & Correction Audit
│   └── dashboard.html                                         # Interactive HTML Leaderboard & Visual Analytics
│
├── models/                                                    # Trained Production & Research Checkpoints
│   ├── best_gru_baseline.pt                                   # Frozen 227,132-Param PyTorch GRU (I-24 Trained Checkpoint)
│   ├── ridge_final.joblib                                     # Production Kinematically Constrained Ridge Model
│   ├── xgboost_final.json                                     # Production Kinematically Constrained XGBoost Model
│   └── scaler_final.joblib                                    # Production Feature Normalization Scaler
│
├── data/                                                      # Certified Datasets, Normalizers & Protocol Manifests
│   ├── train_normalization_statistics.csv                     # Authoritative I-24 16-Feature Mean & Standard Deviation
│   ├── feature_names.npy                                      # Canonical 16 Feature Names Array
│   ├── CARLA_FEATURE_EQUIVALENCE.csv                          # Certified 8/8 Feature Equivalence Table
│   ├── CARLA_RESULTS.csv                                      # CARLA 6-Scenario Quantitative Benchmark Data
│   ├── final_real_world_results.csv                           # Multi-Domain Real-World Benchmark Leaderboard
│   ├── final_config.json                                      # Canonical Global Experiment Configuration
│   ├── final_feature_manifest.json                            # 16-Feature Protocol & Derivation Manifest
│   └── final_training_manifest.json                           # Model Training Parameters Manifest
│
└── carla_demo/                                                # Modular Live CARLA 0.9.16 Integration Suite
    ├── 00_environment_check.py                                # Preflight Hardware, CUDA & Environment Diagnostic
    ├── 01_load_model.py                                       # Frozen GRU Architecture & Strict Checkpoint Loader
    ├── 02_load_normalization.py                               # 32-Channel Input Constructor (16 Normalized + 16 Mask)
    ├── 03_connect_carla.py                                    # 10 FPS Synchronous Simulator Connector
    ├── 04_spawn_scene.py                                      # Spawns Ego Vehicle, Traffic & 1280x720 RGB Camera
    ├── 05_feature_engine.py                                   # Causal Kinematic & Interaction Feature Extractor
    ├── 06_live_inference.py                                   # Live PyTorch Forward Pass & 3D World/Pixel Projections
    ├── 07_visualization.py                                    # Pygame Dashboard HUD with Live Trajectory Overlay
    ├── 08_run_demo.py                                         # Master Interactive Real-Time Demonstration Loop
    ├── 09_benchmark.py                                        # Automated 500-Cycle CUDA Latency Profiler
    ├── 10_cleanup.py                                          # Safe Standalone Teardown & State Restoration
    └── README.md                                              # CARLA Demonstration Architecture Guide
```

---

## Recommended Teacher / Evaluator Walkthrough

To review and evaluate this project thoroughly, follow these four stages:

### Stage 1: The Master Research Notebook
Open and examine **`TrajPilot_Research_20261005.ipynb`** in Jupyter Lab or VS Code.
- Structured into **85 explicitly numbered and explained research steps**.
- Covers the full experimental lifecycle:
  1. *Steps 1–15*: Research charter, dataset acquisition (I-24 MOTION), leakage-free session-disjoint splits, coordinate frame standardization ($Y_{\text{ego\_aligned}}$).
  2. *Steps 16–35*: Causal feature engineering (16 features), missingness masks (32 channels), physics baselines (CV, CA divergence analysis).
  3. *Steps 36–55*: Deep sequence modeling (227,132-parameter GRU), loss curves, horizon-wise error propagation.
  4. *Steps 56–70*: Out-of-distribution evaluation (TGSIM I-294, I-90/I-94, Foggy Bottom arterial), standstill drift diagnosis.
  5. *Steps 71–85*: Kinematically constrained models (Ridge, XGBoost), 8/8 CARLA feature-equivalence certification, 6-scenario CARLA closed-loop evaluation, and final model selection.

### Stage 2: The Cell-by-Cell Word Report
Review **`TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx`**.
- A comprehensive **972 KB academic document** written specifically to explain every single cell of the master notebook.
- Details mathematical formulations, inputs, outputs, biological/physical motivations, and engineering decisions.

### Stage 3: Quantitative Results & Leaderboards
Inspect the multi-domain benchmark leaderboard in **`reports/TRAJPILOT_RESEARCH_REPORT.md`** or open **`reports/dashboard.html`** in any web browser.

### Stage 4: Live CARLA 0.9.16 Interactive Demonstration
If CARLA 0.9.16 is installed on the machine, run:
```cmd
RUN_TRAJPILOT_CARLA.bat
```
This automatically verifies environment prerequisites, checks port 2000, connects to CARLA, spawns traffic, and displays the real-time Pygame HUD with 3.0s trajectory overlay.

---

## Authoritative Benchmark Results

### 1. Multi-Domain Real-World Highway & Urban Leaderboard

| Model | Highway Rank | I-294 FDE (m) *(Free-Flow)* | I-90 FDE (m) *(Congested)* | Mean Highway FDE (m) | Foggy Bottom FDE (m) *(Urban Stop-and-Go)* | Latency (ms) | Physical Violations | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constrained XGBoost** | **#1** | **1.6912** | **1.3831** | **1.5372** | 1.3410 | 0.0420 | **0.00%** | **WINNER (Highway)** |
| **Constrained Ridge** | **#2** | 1.8105 | 1.4120 | 1.6112 | **1.2842** | **0.0011** | **0.00%** | **WINNER (Pareto)** |
| Constant Velocity (CV) | #3 | 2.6742 | 1.7520 | 2.2131 | 1.8920 | 0.0002 | 0.00% | Physics Baseline |
| TrajPilot GRU (Frozen) | #4 | 2.4510 | 2.1105 | 2.2807 | 3.4120 | 2.0960 | 4.82% | Source-Overfit |
| Constant Acceleration (CA) | #5 | 5.8920 | 4.9210 | 5.4065 | 4.1200 | 0.0004 | 12.30% | Divergent ($t^2$) |
| Unconstrained Ridge | #6 | 1.9420 | 1.5830 | 1.7625 | 5.8410 | 0.0010 | 18.40% | Standstill Drift |

### 2. CARLA 0.9.16 Closed-Loop 6-Scenario Evaluation

| Scenario | Maneuver Description | Speed / Profile | CV Baseline FDE (m) | Ridge FDE (m) | XGBoost FDE (m) | Improvement vs. CV |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **S1** | Steady Highway Cruise | 64 mph (28.6 m/s) | **0.0000** | 0.6012 | 0.7214 | CV Optimal |
| **S2** | Dynamic Acceleration | $+2.0\text{ m/s}^2$ | 8.4512 | **7.4610** | 7.5820 | **+11.7%** |
| **S3** | Emergency Deceleration | $-4.5\text{ m/s}^2$ | 9.8120 | **8.9210** | 9.0140 | **+9.1%** |
| **S4** | Lateral Lane Change | $1.5\text{ m/s}$ lateral | 8.0410 | 7.4120 | **7.3110** | **+9.1%** |
| **S5** | Curved Highway ($R=250\text{ m}$) | 52 mph (23.2 m/s) | 5.7991 | **2.2662** | 2.4120 | **+60.9%** |
| **S6** | Congested Traffic Queue | Stop-and-Go | 2.1410 | **1.4120** | 1.4910 | **+34.1%** |

### 3. Inference Latency & Hardware Profile (NVIDIA RTX 3060)

| Stage | Mean Latency | Median Latency | P95 Latency | P99 Latency | FPS Throughput | Automotive Headroom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Constrained Ridge** | **0.0011 ms** | **0.0010 ms** | **0.0015 ms** | **0.0021 ms** | **909,000 FPS** | **>99.9%** |
| **Constrained XGBoost** | **0.0420 ms** | **0.0390 ms** | **0.0580 ms** | **0.0710 ms** | **23,800 FPS** | **99.9%** |
| **TrajPilot GRU (PyTorch)** | **2.7750 ms** | **2.6440 ms** | **3.5380 ms** | **4.1660 ms** | **360.3 FPS** | **95.8%** |

---

## Scientific Integrity Statement

> [!IMPORTANT]
> **RESEARCH INTEGRITY & METHODOLOGICAL TRANSPARENCY**
>
> 1. **Cross-Domain Generalization vs. Direct Equivalence**: Real-world highway dynamics (I-24) differ substantially from synthetic CARLA physics. Demonstrating live inference inside CARLA confirms real-time system integration, but does **not** prove mathematical equivalence between synthetic sensors and physical radar/camera arrays.
> 2. **Explicit Missingness Protocol**: Four lane context features (`L_Approach`, `R_Approach`, `L_Adjacent`, `R_Adjacent`) could not be resolved from CARLA geometry with verified semantic equivalence to I-24. In strict accordance with the TrajPilot missingness protocol, these features are explicitly masked to `0.0` with missingness indicator masks set to `1.0`.
> 3. **The Importance of Negative Findings**: Our research explicitly documents and analyzes negative results: unregularized Constant Acceleration fails catastrophically due to $t^2$ noise amplification; unconstrained Ridge drifts by $>5.8\text{ m}$ at standstill; and unadapted deep GRUs overfit to training session geometry. Transparent documentation of these failure modes led directly to the development of the winning Kinematic Constraint Decoder.

---

## Environment Setup & Quick Start

### Option A: Using Conda (Recommended)
```cmd
conda env create -f environment.yml
conda activate carla_py312
jupyter lab TrajPilot_Research_20261005.ipynb
```

### Option B: Using Pip
```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter lab TrajPilot_Research_20261005.ipynb
```

### Verify System Diagnostics:
```cmd
python diagnostics.py
```
Outputs a complete hardware, GPU, PyTorch, model checkpoint, and normalization health matrix.

---
*Project developed and packaged for academic handover.*
