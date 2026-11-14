# TrajPilot CityFlow — CARLA 0.9.16 Integration Suite

This directory contains the modular execution and demonstration pipeline integrating the frozen **TrajPilot I-24 GRU** trajectory prediction model with the **CARLA 0.9.16** autonomous driving simulator on Windows.

---

## Modular Architecture

| Stage | Script | Purpose |
| :--- | :--- | :--- |
| **00** | `00_environment_check.py` | Preflight diagnostic: Python 3.12, CUDA, PyTorch, CARLA API, model files, port 2000. |
| **01** | `01_load_model.py` | Exact frozen 227,132-parameter GRU architecture and strict checkpoint loader. |
| **02** | `02_load_normalization.py` | Exact I-24 training normalization statistics and 32-channel tensor constructor. |
| **03** | `03_connect_carla.py` | Robust client connector establishing 10 FPS synchronous mode and Traffic Manager. |
| **04** | `04_spawn_scene.py` | Spawns ego vehicle, surrounding traffic, attaches synchronized 1280x720 RGB camera. |
| **05** | `05_feature_engine.py` | Causal feature extraction engine maintaining 30-frame buffer and metadata tracking. |
| **06** | `06_live_inference.py` | Real-time GRU inference, CUDA synchronization, and 3D world/pixel projections. |
| **07** | `07_visualization.py` | Pygame HUD rendering live video, predicted trajectory, telemetry, and disclaimers. |
| **08** | `08_run_demo.py` | Master interactive demonstration loop with complete `try...finally` teardown. |
| **09** | `09_benchmark.py` | Automated 500-cycle latency profiler measuring GPU, feature, and total pipeline times. |
| **10** | `10_cleanup.py` | Standalone emergency teardown restoring simulator state and removing actors. |

---

## Research Integrity & Scientific Findings

> [!CAUTION]
> **CARLA DEMONSTRATION — FEATURE EQUIVALENCE NOT PROVEN**
> 
> Frozen I-24 GRU inference is demonstrated in CARLA using available/derived features. This engineering demonstration does **not** establish cross-domain generalization or mathematical equivalence between CARLA synthetic features and I-24 physical sensor features.

### Feature Classification Audit

1. **Native CARLA Kinematics**:
   - `vx_causal`, `vy_causal`, `acceleration_native`, `turn_rate_valid` are computed directly from ego vehicle IMU and transformation matrices rotated into the local heading frame.
2. **Approximate Derived Kinematics**:
   - `speed_derived` (backward finite difference), `acceleration_causal_0p5s` (0.5s causal backward window), `curvature_valid` (yaw rate / speed ratio), and `jerk_causal` (1-step acceleration derivative).
3. **Approximate Interaction Features**:
   - `SpaceGap`, `RelativeVelocity`, `LateralDistance`, and `time_headway` are derived from leading actor bounding box geometry. When no leading vehicle is in the forward corridor, these return `NaN`.
4. **Unresolved Lane Context Features**:
   - `L_Approach`, `R_Approach`, `L_Adjacent`, `R_Adjacent` are not proven semantically equivalent to I-24 highway lane definitions and are handled via the model's explicit missingness mask (normalized value = 0.0, missingness indicator = 1.0).

---

## How to Run

### 1. Launch CARLA Simulator
```cmd
C:\CARLA_0.9.16\Start_CARLA.bat
```

### 2. Run Master TrajPilot Demo
```cmd
RUN_TRAJPILOT_CARLA.bat
```
Or directly via Python in the `carla0916` or `carla_py312` environment:
```cmd
python carla_demo\08_run_demo.py
```

### 3. Run Performance Benchmark
```cmd
python carla_demo\09_benchmark.py
```
