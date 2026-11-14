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
