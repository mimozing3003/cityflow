"""
TrajPilot CityFlow - Stage 00: Environment & Preflight Diagnostic Check
Validates Python, CUDA, PyTorch, CARLA API, model files, and simulator port.
"""

import sys
import os
import socket
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT = PROJECT_ROOT / "data" / "models" / "gru_baseline" / "best_gru_baseline.pt"
NORM_FILE = PROJECT_ROOT / "data" / "processed" / "i24_sequences" / "gru_inputs" / "train_normalization_statistics.csv"
FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "i24_sequences" / "feature_names.npy"


def check_port_2000(host="127.0.0.1", port=2000, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def run_checks():
    print("=" * 78)
    print("TRAJPILOT CITYFLOW | 00 - ENVIRONMENT PREFLIGHT CHECK")
    print("=" * 78)
    
    results = {}

    # 1. Python Version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_pass = (sys.version_info.major == 3 and sys.version_info.minor in (10, 11, 12))
    results["Python Version"] = (py_pass, py_ver, "Requires Python 3.10-3.12 for CARLA 0.9.16")
    print(f"[{'PASS' if py_pass else 'FAIL'}] Python Version        : {py_ver}")

    # 2. PyTorch & CUDA
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        results["PyTorch"] = (True, torch.__version__, "")
        results["CUDA"] = (cuda_avail, f"{gpu_name} (CUDA {torch.version.cuda})", "GPU acceleration available")
        print(f"[{'PASS'}] PyTorch               : {torch.__version__}")
        print(f"[{'PASS' if cuda_avail else 'WARN'}] CUDA GPU Acceleration : {gpu_name} (CUDA {torch.version.cuda})")
    except ImportError as e:
        results["PyTorch"] = (False, "Not installed", str(e))
        results["CUDA"] = (False, "N/A", "PyTorch missing")
        print(f"[FAIL] PyTorch               : NOT INSTALLED")

    # 3. CARLA Python API
    try:
        import carla
        # CARLA module doesn't always expose __version__, get from client or wheel
        carla_path = getattr(carla, "__file__", "carla")
        results["CARLA API"] = (True, carla_path, "Module imported successfully")
        print(f"[{'PASS'}] CARLA Python API      : Available ({carla_path})")
    except ImportError as e:
        results["CARLA API"] = (False, "Not installed", str(e))
        print(f"[FAIL] CARLA Python API      : NOT INSTALLED ({e})")

    # 4. TrajPilot Checkpoint
    ckpt_pass = CHECKPOINT.exists()
    ckpt_size = f"{CHECKPOINT.stat().st_size:,} bytes" if ckpt_pass else "Missing"
    results["Model Checkpoint"] = (ckpt_pass, str(CHECKPOINT), ckpt_size)
    print(f"[{'PASS' if ckpt_pass else 'FAIL'}] Model Checkpoint      : {ckpt_size}")

    # 5. Normalization Statistics
    norm_pass = NORM_FILE.exists()
    norm_size = f"{NORM_FILE.stat().st_size:,} bytes" if norm_pass else "Missing"
    results["Normalizer"] = (norm_pass, str(NORM_FILE), norm_size)
    print(f"[{'PASS' if norm_pass else 'FAIL'}] Normalization File    : {norm_size}")

    # 6. Feature Names
    feat_pass = FEATURE_FILE.exists()
    feat_size = f"{FEATURE_FILE.stat().st_size:,} bytes" if feat_pass else "Missing"
    results["Feature File"] = (feat_pass, str(FEATURE_FILE), feat_size)
    print(f"[{'PASS' if feat_pass else 'FAIL'}] Feature Names File    : {feat_size}")

    # 7. Simulator Port 2000
    p2000_open = check_port_2000()
    results["Port 2000"] = (p2000_open, "127.0.0.1:2000", "CARLA Simulator Listening" if p2000_open else "CARLA Simulator Not Running")
    print(f"[{'PASS' if p2000_open else 'INFO'}] CARLA Port 2000       : {'OPEN (Server Ready)' if p2000_open else 'NOT ACTIVE (Simulator not yet started)'}")

    print("=" * 78)
    all_critical_pass = py_pass and results["PyTorch"][0] and results["CARLA API"][0] and ckpt_pass and norm_pass and feat_pass
    print(f"PREFLIGHT STATUS: {'ALL CRITICAL CHECKS PASSED' if all_critical_pass else 'FAILURES DETECTED'}")
    print("=" * 78)
    return all_critical_pass


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
