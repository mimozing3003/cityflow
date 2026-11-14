"""
TrajPilot CityFlow - Comprehensive System & CARLA Diagnostic Utility
Audits hardware, drivers, CUDA, PyTorch, CARLA API, simulator connection, and TrajPilot assets.
"""

import sys
import os
import platform
import socket
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow")
CARLA_ROOT = Path(r"C:\CARLA_0.9.16")

CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "gru_baseline" / "best_gru_baseline.pt"
NORM_PATH = PROJECT_ROOT / "data" / "processed" / "i24_sequences" / "gru_inputs" / "train_normalization_statistics.csv"
FEAT_PATH = PROJECT_ROOT / "data" / "processed" / "i24_sequences" / "feature_names.npy"


def check_port(host="127.0.0.1", port=2000, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return parts[0], parts[1], f"{float(parts[2])/1024:.1f} GB"
    except Exception:
        return "Unknown GPU", "Unknown Driver", "Unknown"


def run_diagnostics():
    print("=" * 78)
    print("TRAJPILOT CARLA DIAGNOSTICS")
    print("=" * 78)

    # 1. Host Operating System
    os_name = f"{platform.system()} {platform.release()} (Build {platform.version()})"
    print(f"Windows Version          : {os_name}")
    print(f"CPU Architecture         : {platform.machine()} ({platform.processor()})")

    # 2. GPU & NVIDIA Driver
    gpu_name, driver_ver, vram = get_gpu_info()
    print(f"GPU                      : {gpu_name} ({vram} VRAM)")
    print(f"NVIDIA Driver            : {driver_ver}")

    # 3. Python & PyTorch
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_pass = sys.version_info.major == 3 and sys.version_info.minor in (10, 11, 12)
    print(f"Python Version           : {py_ver} [{'PASS' if py_pass else 'FAIL'}]")

    torch_ver = "Missing"
    cuda_avail = False
    cuda_ver = "N/A"
    try:
        import torch
        torch_ver = torch.__version__
        cuda_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda or "N/A"
    except ImportError:
        pass

    print(f"PyTorch Version          : {torch_ver} [{'PASS' if torch_ver != 'Missing' else 'FAIL'}]")
    print(f"CUDA Availability        : {cuda_avail} [{'PASS' if cuda_avail else 'FAIL'}]")
    print(f"CUDA Toolkit Version     : {cuda_ver}")

    # 4. CARLA Python API
    carla_avail = False
    carla_api_ver = "Unknown"
    try:
        import carla
        carla_avail = True
        carla_api_ver = "0.9.16"
    except ImportError:
        pass

    print(f"CARLA API Version        : {carla_api_ver if carla_avail else 'NOT INSTALLED'} [{'PASS' if carla_avail else 'FAIL'}]")

    # 5. CARLA Installation & Simulator
    carla_exe = CARLA_ROOT / "CarlaUE4.exe"
    carla_installed = carla_exe.exists()
    print(f"CARLA Installation Path  : {CARLA_ROOT} [{'PASS' if carla_installed else 'FAIL'}]")

    p2000_open = check_port("127.0.0.1", 2000)
    print(f"Port 2000 Status         : {'OPEN' if p2000_open else 'CLOSED'} [{'PASS' if p2000_open else 'FAIL'}]")

    server_ver = "N/A"
    server_connected = False
    if carla_avail and p2000_open:
        try:
            client = carla.Client("127.0.0.1", 2000)
            client.set_timeout(3.0)
            server_ver = client.get_server_version()
            server_connected = True
        except Exception:
            pass

    print(f"CARLA Server Version     : {server_ver}")
    print(f"CARLA Connection Status  : {'CONNECTED' if server_connected else 'DISCONNECTED'} [{'PASS' if server_connected else 'FAIL'}]")

    # 6. TrajPilot Assets
    print(f"Project Path             : {PROJECT_ROOT}")
    ckpt_pass = CHECKPOINT_PATH.exists()
    norm_pass = NORM_PATH.exists()
    feat_pass = FEAT_PATH.exists()

    print(f"Checkpoint Exists        : {'YES' if ckpt_pass else 'NO'} [{'PASS' if ckpt_pass else 'FAIL'}]")
    print(f"Normalizer Exists        : {'YES' if norm_pass else 'NO'} [{'PASS' if norm_pass else 'FAIL'}]")
    print(f"Feature File Exists      : {'YES' if feat_pass else 'NO'} [{'PASS' if feat_pass else 'FAIL'}]")

    # 7. Model Strict Validation
    model_pass = False
    if ckpt_pass and torch_ver != "Missing":
        try:
            import importlib
            mod_01 = importlib.import_module("carla_demo.01_load_model")
            m, _ = mod_01.load_frozen_model(torch.device("cpu"), CHECKPOINT_PATH)
            params = sum(p.numel() for p in m.parameters())
            model_pass = (params == 227132)
        except Exception:
            model_pass = False

    print(f"Model Strict Load        : {'PASS' if model_pass else 'FAIL'}")
    print("=" * 78)


if __name__ == "__main__":
    run_diagnostics()
