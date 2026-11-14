import os
import shutil
from pathlib import Path

SOURCE_ROOT = Path(r"c:\Users\SOMUJIT\Desktop\cityflow")
TARGET_DIRS = [
    SOURCE_ROOT / "TRAJPILOT CITYFLOW",
    Path(r"c:\Users\SOMUJIT\Desktop\TRAJPILOT CITYFLOW")
]

for target in TARGET_DIRS:
    print(f"\n=======================================================")
    print(f"PACKAGING TRAJPILOT CITYFLOW INTO: {target}")
    print(f"=======================================================")
    
    # Create directory structure
    (target / "reports").mkdir(parents=True, exist_ok=True)
    (target / "models").mkdir(parents=True, exist_ok=True)
    (target / "data").mkdir(parents=True, exist_ok=True)
    (target / "carla_demo").mkdir(parents=True, exist_ok=True)

    # 1. Primary Notebooks & Core Documents
    primary_files = [
        ("TrajPilot_Research_20261005.ipynb", target / "TrajPilot_Research_20261005.ipynb"),
        ("TrajPilot_Complete_AllOperations.ipynb", target / "TrajPilot_Complete_AllOperations.ipynb"),
        ("TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx", target / "TrajPilot_Research_20261005_CELL_BY_CELL_EXPLANATION.docx"),
        ("RUN_TRAJPILOT_CARLA.bat", target / "RUN_TRAJPILOT_CARLA.bat"),
        ("diagnostics.py", target / "diagnostics.py"),
        ("environment.yml", target / "environment.yml"),
    ]

    for src_name, dst_path in primary_files:
        src = SOURCE_ROOT / src_name
        if src.exists():
            shutil.copy2(src, dst_path)
            print(f"[COPIED] {src_name:40s} -> {dst_path.name} ({src.stat().st_size:,} bytes)")
        else:
            print(f"[MISSING] {src_name}")

    # 2. Reports
    report_files = [
        ("TRAJPILOT_RESEARCH_REPORT.md", target / "reports" / "TRAJPILOT_RESEARCH_REPORT.md"),
        ("CARLA_SETUP_REPORT.md", target / "reports" / "CARLA_SETUP_REPORT.md"),
        ("NOTEBOOK_AUDIT_REPORT.md", target / "reports" / "NOTEBOOK_AUDIT_REPORT.md"),
        ("dashboard.html", target / "reports" / "dashboard.html"),
    ]

    for src_name, dst_path in report_files:
        src = SOURCE_ROOT / src_name
        if src.exists():
            shutil.copy2(src, dst_path)
            print(f"[COPIED] {src_name:40s} -> reports/{dst_path.name}")
        else:
            print(f"[MISSING] {src_name}")

    # 3. Models
    model_files = [
        ("data/models/gru_baseline/best_gru_baseline.pt", target / "models" / "best_gru_baseline.pt"),
        ("data/models/final_model/ridge_final.joblib", target / "models" / "ridge_final.joblib"),
        ("data/models/final_model/xgboost_final.json", target / "models" / "xgboost_final.json"),
        ("data/models/final_model/scaler_final.joblib", target / "models" / "scaler_final.joblib"),
    ]

    for src_rel, dst_path in model_files:
        src = SOURCE_ROOT / src_rel
        if src.exists():
            shutil.copy2(src, dst_path)
            print(f"[COPIED] {src_rel:40s} -> models/{dst_path.name} ({src.stat().st_size:,} bytes)")
        else:
            print(f"[MISSING] {src_rel}")

    # 4. Essential Data & Manifests
    data_files = [
        ("data/processed/i24_sequences/gru_inputs/train_normalization_statistics.csv", target / "data" / "train_normalization_statistics.csv"),
        ("data/processed/i24_sequences/feature_names.npy", target / "data" / "feature_names.npy"),
        ("data/manifests/CARLA_FEATURE_EQUIVALENCE.csv", target / "data" / "CARLA_FEATURE_EQUIVALENCE.csv"),
        ("data/manifests/CARLA_RESULTS.csv", target / "data" / "CARLA_RESULTS.csv"),
        ("data/manifests/final_real_world_results.csv", target / "data" / "final_real_world_results.csv"),
        ("data/manifests/final_config.json", target / "data" / "final_config.json"),
        ("data/manifests/final_feature_manifest.json", target / "data" / "final_feature_manifest.json"),
        ("data/manifests/final_training_manifest.json", target / "data" / "final_training_manifest.json"),
    ]

    for src_rel, dst_path in data_files:
        src = SOURCE_ROOT / src_rel
        if src.exists():
            shutil.copy2(src, dst_path)
            print(f"[COPIED] {src_rel:40s} -> data/{dst_path.name}")
        else:
            print(f"[MISSING] {src_rel}")

    # 5. CARLA Demo Suite
    carla_src = SOURCE_ROOT / "carla_demo"
    if carla_src.exists():
        for item in carla_src.iterdir():
            if item.is_file():
                shutil.copy2(item, target / "carla_demo" / item.name)
                print(f"[COPIED] carla_demo/{item.name:25s} -> carla_demo/{item.name}")

    # 6. Generate requirements.txt
    reqs_content = """# TrajPilot CityFlow Python Dependencies
torch>=2.5.0
torchvision>=0.20.0
numpy>=1.26.0,<2.3.0
pandas>=2.2.0
scipy>=1.12.0
scikit-learn>=1.4.0
xgboost>=2.0.0
joblib>=1.3.0
matplotlib>=3.8.0
seaborn>=0.13.0
pygame>=2.6.0
python-docx>=1.1.0
jupyter>=1.0.0
ipykernel>=6.29.0
"""
    with open(target / "requirements.txt", "w", encoding="utf-8") as f:
        f.write(reqs_content)
    print(f"[CREATED] requirements.txt")

print("\nPackaging script finished successfully!")
