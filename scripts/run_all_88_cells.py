import sys
import os
import time
import shutil
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(".").resolve()
notebook_path = PROJECT_ROOT / "TrajPilot_CityFlow_Master_Research.ipynb"

print("=" * 80)
print(f"RUN ALL CELLS PIPELINE: {notebook_path.name}")
print("=" * 80)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == "code"]
print(f"Total cells to execute: {len(nb.cells)} ({len(code_cells)} code cells)")

# Ensure environmental variables are set
os.environ["TRAJPILOT_ROOT"] = str(PROJECT_ROOT)
os.environ["MPLBACKEND"] = "Agg"

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

start_time = time.time()
print("\nStarting execution in python3 kernel...")

try:
    ep.preprocess(nb, {"metadata": {"path": str(PROJECT_ROOT)}})
    elapsed = time.time() - start_time
    print(f"\nExecution completed successfully in {elapsed:.2f} seconds with 0 errors!")

    # Check that code cells have outputs
    cells_with_output = sum(1 for c in nb.cells if c.cell_type == "code" and len(c.get("outputs", [])) > 0)
    print(f"Verification: {cells_with_output} of {len(code_cells)} code cells have populated outputs.")

    # Save to all target notebooks
    targets = [
        PROJECT_ROOT / "TrajPilot_CityFlow_Master_Research.ipynb",
        PROJECT_ROOT / "TrajPilot_Research_Master.ipynb",
        PROJECT_ROOT / "TrajPilot_Complete_AllOperations.ipynb"
    ]

    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Saved executed notebook to: {target.name} ({target.stat().st_size:,} bytes)")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\nExecution encountered an error after {elapsed:.2f} seconds: {e}")
    # Inspect which cell failed
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.output_type == "error":
                    print(f"\n--- ERROR IN CELL {i+1} ---")
                    print(f"Error Name: {out.get('ename')}")
                    print(f"Error Value: {out.get('evalue')}")
                    tb = out.get("traceback", [])
                    for line in tb[-6:]:
                        print("  ", line)
    sys.exit(1)
