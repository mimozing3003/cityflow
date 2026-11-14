import os
import sys
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
target_nb = PROJECT_ROOT / "TrajPilot_Research_20261005.ipynb"

print("=" * 80)
print(f"EXECUTING ALL 85 CELLS IN NOTEBOOK: {target_nb.name}")
print("=" * 80)

with open(target_nb, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

code_cells = [c for c in nb.cells if c.cell_type == "code"]
print(f"Total cells: {len(nb.cells)} ({len(code_cells)} code cells)")
assert len(code_cells) == 85, f"Expected 85 code cells, found {len(code_cells)}"

# Set environment
os.environ["TRAJPILOT_ROOT"] = str(PROJECT_ROOT)
os.environ["MPLBACKEND"] = "Agg"

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

start_time = time.time()
print("\nExecuting cells via ExecutePreprocessor...")

try:
    ep.preprocess(nb, {"metadata": {"path": str(PROJECT_ROOT)}})
    elapsed = time.time() - start_time
    print(f"\nExecution completed successfully in {elapsed:.2f} seconds with 0 errors!")

    # Check that outputs are populated
    cells_with_output = sum(1 for c in nb.cells if c.cell_type == "code" and len(c.get("outputs", [])) > 0)
    print(f"Outputs verified: {cells_with_output} of {len(code_cells)} code cells have outputs.")

    # Save to all target notebooks
    targets = [
        PROJECT_ROOT / "TrajPilot_Research_20261005.ipynb",
        PROJECT_ROOT / "TrajPilot_CityFlow_Master_Research.ipynb",
        PROJECT_ROOT / "TrajPilot_Research_Master.ipynb",
        PROJECT_ROOT / "TrajPilot_Complete_AllOperations.ipynb"
    ]

    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Successfully saved executed notebook: {target.name} ({target.stat().st_size:,} bytes)")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\nExecution encountered an error after {elapsed:.2f} seconds: {e}")
    # Inspect which cell failed
    code_idx = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            code_idx += 1
            for out in cell.get("outputs", []):
                if out.output_type == "error":
                    print(f"\n--- ERROR IN CODE CELL #{code_idx} (Cell index {i+1}) ---")
                    print(f"Error Name: {out.get('ename')}")
                    print(f"Error Value: {out.get('evalue')}")
                    tb = out.get("traceback", [])
                    for line in tb[-8:]:
                        print("  ", line)
    sys.exit(1)
