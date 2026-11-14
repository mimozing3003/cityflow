import sys
import shutil
from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = Path("TrajPilot_CityFlow_Master_Research.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

print(f"Executing notebook: {notebook_path} ({len(nb.cells)} cells)...")

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

try:
    ep.preprocess(nb, {"metadata": {"path": str(Path.cwd())}})
    print("Execution completed with 0 errors!")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Updated {notebook_path} with execution outputs.")

    # Also update TrajPilot_Research_Master.ipynb with execution outputs
    shutil.copyfile(notebook_path, "TrajPilot_Research_Master.ipynb")
    print("Synchronized executed notebook to TrajPilot_Research_Master.ipynb.")

except Exception as e:
    print(f"ERROR during notebook execution: {e}")
    # Print error details from cells
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for out in cell.get("outputs", []):
                if out.output_type == "error":
                    print(f"\n--- Error in Cell {i} ---")
                    print(out.ename, out.evalue)
                    for trace_line in out.traceback[-5:]:
                        print("  ", trace_line)
    sys.exit(1)
