import nbformat
import json

with open("TrajPilot_Research_Master.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print(f"Total cells in original: {len(nb.cells)}")

# Look for key implementations:
# 1. TrajPilotGRU class
# 2. Physics baselines (CV, CA)
# 3. Model loading / checkpoint loading
# 4. Evaluation logic
# 5. CARLA logic

for i, cell in enumerate(nb.cells):
    src = cell.source
    if "class " in src and "GRU" in src:
        print(f"Cell {i} has GRU class definition:")
        for line in src.splitlines()[:10]:
            print("  ", line)
    if "best_gru_baseline.pt" in src:
        print(f"Cell {i} references best_gru_baseline.pt")
    if "0.561" in src or "0.317" in src:
        print(f"Cell {i} references reference metrics (0.561 / 0.317)")
