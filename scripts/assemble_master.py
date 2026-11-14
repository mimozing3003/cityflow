import shutil
from pathlib import Path
import nbformat as nbf

from cells_p1 import get_cells_part1
from cells_p2 import get_cells_part2
from cells_p3 import get_cells_part3
from cells_p4 import get_cells_part4
from cells_p5_tgsim import get_cells_part5_tgsim
from cells_p6_highway import get_cells_part6_highway
from cells_p7_ablation import get_cells_part7_ablation

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3"
    }
    nb.metadata["language_info"] = {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12.0"
    }

    cells = []
    cells.extend(get_cells_part1())
    cells.extend(get_cells_part2())
    cells.extend(get_cells_part3())
    cells.extend(get_cells_part4())
    cells.extend(get_cells_part5_tgsim())
    cells.extend(get_cells_part6_highway())
    cells.extend(get_cells_part7_ablation())

    nb.cells = cells
    print(f"Total cells assembled: {len(nb.cells)}")

    out_path = Path("TrajPilot_CityFlow_Master_Research.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")

    # Also keep TrajPilot_Research_Master.ipynb synchronized
    target_active = Path("TrajPilot_Research_Master.ipynb")
    shutil.copyfile(out_path, target_active)
    print(f"Synchronized: {target_active}")

    return out_path

if __name__ == "__main__":
    build_notebook()
