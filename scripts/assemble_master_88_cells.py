import os
import sys
import shutil
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(".").resolve()
sys.path.append(str(PROJECT_ROOT / "scripts"))

from cells_p1 import get_cells_part1
from cells_p2 import get_cells_part2
from cells_p3 import get_cells_part3
from cells_p4 import get_cells_part4
from cells_p5_tgsim import get_cells_part5_tgsim
from cells_p6_highway import get_cells_part6_highway
from cells_p7_ablation import get_cells_part7_ablation
from cells_p8_final import get_cells_part8_final

def build_88_cell_notebook():
    print("=" * 80)
    print("ASSEMBLING DEFINITIVE MASTER NOTEBOOK WITH ALL 88 CODE CELLS")
    print("=" * 80)

    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }

    all_parts = [
        ("Part 1: Environment, Configuration, Data Provenance & Split Audit", get_cells_part1()),
        ("Part 2: Causal Feature Construction, Physics Baselines & Causal GRU Architecture", get_cells_part2()),
        ("Part 3: Robustness Analysis, Distribution Shift & CARLA Adaptation Diagnostics", get_cells_part3()),
        ("Part 4: Multi-Scenario CARLA Validation, Predictive Uncertainty & Master Benchmark", get_cells_part4()),
        ("Part 5: TGSIM Real-World Cross-Dataset Generalization & Schema Mapping", get_cells_part5_tgsim()),
        ("Part 6: Highway Multi-Model Benchmark (XGBoost, Ridge, MLP, GRU, TCN, Transformer)", get_cells_part6_highway()),
        ("Part 7: Stage 7 Ablation, LOFO, Model Interpretability, Latency & Pareto Frontier", get_cells_part7_ablation()),
        ("Part 8: Kinematic Constraints, Multi-Domain Verification & CARLA Online Deployment", get_cells_part8_final())
    ]

    total_cells = []
    code_count = 0
    md_count = 0

    for part_title, part_cells in all_parts:
        c_code = sum(1 for c in part_cells if c.cell_type == "code")
        c_md = sum(1 for c in part_cells if c.cell_type == "markdown")
        print(f"Adding {part_title:75s} | Code: {c_code:2d} | MD: {c_md:2d} | Total: {len(part_cells):2d}")
        total_cells.extend(part_cells)
        code_count += c_code
        md_count += c_md

    nb.cells = total_cells

    print("-" * 80)
    print(f"GRAND TOTAL CELLS ASSEMBLED: {len(nb.cells)}")
    print(f"  * Code Cells     : {code_count} (Target: Exactly 88)")
    print(f"  * Markdown Cells : {md_count}")
    print("-" * 80)
    assert code_count == 88, f"ERROR: Expected exactly 88 code cells, found {code_count}!"

    # Target notebooks
    targets = [
        PROJECT_ROOT / "TrajPilot_CityFlow_Master_Research.ipynb",
        PROJECT_ROOT / "TrajPilot_Research_Master.ipynb",
        PROJECT_ROOT / "TrajPilot_Complete_AllOperations.ipynb"
    ]

    for target in targets:
        with open(target, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Successfully written to: {target.name} ({target.stat().st_size:,} bytes)")

    return targets[0]

if __name__ == "__main__":
    build_88_cell_notebook()
