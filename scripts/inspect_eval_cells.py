import nbformat

with open("TrajPilot_Research_Master.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

for idx in [203, 204, 205, 206]:
    if idx < len(nb.cells):
        print(f"=== Cell {idx} ({nb.cells[idx].cell_type}) ===")
        src = nb.cells[idx].source
        print(src[:600])
        print("...")
