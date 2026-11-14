import nbformat

with open("TrajPilot_Research_Master.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

with open("scripts/cell204_source.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[204].source)

print("Saved cell 204 to scripts/cell204_source.py")
