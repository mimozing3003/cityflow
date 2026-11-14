import nbformat

with open("TrajPilot_Research_Master.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

with open("scripts/cell267_source.py", "w", encoding="utf-8") as f:
    f.write(nb.cells[267].source)

print("Saved cell 267 source to scripts/cell267_source.py")
