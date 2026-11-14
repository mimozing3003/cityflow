import json

with open("scripts/cell_specs.json", "r", encoding="utf-8") as f:
    specs = json.load(f)

c74 = specs[73]
print("Cell 74 raw lines:")
for line in c74["raw"].splitlines()[:50]:
    print(line)
