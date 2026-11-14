import json

with open("scripts/cell_specs.json", "r", encoding="utf-8") as f:
    specs = json.load(f)

for s in specs:
    num = s["cell_num"]
    title = s["title"]
    raw = s["raw"]
    py = len(s["py_blocks"])
    md = len(s["md_blocks"])
    txt = len(s["txt_blocks"])
    first_few = raw.splitlines()[0] if raw.splitlines() else ""
    print(f"Cell {num:02d}: {title[:40]:40s} | py:{py} md:{md} txt:{txt} | {len(raw)} chars")
