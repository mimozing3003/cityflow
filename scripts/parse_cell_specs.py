import re
import json

with open("full_user_prompt.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"# (CELL \d{3} [^\n]+)"
matches = list(re.finditer(pattern, text))

cells = []
for i in range(len(matches)):
    start = matches[i].start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    cell_raw = text[start:end].strip()
    title = matches[i].group(1)

    # Extract markdown block if present
    md_blocks = re.findall(r"```markdown\s*(.*?)\s*```", cell_raw, re.DOTALL)
    py_blocks = re.findall(r"```python\s*(.*?)\s*```", cell_raw, re.DOTALL)
    txt_blocks = re.findall(r"```text\s*(.*?)\s*```", cell_raw, re.DOTALL)

    cells.append({
        "cell_num": i + 1,
        "title": title,
        "raw": cell_raw,
        "md_blocks": md_blocks,
        "py_blocks": py_blocks,
        "txt_blocks": txt_blocks,
    })

with open("scripts/cell_specs.json", "w", encoding="utf-8") as f:
    json.dump(cells, f, indent=2, ensure_ascii=False)

print(f"Saved {len(cells)} cell specifications to scripts/cell_specs.json")
