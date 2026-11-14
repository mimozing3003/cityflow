import re
import json

with open("full_user_prompt.txt", "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"# (CELL \d{3} [^\n]+)"
matches = list(re.finditer(pattern, text))

cells_info = []
for i in range(len(matches)):
    start = matches[i].start()
    end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    cell_raw = text[start:end].strip()
    title = matches[i].group(1)

    is_markdown = (
        "MARKDOWN" in title
        or "### Markdown cell" in cell_raw[:120]
        or "```markdown" in cell_raw[:120]
    )

    cells_info.append({
        "index": i + 1,
        "title": title,
        "is_markdown": is_markdown,
        "raw": cell_raw,
    })

print(f"Total cells: {len(cells_info)}")
for c in cells_info:
    cell_type = "MD  " if c["is_markdown"] else "CODE"
    print(f"{c['index']:02d} | {cell_type} | {c['title']}")
