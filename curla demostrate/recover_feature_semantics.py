import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow")
OUT = ROOT / "curla demostrate"
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "SpaceGap",
    "RelativeVelocity",
    "LateralDistance",
    "time_headway",
    "L_Approach",
    "R_Approach",
    "L_Adjacent",
    "R_Adjacent",
]

EXTENSIONS = {".py", ".ipynb", ".md", ".txt", ".json"}
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", ".ipynb_checkpoints",
    "site-packages", "dist", "build"
}
MAX_FILE_BYTES = 25 * 1024 * 1024
CONTEXT_LINES = 5
MAX_MATCHES_PER_FILE = 100

def extract_units(path, text):
    if path.suffix.lower() != ".ipynb":
        return [("file", text)]

    try:
        notebook = json.loads(text)
    except json.JSONDecodeError:
        return [("file", text)]

    units = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", [])
        if isinstance(source, list):
            source = "".join(source)
        units.append((f"cell_{index}_{cell.get('cell_type', 'unknown')}", source))

    return units

def find_context(units, path):
    matches = []

    for unit_name, content in units:
        lines = content.splitlines()

        for index, line in enumerate(lines):
            found = [
                feature for feature in FEATURES
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(feature)}(?![A-Za-z0-9_])",
                             line, flags=re.IGNORECASE)
            ]

            if not found:
                continue

            start = max(0, index - CONTEXT_LINES)
            end = min(len(lines), index + CONTEXT_LINES + 1)

            matches.append({
                "file": str(path.relative_to(ROOT)),
                "unit": unit_name,
                "line": index + 1,
                "features": found,
                "context": "\n".join(
                    f"{n + 1}: {lines[n]}" for n in range(start, end)
                )
            })

            if len(matches) >= MAX_MATCHES_PER_FILE:
                return matches

    return matches

all_matches = []
scanned = []
skipped = []

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    if any(part.lower() in EXCLUDED_DIRS for part in path.parts):
        continue

    if path.suffix.lower() not in EXTENSIONS:
        continue

    try:
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append((str(path), f"too large: {size} bytes"))
            continue

        text = path.read_text(encoding="utf-8-sig", errors="replace")
        scanned.append(str(path.relative_to(ROOT)))
        units = extract_units(path, text)
        all_matches.extend(find_context(units, path))

    except (OSError, PermissionError) as exc:
        skipped.append((str(path), str(exc)))

# Deduplicate repeated excerpts caused by nearby matches.
unique = []
seen = set()

for item in all_matches:
    key = (
        item["file"],
        item["unit"],
        item["line"],
        item["context"]
    )
    if key not in seen:
        seen.add(key)
        unique.append(item)

json_path = OUT / "i24_feature_semantics_search.json"
txt_path = OUT / "i24_feature_semantics_search.txt"
csv_path = OUT / "i24_feature_semantics_search.csv"

json_path.write_text(
    json.dumps({
        "project_root": str(ROOT),
        "features_searched": FEATURES,
        "files_scanned": len(scanned),
        "matches": unique,
        "skipped_files": [
            {"path": path, "reason": reason}
            for path, reason in skipped
        ],
    }, indent=2),
    encoding="utf-8"
)

with txt_path.open("w", encoding="utf-8") as file:
    file.write("TRAJPILOT CITYFLOW — I-24 FEATURE SEMANTICS SEARCH\n")
    file.write("=" * 72 + "\n")
    file.write(f"Files scanned: {len(scanned)}\n")
    file.write(f"Unique matches: {len(unique)}\n\n")

    for item in unique:
        file.write(
            f"\nFILE: {item['file']}\n"
            f"UNIT: {item['unit']}\n"
            f"LINE: {item['line']}\n"
            f"FEATURES: {', '.join(item['features'])}\n"
            f"{'-' * 60}\n"
            f"{item['context']}\n"
        )

    file.write("\n\nSCANNED FILES\n")
    for path in scanned:
        file.write(path + "\n")

    if skipped:
        file.write("\nSKIPPED FILES\n")
        for path, reason in skipped:
            file.write(f"{path}: {reason}\n")

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["file", "unit", "line", "features", "context"]
    )
    writer.writeheader()

    for item in unique:
        writer.writerow({
            **item,
            "features": "; ".join(item["features"])
        })

feature_counts = {
    feature: sum(
        feature.lower() in [f.lower() for f in item["features"]]
        for item in unique
    )
    for feature in FEATURES
}

print("=" * 72)
print("TRAJPILOT CITYFLOW | FEATURE SEMANTICS FORENSIC SEARCH")
print("=" * 72)
print("Project root:", ROOT)
print("Files scanned:", len(scanned))
print("Unique context matches:", len(unique))
print("\nMatches per feature:")

for feature, count in feature_counts.items():
    print(f"  {feature}: {count}")

print("\nFiles containing matches:")
for path in sorted({item["file"] for item in unique}):
    print(" ", path)

print("\nSearch artifacts:")
print(" ", json_path)
print(" ", txt_path)
print(" ", csv_path)

print("\nSkipped files:", len(skipped))
print("Search complete. No source files were modified.")
