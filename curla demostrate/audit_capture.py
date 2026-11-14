import csv
import math
from pathlib import Path

ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow")
DEMO_DIR = ROOT / "curla demostrate"
CSV_PATH = DEMO_DIR / "trajectories_raw.csv"
ADAPTER_PATH = ROOT / "carla_eval" / "carla_adapter.py"

rows = []
columns = []

if CSV_PATH.exists():
    with CSV_PATH.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

print("=" * 65)
print("TRAJPILOT CITYFLOW | AUDIT CAPTURE & ADAPTER")
print("=" * 65)

print("\n[1] CSV STRUCTURE")
print("Path:", CSV_PATH)
print("Rows:", len(rows))
print("Columns:", len(columns))
print("Column names:", columns)

if rows:
    print("\n[2] MISSING VALUES")
    for column in columns:
        missing = sum(
            row.get(column) in (None, "")
            for row in rows
        )
        print(f"{column}: {missing}/{len(rows)} missing")

    print("\n[3] NUMERIC RANGES")
    for column in columns:
        values = []
        for row in rows:
            try:
                value = float(row[column])
                if math.isfinite(value):
                    values.append(value)
            except (ValueError, TypeError, KeyError):
                pass
        if values:
            print(
                f"{column}: min={min(values):.4f}, "
                f"max={max(values):.4f}, "
                f"first={values[0]:.4f}, "
                f"last={values[-1]:.4f}"
            )

    print("\n[4] FIRST THREE ROWS")
    for row in rows[:3]:
        print(row)

    print("\n[5] LAST THREE ROWS")
    for row in rows[-3:]:
        print(row)

print("\n[6] EXISTING CARLA ADAPTER")
if ADAPTER_PATH.exists():
    print("Path:", ADAPTER_PATH)
    with ADAPTER_PATH.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if line_number > 180:
                print("... output limited to first 180 lines ...")
                break
            print(f"{line_number:03d}: {line.rstrip()}")
else:
    print("Adapter not found at:", ADAPTER_PATH)
    print("Search for carla_adapter.py under:", ROOT)

print("\nAudit complete. No files were modified.")
