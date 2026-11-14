import sys
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from scripts.populate_metadata_p1_p4 import STEPS_P1_P4
from scripts.populate_metadata_p5_p8 import STEPS_P5_P8

ALL_STEPS = {}
ALL_STEPS.update(STEPS_P1_P4)
ALL_STEPS.update(STEPS_P5_P8)

print(f"Total steps assembled: {len(ALL_STEPS)} (Target: 85)")
assert len(ALL_STEPS) == 85, f"Expected 85 steps, got {len(ALL_STEPS)}"

# Verify all steps 1 to 85 exist
missing = [i for i in range(1, 86) if i not in ALL_STEPS]
if missing:
    print(f"Missing steps: {missing}")
else:
    print("All 85 steps are 100% present!")

# Save to step_explanations.py
with open("scripts/step_explanations.py", "w", encoding="utf-8") as f:
    f.write('"""\nComprehensive 85-Step Metadata & Explanations Dictionary\n"""\n\n')
    f.write("STEP_METADATA = " + repr(ALL_STEPS) + "\n")

print("Successfully written scripts/step_explanations.py!")
