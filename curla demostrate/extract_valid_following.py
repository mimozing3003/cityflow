import csv
import math
from pathlib import Path

import carla

ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\curla demostrate")
SOURCE = ROOT / "trajectories_aligned.csv"
OUTPUT = ROOT / "trajectories_valid_following.csv"

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)
road_map = client.get_world().get_map()

with SOURCE.open("r", newline="", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)
    fieldnames = reader.fieldnames
    rows = list(reader)

if not rows:
    raise RuntimeError("Source trajectory CSV is empty.")

audited = []

for row in rows:
    ego_loc = carla.Location(
        x=float(row["ego_x"]),
        y=float(row["ego_y"]),
        z=float(row["ego_z"])
    )
    lead_loc = carla.Location(
        x=float(row["lead_x"]),
        y=float(row["lead_y"]),
        z=float(row["lead_z"])
    )

    ego_wp = road_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    lead_wp = road_map.get_waypoint(
        lead_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    ego_yaw = float(row["ego_yaw_deg"])
    lead_yaw = float(row["lead_yaw_deg"])
    yaw_diff = abs((lead_yaw - ego_yaw + 180.0) % 360.0 - 180.0)

    dx = lead_loc.x - ego_loc.x
    dy = lead_loc.y - ego_loc.y
    dz = lead_loc.z - ego_loc.z

    yaw = math.radians(ego_yaw)
    longitudinal_gap = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral_offset = -dx * math.sin(yaw) + dy * math.cos(yaw)

    same_lane = (
        ego_wp is not None
        and lead_wp is not None
        and ego_wp.road_id == lead_wp.road_id
        and ego_wp.lane_id == lead_wp.lane_id
    )

    outside_junction = (
        ego_wp is not None
        and lead_wp is not None
        and not ego_wp.is_junction
        and not lead_wp.is_junction
    )

    valid = (
        same_lane
        and outside_junction
        and yaw_diff <= 5.0
        and abs(lateral_offset) <= 2.0
        and 0.0 < longitudinal_gap <= 60.0
    )

    audited.append({
        "row": row,
        "valid": valid,
        "yaw_diff": yaw_diff,
        "longitudinal_gap": longitudinal_gap,
        "lateral_offset": lateral_offset,
        "ego_road": ego_wp.road_id if ego_wp else None,
        "ego_lane": ego_wp.lane_id if ego_wp else None,
        "lead_road": lead_wp.road_id if lead_wp else None,
        "lead_lane": lead_wp.lane_id if lead_wp else None,
    })

segments = []
start = None

for i, item in enumerate(audited):
    if item["valid"]:
        if start is None:
            start = i
    elif start is not None:
        segments.append((start, i - 1))
        start = None

if start is not None:
    segments.append((start, len(audited) - 1))

if not segments:
    raise RuntimeError("No valid continuous same-lane segment found.")

start_idx, end_idx = max(
    segments,
    key=lambda pair: pair[1] - pair[0] + 1
)

segment = audited[start_idx:end_idx + 1]
valid_rows = [item["row"] for item in segment]

with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(valid_rows)

gaps = [item["longitudinal_gap"] for item in segment]
lateral = [abs(item["lateral_offset"]) for item in segment]
heading = [item["yaw_diff"] for item in segment]

print("=" * 68)
print("TRAJPILOT CITYFLOW | VALID FOLLOWING SEGMENT")
print("=" * 68)
print("Source rows:", len(rows))
print("Rows passing all filters:", sum(item["valid"] for item in audited))
print("Contiguous valid segments:", len(segments))
print("Selected source row indices:", start_idx, "to", end_idx)
print("Selected source step range:",
      valid_rows[0]["step"], "to", valid_rows[-1]["step"])
print("Selected sample count:", len(valid_rows))
print("Duration at 10 Hz:", round(len(valid_rows) * 0.1, 2), "seconds")
print("Same road/lane:",
      segment[0]["ego_road"], segment[0]["ego_lane"])
print("Longitudinal gap range:",
      round(min(gaps), 3), "to", round(max(gaps), 3), "m")
print("Maximum absolute lateral offset:", round(max(lateral), 3), "m")
print("Maximum heading difference:", round(max(heading), 3), "degrees")
print("Saved:", OUTPUT)
print("\nSource file was not modified.")
