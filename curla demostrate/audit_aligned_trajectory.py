import csv
import math
from pathlib import Path

import carla

ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\curla demostrate")
CSV_PATH = ROOT / "trajectories_aligned.csv"

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)
world_map = client.get_world().get_map()

with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise RuntimeError("Trajectory CSV is empty.")

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

    ego_yaw = float(row["ego_yaw_deg"])
    lead_yaw = float(row["lead_yaw_deg"])

    ego_wp = world_map.get_waypoint(
        ego_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )
    lead_wp = world_map.get_waypoint(
        lead_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    dx = lead_loc.x - ego_loc.x
    dy = lead_loc.y - ego_loc.y
    dz = lead_loc.z - ego_loc.z

    yaw = math.radians(ego_yaw)
    longitudinal = dx * math.cos(yaw) + dy * math.sin(yaw)
    lateral = -dx * math.sin(yaw) + dy * math.cos(yaw)
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw_diff = abs((lead_yaw - ego_yaw + 180.0) % 360.0 - 180.0)

    audited.append({
        "step": int(row["step"]),
        "time": float(row["elapsed_from_start_s"]),
        "ego_road": ego_wp.road_id if ego_wp else None,
        "ego_lane": ego_wp.lane_id if ego_wp else None,
        "lead_road": lead_wp.road_id if lead_wp else None,
        "lead_lane": lead_wp.lane_id if lead_wp else None,
        "ego_junction": ego_wp.is_junction if ego_wp else None,
        "lead_junction": lead_wp.is_junction if lead_wp else None,
        "longitudinal": longitudinal,
        "lateral": lateral,
        "distance": distance,
        "yaw_diff": yaw_diff,
        "ego_speed": float(row["ego_speed_mps"]),
        "lead_speed": float(row["lead_speed_mps"])
    })

same_lane = [
    item for item in audited
    if item["ego_road"] == item["lead_road"]
    and item["ego_lane"] == item["lead_lane"]
]
different_lane = [
    item for item in audited
    if item["ego_road"] != item["lead_road"]
    or item["ego_lane"] != item["lead_lane"]
]
max_lateral = max(audited, key=lambda item: abs(item["lateral"]))
min_longitudinal = min(audited, key=lambda item: item["longitudinal"])

print("=" * 70)
print("TRAJPILOT CITYFLOW | ALIGNED TRAJECTORY AUDIT")
print("=" * 70)
print("Rows:", len(audited))
print("Same mapped road and lane:", len(same_lane), "/", len(audited))
print("Different mapped road or lane:", len(different_lane), "/", len(audited))
print(
    "Same-lane fraction:",
    f"{100 * len(same_lane) / len(audited):.1f}%"
)
print(
    "Maximum absolute lateral offset:",
    f"{abs(max_lateral['lateral']):.3f} m at step {max_lateral['step']}"
)
print(
    "Minimum projected longitudinal gap:",
    f"{min_longitudinal['longitudinal']:.3f} m "
    f"at step {min_longitudinal['step']}"
)

print("\nSelected samples:")
selected_steps = {0, 25, 50, 75, 100, 125, 149, max_lateral["step"]}
for item in audited:
    if item["step"] in selected_steps:
        print(
            f"step={item['step']:3d} t={item['time']:5.1f}s | "
            f"ego road/lane={item['ego_road']}/{item['ego_lane']} | "
            f"lead road/lane={item['lead_road']}/{item['lead_lane']} | "
            f"gap={item['longitudinal']:7.2f}m | "
            f"lateral={item['lateral']:7.2f}m | "
            f"distance={item['distance']:7.2f}m | "
            f"heading_diff={item['yaw_diff']:5.1f}deg | "
            f"junction={item['ego_junction']}/{item['lead_junction']}"
        )

print("\nAudit complete. No simulation settings or actors were changed.")
