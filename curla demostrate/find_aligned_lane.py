import math
import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
road_map = world.get_map()
spawn_points = road_map.get_spawn_points()

def angle_difference(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)

candidates = []

for index, spawn in enumerate(spawn_points):
    start = road_map.get_waypoint(
        spawn.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if start is None or start.is_junction:
        continue

    for distance in (15.0, 20.0, 25.0):
        for lead in start.next(distance):
            if lead.is_junction:
                continue

            if (
                lead.road_id != start.road_id
                or lead.lane_id != start.lane_id
            ):
                continue

            heading_error = angle_difference(
                start.transform.rotation.yaw,
                lead.transform.rotation.yaw
            )

            actual_distance = start.transform.location.distance(
                lead.transform.location
            )

            if heading_error > 5.0:
                continue

            if not 14.0 <= actual_distance <= 26.0:
                continue

            candidates.append({
                "spawn_index": index,
                "road_id": start.road_id,
                "lane_id": start.lane_id,
                "distance": actual_distance,
                "heading_error": heading_error,
                "ego": start.transform,
                "lead": lead.transform,
                "ego_junction": start.is_junction,
                "lead_junction": lead.is_junction
            })

candidates.sort(
    key=lambda item: (
        item["heading_error"],
        abs(item["distance"] - 20.0)
    )
)

print("=" * 65)
print("TRAJPILOT CITYFLOW | ALIGNED LANE DIAGNOSTIC")
print("=" * 65)
print("Map:", road_map.name)
print("Spawn points examined:", len(spawn_points))
print("Valid candidate pairs:", len(candidates))

for rank, item in enumerate(candidates[:10], start=1):
    ego = item["ego"]
    lead = item["lead"]

    print(f"\nCandidate {rank}")
    print(
        f"  Spawn index={item['spawn_index']}, "
        f"road={item['road_id']}, lane={item['lane_id']}"
    )
    print(
        f"  Distance={item['distance']:.2f} m, "
        f"heading difference={item['heading_error']:.2f} degrees"
    )
    print(
        f"  Ego:  x={ego.location.x:.2f}, "
        f"y={ego.location.y:.2f}, "
        f"z={ego.location.z:.2f}, "
        f"yaw={ego.rotation.yaw:.2f}"
    )
    print(
        f"  Lead: x={lead.location.x:.2f}, "
        f"y={lead.location.y:.2f}, "
        f"z={lead.location.z:.2f}, "
        f"yaw={lead.rotation.yaw:.2f}"
    )

if not candidates:
    print("\nNo suitable pair found. No actors were spawned.")

print("\nDiagnostic complete. World settings were not changed.")
