import csv
import math
from pathlib import Path

import carla

ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\curla demostrate")
OUTPUT_CSV = ROOT / "trajectories_aligned.csv"

HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 2002
DURATION_SECONDS = 15
DELTA_SECONDS = 0.1

ROOT.mkdir(parents=True, exist_ok=True)

client = carla.Client(HOST, PORT)
client.set_timeout(20.0)

world = client.get_world()
road_map = world.get_map()
original_settings = world.get_settings()
traffic_manager = client.get_trafficmanager(TM_PORT)

actors = []
tm_sync_enabled = False

try:
    ego_waypoint = road_map.get_waypoint(
        carla.Location(x=-15.15, y=69.73, z=0.0),
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if ego_waypoint is None:
        raise RuntimeError("Could not resolve the selected ego waypoint.")

    if ego_waypoint.is_junction:
        raise RuntimeError("Selected ego waypoint is in a junction.")

    lead_candidates = ego_waypoint.next(20.0)

    lead_waypoint = next(
        (
            waypoint for waypoint in lead_candidates
            if waypoint.road_id == ego_waypoint.road_id
            and waypoint.lane_id == ego_waypoint.lane_id
            and not waypoint.is_junction
        ),
        None
    )

    if lead_waypoint is None:
        raise RuntimeError(
            "Could not resolve a lead waypoint 20 m ahead "
            "on the same non-junction lane."
        )

    ego_transform = ego_waypoint.transform
    lead_transform = lead_waypoint.transform

    ego_transform.location.z += 0.5
    lead_transform.location.z += 0.5

    blueprints = world.get_blueprint_library()
    ego_bp = blueprints.find("vehicle.tesla.model3")
    lead_bp = blueprints.find("vehicle.audi.a2")

    ego_bp.set_attribute("role_name", "trajpilot_ego")
    lead_bp.set_attribute("role_name", "trajpilot_lead")

    lead = world.try_spawn_actor(lead_bp, lead_transform)

    if lead is None:
        raise RuntimeError(
            "Lead spawn failed. No trajectory capture was started."
        )

    actors.append(lead)

    ego = world.try_spawn_actor(ego_bp, ego_transform)

    if ego is None:
        raise RuntimeError(
            "Ego spawn failed. The lead actor will be cleaned up."
        )

    actors.append(ego)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = DELTA_SECONDS
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)
    tm_sync_enabled = True
    traffic_manager.set_random_device_seed(42)
    traffic_manager.set_global_distance_to_leading_vehicle(5.0)
    traffic_manager.auto_lane_change(ego, False)
    traffic_manager.auto_lane_change(lead, False)

    lead.set_autopilot(True, TM_PORT)
    ego.set_autopilot(True, TM_PORT)

    world.tick()

    initial_transform = ego.get_transform()
    initial_lead_transform = lead.get_transform()

    print("=" * 65)
    print("TRAJPILOT CITYFLOW | ALIGNED TRAJECTORY CAPTURE")
    print("=" * 65)
    print("Map:", road_map.name)
    print("Server:", client.get_server_version())
    print("Road ID:", ego_waypoint.road_id)
    print("Lane ID:", ego_waypoint.lane_id)
    print(
        "Initial waypoint gap:",
        round(
            ego_waypoint.transform.location.distance(
                lead_waypoint.transform.location
            ),
            3
        ),
        "m"
    )
    print("Ego actor:", ego.id)
    print("Lead actor:", lead.id)
    print("Traffic Manager Port:", TM_PORT)
    print("Output:", OUTPUT_CSV)

    rows = []
    steps = int(DURATION_SECONDS / DELTA_SECONDS)
    first_frame = world.get_snapshot().frame

    for step in range(steps):
        world.tick()

        snapshot = world.get_snapshot()
        ego_t = ego.get_transform()
        lead_t = lead.get_transform()
        ego_v = ego.get_velocity()
        lead_v = lead.get_velocity()

        yaw = math.radians(ego_t.rotation.yaw)
        forward_x = math.cos(yaw)
        forward_y = math.sin(yaw)
        right_x = -forward_y
        right_y = forward_x

        dx = lead_t.location.x - ego_t.location.x
        dy = lead_t.location.y - ego_t.location.y
        dz = lead_t.location.z - ego_t.location.z

        longitudinal_gap = dx * forward_x + dy * forward_y
        lateral_offset = dx * right_x + dy * right_y
        euclidean_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        relative_longitudinal_velocity = (
            (lead_v.x - ego_v.x) * forward_x
            + (lead_v.y - ego_v.y) * forward_y
        )

        ego_speed = math.sqrt(
            ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2
        )
        lead_speed = math.sqrt(
            lead_v.x ** 2 + lead_v.y ** 2 + lead_v.z ** 2
        )

        rows.append({
            "step": step,
            "frame": snapshot.frame,
            "sim_time_s": snapshot.timestamp.elapsed_seconds,
            "elapsed_from_start_s": (snapshot.frame - first_frame)
            * DELTA_SECONDS,
            "road_id": ego_waypoint.road_id,
            "lane_id": ego_waypoint.lane_id,
            "ego_x": ego_t.location.x,
            "ego_y": ego_t.location.y,
            "ego_z": ego_t.location.z,
            "ego_yaw_deg": ego_t.rotation.yaw,
            "ego_vx_mps": ego_v.x,
            "ego_vy_mps": ego_v.y,
            "ego_speed_mps": ego_speed,
            "lead_x": lead_t.location.x,
            "lead_y": lead_t.location.y,
            "lead_z": lead_t.location.z,
            "lead_yaw_deg": lead_t.rotation.yaw,
            "lead_vx_mps": lead_v.x,
            "lead_vy_mps": lead_v.y,
            "lead_speed_mps": lead_speed,
            "longitudinal_gap_m": longitudinal_gap,
            "lateral_offset_m": lateral_offset,
            "euclidean_distance_m": euclidean_distance,
            "relative_longitudinal_velocity_mps":
                relative_longitudinal_velocity,
        })

        if step % 25 == 0:
            print(
                f"Step {step:3d}/{steps} | "
                f"ego={ego_speed:.2f} m/s | "
                f"lead={lead_speed:.2f} m/s | "
                f"longitudinal gap={longitudinal_gap:.2f} m | "
                f"lateral offset={lateral_offset:.2f} m"
            )

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    gaps = [row["longitudinal_gap_m"] for row in rows]
    lateral = [abs(row["lateral_offset_m"]) for row in rows]

    print("\nCAPTURE SUMMARY")
    print("Rows:", len(rows))
    print("Expected rows:", steps)
    print("Minimum longitudinal gap:", round(min(gaps), 3), "m")
    print("Maximum longitudinal gap:", round(max(gaps), 3), "m")
    print("Maximum absolute lateral offset:", round(max(lateral), 3), "m")
    print("CSV saved:", OUTPUT_CSV)

finally:
    for actor in reversed(actors):
        try:
            actor.set_autopilot(False, TM_PORT)
        except Exception:
            pass

        try:
            actor.destroy()
        except Exception:
            pass

    try:
        if tm_sync_enabled:
            traffic_manager.set_synchronous_mode(False)
    except Exception:
        pass

    try:
        world.apply_settings(original_settings)
    except Exception as exc:
        print("Warning: world settings restoration failed:", exc)

    print("Cleanup finished.")
