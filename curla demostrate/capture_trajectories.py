import carla
import csv
import math
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\curla demostrate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "trajectories_raw.csv"

DELTA_SECONDS = 0.1
DURATION_SECONDS = 15.0

print("=" * 65)
print("TRAJPILOT CITYFLOW | INITIALIZING TRAJECTORY CAPTURE")
print("=" * 65)

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
original_settings = world.get_settings()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = DELTA_SECONDS
world.apply_settings(settings)

actors = []

try:
    traffic_manager = client.get_trafficmanager(2002)
    traffic_manager.set_synchronous_mode(True)

    blueprints = world.get_blueprint_library()
    ego_bp = blueprints.find("vehicle.tesla.model3")
    if ego_bp is None:
        ego_bp = blueprints.filter("vehicle.*")[0]
    lead_bp = blueprints.find("vehicle.audi.a2")
    if lead_bp is None:
        lead_bp = blueprints.filter("vehicle.*")[1]

    ego_bp.set_attribute("role_name", "ego")
    lead_bp.set_attribute("role_name", "lead")

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()

    candidate = None
    min_dist = 12.0
    max_dist = 28.0

    for sp in spawn_points:
        wp_ego = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_ego is None:
            continue
        # Look ahead on the same lane for a lead spawn point
        next_wps = wp_ego.next(18.0)
        if not next_wps:
            continue
        wp_lead = next_wps[0]
        if wp_lead.lane_id == wp_ego.lane_id and wp_lead.road_id == wp_ego.road_id:
            candidate = (sp, wp_lead.transform)
            break

    if candidate is None:
        # Fallback to searching pair of spawn points with same orientation and reasonable separation
        for i, sp1 in enumerate(spawn_points):
            wp1 = carla_map.get_waypoint(sp1.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp1 is None:
                continue
            for j, sp2 in enumerate(spawn_points):
                if i == j:
                    continue
                dist = sp1.location.distance(sp2.location)
                if min_dist <= dist <= max_dist:
                    wp2 = carla_map.get_waypoint(sp2.location, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if wp2 and wp1.road_id == wp2.road_id and wp1.lane_id == wp2.lane_id:
                        candidate = (sp1, sp2)
                        break
            if candidate:
                break

    if candidate is None:
        raise RuntimeError(
            "Could not find two waypoints on the same driving lane. "
            "No vehicles were intentionally spawned."
        )

    ego_transform, lead_transform = candidate
    ego_transform.location.z += 0.2
    lead_transform.location.z += 0.2

    lead = world.try_spawn_actor(lead_bp, lead_transform)
    if lead is None:
        raise RuntimeError(
            "Could not spawn the lead vehicle at the selected location."
        )
    actors.append(lead)

    ego = world.try_spawn_actor(ego_bp, ego_transform)
    if ego is None:
        raise RuntimeError(
            "Could not spawn the ego vehicle. Try another lane or spawn point."
        )
    actors.append(ego)

    world.tick()

    print("=" * 65)
    print("TRAJPILOT CITYFLOW | CONTROLLED TRAJECTORY CAPTURE")
    print("=" * 65)
    print("Map:", world.get_map().name)
    print("Server:", client.get_server_version())
    print("Fixed time step:", DELTA_SECONDS, "seconds")
    print("Ego actor ID:", ego.id)
    print("Lead actor ID:", lead.id)
    print("Output:", OUTPUT_CSV)

    start_frame = world.get_snapshot().frame
    rows = []
    steps = int(DURATION_SECONDS / DELTA_SECONDS)

    for step in range(steps):
        lead.apply_control(
            carla.VehicleControl(throttle=0.22, steer=0.0)
        )
        ego.apply_control(
            carla.VehicleControl(throttle=0.18, steer=0.0)
        )
        world.tick()

        snapshot = world.get_snapshot()
        sim_time = snapshot.timestamp.elapsed_seconds
        ego_transform = ego.get_transform()
        lead_transform = lead.get_transform()
        ego_velocity = ego.get_velocity()
        lead_velocity = lead.get_velocity()

        dx = lead_transform.location.x - ego_transform.location.x
        dy = lead_transform.location.y - ego_transform.location.y
        dz = lead_transform.location.z - ego_transform.location.z
        separation = math.sqrt(dx * dx + dy * dy + dz * dz)

        ego_speed = math.sqrt(
            ego_velocity.x ** 2 + ego_velocity.y ** 2 + ego_velocity.z ** 2
        )
        lead_speed = math.sqrt(
            lead_velocity.x ** 2 + lead_velocity.y ** 2 + lead_velocity.z ** 2
        )
        relative_velocity = lead_speed - ego_speed

        rows.append({
            "step": step,
            "sim_time": round(sim_time, 4),
            "ego_x": round(ego_transform.location.x, 4),
            "ego_y": round(ego_transform.location.y, 4),
            "ego_z": round(ego_transform.location.z, 4),
            "ego_yaw": round(ego_transform.rotation.yaw, 4),
            "ego_vx": round(ego_velocity.x, 4),
            "ego_vy": round(ego_velocity.y, 4),
            "ego_speed": round(ego_speed, 4),
            "lead_x": round(lead_transform.location.x, 4),
            "lead_y": round(lead_transform.location.y, 4),
            "lead_z": round(lead_transform.location.z, 4),
            "lead_yaw": round(lead_transform.rotation.yaw, 4),
            "lead_vx": round(lead_velocity.x, 4),
            "lead_vy": round(lead_velocity.y, 4),
            "lead_speed": round(lead_speed, 4),
            "separation": round(separation, 4),
            "relative_velocity": round(relative_velocity, 4)
        })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Captured {len(rows)} steps ({DURATION_SECONDS}s at 10 Hz)")
    print(f"Initial separation : {rows[0]['separation']:.2f} m")
    print(f"Final separation   : {rows[-1]['separation']:.2f} m")
    print(f"Final Ego Speed    : {rows[-1]['ego_speed']:.2f} m/s ({rows[-1]['ego_speed']*3.6:.1f} km/h)")
    print(f"Final Lead Speed   : {rows[-1]['lead_speed']:.2f} m/s ({rows[-1]['lead_speed']*3.6:.1f} km/h)")
    print(f"Successfully saved to: {OUTPUT_CSV}")
    print("=" * 65)

finally:
    print("\nCleaning up spawned actors and restoring settings...")
    for actor in actors:
        if actor and actor.is_alive:
            actor.destroy()
    world.apply_settings(original_settings)
    print("Cleanup completed.")
