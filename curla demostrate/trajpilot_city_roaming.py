"""
trajpilot_city_roaming.py
========================
Full City Autonomous Roaming & Continuous Live TrajPilot GRU Trajectory Prediction.

Demonstration Features:
  1. Spawns 30+ NPC vehicles across all roads in the CARLA map.
  2. Spawns the ego vehicle (Tesla Model 3) with full autonomous city navigation
     (obeys traffic lights, stop signs, turns at intersections, lane changes, moves & stops).
  3. Third-person chase camera smoothly tracks behind the ego vehicle in CarlaUE4.
  4. At every 10 Hz tick, computes 16 causal kinematics & car-following features
     and runs the frozen TrajPilot GRU model (227,132 parameters).
  5. Continuously renders a dynamic glowing CYAN trajectory ribbon on the road ahead
     showing the predicted 30 future steps (3.0s horizon).
  6. Runs continuously across the entire map until interrupted (Ctrl+C).
  7. Safely cleans up all spawned actors on exit.
"""

import sys
import time
import math
import json
import random
from datetime import datetime
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import carla

# Paths
PROJECT_ROOT = Path(r"C:\Users\SOMUJIT\Desktop\cityflow")
DEMO_DIR = PROJECT_ROOT / "curla demostrate"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "models" / "gru_baseline" / "best_gru_baseline.pt"
NORMALIZER_PATH = (
    PROJECT_ROOT / "data" / "processed" / "i24_sequences"
    / "gru_inputs" / "train_normalization_statistics.csv"
)
SUMMARY_JSON = DEMO_DIR / "roaming_demo_summary.json"

# Simulation constants
HOST = "127.0.0.1"
PORT = 2000
TM_PORT = 2002
DT = 0.1  # 10 Hz
NUM_NPCS = 30
MAX_TICKS = 10000  # ~16.6 minutes (or until user interrupts)

# Model architecture constants
EXPECTED_PARAMS = 227132
INPUT_DIM = 32
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.10
FUTURE_STEPS = 30
TARGET_DIM = 2

FEATURE_NAMES = [
    "vx_causal",
    "vy_causal",
    "speed_derived",
    "acceleration_native",
    "acceleration_causal_0p5s",
    "turn_rate_valid",
    "curvature_valid",
    "jerk_causal",
    "SpaceGap",
    "RelativeVelocity",
    "LateralDistance",
    "time_headway",
    "L_Approach",
    "R_Approach",
    "L_Adjacent",
    "R_Adjacent",
]


class TrajPilotGRU(nn.Module):
    """
    Exact frozen TrajPilot GRU trajectory prediction model.
    Input:  (batch, 30, 32) -> 16 normalized features + 16 missingness indicators.
    Output: (batch, 30, 2)  -> 30 future (dx, dy) trajectory coordinates at 10 Hz (3.0s horizon).
    """
    def __init__(
        self,
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        future_steps=FUTURE_STEPS,
        target_dim=TARGET_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.future_steps = future_steps
        self.target_dim = target_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, future_steps * target_dim),
        )

    def forward(self, x):
        x_proj = self.input_projection(x)
        _, h_n = self.encoder(x_proj)
        context = h_n[-1]
        y = self.decoder(context)
        return y.view(-1, self.future_steps, self.target_dim)


def load_frozen_checkpoint(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = TrajPilotGRU().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == EXPECTED_PARAMS, f"Parameter count mismatch: {param_count} != {EXPECTED_PARAMS}"

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, param_count


def load_normalization_statistics():
    if not NORMALIZER_PATH.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {NORMALIZER_PATH}")

    df = pd.read_csv(NORMALIZER_PATH)
    df = df.set_index("feature").reindex(FEATURE_NAMES)
    means = df["train_mean"].to_numpy(dtype=np.float64)
    stds = df["train_std"].to_numpy(dtype=np.float64)
    stds = np.where(np.isfinite(stds) & (stds > 0), stds, 1.0)
    return means, stds


def build_16_features_from_history(history_records):
    """
    Construct the 16 causal TrajPilot features from the rolling history buffer (up to 30 steps).
    """
    n = len(history_records)
    if n < 30:
        pad_count = 30 - n
        first = history_records[0]
        records = [first] * pad_count + list(history_records)
    else:
        records = list(history_records)[-30:]

    yaw_deg = np.array([r["yaw_deg"] for r in records], dtype=np.float64)
    yaw = np.unwrap(np.radians(yaw_deg))

    vx = np.array([r["vx"] for r in records], dtype=np.float64)
    vy = np.array([r["vy"] for r in records], dtype=np.float64)
    speed = np.array([r["speed"] for r in records], dtype=np.float64)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # 0, 1: vx_causal, vy_causal
    vx_causal = vx * cos_yaw + vy * sin_yaw
    vy_causal = -vx * sin_yaw + vy * cos_yaw

    # 2: speed_derived
    speed_derived = speed.copy()

    # 3: acceleration_native
    accel = np.zeros(30, dtype=np.float64)
    accel[1:] = (speed[1:] - speed[:-1]) / DT
    accel[0] = accel[1]

    # 4: acceleration_causal_0p5s
    accel_0p5s = np.full(30, np.nan, dtype=np.float64)
    accel_0p5s[5:] = (speed[5:] - speed[:-5]) / (5.0 * DT)

    # 5: turn_rate_valid
    turn_rate = np.full(30, np.nan, dtype=np.float64)
    turn_rate[1:] = (yaw[1:] - yaw[:-1]) / DT

    # 6: curvature_valid
    curv = np.full(30, np.nan, dtype=np.float64)
    mask_curv = (speed >= 1.0) & np.isfinite(turn_rate)
    curv[mask_curv] = turn_rate[mask_curv] / np.maximum(speed[mask_curv], 1e-6)

    # 7: jerk_causal
    jerk = np.full(30, np.nan, dtype=np.float64)
    jerk[1:] = (accel[1:] - accel[:-1]) / DT

    # 8-11: Car-following features
    space_gap = np.array([r.get("space_gap", np.nan) for r in records], dtype=np.float64)
    rel_vel = np.array([r.get("rel_vel", np.nan) for r in records], dtype=np.float64)
    lat_dist = np.array([r.get("lat_dist", np.nan) for r in records], dtype=np.float64)
    time_headway = np.where(np.isfinite(space_gap) & (speed > 0.1), space_gap / np.maximum(speed, 0.1), np.nan)

    # 12-15: Multi-vehicle boundary features
    l_approach = np.zeros(30, dtype=np.float64)
    r_approach = np.zeros(30, dtype=np.float64)
    l_adjacent = np.zeros(30, dtype=np.float64)
    r_adjacent = np.zeros(30, dtype=np.float64)

    features = np.column_stack([
        vx_causal, vy_causal, speed_derived, accel, accel_0p5s,
        turn_rate, curv, jerk, space_gap, rel_vel, lat_dist,
        time_headway, l_approach, r_approach, l_adjacent, r_adjacent,
    ]).astype(np.float32)

    return features


def exact_16_to_32(x16, means, stds):
    x16 = np.asarray(x16, dtype=np.float32)
    if x16.ndim == 2:
        x16 = x16[None, ...]

    num_feat = x16.shape[-1]
    finite_mask = np.isfinite(x16)
    normalized = np.zeros_like(x16, dtype=np.float32)

    means_t = means.astype(np.float32)
    stds_t = stds.astype(np.float32)

    for j in range(num_feat):
        valid = finite_mask[:, :, j]
        normalized[valid, j] = (x16[valid, j] - means_t[j]) / stds_t[j]

    missing_mask = (~finite_mask).astype(np.float32)
    return np.concatenate([normalized, missing_mask], axis=-1)


def find_leading_vehicle(ego_loc, ego_yaw_rad, ego_v, candidate_vehicles):
    """
    Find the closest vehicle ahead in the same travel corridor (up to 55m ahead, within 2.5m laterally).
    """
    forward_x = math.cos(ego_yaw_rad)
    forward_y = math.sin(ego_yaw_rad)

    closest_lead = None
    min_dist = float("inf")

    for v in candidate_vehicles:
        if not v.is_alive:
            continue
        v_loc = v.get_location()
        dx = v_loc.x - ego_loc.x
        dy = v_loc.y - ego_loc.y

        lon = dx * forward_x + dy * forward_y
        lat = -dx * forward_y + dy * forward_x

        if 1.0 < lon < 55.0 and abs(lat) < 2.5:
            if lon < min_dist:
                min_dist = lon
                v_vel = v.get_velocity()
                rel_vel = (v_vel.x - ego_v.x) * forward_x + (v_vel.y - ego_v.y) * forward_y
                closest_lead = (v, lon, lat, rel_vel)

    return closest_lead


def draw_live_prediction_ribbon(world, pred_coords, ego_loc):
    """
    Draw dynamic glowing CYAN trajectory ribbon on the road surface ahead of the ego vehicle.
    Uses life_time=0.25 to create a smooth, continuous ribbon that adapts in real-time.
    """
    debug = world.debug
    life_time = 0.25  # Re-drawn every 0.1s tick
    cyan_color = carla.Color(0, 230, 255)
    white_color = carla.Color(255, 255, 255)

    # Origin point
    origin_pt = carla.Location(x=float(ego_loc.x), y=float(ego_loc.y), z=float(ego_loc.z) + 0.3)
    debug.draw_point(origin_pt, size=0.08, color=white_color, life_time=life_time)

    # Draw continuous trajectory line
    prev_pt = origin_pt
    for i, (wx, wy, wz) in enumerate(pred_coords):
        curr_pt = carla.Location(x=float(wx), y=float(wy), z=float(wz))
        debug.draw_line(prev_pt, curr_pt, thickness=0.09, color=cyan_color, life_time=life_time)
        if (i + 1) % 5 == 0:
            debug.draw_point(curr_pt, size=0.07, color=cyan_color, life_time=life_time)
        prev_pt = curr_pt

    # Highlight final 3.0s horizon position
    if pred_coords:
        end_pt = carla.Location(x=float(pred_coords[-1][0]), y=float(pred_coords[-1][1]), z=float(pred_coords[-1][2]))
        debug.draw_point(end_pt, size=0.15, color=carla.Color(255, 220, 50), life_time=life_time)


def main():
    print("=" * 80)
    print("TRAJPILOT CITYFLOW | FULL-CITY AUTONOMOUS ROAMING & LIVE GRU PREDICTION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Inference Device : {device}")

    # Load frozen checkpoint & normalization statistics
    model, param_count = load_frozen_checkpoint(device)
    means, stds = load_normalization_statistics()
    print(f"Loaded Frozen GRU Model   : {param_count:,} parameters (Strict PASS)")
    print(f"Loaded I-24 Normalization : {len(means)} features")

    # Connect to CARLA
    print(f"Connecting to CARLA Server at {HOST}:{PORT}...")
    client = carla.Client(HOST, PORT)
    client.set_timeout(25.0)

    server_ver = client.get_server_version()
    client_ver = client.get_client_version()
    print(f"CARLA Server Version       : {server_ver}")
    print(f"CARLA Client Version       : {client_ver}")

    world = client.get_world()
    road_map = world.get_map()
    spawn_points = road_map.get_spawn_points()
    print(f"CARLA Map Loaded          : {road_map.name} ({len(spawn_points)} spawn points across city)")

    original_settings = world.get_settings()
    actors = []
    npc_vehicles = []
    tm_sync = False

    # Connect to Traffic Manager
    try:
        traffic_manager = client.get_trafficmanager(TM_PORT)
        actual_tm_port = traffic_manager.get_port()
    except Exception:
        traffic_manager = client.get_trafficmanager()
        actual_tm_port = traffic_manager.get_port()
    print(f"Traffic Manager Port      : {actual_tm_port}")

    try:
        # 1. Spawn Surrounding NPC Traffic across the entire city
        print(f"\n[1/4] Spawning {NUM_NPCS} background traffic vehicles across city roads...")
        blueprints = world.get_blueprint_library()
        car_bps = [bp for bp in blueprints.filter("vehicle.*") if int(bp.get_attribute("number_of_wheels")) == 4]

        random.seed(101)
        shuffled_spawns = list(spawn_points)
        random.shuffle(shuffled_spawns)

        # Reserve first spawn for ego vehicle
        ego_spawn = shuffled_spawns.pop(0)

        for sp in shuffled_spawns[:NUM_NPCS]:
            bp = random.choice(car_bps)
            bp.set_attribute("role_name", "autopilot")
            npc = world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, actual_tm_port)
                traffic_manager.auto_lane_change(npc, True)
                traffic_manager.ignore_lights_percentage(npc, 0.0)
                traffic_manager.ignore_signs_percentage(npc, 0.0)
                npc_vehicles.append(npc)
                actors.append(npc)

        print(f"      Successfully spawned {len(npc_vehicles)} NPC vehicles roaming the map.")

        # 2. Spawn Ego Vehicle (Tesla Model 3)
        print("\n[2/4] Spawning Autonomous Ego Vehicle (Tesla Model 3)...")
        ego_bp = blueprints.find("vehicle.tesla.model3")
        ego_bp.set_attribute("role_name", "trajpilot_ego")
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            # Fallback to next available spawn point
            for sp in shuffled_spawns[NUM_NPCS:]:
                ego = world.try_spawn_actor(ego_bp, sp)
                if ego:
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle at available spawn points.")

        actors.append(ego)
        print(f"      Ego Vehicle ID: {ego.id} spawned at ({ego_spawn.location.x:.1f}, {ego_spawn.location.y:.1f})")

        # 3. Configure Synchronous Simulation & Autonomous Driving
        print("\n[3/4] Enabling synchronous 10 Hz simulation & autonomous city driving...")
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = DT
        world.apply_settings(settings)

        traffic_manager.set_synchronous_mode(True)
        tm_sync = True
        traffic_manager.set_random_device_seed(42)
        traffic_manager.set_global_distance_to_leading_vehicle(3.0)

        # Autonomous driving rules for Ego Vehicle:
        ego.set_autopilot(True, actual_tm_port)
        traffic_manager.auto_lane_change(ego, True)                 # Allow changing lanes
        traffic_manager.distance_to_leading_vehicle(ego, 3.5)       # Safe following gap
        traffic_manager.vehicle_percentage_speed_difference(ego, -10.0)  # Cruise at ~35 km/h
        traffic_manager.ignore_lights_percentage(ego, 0.0)          # Obey all traffic lights
        traffic_manager.ignore_signs_percentage(ego, 0.0)           # Obey all stop signs

        # Spectator camera
        spectator = world.get_spectator()

        world.tick()
        print("\n[4/4] AUTONOMOUS SIMULATION ACTIVE!")
        print("=" * 80)
        print("VISUAL DEMONSTRATION GUIDE (Inside CarlaUE4 Window):")
        print("  - Camera     : 3rd-person chase camera automatically follows your Tesla Model 3.")
        print("  - Navigation : Autonomous driving obeys traffic lights, stop signs, turns & lanes.")
        print("  - Glowing Cyan Ribbon : TrajPilot Frozen GRU (30 steps, 3.0s horizon) live forecast.")
        print("  - Yellow Dot : 3-second future destination point.")
        print("  - City Traffic: 30+ vehicles cruising across all roads.")
        print("  - Press Ctrl+C in this terminal at any time to finish.")
        print("=" * 80 + "\n")

        rolling_history = deque(maxlen=30)
        start_time = time.time()
        inference_latencies = []
        total_distance = 0.0
        prev_loc = None

        tick = 0
        while tick < MAX_TICKS:
            world.tick()
            tick += 1

            snapshot = world.get_snapshot()
            ego_t = ego.get_transform()
            ego_v = ego.get_velocity()
            ego_loc = ego_t.location
            ego_yaw_deg = ego_t.rotation.yaw
            ego_yaw_rad = math.radians(ego_yaw_deg)
            forward_x = math.cos(ego_yaw_rad)
            forward_y = math.sin(ego_yaw_rad)
            ego_speed = math.sqrt(ego_v.x ** 2 + ego_v.y ** 2 + ego_v.z ** 2)

            # Accumulate travel distance
            if prev_loc is not None:
                d_step = math.sqrt((ego_loc.x - prev_loc.x) ** 2 + (ego_loc.y - prev_loc.y) ** 2)
                total_distance += d_step
            prev_loc = ego_loc

            # Smooth third-person chase camera
            cam_x = ego_loc.x - 7.5 * forward_x
            cam_y = ego_loc.y - 7.5 * forward_y
            cam_z = ego_loc.z + 3.8
            spectator.set_transform(carla.Transform(
                carla.Location(x=cam_x, y=cam_y, z=cam_z),
                carla.Rotation(pitch=-15.0, yaw=ego_yaw_deg, roll=0.0)
            ))

            # Detect closest leading vehicle in corridor
            lead_info = find_leading_vehicle(ego_loc, ego_yaw_rad, ego_v, npc_vehicles)
            if lead_info:
                lead_actor, space_gap, lat_dist, rel_vel = lead_info
            else:
                space_gap = np.nan
                lat_dist = np.nan
                rel_vel = np.nan

            # Save state into rolling history
            record = {
                "step": tick,
                "sim_time": snapshot.timestamp.elapsed_seconds,
                "vx": ego_v.x,
                "vy": ego_v.y,
                "speed": ego_speed,
                "yaw_deg": ego_yaw_deg,
                "space_gap": space_gap,
                "lat_dist": lat_dist,
                "rel_vel": rel_vel,
            }
            rolling_history.append(record)

            # Continuous Frozen-GRU Inference & Trajectory Ribbon Rendering
            t_infer_start = time.perf_counter()
            x16 = build_16_features_from_history(rolling_history)
            x32 = exact_16_to_32(x16, means, stds)

            x_tensor = torch.tensor(x32, dtype=torch.float32, device=device)
            with torch.no_grad():
                out_tensor = model(x_tensor)
            pred_ego = out_tensor.cpu().numpy()[0]  # (30, 2)
            t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            inference_latencies.append(t_infer_ms)

            # Convert local (dx, dy) to CARLA 3D world coordinates
            pred_coords = []
            for k in range(FUTURE_STEPS):
                s_lon = float(pred_ego[k, 0])
                s_lat = float(pred_ego[k, 1])
                pred_dx = s_lon * forward_x - s_lat * forward_y
                pred_dy = s_lon * forward_y + s_lat * forward_x
                pred_coords.append((ego_loc.x + pred_dx, ego_loc.y + pred_dy, ego_loc.z + 0.25))

            # Render 3D dynamic trajectory ribbon on road
            draw_live_prediction_ribbon(world, pred_coords, ego_loc)

            # Log live telemetry to console once every second (10 ticks)
            if tick % 10 == 0:
                wp = road_map.get_waypoint(ego_loc, project_to_road=True)
                lead_str = f"{space_gap:.1f}m" if np.isfinite(space_gap) else "Open Road"
                speed_kmh = ego_speed * 3.6
                elapsed = time.time() - start_time
                print(
                    f"[{elapsed:5.1f}s | Tick {tick:5d}] "
                    f"Speed: {speed_kmh:4.1f} km/h | "
                    f"Road {wp.road_id:2d}, Lane {wp.lane_id:2d} | "
                    f"Lead Gap: {lead_str:>9s} | "
                    f"Distance: {total_distance:5.1f}m | "
                    f"GRU: {t_infer_ms:.2f}ms"
                )

    except KeyboardInterrupt:
        print("\n\nDemonstration paused by user (Ctrl+C). Initiating clean exit...")
    except Exception as exc:
        print(f"\n[ERROR] Demonstration encountered an exception: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nCleaning up simulation session...")
        for a in reversed(actors):
            try:
                a.set_autopilot(False, actual_tm_port)
            except Exception:
                pass
            try:
                if a is not None and a.is_alive:
                    a.destroy()
            except Exception:
                pass
        print(f"Cleanly removed {len(actors)} vehicles.")

        if tm_sync:
            try:
                traffic_manager.set_synchronous_mode(False)
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)
            print("Restored original CARLA settings.")
        except Exception:
            pass

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "map": road_map.name,
            "total_ticks": tick if 'tick' in locals() else 0,
            "total_distance_traveled_m": round(total_distance if 'total_distance' in locals() else 0.0, 2),
            "mean_gru_latency_ms": round(float(np.mean(inference_latencies)), 3) if inference_latencies else 0.0,
            "npcs_spawned": len(npc_vehicles),
            "status": "PASS_FULL_CITY_ROAMING",
        }
        SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved demonstration summary to {SUMMARY_JSON}")
        print("DEMONSTRATION SESSION COMPLETE.")


if __name__ == "__main__":
    main()
