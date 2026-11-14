"""
TrajPilot CityFlow - Stage 08: Master Live Interactive CARLA Demonstration
Assembles the complete pipeline into a robust, synchronized real-time demonstration.
"""

import sys
import time
import math
from pathlib import Path
from collections import deque
import numpy as np

# Ensure module imports work regardless of working directory
DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

import importlib
mod_00 = importlib.import_module("00_environment_check")
mod_03 = importlib.import_module("03_connect_carla")
mod_04 = importlib.import_module("04_spawn_scene")
mod_05 = importlib.import_module("05_feature_engine")
mod_06 = importlib.import_module("06_live_inference")
mod_07 = importlib.import_module("07_visualization")

connect_carla = mod_03.connect_carla
CarlaScene = mod_04.CarlaScene
CarlaFeatureEngine = mod_05.CarlaFeatureEngine
TrajPilotInferenceEngine = mod_06.TrajPilotInferenceEngine
CarlaVisualizer = mod_07.CarlaVisualizer


def run_demo(max_frames=1000, fps=10):
    print("=" * 78)
    print("TRAJPILOT CITYFLOW | 08 - LIVE CARLA DEMONSTRATION")
    print("=" * 78)

    # 1. Environment preflight
    if not mod_00.run_checks():
        print("[ABORT] Preflight environment check failed.")
        return

    # 2. Connect to CARLA
    client, world, tm, original_settings = connect_carla(fps=fps)
    
    scene = CarlaScene(world, tm, num_npcs=15)
    feature_engine = CarlaFeatureEngine(fps=fps, history_len=30)
    inference_engine = TrajPilotInferenceEngine()
    visualizer = CarlaVisualizer()

    past_breadcrumbs = deque(maxlen=20)

    try:
        # 3. Spawn Ego & Scene Actors
        ego_veh, camera_sensor = scene.spawn()

        print("\nStarting live inference loop (Press ESC or Q on visualization window to exit)...")
        print("-" * 78)

        frame_count = 0
        t_last = time.perf_counter()

        while frame_count < max_frames:
            # Advance simulation by exactly 1 step (10 FPS delta = 0.1s)
            world.tick()
            frame_count += 1

            # Fetch synchronized sensor frame
            frame_id, rgb_image = scene.get_latest_image(timeout=1.0)
            if rgb_image is None:
                continue

            # Update feature engine
            all_actors = world.get_actors()
            current_feats, meta_dict = feature_engine.update(ego_veh, all_actors)
            history_16 = feature_engine.get_history_array()

            # Run frozen GRU trajectory inference
            pred_rel_traj, latency_ms = inference_engine.predict(history_16)

            # Project predicted trajectory to world and camera pixel coordinates
            ego_tf = ego_veh.get_transform()
            world_pred_points = inference_engine.project_to_world(pred_rel_traj, ego_tf)
            screen_pred_points = inference_engine.project_to_camera_pixels(world_pred_points, camera_sensor)

            # Record and project past breadcrumbs
            past_breadcrumbs.append(ego_tf.location)
            screen_past_points = inference_engine.project_to_camera_pixels(list(past_breadcrumbs), camera_sensor)

            # Calculate live FPS
            t_now = time.perf_counter()
            live_fps = 1.0 / max(t_now - t_last, 1e-4)
            t_last = t_now

            # Telemetry dictionary
            telemetry = {
                "fps": live_fps,
                "latency_ms": latency_ms,
                "device": str(inference_engine.device).upper(),
                "speed": float(current_feats[2]),  # speed_derived (m/s)
                "num_vehicles": len(scene.npc_vehicles) + 1
            }

            # Render frame
            running = visualizer.render_frame(
                rgb_image=rgb_image,
                pred_screen_points=screen_pred_points,
                past_screen_points=screen_past_points,
                telemetry=telemetry,
                features_metadata=meta_dict
            )

            if not running:
                print("Exit requested by user.")
                break

            if frame_count % 50 == 0:
                avail_count = sum(1 for m in meta_dict.values() if m.available)
                print(f"[Frame {frame_count:4d}] FPS: {live_fps:4.1f} | Latency: {latency_ms:5.2f}ms | Features: {avail_count}/16 | Speed: {telemetry['speed']*3.6:4.1f} km/h")

    except Exception as e:
        print(f"\n[ERROR] Demo interrupted by exception: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nRestoring simulator and tearing down session...")
        ego_veh = None
        camera_sensor = None
        all_actors = None
        try:
            visualizer.close()
        except Exception:
            pass
        try:
            scene.destroy()
            world.tick()
        except Exception as ex:
            print(f"Error destroying scene: {ex}")
        try:
            if tm is not None:
                tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_settings)
            print("Restored original CARLA settings.")
        except Exception as ex:
            print(f"Error restoring world settings: {ex}")
        print("Demo completed.")


if __name__ == "__main__":
    run_demo()
