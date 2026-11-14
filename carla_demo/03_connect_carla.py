"""
TrajPilot CityFlow - Stage 03: CARLA Simulator Connection & Synchronous World Setup
Connects to CARLA 0.9.16 at 127.0.0.1:2000, establishes 10 FPS synchronous mode, and configures Traffic Manager.
"""

import sys
import time
from pathlib import Path

try:
    import carla
except ImportError:
    carla = None


def connect_carla(host="127.0.0.1", port=2000, timeout=15.0, map_name=None, fps=10):
    if carla is None:
        raise ImportError(
            "\n[CAUSE] 'carla' module is not installed in the active environment.\n"
            "[FIX] Install the CARLA 0.9.16 wheel or activate 'carla0916' environment.\n"
            "[COMMAND] pip install carla==0.9.16\n"
        )

    print("=" * 78)
    print("TRAJPILOT CITYFLOW | 03 - CARLA SIMULATOR CONNECTION")
    print("=" * 78)
    print(f"Connecting to CARLA Server at {host}:{port} (timeout={timeout}s)...")

    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        server_ver = client.get_server_version()
        print(f"Server Connection : SUCCESS")
        print(f"Server Version    : {server_ver}")
    except Exception as e:
        print(f"[FAIL] Could not connect to CARLA server at {host}:{port}.")
        print("\n================ DIAGNOSTIC REPORT ================")
        print("[CAUSE] CARLA Simulator is not running or blocked on port 2000.")
        print("[FIX]   Start the CARLA Simulator server executable before running.")
        print("[COMMAND] C:\\CARLA_0.9.16\\Start_CARLA.bat")
        print("          or execute: C:\\CARLA_0.9.16\\CarlaUE4.exe -quality-level=Epic -RenderOffScreen\n")
        raise ConnectionError(f"CARLA connection failed: {e}") from e

    world = client.get_world()
    current_map = world.get_map().name
    print(f"Current Map       : {current_map}")

    # Load requested map if specified and not already active
    if map_name and map_name not in current_map:
        print(f"Loading requested map: {map_name}...")
        try:
            world = client.load_world(map_name)
            time.sleep(1.0)
            print(f"Loaded Map        : {world.get_map().name}")
        except Exception as ex:
            print(f"Warning: Could not load {map_name} ({ex}), continuing with {current_map}")

    # Record original settings for safe teardown
    original_settings = world.get_settings()
    
    # Configure 10 FPS Synchronous Mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    # Configure Traffic Manager Synchronous Mode
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(-10.0)

    print(f"Synchronous Mode  : ENABLED (fixed_delta_seconds={settings.fixed_delta_seconds:.3f} -> {fps} FPS)")
    print(f"Traffic Manager   : SYNCHRONOUS on port 8000")
    print(f"Active Actors     : {len(world.get_actors())}")
    print("=" * 78)
    return client, world, tm, original_settings


if __name__ == "__main__":
    try:
        c, w, tm_inst, orig = connect_carla()
        print("Testing single world tick...")
        w.tick()
        print("World tick completed successfully.")
        print("Restoring original settings...")
        w.apply_settings(orig)
        print("Teardown clean.")
    except Exception as err:
        print(f"Connection test stopped: {err}")
