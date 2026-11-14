"""
TrajPilot CityFlow - Stage 10: Robust Teardown & Simulator State Restoration
Safely cleans up CARLA world settings, disables frozen synchronous modes, and destroys all spawned actors.
"""

import sys
import pygame

try:
    import carla
except ImportError:
    carla = None


def cleanup_carla(host="127.0.0.1", port=2000):
    print("=" * 78)
    print("TRAJPILOT CITYFLOW | 10 - SIMULATOR CLEANUP & RESTORATION")
    print("=" * 78)

    # 1. Close Pygame if active
    if pygame.get_init():
        pygame.quit()
        print("Closed Pygame display context.")

    if carla is None:
        print("CARLA Python API not imported. Skipping server cleanup.")
        return

    try:
        client = carla.Client(host, port)
        client.set_timeout(5.0)
        world = client.get_world()
    except Exception as e:
        print(f"CARLA server not running at {host}:{port} ({e}). Nothing to clean up.")
        return

    # 2. Reset Traffic Manager & World Synchronous Mode
    try:
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(False)
    except Exception:
        pass

    try:
        settings = world.get_settings()
        if settings.synchronous_mode:
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            print("Restored asynchronous mode (synchronous_mode = False).")
    except Exception as ex:
        print(f"Warning restoring settings: {ex}")

    # 3. Destroy spawned actors
    destroyed_count = 0
    actors = world.get_actors()
    for actor in actors:
        # Target demo vehicles and sensors
        role = actor.attributes.get("role_name", "")
        type_id = actor.type_id
        if role in ("ego", "autopilot") or type_id.startswith("sensor.camera"):
            try:
                actor.destroy()
                destroyed_count += 1
            except Exception:
                pass

    print(f"Cleaned up {destroyed_count} spawned demonstration actors/sensors.")
    print("Simulator restored to clean initial state.")
    print("=" * 78)


if __name__ == "__main__":
    cleanup_carla()
