
import sys
import time
import numpy as np

try:
    import carla
except ImportError as exc:
    raise RuntimeError(
        "CARLA Python API is not importable in this environment."
    ) from exc

sys.path.insert(
    0,
    r"C:\Users\SOMUJIT\Desktop\cityflow\carla_eval"
)

from exact_gru_input_adapter import exact_16_to_32

HOST = "127.0.0.1"
PORT = 2000
TIMEOUT = 10.0

def connect():

    client = carla.Client(
        HOST,
        PORT
    )

    client.set_timeout(
        TIMEOUT
    )

    world = client.get_world()

    return client, world

def vehicle_snapshot(vehicle):

    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()

    return {
        "x": transform.location.x,
        "y": transform.location.y,
        "z": transform.location.z,
        "yaw": transform.rotation.yaw,
        "vx": velocity.x,
        "vy": velocity.y,
        "vz": velocity.z,
        "ax": acceleration.x,
        "ay": acceleration.y,
        "az": acceleration.z
    }

if __name__ == "__main__":

    client, world = connect()

    vehicles = world.get_actors().filter(
        "vehicle.*"
    )

    print("CARLA connection: PASS")
    print("Map:", world.get_map().name)
    print("Vehicles:", len(vehicles))

    if len(vehicles) > 0:

        vehicle = vehicles[0]

        print(
            "Vehicle:",
            vehicle.id,
            vehicle.type_id
        )

        print(
            vehicle_snapshot(vehicle)
        )
