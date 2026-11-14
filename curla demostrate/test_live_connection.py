import carla
import sys

print("=" * 65)
print("TRAJPILOT CITYFLOW | LIVE CARLA CONNECTION TEST")
print("=" * 65)

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(15.0)

try:
    server_version = client.get_server_version()
    client_version = client.get_client_version()
    world = client.get_world()
    print("Python:", sys.version.split()[0])
    print("Client API:", client_version)
    print("Server API:", server_version)
    print("Map:", world.get_map().name)
    print("Actors:", len(world.get_actors()))
    print("Connection: PASS")
except Exception as exc:
    print("Connection: FAIL")
    print("Error:", repr(exc))
    raise

print("=" * 65)
