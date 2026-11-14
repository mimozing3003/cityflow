import carla
import json
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\SOMUJIT\Desktop\cityflow\curla demostrate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(20.0)

world = client.get_world()
blueprints = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()

vehicles = sorted(
    [
        {
            "id": bp.id,
            "role_name": bp.get_attribute("role_name").as_str()
            if bp.has_attribute("role_name")
            else None
        }
        for bp in blueprints.filter("vehicle.*")
    ],
    key=lambda item: item["id"]
)

formatted_spawn_points = [
    {
        "index": i,
        "x": round(sp.location.x, 2),
        "y": round(sp.location.y, 2),
        "z": round(sp.location.z, 2),
        "yaw": round(sp.rotation.yaw, 2)
    }
    for i, sp in enumerate(spawn_points[:10])
]

report = {
    "server_version": client.get_server_version(),
    "map": world.get_map().name,
    "spawn_point_count": len(spawn_points),
    "vehicle_blueprint_count": len(vehicles),
    "vehicle_blueprints": [v["id"] for v in vehicles],
    "first_spawn_points": formatted_spawn_points
}

report_path = OUTPUT_DIR / "scene_inspection_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("=" * 65)
print("TRAJPILOT CITYFLOW | SCENE PREFLIGHT INSPECTION")
print("=" * 65)
print(f"CARLA Server Version     : {report['server_version']}")
print(f"Active Map               : {report['map']}")
print(f"Total Spawn Points       : {report['spawn_point_count']}")
print(f"Vehicle Blueprints Total : {report['vehicle_blueprint_count']}")
print(f"Sample Blueprints        : {report['vehicle_blueprints'][:5]}")
print("\nFirst 5 Spawn Points (x, y, z, yaw):")
for sp in report["first_spawn_points"][:5]:
    print(f"  SpawnPoint[{sp['index']}]: x={sp['x']}, y={sp['y']}, z={sp['z']}, yaw={sp['yaw']}")
print(f"\nFull report written to: {report_path}")
print("=" * 65)
