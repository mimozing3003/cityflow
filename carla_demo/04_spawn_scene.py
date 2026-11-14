"""
TrajPilot CityFlow - Stage 04: Scene Spawning, Ego Vehicle & Sensor Rig
Spawns ego vehicle, surrounding traffic, attaches synchronized RGB camera, and manages actor registry.
"""

import random
import queue
import numpy as np

try:
    import carla
except ImportError:
    carla = None


class CarlaScene:
    def __init__(self, world, tm, num_npcs=15, camera_width=1280, camera_height=720, camera_fov=90):
        self.world = world
        self.tm = tm
        self.bp_lib = world.get_blueprint_library()
        self.spawn_points = world.get_map().get_spawn_points()
        self.num_npcs = num_npcs
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fov = camera_fov
        
        self.ego_vehicle = None
        self.npc_vehicles = []
        self.camera_sensor = None
        self.image_queue = queue.Queue(maxsize=10)
        self.all_actor_ids = []

    def spawn(self):
        print("=" * 78)
        print("TRAJPILOT CITYFLOW | 04 - SCENE & ACTOR SPAWNING")
        print("=" * 78)
        if len(self.spawn_points) < 5:
            raise RuntimeError(f"Map has insufficient spawn points ({len(self.spawn_points)})")

        random.seed(42)
        shuffled_spawns = list(self.spawn_points)
        random.shuffle(shuffled_spawns)

        # 1. Spawn Ego Vehicle
        ego_bp = self.bp_lib.filter("vehicle.tesla.model3")
        if not ego_bp:
            ego_bp = self.bp_lib.filter("vehicle.*")
        ego_bp = ego_bp[0]
        ego_bp.set_attribute("role_name", "ego")

        ego_spawn = shuffled_spawns.pop(0)
        self.ego_vehicle = self.world.spawn_actor(ego_bp, ego_spawn)
        self.all_actor_ids.append(self.ego_vehicle.id)
        self.ego_vehicle.set_autopilot(True, self.tm.get_port())
        print(f"Ego Vehicle       : {ego_bp.id} (ID: {self.ego_vehicle.id}) spawned at {ego_spawn.location}")

        # 2. Attach RGB Camera
        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(self.camera_width))
        cam_bp.set_attribute("image_size_y", str(self.camera_height))
        cam_bp.set_attribute("fov", str(self.camera_fov))
        
        # Chase camera transform: behind and slightly elevated
        cam_transform = carla.Transform(
            carla.Location(x=-6.0, z=2.8),
            carla.Rotation(pitch=-14.0)
        )
        self.camera_sensor = self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.ego_vehicle)
        self.all_actor_ids.append(self.camera_sensor.id)

        # Synchronous image queue listener
        def _on_image(image):
            # Convert raw BGRA buffer to RGB numpy array
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            rgb_array = array[:, :, [2, 1, 0]]  # Convert CARLA BGRA to RGB
            # Non-blocking put, discard old if congested
            try:
                if self.image_queue.full():
                    self.image_queue.get_nowait()
                self.image_queue.put_nowait((image.frame, rgb_array))
            except queue.Empty:
                pass

        self.camera_sensor.listen(_on_image)
        print(f"Camera Rig        : RGB {self.camera_width}x{self.camera_height} (FOV {self.camera_fov}) attached")

        # 3. Spawn Surrounding Traffic
        vehicle_bps = self.bp_lib.filter("vehicle.*")
        # Filter out 2-wheelers and large trucks for smoother traffic flow
        vehicle_bps = [bp for bp in vehicle_bps if int(bp.get_attribute("number_of_wheels")) == 4]

        spawned_count = 0
        for spawn_pt in shuffled_spawns[:self.num_npcs]:
            bp = random.choice(vehicle_bps)
            bp.set_attribute("role_name", "autopilot")
            npc = self.world.try_spawn_actor(bp, spawn_pt)
            if npc:
                npc.set_autopilot(True, self.tm.get_port())
                self.npc_vehicles.append(npc)
                self.all_actor_ids.append(npc.id)
                spawned_count += 1

        print(f"Surrounding NPCs  : {spawned_count}/{self.num_npcs} vehicles spawned successfully")
        print("=" * 78)
        return self.ego_vehicle, self.camera_sensor

    def get_latest_image(self, timeout=2.0):
        try:
            return self.image_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None

    def destroy(self):
        print("Tearing down CARLA scene actors...")
        if self.camera_sensor and self.camera_sensor.is_alive:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None
        for npc in self.npc_vehicles:
            if npc and npc.is_alive:
                npc.destroy()
        self.npc_vehicles.clear()
        if self.ego_vehicle and self.ego_vehicle.is_alive:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
        print("Scene teardown completed cleanly.")
