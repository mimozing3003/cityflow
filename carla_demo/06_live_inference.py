"""
TrajPilot CityFlow - Stage 06: Live Frozen GRU Inference Pipeline
Takes 30-frame history, constructs 32-channel tensor, executes model under torch.no_grad,
and transforms predicted relative displacements to CARLA world/visual coordinates.
"""

import time
import math
import numpy as np
import torch

try:
    import carla
except ImportError:
    carla = None

import importlib
import sys
from pathlib import Path

# Handle numbered module imports cleanly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

mod_01 = importlib.import_module("01_load_model")
mod_02 = importlib.import_module("02_load_normalization")
load_frozen_model = mod_01.load_frozen_model
TrajPilotNormalizer = mod_02.TrajPilotNormalizer


class TrajPilotInferenceEngine:
    """
    Manages frozen GRU live forward pass and coordinate projections.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = load_frozen_model(self.device)
        self.normalizer = TrajPilotNormalizer()
        self.is_cuda = (self.device.type == "cuda")

    def predict(self, history_16):
        """
        history_16: (30, 16) array of causal features (can contain NaNs).
        Returns:
            predicted_traj: (30, 2) numpy array of predicted (dx, dy) offsets relative to ego
            latency_ms: float inference time in milliseconds
        """
        # 1. Construct 30x32 tensor with missingness indicators
        input_32 = self.normalizer.transform_history(history_16)
        tensor_x = torch.from_numpy(input_32).unsqueeze(0).to(self.device)  # (1, 30, 32)

        # 2. Synchronize before timing
        if self.is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 3. Model forward pass under torch.no_grad()
        with torch.no_grad():
            output = self.model(tensor_x)  # (1, 30, 2)

        # 4. Synchronize after timing
        if self.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        pred_traj = output[0].cpu().numpy()  # (30, 2)
        return pred_traj, latency_ms

    @staticmethod
    def project_to_world(pred_rel_traj, ego_transform):
        """
        Transforms ego-relative (dx, dy) displacements into CARLA world coordinates.
        pred_rel_traj: (30, 2) where column 0 is forward displacement (m), column 1 is lateral (m).
        ego_transform: carla.Transform of ego vehicle.
        Returns: list of carla.Location objects.
        """
        if carla is None:
            return []

        ego_loc = ego_transform.location
        yaw_rad = math.radians(ego_transform.rotation.yaw)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        world_points = []
        for i in range(len(pred_rel_traj)):
            fwd = float(pred_rel_traj[i, 0])
            lat = float(pred_rel_traj[i, 1])

            # Rotate relative (fwd, lat) by ego heading
            world_x = ego_loc.x + (fwd * cos_y - lat * sin_y)
            world_y = ego_loc.y + (fwd * sin_y + lat * cos_y)
            world_z = ego_loc.z + 0.3  # float slightly above road
            world_points.append(carla.Location(x=world_x, y=world_y, z=world_z))

        return world_points

    @staticmethod
    def project_to_camera_pixels(world_points, camera_actor, image_w=1280, image_h=720):
        """
        Projects 3D CARLA world locations onto 2D camera viewport pixels.
        Returns list of (px, py) tuples.
        """
        if carla is None or not world_points or camera_actor is None:
            return []

        cam_tf = camera_actor.get_transform()
        cam_loc = cam_tf.location
        
        # Camera intrinsic matrix from FOV
        fov = float(camera_actor.attributes.get("fov", 90.0))
        focal = image_w / (2.0 * math.tan(math.radians(fov) / 2.0))
        c_u = image_w / 2.0
        c_v = image_h / 2.0

        # World to camera matrix
        w2c = np.array(cam_tf.get_inverse_matrix())

        pixel_points = []
        for loc in world_points:
            # 3D homogeneous coordinate
            p_world = np.array([loc.x, loc.y, loc.z, 1.0])
            # Camera frame (Unreal coordinate system: x=fwd, y=right, z=up)
            p_cam = w2c @ p_world
            
            # Discard points behind camera
            if p_cam[0] <= 0.5:
                continue

            # Project to camera plane (Unreal: y=right -> u, -z=down -> v, x=fwd -> depth)
            u = (focal * p_cam[1] / p_cam[0]) + c_u
            v = (-focal * p_cam[2] / p_cam[0]) + c_v

            if 0 <= u < image_w and 0 <= v < image_h:
                pixel_points.append((int(u), int(v)))

        return pixel_points
