"""
TrajPilot CityFlow - Stage 05: Feature Engineering Engine & Metadata Auditor
Computes 16 canonical features per frame from CARLA telemetry.
Adheres strictly to research integrity: clearly delineates native, approximate, and unresolved features.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    import carla
except ImportError:
    carla = None

CANONICAL_FEATURES = [
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
    "R_Adjacent"
]


@dataclass
class FeatureMetadata:
    feature_name: str
    source: str
    method: str
    units: str
    exact_or_approximate: str  # "NATIVE", "APPROXIMATE", "UNRESOLVED"
    available: bool
    missing_reason: str = ""


class CarlaFeatureEngine:
    """
    Stateful causal feature computer maintaining a 30-frame history buffer (3.0s at 10 FPS).
    Ensures scientific honesty by returning NaN for unresolved or out-of-range interaction features.
    """
    def __init__(self, fps: float = 10.0, history_len: int = 30):
        self.fps = fps
        self.dt = 1.0 / fps
        self.history_len = history_len

        # Ring buffers for causal kinematics (up to 30 frames)
        self.raw_history = deque(maxlen=history_len)  # stores dict of raw kinematic telemetry
        self.feature_history = deque(maxlen=history_len)  # stores (16,) float32 arrays
        self.last_metadata: Dict[str, FeatureMetadata] = {}

    def reset(self):
        self.raw_history.clear()
        self.feature_history.clear()
        self.last_metadata.clear()

    def update(self, ego_vehicle, world_actors) -> Tuple[np.ndarray, Dict[str, FeatureMetadata]]:
        """
        Computes 16 features for the current frame, appends to buffer, and returns current feature vector and metadata.
        """
        # 1. Extract Ego Kinematics in Ego Local Reference Frame
        ego_tf = ego_vehicle.get_transform()
        ego_vel = ego_vehicle.get_velocity()  # world coords (m/s)
        ego_acc = ego_vehicle.get_acceleration()  # world coords (m/s^2)
        ego_ang_vel = ego_vehicle.get_angular_velocity()  # degrees/s

        # Ego Heading (Yaw in radians)
        yaw_rad = math.radians(ego_tf.rotation.yaw)
        cos_yaw = math.cos(yaw_rad)
        sin_yaw = math.sin(yaw_rad)

        # Transform world velocity to ego longitudinal (vx) and lateral (vy)
        vx_local = ego_vel.x * cos_yaw + ego_vel.y * sin_yaw
        vy_local = -ego_vel.x * sin_yaw + ego_vel.y * cos_yaw
        speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

        # Native acceleration magnitude along forward heading
        acc_native = ego_acc.x * cos_yaw + ego_acc.y * sin_yaw

        # Turn rate in rad/s (yaw rate)
        turn_rate = math.radians(ego_ang_vel.z)

        # Store raw record for causal window computations
        raw_record = {
            "vx": vx_local,
            "vy": vy_local,
            "speed": speed,
            "acc_native": acc_native,
            "turn_rate": turn_rate,
            "loc": ego_tf.location
        }
        self.raw_history.append(raw_record)

        # 2. Causal Kinematic Window Computations
        # speed_derived: derived from consecutive positions
        if len(self.raw_history) >= 2:
            prev_rec = self.raw_history[-2]
            dist = raw_record["loc"].distance(prev_rec["loc"])
            speed_derived = dist / self.dt
        else:
            speed_derived = speed

        # acceleration_causal_0p5s: 0.5-second backward window (5 frames at 10 FPS)
        w_5 = min(5, len(self.raw_history))
        if w_5 >= 2:
            dv = self.raw_history[-1]["speed"] - self.raw_history[-w_5]["speed"]
            acc_0p5s = dv / ((w_5 - 1) * self.dt)
        else:
            acc_0p5s = acc_native

        # jerk_causal: derivative of acceleration over 1-step
        if len(self.raw_history) >= 2:
            da = self.raw_history[-1]["acc_native"] - self.raw_history[-2]["acc_native"]
            jerk = da / self.dt
        else:
            jerk = 0.0

        # curvature_valid: turn_rate / speed (with numerical safety)
        if speed > 0.5:
            curvature = turn_rate / speed
        else:
            curvature = 0.0

        # 3. Leading Vehicle Interaction Extraction (CARLA Scene Geometry)
        # Search for closest leading vehicle within a 75m cone ahead of ego
        leader, space_gap, rel_vel, lat_dist = self._find_leading_vehicle(ego_vehicle, ego_tf, world_actors)

        if leader is not None and space_gap > 0.1:
            time_headway = space_gap / max(speed, 0.5)
            interaction_avail = True
        else:
            space_gap = np.nan
            rel_vel = np.nan
            lat_dist = np.nan
            time_headway = np.nan
            interaction_avail = False

        # 4. Lane Context Features
        # These features are unresolved: writing 0.0 would falsely mark them as observed
        # valid values in the model input. NaN activates the explicit missingness channels.
        l_approach = np.nan
        r_approach = np.nan
        l_adjacent = np.nan
        r_adjacent = np.nan

        # 5. Assemble 16-feature vector
        features = np.array([
            vx_local,
            vy_local,
            speed_derived,
            acc_native,
            acc_0p5s,
            turn_rate,
            curvature,
            jerk,
            space_gap,
            rel_vel,
            lat_dist,
            time_headway,
            l_approach,
            r_approach,
            l_adjacent,
            r_adjacent
        ], dtype=np.float32)

        # 6. Metadata Audit
        metadata = {
            "vx_causal": FeatureMetadata("vx_causal", "CARLA IMU/Transform", "Causal longitudinal velocity", "m/s", "NATIVE", True),
            "vy_causal": FeatureMetadata("vy_causal", "CARLA IMU/Transform", "Causal lateral velocity", "m/s", "NATIVE", True),
            "speed_derived": FeatureMetadata("speed_derived", "CARLA Location Δ", "Backward finite difference", "m/s", "APPROXIMATE", True),
            "acceleration_native": FeatureMetadata("acceleration_native", "CARLA Actor IMU", "Forward vehicle acceleration", "m/s²", "NATIVE", True),
            "acceleration_causal_0p5s": FeatureMetadata("acceleration_causal_0p5s", "0.5s Causal Window", "Δv / 0.5s backward window", "m/s²", "APPROXIMATE", True),
            "turn_rate_valid": FeatureMetadata("turn_rate_valid", "CARLA Gyroscope", "Yaw angular velocity", "rad/s", "NATIVE", True),
            "curvature_valid": FeatureMetadata("curvature_valid", "Kinematic Ratio", "turn_rate / speed", "1/m", "APPROXIMATE", True),
            "jerk_causal": FeatureMetadata("jerk_causal", "1-step Acceleration Δ", "da / dt", "m/s³", "APPROXIMATE", True),
            "SpaceGap": FeatureMetadata("SpaceGap", "CARLA Actor BoundingBoxes", "Euclidean bumper gap to leader", "m", "APPROXIMATE", interaction_avail, "No leader ahead" if not interaction_avail else ""),
            "RelativeVelocity": FeatureMetadata("RelativeVelocity", "CARLA Actor Velocities", "ego_v - leader_v", "m/s", "APPROXIMATE", interaction_avail, "No leader ahead" if not interaction_avail else ""),
            "LateralDistance": FeatureMetadata("LateralDistance", "CARLA Actor Coordinates", "Ego lateral offset to leader", "m", "APPROXIMATE", interaction_avail, "No leader ahead" if not interaction_avail else ""),
            "time_headway": FeatureMetadata("time_headway", "Gap / Speed Ratio", "SpaceGap / ego speed", "s", "APPROXIMATE", interaction_avail, "No leader ahead" if not interaction_avail else ""),
            "L_Approach": FeatureMetadata("L_Approach", "CARLA Waypoint/Map", "Left lane boundary proximity", "prob", "UNRESOLVED", False, "I-24 semantic equivalence unproven"),
            "R_Approach": FeatureMetadata("R_Approach", "CARLA Waypoint/Map", "Right lane boundary proximity", "prob", "UNRESOLVED", False, "I-24 semantic equivalence unproven"),
            "L_Adjacent": FeatureMetadata("L_Adjacent", "CARLA Waypoint/Map", "Left adjacent lane presence", "flag", "UNRESOLVED", False, "I-24 semantic equivalence unproven"),
            "R_Adjacent": FeatureMetadata("R_Adjacent", "CARLA Waypoint/Map", "Right adjacent lane presence", "flag", "UNRESOLVED", False, "I-24 semantic equivalence unproven")
        }

        self.last_metadata = metadata
        self.feature_history.append(features)
        return features, metadata

    def _find_leading_vehicle(self, ego, ego_tf, world_actors):
        ego_loc = ego_tf.location
        ego_yaw = math.radians(ego_tf.rotation.yaw)
        cos_y = math.cos(ego_yaw)
        sin_y = math.sin(ego_yaw)

        min_dist = 75.0  # search range
        closest_leader = None
        closest_rel_vel = 0.0
        closest_lat_dist = 0.0

        for actor in world_actors:
            if actor.id == ego.id or "vehicle" not in actor.type_id:
                continue
            act_loc = actor.get_location()
            dx = act_loc.x - ego_loc.x
            dy = act_loc.y - ego_loc.y

            # Transform to ego longitudinal (fwd) and lateral (lat)
            fwd = dx * cos_y + dy * sin_y
            lat = -dx * sin_y + dy * cos_y

            # Vehicle must be ahead (fwd > 2.0m) and roughly within lane corridor (|lat| < 2.5m)
            if 2.0 < fwd < min_dist and abs(lat) < 2.5:
                min_dist = fwd
                closest_leader = actor
                closest_lat_dist = lat

                # Relative velocity along heading
                ego_v = ego.get_velocity()
                act_v = actor.get_velocity()
                ego_v_fwd = ego_v.x * cos_y + ego_v.y * sin_y
                act_v_fwd = act_v.x * cos_y + act_v.y * sin_y
                closest_rel_vel = ego_v_fwd - act_v_fwd

        if closest_leader is not None:
            # Approximate bumper-to-bumper gap (subtract ~4.5m vehicle lengths)
            bumper_gap = max(0.5, min_dist - 4.5)
            return closest_leader, bumper_gap, closest_rel_vel, closest_lat_dist
        return None, 0.0, 0.0, 0.0

    def is_history_ready(self) -> bool:
        return len(self.feature_history) >= self.history_len

    def get_history_array(self) -> np.ndarray:
        """
        Returns (30, 16) array. If fewer than 30 frames collected, prepends earliest frame (causal padding).
        """
        if len(self.feature_history) == 0:
            return np.full((self.history_len, 16), np.nan, dtype=np.float32)

        arr = np.array(self.feature_history, dtype=np.float32)
        if len(arr) < self.history_len:
            pad_len = self.history_len - len(arr)
            pad = np.repeat(arr[:1], pad_len, axis=0)
            return np.vstack([pad, arr])
        return arr[-self.history_len:]
