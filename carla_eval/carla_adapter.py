from collections import deque
import math
import numpy as np

FEATURES_16 = [
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

class CarlaTrajectoryAdapter:
    def __init__(self, history_steps=30, dt=0.1):
        self.history_steps = history_steps
        self.dt = dt
        self.history = deque(maxlen=history_steps)

    def actor_state(self, actor):
        transform = actor.get_transform()
        velocity = actor.get_velocity()
        acceleration = actor.get_acceleration()

        speed = math.sqrt(
            velocity.x ** 2 +
            velocity.y ** 2 +
            velocity.z ** 2
        )

        acceleration_magnitude = math.sqrt(
            acceleration.x ** 2 +
            acceleration.y ** 2 +
            acceleration.z ** 2
        )

        yaw = math.radians(
            transform.rotation.yaw
        )

        vx = (
            velocity.x * math.cos(yaw) +
            velocity.y * math.sin(yaw)
        )

        vy = (
            -velocity.x * math.sin(yaw) +
            velocity.y * math.cos(yaw)
        )

        return {
            "x": transform.location.x,
            "y": transform.location.y,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "acceleration": acceleration_magnitude,
            "yaw": yaw
        }

    def update(self, actor):
        state = self.actor_state(actor)
        self.history.append(state)
        return state

    def build_history(self):
        if len(self.history) != self.history_steps:
            return None

        rows = list(self.history)

        vx = np.array(
            [x["vx"] for x in rows],
            dtype=np.float32
        )

        vy = np.array(
            [x["vy"] for x in rows],
            dtype=np.float32
        )

        speed = np.array(
            [x["speed"] for x in rows],
            dtype=np.float32
        )

        acceleration = np.array(
            [x["acceleration"] for x in rows],
            dtype=np.float32
        )

        yaw = np.unwrap(
            np.array(
                [x["yaw"] for x in rows],
                dtype=np.float32
            )
        )

        causal_acceleration = np.full(
            self.history_steps,
            np.nan,
            dtype=np.float32
        )

        causal_acceleration[5:] = (
            speed[5:] - speed[:-5]
        ) / (5.0 * self.dt)

        jerk = np.full(
            self.history_steps,
            np.nan,
            dtype=np.float32
        )

        jerk[1:] = (
            acceleration[1:] -
            acceleration[:-1]
        ) / self.dt

        turn_rate = np.full(
            self.history_steps,
            np.nan,
            dtype=np.float32
        )

        turn_rate[1:] = (
            yaw[1:] - yaw[:-1]
        ) / self.dt

        curvature = np.full(
            self.history_steps,
            np.nan,
            dtype=np.float32
        )

        valid_speed = speed >= 1.0

        curvature[valid_speed] = (
            turn_rate[valid_speed] /
            np.maximum(speed[valid_speed], 1e-6)
        )

        features = np.column_stack(
            [
                vx,
                vy,
                speed,
                acceleration,
                causal_acceleration,
                turn_rate,
                curvature,
                jerk,
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan),
                np.full(self.history_steps, np.nan)
            ]
        ).astype(np.float32)

        return features
