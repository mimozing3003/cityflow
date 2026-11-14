"""
TrajPilot CityFlow - CARLA feature validity gate.

This module tests engineering contracts of the current CARLA feature engine.
It does not certify equivalence with I-24 or validate prediction accuracy.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "carla_demo" / "05_feature_engine.py"
SPEC = importlib.util.spec_from_file_location("trajpilot_feature_engine", ENGINE_PATH)
ENGINE_MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENGINE_MODULE
SPEC.loader.exec_module(ENGINE_MODULE)
CarlaFeatureEngine = ENGINE_MODULE.CarlaFeatureEngine
CANONICAL_FEATURES = ENGINE_MODULE.CANONICAL_FEATURES


class Vec:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        return float(np.linalg.norm([self.x-other.x, self.y-other.y, self.z-other.z]))


class Rotation:
    def __init__(self, yaw=0.0):
        self.yaw = yaw


class Transform:
    def __init__(self, loc, yaw=0.0):
        self.location = loc
        self.rotation = Rotation(yaw)


class Actor:
    type_id = "vehicle.tesla.model3"

    def __init__(self, actor_id, x, y, yaw=0.0, vx=0.0, vy=0.0, ax=0.0, ay=0.0):
        self.id = actor_id
        self._transform = Transform(Vec(x, y, 0.0), yaw)
        self._velocity = Vec(vx, vy, 0.0)
        self._acceleration = Vec(ax, ay, 0.0)
        self._angular_velocity = Vec(0.0, 0.0, 0.0)

    def get_transform(self):
        return self._transform

    def get_velocity(self):
        return self._velocity

    def get_acceleration(self):
        return self._acceleration

    def get_angular_velocity(self):
        return self._angular_velocity

    def get_location(self):
        return self._transform.location


class FeatureValidityGateTests(unittest.TestCase):
    def test_feature_order_and_shape(self):
        engine = CarlaFeatureEngine(fps=10.0, history_len=30)
        ego = Actor(1, 0, 0, vx=10)
        features, metadata = engine.update(ego, [ego])
        self.assertEqual(features.shape, (16,))
        self.assertEqual(list(metadata), CANONICAL_FEATURES)

    def test_unresolved_lane_features_are_explicitly_missing(self):
        engine = CarlaFeatureEngine(fps=10.0, history_len=30)
        ego = Actor(1, 0, 0, vx=10)
        features, metadata = engine.update(ego, [ego])
        for index, name in enumerate(CANONICAL_FEATURES[12:], start=12):
            self.assertFalse(metadata[name].available, name)
            self.assertEqual(metadata[name].exact_or_approximate, "UNRESOLVED")
            self.assertTrue(np.isnan(features[index]), name)

    def test_no_leader_means_missing_interactions(self):
        engine = CarlaFeatureEngine(fps=10.0, history_len=30)
        ego = Actor(1, 0, 0, vx=10)
        features, metadata = engine.update(ego, [ego])
        for index, name in enumerate(CANONICAL_FEATURES[8:12], start=8):
            self.assertFalse(metadata[name].available, name)
            self.assertTrue(np.isnan(features[index]), name)

    def test_leader_relative_velocity_sign_and_headway(self):
        engine = CarlaFeatureEngine(fps=10.0, history_len=30)
        ego = Actor(1, 0, 0, vx=10)
        leader = Actor(2, 20, 0, vx=8)
        features, metadata = engine.update(ego, [ego, leader])
        self.assertTrue(metadata["SpaceGap"].available)
        self.assertAlmostEqual(float(features[8]), 15.5, places=5)
        self.assertAlmostEqual(float(features[9]), 2.0, places=5)
        self.assertAlmostEqual(float(features[11]), 1.55, places=5)

    def test_moving_ego_no_leader_metadata_contract(self):
        engine = CarlaFeatureEngine(fps=10.0, history_len=30)
        ego = Actor(1, 0, 0, vx=5)
        _, metadata = engine.update(ego, [ego])
        self.assertFalse(metadata["L_Approach"].available)
        self.assertFalse(metadata["R_Approach"].available)
        self.assertFalse(metadata["L_Adjacent"].available)
        self.assertFalse(metadata["R_Adjacent"].available)


if __name__ == "__main__":
    unittest.main()
