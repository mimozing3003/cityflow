"""Tests for frame-exact camera retrieval without a live CARLA server."""
import importlib.util
from pathlib import Path
import queue
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "carla_demo" / "04_spawn_scene.py"
SPEC = importlib.util.spec_from_file_location("trajpilot_scene", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
CarlaScene = MODULE.CarlaScene


class CameraFrameSyncTests(unittest.TestCase):
    def make_scene(self):
        scene = CarlaScene.__new__(CarlaScene)
        scene.image_queue = queue.Queue(maxsize=10)
        scene._pending_image = None
        return scene

    def test_stale_frame_is_discarded_before_exact_frame(self):
        scene = self.make_scene()
        scene.image_queue.put((40, "stale"))
        scene.image_queue.put((41, "matching"))
        frame, image = scene.get_image_for_frame(41, timeout=0.01)
        self.assertEqual(frame, 41)
        self.assertEqual(image, "matching")

    def test_future_frame_is_held_until_its_tick(self):
        scene = self.make_scene()
        scene.image_queue.put((52, "future"))
        frame, image = scene.get_image_for_frame(51, timeout=0.01)
        self.assertIsNone(frame)
        self.assertIsNone(image)
        frame, image = scene.get_image_for_frame(52, timeout=0.01)
        self.assertEqual(frame, 52)
        self.assertEqual(image, "future")

    def test_missing_frame_does_not_return_wrong_timestamp(self):
        scene = self.make_scene()
        scene.image_queue.put((62, "next frame"))
        frame, image = scene.get_image_for_frame(61, timeout=0.01)
        self.assertIsNone(frame)
        self.assertIsNone(image)
        self.assertEqual(scene._pending_image, (62, "next frame"))


if __name__ == "__main__":
    unittest.main()
