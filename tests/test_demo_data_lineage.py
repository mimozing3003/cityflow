import unittest
from pathlib import Path

from scripts.audit_demo_trajectory_lineage import audit_demo_data

ROOT = Path(__file__).resolve().parents[1]


class DemoTrajectoryLineageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = audit_demo_data(ROOT)

    def test_source_logs_have_contiguous_ten_hz_timing(self):
        self.assertEqual(self.report["raw_log"]["rows"], 150)
        self.assertEqual(self.report["raw_log"]["simulation_time_deltas_seconds"], [0.1])
        self.assertEqual(self.report["aligned_log"]["rows"], 150)
        self.assertEqual(self.report["aligned_log"]["frame_deltas"], [1])
        self.assertEqual(self.report["aligned_log"]["elapsed_time_deltas_seconds"], [0.1])

    def test_valid_following_file_is_prefix_not_independent_recording(self):
        subset = self.report["valid_following_subset"]
        self.assertEqual(subset["rows"], 60)
        self.assertTrue(subset["is_exact_prefix_of_aligned_log"])
        self.assertAlmostEqual(subset["fraction_of_aligned_rows"], 0.4)

    def test_following_screen_is_explicitly_heuristic(self):
        screen = self.report["heuristic_following_screen"]
        self.assertEqual(screen["passing_rows"], 114)
        self.assertEqual(screen["contiguous_passing_runs"][0]["start_step"], 0)
        self.assertEqual(screen["contiguous_passing_runs"][0]["end_step"], 113)
        self.assertIn("not ground-truth", screen["scope_warning"])

    def test_prediction_errors_recompute_from_displacements(self):
        pred = self.report["prediction_demo"]
        self.assertEqual(pred["rows"], 30)
        self.assertLess(pred["max_stored_vs_recomputed_error_delta_m"], 0.002)
        self.assertAlmostEqual(pred["fde_3s"]["stored_error_m"], 14.365, places=3)
        self.assertAlmostEqual(pred["fde_1s"]["stored_error_m"], 4.3229, places=3)
        self.assertAlmostEqual(pred["fde_2s"]["stored_error_m"], 9.8318, places=3)

    def test_demo_lineage_is_not_claimed_as_dataset_provenance(self):
        lineage = self.report["lineage_checks"]
        self.assertFalse(lineage["raw_to_aligned_transform_proven"])
        self.assertFalse(lineage["step14_prediction_to_aligned_log_join_proven"])


if __name__ == "__main__":
    unittest.main()
