import unittest
import numpy as np

from scripts.benchmark_artifacts import build_prediction_rows, build_window_metadata_rows


class BenchmarkArtifactTests(unittest.TestCase):
    def meta(self):
        return {
            "window_id": "win-001",
            "trajectory_id": "track-1-run-2",
            "recording_id": "2",
            "vehicle_id": "track-1",
            "source_file": "source.csv",
            "source_file_sha256": "a" * 64,
            "window_start_index": 10,
            "segment_start_index": 0,
            "segment_stop_index": 100,
            "history_start_time_s": 1.0,
            "prediction_origin_time_s": 3.9,
            "future_timestamps_s": [4.0 + 0.1 * i for i in range(30)],
        }

    def test_long_prediction_rows_preserve_lineage_and_errors(self):
        prediction = np.zeros((1, 30, 2), dtype=float)
        truth = np.ones((1, 30, 2), dtype=float)
        rows = build_prediction_rows("TGSIM_I294_L2", prediction, truth, [self.meta()])
        self.assertEqual(len(rows), 30)
        self.assertEqual(rows[0]["trajectory_id"], "track-1-run-2")
        self.assertEqual(rows[0]["recording_id"], "2")
        self.assertEqual(rows[0]["horizon_step"], 1)
        self.assertAlmostEqual(rows[0]["displacement_error_m"], np.sqrt(2.0))
        self.assertEqual(rows[-1]["horizon_step"], 30)
        self.assertAlmostEqual(rows[-1]["horizon_seconds"], 3.0)

    def test_prediction_metadata_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            build_prediction_rows("D", np.zeros((1, 30, 2)), np.zeros((2, 30, 2)), [self.meta()])

    def test_metadata_fields_export_one_row_per_window(self):
        rows = build_window_metadata_rows("D", "test", [self.meta()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["target_start_time_s"], 4.0)
        self.assertEqual(rows[0]["target_end_time_s"], 6.9)
        self.assertEqual(rows[0]["recording_id"], "2")

    def test_future_timestamps_required(self):
        meta = self.meta()
        meta["future_timestamps_s"] = [1.0]
        with self.assertRaises(ValueError):
            build_prediction_rows("D", np.zeros((1, 30, 2)), np.zeros((1, 30, 2)), [meta])


if __name__ == "__main__":
    unittest.main()
