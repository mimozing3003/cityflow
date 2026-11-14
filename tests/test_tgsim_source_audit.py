import csv
import tempfile
import unittest
from pathlib import Path

from scripts.audit_tgsim_source_data import audit_csv, resolve_fields, SchemaError


class TGSIMSourceAuditTests(unittest.TestCase):
    def write_fixture(self, tmp, headers, rows, filename="data.csv"):
        path = Path(tmp) / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_i294_hyphenated_published_fields_and_run_lineage(self):
        headers = ["ID", "time", "xloc-kf", "yloc-kf", "lane-kf", "speed-kf",
                   "acceleration-kf", "run_index"]
        rows = [
            ["v1", "0.0", "2120", "20", "2", "10", "0.1", "1"],
            ["v1", "0.1", "2121", "20", "2", "10", "0.1", "1"],
            ["v1", "0.2", "2131", "20", "2", "10", "0.1", "1"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_fixture(tmp, headers, rows)
            result = audit_csv("TGSIM_I294_L2", path)
            self.assertEqual(result["audit"]["rows"], 3)
            self.assertEqual(result["audit"]["distinct_track_groups"], 1)
            self.assertEqual(result["lineage_gate"]["status"], "PASS_RECORDING_ID_PRESENT")
            self.assertEqual(result["audit"]["interpolated_overpass_rows"], 1)
            self.assertEqual(result["schema"]["resolved_semantic_to_source_fields"]["x_m"], "xloc-kf")

    def test_normalized_underscore_aliases_are_explicitly_supported(self):
        headers = ["id", "time", "xloc_kf", "yloc_kf", "lane_kf", "speed_kf",
                   "acceleration_kf", "run_index"]
        rows = [["v2", "0", "10", "5", "3", "4", "0.2", "2"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_fixture(tmp, headers, rows)
            result = audit_csv("TGSIM_I294_L2", path)
            self.assertEqual(result["schema"]["resolved_semantic_to_source_fields"]["track_id"], "id")
            self.assertEqual(result["schema"]["resolved_semantic_to_source_fields"]["x_m"], "xloc_kf")

    def test_missing_recording_id_blocks_lineage_even_when_track_ids_exist(self):
        headers = ["ID", "time", "xloc-kf", "yloc-kf"]
        rows = [["same_id", "0", "1", "1"], ["same_id", "0.1", "2", "1"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_fixture(tmp, headers, rows)
            result = audit_csv("TGSIM_I90_I94_MOVING", path)
            self.assertEqual(result["lineage_gate"]["status"], "BLOCKED_MISSING_RECORDING_ID")
            self.assertEqual(result["audit"]["distinct_track_groups"], 1)

    def test_ambiguous_aliases_are_rejected(self):
        headers = ["ID", "id", "time", "xloc-kf", "yloc-kf"]
        schema = {
            "fields": {
                "track_id": ["ID", "id"],
                "time_s": ["time"],
                "x_m": ["xloc-kf"],
                "y_m": ["yloc-kf"],
            },
            "expected_vectors": [],
        }
        with self.assertRaises(SchemaError):
            resolve_fields(headers, schema)

    def test_duplicate_timestamps_are_reported(self):
        headers = ["ID", "time", "xloc-kf", "yloc-kf", "run_index"]
        rows = [["v1", "0.0", "1", "1", "1"], ["v1", "0.0", "1", "1", "1"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_fixture(tmp, headers, rows)
            result = audit_csv("TGSIM_I294_L2", path)
            self.assertEqual(result["audit"]["duplicate_track_timestamp_keys"], 1)
            self.assertEqual(result["structural_status"], "FAIL")

    def test_foggy_bottom_requires_both_vector_components(self):
        headers = ["ID", "time", "xloc-kf", "yloc-kf", "speed-kf-x",
                   "acceleration-kf-x", "type-most-common"]
        rows = [["p1", "0", "1", "1", "0.1", "0.0", "0"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_fixture(tmp, headers, rows)
            with self.assertRaises(SchemaError):
                audit_csv("TGSIM_FOGGY_BOTTOM", path)

    def test_foggy_bottom_accepts_both_vector_components_and_type(self):
        headers = ["ID", "time", "xloc-kf", "yloc-kf", "speed-kf-x", "speed-kf-y",
                   "acceleration-kf-x", "acceleration-kf-y", "type-most-common"]
        rows = [["p1", "0", "1", "1", "0.1", "0.2", "0.0", "0.0", "0"]]
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_fixture(tmp, headers, rows)
            result = audit_csv("TGSIM_FOGGY_BOTTOM", path)
            self.assertEqual(result["schema"]["resolved_semantic_to_source_fields"]["speed_y_mps"], "speed-kf-y")
            self.assertEqual(result["audit"]["road_user_type_counts"], {"0": 1})
            self.assertEqual(result["lineage_gate"]["status"], "BLOCKED_MISSING_RECORDING_ID")


if __name__ == "__main__":
    unittest.main()
