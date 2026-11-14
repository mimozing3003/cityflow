import tempfile
import unittest
from pathlib import Path

from scripts.generate_feature_mapping import DATASETS, FEATURES, build_mapping_rows, write_mapping


class TGSIMFeatureMappingTests(unittest.TestCase):
    def test_all_datasets_have_sixteen_canonical_features(self):
        rows = build_mapping_rows()
        self.assertEqual(len(FEATURES), 16)
        self.assertEqual(len(rows), 3 * 16)
        for dataset_name in DATASETS:
            selected = [row for row in rows if row["target_dataset"] == dataset_name]
            self.assertEqual(len(selected), 16)
            self.assertEqual([row["canonical_feature"] for row in selected],
                             [feature["canonical_feature"] for feature in FEATURES])

    def test_kalman_filtered_acceleration_is_not_marked_native_or_exact(self):
        rows = build_mapping_rows()
        acceleration = [row for row in rows if row["canonical_feature"] == "acceleration_native"]
        self.assertTrue(all(row["equivalence_status"] == "SOURCE_KF_FILTERED_NOT_NATIVE_IMU"
                            for row in acceleration))
        self.assertTrue(all("Kalman-filtered" in row["semantic_caveat"] for row in acceleration))
        self.assertTrue(all(row["equivalence_status"] not in {"EXACT", "NATIVE"}
                            for row in acceleration))

    def test_foggy_bottom_has_vector_kinematics_and_road_user_type(self):
        row = next(row for row in build_mapping_rows()
                   if row["target_dataset"] == "TGSIM_FOGGY_BOTTOM"
                   and row["canonical_feature"] == "acceleration_native")
        self.assertEqual(row["speed_fields"], "speed-kf-x;speed-kf-y")
        self.assertEqual(row["acceleration_fields"], "acceleration-kf-x;acceleration-kf-y")
        self.assertEqual(row["road_user_type_fields"], "type-most-common")

    def test_interaction_features_require_track_association(self):
        rows = [row for row in build_mapping_rows()
                if row["canonical_feature"] in {"SpaceGap", "RelativeVelocity", "time_headway"}]
        self.assertTrue(all(row["equivalence_status"] == "DERIVED_REQUIRES_LEADER_ASSOCIATION"
                            for row in rows))

    def test_lane_semantics_remain_unresolved(self):
        rows = [row for row in build_mapping_rows()
                if row["canonical_feature"] in {"L_Approach", "R_Approach", "L_Adjacent", "R_Adjacent"}]
        self.assertTrue(all(row["equivalence_status"] == "UNRESOLVED_SOURCE_SEMANTICS"
                            for row in rows))

    def test_writer_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "mapping.csv"
            rows = write_mapping(path)
            self.assertTrue(path.exists())
            self.assertEqual(len(rows), 48)
            content = path.read_text(encoding="utf-8")
            self.assertIn("SOURCE_KF_FILTERED_NOT_NATIVE_IMU", content)


if __name__ == "__main__":
    unittest.main()
