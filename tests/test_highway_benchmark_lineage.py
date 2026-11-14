import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class HighwayBenchmarkLineageRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=(ROOT/"scripts/run_highway_benchmark.py").read_text(encoding="utf-8")

    def test_extractor_preserves_window_identity_and_source_hash(self):
        for token in [
            '"source_file_sha256"', '"trajectory_id"', '"recording_id"',
            '"window_start_index"', '"future_timestamps_s"', '"window_id"',
        ]:
            self.assertIn(token,self.source)

    def test_benchmark_saves_long_predictions_and_partition_metadata(self):
        self.assertIn("highway_predictions_long.csv",self.source)
        self.assertIn("build_prediction_rows(",self.source)
        self.assertIn("build_window_metadata_rows(",self.source)
        self.assertIn("trajectory_split_manifest.csv",self.source)

    def test_main_benchmark_uses_clustered_paired_statistics(self):
        self.assertIn("paired_cluster_test(",self.source)
        self.assertIn("cluster_bootstrap_ci(",self.source)
        self.assertIn("holm_adjust(",self.source)
        self.assertNotIn("stats.ttest_rel(cv_err, m_err)",self.source)

    def test_recording_overlap_is_reported_not_assumed_away(self):
        self.assertIn("recording_split_overlap_count",self.source)
        self.assertIn("recording_session_disjoint_split",self.source)
        self.assertIn("NOT_ESTABLISHED_UNLESS_RECORDING_SPLIT_DISJOINT",self.source)


if __name__=="__main__": unittest.main()
