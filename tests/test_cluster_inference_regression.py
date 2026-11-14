import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ClusterInferenceRegressionTests(unittest.TestCase):
    def test_ablation_pipeline_uses_cluster_bootstrap_and_cluster_pairs(self):
        source = (ROOT / "scripts/run_highway_ablation.py").read_text(encoding="utf-8")
        self.assertIn("cluster_bootstrap_ci(", source)
        self.assertIn("paired_cluster_test(", source)
        self.assertIn("holm_adjust(", source)
        self.assertIn('ddata["test"]["traj_id"]', source)
        self.assertNotIn("rng_b.randint(0, N, size=N)", source)

    def test_report_marks_historical_confidence_intervals_provisional(self):
        report = (ROOT / "TRAJPILOT_RESEARCH_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("overlapping windows", report.lower())
        self.assertIn("must be regenerated", report.lower())


if __name__ == "__main__":
    unittest.main()
