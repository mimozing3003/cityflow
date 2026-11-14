import unittest

import numpy as np

from scripts.cluster_statistics import cluster_bootstrap_ci, holm_adjust, paired_cluster_test


class ClusterStatisticsTests(unittest.TestCase):
    def test_bootstrap_reports_clusters_not_only_windows(self):
        values = np.array([1.0, 1.2, 1.1, 10.0, 10.2, 9.8])
        groups = np.array(["a", "a", "a", "b", "b", "b"])
        result = cluster_bootstrap_ci(values, groups, n_boot=500, seed=7)
        self.assertEqual(result["n_observations"], 6)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["bootstrap_unit"], "trajectory_id")
        self.assertAlmostEqual(result["estimate"], values.mean())
        self.assertLessEqual(result["ci_lower"], result["ci_upper"])

    def test_paired_test_collapses_windows_to_track_means(self):
        groups = np.array(["a", "a", "a", "b", "b", "c", "c", "c"])
        base = np.array([5, 5, 5, 4, 4, 3, 3, 3], dtype=float)
        model = base - 1.0
        result = paired_cluster_test(base, model, groups)
        self.assertEqual(result["n_clusters"], 3)
        self.assertAlmostEqual(result["mean_difference_base_minus_model"], 1.0)
        self.assertEqual(result["test_unit"], "per-trajectory mean")

    def test_paired_test_requires_multiple_clusters(self):
        with self.assertRaises(ValueError):
            paired_cluster_test([1, 2], [0, 1], ["same", "same"])

    def test_unequal_model_cluster_ids_rejected(self):
        with self.assertRaises(ValueError):
            paired_cluster_test([1, 2, 3], [0, 1, 2], ["a", "a", "b"])

    def test_holm_adjustment_is_monotone_and_restores_order(self):
        raw = [0.04, 0.001, 0.02, 0.5]
        adjusted = holm_adjust(raw)
        self.assertAlmostEqual(adjusted[0], 0.06)
        self.assertAlmostEqual(adjusted[1], 0.004)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertAlmostEqual(adjusted[3], 0.5)

    def test_bad_pvalues_rejected(self):
        with self.assertRaises(ValueError):
            holm_adjust([0.2, float("nan")])


if __name__ == "__main__":
    unittest.main()
