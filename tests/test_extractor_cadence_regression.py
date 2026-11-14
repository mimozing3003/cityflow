import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTRACTORS = [
    "scripts/run_highway_benchmark.py",
    "scripts/run_highway_ablation.py",
    "scripts/train_tgsim_physical_model.py",
    "scripts/run_tgsim_cross_dataset.py",
]


class ExtractorCadenceRegressionTests(unittest.TestCase):
    def test_all_tgsim_extractors_use_exact_cadence_windowing(self):
        for relative in EXTRACTORS:
            with self.subTest(script=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("iter_window_starts", source)
                self.assertIn("nominal_dt=0.1", source)
                self.assertNotIn("np.max(np.diff(times)) > 0.25", source)

    def test_heading_difference_uses_signed_previous_velocity_components(self):
        for relative in EXTRACTORS:
            with self.subTest(script=relative):
                source = (ROOT / relative).read_text(encoding="utf-8")
                self.assertNotIn("max(feats[t-1, 0], 1e-4)", source)
                self.assertNotIn("max(feats50[t-1, 0], 1e-4)", source)


if __name__ == "__main__":
    unittest.main()
