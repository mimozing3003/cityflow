import unittest

from scripts.group_splitting import split_group_keys


class RecordingDisjointSplitTests(unittest.TestCase):
    def setUp(self):
        self.keys = [("veh_a", "run_1"), ("veh_b", "run_1"),
                     ("veh_c", "run_2"), ("veh_d", "run_2"),
                     ("veh_e", "run_3"), ("veh_f", "run_3"),
                     ("veh_g", "run_4"), ("veh_h", "run_4")]
        self.to_recording = {key: key[1] for key in self.keys}

    def test_recording_split_never_leaks_runs_between_partitions(self):
        split, audit = split_group_keys(self.keys, self.to_recording, "recording", seed=7)
        self.assertTrue(audit["recording_disjoint"])
        self.assertEqual(audit["recording_split_overlap_count"], 0)
        for part, keys in split.items():
            run_ids = {self.to_recording[key] for key in keys}
            self.assertTrue(run_ids)
            for other_part, other_keys in split.items():
                if part != other_part:
                    self.assertFalse(run_ids & {self.to_recording[key] for key in other_keys})

    def test_recording_split_is_deterministic(self):
        a, _ = split_group_keys(self.keys, self.to_recording, "recording", seed=21)
        b, _ = split_group_keys(self.keys, self.to_recording, "recording", seed=21)
        self.assertEqual(a, b)

    def test_recording_split_requires_run_ids(self):
        mapping = dict(self.to_recording)
        mapping[self.keys[0]] = ""
        with self.assertRaises(ValueError):
            split_group_keys(self.keys, mapping, "recording")

    def test_recording_split_requires_at_least_three_runs(self):
        keys = self.keys[:4]
        mapping = {key: key[1] for key in keys}
        with self.assertRaises(ValueError):
            split_group_keys(keys, mapping, "recording")

    def test_recording_split_allocates_four_runs_as_two_one_one(self):
        split, audit = split_group_keys(self.keys, self.to_recording, "recording", seed=10)
        self.assertEqual(len(audit["train_recording_ids"]), 2)
        self.assertEqual(len(audit["val_recording_ids"]), 1)
        self.assertEqual(len(audit["test_recording_ids"]), 1)

    def test_trajectory_split_is_explicitly_not_recording_disjoint(self):
        split, audit = split_group_keys(self.keys, self.to_recording, "trajectory", seed=42)
        self.assertFalse(audit["recording_disjoint"])
        self.assertIsNone(audit["recording_split_overlap_count"])
        all_keys = set().union(*split.values())
        self.assertEqual(all_keys, set(self.keys))


if __name__ == "__main__":
    unittest.main()
