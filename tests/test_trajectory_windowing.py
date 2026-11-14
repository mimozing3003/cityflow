import unittest

from scripts.trajectory_windowing import contiguous_segments, has_complete_window, iter_window_starts


class ExactCadenceWindowingTests(unittest.TestCase):
    def test_uniform_ten_hz_track_is_one_segment(self):
        times = [i * 0.1 for i in range(100)]
        self.assertEqual(contiguous_segments(times), [(0, 100)])

    def test_two_tenth_second_gap_splits_segment(self):
        times = [i * 0.1 for i in range(40)] + [4.1 + i * 0.1 for i in range(40)]
        segments = contiguous_segments(times)
        self.assertEqual(segments, [(0, 40), (40, 80)])
        self.assertFalse(has_complete_window(times, history_steps=30, future_steps=30))

    def test_duplicate_timestamp_splits_segment(self):
        times = [i * 0.1 for i in range(40)]
        times[20] = times[19]
        self.assertEqual(contiguous_segments(times), [(0, 20), (20, 40)])

    def test_out_of_order_timestamp_splits_segment(self):
        times = [i * 0.1 for i in range(40)]
        times[20] = times[19] - 0.1
        self.assertEqual(contiguous_segments(times), [(0, 20), (20, 40)])

    def test_nonfinite_timestamp_is_excluded_from_segments(self):
        times = [i * 0.1 for i in range(20)] + [float("nan")] + [2.0 + i * 0.1 for i in range(20)]
        self.assertEqual(contiguous_segments(times), [(0, 20), (21, 41)])

    def test_small_timestamp_jitter_within_tolerance_is_accepted(self):
        times = [0.0, 0.101, 0.199, 0.301, 0.4]
        self.assertEqual(contiguous_segments(times), [(0, 5)])

    def test_window_starts_never_cross_gap(self):
        times = [i * 0.1 for i in range(65)] + [6.6 + i * 0.1 for i in range(65)]
        starts = list(iter_window_starts(times, history_steps=30, future_steps=30, stride=5))
        self.assertEqual([start for _, _, start in starts], [0, 5, 10, 65, 70, 75])
        for seg_start, seg_stop, start in starts:
            self.assertGreaterEqual(start, seg_start)
            self.assertLessEqual(start + 60, seg_stop)

    def test_exact_sixty_samples_make_one_window(self):
        times = [i * 0.1 for i in range(60)]
        self.assertEqual(list(iter_window_starts(times)), [(0, 60, 0)])

    def test_short_track_has_no_window(self):
        times = [i * 0.1 for i in range(59)]
        self.assertEqual(list(iter_window_starts(times)), [])
        self.assertFalse(has_complete_window(times))

    def test_invalid_parameters_rejected(self):
        with self.assertRaises(ValueError):
            contiguous_segments([0, 1], nominal_dt=0)
        with self.assertRaises(ValueError):
            list(iter_window_starts([0, 0.1], history_steps=0))


if __name__ == "__main__":
    unittest.main()
