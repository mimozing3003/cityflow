"""Cadence-safe splitting and window enumeration for trajectory tracks."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
import math


def contiguous_segments(
    times: Sequence[float],
    nominal_dt: float = 0.1,
    tolerance: float = 0.025,
) -> list[tuple[int, int]]:
    """Return [start, stop) segments whose adjacent timestamps match nominal_dt.

    A non-finite timestamp or any non-positive/out-of-tolerance delta breaks
    continuity. The sample at the right side of a break starts a new segment.
    """
    if nominal_dt <= 0 or tolerance < 0 or tolerance >= nominal_dt:
        raise ValueError("Require nominal_dt > 0 and 0 <= tolerance < nominal_dt.")
    if len(times) == 0:
        return []

    values = []
    for value in times:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(float("nan"))

    segments: list[tuple[int, int]] = []
    segment_start = 0

    if not math.isfinite(values[0]):
        segment_start = 1

    for index in range(1, len(values)):
        previous = values[index - 1]
        current = values[index]
        valid_pair = math.isfinite(previous) and math.isfinite(current)
        delta = current - previous if valid_pair else float("nan")

        if not valid_pair:
            if index - segment_start >= 1:
                segments.append((segment_start, index))
            segment_start = index if math.isfinite(current) else index + 1
            continue

        if delta <= 0:
            # The current row is duplicated or out of order. Drop it rather
            # than letting it become the first sample of the next segment.
            if index - segment_start >= 1:
                segments.append((segment_start, index))
            segment_start = index + 1
            continue

        if abs(delta - nominal_dt) > tolerance:
            # A positive gap breaks continuity, but the current timestamp is
            # valid and may begin a new segment.
            if index - segment_start >= 1:
                segments.append((segment_start, index))
            segment_start = index

    if len(values) - segment_start >= 1:
        segments.append((segment_start, len(values)))

    return [(start, stop) for start, stop in segments if stop > start]


def iter_window_starts(
    times: Sequence[float],
    history_steps: int = 30,
    future_steps: int = 30,
    stride: int = 1,
    nominal_dt: float = 0.1,
    tolerance: float = 0.025,
) -> Iterator[tuple[int, int, int]]:
    """Yield (segment_start, segment_stop, window_start) for complete windows."""
    if history_steps <= 0 or future_steps <= 0 or stride <= 0:
        raise ValueError("history_steps, future_steps, and stride must be positive.")
    required = history_steps + future_steps
    for segment_start, segment_stop in contiguous_segments(times, nominal_dt, tolerance):
        last_start = segment_stop - required
        for start in range(segment_start, last_start + 1, stride):
            yield segment_start, segment_stop, start


def has_complete_window(
    times: Sequence[float],
    history_steps: int = 30,
    future_steps: int = 30,
    nominal_dt: float = 0.1,
    tolerance: float = 0.025,
) -> bool:
    return next(
        iter_window_starts(
            times,
            history_steps=history_steps,
            future_steps=future_steps,
            stride=1,
            nominal_dt=nominal_dt,
            tolerance=tolerance,
        ),
        None,
    ) is not None
