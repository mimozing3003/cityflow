# CARLA Runtime Warm-up and Frame Synchronization Fix

## Problem

The previous live loop advanced the simulator, then waited for an image using a generic FIFO dequeue. If no image arrived before timeout, it skipped the feature-engine update entirely. This could create a larger time gap between consecutive feature samples while the feature engine continued to assume a fixed 0.1 s interval. If an image did arrive, the FIFO dequeue could return an older frame while current ego telemetry was used for inference and overlay.

A second problem was that the feature engine pads incomplete history by repeating its first observed frame. The live loop previously passed this padded sequence into the GRU immediately, before collecting 30 actual 10 Hz feature observations.

## Changes

- The live loop now updates feature history on every synchronous simulation tick, irrespective of camera-image availability.
- The scene now offers `get_image_for_frame(expected_frame)`, which discards stale camera frames, returns only an exact frame match, and retains a future frame rather than pairing it with the wrong tick.
- Inference starts only after `CarlaFeatureEngine.is_history_ready()` confirms that 30 actual feature frames have been collected.
- During warm-up, the dashboard reports `WARM-UP n/30`, latency is marked as not run, and no predicted trajectory is drawn.
- Camera timeout now affects image availability only; it no longer suppresses feature updates or causes inference to consume stale image frames.

## Tests

Synthetic tests cover:
- warm-up is false after 29 observations and true after 30;
- stale camera frames are discarded;
- a future frame is held until its matching tick;
- a missing frame is never returned under a wrong timestamp.

These tests validate code-level contracts only. They do not exercise a live CARLA server, guarantee camera sensor timing on the target machine, or establish I-24/CARLA feature/target parity.

## Remaining risk

The exact output alignment and coordinate-frame interpretation of the frozen I-24 target still need source-level verification. This change fixes the runtime warm-up and timestamp pairing behavior without asserting that the underlying feature semantics are equivalent.
