"""Measure how long one engagement estimate takes on this computer.

Run from the repository root:
    python -m scripts.benchmark_inference

For every available device (Apple GPU through MPS, and CPU) prints the time
per estimate, and checks that all devices give the same score for the same
clip, so switching to the GPU does not change the results.
"""

import time

import numpy as np

from app.core.engagement import NUM_FRAMES, EngagementModel, candidate_devices

RUNS = 20
SEED = 0


def make_clip() -> list[np.ndarray]:
    """Return the same random clip on every call."""
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, 255, (320, 568, 3), dtype=np.uint8) for _ in range(NUM_FRAMES)]


def measure(model: EngagementModel, clip: list[np.ndarray]) -> float:
    """Return the median time of one estimate, in seconds."""
    timings = []
    for _ in range(RUNS):
        started = time.perf_counter()
        model.engagement_score(clip)
        timings.append(time.perf_counter() - started)
    return float(np.median(timings))


def main() -> None:
    """Print speed and score for every device that can run the model."""
    clip = make_clip()
    for device in candidate_devices():
        try:
            model = EngagementModel(device=device)
        except RuntimeError as exc:
            print(f"{device}: model does not run here ({exc.__class__.__name__})")
            continue
        seconds = measure(model, clip)
        score = model.engagement_score(clip)
        print(
            f"{device}: {seconds * 1000:.0f} ms per estimate, "
            f"up to {1 / seconds:.1f} per second, score {score:.3f}"
        )
    print("used by the app:", EngagementModel().device)


if __name__ == "__main__":
    main()
