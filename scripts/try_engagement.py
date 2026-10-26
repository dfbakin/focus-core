"""Manual smoke test of the engagement model.

Run from the repository root:
    python -m scripts.try_engagement               (random frames)
    python -m scripts.try_engagement path/to.mp4   (a clip from the middle)
"""

import sys

import cv2
import numpy as np

from app.core.engagement import NUM_FRAMES, EngagementModel

STRIDE = 8


def read_middle_clip(video_path: str) -> list[np.ndarray]:
    """Read NUM_FRAMES frames, every STRIDE-th, from the middle of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0, (total - NUM_FRAMES * STRIDE) // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames: list[np.ndarray] = []
    for pos in range(NUM_FRAMES * STRIDE):
        if not cap.grab():
            break
        if pos % STRIDE == 0:
            ok, frame = cap.retrieve()
            if ok:
                frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames could be read from: {video_path}")
    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])
    return frames


def main() -> None:
    """Print model outputs for random frames or for the given video."""
    model = EngagementModel()
    if len(sys.argv) > 1:
        frames = read_middle_clip(sys.argv[1])
    else:
        frames = [
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            for _ in range(NUM_FRAMES)
        ]

    probs = model.predict(frames)
    print("probs [low, mid, high]:", [round(p, 3) for p in probs.tolist()])
    print("pred class:", int(probs.argmax()))
    print("score 0..100:", round(model.engagement_score(frames), 1))


if __name__ == "__main__":
    main()
