"""Build a demo video from dataset clips of different engagement levels.

The clips are concatenated in the order given by --plan, so during a demo
lecture the indicator visibly changes color. All clips are resized to the
size of the first one.

Run from the repository root:
    python -m scripts.make_demo_video --data-dir ~/focus-core/data/ouc-cge/videos
    python -m scripts.make_demo_video --data-dir ... --plan high:3,mid:3,low:3,high:2
"""

import argparse
import random
from pathlib import Path

import cv2

DEFAULT_PLAN = "high:3,mid:3,low:3,high:2"
DEFAULT_OUTPUT = "demo_lecture.mp4"
DEFAULT_SEED = 42
FALLBACK_FPS = 25.0


def parse_plan(plan: str) -> list[tuple[str, int]]:
    """Turn 'high:3,low:2' into [('high', 3), ('low', 2)]."""
    steps = []
    for part in plan.split(","):
        level, count = part.split(":")
        steps.append((level.strip(), int(count)))
    return steps


def pick_clips(data_dir: Path, plan: list[tuple[str, int]], seed: int) -> list[Path]:
    """Randomly choose the requested number of clips for every step of the plan."""
    rng = random.Random(seed)
    chosen = []
    for level, count in plan:
        candidates = sorted((data_dir / level).glob("*.mp4"))
        if len(candidates) < count:
            raise SystemExit(f"Not enough clips in {data_dir / level}")
        chosen.extend(rng.sample(candidates, count))
    return chosen


def write_video(clips: list[Path], output: Path) -> int:
    """Concatenate clips into one mp4 file and return the number of frames."""
    writer = None
    size = None
    frames_written = 0
    for clip in clips:
        capture = cv2.VideoCapture(str(clip))
        if writer is None:
            fps = capture.get(cv2.CAP_PROP_FPS) or FALLBACK_FPS
            size = (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output), fourcc, fps, size)
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            frames_written += 1
        capture.release()
    if writer is not None:
        writer.release()
    return frames_written


def main() -> None:
    """Parse arguments, pick clips and write the demo video."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="folder with low/, mid/ and high/ subfolders")
    parser.add_argument("--plan", default=DEFAULT_PLAN,
                        help="levels and clip counts in order, e.g. high:3,low:2")
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    plan = parse_plan(args.plan)
    clips = pick_clips(args.data_dir.expanduser(), plan, args.seed)
    frames = write_video(clips, args.output)

    print(f"Saved {args.output} ({frames} frames) from clips:")
    for clip in clips:
        print(f"  {clip.parent.name}/{clip.name}")


if __name__ == "__main__":
    main()
