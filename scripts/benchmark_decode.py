"""Compare clip-decoding strategies to locate the data-loading bottleneck.

Strategies:
  A. current:      per-frame cap.set(POS_FRAMES)+read, BOTH pathways (8+32 frames)
  B. single+seek:  per-frame cap.set(POS_FRAMES)+read, slow pathway only (8 frames)
  C. single+grab:  one seek to start, then grab()/retrieve() only needed frames
  D. decord:       (if installed) decord.VideoReader batched frame fetch

Run:
    poetry run python scripts/benchmark_decode.py --n 24
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/ouc-cge")
WINDOW = 64


def valid_paths(n, window=WINDOW):
    df = pd.read_csv(ROOT / "data_labels" / "train.csv", sep=" ", names=["path", "label"])
    out = []
    for p in df["path"]:
        if len(out) >= n:
            break
        vp = ROOT / str(p)
        cap = cv2.VideoCapture(str(vp))
        if cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= window + 2:
            out.append(vp)
        cap.release()
    return out


def decode_seek(vp, start, num_frames, rate):
    cap = cv2.VideoCapture(str(vp))
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + i * rate)
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def decode_grab(vp, start, num_frames, rate):
    """One seek to `start`, then sequential grab(); decode only every `rate`-th."""
    cap = cv2.VideoCapture(str(vp))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    needed = {i * rate for i in range(num_frames)}
    pos = 0
    last = max(needed)
    while pos <= last:
        if pos in needed:
            ok, f = cap.retrieve() if cap.grab() else (False, None)
            if ok:
                frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        else:
            cap.grab()
        pos += 1
    cap.release()
    return frames


def time_strategy(name, fn, paths, reps=2):
    t0 = time.perf_counter()
    total = 0
    for _ in range(reps):
        for vp in paths:
            cap = cv2.VideoCapture(str(vp))
            tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            start = np.random.randint(0, max(1, tf - WINDOW))
            total += fn(vp, start)
    dt = time.perf_counter() - t0
    nsamp = reps * len(paths)
    print(f"  {name:<28} {nsamp/dt:6.2f} samples/s   "
          f"({1000*dt/nsamp:6.1f} ms/sample, {total} frames)")
    return nsamp / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()

    print(f"Collecting {args.n} valid clips ...")
    paths = valid_paths(args.n)
    print(f"Got {len(paths)} clips. Timing decode strategies (single worker):\n")

    def A(vp, start):
        f1 = decode_seek(vp, start, 8, 8)   # slow
        f2 = decode_seek(vp, start, 32, 2)  # fast
        return len(f1) + len(f2)

    def B(vp, start):
        return len(decode_seek(vp, start, 8, 8))

    def C(vp, start):
        return len(decode_grab(vp, start, 8, 8))

    time_strategy("A current (2 pathways, seek)", A, paths)
    time_strategy("B slow-only (seek)", B, paths)
    time_strategy("C slow-only (grab/retrieve)", C, paths)

    try:
        import decord  # noqa: F401
        from decord import VideoReader, cpu

        def D(vp, start):
            vr = VideoReader(str(vp), ctx=cpu(0))
            idx = [start + i * 8 for i in range(8)]
            batch = vr.get_batch(idx)
            return batch.shape[0]

        time_strategy("D decord slow-only (batched)", D, paths)
    except Exception as e:
        print(f"  D decord: not available ({type(e).__name__})")


if __name__ == "__main__":
    main()
