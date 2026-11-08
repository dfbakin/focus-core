#!/usr/bin/env python
"""Reference scales for frame RMSE at 128x72 (06_temporal_structure.md, analysis A).

Decodes the first frames of 300 random 320p clips (numpy default_rng(0) over the 7,701 readable
clips) with cv2 (default VideoCapture, setNumThreads(1), as in 02_frame_cache_stream.py), converts
BGR -> RGB, resizes to 128x72 with INTER_AREA (the same pipeline as the *_hires.npz cache) and records,
as float32 on the 0-255 scale:

  * rmse_lag_L     = RMSE(frame 0, frame L) for L in LAGS (RGB) and gray_lag_L (grayscale, BT.601);
  * rmse_step_t    = RMSE(frame t, frame t+1) for t = 0..STEPS-1 (RGB), to see ffmpeg's padding:
                     the 320p copies start with duplicated frames when the raw first pts > 0;
  * n_read         = number of frames read (at most max(LAGS) + 1).

Frames 0 and 1 give the "consecutive-frame floor" asked for in the brief; because frame 1 is often
a padded duplicate of frame 0, the floor is also reported for steps t >= 3 (raw-to-raw frames).

Output: outputs/research/temporal/ref_frame_lags.csv (one row per sampled clip).
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04a_reference_scales.py
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "ouc-cge-320"
OUT = REPO / "outputs" / "research" / "temporal"
LAGS = [1, 2, 3, 4, 6, 8, 10, 15, 20, 30, 45, 60, 90, 120, 150]
STEPS = 12
N_SAMPLE = 300
cv2.setNumThreads(1)


def gray(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def work(path):
    cap = cv2.VideoCapture(str(ROOT / path))
    frames = []
    while len(frames) <= max(LAGS):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (128, 72), interpolation=cv2.INTER_AREA)
        frames.append(f.astype(np.float32))
    cap.release()
    rec = {"path": path, "n_read": len(frames)}
    if not frames:
        return rec
    f0 = frames[0]
    for L in LAGS:
        if L < len(frames):
            rec[f"rmse_lag_{L}"] = float(np.sqrt(((frames[L] - f0) ** 2).mean()))
            rec[f"gray_lag_{L}"] = float(np.sqrt(((gray(frames[L]) - gray(f0)) ** 2).mean()))
    for t in range(STEPS):
        if t + 1 < len(frames):
            rec[f"rmse_step_{t}"] = float(np.sqrt(((frames[t + 1] - frames[t]) ** 2).mean()))
            rec[f"gray_step_{t}"] = float(np.sqrt(((gray(frames[t + 1]) - gray(frames[t])) ** 2).mean()))
    return rec


def main():
    sys.path.insert(0, str(REPO / "scripts" / "review"))
    from common import load_cache
    c = load_cache(REPO / "outputs" / "review_cache")
    paths = c["path"][c["ok"]]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(paths), N_SAMPLE, replace=False)
    t0 = time.time()
    with Pool(8) as pool:
        recs = pool.map(work, [paths[i] for i in sel], chunksize=4)
    df = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "ref_frame_lags.csv", index=False)
    print(f"{len(df)} clips decoded in {time.time() - t0:.1f} s")
    print(df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.round(2).to_string())


if __name__ == "__main__":
    main()
