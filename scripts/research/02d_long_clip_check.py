#!/usr/bin/env python
"""Independent check of the streaming cache on clips the original 01_frame_cache.py cannot
decode on this machine (the 9,619-frame padded clips need about 5 GB each).

For each selected clip: decode with cv2.VideoCapture(path) + cap.read() until it fails (as
01_frame_cache.decode), but store the 32x32 RGB INTER_AREA thumbnail of EVERY frame (about 3 KB
per frame) plus the full first and last frame; after the loop take thumbs[0], thumbs[n // 2],
thumbs[n - 1] with n = len(list), exactly the index logic of 01 with no frame-count hint.
Compare n_frames, n_reported, fps, w, h, thumbs with frames_{split}.npz and the 128x72 first /
last frames with frames_{split}_hires.npz of --cache.

Selection: every clip with clip-audit n_frames > 9000 (the padded clips) plus --n-random
random readable clips (seed 0).

Usage:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02d_long_clip_check.py \
      --root data/ouc-cge-320 --cache outputs/review_cache --workers 4
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SPLITS = ["train", "val", "test"]
cv2.setNumThreads(1)


def ref_decode(full_path):
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return None
    n_rep, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    th, first, last = [], None, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if first is None:
            first = f
        last = f
        th.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (32, 32), interpolation=cv2.INTER_AREA))
    cap.release()
    n = len(th)
    hi = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_BGR2RGB), (128, 72), interpolation=cv2.INTER_AREA) for x in (first, last)]
    return dict(path=full_path, n_frames=n, n_reported=n_rep, fps=fps, h=first.shape[0], w=first.shape[1],
                thumbs=np.stack([th[0], th[n // 2], th[n - 1]]), first=hi[0], last=hi[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data/ouc-cge-320")
    ap.add_argument("--cache", type=Path, default=REPO / "outputs/review_cache")
    ap.add_argument("--audit", type=Path, default=REPO / "outputs/research/clip_audit.csv")
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    a = pd.read_csv(args.audit)
    a = a[a.readable.astype(bool)]
    long_ = a[a.n_frames > 9000]
    rest = a[a.n_frames <= 9000]
    rnd = rest.iloc[np.sort(np.random.default_rng(0).choice(len(rest), args.n_random, replace=False))]
    sel = pd.concat([long_, rnd])
    print(f"checking {len(long_)} long clips (n_frames>9000) + {len(rnd)} random clips", flush=True)

    # Each NpzFile[key] access decompresses the whole array, and a row of it is a view that keeps
    # the whole array alive: read every key once and keep copies of the selected rows only.
    need = set(sel.path)
    cache, hires = {}, {}
    for s in SPLITS:
        with np.load(args.cache / f"frames_{s}.npz", allow_pickle=False) as z:
            paths = z["path"]
            rows = np.where(np.isin(paths, list(need)))[0]
            arr = {k: z[k][rows] for k in ("n_frames", "n_reported", "fps", "w", "h", "thumbs")}
        with np.load(args.cache / f"frames_{s}_hires.npz", allow_pickle=False) as h:
            assert np.array_equal(h["path"], paths)
            first = h["first"][rows]
            last = h["last"][rows]
        for j, i in enumerate(rows):
            cache[paths[i]] = {k: arr[k][j] for k in arr}
            hires[paths[i]] = dict(first=first[j], last=last[j])
    assert set(cache) == need

    rows = []
    with Pool(args.workers) as pool:
        for path, r in zip(sel.path, pool.imap(ref_decode, [str(args.root / p) for p in sel.path])):
            c, h = cache[path], hires[path]
            row = dict(path=path, n_frames=r["n_frames"], long=r["n_frames"] > 9000)
            row["meta_equal"] = all(int(c[k]) == int(r[k]) for k in ("n_frames", "n_reported", "w", "h")) and \
                np.float32(r["fps"]) == c["fps"]
            row["thumbs_equal"] = np.array_equal(c["thumbs"], r["thumbs"])
            row["hires_first_equal"] = np.array_equal(h["first"], r["first"])
            row["hires_last_equal"] = np.array_equal(h["last"], r["last"])
            rows.append(row)
    df = pd.DataFrame(rows)
    out = REPO / "outputs/research/long_clip_check.csv"
    df.to_csv(out, index=False)
    for grp, g in df.groupby("long"):
        print(f"{'long (>9000 frames)' if grp else 'random':20s} n={len(g):3d} "
              + " ".join(f"{k}={int(g[k].sum())}/{len(g)}" for k in ("meta_equal", "thumbs_equal",
                                                                    "hires_first_equal", "hires_last_equal")))
    bad = df[~df[["meta_equal", "thumbs_equal", "hires_first_equal", "hires_last_equal"]].all(axis=1)]
    print("ALL IDENTICAL" if bad.empty else f"DIFFERENCES:\n{bad.to_string()}")
    print(f"-> {out}")
    sys.exit(0 if bad.empty else 1)


if __name__ == "__main__":
    main()
