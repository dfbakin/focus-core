#!/usr/bin/env python
"""Frame distances along the clip-ID axis at 128x72 (06_temporal_structure.md, analysis A).

Inputs: outputs/review_cache/frames_{train,val,test}_hires.npz (first/last frame of every 320p clip,
RGB, 128x72, INTER_AREA) and frames_{split}.npz (metadata, via scripts/review/common.load_cache).

For every readable clip N of a class folder and k = 1..8 (skipping IDs that are missing or unreadable):
  d_k(N) = RMSE(last(N),   first(N+k))   forward continuation: N+k starts where N ends
  b_k(N) = RMSE(last(N+k), first(N))     backward continuation: N+k ends where N starts
  f_k(N) = RMSE(first(N),  first(N+k))
on the 0-255 scale, RGB (columns d1..d8, b1..b8, f1..f8) and grayscale BT.601 (gd1.., gb1.., gf1..).
Also per clip: fl = RMSE(first(N), last(N)) (RGB, gfl gray).
Reference pairs (numpy default_rng(0)): 20,000 random same-class pairs and 20,000 random cross-class
pairs, RMSE(first, first) and RMSE(last(i), first(j)), RGB and gray.

Outputs (outputs/research/temporal/): succ_dist.csv (one row per readable clip), ref_pairs.csv.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04c_successor_distances.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import load_cache  # noqa: E402

OUT = REPO / "outputs" / "research" / "temporal"
K = 8
W_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def load_hires():
    parts = [np.load(REPO / "outputs" / "review_cache" / f"frames_{s}_hires.npz", allow_pickle=True)
             for s in ("train", "val", "test")]
    return (np.concatenate([p["path"] for p in parts]).astype(str), np.concatenate([p["first"] for p in parts]),
            np.concatenate([p["last"] for p in parts]))


def rmse_pairs(A, ia, B, ib, gray=False, chunk=1024):
    """RMSE between rows A[ia] and B[ib] (uint8 images), float32, in chunks."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        x = A[ia[s:s + chunk]].astype(np.float32)
        y = B[ib[s:s + chunk]].astype(np.float32)
        if gray:
            x = x @ W_GRAY
            y = y @ W_GRAY
        out[s:s + chunk] = np.sqrt(((x - y) ** 2).reshape(len(x), -1).mean(axis=1))
    return out


def main():
    t0 = time.time()
    c = load_cache(REPO / "outputs" / "review_cache")
    hp, first, last = load_hires()
    assert (hp == c["path"]).all()
    ok = c["ok"]
    rows = np.flatnonzero(ok)
    df = pd.DataFrame({"row": rows, "path": c["path"][rows], "cls": c["cls_folder"][rows],
                       "vid": c["view_id"][rows], "label": c["label"][rows], "split": c["split"][rows],
                       "n_frames": c["n_frames"][rows]})
    df = df.sort_values(["cls", "vid"]).reset_index(drop=True)
    key = {(a, b): r for a, b, r in zip(df.cls, df.vid, df.row)}
    for k in range(1, K + 1):
        nb = np.array([key.get((a, b + k), -1) for a, b in zip(df.cls, df.vid)])
        df[f"nb{k}"] = nb
        has = nb >= 0
        src = df.row.values[has]
        dst = nb[has]
        for g, pre in ((False, ""), (True, "g")):
            for name, A, B, ia, ib in (("d", last, first, src, dst), ("b", last, first, dst, src),
                                        ("f", first, first, src, dst)):
                v = np.full(len(df), np.nan, dtype=np.float32)
                v[has] = rmse_pairs(A, ia, B, ib, gray=g)
                df[f"{pre}{name}{k}"] = v
        print(f"k={k} done ({time.time() - t0:.1f} s)", flush=True)
    r = df.row.values
    df["fl"] = rmse_pairs(first, r, last, r)
    df["gfl"] = rmse_pairs(first, r, last, r, gray=True)
    df.drop(columns=[f"nb{k}" for k in range(1, K + 1)]).to_csv(OUT / "succ_dist.csv", index=False)

    rng = np.random.default_rng(0)
    lab = c["label"]
    ref = []
    for name, same in (("random_same_class", True), ("random_cross_class", False)):
        a = rng.choice(rows, 60000)
        b = rng.choice(rows, 60000)
        m = (a != b) & ((lab[a] == lab[b]) == same)
        a, b = a[m][:20000], b[m][:20000]
        ref.append(pd.DataFrame({"pair_type": name, "a": a, "b": b,
                                 "ff": rmse_pairs(first, a, first, b), "lf": rmse_pairs(last, a, first, b),
                                 "gff": rmse_pairs(first, a, first, b, True),
                                 "glf": rmse_pairs(last, a, first, b, True)}))
    pd.concat(ref).to_csv(OUT / "ref_pairs.csv", index=False)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
