#!/usr/bin/env python
"""Temporary dataset roots for checking 02_frame_cache_stream.py against 01_frame_cache.py.

01_frame_cache.py can only restrict the clip set with --limit (first N rows per split), so the
check builds small dataset roots in the project format with common.write_dataset_root
(data_labels/{train,val,test}.csv + a relative `videos` symlink to data/ouc-cge-320/videos):

  outputs/research/tmp_root/       30 random clips shorter than 15 s (clip audit duration_s),
                                   10 per published split, numpy seed 0, in published split.
  outputs/research/tmp_root_edge/  edge cases: the 4 clips missing from the 320p copies, the 6
                                   shortest readable clips, 2 ordinary clips per split, and the
                                   malformed `videos/anonymity/a.mp4` line (no label) appended to
                                   test.csv, as in the published manifest.
  outputs/research/tmp_root/bad_hints_audit.csv
                                   copy of the audit rows of tmp_root with n_frames shifted by
                                   -5 / +1 / +7 / -1 (cycled) to force the second read pass.

Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02c_make_tmp_roots.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import SPLITS, load_all_manifests, write_dataset_root  # noqa: E402

LINK = "../../../data/ouc-cge-320/videos"  # relative to outputs/research/<root>


def main():
    audit = pd.read_csv(REPO / "outputs/research/clip_audit.csv")
    man = load_all_manifests(REPO / "data/ouc-cge-320")
    a = man.merge(audit[["path", "readable", "n_frames", "duration_s"]], on="path", how="left")
    rng = np.random.default_rng(0)

    # --- 30 random short clips, 10 per split ------------------------------------------
    rows = []
    for s in SPLITS:
        pool = a[(a.split == s) & a.readable.fillna(False).astype(bool) & (a.duration_s < 15)]
        rows.append(pool.iloc[np.sort(rng.choice(len(pool), 10, replace=False))])
    sel = pd.concat(rows, ignore_index=True)
    root = REPO / "outputs/research/tmp_root"
    write_dataset_root(sel, root, LINK)
    sel[["path", "label", "split", "n_frames", "duration_s"]].to_csv(root / "selection.csv", index=False)
    print(f"{root}: {len(sel)} clips, duration {sel.duration_s.min():.2f}-{sel.duration_s.max():.2f} s, "
          f"n_frames {int(sel.n_frames.min())}-{int(sel.n_frames.max())}")

    bad = audit[audit.path.isin(sel.path)].copy().reset_index(drop=True)
    shifts = np.resize([-5, 1, 7, -1], len(bad))
    bad["n_frames"] = (bad["n_frames"] + shifts).clip(lower=0)
    bad.to_csv(root / "bad_hints_audit.csv", index=False)

    # --- edge cases -------------------------------------------------------------------
    missing = a[~a.readable.fillna(False).astype(bool)]
    shortest = a[a.readable.fillna(False).astype(bool)].nsmallest(6, "n_frames")
    ordinary = pd.concat([a[(a.split == s) & (a.duration_s < 15)].iloc[[0, 1]] for s in SPLITS])
    edge = pd.concat([missing, shortest, ordinary]).drop_duplicates("path")
    edge = pd.concat([edge[edge.split == s] for s in SPLITS], ignore_index=True)
    root_e = REPO / "outputs/research/tmp_root_edge"
    write_dataset_root(edge, root_e, LINK)
    with open(root_e / "data_labels" / "test.csv", "a") as fh:
        fh.write("videos/anonymity/a.mp4\n")
    edge[["path", "label", "split", "n_frames"]].to_csv(root_e / "selection.csv", index=False)
    print(f"{root_e}: {len(edge)} clips + malformed row; missing={len(missing)} "
          f"shortest n_frames={shortest.n_frames.astype(int).tolist()}")


if __name__ == "__main__":
    main()
