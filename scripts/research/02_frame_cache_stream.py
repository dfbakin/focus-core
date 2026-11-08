#!/usr/bin/env python
"""Streaming frame cache for OUC-CGE: same output as scripts/review/01_frame_cache.py --thumbs-only,
without holding a decoded clip in memory.

01_frame_cache.py::decode() appends every decoded frame of a clip to a list, then takes the
frames at indices 0, n // 2 and n - 1 (n = number of successful cap.read() calls). The 57
padded 320-second clips of the 320p copies have 9,619 frames, about 5 GB each, which does not
fit on an 8-core / 7 GB machine. This script reads the clip with the same cv2 calls
(cv2.VideoCapture(path) with default arguments, cap.read() until it fails, cv2.setNumThreads(1))
but keeps only three frames:

  * frame 0;
  * the frame at index c = n_hint // 2, where n_hint is the expected frame count;
  * the most recent frame (which is the last one when the loop ends).

n_hint comes from the clip audit (outputs/research/clip_audit.csv, column n_frames: exact
sequential grab() count of the same 320p files) when the audit row exists and its file_bytes
equals the size of the file on disk; otherwise from a first counting pass with cap.grab().
After the read() pass the actual count n is known. If n != n_hint the clip is read a second
time with c = n // 2, so the centre index is always n // 2 of the read() count, exactly as in
01_frame_cache.py, whatever the hint was. The thumbnails are computed with the original code
(BGR -> RGB with cv2.cvtColor, cv2.resize to 32x32 with INTER_AREA, uint8).

Outputs in --out, per split (train/val/test):
  frames_{split}.npz        keys, dtypes, shapes and values of 01_frame_cache.py --thumbs-only:
                            path, label, split, cls_folder, view_id, ok, n_frames, n_reported,
                            fps, w, h, thumbs (N, 3, 32, 32, 3) uint8; no feats key.
  frames_{split}_hires.npz  path, view_id, label, split, ok, first, last; first/last are the
                            first and last decoded frames, BGR -> RGB, cv2.resize to 128x72
                            (width x height) with INTER_AREA, uint8 (N, 72, 128, 3). Zero rows
                            for unreadable clips. Written unless --no-hires.
  frames_stream_meta.csv    per clip: path, split, hint source, n_hint, passes, worker peak RSS.

Usage (repository root, CPU only):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02_frame_cache_stream.py \
      --root data/ouc-cge-320 --out outputs/review_cache --workers 8
  ... --limit 10 --out outputs/research/frame_cache_stream_limit10      # first 10 clips per split
  ... --no-audit                                                         # always count with grab()
"""
import argparse
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import SPLITS, load_all_manifests  # noqa: E402

cv2.setNumThreads(1)  # as in 01_frame_cache.py

HIRES_W, HIRES_H = 128, 72


def count_frames_grab(full_path: str) -> int:
    """Sequential grab() count (fallback hint when the audit has no matching row)."""
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return 0
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


def read_pass(full_path: str, n_hint: int):
    """One sequential cap.read() pass, as in 01_frame_cache.decode, keeping three frames only.

    Returns None if the file does not open, else dict(n_reported, fps, n, first, centre, last)
    where centre is the frame at index n_hint // 2 (None if the clip is shorter than that).
    """
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        return None
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    c = n_hint // 2
    first = centre = last = None
    k = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k == 0:
            first = f
        if k == c:
            centre = f
        last = f
        k += 1
    cap.release()
    return dict(n_reported=n_reported, fps=fps, n=k, first=first, centre=centre, last=last)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def decode(args):
    idx, full_path, n_hint, want_hires = args
    out = dict(idx=idx, ok=False, n_frames=0, n_reported=0, fps=0.0, w=0, h=0,
               hint_src="audit" if n_hint is not None else "grab", n_hint=-1, passes=0)
    if n_hint is None:
        n_hint = count_frames_grab(full_path)
    out["n_hint"] = int(n_hint)
    r = read_pass(full_path, n_hint)
    out["passes"] = 1
    if r is None:  # 01: `if not cap.isOpened(): return out` (n_reported=0, fps=0.0)
        out["rss_mb"] = peak_rss_mb()
        return out
    out["n_reported"], out["fps"] = r["n_reported"], r["fps"]  # from the first open, as in 01
    n = r["n"]
    if n == 0:  # 01: `if n == 0: return out`
        out["rss_mb"] = peak_rss_mb()
        return out
    if n != n_hint:  # wrong hint: read again with the exact count
        first_pass_n = n
        r = read_pass(full_path, n)
        out["passes"] = 2
        if r is None or r["n"] != first_pass_n:
            raise RuntimeError(f"{full_path}: read() count changed between passes "
                               f"({first_pass_n} then {None if r is None else r['n']})")
    sel = (r["first"], r["centre"], r["last"])  # indices 0, n // 2, n - 1
    out.update(ok=True, n_frames=n, h=int(r["first"].shape[0]), w=int(r["first"].shape[1]))
    thumbs = []
    for f in sel:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        thumbs.append(cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA))
    out["thumbs"] = np.stack(thumbs)
    if want_hires:
        out["first"] = cv2.resize(cv2.cvtColor(sel[0], cv2.COLOR_BGR2RGB), (HIRES_W, HIRES_H),
                                  interpolation=cv2.INTER_AREA)
        out["last"] = cv2.resize(cv2.cvtColor(sel[2], cv2.COLOR_BGR2RGB), (HIRES_W, HIRES_H),
                                 interpolation=cv2.INTER_AREA)
    out["rss_mb"] = peak_rss_mb()
    return out


def load_hints(audit: Path, root: Path, paths) -> list:
    """n_frames of the audit row of each path if its file_bytes equals the size on disk, else None."""
    if audit is None or not audit.exists():
        return [None] * len(paths)
    a = pd.read_csv(audit, usecols=["path", "readable", "n_frames", "file_bytes"])
    a = a.drop_duplicates("path").set_index("path")
    hints = []
    for p in paths:
        full = root / p
        if p not in a.index or not full.is_file():
            hints.append(None)
            continue
        row = a.loc[p]
        if pd.isna(row.file_bytes) or int(row.file_bytes) != os.path.getsize(full):
            hints.append(None)
            continue
        hints.append(int(row.n_frames))
    return hints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="decode processes (each holds at most four decoded frames)")
    ap.add_argument("--thumbs-only", action="store_true",
                    help="accepted for CLI compatibility with 01_frame_cache.py; always on here")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N clips per split")
    ap.add_argument("--audit", type=Path, default=REPO / "outputs/research/clip_audit.csv",
                    help="clip audit CSV with exact n_frames per path (fast path for the centre index)")
    ap.add_argument("--no-audit", action="store_true", help="ignore the audit, count frames with grab()")
    ap.add_argument("--no-hires", action="store_true", help="do not write frames_{split}_hires.npz")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    want_hires = not args.no_hires

    df = load_all_manifests(args.root)
    if args.limit:
        df = df.groupby("split").head(args.limit).reset_index(drop=True)
    print(f"manifest rows: {len(df)} " + " ".join(f"{s}={int((df.split == s).sum())}" for s in SPLITS), flush=True)

    hints = load_hints(None if args.no_audit else args.audit, args.root, list(df["path"]))
    n_audit = sum(h is not None for h in hints)
    print(f"frame-count hints: {n_audit} from {args.audit if not args.no_audit else '(audit disabled)'}, "
          f"{len(hints) - n_audit} by a grab() counting pass", flush=True)

    t0 = time.time()
    jobs = [(i, str(args.root / p), hints[i], want_hires) for i, p in enumerate(df["path"])]
    # longest clips first (hint, else file size as a proxy), so a long clip does not finish last
    est = [h if h is not None else (os.path.getsize(j[1]) // 2000 if os.path.isfile(j[1]) else 0)
           for h, j in zip(hints, jobs)]
    order = np.argsort(-np.asarray(est, dtype=np.int64), kind="stable")
    jobs = [jobs[i] for i in order]

    n = len(df)
    ok = np.zeros(n, dtype=bool)
    thumbs = np.zeros((n, 3, 32, 32, 3), dtype=np.uint8)
    first = np.zeros((n, HIRES_H, HIRES_W, 3), dtype=np.uint8) if want_hires else None
    last = np.zeros((n, HIRES_H, HIRES_W, 3), dtype=np.uint8) if want_hires else None
    meta_keys = ("n_frames", "n_reported", "fps", "w", "h")
    meta = {k: [None] * n for k in meta_keys}
    run = [None] * n
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(decode, jobs, chunksize=1)):
            i = r["idx"]
            ok[i] = r["ok"]
            for key in meta_keys:
                meta[key][i] = r[key]
            if r["ok"]:
                thumbs[i] = r["thumbs"]
                if want_hires:
                    first[i], last[i] = r["first"], r["last"]
            run[i] = dict(hint_src=r["hint_src"], n_hint=r["n_hint"], passes=r["passes"], rss_mb=r["rss_mb"])
            if (k + 1) % 500 == 0:
                print(f"decoded {k + 1}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    print(f"decode done in {time.time() - t0:.0f}s", flush=True)
    meta = {k: np.array(v) for k, v in meta.items()}  # same construction as 01 (np.array of python scalars)
    run = pd.DataFrame(run)
    print(f"read passes: {int((run.passes == 1).sum())} clips x1, {int((run.passes == 2).sum())} clips x2 "
          f"(hint != read() count); worker peak RSS max {run.rss_mb.max():.0f} MB", flush=True)

    print(f"unreadable clips: {int((~ok).sum())}", flush=True)
    for i in np.where(~ok)[0]:
        print("  BAD", df.path[i], df.split[i], flush=True)
    print("--thumbs-only: no ResNet50 features (02, 03 and 04 need them; 03b and 07 do not)", flush=True)

    for s in SPLITS:
        m = (df.split == s).values
        arrays = dict(
            path=df.path.values[m].astype(str), label=df.label.values[m].astype(np.int64),
            split=df.split.values[m].astype(str), cls_folder=df.cls_folder.values[m].astype(str),
            view_id=df.view_id.values[m].astype(np.int64), ok=ok[m],
            n_frames=meta["n_frames"][m].astype(np.int64), n_reported=meta["n_reported"][m].astype(np.int64),
            fps=meta["fps"][m].astype(np.float32), w=meta["w"][m].astype(np.int64), h=meta["h"][m].astype(np.int64),
            thumbs=thumbs[m],
        )
        np.savez_compressed(args.out / f"frames_{s}.npz", **arrays)
        if want_hires:
            hires = dict(path=arrays["path"], view_id=arrays["view_id"], label=arrays["label"],
                         split=arrays["split"], ok=arrays["ok"], first=first[m], last=last[m])
            np.savez_compressed(args.out / f"frames_{s}_hires.npz", **hires)
            del hires
        print(f"saved {s}: {int(m.sum())} clips, ok={int(ok[m].sum())}", flush=True)

    out_meta = pd.concat([df[["path", "split"]].reset_index(drop=True), run], axis=1)
    out_meta["n_frames"] = meta["n_frames"]
    out_meta.to_csv(args.out / "frames_stream_meta.csv", index=False)

    nf = meta["n_frames"][ok]
    print("n_frames percentiles (1,5,25,50,75,95,99):", np.percentile(nf, [1, 5, 25, 50, 75, 95, 99]).round(0))
    print("clips with reported frame count < 64 (dropped by OUCCGEDataset window filter):",
          int((meta["n_reported"] < 64).sum()), "of", n)
    print("reported != actual frame count:", int((meta["n_reported"][ok] != nf).sum()))
    print(f"total wall {time.time() - t0:.0f}s; main process peak RSS {peak_rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
