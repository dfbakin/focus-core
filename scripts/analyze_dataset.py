"""Analyze the OUC-CGE dataset: frame-count / duration distribution, corruption.

Probes every clip referenced in the train/val/test manifests and reports:
  - label distribution per split
  - frame-count / duration / fps / resolution distributions
  - how many clips are "too short" for the sampling window (various thresholds)
  - how many clips are corrupted (cannot open, or first/last frame undecodable)

Header metadata (frame count, fps, size) is read from the container, which is
fast. Corruption is checked by actually decoding the first and last frame,
since truncated files typically fail near the end.

Usage:
    poetry run python scripts/analyze_dataset.py
    poetry run python scripts/analyze_dataset.py --workers 16 --no-decode-check
    poetry run python scripts/analyze_dataset.py --window 64
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("data/ouc-cge")
LABELS_DIR = ROOT / "data_labels"
SPLITS = ["train", "val", "test"]


def probe_video(args):
    """Probe a single video. Returns a dict of metadata + corruption flags."""
    rel_path, label, split, decode_check = args
    if not isinstance(rel_path, str):
        return {
            "split": split, "path": str(rel_path), "label": -1,
            "label_malformed": True, "exists": False, "opened": False,
            "total_frames": 0, "fps": 0.0, "width": 0, "height": 0,
            "first_frame_ok": False, "last_frame_ok": False,
        }
    video_path = ROOT / rel_path
    try:
        label_int = int(label)
    except (ValueError, TypeError):
        label_int = -1  # malformed / missing label in manifest
    info = {
        "split": split,
        "path": rel_path,
        "label": label_int,
        "label_malformed": label_int == -1,
        "exists": video_path.exists(),
        "opened": False,
        "total_frames": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "first_frame_ok": False,
        "last_frame_ok": False,
        "real_readable_frames": -1,  # only filled if decode_check and mismatch
    }
    if not info["exists"]:
        return info

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return info
    info["opened"] = True
    info["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info["fps"] = float(cap.get(cv2.CAP_PROP_FPS))
    info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if decode_check:
        ok_first, _ = cap.read()
        info["first_frame_ok"] = bool(ok_first)
        # Try to seek to the reported last frame and decode it.
        tf = info["total_frames"]
        if tf > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, tf - 1))
            ok_last, _ = cap.read()
            info["last_frame_ok"] = bool(ok_last)
    cap.release()
    return info


def percentiles(arr, ps=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    if len(arr) == 0:
        return {}
    arr = np.asarray(arr, dtype=float)
    return {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in ps}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(16, cv2.getNumThreads() or 8))
    parser.add_argument(
        "--window",
        type=int,
        default=64,
        help="Sampling window in frames used by the dataset (rate*num_frames).",
    )
    parser.add_argument(
        "--no-decode-check",
        action="store_true",
        help="Skip first/last-frame decode (faster, no corruption check).",
    )
    parser.add_argument("--out", type=str, default="docs/dataset_analysis.json")
    args = parser.parse_args()
    decode_check = not args.no_decode_check

    tasks = []
    split_rows = {}
    for split in SPLITS:
        df = pd.read_csv(LABELS_DIR / f"{split}.csv", sep=" ", names=["path", "label"])
        split_rows[split] = len(df)
        for _, row in df.iterrows():
            tasks.append((row["path"], row["label"], split, decode_check))

    print(f"Probing {len(tasks)} clips with {args.workers} workers "
          f"(decode_check={decode_check}) ...")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for info in tqdm(ex.map(probe_video, tasks, chunksize=16), total=len(tasks)):
            results.append(info)

    df = pd.DataFrame(results)

    report = {"n_total": len(df), "splits": split_rows, "window_frames": args.window}

    # ---- Integrity ----
    missing = df[~df["exists"]]
    not_opened = df[df["exists"] & ~df["opened"]]
    report["integrity"] = {
        "missing_files": int(len(missing)),
        "cannot_open": int(len(not_opened)),
        "malformed_label_rows": int(df.get("label_malformed", pd.Series(dtype=bool)).sum()),
    }
    if decode_check:
        opened = df[df["opened"]]
        bad_first = opened[~opened["first_frame_ok"]]
        bad_last = opened[~opened["last_frame_ok"]]
        report["integrity"]["first_frame_unreadable"] = int(len(bad_first))
        report["integrity"]["last_frame_unreadable"] = int(len(bad_last))

    # ---- Label distribution ----
    report["label_distribution"] = {}
    for split in SPLITS:
        sub = df[df["split"] == split]
        counts = Counter(sub["label"].tolist())
        total = len(sub)
        report["label_distribution"][split] = {
            str(k): {"count": int(v), "pct": round(100 * v / total, 2) if total else 0}
            for k, v in sorted(counts.items())
        }

    # ---- Frame / duration / fps / resolution ----
    opened = df[df["opened"] & (df["total_frames"] > 0)]
    frames = opened["total_frames"].tolist()
    fps_vals = opened["fps"].tolist()
    durations = [
        f / s for f, s in zip(opened["total_frames"], opened["fps"]) if s > 0
    ]
    report["frame_count"] = percentiles(frames)
    report["duration_sec"] = percentiles(durations)
    report["fps_values"] = {
        str(round(k, 2)): int(v) for k, v in Counter(round(f, 2) for f in fps_vals).items()
    }
    report["resolution"] = {
        f"{w}x{h}": int(v)
        for (w, h), v in Counter(zip(opened["width"], opened["height"])).items()
    }

    # ---- "Too short" analysis at several thresholds ----
    thresholds = sorted({args.window, 32, 64, 96, 128, 150, 300})
    report["too_short"] = {}
    for thr in thresholds:
        n = int((opened["total_frames"] < thr).sum())
        report["too_short"][f"<{thr}_frames"] = {
            "count": n,
            "pct": round(100 * n / len(opened), 2) if len(opened) else 0,
        }

    # ---- Combined "would be dropped" by a robust filter ----
    droppable = df[
        (~df["exists"])
        | (~df["opened"])
        | (df["total_frames"] < args.window)
    ]
    if decode_check:
        droppable = pd.concat([
            droppable,
            df[df["opened"] & (~df["first_frame_ok"] | ~df["last_frame_ok"])],
        ]).drop_duplicates(subset=["split", "path"])
    report["droppable_total"] = {
        "count": int(len(droppable)),
        "pct": round(100 * len(droppable) / len(df), 2),
    }
    report["droppable_by_split"] = {
        split: int((droppable["split"] == split).sum()) for split in SPLITS
    }

    # Save full per-clip table + JSON report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    df.to_csv(out_path.with_suffix(".csv"), index=False)

    print("\n" + "=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70)
    print(f"Saved report -> {out_path}")
    print(f"Saved per-clip table -> {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
