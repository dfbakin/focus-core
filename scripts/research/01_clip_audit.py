#!/usr/bin/env python
"""Clip audit of OUC-CGE (320p copies): existence, decodability, exact frame count, fps,
resolution, duration, file size, audio streams, container metadata, file hash.

One row per record of data_labels/{train,val,test}.csv (all records, also the malformed
`videos/anonymity/a.mp4` row without a label, which gets label NA).

Per clip (worker process, one frame in memory at a time):
  * cv2 (FFmpeg backend, 1 decode thread): CAP_PROP_FRAME_COUNT (n_reported), CAP_PROP_FPS (fps),
    exact frame count by sequential grab() until it fails (n_frames; grab() decodes every frame,
    retrieve() is only called for the first frame to read its shape), width/height of that frame.
    duration_s = n_frames / fps.
  * PyAV (demux only, no decode): container format and duration, per-stream type / codec,
    video stream frame count from the header, average / base frame rate, stream duration,
    number of video packets, first / last video pts, audio stream count and codec, the
    `encoder` tags of container and video stream.
  * md5 of the file bytes (exact-duplicate detection).

Usage (from the repository root):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/01_clip_audit.py \
      --root data/ouc-cge-320 --manifests data/ouc-cge/data_labels --out outputs/research/clip_audit.csv
  ... --sample 200 --out outputs/research/clip_audit_sample200.csv      # timing run
"""
import argparse
import hashlib
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ["train", "val", "test"]


def read_manifests(mdir: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        with open(mdir / f"{split}.csv") as fh:
            for ln, line in enumerate(fh, 1):
                parts = line.strip().split(" ")
                if not parts or not parts[0]:
                    continue
                path = parts[0]
                label = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else pd.NA
                rows.append(dict(path=path, label=label, split=split, manifest_line=ln))
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("Int64")
    return df


def md5_file(p: str) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probe(args):
    idx, full = args
    import cv2
    cv2.setNumThreads(1)
    t0 = time.time()
    out = dict(idx=idx, exists=os.path.isfile(full), readable=False, n_frames=0, n_reported=np.nan,
               fps=np.nan, width=np.nan, height=np.nan, file_bytes=np.nan, md5="", has_audio=pd.NA,
               audio_codec="", n_audio_streams=np.nan, n_streams=np.nan, container_format="",
               container_duration_s=np.nan, video_codec="", pix_fmt="", av_avg_rate=np.nan,
               av_base_rate=np.nan, av_stream_frames=np.nan, av_stream_duration_s=np.nan,
               av_n_video_packets=np.nan, av_first_pts_s=np.nan, av_last_pts_s=np.nan,
               container_encoder="", video_encoder="", cv2_error="", av_error="", probe_s=np.nan)
    if not out["exists"]:
        out["probe_s"] = time.time() - t0
        return out
    out["file_bytes"] = os.path.getsize(full)
    out["md5"] = md5_file(full)
    # --- cv2: exact frame count by sequential decode
    try:
        cap = cv2.VideoCapture(full, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if cap.isOpened():
            out["n_reported"] = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            out["fps"] = cap.get(cv2.CAP_PROP_FPS)
            n = 0
            if cap.grab():
                ok, f = cap.retrieve()
                if ok and f is not None:
                    out["height"], out["width"] = int(f.shape[0]), int(f.shape[1])
                n = 1
                while cap.grab():
                    n += 1
            out["n_frames"] = n
            out["readable"] = n > 0
            if n > 0 and np.isnan(out["width"]):
                out["width"] = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                out["height"] = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        else:
            out["cv2_error"] = "not opened"
        cap.release()
    except Exception as e:  # noqa: BLE001
        out["cv2_error"] = repr(e)[:200]
    # --- PyAV: streams and container metadata, demux only
    try:
        import av
        with av.open(full) as c:
            out["container_format"] = c.format.name
            out["container_duration_s"] = c.duration / 1e6 if c.duration is not None else np.nan
            out["container_encoder"] = c.metadata.get("encoder", "")
            out["n_streams"] = len(c.streams)
            aud = list(c.streams.audio)
            out["n_audio_streams"] = len(aud)
            out["has_audio"] = len(aud) > 0
            out["audio_codec"] = ";".join(a.codec_context.name for a in aud)
            if c.streams.video:
                v = c.streams.video[0]
                out["video_codec"] = v.codec_context.name
                out["pix_fmt"] = v.codec_context.pix_fmt or ""
                out["video_encoder"] = v.metadata.get("encoder", "")
                out["av_avg_rate"] = float(v.average_rate) if v.average_rate else np.nan
                out["av_base_rate"] = float(v.base_rate) if v.base_rate else np.nan
                out["av_stream_frames"] = v.frames
                tb = float(v.time_base)
                out["av_stream_duration_s"] = v.duration * tb if v.duration is not None else np.nan
                npk, pts = 0, []
                for pk in c.demux(v):
                    if pk.size == 0:
                        continue
                    npk += 1
                    if pk.pts is not None:
                        pts.append(pk.pts)
                out["av_n_video_packets"] = npk
                if pts:
                    out["av_first_pts_s"] = min(pts) * tb
                    out["av_last_pts_s"] = max(pts) * tb
    except Exception as e:  # noqa: BLE001
        out["av_error"] = repr(e)[:200]
    out["probe_s"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/ouc-cge-320"))
    ap.add_argument("--manifests", type=Path, default=Path("data/ouc-cge/data_labels"))
    ap.add_argument("--out", type=Path, default=Path("outputs/research/clip_audit.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sample", type=int, default=0, help="random sample of N records (seed 0)")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    df = read_manifests(args.manifests)
    if args.sample:
        df = df.sample(n=args.sample, random_state=0).reset_index(drop=True)
    print(f"records: {len(df)} " + " ".join(f"{s}={int((df.split == s).sum())}" for s in SPLITS), flush=True)
    jobs = [(i, str(args.root / p)) for i, p in enumerate(df.path)]
    # longest files first, so a long clip does not finish last on a single worker
    sizes = [os.path.getsize(j[1]) if os.path.isfile(j[1]) else 0 for j in jobs]
    jobs = [jobs[i] for i in np.argsort(sizes)[::-1]]
    t0 = time.time()
    res = [None] * len(jobs)
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(probe, jobs, chunksize=2)):
            res[r["idx"]] = r
            if (k + 1) % 500 == 0:
                print(f"probed {k + 1}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0
    r = pd.DataFrame(res).drop(columns="idx")
    out = pd.concat([df.reset_index(drop=True), r], axis=1)
    out["duration_s"] = out["n_frames"] / out["fps"]
    out.loc[~out["readable"], "duration_s"] = np.nan
    first = ["path", "label", "split", "exists", "readable", "n_frames", "fps", "width", "height",
             "duration_s", "file_bytes", "has_audio", "audio_codec"]
    out = out[first + [c for c in out.columns if c not in first]]
    out.to_csv(args.out, index=False)
    print(f"wall {wall:.1f}s for {len(out)} records ({args.workers} workers); "
          f"sum of per-clip probe time {out.probe_s.sum():.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
