"""Offline spatial down-resize of OUC-CGE clips to speed up the data pipeline.

The training bottleneck is decoding full-resolution (mostly 1080p HEVC) video on
the fly: to sample 8 frames the loader decodes ~57 consecutive frames and builds
a ~200 MB float tensor per clip. Re-encoding every clip once to a small short
side (default 320) makes decode ~10-18x cheaper, shrinks the per-sample tensor,
and shrinks the whole dataset enough to live in the OS page cache.

We only change *resolution* (and drop the audio track) — the full frame
sequence and fps are preserved, so the dataset keeps doing random temporal
sampling/augmentation exactly as before. Just point training at the new root:

    python -m src.train +experiment=baseline_ouc_cge data.root=data/ouc-cge-320

Decode runs on CPU (scales across all cores); encode runs on the GPU via NVENC.
The job is resumable: clips whose output already decodes are skipped.

Usage:
    python scripts/preprocess_resize.py --workers 32
    python scripts/preprocess_resize.py --short-side 320 --encoder h264_nvenc --cq 24
"""

import argparse
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SPLITS = ["train", "val", "test"]
# Short side -> 320, preserving aspect ratio, force even dims (-2) for yuv420p.
_SCALE = "scale='if(gt(iw,ih),-2,{s})':'if(gt(iw,ih),{s},-2)'"


def collect_paths(src_root: Path) -> list[str]:
    """Unique relative clip paths referenced by any manifest."""
    paths: set[str] = set()
    for split in SPLITS:
        csv = src_root / "data_labels" / f"{split}.csv"
        df = pd.read_csv(csv, sep=" ", names=["path", "label"])
        for p in df["path"]:
            if isinstance(p, str) and p.strip():
                paths.add(p.strip())
    return sorted(paths)


def _has_frames(path: Path) -> bool:
    """True if `path` is a decodable video with >0 frames (resume check)."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return out.returncode == 0 and out.stdout.strip().isdigit() and int(out.stdout) > 0
    except Exception:
        return False


def transcode_one(task) -> tuple[str, str]:
    (rel, src_root, dst_root, short_side, encoder, cq, gop, threads,
     overwrite, max_seconds) = task
    src = Path(src_root) / rel
    dst = Path(dst_root) / rel
    if not src.exists():
        return rel, "missing_src"
    if not overwrite and _has_frames(dst):
        return rel, "skip"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    # Encoder-specific preset + quality flags (NVENC uses p1-p7/-cq; x26x use
    # named presets/-crf).
    if "nvenc" in encoder:
        quality = ["-preset", "p4", "-cq", str(cq)]
    else:
        quality = ["-preset", "veryfast", "-crf", str(cq)]
    # `-t` before `-i` stops *decoding* after max_seconds, which bounds the cost
    # of the few very long clips (up to 323 s). The sampler only needs a 64-frame
    # window, so a 30 s cap leaves >99% of clips (p99 ~14.5 s) untouched.
    limit = ["-t", str(max_seconds)] if max_seconds and max_seconds > 0 else []
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-threads", str(threads), *limit, "-i", str(src),
        "-vf", _SCALE.format(s=short_side),
        "-c:v", encoder, *quality,
        "-g", str(gop), "-pix_fmt", "yuv420p", "-an", str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return rel, "timeout"
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return rel, "fail"
    tmp.replace(dst)
    return rel, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="data/ouc-cge")
    ap.add_argument("--dst-root", default="data/ouc-cge-320")
    ap.add_argument("--short-side", type=int, default=320)
    ap.add_argument("--encoder", default="h264_nvenc")
    ap.add_argument("--cq", type=int, default=24)
    ap.add_argument("--gop", type=int, default=15)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--threads", type=int, default=2, help="ffmpeg decode threads per job")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="transcode only first N clips (0=all)")
    ap.add_argument("--max-seconds", type=int, default=30,
                    help="cap decoded duration per clip (0=full clip)")
    args = ap.parse_args()

    src_root, dst_root = Path(args.src_root), Path(args.dst_root)
    rels = collect_paths(src_root)
    if args.limit:
        rels = rels[: args.limit]
    print(f"{len(rels)} unique clips -> {dst_root} "
          f"(short_side={args.short_side}, {args.encoder} cq{args.cq} g{args.gop}, "
          f"{args.workers} workers x {args.threads} decode threads)")

    # Mirror the manifests so the dataset can use --root data/ouc-cge-320 directly.
    (dst_root / "data_labels").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        shutil.copy2(src_root / "data_labels" / f"{split}.csv",
                     dst_root / "data_labels" / f"{split}.csv")

    tasks = [
        (rel, str(src_root), str(dst_root), args.short_side, args.encoder,
         args.cq, args.gop, args.threads, args.overwrite, args.max_seconds)
        for rel in rels
    ]

    counts: dict[str, int] = {}
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(transcode_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            rel, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status in ("fail", "timeout", "missing_src"):
                failures.append(f"{status}\t{rel}")

    print("\n=== transcode summary ===")
    for k in sorted(counts):
        print(f"  {k:12s}: {counts[k]}")
    if failures:
        log = dst_root / "transcode_failures.txt"
        log.write_text("\n".join(failures))
        print(f"  {len(failures)} non-ok clips logged -> {log}")
    try:
        src_sz = sum(f.stat().st_size for f in (src_root / "videos").rglob("*.mp4"))
        dst_sz = sum(f.stat().st_size for f in (dst_root / "videos").rglob("*.mp4"))
        print(f"  size: {src_sz/1e9:.1f} GB -> {dst_sz/1e9:.1f} GB "
              f"({100*dst_sz/src_sz:.0f}% of original)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
