#!/usr/bin/env python
"""Header audit of the raw OUC-CGE release (data/ouc-cge/videos/{high,mid,low}/viewN.mp4).

Reads container and stream headers with PyAV and demuxes only the first few
video packets of each file; nothing is decoded, except files whose header
reports <= 5 video frames, a video duration < 1 s or an average frame rate
< 5 fps ("broken" clips), which are decoded completely to count the real frames and record the first/last
decoded pts.

Files are opened read-only and never modified. Results are written
incrementally to a JSON-lines file (one record per file, so the run can be
resumed) and converted to CSV at the end.

Optional second pass (--md5): md5 of the file bytes, written incrementally to
a separate CSV.

Optional third pass (--packets): demux (no decode) every video packet of every
file and store (file, pts, dts, size, keyframe, 64-bit blake2b of the packet
bytes) in outputs/research/raw_video_packets.npz. Byte-identical encoded frames
in two files show that the files were cut (stream copy) from the same source
and overlap in time, even when their timestamps were rebased.

Run with the GPU hidden:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit.py
    CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit.py --md5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("high", "mid", "low")
N_PROBE_PACKETS = 16
BROKEN_MAX_FRAMES = 5
BROKEN_MAX_DUR_S = 1.0
SPARSE_MAX_RATE = 5.0  # also decode files whose header average rate is < 5 fps (frames spread over a long span)


def _num(x):
    """Rational/None -> float/None (JSON friendly)."""
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return str(x)


def _tags(prefix: str, md) -> dict:
    out = {}
    for k, v in dict(md or {}).items():
        out[f"{prefix}{k}"] = v
    return out


def probe(relpath: str, root: str) -> dict:
    import av  # imported in the worker

    t0 = time.time()
    path = os.path.join(root, relpath)
    cls, base = relpath.split("/")
    m = re.match(r"view(\d+)\.mp4$", base)
    rec: dict = {
        "relpath": relpath,
        "cls": cls,
        "basename": base,
        "clip_id": int(m.group(1)) if m else None,
        "error": None,
        "error_stage": None,
    }
    stage = "stat"
    try:
        rec["file_bytes"] = os.path.getsize(path)
        stage = "open"
        with av.open(path, mode="r") as c:
            stage = "container"
            rec["format_name"] = c.format.name
            rec["container_duration_s"] = c.duration / 1e6 if c.duration is not None else None
            rec["container_start_time_s"] = c.start_time / 1e6 if c.start_time is not None else None
            rec["container_bit_rate"] = c.bit_rate
            rec.update(_tags("ctag_", c.metadata))
            rec["n_streams"] = len(c.streams)
            rec["stream_types"] = ",".join(s.type for s in c.streams)
            vids = [s for s in c.streams if s.type == "video"]
            auds = [s for s in c.streams if s.type == "audio"]
            rec["n_video_streams"] = len(vids)
            rec["n_audio_streams"] = len(auds)
            rec["n_other_streams"] = len(c.streams) - len(vids) - len(auds)
            other = [s for s in c.streams if s.type not in ("video", "audio")]
            rec["other_stream_codecs"] = ",".join(
                f"{s.type}:{getattr(s.codec_context, 'name', None) if s.codec_context else None}" for s in other
            ) or None
            for j, s in enumerate(other):
                rec.update(_tags(f"o{j}tag_", s.metadata))

            # ---- audio streams
            stage = "audio"
            for i, s in enumerate(auds):
                cc = s.codec_context
                p = f"a{i}_"
                rec[p + "codec"] = cc.name if cc else None
                rec[p + "profile"] = getattr(s, "profile", None)
                rec[p + "sample_rate"] = cc.sample_rate if cc else None
                rec[p + "layout"] = cc.layout.name if cc else None
                rec[p + "channels"] = len(cc.layout.channels) if cc else None
                rec[p + "sample_fmt"] = cc.format.name if cc and cc.format else None
                rec[p + "time_base"] = str(s.time_base)
                rec[p + "frames"] = s.frames
                rec[p + "duration_ticks"] = s.duration
                rec[p + "start_time_ticks"] = s.start_time
                rec[p + "duration_s"] = (
                    float(s.duration * s.time_base) if s.duration is not None and s.time_base else None
                )
                rec[p + "bit_rate"] = cc.bit_rate if cc else None
                rec.update(_tags(p + "tag_", s.metadata))
            rec["audio_codecs"] = ",".join(str(rec.get(f"a{i}_codec")) for i in range(len(auds))) or None

            if not vids:
                rec["error"] = "no video stream"
                rec["error_stage"] = "video"
                return rec
            v = vids[0]
            stage = "video"
            cc = v.codec_context
            tb = v.time_base
            rec["v_index"] = v.index
            rec["v_codec"] = cc.name
            rec["v_profile"] = getattr(v, "profile", None)
            rec["v_width"] = cc.width
            rec["v_height"] = cc.height
            rec["v_pix_fmt"] = cc.pix_fmt
            rec["v_average_rate"] = _num(v.average_rate)
            rec["v_guessed_rate"] = _num(v.guessed_rate)
            rec["v_base_rate"] = _num(v.base_rate)
            rec["v_time_base"] = str(tb)
            rec["v_frames"] = v.frames
            rec["v_duration_ticks"] = v.duration
            rec["v_duration_s"] = float(v.duration * tb) if v.duration is not None and tb else None
            rec["v_start_time_ticks"] = v.start_time
            rec["v_start_time_s"] = float(v.start_time * tb) if v.start_time is not None and tb else None
            rec["v_bit_rate"] = cc.bit_rate
            rec.update(_tags("vtag_", v.metadata))

            # ---- first packets (demux only, no decode)
            stage = "packets"
            pts, dts, key, sizes = [], [], [], []
            for pkt in c.demux(v):
                if pkt.size == 0 and pkt.pts is None:  # flush packet
                    continue
                pts.append(pkt.pts)
                dts.append(pkt.dts)
                key.append(bool(pkt.is_keyframe))
                sizes.append(pkt.size)
                if len(pts) >= N_PROBE_PACKETS:
                    break
            rec["n_probe_packets"] = len(pts)
            if pts:
                rec["pkt0_pts"] = pts[0]
                rec["pkt0_dts"] = dts[0]
                rec["pkt0_key"] = key[0]
                rec["pkt0_pts_s"] = float(pts[0] * tb) if pts[0] is not None else None
                rec["pkt0_dts_s"] = float(dts[0] * tb) if dts[0] is not None else None
            if len(pts) >= 2:
                rec["pkt1_pts"] = pts[1]
                rec["pkt1_dts"] = dts[1]
                rec["pkt1_minus_pkt0_pts"] = (
                    pts[1] - pts[0] if pts[0] is not None and pts[1] is not None else None
                )
            good = sorted(p for p in pts if p is not None)
            if good:
                rec["probe_min_pts"] = good[0]
                rec["probe_min_pts_s"] = float(good[0] * tb)
            if len(good) >= 2:
                diffs = [b - a for a, b in zip(good, good[1:]) if b > a]
                if diffs:
                    diffs.sort()
                    rec["probe_tick_step_median"] = diffs[len(diffs) // 2]
                    rec["probe_tick_step_min"] = diffs[0]
            rec["probe_pts_reordered"] = any(
                b is not None and a is not None and b < a for a, b in zip(pts, pts[1:])
            )
            rec["probe_n_key"] = sum(key)

        # ---- broken clips: full decode (in a fresh container)
        vf = rec.get("v_frames")
        vd = rec.get("v_duration_s")
        vr = rec.get("v_average_rate")
        if (
            (vf is not None and vf <= BROKEN_MAX_FRAMES)
            or (vd is not None and vd < BROKEN_MAX_DUR_S)
            or (isinstance(vr, float) and vr < SPARSE_MAX_RATE)
        ):
            stage = "full_decode"
            rec["full_decoded"] = True
            with av.open(path, mode="r") as c2:
                v2 = c2.streams.video[0]
                v2.codec_context.thread_count = 1
                tb2 = v2.time_base
                n_pkt = 0
                pk_pts = []
                n_dec = 0
                first = last = None
                for pkt in c2.demux(v2):
                    if pkt.size:
                        n_pkt += 1
                        pk_pts.append(pkt.pts)
                    for fr in pkt.decode():
                        n_dec += 1
                        if first is None:
                            first = fr.pts
                            rec["dec_width"] = fr.width
                            rec["dec_height"] = fr.height
                        last = fr.pts
                        del fr
                rec["dec_n_packets"] = n_pkt
                rec["dec_packet_pts"] = ",".join(str(p) for p in pk_pts[:20])
                rec["dec_n_frames"] = n_dec
                rec["dec_first_pts"] = first
                rec["dec_last_pts"] = last
                rec["dec_first_pts_s"] = float(first * tb2) if first is not None else None
                rec["dec_last_pts_s"] = float(last * tb2) if last is not None else None
                # audio extent of the broken clip (headers already recorded above)
        else:
            rec["full_decoded"] = False
    except Exception as e:  # record, never crash
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["error_stage"] = stage
        rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-1]
    rec["probe_s"] = round(time.time() - t0, 4)
    return rec


def _probe_star(args):
    return probe(*args)


def md5_file(args) -> dict:
    relpath, root = args
    t0 = time.time()
    h = hashlib.md5()
    try:
        with open(os.path.join(root, relpath), "rb") as f:
            while True:
                b = f.read(8 << 20)
                if not b:
                    break
                h.update(b)
        return {"relpath": relpath, "md5": h.hexdigest(), "md5_s": round(time.time() - t0, 3), "md5_error": None}
    except Exception as e:
        return {"relpath": relpath, "md5": None, "md5_s": round(time.time() - t0, 3), "md5_error": repr(e)}


def packets_file(args) -> dict:
    import av
    import numpy as np

    relpath, root = args
    t0 = time.time()
    pts, dts, size, key, h = [], [], [], [], []
    err = None
    try:
        with av.open(os.path.join(root, relpath), mode="r") as c:
            v = c.streams.video[0]
            for pkt in c.demux(v):
                if not pkt.size:
                    continue
                pts.append(pkt.pts if pkt.pts is not None else -(2**62))
                dts.append(pkt.dts if pkt.dts is not None else -(2**62))
                size.append(pkt.size)
                key.append(bool(pkt.is_keyframe))
                h.append(int.from_bytes(hashlib.blake2b(bytes(pkt), digest_size=8).digest(), "little"))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {
        "relpath": relpath,
        "pts": np.array(pts, dtype=np.int64),
        "dts": np.array(dts, dtype=np.int64),
        "size": np.array(size, dtype=np.int32),
        "key": np.array(key, dtype=bool),
        "hash": np.array(h, dtype=np.uint64),
        "error": err,
        "s": time.time() - t0,
    }


def list_files(root: Path, only: list[str] | None) -> tuple[list[str], list[str]]:
    mp4, other = [], []
    for cls in CLASSES:
        d = root / cls
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            rel = f"{cls}/{p.name}"
            (mp4 if p.suffix == ".mp4" else other).append(rel)
    for p in sorted(root.iterdir()):
        if p.is_file():
            other.append(p.name)
    if only:
        mp4 = [r for r in mp4 if r in set(only)]
    return mp4, other


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data/ouc-cge/videos"))
    ap.add_argument("--out", default=str(ROOT / "outputs/research/raw_header_audit.csv"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="relpaths (cls/viewN.mp4) to restrict to")
    ap.add_argument("--md5", action="store_true", help="md5 pass instead of the header pass")
    ap.add_argument("--md5-only-list", help="text file of relpaths to hash (default: all)")
    ap.add_argument("--packets", action="store_true", help="per-packet hash pass (demux only)")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing partial jsonl")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files, other = list_files(root, args.only)
    print(f"[{time.strftime('%T')}] {len(files)} mp4 files under {root}; non-mp4: {other}", flush=True)

    if args.packets:
        import numpy as np

        t0 = time.time()
        rel, fidx, P, D, S, K, H, errs = [], [], [], [], [], [], [], {}
        with Pool(args.workers) as pool:
            for k, r in enumerate(pool.imap(packets_file, [(x, str(root)) for x in files], chunksize=8)):
                rel.append(r["relpath"])
                n = len(r["pts"])
                fidx.append(np.full(n, k, dtype=np.int32))
                P.append(r["pts"]); D.append(r["dts"]); S.append(r["size"]); K.append(r["key"]); H.append(r["hash"])
                if r["error"]:
                    errs[r["relpath"]] = r["error"]
                if (k + 1) % 1000 == 0:
                    print(f"[{time.strftime('%T')}] packets {k + 1}/{len(files)} {time.time() - t0:.1f}s", flush=True)
        npz = out.with_name("raw_video_packets.npz")
        np.savez(npz, relpath=np.array(rel), fidx=np.concatenate(fidx), pts=np.concatenate(P), dts=np.concatenate(D),
                 size=np.concatenate(S), key=np.concatenate(K), hash=np.concatenate(H),
                 errors=np.array([f"{a}\t{b}" for a, b in errs.items()]))
        wall = time.time() - t0
        print(f"packet pass: {len(files)} files, {sum(len(x) for x in P)} packets in {wall:.1f}s; errors {len(errs)}; wrote {npz}",
              flush=True)
        with open(out.with_name("raw_header_audit_timing.txt"), "a") as f:
            f.write(f"{time.strftime('%F %T')} packet pass files={len(files)} wall_s={wall:.1f} workers={args.workers}\n")
        return 0

    if args.md5:
        md5_out = out.with_name("raw_md5.csv")
        if args.md5_only_list:
            keep = {l.strip() for l in open(args.md5_only_list) if l.strip()}
            files = [f for f in files if f in keep]
        done = set()
        if md5_out.exists() and not args.fresh:
            import csv

            with open(md5_out) as f:
                done = {r["relpath"] for r in csv.DictReader(f)}
        todo = [f for f in files if f not in done]
        print(f"md5: {len(todo)} to do ({len(done)} already)", flush=True)
        t0 = time.time()
        nbytes = 0
        new = not md5_out.exists() or args.fresh
        with open(md5_out, "w" if new else "a") as f, Pool(args.workers) as pool:
            if new:
                f.write("relpath,md5,md5_s,md5_error\n")
            for k, r in enumerate(pool.imap_unordered(md5_file, [(x, str(root)) for x in todo], chunksize=4), 1):
                f.write(f"{r['relpath']},{r['md5'] or ''},{r['md5_s']},{r['md5_error'] or ''}\n")
                try:
                    nbytes += os.path.getsize(root / r["relpath"])
                except OSError:
                    pass
                if k % 250 == 0:
                    f.flush()
                    el = time.time() - t0
                    print(
                        f"[{time.strftime('%T')}] md5 {k}/{len(todo)} {el:.0f}s "
                        f"{nbytes / el / 1e6:.0f} MB/s eta {el / k * (len(todo) - k):.0f}s",
                        flush=True,
                    )
        print(f"md5 done in {time.time() - t0:.1f}s, {nbytes / 1e9:.2f} GB", flush=True)
        return 0

    jsonl = out.with_suffix(".jsonl")
    done = set()
    if jsonl.exists() and not args.fresh:
        with open(jsonl) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["relpath"])
                except Exception:
                    pass
    todo = [f for f in files if f not in done]
    print(f"headers: {len(todo)} to do ({len(done)} already in {jsonl.name})", flush=True)
    t0 = time.time()
    mode = "w" if args.fresh or not jsonl.exists() else "a"
    n_err = 0
    with open(jsonl, mode) as f, Pool(args.workers) as pool:
        for k, rec in enumerate(pool.imap(_probe_star, [(x, str(root)) for x in todo], chunksize=8), 1):
            f.write(json.dumps(rec, default=str) + "\n")
            n_err += rec.get("error") is not None
            if k % 500 == 0:
                f.flush()
                print(f"[{time.strftime('%T')}] {k}/{len(todo)} {time.time() - t0:.1f}s errors={n_err}", flush=True)
    wall = time.time() - t0
    print(f"header pass: {len(todo)} files in {wall:.1f}s ({n_err} errors)", flush=True)

    import pandas as pd

    df = pd.read_json(jsonl, lines=True, dtype=False)
    if args.only:
        df = df[df.relpath.isin(files)]
    df = df.drop_duplicates("relpath", keep="last").sort_values(["cls", "clip_id"])
    first = ["relpath", "cls", "basename", "clip_id", "file_bytes", "error", "error_stage"]
    cols = first + sorted(c for c in df.columns if c not in first)
    df[cols].to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {len(cols)} columns)", flush=True)
    with open(out.with_name("raw_header_audit_timing.txt"), "a") as f:
        f.write(
            f"{time.strftime('%F %T')} header pass files={len(todo)} wall_s={wall:.1f} "
            f"workers={args.workers} errors={n_err}\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
