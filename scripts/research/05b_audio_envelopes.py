#!/usr/bin/env python
"""Loudness envelope of the first audio track of every raw clip (input to the Tier-1 event search).

Same envelope as 04e/04f (PyAV decode of audio stream 0, mono mean, RMS in 10 ms windows,
log10(rms + 1e-5), 100 Hz), applied to all 7,701 readable raw files instead of a sample. In addition
the container time of the first decoded audio sample is kept (a_start_s), so that the envelope can be
placed on a recording's time axis: sample i sits at container time a_start_s + i / 100, and container
time maps to recording time through the clip origin solved from the packet offsets (05a).

Output: outputs/research/sessions/audio_envelopes.npz
  rel (n,) str, a_start_s (n,), n_samples (n,), sr (n,), n_aframes (n,), status (n,) str, n_bad_packets (n,),
  offsets (n + 1,) int64 into env, env (sum n_samples,) float32
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/05b_audio_envelopes.py [--workers 3]
"""
import argparse
import resource
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "ouc-cge" / "videos"
OUT = REPO / "outputs" / "research" / "sessions"
HZ = 100


def envelope(rel):
    """Adapted from 04e_audio_sync.envelope (packed/planar handling, mono mean, 10 ms log-RMS).
    Differences: packets are decoded one by one and a packet that fails to decode is skipped (49 files
    in high 2010-2620 contain one corrupt AAC packet, which made the whole-stream decode of 04e fail);
    every decoded frame is placed at its own pts, so a skipped packet leaves a NaN gap instead of
    shifting the rest of the envelope. a_start_s = container time of the first decoded sample."""
    import av
    try:
        with av.open(str(RAW / rel)) as c:
            if not c.streams.audio:
                return rel, None, np.nan, 0, 0, "no_audio_stream", 0
            st = c.streams.audio[0]
            sr = st.rate
            pieces, n_bad = [], 0
            for pkt in c.demux(st):
                try:
                    frames = pkt.decode()
                except Exception:  # noqa: BLE001
                    n_bad += 1
                    continue
                for fr in frames:
                    a = fr.to_ndarray().astype(np.float32)
                    if a.ndim == 2 and not fr.format.is_planar:
                        a = a.reshape(-1, len(fr.layout.channels)).T
                    t = float(fr.pts * fr.time_base) if fr.pts is not None else np.nan
                    pieces.append((t, a.mean(axis=0) if a.ndim == 2 else a))
        if not pieces:
            return rel, None, np.nan, sr, 0, "no_audio_frames", n_bad
        ts = np.array([t for t, _ in pieces])
        if np.isnan(ts).any():  # no pts: concatenate in decode order
            x = np.concatenate([a for _, a in pieces])
            a0 = 0.0
        else:
            a0 = float(ts.min())
            pos = np.round((ts - a0) * sr).astype(np.int64)
            end = int(max(p + len(a) for p, (_, a) in zip(pos, pieces)))
            x = np.full(end, np.nan, np.float32)
            for p, (_, a) in zip(pos, pieces):
                x[p: p + len(a)] = a
        w = sr // HZ
        n = len(x) // w
        if n < 1:
            return rel, None, a0, sr, len(pieces), "too_short", n_bad
        blk = x[: n * w].reshape(n, w)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            env = np.log10(np.sqrt(np.nanmean(blk ** 2, axis=1)) + 1e-5).astype(np.float32)
        env[np.isnan(blk).all(1)] = np.nan
        return rel, env, a0, sr, len(pieces), "ok" if n_bad == 0 else "ok_skipped_packets", n_bad
    except Exception as e:  # noqa: BLE001
        return rel, None, np.nan, 0, 0, f"error:{type(e).__name__}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    C = pd.read_csv(REPO / "outputs" / "research" / "temporal" / "clip_temporal.csv", usecols=["rel"])
    files = sorted(C.rel)
    res = {}
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(envelope, files, chunksize=16)):
            res[r[0]] = r
            if k % 1000 == 0:
                print(f"{k}/{len(files)} {time.time() - t0:.0f} s", flush=True)
    rels = np.array(files)
    envs = [res[f][1] for f in files]
    ns = np.array([0 if e is None else len(e) for e in envs], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(ns)]
    env = np.concatenate([e for e in envs if e is not None]).astype(np.float32)
    np.savez(OUT / "audio_envelopes.npz", rel=rels, a_start_s=np.array([res[f][2] for f in files], dtype=np.float64),
             n_samples=ns, sr=np.array([res[f][3] for f in files]), n_aframes=np.array([res[f][4] for f in files]),
             status=np.array([res[f][5] for f in files]), n_bad_packets=np.array([res[f][6] for f in files]),
             offsets=offsets, env=env)
    st = pd.Series([res[f][5] for f in files]).value_counts()
    print(st.to_string())
    print(f"envelope seconds total {ns.sum() / HZ:.0f}; wall {time.time() - t0:.1f} s; "
          f"peak RSS self {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB, "
          f"children {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
