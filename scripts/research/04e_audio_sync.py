#!/usr/bin/env python
"""Are interleaved partners recorded at the same instant? Audio-envelope synchrony test.

Each raw camera file carries its own audio (audio packets are not byte-identical across partner
files, unlike the video packets of overlapping cuts), so simultaneity is tested on the decoded audio:
two microphones in one room recording the same speech have strongly correlated loudness envelopes
at a lag close to 0; clips from different times do not.

Pairs (numpy default_rng(0)): from every frame-segmented run (runs_frames.csv, 04d) with |period|
p >= 2 and >= 15 clips, up to 25 anchors N, and for each the partner candidates (N, N+j), j = 1..p-1,
and the same-source pair (N, N+p) (same camera, 5-10 s later, overlap ~0.3 s). Control: 600 random
same-class pairs from the same raw batch combo, and adjacent pairs (N, N+1) of period-1 runs.
Audio: first audio stream of the raw file (data/ouc-cge/videos), PyAV decode, mono mean, RMS in 10 ms
windows -> log10 envelope at 100 Hz; score = max over lags |l| <= 1 s of the Pearson correlation of
the two envelopes over their overlap (>= 2 s required).

Outputs: outputs/research/temporal/audio_sync_pairs.csv (one row per pair), audio_sync_summary.json.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04e_audio_sync.py
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "ouc-cge" / "videos"
T = REPO / "outputs" / "research" / "temporal"
MAXLAG = 100      # 1 s at 100 Hz
MINOV = 200       # 2 s


def envelope(rel):
    import av
    try:
        with av.open(str(RAW / rel)) as c:
            if not c.streams.audio:
                return rel, None
            st = c.streams.audio[0]
            sr = st.rate
            chunks = []
            for fr in c.decode(st):
                a = fr.to_ndarray().astype(np.float32)
                if a.ndim == 2 and not fr.format.is_planar:
                    a = a.reshape(-1, len(fr.layout.channels)).T
                chunks.append(a.mean(axis=0) if a.ndim == 2 else a)
        x = np.concatenate(chunks)
        w = sr // 100
        n = len(x) // w
        env = np.log10(np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1)) + 1e-5)
        return rel, env.astype(np.float32)
    except Exception:
        return rel, None


def score(a, b):
    best, lag = -2.0, 0
    for l in range(-MAXLAG, MAXLAG + 1):
        if l >= 0:
            x, y = a[l:], b
        else:
            x, y = a, b[-l:]
        n = min(len(x), len(y))
        if n < MINOV:
            continue
        x, y = x[:n], y[:n]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best:
            best, lag = r, l
    return (best if best > -2 else np.nan), lag


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    C = pd.read_csv(T / "clip_temporal.csv")
    RT = pd.read_csv(T / "runs_frames.csv")
    have = set(zip(C.cls, C.vid))
    combo = dict(zip(zip(C.cls, C.vid), C.combo))
    pairs = []
    for r in RT.itertuples():
        p = abs(int(r.period))
        if r.n_clips < 15 or p == 0:
            continue
        ids = [v for v in range(r.id_start, r.id_end + 1) if (r.cls, v) in have]
        if p == 1:
            anchors = rng.choice([v for v in ids if (r.cls, v + 1) in have and v + 1 <= r.id_end], size=min(8, len(ids) - 1), replace=False)
            pairs += [(r.cls, int(n), int(n) + 1, "adjacent_period1", p, r.id_start, 1) for n in anchors]
            continue
        cand = [v for v in ids if v + p <= r.id_end]
        anchors = rng.choice(cand, size=min(25, len(cand)), replace=False)
        for n in sorted(anchors):
            for j in range(1, p + 1):
                if (r.cls, n + j) in have:
                    pairs.append((r.cls, int(n), int(n + j), "same_source" if j == p else "partner", p, r.id_start, j))
    for cls in ("low", "mid", "high"):
        ids = C[C.cls == cls].vid.values
        a, b = rng.choice(ids, 3000), rng.choice(ids, 3000)
        sel = [(x, y) for x, y in zip(a, b) if abs(x - y) > 20 and combo[(cls, x)] == combo[(cls, y)]][:200]
        pairs += [(cls, int(x), int(y), "random_same_combo", 0, -1, 0) for x, y in sel]
    P = pd.DataFrame(pairs, columns=["cls", "a", "b", "kind", "period", "run_start", "j"])
    files = sorted({f"{c}/view{v}.mp4" for c, v in zip(P.cls, P.a)} | {f"{c}/view{v}.mp4" for c, v in zip(P.cls, P.b)})
    print(f"{len(P)} pairs, {len(files)} files to decode")
    with Pool(8) as pool:
        env = dict(pool.imap_unordered(envelope, files, chunksize=8))
    print(f"decoded in {time.time() - t0:.1f} s; missing audio {sum(v is None for v in env.values())}")
    sc = []
    for r in P.itertuples():
        ea, eb = env.get(f"{r.cls}/view{r.a}.mp4"), env.get(f"{r.cls}/view{r.b}.mp4")
        sc.append(score(ea, eb) if ea is not None and eb is not None else (np.nan, 0))
    P["env_corr"] = [s[0] for s in sc]
    P["lag_s"] = [s[1] / 100 for s in sc]
    P["phase"] = (P.a - P.run_start) % P.period.where(P.period > 0, 1)
    P.to_csv(T / "audio_sync_pairs.csv", index=False)
    rand = P[P.kind == "random_same_combo"].env_corr.dropna()
    thr = float(np.percentile(rand, 99))
    out = {"n_pairs": int(len(P)), "threshold_p99_random": round(thr, 3), "wall_s": round(time.time() - t0, 1)}
    for k, g in P.groupby("kind"):
        c = g.env_corr.dropna()
        out[k] = {"n": int(len(c)), "corr_p25_p50_p75": [round(float(np.percentile(c, q)), 3) for q in (25, 50, 75)],
                  "frac_above_thr": round(float((c > thr).mean()), 3),
                  "abs_lag_s_median_above_thr": round(float(g[g.env_corr > thr].lag_s.abs().median()), 3) if (c > thr).any() else None}
    # per run: share of partner pairs above threshold, by (phase, j)
    runs = []
    for (cls, rs, p), g in P[P.kind.isin(["partner", "same_source"])].groupby(["cls", "run_start", "period"]):
        part = g[g.kind == "partner"]
        rec = {"cls": cls, "run_start": int(rs), "period": int(p), "n_partner_pairs": int(len(part)),
               "partner_frac_sync": round(float((part.env_corr > thr).mean()), 3),
               "same_source_frac_sync": round(float((g[g.kind == 'same_source'].env_corr > thr).mean()), 3),
               "partner_corr_median": round(float(part.env_corr.median()), 3)}
        tab = part.groupby(["phase", "j"]).env_corr.apply(lambda s: (s > thr).mean())
        rec["sync_by_phase_j"] = {f"{int(a)}|{int(b)}": round(float(v), 2) for (a, b), v in tab.items()}
        runs.append(rec)
    out["runs"] = runs
    with open(T / "audio_sync_summary.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    print(pd.DataFrame(runs).drop(columns=["sync_by_phase_j"]).to_string())
    print(f"wall {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
