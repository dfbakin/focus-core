#!/usr/bin/env python
"""Tier 1 of the session reconstruction: events (recordings filmed at the same time in the same room).

Audio evidence (i)
  Every recording of 05a gets a loudness envelope on its own time axis: the 100 Hz log-RMS envelope of each
  clip (05b) is placed at recording time  origin + unit_shift + a_start + i / 100  (origin from the packet
  offsets, exact; frame-based for packetless clips and across probable merges, flagged in 05a), averaged
  where clips overlap, resampled to 20 Hz, and high-passed (minus its 5-s running mean). The first 30 ms of
  each clip (AAC priming dip, repeated at every 5-s cut) and digital-silence dropouts (log10 RMS < -4.5, 317
  clips) are masked, because periodic or flat artefacts correlate between any two recordings cut on one grid. For every pair of
  recordings the masked Pearson correlation is computed at every lag with FFTs; the score is
    r60   = max over recording-time lags |L| <= 60 s (L = 0: both recordings start together), overlap >= n_min
    rfull = max over all lags with the same overlap rule; second pass (see Events)
  Only lags whose overlap is >= 0.8 P are scanned (P = valid audio seconds of the shorter recording), so the
  statistic matches the null windows. Persistence: at the best lag, each half of the overlap must on its own
  exceed the 99th-percentile threshold of its length bin (a short common sound cannot carry a link).
  ALL pairs are scored, same class and cross class. The first run of this script used cross-class pairs as the
  null (as briefed: different class folders = different lesson phases = different times). That null contained
  real synchronous pairs (r60 0.81-0.90 over 2-7 minutes, same people seen from other angles; see 08 sec. 3),
  so cross-class pairs are candidates here, and the null uses only pairs that cannot be simultaneous:
    B  same recording, two time-disjoint windows (P s against P + 120 s, at least P + 60 s apart): same room,
       same microphone, same people, different minutes. Up to 10,000 per P bin.
    R  every recording pair with the partner's envelope reversed in time (both orders): exactly the candidates'
       lengths, gaps and lag scan, no possible synchrony.
  Threshold per P bin (edges 0/10/20/40/80/160/320/640/inf s) = max(q-th percentile of B, of R); primary
  q = 99.9, reported 99 and 99.99. Expected false links per bin = (1 - q/100) x candidate pairs in the bin.

Events
  Links above the primary threshold and persistent are merged greedily by descending null percentile, then r60 (Kruskal); the
  event time axis follows the tree (offset(b) = offset(a) + L60). A second pass adds pairs whose rfull exceeds
  the 99.9th percentile of the reversed null scanned over the full lag range (and is persistent), for recordings
  that start more than 60 s apart. A link that closes a cycle is checked
  against the tree (|residual| <= 1 s, else "cycle_conflict"). A link with r >= 0.95 between two recordings of the
  same scene cluster (27.5) is a re-encoded or re-cut copy of the same camera's footage in another batch: it
  joins the event as "duplicate" and does not count as a second camera (all 15 accepted pairs with r >= 0.95
  are same-cluster; the highest different-cluster pair is 0.938). The paper's "3 cameras per room" is NOT used to
  build events; it is reported as a check (max number of distinct recordings present at the same instant).
Structure (ii): recordings holding >= 5 clips of an interleaved ID run typed "cameras" (06 sec. 6) are column
  pairs; the structural lag is the median start-time difference of partner clips (N, N + j), 0 < |j| < |p|.
  The pair is first tested by audio in a narrow window (structural lag +- 10 s, same threshold and persistence):
  a pass is an audio link ("audio_structure_guided"). Without audio confirmation the pair is linked as a
  FALLBACK only when the recording to attach is alone and has < 30 s of valid audio (audio cannot test it);
  when both have audio and are silent at the structural lag, they are NOT linked.
Footage identity (exact or near-exact, offsets unknown): recordings that share an encoded packet (only through the
  broken export families, which 05a keeps out of the time axes) or that have a clip pair at 32x32 clip distance
  <= 3.0 (the same frame up to re-encoding) are joined into one event (tiers "packets", "frames_duplicate").
Prior (iii), FALLBACK: a recording still alone with < 30 s of valid audio is attached to the event of the nearest
  recording in ID (same class, same codec and frame size, same room layout, |ID gap| <= 12) whose own event
  tier is audio or structural (no chaining of priors); time offset unknown.
Room layout per recording is the agent's visual label from the contact sheet (05f figure
sessions_recordings_sheet.png); only 8 recordings show the rectangular-desk "checkerboard" room.

Outputs (outputs/research/sessions/): audio_pairs.csv (every recording pair: r60, rfull, lags, null percentile,
decision), audio_null.csv, audio_thresholds.csv, events.csv (one row per recording: event, tier, offset),
event_table.csv (one row per event), structural_links.csv, prior_links.csv, compare_04f_column_pairs.csv,
tier1_summary.json.
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/05c_events_audio.py [--workers 3]
"""
import argparse
import importlib
import json
import resource
import sys
import time
import warnings
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "sessions"
T = REPO / "outputs" / "research" / "temporal"
HZ_IN = 100
HZ = 20
HP_S = 5.0
MAXLAG_S = 60.0
BINS = [0, 10, 20, 40, 80, 160, 320, 640, np.inf]
N_NULL_B = 10000
QS = (99.0, 99.9, 99.99)
PRIMARY_Q = 99.9
MAX_CAMERAS = 3
JUNCTION_TOL_S = 0.5
CYCLE_TOL_S = 1.0
DUP_R = 0.95   # duplicate footage: r >= 0.95 AND same scene cluster (27.5); see 08 sec. 3
SILENCE_FLOOR = -4.5
FRAME_DUP = 3.0     # 32x32 clip distance at or below which two clips show the same frame up to re-encoding
MIN_AUDIO_S = 30.0   # below this much valid audio a recording cannot be tested by audio (fallbacks allowed)
# agent's visual labels (contact sheet of one frame per recording, 05f): rectangular desks + blackboard.
# keyed by (class, lowest clip ID of the recording) so that the labels do not depend on recording numbering
CHECKERBOARD = {("high", 2227), ("high", 2318), ("high", 2481), ("low", 543), ("low", 544), ("low", 593),
                ("mid", 12), ("mid", 13)}

ENV = {}   # recording -> (high-passed envelope, mask, tmin_s); filled in main, shared with workers by fork


def next_pow2(n):
    return 1 << int(np.ceil(np.log2(max(n, 2))))


def corr_all(x, mx, y, my):
    """Masked Pearson correlation of x[t + l] with y[t] for every integer lag l in [-(m-1), n-1].
    Returns lags (samples), r, n_overlap."""
    n, m = len(x), len(y)
    N = next_pow2(n + m)
    rf, irf = np.fft.rfft, np.fft.irfft
    FX, FX2, FMX = rf(x, N), rf(x * x, N), rf(mx, N)
    FY, FY2, FMY = rf(y, N), rf(y * y, N), rf(my, N)

    def c(A, B):
        return irf(A * np.conj(B), N)
    lags = np.arange(-(m - 1), n)
    idx = lags % N
    Nn = np.rint(c(FMX, FMY)[idx])
    Sx, Sy = c(FX, FMY)[idx], c(FMX, FY)[idx]
    Sxx, Syy, Sxy = c(FX2, FMY)[idx], c(FMX, FY2)[idx], c(FX, FY)[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy - Sx * Sy / Nn
        vx = Sxx - Sx * Sx / Nn
        vy = Syy - Sy * Sy / Nn
        r = cov / np.sqrt(vx * vy)
    bad = (Nn < 2) | (vx <= 1e-9 * np.maximum(Nn, 1)) | (vy <= 1e-9 * np.maximum(Nn, 1))
    r[bad] = np.nan
    return lags, r, Nn


def halves(x, mx, y, my, l):
    """Pearson r on the first and second half (by valid-sample count) of the overlap at lag l (samples)."""
    n, m = len(x), len(y)
    t = np.arange(max(0, -l), min(m, n - l))
    v = t[(mx[t + l] > 0) & (my[t] > 0)]
    if len(v) < 2 * 3 * HZ:
        return np.nan, np.nan
    out = []
    for part in (v[: len(v) // 2], v[len(v) // 2:]):
        a, b = x[part + l], y[part]
        out.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-9 and b.std() > 1e-9 else np.nan)
    return tuple(out)


def score(x, mx, tx, y, my, ty, full=True):
    """r60 / rfull of two envelopes with time origins tx, ty (s). L = rec_x - rec_y time offset.
    Only lags at which the overlap covers >= 80% of the shorter recording's valid audio are scanned, so that
    the statistic matches the null windows (which always overlap fully) and a short common sound near the edge
    of the lag window cannot carry a link on a small overlap."""
    P = min(mx.sum(), my.sum()) / HZ
    nmin = max(3.0, 0.8 * P) * HZ
    out = dict(P=P, r60=np.nan, L60=np.nan, n60=0.0, r_h1=np.nan, r_h2=np.nan, rfull=np.nan, Lfull=np.nan, nfull=0.0,
               rf_h1=np.nan, rf_h2=np.nan)
    if P < 3.0:
        return out
    lags, r, Nn = corr_all(x, mx, y, my)
    L = tx - ty + lags / HZ
    ok = (Nn >= nmin) & np.isfinite(r)
    w = ok & (np.abs(L) <= MAXLAG_S)
    if w.any():
        i = np.flatnonzero(w)[np.nanargmax(r[w])]
        h1, h2 = halves(x, mx, y, my, int(lags[i]))
        out.update(r60=float(r[i]), L60=float(L[i]), n60=float(Nn[i] / HZ), r_h1=h1, r_h2=h2)
    if full and ok.any():
        i = np.flatnonzero(ok)[np.nanargmax(r[ok])]
        h1, h2 = halves(x, mx, y, my, int(lags[i]))
        out.update(rfull=float(r[i]), Lfull=float(L[i]), nfull=float(Nn[i] / HZ), rf_h1=h1, rf_h2=h2)
    return out


def pair_job(args):
    a, b = args
    xa, ma, ta = ENV[a]
    xb, mb, tb = ENV[b]
    return a, b, score(xa, ma, ta, xb, mb, tb)


def reversed_job(args):
    """null R: the same pair with the partner's envelope reversed in time (same lengths, same lag scan)."""
    a, b, which = args
    xa, ma, ta = ENV[a]
    xb, mb, tb = ENV[b]
    if which == "b":
        xb, mb = xb[::-1].copy(), mb[::-1].copy()
    else:
        xa, ma = xa[::-1].copy(), ma[::-1].copy()
    s = score(xa, ma, ta, xb, mb, tb, full=True)
    return dict(kind="null_R_reversed", a=a, b=b, which=which, **{k: s[k] for k in ("P", "r60", "L60", "n60", "r_h1", "r_h2", "rfull", "rf_h1", "rf_h2")})


def window_job(args):
    """null B: window of length P (s) of recording a against a disjoint window of P + 120 s of the same recording."""
    a, sa, b, sb, P = args
    kind = "null_B_same_recording"
    xa, ma, _ = ENV[a]
    xb, mb, _ = ENV[b]
    na = int(P * HZ)
    nb = min(len(xb), na + int(2 * MAXLAG_S * HZ))
    x, mx = xa[sa: sa + na], ma[sa: sa + na]
    y, my = xb[sb: sb + nb], mb[sb: sb + nb]
    tx, ty = 0.0, -((len(y) - len(x)) / 2) / HZ
    s = score(x, mx, tx, y, my, ty, full=False)
    return dict(kind=kind, a=a, b=b, P_target=P, **{k: s[k] for k in ("P", "r60", "L60", "n60", "r_h1", "r_h2")})


def highpass(x, m):
    k = int(HP_S * HZ)
    ker = np.ones(k)
    num = np.convolve(x * m, ker, mode="same")
    den = np.convolve(m, ker, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(den > 0, num / den, 0.0)
    return (x - mu) * m


def build_envelopes(C, Z):
    rel_idx = {r: i for i, r in enumerate(Z["rel"])}
    off, env, a0, status = Z["offsets"], Z["env"], Z["a_start_s"], Z["status"]
    info = {}
    for rec, g in C.groupby("recording"):
        pieces = []
        for r in g.itertuples():
            i = rel_idx[r.rel]
            if not str(status[i]).startswith("ok") or r.no_audio:
                continue
            e = env[off[i]: off[i + 1]].astype(np.float64)
            e[:3] = np.nan                  # AAC priming dip in the first 20-30 ms of every clip
            e[e < SILENCE_FLOOR] = np.nan   # digital-silence dropouts (log10 RMS at the 1e-5 floor)
            t = r.origin_s + r.unit_shift_s + a0[i]
            pieces.append((t, e))
        if not pieces:
            info[rec] = dict(valid_s=0.0, tmin=np.nan)
            continue
        tmin = min(t for t, _ in pieces)
        tmax = max(t + len(e) / HZ_IN for t, e in pieces)
        n = int(np.ceil((tmax - tmin) * HZ)) + 1
        s, cnt = np.zeros(n), np.zeros(n)
        for t, e in pieces:
            # resample 100 Hz -> 20 Hz by averaging 5 samples (log domain), placed at the nearest 20 Hz slot
            k = len(e) // (HZ_IN // HZ)
            ee = e[: k * (HZ_IN // HZ)].reshape(k, HZ_IN // HZ)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = np.nanmean(np.where(np.isfinite(ee), ee, np.nan), axis=1) if np.isnan(ee).any() else ee.mean(1)
            j0 = int(round((t - tmin) * HZ))
            j = np.arange(j0, j0 + k)
            ok = np.isfinite(v) & (j >= 0) & (j < n)
            np.add.at(s, j[ok], v[ok])
            np.add.at(cnt, j[ok], 1)
        m = (cnt > 0).astype(np.float64)
        x = np.where(cnt > 0, s / np.maximum(cnt, 1), 0.0)
        xh = highpass(x, m)
        ENV[rec] = (xh, m, float(tmin))
        info[rec] = dict(valid_s=float(m.sum() / HZ), tmin=float(tmin), env_len_s=n / HZ, n_audio_clips=len(pieces))
    return info


def guided(a, b, Lc, halfwidth=10.0):
    """best masked r of recordings a, b at lags L (rec_a - rec_b) within Lc +- halfwidth s, overlap >= 0.8 P."""
    xa, ma, ta = ENV[a]
    xb, mb, tb = ENV[b]
    P = min(ma.sum(), mb.sum()) / HZ
    out = dict(P=P, r=np.nan, L=np.nan, r_h1=np.nan, r_h2=np.nan)
    if P < 3.0:
        return out
    lags, r, Nn = corr_all(xa, ma, xb, mb)
    L = ta - tb + lags / HZ
    w = (Nn >= max(3.0, 0.8 * P) * HZ) & np.isfinite(r) & (np.abs(L - Lc) <= halfwidth)
    if w.any():
        i = np.flatnonzero(w)[np.nanargmax(r[w])]
        h1, h2 = halves(xa, ma, xb, mb, int(lags[i]))
        out.update(r=float(r[i]), L=float(L[i]), r_h1=h1, r_h2=h2)
    return out


def bin_of(P):
    return int(np.searchsorted(BINS, P, side="right") - 1)


def concurrency(offsets, span, dup_of):
    """max number of distinct recordings (duplicates counted once) present at the same instant; each interval is
    shrunk by 0.5 s at both ends so that a junction overlap (median 0.3 s) does not count as simultaneity."""
    ev = []
    for r, o in offsets.items():
        if r in dup_of or not np.isfinite(o):
            continue
        ev += [(o + JUNCTION_TOL_S, 1), (o + span[r] - JUNCTION_TOL_S, -1)]
    ev.sort(key=lambda e: (e[0], e[1]))
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(0)
    C = pd.read_csv(OUT / "clip_recordings.csv")
    RECS = pd.read_csv(OUT / "recordings.csv").set_index("recording")
    Z = np.load(OUT / "audio_envelopes.npz", allow_pickle=False)
    Z = {k: Z[k] for k in Z.files}
    info = build_envelopes(C, Z)
    del Z
    RECS["valid_audio_s"] = [info[r]["valid_s"] for r in RECS.index]
    RECS["layout"] = ["checkerboard" if (r.cls, r.id_min) in CHECKERBOARD else "round" for r in RECS.itertuples()]
    assert (RECS.layout == "checkerboard").sum() == len(CHECKERBOARD), "layout keys do not match the recordings"
    recs = sorted(ENV)
    cls = RECS.cls.to_dict()
    span = RECS.span_s.to_dict()
    print(f"envelopes: {len(ENV)} of {len(RECS)} recordings have audio; {time.time() - t0:.1f} s", flush=True)
    pairs = [(a, b) for i, a in enumerate(recs) for b in recs[i + 1:]]
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(pair_job, pairs, chunksize=8))
        rres = list(pool.imap_unordered(reversed_job, [(a, b, w) for a, b in pairs for w in ("a", "b")], chunksize=8))
    PR = pd.DataFrame([dict(a=a, b=b, **s) for a, b, s in res]).sort_values(["a", "b"]).reset_index(drop=True)
    PR["cls_a"] = PR.a.map(cls)
    PR["cls_b"] = PR.b.map(cls)
    PR["kind"] = np.where(PR.cls_a == PR.cls_b, "same_class", "cross_class")
    NR = pd.DataFrame(rres)
    print(f"{len(PR)} pairs and {len(NR)} reversed pairs scored, {time.time() - t0:.1f} s", flush=True)

    # ---- null B: same recording, disjoint windows
    vlen = {r: len(ENV[r][0]) for r in recs}
    jobs = []
    for bi in range(len(BINS) - 1):
        lo, hi = max(BINS[bi], 3.0), min(BINS[bi + 1], 700.0)
        for _ in range(N_NULL_B):
            P = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            na, nb, gap = int(P * HZ), int(P * HZ) + int(2 * MAXLAG_S * HZ), int((P + 60) * HZ)
            pool_a = [r for r in recs if vlen[r] >= na + nb + gap and RECS.valid_audio_s[r] * HZ >= na + nb]
            if not pool_a:
                break
            a = pool_a[rng.integers(len(pool_a))]
            sa = int(rng.integers(0, vlen[a] - na - nb - gap + 1))
            sb = int(rng.integers(sa + na + gap, vlen[a] - nb + 1))
            jobs.append((a, sa, a, sb, P))      # x = earlier window, y = later window: disjoint at every lag
    with Pool(args.workers) as pool:
        bres = list(pool.imap_unordered(window_job, jobs, chunksize=64))
    NB = pd.DataFrame(bres)
    NB["kind"] = "null_B_same_recording"
    print(f"{len(NB)} null-B windows, {time.time() - t0:.1f} s", flush=True)

    NULL = pd.concat([NB[["kind", "a", "b", "P", "r60"]], NR[["kind", "a", "b", "P", "r60"]]], ignore_index=True)
    NULL["bin"] = NULL.P.map(bin_of)
    NULL.to_csv(OUT / "audio_null.csv", index=False)
    PR["bin"] = PR.P.map(bin_of)
    thr_rows, thr = [], {}
    for bi in range(len(BINS) - 1):
        gb = NB[NB.P.map(bin_of) == bi].r60.dropna().values
        gr = NR[NR.P.map(bin_of) == bi].r60.dropna().values
        gx = PR[(PR.kind == "cross_class") & (PR.bin == bi)].r60.dropna().values
        gs = PR[(PR.kind == "same_class") & (PR.bin == bi)].r60.dropna().values
        row = dict(bin=bi, P_lo=BINS[bi], P_hi=BINS[bi + 1], n_null_B=len(gb), n_null_R=len(gr),
                   n_same_class=len(gs), n_cross_class=len(gx),
                   nullB_median=float(np.median(gb)) if len(gb) else np.nan, nullB_max=float(gb.max()) if len(gb) else np.nan,
                   nullR_median=float(np.median(gr)) if len(gr) else np.nan, nullR_max=float(gr.max()) if len(gr) else np.nan,
                   cross_median=float(np.median(gx)) if len(gx) else np.nan, same_median=float(np.median(gs)) if len(gs) else np.nan)
        thr[bi] = {}
        for q in QS:
            qb = float(np.percentile(gb, q)) if len(gb) >= 100 else np.nan
            qr = float(np.percentile(gr, q)) if len(gr) >= 100 else np.nan
            row[f"nullB_q{q}"], row[f"nullR_q{q}"] = qb, qr
            t = np.nanmax([qb, qr]) if np.isfinite([qb, qr]).any() else np.nan
            row[f"thr_q{q}"] = t
            thr[bi][q] = t
            row[f"same_above_q{q}"] = int((gs > t).sum())
            row[f"cross_above_q{q}"] = int((gx > t).sum())
            row[f"expected_false_q{q}"] = (1 - q / 100) * (len(gs) + len(gx))
        thr_rows.append(row)
        gf = NR[NR.P.map(bin_of) == bi].rfull.dropna().values
        for q in QS:
            row[f"thr_full_q{q}"] = float(np.percentile(gf, q)) if len(gf) >= 100 else np.nan
    TH = pd.DataFrame(thr_rows)
    TH.to_csv(OUT / "audio_thresholds.csv", index=False)
    thr_full = {int(r.bin): {q: r[f"thr_full_q{q}"] for q in QS} for _, r in TH.iterrows()}
    # null percentile of each pair: within its bin, the larger of the two nulls' percentiles is NOT used;
    # we report the percentile in the pooled B u R null of the bin (both are valid nulls)
    pooled = {bi: np.sort(np.r_[NB[NB.P.map(bin_of) == bi].r60.dropna().values, NR[NR.P.map(bin_of) == bi].r60.dropna().values])
              for bi in range(len(BINS) - 1)}

    def null_pct(P, r):
        g = pooled[bin_of(P)]
        if not np.isfinite(r) or not len(g):
            return np.nan
        return float(np.searchsorted(g, r, side="left") / len(g) * 100)
    PR["null_pct"] = [null_pct(p, r) for p, r in zip(PR.P, PR.r60)]
    for q in QS:
        PR[f"above_q{q}"] = [bool(np.isfinite(r) and np.isfinite(thr[b][q]) and r > thr[b][q]) for b, r in zip(PR.bin, PR.r60)]
    # persistence: each half of the overlap must on its own exceed the 99th-percentile threshold of its length bin
    PR["half_thr"] = [thr[bin_of(p / 2)][99.0] for p in PR.P]
    PR["persistent"] = (PR[["r_h1", "r_h2"]].min(axis=1) > PR.half_thr)
    for q in QS:
        PR[f"accept_q{q}"] = PR[f"above_q{q}"] & PR.persistent
    PR["strength"] = np.select([PR["above_q99.99"], PR["above_q99.9"], PR["above_q99.0"]], ["strong", "primary", "weak"], "none")

    # ---- re-encoded duplicates: very high r and near-identical frames at the aligned time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    m05a = importlib.import_module("05a_recordings")
    clips_by_rec = {r: g.sort_values("t_start_s") for r, g in C.groupby("recording")}
    probe, need = {}, []
    for r in PR[PR["accept_q99.9"] & (PR.r60 >= 0.97)].itertuples():
        ga, gb = clips_by_rec[r.a], clips_by_rec[r.b]
        ca = ga.iloc[len(ga) // 2]
        tb = ca.t_start_s - r.L60          # rec_a = rec_b + L
        cover = gb[(gb.t_start_s <= tb + 1.0) & (gb.t_end_s >= tb)]
        if len(cover):
            cb = cover.iloc[int((cover.t_start_s - tb).abs().values.argmin())]
            probe[(r.a, r.b)] = (int(ca.row), int(cb.row), float(cb.t_start_s - tb))
            need += [int(ca.row), int(cb.row)]
    F, _ = m05a.load_rows(sorted(set(need)), [])
    dupd = {k: float(np.sqrt(((F[a].astype(np.float32) - F[b].astype(np.float32)) ** 2).mean())) for k, (a, b, _) in probe.items()}
    del F
    PR["dup_frame_rmse"] = [dupd.get((a, b), np.nan) for a, b in zip(PR.a, PR.b)]
    PR["dup_frame_dt"] = [probe[(a, b)][2] if (a, b) in probe else np.nan for a, b in zip(PR.a, PR.b)]

    # ---- duplicate footage (a re-encoded or re-cut copy of the same camera in another batch): at r >= 0.95 every
    # accepted pair is in the same scene cluster, and the highest different-cluster pair reaches 0.938
    c27 = RECS.c27.to_dict()
    PR["same_c27"] = [c27[a] == c27[b] for a, b in zip(PR.a, PR.b)]
    PR["r_best"] = np.where(PR["accept_q99.9"], PR.r60, np.where(
        [bool(np.isfinite(r) and np.isfinite(thr_full[bin_of(p)][PRIMARY_Q]) and r > thr_full[bin_of(p)][PRIMARY_Q]) for p, r in zip(PR.P, PR.rfull)],
        PR.rfull, np.nan))
    PR["dup_candidate"] = (PR.r_best >= DUP_R) & PR.same_c27

    # ---- events: Kruskal over primary links
    parent = {r: r for r in RECS.index}
    members = {r: {r: 0.0} for r in RECS.index}
    dup_of = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    nclips = RECS.n_clips.to_dict()

    def dup_pair(a, b):
        """(duplicate, kept): the copy with fewer clips is the duplicate (ties: the higher recording id)."""
        k = sorted((a, b), key=lambda r: (-nclips[r], r))
        return k[1], k[0]

    def union(a, b, L, how):
        ra, rb = find(a), find(b)
        if not np.isfinite(L):
            # offset unknown: the group with more known offsets keeps them, the other group's become unknown
            fin = lambda root: sum(np.isfinite(v) for v in members[root].values())  # noqa: E731
            if fin(rb) > fin(ra):
                a, b, ra, rb = b, a, rb, ra
        shift = members[ra][a] + L - members[rb][b] if np.isfinite(L) else np.nan
        merged = dict(members[ra])
        merged.update({k: (v + shift if np.isfinite(shift) else np.nan) for k, v in members[rb].items()})
        parent[rb] = ra
        members[ra] = merged
        del members[rb]
        for k in merged:
            ev_tier.setdefault(k, how)
    ev_tier = {}
    PR["decision"] = ""
    PR["cycle_residual_s"] = np.nan
    order = PR[PR["accept_q99.9"]].sort_values(["null_pct", "r60"], ascending=False).index
    for i in order:
        r = PR.loc[i]
        a, b, L = int(r.a), int(r.b), float(r.L60)
        is_dup = bool(r.dup_candidate)
        ra, rb = find(a), find(b)
        if ra == rb:
            res_ = L - (members[ra][b] - members[ra][a])
            PR.at[i, "cycle_residual_s"] = res_
            PR.at[i, "decision"] = ("cycle_ok" if abs(res_) <= CYCLE_TOL_S else "cycle_conflict") + (":duplicate" if is_dup else "")
            if is_dup:
                dup_of.setdefault(*dup_pair(a, b))
            continue
        union(a, b, L, "audio")        # same instant: off_a + t_a = off_b + t_b and t_a - t_b = L  ->  off_b = off_a + L
        if is_dup:
            dup_of.setdefault(*dup_pair(a, b))
        PR.at[i, "decision"] = "tree_link" + (":duplicate" if is_dup else "")
    # second pass: full lag range (the +-60 s scan assumes that synchronous recordings start within a minute of
    # each other; a recording that starts late is missed). Threshold: 99.9th percentile of the reversed null
    # scanned over the full lag range, same overlap rule; persistence at the full-range lag.
    nullR_full_sorted = {bi: np.sort(NR[NR.P.map(bin_of) == bi].rfull.dropna().values) for bi in range(len(BINS) - 1)}
    PR["null_pct_full"] = [float(np.searchsorted(nullR_full_sorted[bin_of(p)], r) / len(nullR_full_sorted[bin_of(p)]) * 100)
                           if np.isfinite(r) and len(nullR_full_sorted[bin_of(p)]) else np.nan for p, r in zip(PR.P, PR.rfull)]
    PR["accept_full"] = [bool(np.isfinite(r) and np.isfinite(thr_full[bin_of(p)][PRIMARY_Q]) and r > thr_full[bin_of(p)][PRIMARY_Q]
                              and np.isfinite(h1) and np.isfinite(h2) and min(h1, h2) > thr_full[bin_of(p / 2)][99.0])
                         for p, r, h1, h2 in zip(PR.P, PR.rfull, PR.rf_h1, PR.rf_h2)]
    for i in PR[PR.accept_full & ~PR["accept_q99.9"]].sort_values("rfull", ascending=False).index:
        r = PR.loc[i]
        a, b, L = int(r.a), int(r.b), float(r.Lfull)
        ra, rb = find(a), find(b)
        is_dup = bool(r.dup_candidate)
        if is_dup:
            dup_of.setdefault(*dup_pair(a, b))
        if ra == rb:
            res_ = L - (members[ra][b] - members[ra][a])
            PR.at[i, "cycle_residual_s"] = res_
            PR.at[i, "decision"] = ("full_lag:cycle_ok" if abs(res_) <= CYCLE_TOL_S else "full_lag:cycle_conflict") + (":duplicate" if is_dup else "")
            continue
        union(a, b, L, "audio")
        PR.at[i, "decision"] = "tree_link:full_lag" + (":duplicate" if is_dup else "")
    audio_roots = {find(r) for r in RECS.index if len(members[find(r)]) > 1}

    # ---- (ii) structure: columns of interleaved camera runs. The structure gives a lag (partner clips are
    # exported in 5-s lock-step), which allows an audio test in a narrow window (L_struct +- 10 s) that the
    # +-60 s scan around the recording starts misses when a recording starts late. Outcomes:
    #   audio_structure_guided   guided r above the primary threshold and persistent -> an audio link;
    #   fallback_link            no audio confirmation, and the recording to attach has < MIN_AUDIO_S of audio
    #                            (audio cannot test it) and is alone -> structural fallback;
    #   no_link:audio_silent     both have audio, no synchrony at the structural lag -> not linked (in the
    #                            period 4-6 runs this is the rule: two or three audio groups exported in lock-step)
    TY = pd.read_csv(T / "partner_tests_by_run_typed.csv")
    cam_runs = TY[TY.type.astype(str).str.startswith("cameras")]
    clip_rec = dict(zip(zip(C.cls, C.vid), C.recording))
    clip_t = dict(zip(zip(C.cls, C.vid), C.t_start_s))
    valid_audio = RECS.valid_audio_s.to_dict()
    struct_rows = []
    for run in cam_runs.itertuples():
        p = abs(int(run.period))
        g = C[(C.cls == run.cls) & (C.vid >= run.id_start) & (C.vid <= run.id_end)]
        cnt = g.recording.value_counts()
        rr = [int(x) for x in cnt[cnt >= 5].index]
        for i_, a in enumerate(rr):
            for b in rr[i_ + 1:]:
                d = []
                for v in g[g.recording == a].vid:
                    for j in range(-(p - 1), p):
                        if j and clip_rec.get((run.cls, v + j)) == b:
                            d.append(clip_t[(run.cls, v)] - clip_t[(run.cls, v + j)])
                if len(d) < 3:
                    continue
                L = float(np.median(d))      # rec_a - rec_b at the same moment (partner clips)
                same_event = find(a) == find(b)
                alone_a, alone_b = len(members[find(a)]) == 1, len(members[find(b)]) == 1
                rec = dict(run=f"{run.cls}:{run.id_start}-{run.id_end}", run_type=run.type, period=int(run.period), a=a, b=b,
                           L_struct=L, L_iqr=float(np.subtract(*np.percentile(d, [75, 25]))), n_partner_pairs=len(d),
                           a_alone=alone_a, b_alone=alone_b, same_event_before=same_event,
                           valid_audio_a=valid_audio[a], valid_audio_b=valid_audio[b])
                g_ = guided(a, b, L) if (a in ENV and b in ENV) else dict(P=0.0, r=np.nan, L=np.nan, r_h1=np.nan, r_h2=np.nan)
                rec.update({f"guided_{k}": v for k, v in g_.items()})
                ok_audio = (np.isfinite(g_["r"]) and g_["r"] > thr[bin_of(g_["P"])][PRIMARY_Q]
                            and min(g_["r_h1"], g_["r_h2"]) > thr[bin_of(g_["P"] / 2)][99.0])
                rec["guided_audio_ok"] = bool(ok_audio)
                if same_event:
                    pred = members[find(a)][b] - members[find(a)][a]
                    rec["event_offset_b_minus_a"] = pred
                    rec["struct_offset_b_minus_a"] = L
                    rec["decision"] = "same_event_already"
                elif ok_audio:
                    union(a, b, g_["L"], "audio")
                    rec["decision"] = "audio_structure_guided"
                elif (alone_b and valid_audio[b] < MIN_AUDIO_S) or (alone_a and valid_audio[a] < MIN_AUDIO_S):
                    if alone_b and valid_audio[b] < MIN_AUDIO_S:
                        union(a, b, L, "structural")
                    else:
                        union(b, a, -L, "structural")
                    rec["decision"] = "fallback_link"
                else:
                    rec["decision"] = "no_link:audio_silent_at_structural_lag"
                struct_rows.append(rec)
    ST = pd.DataFrame(struct_rows)

    # ---- footage identity (exact or near-exact, timing unknown): recordings that share an encoded packet (only
    # possible through the broken export families, whose packets 05a keeps out of the time axis), and recordings
    # with a clip pair at 32x32 clip distance <= FRAME_DUP (the same frame up to re-encoding: re-encoded copies in
    # another batch, e.g. HEVC clips without audio). Both are the same camera at the same time, so they must share
    # an event; their offset on the event axis stays unknown unless audio gave one.
    ident_rows = []
    PPK = pd.read_csv(T / "packet_pairs_offsets.csv", usecols=["cls_a", "id_a", "cls_b", "id_b"])
    pk_rec = defaultdict(int)
    for ka, kb in zip(zip(PPK.cls_a, PPK.id_a), zip(PPK.cls_b, PPK.id_b)):
        ra_, rb_ = clip_rec.get(ka), clip_rec.get(kb)
        if ra_ is not None and rb_ is not None and ra_ != rb_:
            pk_rec[(min(ra_, rb_), max(ra_, rb_))] += 1
    m05e = importlib.import_module("05e_table_folds")
    Rd = C.sort_values(["cls", "vid"]).reset_index(drop=True)
    Dm = m05e.thumb_distance_matrix(Rd.path.tolist())
    rec_arr = Rd.recording.values
    fr_rec = defaultdict(lambda: [0, np.inf])
    for i0 in range(0, len(Rd), 512):
        blk = Dm[i0:i0 + 512]
        ii, jj = np.nonzero(blk <= FRAME_DUP)
        ii = ii + i0
        for i, j in zip(ii, jj):
            if j <= i or rec_arr[i] == rec_arr[j]:
                continue
            k = (min(rec_arr[i], rec_arr[j]), max(rec_arr[i], rec_arr[j]))
            fr_rec[k][0] += 1
            fr_rec[k][1] = min(fr_rec[k][1], float(Dm[i, j]))
    del Dm
    for (a, b), n in sorted(pk_rec.items()):
        before = find(a) == find(b)
        if not before:
            union(a, b, np.nan, "packets")
        ident_rows.append(dict(a=a, b=b, kind="packets", n_clip_pairs=n, min_d32=np.nan, cls_a=cls[a], cls_b=cls[b],
                               decision="same_event_already" if before else "identity_link"))
    for (a, b), (n, dmin) in sorted(fr_rec.items()):
        before = find(a) == find(b)
        if not before:
            union(a, b, np.nan, "frames_duplicate")
        ident_rows.append(dict(a=a, b=b, kind="frames_duplicate", n_clip_pairs=n, min_d32=dmin, cls_a=cls[a], cls_b=cls[b],
                               decision="same_event_already" if before else "identity_link"))
    IDL = pd.DataFrame(ident_rows)
    IDL.to_csv(OUT / "identity_links.csv", index=False)

    # ---- (iii) prior fallback: only for recordings that audio cannot test, attached to an audio/structural event
    def fam(combo):
        return "|".join(str(combo).split("|")[:2]) if isinstance(combo, str) else ""
    prior_rows = []
    tier_now = dict(ev_tier)
    for r in RECS.index:
        if len(members[find(r)]) > 1:
            continue
        rr = RECS.loc[r]
        if rr.valid_audio_s >= MIN_AUDIO_S:
            prior_rows.append(dict(rec=r, attached_to=np.nan, id_gap=np.nan, valid_audio_s=rr.valid_audio_s,
                                   n_clips=int(rr.n_clips), decision="not_eligible:has_audio"))
            continue
        best = None
        for q in RECS.index:
            if q == r or RECS.cls[q] != rr.cls or len(members[find(q)]) < 2 or tier_now.get(q) not in ("audio", "structural", "packets", "frames_duplicate"):
                continue
            gap = min(abs(int(RECS.id_min[q]) - int(rr.id_max)), abs(int(rr.id_min) - int(RECS.id_max[q])))
            if gap > 12 or fam(RECS.combo[q]) != fam(rr.combo) or RECS.layout[q] != rr.layout:
                continue
            if best is None or gap < best[1]:
                best = (q, gap)
        if best is None:
            prior_rows.append(dict(rec=r, attached_to=np.nan, id_gap=np.nan, valid_audio_s=rr.valid_audio_s,
                                   n_clips=int(rr.n_clips), decision="no_candidate"))
            continue
        q, gap = best
        prior_rows.append(dict(rec=r, attached_to=q, id_gap=int(gap), same_c27=bool(RECS.c27[q] == rr.c27),
                               valid_audio_s=rr.valid_audio_s, n_clips=int(rr.n_clips), decision="fallback_link"))
        union(q, r, np.nan, "prior")
    PRI = pd.DataFrame(prior_rows)

    # ---- event table
    roots = sorted(members, key=lambda k: (-len(members[k]), -RECS.n_clips[list(members[k])].sum(), min(members[k])))
    ev_id, ev_off = {}, {}
    for k, root in enumerate(roots):
        offs = [o for o in members[root].values() if np.isfinite(o)]
        o0 = min(offs) if offs else 0.0
        for r, o in members[root].items():
            ev_id[r] = k
            ev_off[r] = o - o0
    E = RECS.copy()
    E["event"] = E.index.map(ev_id)
    E["event_offset_s"] = E.index.map(ev_off)
    E["event_size"] = E.event.map(E.event.value_counts())
    E["event_tier"] = [ev_tier.get(r, "none") if E.event_size[r] > 1 else "none" for r in E.index]
    E["duplicate_of_recording"] = E.index.map(dup_of)
    best_r, best_p = {}, {}
    for r in PR[PR.decision.str.startswith(("tree_link", "cycle_ok", "full_lag:cycle_ok"))].itertuples():
        is_full = "full_lag" in str(r.decision)
        rb_ = r.rfull if is_full else r.r60
        pb_ = r.null_pct_full if is_full else r.null_pct
        for x in (r.a, r.b):
            if (pb_, rb_) > (best_p.get(x, -1), best_r.get(x, -1)):
                best_p[x], best_r[x] = pb_, rb_
    E["best_audio_r"] = E.index.map(best_r)
    E["best_audio_null_pct"] = E.index.map(best_p)
    conc = {e: concurrency({q: ev_off[q] for q in E.index[E.event == e]}, span, dup_of) for e in E.event.unique()}
    E["event_max_concurrent"] = E.event.map(conc)
    # weakest tree link that holds each recording in its event (for the doubtful-decision list)
    E.to_csv(OUT / "events.csv")

    ET = E.groupby("event").agg(n_recordings=("n_clips", "size"), n_clips=("n_clips", "sum"),
                                classes=("cls", lambda s: ",".join(sorted(set(s)))),
                                tiers=("event_tier", lambda s: ",".join(sorted(set(s)))),
                                layouts=("layout", lambda s: ",".join(sorted(set(s)))),
                                max_concurrent=("event_max_concurrent", "first"),
                                n_duplicates=("duplicate_of_recording", lambda s: int(s.notna().sum())),
                                recordings=("n_clips", lambda s: ",".join(map(str, s.index))))
    ET["span_s"] = [float(np.nanmax([ev_off[r] + span[r] for r in E.index[E.event == e]])) for e in ET.index]
    tree = PR[PR.decision.str.startswith("tree_link")]
    ET["n_audio_tree_links"] = [int(((tree.a.map(ev_id) == e)).sum()) for e in ET.index]
    ET["weakest_audio_link_null_pct"] = [float(tree[tree.a.map(ev_id) == e].null_pct.min()) if (tree.a.map(ev_id) == e).any() else np.nan for e in ET.index]
    ET["n_cross_class_audio_links"] = [int(((tree.a.map(ev_id) == e) & (tree.kind == "cross_class")).sum()) for e in ET.index]
    ET.to_csv(OUT / "event_table.csv")
    PR.to_csv(OUT / "audio_pairs.csv", index=False)
    ST.to_csv(OUT / "structural_links.csv", index=False)
    PRI.to_csv(OUT / "prior_links.csv", index=False)

    # ---- consistency: packets shared inside events (impossible for different cameras)
    R = pd.read_csv(T / "packet_pairs_offsets.csv", usecols=["cls_a", "id_a", "cls_b", "id_b"])
    rec_a = [clip_rec.get(k) for k in zip(R.cls_a, R.id_a)]
    rec_b = [clip_rec.get(k) for k in zip(R.cls_b, R.id_b)]
    shared_inside = sum(1 for x, y in zip(rec_a, rec_b) if x is not None and y is not None and x != y and ev_id[x] == ev_id[y])

    # ---- comparison with the 28 synchronous different-camera column pairs of 04f
    AC4 = pd.read_csv(T / "audio_columns.csv")
    AC4 = AC4[AC4.kind == "sync_different_camera"]
    cmp_rows = []
    for r in AC4.itertuples():
        p = abs(int(r.period))
        ids = range(int(r.id_start), int(r.id_end) + 1)
        cols = {}
        for name, cj in (("a", r.col_a), ("b", r.col_b)):
            col = sorted([v for v in ids if (v - r.id_start) % p == cj and (r.cls, v) in clip_rec], reverse=r.period < 0)
            recs_col = pd.Series([clip_rec[(r.cls, v)] for v in col])
            cols[name] = (col, int(recs_col.mode().iloc[0]), float((recs_col == recs_col.mode().iloc[0]).mean()))
        (ca, ra, fa), (cb, rb, fb) = cols["a"], cols["b"]
        same_event = ev_id[ra] == ev_id[rb]
        # 04f: corr(a[t + l], b[t]) on stitched column timelines starting at each column's first clip
        # -> rec_a - rec_b = l + t_first(a) - t_first(b), valid when the column's first clip is in the recording
        pred = r.lag_s + clip_t[(r.cls, ca[0])] - clip_t[(r.cls, cb[0])]
        pr = PR[(PR.a == min(ra, rb)) & (PR.b == max(ra, rb))]
        L_dir = (float(pr.L60.iloc[0]) if ra < rb else -float(pr.L60.iloc[0])) if len(pr) else np.nan
        ev_diff = ev_off[ra] - ev_off[rb] if same_event else np.nan     # off_a - off_b = rec_b - rec_a at one instant
        cmp_rows.append(dict(cls=r.cls, run=f"{r.id_start}-{r.id_end}", col_a=r.col_a, col_b=r.col_b, score_04f=r.score,
                             lag_04f=r.lag_s, rec_a=ra, rec_b=rb, col_purity_a=fa, col_purity_b=fb, same_recording=ra == rb,
                             same_event=same_event, r60=float(pr.r60.iloc[0]) if len(pr) else np.nan,
                             L_pair_rec_a_minus_rec_b=L_dir, L_pred_from_04f=pred, abs_diff_pair_vs_04f=abs(L_dir - pred),
                             L_event_rec_a_minus_rec_b=-ev_diff if same_event else np.nan))
    CMP = pd.DataFrame(cmp_rows)
    CMP.to_csv(OUT / "compare_04f_column_pairs.csv", index=False)

    # ---- summary
    C2 = C.assign(event=C.recording.map(ev_id))
    C2["event_size"] = C2.event.map(E.groupby("event").size())
    C2["event_tier"] = C2.recording.map(E.event_tier)
    C2["event_classes"] = C2.event.map(ET.classes)
    S = {
        "n_recordings": int(len(RECS)), "n_recordings_with_audio": int(len(ENV)),
        "recordings_without_audio": {int(r): int(RECS.n_clips[r]) for r in RECS.index if r not in ENV},
        "n_pairs": int(len(PR)), "n_same_class_pairs": int((PR.kind == "same_class").sum()),
        "n_cross_class_pairs": int((PR.kind == "cross_class").sum()),
        "n_null_B": int(len(NB)), "n_null_R": int(len(NR)),
        "links_above": {str(q): {"same_class": int(PR[(PR.kind == "same_class")][f"above_q{q}"].sum()),
                                 "cross_class": int(PR[(PR.kind == "cross_class")][f"above_q{q}"].sum())} for q in QS},
        "links_full_lag_accepted_not_in_60s": int((PR.accept_full & ~PR["accept_q99.9"]).sum()),
        "null_R_full_lag_persistent_above": int(((NR.rfull > NR.P.map(lambda p: thr_full[bin_of(p)][PRIMARY_Q])) & (NR[["rf_h1", "rf_h2"]].min(axis=1) > NR.P.map(lambda p: thr_full[bin_of(p / 2)][99.0]))).sum()),
        "links_above_and_persistent": {str(q): {"same_class": int(PR[(PR.kind == "same_class")][f"accept_q{q}"].sum()),
                                                "cross_class": int(PR[(PR.kind == "cross_class")][f"accept_q{q}"].sum())} for q in QS},
        "null_persistent_and_above_q99.9": {"B": int(((NB.r60 > NB.P.map(lambda p: thr[bin_of(p)][99.9])) & (NB[["r_h1", "r_h2"]].min(axis=1) > NB.P.map(lambda p: thr[bin_of(p / 2)][99.0]))).sum()),
                                            "R": int(((NR.r60 > NR.P.map(lambda p: thr[bin_of(p)][99.9])) & (NR[["r_h1", "r_h2"]].min(axis=1) > NR.P.map(lambda p: thr[bin_of(p / 2)][99.0]))).sum())},
        "expected_false_links": {str(q): round(float(TH[f"expected_false_q{q}"].sum()), 2) for q in QS},
        "nullB_vs_nullR_q99.9_by_bin": {int(r.bin): [round(r["nullB_q99.9"], 3), round(r["nullR_q99.9"], 3)] for _, r in TH.iterrows()},
        "decisions": PR[PR.decision != ""].decision.value_counts().to_dict(),
        "cycle_residual_abs_s": PR.cycle_residual_s.dropna().abs().describe().round(3).to_dict(),
        "n_events_total": int(E.event.nunique()), "n_events_ge2": int((ET.n_recordings >= 2).sum()),
        "event_size_recordings": ET.n_recordings.value_counts().sort_index().to_dict(),
        "events_ge2_by_classes": ET[ET.n_recordings >= 2].classes.value_counts().to_dict(),
        "recordings_by_event_tier": E.event_tier.value_counts().to_dict(),
        "clips_by_event_tier": C2.event_tier.value_counts().to_dict(),
        "clips_by_event_tier_pct": (C2.event_tier.value_counts(normalize=True) * 100).round(2).to_dict(),
        "clips_in_events_ge2": int((C2.event_size > 1).sum()),
        "clips_in_cross_class_events": int(C2.event_classes.str.contains(",").sum()),
        "singleton_recordings_by_class": E[E.event_size == 1].groupby("cls").size().to_dict(),
        "singleton_clips_by_class": C2[C2.event_size == 1].groupby("cls").size().to_dict(),
        "singleton_recordings_by_audio": {"valid_audio_lt_3s": int(((E.event_size == 1) & (E.valid_audio_s < 3)).sum()),
                                          "valid_audio_3_30s": int(((E.event_size == 1) & (E.valid_audio_s >= 3) & (E.valid_audio_s < 30)).sum()),
                                          "valid_audio_ge_30s": int(((E.event_size == 1) & (E.valid_audio_s >= 30)).sum())},
        "singleton_clips_no_audio_flag": int(((C2.event_size == 1) & C2.no_audio).sum()),
        "clips_no_audio_flag_total": int(C2.no_audio.sum()),
        "check_max_concurrent_by_event": ET[ET.n_recordings >= 2].max_concurrent.value_counts().sort_index().to_dict(),
        "check_events_gt3_concurrent": ET[ET.max_concurrent > MAX_CAMERAS].index.tolist(),
        "check_events_mixed_layout": ET[ET.layouts.str.contains(",")].index.tolist(),
        "check_shared_packet_pairs_inside_events": int(shared_inside),
        "structural": ST.decision.value_counts().to_dict() if len(ST) else {},
        "structural_offset_vs_event_offset_abs_diff_s": (ST[ST.decision == "same_event_already"].event_offset_b_minus_a - ST[ST.decision == "same_event_already"].struct_offset_b_minus_a).abs().describe().round(2).to_dict() if len(ST) else {},
        "prior": PRI.decision.value_counts().to_dict() if len(PRI) else {},
        "identity_links": {f"{k[0]}:{k[1]}": int(v) for k, v in IDL.groupby(["kind", "decision"]).size().items()} if len(IDL) else {},
        "identity_links_cross_class": int((IDL.cls_a != IDL.cls_b).sum()) if len(IDL) else 0,
        "compare_04f": {"n": int(len(CMP)), "same_event": int(CMP.same_event.sum()),
                        "abs_diff_pair_vs_04f_s": CMP.abs_diff_pair_vs_04f.describe().round(2).to_dict()},
        "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb_self": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "peak_rss_mb_children": round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024, 1),
    }
    with open(OUT / "tier1_summary.json", "w") as fh:
        json.dump(S, fh, indent=1, default=str)
    pd.set_option("display.width", 250)
    print(TH.to_string())
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
