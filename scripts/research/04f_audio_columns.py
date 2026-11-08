#!/usr/bin/env python
"""Camera columns of the multi-stream runs: are they one session recorded simultaneously?

For every frame-segmented run (runs_frames.csv, 04d) with |period| p >= 2 and >= 15 clips, the clips
are split into p columns by (ID - id_start) mod p ("sources"); each column is ordered along time (ID
ascending if p > 0, descending if p < 0). A column timeline is stitched from the clips' audio loudness
envelopes (first audio track, 100 Hz log-RMS, as in 04e): each clip contributes its envelope up to the
start of the next clip of the column, where the start offset is the exact packet offset when the two
clips share video packets and the clip's own duration otherwise. For every pair of columns of a run the
max Pearson correlation over lags |l| <= 30 s is computed on a window of up to 120 s from the middle
of the run (both columns over the same time span). Null distribution: column 0 of a run against
column 0 of a different run of the same class (default_rng(0)), same window and lag range.

A column pair is "synchronous" if its score exceeds the largest null score (0.6 as a sensitivity check);
it is "same camera" if >= 5 clip pairs across the two columns share video packets (overlapping cuts
of one recording), otherwise "different cameras". For synchronous different-camera pairs, with the
lag, every clip of one column is mapped onto the other column's timeline, and clips of the two
columns whose time windows overlap by >= 1 s are "simultaneous views" of each other.

Outputs (outputs/research/temporal/): audio_columns.csv (one row per column pair), audio_null.csv,
simultaneous_views.csv (clip pairs), audio_columns_summary.json (including test clips with a
simultaneous view in train).
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04f_audio_columns.py
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
HZ = 100
MAXLAG = 30 * HZ
WIN = 120 * HZ
MIN_OV = 20 * HZ


def envelope(rel):
    import av
    try:
        with av.open(str(RAW / rel)) as c:
            if not c.streams.audio:
                return rel, None
            st = c.streams.audio[0]
            sr = st.rate
            ch = [fr.to_ndarray().astype(np.float32).mean(0) for fr in c.decode(st)]
        x = np.concatenate(ch)
        w = sr // HZ
        n = len(x) // w
        return rel, np.log10(np.sqrt((x[: n * w].reshape(n, w) ** 2).mean(1)) + 1e-5).astype(np.float32)
    except Exception:
        return rel, None


def xcorr(a, b):
    """max over |l| <= MAXLAG of corr(a[t+l], b[t]) (normalised over the overlap); via cumulative sums."""
    best, lag = -2.0, 0
    na, nb = len(a), len(b)
    for l in range(-MAXLAG, MAXLAG + 1, 2):
        if l >= 0:
            x, y = a[l:], b
        else:
            x, y = a, b[-l:]
        n = min(len(x), len(y))
        if n < MIN_OV:
            continue
        x, y = x[:n], y[:n]
        sx, sy = x.std(), y.std()
        if sx < 1e-6 or sy < 1e-6:
            continue
        r = float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))
        if r > best:
            best, lag = r, l
    # refine at full resolution around the best lag
    for l in range(lag - 2, lag + 3):
        if abs(l) > MAXLAG:
            continue
        x, y = (a[l:], b) if l >= 0 else (a, b[-l:])
        n = min(len(x), len(y))
        if n < MIN_OV:
            continue
        x, y = x[:n], y[:n]
        if x.std() < 1e-6 or y.std() < 1e-6:
            continue
        r = float(((x - x.mean()) * (y - y.mean())).mean() / (x.std() * y.std()))
        if r > best:
            best, lag = r, l
    return (best if best > -2 else np.nan), lag


def job(args):
    key, a, b = args
    return key, xcorr(a, b)


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    C = pd.read_csv(T / "clip_temporal.csv")
    RT = pd.read_csv(T / "runs_frames.csv")
    R = pd.read_csv(T / "packet_pairs_offsets.csv")
    off = {}
    for r in R[R.kind == "partial"].itertuples():
        off[(r.cls_a, r.id_a, r.id_b)] = abs(int(r.start_b_minus_a))
    have = set(zip(C.cls, C.vid))
    dur = dict(zip(zip(C.cls, C.vid), C.dur_s))
    split = dict(zip(zip(C.cls, C.vid), C.split))
    runs = RT[(RT.period.abs() >= 2) & (RT.n_clips >= 15)].reset_index(drop=True)
    cols = []   # (run_idx, j, [ids in time order])
    for i, r in runs.iterrows():
        p = abs(int(r.period))
        ids = [v for v in range(r.id_start, r.id_end + 1) if (r.cls, v) in have]
        for j in range(p):
            col = sorted([v for v in ids if (v - r.id_start) % p == j], reverse=r.period < 0)
            if len(col) >= 3:
                cols.append((i, j, col))
    files = sorted({f"{runs.cls[i]}/view{v}.mp4" for i, _, col in cols for v in col})
    print(f"{len(runs)} runs, {len(cols)} columns, {len(files)} files")
    with Pool(8) as pool:
        env = dict(pool.imap_unordered(envelope, files, chunksize=8))
    print(f"decoded in {time.time() - t0:.1f} s")

    # stitch columns; start time of every clip on its column timeline (samples)
    stitched, starts, n_pk_steps = {}, {}, {}
    for i, j, col in cols:
        cls = runs.cls[i]
        parts, st, t, npk = [], {}, 0, 0
        for a, b in zip(col, col[1:] + [None]):
            e = env.get(f"{cls}/view{a}.mp4")
            if b is not None:
                o = off.get((cls, min(a, b), max(a, b)))
                npk += o is not None
                step = int(round((o / 30 if o is not None else dur[(cls, a)]) * HZ))
            else:
                step = int(round(dur[(cls, a)] * HZ))
            seg = e[:step] if e is not None else np.full(step, np.nan, np.float32)
            if len(seg) < step:
                seg = np.r_[seg, np.full(step - len(seg), np.nan, np.float32)]
            st[a] = t
            parts.append(seg)
            t += step
        x = np.concatenate(parts)
        x = np.where(np.isfinite(x), x, np.nanmedian(x) if np.isfinite(x).any() else 0).astype(np.float32)
        stitched[(i, j)] = x
        starts[(i, j)] = st
        n_pk_steps[(i, j)] = (npk, len(col) - 1)

    def window(x, L):
        mid = L // 2
        lo = max(0, mid - WIN // 2)
        return lo, x[lo: lo + WIN]

    tasks, meta = [], {}
    for i in range(len(runs)):
        js = sorted(j for (ii, j) in stitched if ii == i)
        for a_ in range(len(js)):
            for b_ in range(a_ + 1, len(js)):
                xa, xb = stitched[(i, js[a_])], stitched[(i, js[b_])]
                L = min(len(xa), len(xb))
                lo, wa = window(xa, L)
                _, wb = window(xb, L)
                tasks.append((("run", i, js[a_], js[b_]), wa, wb))
                meta[("run", i, js[a_], js[b_])] = lo
    # null: column 0 of run i vs column 0 of a random other run of the same class
    for i in range(len(runs)):
        others = [k for k in range(len(runs)) if k != i and runs.cls[k] == runs.cls[i] and (k, 0) in stitched]
        if not others or (i, 0) not in stitched:
            continue
        for k in rng.choice(others, size=min(3, len(others)), replace=False):
            xa, xb = stitched[(i, 0)], stitched[(int(k), 0)]
            L = min(len(xa), len(xb))
            tasks.append((("null", i, int(k), 0), window(xa, L)[1], window(xb, L)[1]))
    print(f"{len(tasks)} cross-correlations")
    with Pool(8) as pool:
        res = dict(pool.imap_unordered(job, tasks, chunksize=2))
    null = pd.DataFrame([(k[1], k[2], v[0], v[1] / HZ) for k, v in res.items() if k[0] == "null"],
                        columns=["run_a", "run_b", "score", "lag_s"])
    null.to_csv(T / "audio_null.csv", index=False)
    thr = float(np.nanmax(null.score))          # primary: the largest null score
    thr_hi = 0.6                                 # conservative sensitivity threshold
    pk_any = {(r.cls_a, r.id_a, r.id_b) for r in R[R.kind != "identical"].itertuples()} | \
             {(r.cls_a, r.id_a, r.id_b) for r in R[R.kind == "identical"].itertuples()}
    rows = []
    for k, (s, l) in res.items():
        if k[0] != "run":
            continue
        _, i, ja, jb = k
        r = runs.loc[i]
        rows.append(dict(cls=r.cls, id_start=int(r.id_start), id_end=int(r.id_end), period=int(r.period), n_clips=int(r.n_clips),
                         col_a=ja, col_b=jb, score=round(s, 3), lag_s=l / HZ, sync=bool(s > thr),
                         pk_steps_a=f"{n_pk_steps[(i, ja)][0]}/{n_pk_steps[(i, ja)][1]}",
                         pk_steps_b=f"{n_pk_steps[(i, jb)][0]}/{n_pk_steps[(i, jb)][1]}",
                         n_cross_pk=sum((r.cls, min(x, y), max(x, y)) in pk_any for x in cols_lookup(cols, i, ja)
                                        for y in cols_lookup(cols, i, jb) if abs(x - y) <= 3 * abs(int(r.period))),
                         c27_a=C.set_index(["cls", "vid"]).loc[[(r.cls, v) for v in cols_lookup(cols, i, ja)]].c27.mode().iloc[0],
                         c27_b=C.set_index(["cls", "vid"]).loc[[(r.cls, v) for v in cols_lookup(cols, i, jb)]].c27.mode().iloc[0],
                         run_i=i, lag_n=l))
    A = pd.DataFrame(rows).sort_values(["cls", "id_start", "col_a", "col_b"])
    A["sync_hi"] = A.score > thr_hi
    A["same_camera"] = A.n_cross_pk >= 5   # same camera: the columns share video packets (overlapping cuts); 1-3 pairs are run-boundary artefacts
    A["kind"] = np.select([A.sync & ~A.same_camera, A.sync & A.same_camera], ["sync_different_camera", "sync_same_camera"], "not_sync")
    # simultaneous views: clips of synchronous column pairs whose windows overlap >= 1 s
    sv = []
    for r in A[A.kind == "sync_different_camera"].itertuples():
        sa, sb = starts[(r.run_i, r.col_a)], starts[(r.run_i, r.col_b)]
        for a, ta in sa.items():
            ea = ta + dur[(r.cls, a)] * HZ
            for b, tb in sb.items():
                # corr(a[t + l], b[t]): b's time t corresponds to a's time t + l
                tb2, eb2 = tb + r.lag_n, tb + r.lag_n + dur[(r.cls, b)] * HZ
                ov = min(ea, eb2) - max(ta, tb2)
                if ov >= 1 * HZ:
                    sv.append(dict(cls=r.cls, id_a=a, id_b=b, overlap_s=ov / HZ, split_a=split[(r.cls, a)],
                                   split_b=split[(r.cls, b)], run_start=r.id_start, period=r.period, score=r.score,
                                   sync_hi=bool(r.sync_hi)))
    SV = pd.DataFrame(sv)
    A.drop(columns=["run_i", "lag_n"]).to_csv(T / "audio_columns.csv", index=False)
    SV.to_csv(T / "simultaneous_views.csv", index=False)
    te = C[C.split == "test"]
    simul = {}
    for r in SV.itertuples():
        simul.setdefault((r.cls, r.id_a), set()).add(r.split_b)
        simul.setdefault((r.cls, r.id_b), set()).add(r.split_a)
    te_keys = list(zip(te.cls, te.vid))
    has_sv = np.array([k in simul for k in te_keys])
    sv_train = np.array(["train" in simul.get(k, set()) for k in te_keys])
    simul_hi = {}
    for r in SV[SV.sync_hi].itertuples() if len(SV) else []:
        simul_hi.setdefault((r.cls, r.id_a), set()).add(r.split_b)
        simul_hi.setdefault((r.cls, r.id_b), set()).add(r.split_a)
    sv_train_hi = np.array(["train" in simul_hi.get(k, set()) for k in te_keys])
    has_sv_hi = np.array([k in simul_hi for k in te_keys])
    runs_sync = A.groupby(["cls", "id_start"]).sync.any()
    out = {"n_runs": int(len(runs)), "n_columns": len(cols), "n_column_pairs": int(len(A)),
           "null_n": int(len(null)), "null_score_p50_p95_p99_max": [round(float(np.nanpercentile(null.score, q)), 3) for q in (50, 95, 99, 100)],
           "threshold": round(thr, 3),
           "n_sync_column_pairs": int(A.sync.sum()), "n_runs_with_sync_pair": int(runs_sync.sum()),
           "column_pairs_by_kind": A.kind.value_counts().to_dict(),
           "n_runs_with_sync_different_camera": int(A[A.kind == "sync_different_camera"][["cls", "id_start"]].drop_duplicates().shape[0]),
           "n_runs_with_sync_different_camera_score_gt_0.6": int(A[(A.kind == "sync_different_camera") & A.sync_hi][["cls", "id_start"]].drop_duplicates().shape[0]),
           "clips_in_runs_with_sync_different_camera": int(A[A.kind == "sync_different_camera"][["cls", "id_start", "n_clips"]].drop_duplicates().n_clips.sum()),
           "different_camera_sync_lag_abs_s": [round(float(x), 2) for x in A[A.kind == "sync_different_camera"].lag_s.abs().describe(percentiles=[.25, .5, .75]).loc[["min", "25%", "50%", "75%", "max"]]],
           "runs_with_sync_different_camera_by_period": A[A.kind == "sync_different_camera"][["cls", "id_start", "period"]].drop_duplicates().groupby("period").size().to_dict(),
           "runs_by_period": runs.groupby("period").size().to_dict(),
           "sync_lag_abs_s": [round(float(x), 2) for x in A[A.sync].lag_s.abs().describe(percentiles=[.5]).loc[["min", "50%", "max"]]],
           "sync_pairs_same_c27": round(float((A[A.sync].c27_a == A[A.sync].c27_b).mean()), 3),
           "nonsync_pairs_same_c27": round(float((A[~A.sync].c27_a == A[~A.sync].c27_b).mean()), 3),
           "n_simultaneous_clip_pairs": int(len(SV)),
           "frac_simultaneous_pairs_different_split": round(float((SV.split_a != SV.split_b).mean()), 4) if len(SV) else None,
           "n_test": int(len(te)), "frac_test_with_simultaneous_view": round(float(has_sv.mean()), 4),
           "frac_test_with_simultaneous_view_in_train": round(float(sv_train.mean()), 4),
           "frac_test_with_simultaneous_view_score_gt_0.6": round(float(has_sv_hi.mean()), 4),
           "frac_test_with_simultaneous_view_in_train_score_gt_0.6": round(float(sv_train_hi.mean()), 4),
           "simultaneous_overlap_s": [round(float(x), 2) for x in SV.overlap_s.describe(percentiles=[.5]).loc[["min", "50%", "max"]]] if len(SV) else None,
           "simultaneous_views_per_test_clip_median": float(np.median([len([1 for r in SV.itertuples() if (r.cls, r.id_a) == k or (r.cls, r.id_b) == k]) for k in te_keys if k in simul])) if simul else None,
           "wall_s": round(time.time() - t0, 1)}
    # union with the temporal neighbours of the chains (04d, primary cut) and packet-shared footage
    L = pd.read_csv(T / "chain_links.csv")
    nb = {}
    for r in L.itertuples():
        nb.setdefault((r.cls, r.src), []).append(split[(r.cls, r.dst)])
        nb.setdefault((r.cls, r.dst), []).append(split[(r.cls, r.src)])
    temporal = np.array(["train" in nb.get(k, []) for k in te_keys])
    pk = (te.test_cov_train.fillna(0).values > 0)
    pk_all = (te.test_cov_train.fillna(0).values >= 1)
    out["union"] = {"temporal_neighbour_in_train": round(float(temporal.mean()), 4),
                    "simultaneous_view_in_train": round(float(sv_train.mean()), 4),
                    "temporal_or_simultaneous": round(float((temporal | sv_train).mean()), 4),
                    "packet_shared_with_train": round(float(pk.mean()), 4),
                    "packet_entirely_in_train": round(float(pk_all.mean()), 4),
                    "temporal_or_simultaneous_or_packet": round(float((temporal | sv_train | pk).mean()), 4),
                    "none_of_the_three": int((~(temporal | sv_train | pk)).sum())}
    with open(T / "audio_columns_summary.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))
    print(A.drop(columns=["run_i", "lag_n"]).to_string())


def cols_lookup(cols, i, j):
    for ii, jj, col in cols:
        if ii == i and jj == j:
            return col
    return []


if __name__ == "__main__":
    main()
