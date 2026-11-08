#!/usr/bin/env python
"""Tier 0 of the session reconstruction: recordings (one camera's continuous footage in one class folder).

Evidence, from strong to weak:
  exact     two files share a byte-identical encoded video packet (>= 1 KB). Every one of the 16,037 sharing
            pairs has a single frame offset (04b), so the packet graph fixes the start of every clip on a
            common time axis: origin(b) = origin(a) + mode(pts_a - pts_b) / 15360 s. A connected component of
            the packet graph is an "exact unit"; clips without any shared packet (HEVC, h264 1064x598, ...)
            are exact units of their own. Containment and re-cut pairs at distant IDs are included: they are the
            same encoded footage.
  probable  a junction between the end of one unit and the start of another (frames only). A candidate
            tail(A) -> head(B) must satisfy all of: same class; same raw batch combo; same frame run of
            04d (runs_frames.csv) and, for runs with |period| >= 2, an ID step that is a multiple of the period
            (same camera column); ID step sign equal to the run's time direction (sign of its period; the
            unit's own packet-link direction where the run has period 0); same scene cluster at 27.5;
            |ID step| <= 3 * max(|period|, 1) + 2; and RMSE(last(tail), first(head)) at 128x72 <= CUT.
            CUT = 38.6 = p95 of the RMSE at the 6,478 packet-verified junctions of 04d (equal to the median
            RMSE after a 3-5 s lag, 06 sec. 2): a merge is never further apart than 95% of real junctions.
            Candidates are accepted greedily by ascending RMSE, one successor and one predecessor per unit,
            no cycles. merge_conf = share of packet-verified junctions whose RMSE is >= the merge's RMSE
            (1 = as close as the closest real junctions, 0.05 = at the cut).
Recording = connected component of exact units under probable merges. Time axis: exact within a unit
(packets; frame-based for packetless singletons, i.e. their own duration); across a probable merge the head
of B is placed at the end of A minus the median packet overlap (9 frames) when either side has no packet
data, else after a gap estimated from the junction RMSE via the RMSE-lag table of 06 sec. 2 (source
"frames_merge").

Outputs (outputs/research/sessions/):
  clip_recordings.csv   one row per readable clip: exact unit, recording, t_start/t_end on the recording
                        axis and their source, camera column, flags
  units.csv             one row per exact unit
  recordings.csv        one row per recording
  merges.csv            every candidate merge with every criterion, accepted or not (the reason)
  tier0_summary.json
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/05a_recordings.py
"""
import json
import resource
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
T = REPO / "outputs" / "research" / "temporal"
OUT = REPO / "outputs" / "research" / "sessions"
CACHE = REPO / "outputs" / "review_cache"
TB = 15360.0
FPS = 30.0
MED_OVERLAP_S = 9 / FPS
# 06 sec. 2: median RMSE at the junction of packet-verified neighbours by |lag| in frames
LAG_TAB = [(0, 2.2), (1.5, 4.5), (4, 8.3), (7.5, 12.6), (12, 15.8), (22, 21.1), (60, 29.9), (125, 38.6)]


def pct(s, ps=(5, 25, 50, 75, 95)):
    s = pd.Series(s).dropna()
    return {f"p{p}": round(float(np.percentile(s, p)), 2) for p in ps} | {"n": int(len(s))} if len(s) else {"n": 0}


def lag_from_rmse(d):
    xs = [b for _, b in LAG_TAB]
    ys = [a for a, _ in LAG_TAB]
    return float(np.interp(d, xs, ys))


def load_clips():
    C = pd.read_csv(T / "clip_temporal.csv")
    H = pd.read_csv(REPO / "outputs" / "research" / "raw_audit_joined.csv", low_memory=False)
    H = H[["relpath", "v_start_time_s", "v_duration_s", "v_frames", "v_codec", "dec_last_pts_s", "full_decoded",
           "a0_frames", "a0_duration_s", "n_audio_streams", "md5"]].rename(columns={"relpath": "rel"})
    C = C.merge(H, on="rel", how="left")
    A = pd.read_csv(REPO / "outputs" / "research" / "clip_audit.csv", usecols=["path", "duration_s"])
    C = C.merge(A.rename(columns={"duration_s": "dur_320p_s"}), on="path", how="left")
    C["key"] = list(zip(C.cls, C.vid))
    fd = C.full_decoded.astype(str).str.lower().eq("true")
    sparse = fd & ((C.dec_last_pts_s - C.v_start_time_s) > 2 * C.real_frames / FPS + 1.0)
    C["unreliable_timing"] = (C.real_frames <= 5) | sparse
    return C


def runs_lookup(C):
    RF = pd.read_csv(T / "runs_frames.csv")
    TY = pd.read_csv(T / "partner_tests_by_run_typed.csv")[["cls", "id_start", "id_end", "type"]]
    RF = RF.merge(TY, on=["cls", "id_start", "id_end"], how="left")
    RF["type"] = np.where(RF.type.isna(), np.where(RF.period.abs() == 1, "period1", np.where(RF.period == 0, "none", "short/untyped")), RF.type)
    run_of = {}
    for i, r in RF.iterrows():
        for v in range(int(r.id_start), int(r.id_end) + 1):
            run_of[(r.cls, v)] = i
    C["run_i"] = [run_of.get(k, -1) for k in C.key]
    C["run_key"] = [f"{RF.cls[i]}:{RF.id_start[i]}-{RF.id_end[i]}" if i >= 0 else "" for i in C.run_i]
    C["run_period"] = [int(RF.period[i]) if i >= 0 else 0 for i in C.run_i]
    C["run_type"] = [RF.type[i] if i >= 0 else "none" for i in C.run_i]
    col = []
    for r in C.itertuples():
        p = abs(r.run_period)
        if r.run_i >= 0 and p >= 2:
            col.append(f"{r.run_key}/p{r.run_period}/c{(r.vid - RF.id_start[r.run_i]) % p}")
        elif r.run_i >= 0:
            col.append(f"{r.run_key}/p{r.run_period}")
        else:
            col.append("")
    C["camera_column_clip"] = col
    return C, RF


def packet_units(C, R):
    """Connected components of the packet graph with exact offsets. Clips with unreliable timing (<= 5 frames,
    or frames spread over far more time than they cover: the broken export families of 03 sec. d) may link
    only to each other: their few frames sit at 0.07 s and 320 s of a source and would otherwise bridge
    unrelated parts of a camera column with a wrong time offset (observed: a 120-s jump in high 1401-1768)."""
    bad = set(C.key[C.unreliable_timing])
    adj = defaultdict(list)
    for r in R.itertuples():
        a, b = (r.cls_a, r.id_a), (r.cls_b, r.id_b)
        if (a in bad) != (b in bad):
            continue
        d = r.off_pts_ticks / TB
        adj[a].append((b, d))
        adj[b].append((a, -d))
    origin, unit = {}, {}
    k = 0
    for n in C.key:
        if n in unit:
            continue
        q = deque([n])
        origin[n], unit[n] = 0.0, k
        while q:
            x = q.popleft()
            for y, d in adj[x]:
                if y not in unit:
                    unit[y], origin[y] = k, origin[x] + d
                    q.append(y)
        k += 1
    res = np.array([abs(origin[(r.cls_b, r.id_b)] - origin[(r.cls_a, r.id_a)] - r.off_pts_ticks / TB) for r in R.itertuples()
                    if unit[(r.cls_a, r.id_a)] == unit[(r.cls_b, r.id_b)]])
    C["unit"] = [unit[n] for n in C.key]
    C["origin_s"] = [origin[n] for n in C.key]
    C["t0"] = C.origin_s + C.v_start_time_s
    fd = C.full_decoded.astype(str).str.lower().eq("true") & C.dec_last_pts_s.notna()
    C["t1"] = np.where(fd, C.origin_s + C.dec_last_pts_s + 1 / FPS, C.t0 + C.v_duration_s)
    C["in_pk"] = C.in_pk_graph.astype(bool)
    # normalise every unit so that it starts at 0
    m0 = C.groupby("unit").t0.transform("min")
    C["t0"] -= m0
    C["t1"] -= m0
    C["origin_s"] -= m0
    return C, res


def unit_direction(C, L):
    """Sign of the ID step along time inside each unit, from the 04d packet chain links."""
    unit = dict(zip(C.key, C.unit))
    pl = L[L.type == "packet"]
    steps = defaultdict(list)
    for r in pl.itertuples():
        steps[unit[(r.cls, r.src)]].append(np.sign(r.dst - r.src))
    return {u: int(np.sign(np.median(s))) for u, s in steps.items() if len(s) and np.median(s) != 0}


def load_rows(rows_first, rows_last):
    """first/last 128x72 frames for the requested cache rows; one npz array at a time."""
    out_f, out_l = {}, {}
    base = 0
    for s in ("train", "val", "test"):
        z = np.load(CACHE / f"frames_{s}_hires.npz", allow_pickle=True)
        n = len(z["view_id"])
        for name, rows, out in (("first", rows_first, out_f), ("last", rows_last, out_l)):
            want = [r for r in rows if base <= r < base + n]
            if want:
                arr = z[name]
                for r in want:
                    out[r] = arr[r - base].copy()
                del arr
        base += n
        del z
    return out_f, out_l


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    C = load_clips()
    R = pd.read_csv(T / "packet_pairs_offsets.csv")
    L = pd.read_csv(T / "chain_links.csv")
    C, RF = runs_lookup(C)
    C, res = packet_units(C, R)
    udir = unit_direction(C, L)
    S = {"n_clips": int(len(C)), "packet_offset_residual_max_s": float(res.max()),
         "n_packet_pairs": int(len(R))}

    # ---- overlap without shared packets inside a unit (re-encoded copies or a wrong bridge)
    share = set()
    for r in R.itertuples():
        share.add(((r.cls_a, r.id_a), (r.cls_b, r.id_b)))
        share.add(((r.cls_b, r.id_b), (r.cls_a, r.id_a)))
    ov_rows = []
    for u, g in C[C.in_pk].groupby("unit"):
        if len(g) < 2:
            continue
        g = g.sort_values("t0")
        t0, t1, keys, rows = g.t0.values, g.t1.values, list(g.key), g.row.values
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if t0[j] >= t1[i]:
                    break
                ov = min(t1[i], t1[j]) - t0[j]
                if ov >= 1.0 and (keys[i], keys[j]) not in share:
                    ov_rows.append(dict(unit=u, a=keys[i], b=keys[j], row_a=rows[i], row_b=rows[j], ov=ov,
                                        dt0=t0[j] - t0[i]))
    OV = pd.DataFrame(ov_rows)

    # ---- unit table
    U = C.groupby("unit").agg(cls=("cls", "first"), n_clips=("vid", "size"), id_min=("vid", "min"), id_max=("vid", "max"),
                              span_s=("t1", "max"), in_pk=("in_pk", "max"), combo=("combo", lambda s: s.mode().iloc[0] if s.notna().any() else ""),
                              c27=("c27", lambda s: int(s.mode().iloc[0])),
                              n_chains=("chain_primary", "nunique"))
    U["direction"] = [udir.get(u, 0) for u in U.index]

    # ---- candidate probable merges: tail clips of A -> head clips of B
    Lpk = L[L.type == "packet"]
    jd = np.sort(Lpk.d_junction.dropna().values)
    CUT = float(np.percentile(jd, 95))
    tails, heads = {}, {}
    for u, g in C.groupby("unit"):
        tails[u] = g[g.t1 >= g.t1.max() - 6.0]
        heads[u] = g[g.t0 <= g.t0.min() + 6.0]
    by_cls_id = {(r.cls, r.vid): r for r in C.itertuples()}
    cand = []
    for u, tg in tails.items():
        for a in tg.itertuples():
            pa = a.run_period
            for step in range(-26, 27):
                if step == 0:
                    continue
                b = by_cls_id.get((a.cls, a.vid + step))
                if b is None or b.unit == u:
                    continue
                if b.Index not in heads[b.unit].index:
                    continue
                cand.append((a, b, step))
    rows_f = sorted({b.row for _, b, _ in cand})
    rows_l = sorted({a.row for a, _, _ in cand})
    F, Lf = load_rows(rows_f, rows_l)
    recs = []
    for a, b, step in cand:
        d = float(np.sqrt(((Lf[a.row].astype(np.float32) - F[b.row].astype(np.float32)) ** 2).mean()))
        p = a.run_period if a.run_i == b.run_i else 0
        same_run = a.run_i == b.run_i and a.run_i >= 0
        dir_ref = int(np.sign(p)) if p != 0 else (U.direction[a.unit] or U.direction[b.unit])
        recs.append(dict(cls=a.cls, unit_a=a.unit, unit_b=b.unit, tail_id=a.vid, head_id=b.vid, id_step=step, d128=d,
                         same_combo=a.combo == b.combo, same_c27=a.c27 == b.c27, same_run=same_run,
                         run_key=a.run_key if same_run else f"{a.run_key}|{b.run_key}", run_period=p,
                         run_type=a.run_type if same_run else "",
                         column_ok=(abs(p) < 2) or (step % abs(p) == 0),
                         dir_ref=dir_ref, direction_ok=dir_ref != 0 and np.sign(step) == dir_ref,
                         step_ok=abs(step) <= 3 * max(abs(p), 1) + 2,
                         both_pk=bool(a.in_pk and b.in_pk), split_a=a.split, split_b=b.split,
                         tail_t1=a.t1, head_t0=b.t0))
    M = pd.DataFrame(recs)
    # best tail/head pair per unit pair
    M = M.sort_values("d128").drop_duplicates(["unit_a", "unit_b"]).reset_index(drop=True)
    M["d_ok"] = M.d128 <= CUT
    crit = ["same_combo", "same_c27", "same_run", "column_ok", "direction_ok", "step_ok", "d_ok"]
    M["all_ok"] = M[crit].all(axis=1)
    M["failed"] = [";".join(c for c in crit if not r[c]) for _, r in M[crit].iterrows()]
    M["merge_conf"] = [float((jd >= d).mean()) for d in M.d128]
    # greedy acceptance
    succ, pred, parent = {}, {}, {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    status = []
    for r in M.itertuples():
        if not r.all_ok:
            status.append("rejected:" + r.failed)
            continue
        if r.unit_a in succ:
            status.append("rejected:tail_taken")
            continue
        if r.unit_b in pred:
            status.append("rejected:head_taken")
            continue
        if find(r.unit_a) == find(r.unit_b):
            status.append("rejected:cycle")
            continue
        succ[r.unit_a], pred[r.unit_b] = r.unit_b, r.unit_a
        parent[find(r.unit_b)] = find(r.unit_a)
        status.append("accepted")
    M["status"] = status
    A = M[M.status == "accepted"].copy()
    # ---- recordings: walk unit chains, place units on a common axis
    rec_of, shift, tsrc_unit = {}, {}, {}
    rec_id = 0
    for u in U.index:
        if u in pred:
            continue
        x, off = u, 0.0
        while True:
            rec_of[x], shift[x] = rec_id, off
            if x not in succ:
                break
            y = succ[x]
            r = A[(A.unit_a == x) & (A.unit_b == y)].iloc[0]
            end_a = off + U.span_s[x]
            if r.both_pk:
                gap = lag_from_rmse(r.d128) / FPS
            else:
                gap = -MED_OVERLAP_S
            # head of y starts at end of x plus gap; y's own axis starts at its first clip (t0 = 0)
            off = end_a + gap
            x = y
        rec_id += 1
    C["recording"] = C.unit.map(rec_of)
    C["unit_shift_s"] = C.unit.map(shift)
    C["t_start_s"] = C.t0 + C.unit_shift_s
    C["t_end_s"] = C.t1 + C.unit_shift_s
    merged_units = set(A.unit_a) | set(A.unit_b)
    C["t_source"] = np.where(C.in_pk, "packets", "frames")
    C.loc[C.unit.isin(merged_units) & (C.unit_shift_s > 0), "t_source"] = C.loc[C.unit.isin(merged_units) & (C.unit_shift_s > 0), "t_source"] + "+frames_merge"
    usize = C.groupby("unit").size()
    rsize = C.groupby("recording").unit.nunique()
    C["recording_tier"] = np.where(C.recording.map(rsize) > 1, "probable",
                                   np.where(C.unit.map(usize) > 1, "exact", "singleton"))
    # time-axis source per clip: packets for clips in multi-clip packet units; frames otherwise
    C.loc[(C.unit.map(usize) == 1) & ~C.unit.isin(merged_units), "t_source"] = "single_clip"
    C.loc[C.unreliable_timing, "t_source"] = "unknown"

    # camera column of the recording: majority over its clips
    rc = C.groupby("recording").camera_column_clip.agg(lambda s: s.mode().iloc[0] if (s != "").any() else "")
    C["camera_column"] = C.recording.map(rc)
    RECS = C.groupby("recording").agg(cls=("cls", "first"), n_clips=("vid", "size"), n_units=("unit", "nunique"),
                                      id_min=("vid", "min"), id_max=("vid", "max"), span_s=("t_end_s", "max"),
                                      combo=("combo", lambda s: s.mode().iloc[0] if s.notna().any() else ""),
                                      c27=("c27", lambda s: int(s.mode().iloc[0])), camera_column=("camera_column", "first"),
                                      tier=("recording_tier", "first"), n_train=("split", lambda s: int((s == "train").sum())),
                                      n_val=("split", lambda s: int((s == "val").sum())), n_test=("split", lambda s: int((s == "test").sum())))
    RECS["units"] = C.groupby("recording").unit.agg(lambda s: ",".join(map(str, sorted(set(s)))))
    RECS["chains_04d"] = C.groupby("recording").chain_primary.agg(lambda s: ",".join(map(str, sorted(set(s)))))

    # ---- flags
    md5n = C.md5.map(C.md5.value_counts())
    C["byte_duplicate"] = md5n > 1
    first_of_md5 = C.sort_values(["cls", "vid"]).drop_duplicates("md5").key
    C["byte_duplicate_redundant"] = C.byte_duplicate & ~C.key.isin(set(first_of_md5))
    C["fragment_le5_frames"] = C.real_frames <= 5
    C["padded_320p"] = (C.dur_320p_s - C.v_duration_s) > 1.0   # 03 sec. e: 241 clips
    md5_of = dict(zip(C.key, C.md5))
    contained = set()
    for r in R.itertuples():
        a, b = (r.cls_a, r.id_a), (r.cls_b, r.id_b)
        if md5_of.get(a) == md5_of.get(b):
            continue
        if r.kind == "contained_a_in_b":
            contained.add(a)
        elif r.kind == "contained_b_in_a":
            contained.add(b)
        elif r.kind == "identical":     # same packets, different files: flag the higher ID only
            contained.add(max(a, b, key=lambda k: k[1]))
    C["recut_duplicate_footage"] = C.key.isin(contained)
    C["audio_s"] = C.a0_duration_s.fillna(0)
    C["no_audio"] = (C.n_audio_streams.fillna(0) == 0) | (C.audio_s < 0.5)
    C["audio_single_frame"] = C.a0_frames.fillna(0) == 1
    C["hevc_no_packets"] = C.v_codec == "hevc"
    C["no_shared_packets"] = ~C.in_pk

    keep = ["path", "rel", "cls", "vid", "label", "split", "row", "c27", "c17", "combo", "batch", "v_codec", "n_frames", "v_frames",
            "real_frames", "v_start_time_s", "v_duration_s", "a0_duration_s", "n_audio_streams", "md5", "in_pk", "chain_primary",
            "chainpos_primary", "unit", "origin_s", "t0", "t1", "recording", "unit_shift_s", "t_start_s", "t_end_s", "t_source",
            "recording_tier", "run_key", "run_period", "run_type", "camera_column_clip", "camera_column",
            "byte_duplicate", "byte_duplicate_redundant", "fragment_le5_frames", "padded_320p", "recut_duplicate_footage",
            "no_audio", "audio_single_frame", "hevc_no_packets", "no_shared_packets", "unreliable_timing"]
    C[keep].to_csv(OUT / "clip_recordings.csv", index=False)
    U.to_csv(OUT / "units.csv")
    RECS.to_csv(OUT / "recordings.csv")
    M.drop(columns=[]).to_csv(OUT / "merges.csv", index=False)
    if len(OV):
        # check: at (nearly) equal start times, the first frames of such pairs should be the same picture
        # (re-encoded copies of one footage); different cameras would differ by RMSE ~60-100
        sel = OV.dt0.abs() < 0.2
        Fo, _ = load_rows(sorted(set(OV.row_a[sel]) | set(OV.row_b[sel])), [])
        OV["d_first_aligned"] = [float(np.sqrt(((Fo[a].astype(np.float32) - Fo[b].astype(np.float32)) ** 2).mean())) if s_ else np.nan
                                 for a, b, s_ in zip(OV.row_a, OV.row_b, sel)]
        del Fo
        OV.to_csv(OUT / "unit_overlaps_without_packets.csv", index=False)

    # ---- summary
    vc_t = C.recording_tier.value_counts()
    S.update({
        "n_exact_units": int(len(U)), "n_exact_units_ge2": int((U.n_clips >= 2).sum()),
        "n_packet_units_ge2": int(((U.n_clips >= 2) & U.in_pk).sum()),
        "clips_in_packet_units_ge2": int(C[C.unit.map(usize) >= 2].shape[0]),
        "n_chains_04d": int(C.chain_primary.nunique()),
        "n_chains_04d_ge2": int((C.chain_primary.value_counts() >= 2).sum()),
        "junction_cut_d128": round(CUT, 2), "exact_junction_d128": pct(jd, (5, 25, 50, 75, 95, 99)),
        "n_candidates": int(len(M)), "n_candidates_all_ok": int(M.all_ok.sum()),
        "candidate_status": M.status.str.split(":").str[0].value_counts().to_dict(),
        "rejected_by_single_criterion": M[(M.failed.str.count(";") == 0) & (M.failed != "")].failed.value_counts().to_dict(),
        "n_probable_merges": int(len(A)), "merges_both_have_packets": int(A.both_pk.sum()),
        "merge_d128": pct(A.d128, (5, 25, 50, 75, 95)), "merge_conf": pct(A.merge_conf, (5, 25, 50, 75, 95)),
        "merge_id_step": A.id_step.value_counts().to_dict(),
        "merges_crossing_published_split": int((A.split_a != A.split_b).sum()),
        "n_recordings": int(C.recording.nunique()),
        "n_recordings_ge2_clips": int((C.groupby("recording").size() >= 2).sum()),
        "recordings_by_tier": RECS.tier.value_counts().to_dict(),
        "clips_by_tier": vc_t.to_dict(), "clips_by_tier_pct": (vc_t / len(C) * 100).round(2).to_dict(),
        "t_source_clips": C.t_source.value_counts().to_dict(),
        "recordings_per_class": RECS.groupby("cls").size().to_dict(),
        "recordings_ge2_per_class": RECS[RECS.n_clips >= 2].groupby("cls").size().to_dict(),
        "recording_span_s_ge2": pct(RECS[RECS.n_clips >= 2].span_s, (25, 50, 75, 90)),
        "recording_span_s_max": round(float(RECS.span_s.max()), 1),
        "clip_weighted_recording_n_clips": pct(C.recording.map(C.recording.value_counts()), (25, 50, 75)),
        "unit_overlap_pairs_without_packets": int(len(OV)),
        "unit_overlap_pairs_without_packets_by_unit": OV.groupby("unit").size().to_dict() if len(OV) else {},
        "unit_overlap_first_frame_rmse_at_equal_start": OV.d_first_aligned.describe().round(2).to_dict() if len(OV) else {},
        "flags": {f: int(C[f].sum()) for f in ["byte_duplicate", "byte_duplicate_redundant", "fragment_le5_frames", "padded_320p",
                                               "recut_duplicate_footage", "no_audio", "audio_single_frame", "hevc_no_packets",
                                               "no_shared_packets", "unreliable_timing"]},
        "recordings_spanning_train_and_test": int(((RECS.n_train > 0) & (RECS.n_test > 0)).sum()),
        "wall_s": round(time.time() - t_start, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    })
    with open(OUT / "tier0_summary.json", "w") as fh:
        json.dump(S, fh, indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
