#!/usr/bin/env python
"""Per-clip session table, leakage-free fold definitions, their verification, and the doubtful-decision list.

Inputs: outputs/research/sessions/{clip_recordings.csv, recordings.csv, merges.csv, events.csv, event_table.csv,
audio_pairs.csv, audio_thresholds.csv, audio_null.csv, prior_links.csv, session_candidates.csv,
session_links.csv}, outputs/research/temporal/packet_pairs_offsets.csv, outputs/review_cache/frames_*.npz
(32x32 thumbnails), data/ouc-cge/data_labels/*.csv.

Table: outputs/research/sessions/clip_sessions.csv (one row per labelled clip, 7,705; the 4 unreadable clips have
no recording and are excluded from the folds).

Folds (outputs/research/sessions/folds/), readable clips only:
  event_disjoint_k5.csv      StratifiedGroupKFold(5, shuffle, random_state=0), groups = event_id (a recording
                             alone is its own event), y = label
  loeo.csv                   leave-one-event-out for the 10 largest events (by clips): column loeo_k is "test"
                             for the event, "train_session_candidate" for clips whose session_candidate_id equals
                             the event's (Tier-2 candidate partners: exclude them for a strict test), "train" else
  session_candidate_k5.csv   StratifiedGroupKFold(5) with groups = session_candidate_id (CANDIDATE-based; events
                             without a candidate partner are their own group)
Verification per fold scheme (and the published split for contrast): file pairs sharing an encoded packet
across folds (packet_pairs_offsets.csv), accepted audio links across folds, the minimum and the distribution of
the cross-fold 32x32 thumbnail distance (review clip distance: min over the 3x3 first/centre/last RMSE, float64,
computed blockwise; identical to scripts/review/07_make_grouped_splits.pairwise_min_rmse), class counts per fold,
and nn_stats() of scripts/review/07_make_grouped_splits.py (imported) for every fold as test against the rest.

Doubtful decisions: every Tier-0 merge and every Tier-1 link that is a bridge (removing it splits a recording or an
event; links confirmed by another path affect no clip), plus every prior fallback. confidence: merges ->
merge_conf; audio -> null percentile of the weaker half of the overlap in the pooled null of its length bin;
priors -> 0.25 (no evidence beyond adjacency). doubt = (1 - confidence) x clips on the smaller side.
Output: doubtful_decisions.csv (all, ranked), fold_verification.json, fold_verification.csv.
Usage: OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/05e_table_folds.py
"""
import importlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "sessions"
FOLDS = OUT / "folds"
T = REPO / "outputs" / "research" / "temporal"
CACHE = REPO / "outputs" / "review_cache"
sys.path.insert(0, str(REPO / "scripts" / "review"))
sys.dont_write_bytecode = True        # do not leave a __pycache__ entry under scripts/review/
g07 = importlib.import_module("07_make_grouped_splits")
from common import load_all_manifests  # noqa: E402

N_LOEO = 10
PRIMARY_Q = 99.9
BINS = [0, 10, 20, 40, 80, 160, 320, 640, np.inf]


def bin_of(P):
    return int(np.searchsorted(BINS, P, side="right") - 1)


def thumb_distance_matrix(paths_order):
    """(N, N) float32 min over 3x3 frame pairs of the 32x32 RMSE, float64 arithmetic, blockwise (no N x N float64)."""
    parts = [np.load(CACHE / f"frames_{s}.npz", allow_pickle=True) for s in ("train", "val", "test")]
    paths = np.concatenate([p["path"].astype(str) for p in parts])
    ok = np.concatenate([p["ok"] for p in parts])
    th = []
    for p in parts:
        th.append(p["thumbs"])
    X = np.concatenate(th)                      # (7705, 3, 32, 32, 3) uint8, 71 MB
    del th, parts
    idx = {pp: i for i, pp in enumerate(paths)}
    sel = np.array([idx[p] for p in paths_order])
    assert ok[sel].all()
    X = X[sel].reshape(len(sel), 3, -1)
    N, F, Dd = X.shape
    sq = np.zeros((N, F))
    for i0 in range(0, N, 1024):
        sq[i0:i0 + 1024] = (X[i0:i0 + 1024].astype(np.float64) ** 2).sum(-1)
    D = np.full((N, N), np.inf, np.float32)
    B = 1024
    for i0 in range(0, N, B):
        for a in range(F):
            A = X[i0:i0 + B, a].astype(np.float64)
            for j0 in range(0, N, 2048):
                best = None
                for b in range(F):
                    Bm = X[j0:j0 + 2048, b].astype(np.float64)
                    d2 = sq[i0:i0 + B, a][:, None] + sq[j0:j0 + 2048, b][None, :] - 2.0 * (A @ Bm.T)
                    best = d2 if best is None else np.minimum(best, d2)
                np.maximum(best, 0, out=best)
                blk = np.sqrt(best / Dd).astype(np.float32)
                D[i0:i0 + B, j0:j0 + 2048] = np.minimum(D[i0:i0 + B, j0:j0 + 2048], blk)
    D = np.minimum(D, D.T)
    np.fill_diagonal(D, 0)
    return D


def components(nodes, edges):
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    return {n: find(n) for n in nodes}


def main():
    t0 = time.time()
    FOLDS.mkdir(parents=True, exist_ok=True)
    M = load_all_manifests(REPO / "data" / "ouc-cge")
    M = M[M.cls_folder.notna()].rename(columns={"split": "split_published"})
    CR = pd.read_csv(OUT / "clip_recordings.csv")
    RECS = pd.read_csv(OUT / "recordings.csv").set_index("recording")
    E = pd.read_csv(OUT / "events.csv").set_index("recording")
    ET = pd.read_csv(OUT / "event_table.csv").set_index("event")
    PR = pd.read_csv(OUT / "audio_pairs.csv")
    NULL = pd.read_csv(OUT / "audio_null.csv")
    MG = pd.read_csv(OUT / "merges.csv")
    PRI = pd.read_csv(OUT / "prior_links.csv")
    SC = pd.read_csv(OUT / "session_candidates.csv").set_index("event")

    # ---------------------------------------------------------------- table
    D = M[["path", "label", "split_published", "cls_folder", "view_id"]].rename(columns={"cls_folder": "class"})
    D = D.merge(CR.drop(columns=["label"]).rename(columns={"split": "split_check"}), on="path", how="left")
    D["readable"] = D.recording.notna()
    D["scene_cluster_27_5"] = D.c27
    D["recording_id"] = D.recording.astype("Int64")
    D["exact_unit_id"] = D.unit.astype("Int64")
    D["chain_04d"] = D.chain_primary.astype("Int64")
    src = {"packets": "packets", "packets+frames_merge": "packets", "frames": "frames", "frames+frames_merge": "frames",
           "single_clip": "frames", "unknown": "unknown"}
    D["t_source_detail"] = D.t_source
    D["t_source"] = D.t_source.map(src).fillna("unknown")
    D.loc[D.t_source_detail.astype(str).str.contains("frames_merge"), "t_source"] = D.t_source + "+merge_estimate"
    rec = D.recording
    D["event_id"] = rec.map(E.event).astype("Int64")
    D["event_tier"] = rec.map(E.event_tier).fillna("none")
    D["event_offset_s"] = rec.map(E.event_offset_s)
    D["t_event_s"] = D.event_offset_s + D.t_start_s
    D["event_size_recordings"] = D.event_id.map(ET.n_recordings)
    D["event_classes"] = D.event_id.map(ET.classes)
    D["event_audio_r"] = rec.map(E.best_audio_r)
    D["event_audio_null_pct"] = rec.map(E.best_audio_null_pct)
    D["event_confidence"] = np.where(D.event_tier == "audio",
                                     [f"r={r:.3f};null_pct={p:.3f}" if np.isfinite(r) else "NA" for r, p in
                                      zip(D.event_audio_r.fillna(np.nan), D.event_audio_null_pct.fillna(np.nan))], "NA")
    D["duplicate_of_recording"] = rec.map(E.duplicate_of_recording)
    D["layout"] = rec.map(E.layout)
    D["session_candidate_id"] = D.event_id.map(SC.session_candidate_id).astype("Int64")
    D["session_candidate_size_events"] = D.event_id.map(SC.session_size_events)
    D["session_score"] = D.event_id.map(SC.session_score)
    D.loc[D.session_candidate_size_events == 1, "session_score"] = np.nan
    D["unreadable"] = ~D.readable
    D["reencoded_duplicate_recording"] = D.duplicate_of_recording.notna()
    flag_cols = ["byte_duplicate", "fragment_le5_frames", "padded_320p", "recut_duplicate_footage", "no_audio",
                 "hevc_no_packets", "unreliable_timing", "reencoded_duplicate_recording", "no_shared_packets",
                 "audio_single_frame", "unreadable"]
    for f in flag_cols:
        D[f] = D[f].astype("boolean").fillna(False).astype(bool) if f != "unreadable" else D[f]
    D["flags"] = [";".join(f for f in flag_cols if r[f]) for _, r in D[flag_cols].iterrows()]
    D["clean_subset"] = D.readable & ~(D.fragment_le5_frames | D.byte_duplicate_redundant.astype("boolean").fillna(False).astype(bool)
                                       | D.recut_duplicate_footage | D.padded_320p | D.unreliable_timing
                                       | D.reencoded_duplicate_recording)
    cols = ["path", "class", "label", "split_published", "scene_cluster_27_5", "recording_id", "recording_tier",
            "exact_unit_id", "chain_04d", "camera_column", "t_start_s", "t_end_s", "t_source", "t_source_detail",
            "event_id", "event_tier", "event_confidence", "event_audio_r", "event_audio_null_pct", "event_offset_s",
            "t_event_s", "event_size_recordings", "event_classes", "duplicate_of_recording", "layout",
            "session_candidate_id", "session_candidate_size_events", "session_score", "flags"] + flag_cols + \
           ["byte_duplicate_redundant", "readable", "clean_subset"]
    T_ = D[cols].sort_values(["class", "path"]).reset_index(drop=True)
    T_.to_csv(OUT / "clip_sessions.csv", index=False)
    print(f"table {len(T_)} rows, {time.time() - t0:.1f} s", flush=True)

    # ---------------------------------------------------------------- folds
    R = T_[T_.readable].reset_index(drop=True)
    y = R.label.values
    folds = {}
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    g_ev = R.event_id.astype(int).values
    f_ev = np.empty(len(R), int)
    for k, (_, te) in enumerate(sgk.split(np.zeros(len(R)), y, g_ev)):
        f_ev[te] = k
    folds["event_disjoint_k5"] = f_ev
    pd.DataFrame(dict(path=R.path, label=R.label, cls=R["class"], group_event_id=g_ev, fold=f_ev)).to_csv(
        FOLDS / "event_disjoint_k5.csv", index=False)
    multi = R.session_candidate_size_events > 1
    g_ses = np.where(multi, "S" + R.session_candidate_id.astype(str), "E" + R.event_id.astype(str))
    f_ses = np.empty(len(R), int)
    for k, (_, te) in enumerate(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(len(R)), y, g_ses)):
        f_ses[te] = k
    folds["session_candidate_k5"] = f_ses
    pd.DataFrame(dict(path=R.path, label=R.label, cls=R["class"], group=g_ses, group_is_session_candidate=multi,
                      fold=f_ses, note="CANDIDATE-based: Tier-2 links are unverified")).to_csv(
        FOLDS / "session_candidate_k5.csv", index=False)
    top = R.groupby("event_id").size().sort_values(ascending=False).head(N_LOEO).index.tolist()
    LO = pd.DataFrame(dict(path=R.path, label=R.label, cls=R["class"], event_id=R.event_id, session_candidate_id=R.session_candidate_id))
    for k, ev in enumerate(top):
        sid = R.loc[R.event_id == ev, "session_candidate_id"].iloc[0]
        role = np.where(R.event_id == ev, "test", np.where((R.session_candidate_id == sid) & (R.session_candidate_size_events > 1),
                                                           "train_session_candidate", "train"))
        LO[f"loeo_{k}"] = role
    LO.to_csv(FOLDS / "loeo.csv", index=False)
    loeo_meta = [dict(fold=k, event_id=int(ev), n_test=int((R.event_id == ev).sum()),
                      test_classes=R.loc[R.event_id == ev, "class"].value_counts().to_dict(),
                      n_train_session_candidate=int((LO[f"loeo_{k}"] == "train_session_candidate").sum())) for k, ev in enumerate(top)]

    # ---------------------------------------------------------------- verification
    Dm = thumb_distance_matrix(R.path.tolist())
    print(f"distance matrix {Dm.shape}, {time.time() - t0:.1f} s", flush=True)
    idx_of = {(c, int(v)): i for i, (c, v) in enumerate(zip(R["class"], R.path.str.extract(r"view(\d+)\.mp4")[0].astype(int)))}
    PP = pd.read_csv(T / "packet_pairs_offsets.csv", usecols=["cls_a", "id_a", "cls_b", "id_b"])
    pk_i = np.array([idx_of.get((c, int(v)), -1) for c, v in zip(PP.cls_a, PP.id_a)])
    pk_j = np.array([idx_of.get((c, int(v)), -1) for c, v in zip(PP.cls_b, PP.id_b)])
    okp = (pk_i >= 0) & (pk_j >= 0)
    pk_i, pk_j = pk_i[okp], pk_j[okp]
    rec_clips = R.groupby("recording_id").apply(lambda g: g.index.values, include_groups=False).to_dict()
    acc = PR[PR["accept_q99.9"] | PR.accept_full]
    split_code = R.split_published.map({"train": 0, "val": 1, "test": 2}).values
    schemes = dict(folds)
    schemes["published_split"] = split_code
    ver_rows, ver = [], {}
    for name, f in schemes.items():
        n_pk = int((f[pk_i] != f[pk_j]).sum())
        n_audio = 0
        for r in acc.itertuples():
            fa = set(f[rec_clips[r.a]]) if r.a in rec_clips else set()
            fb = set(f[rec_clips[r.b]]) if r.b in rec_clips else set()
            if fa and fb and (fa != fb or len(fa) > 1):
                n_audio += 1
        res = dict(scheme=name, packet_pairs_across_folds=n_pk, audio_links_across_folds=n_audio, folds={})
        nn_all = np.empty(len(R), np.float32)
        for k in sorted(set(f)):
            q = np.flatnonzero(f == k)
            ref = np.flatnonzero(f != k) if name != "published_split" else np.flatnonzero(f == 0)
            if name == "published_split" and k == 0:
                continue
            st = g07.nn_stats(Dm, q, ref)
            sub = Dm[np.ix_(q, np.flatnonzero(f != k))]
            nn_all[q] = sub.min(axis=1)
            del sub
            cc = R.iloc[q]["class"].value_counts().to_dict()
            res["folds"][int(k)] = dict(n=int(len(q)), classes=cc, nn_stats_vs_rest=st)
            ver_rows.append(dict(scheme=name, fold=int(k), n=len(q), low=cc.get("low", 0), mid=cc.get("mid", 0), high=cc.get("high", 0),
                                 nn_median=st["median"], nn_p5=st["p5"], nn_min=st["min"], frac_le_10=st["frac_le_10"],
                                 frac_le_17_5=st["frac_le_17_5"], frac_le_27_5=st["frac_le_27_5"]))
        if name != "published_split":
            res["min_cross_fold_thumb_distance"] = float(nn_all.min())
            res["cross_fold_nn"] = dict(median=float(np.median(nn_all)), p5=float(np.percentile(nn_all, 5)),
                                        frac_le_10=float((nn_all <= 10).mean()), frac_le_17_5=float((nn_all <= 17.5).mean()),
                                        frac_le_27_5=float((nn_all <= 27.5).mean()))
        ver[name] = res
        print(name, n_pk, n_audio, res.get("min_cross_fold_thumb_distance"), flush=True)
    ver["loeo"] = []
    for m_ in loeo_meta:
        k = m_["fold"]
        role = LO[f"loeo_{k}"].values
        q = np.flatnonzero(role == "test")
        for variant, refmask in (("train_all", role != "test"), ("train_strict", role == "train")):
            ref = np.flatnonzero(refmask)
            st = g07.nn_stats(Dm, q, ref)
            fq = np.zeros(len(R), int)
            fq[q] = 1
            fq[~refmask & (role != "test")] = 2      # excluded clips: neither side
            n_pk = int(((fq[pk_i] == 1) & (fq[pk_j] == 0) | (fq[pk_i] == 0) & (fq[pk_j] == 1)).sum())
            m_[f"{variant}_nn"] = st
            m_[f"{variant}_packet_pairs_across"] = n_pk
        ver["loeo"].append(m_)
    del Dm
    with open(OUT / "fold_verification.json", "w") as fh:
        json.dump(ver, fh, indent=1, default=str)
    pd.DataFrame(ver_rows).to_csv(OUT / "fold_verification.csv", index=False)

    # ---------------------------------------------------------------- doubtful decisions
    clips_per_rec = R.groupby("recording_id").size().to_dict()
    clips_per_unit = CR.groupby("unit").size().to_dict()
    dd = []
    # Tier-0 merges: bridges within recordings (every merge is a bridge: one successor/predecessor chain)
    A0 = MG[MG.status == "accepted"]
    for r in A0.itertuples():
        rec_id = CR.loc[CR.unit == r.unit_a, "recording"].iloc[0]
        units = CR[CR.recording == rec_id].unit.unique().tolist()
        edges = [(x.unit_a, x.unit_b) for x in A0.itertuples() if x.unit_a in units and not (x.unit_a == r.unit_a and x.unit_b == r.unit_b)]
        comp = components(units, edges)
        sizes = defaultdict(int)
        for u in units:
            sizes[comp[u]] += clips_per_unit[u]
        affected = min(sizes.values()) if len(sizes) > 1 else 0
        dd.append(dict(tier="0 recording", decision=f"merge unit {r.unit_a} -> {r.unit_b} ({r.cls} {r.tail_id}->{r.head_id})",
                       evidence="frames (junction RMSE)", value=f"d128={r.d128:.1f}", confidence=float(r.merge_conf),
                       clips_affected=int(affected), a=int(r.unit_a), b=int(r.unit_b), cls=r.cls, kind="merge",
                       reason=f"junction RMSE {r.d128:.1f} vs exact junctions (conf = share of packet junctions this far or farther); id step {r.id_step}"))
    # Tier-1 links of every kind: bridges in the event graph (audio, structure-guided audio, footage identity, prior)
    pooled = {bi: np.sort(NULL[NULL.bin == bi].r60.dropna().values) for bi in range(len(BINS) - 1)}
    ST = pd.read_csv(OUT / "structural_links.csv")
    IDL = pd.read_csv(OUT / "identity_links.csv") if (OUT / "identity_links.csv").exists() else pd.DataFrame()
    edges = []
    for _, r in acc.iterrows():
        full = not bool(r["accept_q99.9"])
        h = min(r.rf_h1, r.rf_h2) if full else min(r.r_h1, r.r_h2)
        g = pooled[bin_of(r.P / 2)]
        conf = float(np.searchsorted(g, h) / len(g)) if len(g) and np.isfinite(h) else 0.0
        rr, LL = (r.rfull, r.Lfull) if full else (r.r60, r.L60)
        edges.append(dict(a=int(r.a), b=int(r.b), kind="audio", confidence=conf, cls=f"{r.cls_a}/{r.cls_b}",
                          evidence="audio" + (" full-lag" if full else " +-60 s") + (" duplicate" if bool(r.dup_candidate) else ""),
                          value=f"r={rr:.3f} L={LL:.1f}s P={r.P:.0f}s halves={(r.rf_h1 if full else r.r_h1):.2f}/{(r.rf_h2 if full else r.r_h2):.2f}",
                          reason=("short overlap; " if r.P < 40 else "") + ("full-lag scan; " if full else "")
                          + ("cross-class; " if r.cls_a != r.cls_b else "") + ("duplicate footage by r>=0.95 & same scene; " if bool(r.dup_candidate) else "")
                          + f"weaker half at null percentile {100 * conf:.2f}"))
    for r in ST[ST.decision == "audio_structure_guided"].itertuples():
        g = pooled[bin_of(r.guided_P / 2)]
        h = min(r.guided_r_h1, r.guided_r_h2)
        conf = float(np.searchsorted(g, h) / len(g)) if len(g) and np.isfinite(h) else 0.0
        edges.append(dict(a=int(r.a), b=int(r.b), kind="audio", confidence=conf, cls=RECS.cls[int(r.a)],
                          evidence="audio at structural lag", value=f"r={r.guided_r:.3f} L={r.guided_L:.1f}s (struct {r.L_struct:.1f}s)",
                          reason=f"camera-run columns; audio tested in +-10 s around the structural lag; weaker half at null percentile {100 * conf:.2f}"))
    for r in ST[ST.decision == "fallback_link"].itertuples():
        edges.append(dict(a=int(r.a), b=int(r.b), kind="structural", confidence=0.4, cls=RECS.cls[int(r.a)],
                          evidence="structural FALLBACK", value=f"run {r.run} L={r.L_struct:.1f}s",
                          reason="interleaved camera run, the attached recording has < 30 s of audio"))
    for r in IDL.itertuples() if len(IDL) else []:
        if r.decision != "identity_link":
            continue
        conf = 1.0 if r.kind == "packets" else float(max(0.5, 1 - r.min_d32 / 6.0))
        edges.append(dict(a=int(r.a), b=int(r.b), kind=r.kind, confidence=conf, cls=f"{r.cls_a}/{r.cls_b}",
                          evidence="shared packets (broken family)" if r.kind == "packets" else "near-identical frames (re-encoded copy)",
                          value=f"pairs={r.n_clip_pairs}" + ("" if r.kind == "packets" else f" min_d32={r.min_d32:.2f}"),
                          reason="exact footage identity, timing unknown" if r.kind == "packets" else
                          "same frame up to re-encoding at 32x32; could also be a static scene seen by one camera at two times"))
    for r in PRI[PRI.decision == "fallback_link"].itertuples():
        edges.append(dict(a=int(r.rec), b=int(r.attached_to), kind="prior", confidence=0.25, cls=RECS.cls[int(r.rec)],
                          evidence="prior FALLBACK (ID adjacency, batch, layout)", value=f"id_gap={int(r.id_gap)} audio={r.valid_audio_s:.0f}s",
                          reason="no audio (< 30 s) and no structure; attached by ID adjacency within the same codec/size and layout"))
    for k, e in enumerate(edges):
        ev = E.event[e["a"]]
        if E.event[e["b"]] != ev:
            continue
        recs = E.index[E.event == ev].tolist()
        others = [(x["a"], x["b"]) for j, x in enumerate(edges) if j != k and x["a"] in recs and x["b"] in recs]
        comp = components(recs, others)
        sizes = defaultdict(int)
        for q in recs:
            sizes[comp[q]] += clips_per_rec.get(q, 0)
        affected = min(sizes.values()) if len(sizes) > 1 else 0
        dd.append(dict(tier="1 event", decision=f"{e['kind']} link rec {e['a']} - rec {e['b']} ({e['cls']})", evidence=e["evidence"],
                       value=e["value"], confidence=e["confidence"], clips_affected=int(affected), a=e["a"], b=e["b"],
                       cls=e["cls"], kind=e["kind"], reason=e["reason"]))
    DD = pd.DataFrame(dd)
    DD["doubt"] = (1 - DD.confidence) * DD.clips_affected
    DD = DD.sort_values(["doubt", "clips_affected"], ascending=False).reset_index(drop=True)
    DD.to_csv(OUT / "doubtful_decisions.csv", index=False)

    S = {"table_rows": int(len(T_)), "readable": int(R.shape[0]), "clean_subset": int(T_.clean_subset.sum()),
         "clean_subset_by_class": T_[T_.clean_subset].groupby("class").size().to_dict(),
         "flags": {f: int(T_[f].sum()) for f in flag_cols},
         "t_source": T_.t_source.value_counts().to_dict(),
         "recording_tier_clips": T_.recording_tier.value_counts().to_dict(),
         "event_tier_clips": T_.event_tier.value_counts().to_dict(),
         "clips_in_multi_session_candidates": int((T_.session_candidate_size_events > 1).sum()),
         "n_groups": {"event_disjoint_k5": int(len(set(g_ev))), "session_candidate_k5": int(len(set(g_ses)))},
         "loeo_events": loeo_meta,
         "doubtful_top15": DD.head(15)[["tier", "decision", "evidence", "value", "confidence", "clips_affected", "reason"]].to_dict("records"),
         "n_bridge_decisions": int((DD.clips_affected > 0).sum()),
         "wall_s": round(time.time() - t0, 1), "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
    with open(OUT / "table_folds_summary.json", "w") as fh:
        json.dump(S, fh, indent=1, default=str)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(ver_rows).round(3).to_string())
    print(json.dumps({k: v for k, v in S.items() if k not in ("doubtful_top15", "loeo_events")}, indent=1, default=str))
    print(DD.head(20)[["tier", "decision", "evidence", "value", "confidence", "clips_affected"]].to_string())


if __name__ == "__main__":
    main()
