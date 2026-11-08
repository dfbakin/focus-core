#!/usr/bin/env python
"""Temporal and camera structure along the OUC-CGE clip-ID axis (06_temporal_structure.md, A-C).

Inputs (all precomputed, nothing is decoded here):
  outputs/research/temporal/succ_dist.csv, ref_pairs.csv   (04c_successor_distances.py, 128x72 RMSE)
  outputs/research/temporal/ref_frame_lags.csv             (04a_reference_scales.py)
  outputs/research/temporal/packet_pairs_offsets.csv       (04b_packet_offsets.py, packet ground truth)
  outputs/research/raw_video_packets.npz                   (test/train packet coverage)
  outputs/research/raw_audit_joined.csv, raw_audit_id_runs.csv (raw container metadata)
  outputs/review_out/scene_clusters.csv, exp3b_clusters_by_thr.csv (scene clusters at 27.5 / 17.5)
  outputs/review_cache/frames_*.npz                        (32x32 thumbs for the cross-chain search)

A. Per clip the best temporal neighbour among N+1..N+8 in both directions:
   forward  d_k(N) = RMSE(last(N), first(N+k)),  backward b_k(N) = RMSE(last(N+k), first(N)).
   k*, dir* = argmin over the 16 candidates, m(N) = the minimum. A continuation is m <= cut. The
   log-histogram of m (Gaussian KDE, Scott bandwidth) has three modes (frame-exact junction ~2.3,
   junction inside an overlap ~16, different room ~90); the primary cut is the upper valley (between
   the continuation and the different-room modes), the alternatives are the lower valley ("strict":
   frame-exact junctions only) and the median RMSE after 0.5 s of motion ("intermediate"). Signed period label
   lab(N) = dir* x k* (ID step per forward time step) if continuation else 0; the ID axis is
   segmented by the majority label in a centred window of 15 clips; runs = maximal stretches of
   constant majority label. The same segmentation is done on the packet successor (exact).
B. Interleaved partners in runs with |period| in {2, 3, 4}: cluster, packet, duration, first-pts
   and container-metadata agreement of partners (N, N+j), j < |p|, against same-source pairs
   (N, N+|p|) and random same-class pairs (default_rng(0)).
C. Chains: one successor and one predecessor per clip; links from packet overlap first (exact, by
   ascending start offset), then frame RMSE links for pairs without shared packets (ascending d);
   no cycles. Label runs, the published split along chains, and test clips with a temporal
   neighbour / simultaneous view / shared footage in train.

Outputs: outputs/research/temporal/{clip_temporal.csv, runs_frames.csv, runs_packets.csv,
pr_sweep.csv, packet_offsets_by_iddiff.csv, packet_offsets_by_period.csv, partner_tests.csv,
partner_tests_by_run.csv, chains.csv, chain_links.csv, superchain_links.csv, disagreements.csv,
summary.json}.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04d_temporal_analysis.py
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import load_cache  # noqa: E402

T = REPO / "outputs" / "research" / "temporal"
K = 8
WIN = 15
CLS = ["low", "mid", "high"]
SUMMARY = {}


def pct(s, ps=(5, 25, 50, 75, 95)):
    s = pd.Series(s).dropna()
    return {f"p{p}": round(float(np.percentile(s, p)), 3) for p in ps} | {"n": int(len(s))} if len(s) else {"n": 0}


# ----------------------------------------------------------------------------- loading
def load():
    D = pd.read_csv(T / "succ_dist.csv")
    D["rel"] = D.path.str.replace("videos/", "", regex=False)
    sc = pd.read_csv(REPO / "outputs" / "review_out" / "scene_clusters.csv")[["path", "cluster"]]
    st = pd.read_csv(REPO / "outputs" / "review_out" / "exp3b_clusters_by_thr.csv")[["path", "cluster_thr17.5"]]
    D = D.merge(sc.rename(columns={"cluster": "c27"}), on="path", how="left")
    D = D.merge(st.rename(columns={"cluster_thr17.5": "c17"}), on="path", how="left")
    J = pd.read_csv(REPO / "outputs" / "research" / "raw_audit_joined.csv", low_memory=False)
    J = J[["relpath", "combo", "first_pts_s", "real_frames", "n_audio", "resolution", "v_codec", "md5"]]
    D = D.merge(J.rename(columns={"relpath": "rel"}), on="rel", how="left")
    runs = pd.read_csv(REPO / "outputs" / "research" / "raw_audit_id_runs.csv")
    bid = {}
    for r in runs.itertuples():
        for i in range(r.id_from, r.id_to + 1):
            bid[(r.cls, i)] = f"{r.cls}:{r.id_from}-{r.id_to}"
    D["batch"] = [bid.get((c, v)) for c, v in zip(D.cls, D.vid)]
    D["dur_s"] = np.where(D.real_frames.notna(), D.real_frames, D.n_frames) / 30.0
    R = pd.read_csv(T / "packet_pairs_offsets.csv")
    return D, R, runs


# ----------------------------------------------------------------------------- reference scales
def reference_scales(D, R):
    L = pd.read_csv(T / "ref_frame_lags.csv")
    steps = L[[f"rmse_step_{t}" for t in range(3, 12)]].median(axis=1)
    gsteps = L[[f"gray_step_{t}" for t in range(3, 12)]].median(axis=1)
    rp = pd.read_csv(T / "ref_pairs.csv")
    ref = {
        "pad_duplicate_step_0_1": pct(L.rmse_step_0),
        "pad_duplicate_step_1_2": pct(L.rmse_step_1),
        "raw_consecutive_frame_step_t3_11_per_clip_median": pct(steps),
        "raw_consecutive_frame_step_gray": pct(gsteps),
        "lag_curve_median_rgb": {int(c.split("_")[-1]): round(float(L[c].median()), 2)
                                 for c in L.columns if c.startswith("rmse_lag_")},
        "lag_curve_median_gray": {int(c.split("_")[-1]): round(float(L[c].median()), 2)
                                  for c in L.columns if c.startswith("gray_lag_")},
        "within_clip_first_last_rgb": pct(D.fl),
        "within_clip_first_last_gray": pct(D.gfl),
    }
    for t, g in rp.groupby("pair_type"):
        ref[f"{t}_first_first_rgb"] = pct(g.ff, (0.1, 1, 5, 50))
        ref[f"{t}_first_first_gray"] = pct(g.gff, (0.1, 1, 5, 50))
        ref[f"{t}_last_first_rgb"] = pct(g.lf, (0.1, 1, 5, 50))
    ref["n_sampled_clips_for_floor"] = int(len(L))
    return ref, float(steps.median())


# ----------------------------------------------------------------------------- packet truth
def packet_truth(R):
    """dict (cls, idN, k) -> (rel, offset o of N+k vs N, lag at the junction) for 1 <= k <= 8."""
    Q = R[(R.kind != "identical") & (R.id_diff >= 1) & (R.id_diff <= K)]
    truth = {}
    for r in Q.itertuples():
        o = r.start_b_minus_a
        if r.kind == "partial" and o > 0:
            rel, lag = "fwd", r.n_a - 1 - o          # last(N) is frame `lag` of N+k
        elif r.kind == "partial" and o < 0:
            rel, lag = "bwd", r.n_b - 1 + o          # last(N+k) is frame `lag` of N
        else:
            rel, lag = "contain", np.nan
        truth[(r.cls_a, r.id_a, r.id_diff)] = (rel, o, lag)
    return truth


def packet_successors(R, D):
    """Immediate packet successor/predecessor per clip: partial overlap, later start, later end,
    |ID diff| <= 8, smallest start offset. Returns dicts keyed by (cls, id)."""
    Q = R[(R.kind == "partial") & (R.id_diff.abs() <= K)]
    e = pd.concat([
        pd.DataFrame({"cls": Q.cls_a, "src": Q.id_a, "dst": Q.id_b, "off": Q.start_b_minus_a, "ext": Q.end_b_minus_a,
                      "n_src": Q.n_a}),
        pd.DataFrame({"cls": Q.cls_a, "src": Q.id_b, "dst": Q.id_a, "off": -Q.start_b_minus_a, "ext": -Q.end_b_minus_a,
                      "n_src": Q.n_b})])
    e = e[(e.off > 0) & (e.ext > 0)].copy()
    e["step"] = e.dst - e.src
    return e


# ----------------------------------------------------------------------------- A
def best_neighbour(D):
    C = np.stack([D[f"d{k}"] for k in range(1, K + 1)] + [D[f"b{k}"] for k in range(1, K + 1)], 1)
    C = np.where(np.isnan(C), np.inf, C)
    j = C.argmin(1)
    D["m"] = C.min(1)
    D["kstar"] = j % K + 1
    D["dirstar"] = np.where(j < K, 1, -1)
    D["second"] = np.sort(C, 1)[:, 1]
    Cf = C[:, :K]
    D["m_fwd"] = Cf.min(1)
    D["kstar_fwd"] = Cf.argmin(1) + 1
    G = np.stack([D[f"gd{k}"] for k in range(1, K + 1)] + [D[f"gb{k}"] for k in range(1, K + 1)], 1)
    G = np.where(np.isnan(G), np.inf, G)
    D["m_gray"] = G.min(1)
    D["kstar_gray"] = G.argmin(1) % K + 1
    D["dirstar_gray"] = np.where(G.argmin(1) < K, 1, -1)
    D.loc[~np.isfinite(D.m), ["m", "kstar", "dirstar"]] = np.nan
    return D


def kde_valley(x):
    """Gaussian KDE (Scott bandwidth) of log10(x); returns all local modes and minima."""
    lx = np.log10(np.clip(x[np.isfinite(x)], 0.5, None))
    kde = gaussian_kde(lx)
    g = np.linspace(lx.min(), lx.max(), 2000)
    y = kde(g)
    mx = [i for i in range(1, len(g) - 1) if y[i] >= y[i - 1] and y[i] > y[i + 1]]
    mn = [i for i in range(1, len(g) - 1) if y[i] <= y[i - 1] and y[i] < y[i + 1]]
    return dict(modes=[round(float(10 ** g[i]), 2) for i in mx], density_modes=[round(float(y[i]), 3) for i in mx],
                valleys=[round(float(10 ** g[i]), 2) for i in mn], density_valleys=[round(float(y[i]), 3) for i in mn],
                bandwidth_log10=round(float(kde.factor * lx.std(ddof=1)), 4))


def majority_runs(ids, labs, win=WIN):
    """Centred-window majority of labs (ties: own label if tied, else smallest |label|, then
    positive). Returns per-clip majority and run index."""
    n = len(labs)
    h = win // 2
    maj = np.empty(n, dtype=int)
    for i in range(n):
        w = labs[max(0, i - h): i + h + 1]
        cnt = Counter(w.tolist())
        top = max(cnt.values())
        tied = [v for v, c in cnt.items() if c == top]
        maj[i] = labs[i] if labs[i] in tied else sorted(tied, key=lambda v: (abs(v), -v))[0]
    run = np.r_[0, np.cumsum(maj[1:] != maj[:-1])]
    return maj, run


def runs_table(D, labcol, majcol, runcol, R, extra_pk=None):
    rows = []
    for (cls, rid), g in D.groupby(["cls", runcol], sort=False):
        p = int(g[majcol].iloc[0])
        cl = g.c27.value_counts()
        dom = "; ".join(f"{int(c)}:{v / len(g):.0%}" for c, v in cl.head(3).items())
        cb = g.combo.fillna("(unreadable)").value_counts()
        batches = list(dict.fromkeys(g.batch.dropna()))
        rec = dict(cls=cls, id_start=int(g.vid.min()), id_end=int(g.vid.max()), period=p, n_clips=len(g),
                   frac_cont=round(float((g[labcol] != 0).mean()), 3),
                   frac_lab_eq_period=round(float((g[labcol] == p).mean()), 3),
                   clusters27=dom, n_clusters27=int(g.c27.nunique()),
                   labels="; ".join(f"{int(k)}:{v}" for k, v in g.label.value_counts().items()),
                   splits="; ".join(f"{k}:{v}" for k, v in g.split.value_counts().items()),
                   combo=f"{cb.index[0]} ({cb.iloc[0] / len(g):.0%})", n_combos=int(len(cb)),
                   raw_batches=", ".join(b.split(":")[1] for b in batches[:6]) + (" ..." if len(batches) > 6 else ""),
                   n_raw_batches=len(batches),
                   frac_first_pts_ge_1s=round(float((g.first_pts_s >= 1).mean()), 3),
                   median_raw_frames=float(g.real_frames.median()),
                   median_m=round(float(g.m.median()), 2))
        if "pk_lab" in g:
            pk = g.pk_lab[g.pk_lab != 0]
            rec["frac_pk_lab_eq_period"] = round(float((g.pk_lab == p).mean()), 3)
            rec["pk_period_mode"] = int(pk.mode().iloc[0]) if len(pk) else 0
        if p != 0:
            q = R[(R.cls_a == cls) & (R.id_diff == abs(p)) & (R.id_a >= g.vid.min()) & (R.id_b <= g.vid.max())
                  & (R.kind == "partial")]
            rec["n_pk_pairs_at_period"] = len(q)
            rec["median_stride_frames"] = float(q.start_b_minus_a.abs().median()) if len(q) else np.nan
            rec["median_overlap_frames"] = float(q.overlap_frames.median()) if len(q) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    D, R, idruns = load()
    D = D.sort_values(["cls", "vid"]).reset_index(drop=True)
    ref, raw_floor = reference_scales(D, R)
    truth = packet_truth(R)
    D = best_neighbour(D)

    # cross-clip same-raw-frame floor: packet-verified forward pairs whose junction lag is 0
    jl = []
    for r in D.itertuples():
        for k in range(1, K + 1):
            t = truth.get((r.cls, r.vid, k))
            if t and t[0] in ("fwd", "bwd") and np.isfinite(t[2]):
                jl.append((t[0], k, t[2], getattr(r, f"d{k}") if t[0] == "fwd" else getattr(r, f"b{k}")))
    JL = pd.DataFrame(jl, columns=["rel", "k", "lag", "d"]).dropna()
    ref["same_raw_frame_across_clips_lag0"] = pct(JL[JL.lag == 0].d)
    ref["junction_lag_bins"] = {f"{a}-{b}": pct(JL[(JL.lag >= a) & (JL.lag < b)].d, (25, 50, 75))
                                for a, b in [(0, 1), (1, 3), (3, 6), (6, 10), (10, 15), (15, 30), (30, 90), (90, 160)]}
    ref["junction_lag_all"] = pct(JL.lag)
    SUMMARY["reference_scales"] = ref

    # ---- cuts
    hv = kde_valley(D.m.values)
    hv_fwd = kde_valley(D.m_fwd.values)
    # three modes expected: frame-exact junction (~2), junction inside an overlap (~15), different room (~90)
    assert len(hv["valleys"]) >= 2, hv
    exact = hv["valleys"][0]                 # valley between the frame-exact and the main mode
    primary = hv["valleys"][-1]              # valley between the main mode and the different-room mode
    mid = ref["lag_curve_median_rgb"][15]    # median RMSE after 0.5 s of motion (frame 0 vs 15)
    strict = exact
    cuts = {"strict": exact, "intermediate": round(mid, 1), "primary_valley": primary}
    SUMMARY["histogram_min_d_both_directions"] = hv
    SUMMARY["histogram_min_d_forward_only"] = hv_fwd
    SUMMARY["cuts"] = cuts
    SUMMARY["histogram_counts_log_bins"] = {
        f"{a:.2f}": int(c) for a, c in zip(*(lambda h: (h[1][:-1], h[0]))(np.histogram(
            D.m.dropna(), bins=np.logspace(np.log10(1), np.log10(140), 29))))}
    for name, c in cuts.items():
        D[f"cont_{name}"] = D.m <= c
    D["cont"] = D.cont_primary_valley

    # ---- packet validation of the frame argmin
    inpk = set(R[R.kind != "identical"].rel_a) | set(R[R.kind != "identical"].rel_b)
    D["in_pk_graph"] = D.rel.isin(inpk)
    D["pk_rel_pred"] = [truth.get((c, v, k), (None,))[0] if np.isfinite(k) else None
                        for c, v, k in zip(D.cls, D.vid, D.kstar.fillna(-1))]
    D["pred_dir_ok"] = [(p == "fwd" and s == 1) or (p == "bwd" and s == -1) for p, s in zip(D.pk_rel_pred, D.dirstar)]
    D["pred_any_ok"] = D.pk_rel_pred.notna()
    D["has_dir_truth"] = [any(truth.get((c, v, k), (None,))[0] in ("fwd", "bwd") for k in range(1, K + 1))
                          for c, v in zip(D.cls, D.vid)]
    D["pk_rel_pred_fwd"] = [truth.get((c, v, int(k)), (None,))[0] for c, v, k in zip(D.cls, D.vid, D.kstar_fwd)]
    E = D[D.in_pk_graph]
    sweep = []
    for c in sorted(set([2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 200] + list(cuts.values()))):
        pc = E.m <= c
        pf = E.m_fwd <= c
        sweep.append(dict(cut=c, n_cont=int(pc.sum()), frac_cont=round(float(pc.mean()), 4),
                          precision_dir=round(float(E[pc].pred_dir_ok.mean()), 4),
                          precision_any_packet=round(float(E[pc].pred_any_ok.mean()), 4),
                          recall_dir=round(float((pc & E.pred_dir_ok).sum() / E.has_dir_truth.sum()), 4),
                          fwd_only_n=int(pf.sum()),
                          fwd_only_precision=round(float((E[pf].pk_rel_pred_fwd == "fwd").mean()), 4),
                          fwd_only_recall=round(float((pf & (E.pk_rel_pred_fwd == "fwd")).sum() / E.has_dir_truth.sum()), 4)))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(T / "pr_sweep.csv", index=False)
    SUMMARY["packet_validation"] = {"clips_in_packet_graph": int(len(E)), "clips_with_dir_truth": int(E.has_dir_truth.sum()),
                                    "clips_not_in_packet_graph": int((~D.in_pk_graph).sum()),
                                    "not_in_graph_by_combo": D[~D.in_pk_graph].combo.fillna("NA").value_counts().head(8).to_dict()}
    gray_agree = ((D.kstar == D.kstar_gray) & (D.dirstar == D.dirstar_gray)).mean()
    SUMMARY["rgb_vs_gray"] = {"argmin_agreement": round(float(gray_agree), 4),
                              "median_m_rgb": round(float(D.m.median()), 2), "median_m_gray": round(float(D.m_gray.median()), 2)}
    # per class / per codec precision at the cuts
    pcs = {}
    for name, c in cuts.items():
        for grp in ("cls", "v_codec"):
            for g, e in E.groupby(grp):
                pc = e.m <= c
                pcs[f"{name}|{grp}={g}"] = dict(n=int(len(e)), n_cont=int(pc.sum()),
                                                precision_dir=round(float(e[pc].pred_dir_ok.mean()), 3) if pc.any() else None,
                                                recall_dir=round(float((pc & e.pred_dir_ok).sum() / max(1, e.has_dir_truth.sum())), 3))
    SUMMARY["packet_validation_by_group"] = pcs

    # disagreements at the primary cut
    dis = []
    for r in E.itertuples():
        if r.cont and not r.pred_any_ok:
            dis.append(dict(type="frames_cont_no_shared_packet", cls=r.cls, vid=r.vid, k=r.kstar, dir=r.dirstar, m=r.m,
                            combo=r.combo, same_c27=None))
        elif r.cont and r.pred_any_ok and not r.pred_dir_ok:
            dis.append(dict(type="frames_cont_packet_other_relation", cls=r.cls, vid=r.vid, k=r.kstar, dir=r.dirstar,
                            m=r.m, combo=r.combo, pk=r.pk_rel_pred))
        if r.has_dir_truth and not (r.cont and r.pred_dir_ok):
            tk = [(k, truth[(r.cls, r.vid, k)]) for k in range(1, K + 1)
                  if truth.get((r.cls, r.vid, k), (None,))[0] in ("fwd", "bwd")]
            dis.append(dict(type="packet_neighbour_missed" if not r.cont else "packet_neighbour_wrong_k", cls=r.cls,
                            vid=r.vid, k=r.kstar, dir=r.dirstar, m=r.m, combo=r.combo,
                            truth_k=tk[0][0], truth_rel=tk[0][1][0], truth_lag=tk[0][1][2]))
    dis = pd.DataFrame(dis)
    dis.to_csv(T / "disagreements.csv", index=False)
    dsum = {}
    for t, g in dis.groupby("type"):
        dsum[t] = {"n": int(len(g)), "by_cls": g.cls.value_counts().to_dict(),
                   "by_combo": g.combo.fillna("NA").value_counts().head(5).to_dict(), "m": pct(g.m, (25, 50, 75))}
        if "truth_lag" in g:
            dsum[t]["truth_lag"] = pct(g.truth_lag, (25, 50, 75))
            dsum[t]["truth_k"] = g.truth_k.value_counts().head(5).to_dict()
            dsum[t]["frames_k"] = g.k.value_counts().head(5).to_dict()
    SUMMARY["disagreements_primary"] = dsum

    # ---- segmentation, frames
    D["lab"] = np.where(D.cont, D.dirstar * D.kstar, 0).astype(int)
    D["lab_strict"] = np.where(D.cont_strict, D.dirstar * D.kstar, 0).astype(int)
    D["lab_mid"] = np.where(D.cont_intermediate, D.dirstar * D.kstar, 0).astype(int)
    # packets: signed ID step to the immediate packet successor (else from the predecessor)
    e = packet_successors(R, D)
    succ = e.sort_values(["off", "step"], key=lambda s: s.abs() if s.name == "step" else s).groupby(["cls", "src"]).head(1)
    pred = e.sort_values(["off"]).groupby(["cls", "dst"]).head(1)
    sdict = {(c, s): st for c, s, st in zip(succ.cls, succ.src, succ.step)}
    pdict = {(c, d): st for c, d, st in zip(pred.cls, pred.dst, pred.step)}
    D["pk_lab"] = [int(sdict.get((c, v), pdict.get((c, v), 0))) for c, v in zip(D.cls, D.vid)]
    for labcol, mj, rn in (("lab", "maj", "run"), ("lab_strict", "maj_strict", "run_strict"),
                           ("lab_mid", "maj_mid", "run_mid"), ("pk_lab", "pk_maj", "pk_run")):
        D[mj] = 0
        D[rn] = 0
        for cls in CLS:
            m = (D.cls == cls).values
            mj_, rn_ = majority_runs(D.vid.values[m], D[labcol].values[m])
            D.loc[m, mj] = mj_
            D.loc[m, rn] = rn_
    RT = runs_table(D, "lab", "maj", "run", R)
    RT.to_csv(T / "runs_frames.csv", index=False)
    RP = runs_table(D, "pk_lab", "pk_maj", "pk_run", R)
    RP.to_csv(T / "runs_packets.csv", index=False)
    agree = D[D.pk_maj != 0]
    seg = {}
    for name, (mj, labc) in {"primary": ("maj", "lab"), "strict": ("maj_strict", "lab_strict"),
                             "intermediate": ("maj_mid", "lab_mid")}.items():
        seg[name] = {
            "frac_clips_maj_eq_packet_maj": round(float((agree[mj] == agree.pk_maj).mean()), 4),
            "frac_clips_by_period": {int(k): round(float(v), 4) for k, v in D[mj].value_counts(normalize=True).items()},
            "frac_clips_by_period_per_cls": {c: {int(k): round(float(v), 4) for k, v in
                                                 D[D.cls == c][mj].value_counts(normalize=True).head(8).items()} for c in CLS},
            "n_runs": int(D.groupby(["cls", mj.replace("maj", "run")]).ngroups),
            "frac_cont": round(float((D[labc] != 0).mean()), 4)}
    seg["packets"] = {"frac_clips_by_period": {int(k): round(float(v), 4) for k, v in D.pk_maj.value_counts(normalize=True).items()},
                      "n_runs": int(D.groupby(["cls", "pk_run"]).ngroups),
                      "frac_clips_with_pk_lab": round(float((D.pk_lab != 0).mean()), 4)}
    big = RT[RT.n_clips >= 15]
    seg["runs_ge15_primary"] = {"n": int(len(big)), "clips": int(big.n_clips.sum()),
                                "clips_by_period": {int(k): int(v) for k, v in big.groupby("period").n_clips.sum().items()}}
    SUMMARY["segmentation"] = seg

    # ---- packet offsets by |ID diff| (item 2 of the packet brief)
    Q = R[(R.kind != "identical") & (R.id_diff.between(1, 3))].copy()
    Q["stride"] = Q.start_b_minus_a.abs()
    Q["time_dir"] = np.sign(Q.start_b_minus_a)
    rows = []
    for (cls, dd), g in Q.groupby(["cls_a", "id_diff"]):
        vc = g.stride.value_counts(normalize=True).head(5)
        rows.append(dict(cls=cls, id_diff=dd, n_pairs=len(g), frac_partial=round(float((g.kind == "partial").mean()), 3),
                         frac_contained=round(float(g.kind.str.startswith("contained").mean()), 3),
                         frac_higher_id_later=round(float((g.time_dir > 0).mean()), 3),
                         frac_higher_id_earlier=round(float((g.time_dir < 0).mean()), 3),
                         frac_same_start=round(float((g.time_dir == 0).mean()), 3),
                         median_stride=float(g.stride.median()),
                         top_strides="; ".join(f"{int(k)}:{v:.0%}" for k, v in vc.items()),
                         median_overlap_partial=float(g[g.kind == "partial"].overlap_frames.median())))
    OFF = pd.DataFrame(rows)
    OFF.to_csv(T / "packet_offsets_by_iddiff.csv", index=False)
    # by frame-run period (pairs whose two clips lie in the same frame run)
    runkey = {(c, v): (r, p) for c, v, r, p in zip(D.cls, D.vid, D.run, D.maj)}
    Q["run_a"] = [runkey.get((c, v), (None, None))[0] for c, v in zip(Q.cls_a, Q.id_a)]
    Q["run_b"] = [runkey.get((c, v), (None, None))[0] for c, v in zip(Q.cls_a, Q.id_b)]
    Q["period"] = [runkey.get((c, v), (None, None))[1] for c, v in zip(Q.cls_a, Q.id_a)]
    Q2 = Q[(Q.run_a == Q.run_b) & Q.period.notna()]
    rows = []
    for (cls, p, dd), g in Q2.groupby(["cls_a", "period", "id_diff"]):
        vc = g.stride.value_counts(normalize=True).head(4)
        rows.append(dict(cls=cls, period=int(p), id_diff=dd, n_pairs=len(g),
                         frac_partial=round(float((g.kind == "partial").mean()), 3),
                         frac_higher_id_later=round(float((g.time_dir > 0).mean()), 3),
                         median_stride=float(g.stride.median()),
                         top_strides="; ".join(f"{int(k)}:{v:.0%}" for k, v in vc.items()),
                         median_overlap=float(g.overlap_frames.median())))
    pd.DataFrame(rows).to_csv(T / "packet_offsets_by_period.csv", index=False)
    succ_all = e.sort_values("off").groupby(["cls", "src"]).head(1)
    SUMMARY["packet_successor_stride"] = {
        c: {"n": int(len(g)), "stride_top": {int(k): round(float(v), 3) for k, v in g.off.value_counts(normalize=True).head(6).items()},
            "overlap_frames": pct(g.n_src - g.off, (5, 25, 50, 75, 95)),
            "id_step_top": {int(k): round(float(v), 3) for k, v in g.step.value_counts(normalize=True).head(8).items()}}
        for c, g in succ_all.groupby("cls")}

    # ---- B. partners
    rng = np.random.default_rng(0)
    pk_pairs = {}
    for r in R[R.kind != "identical"].itertuples():
        pk_pairs[(r.cls_a, r.id_a, r.id_b)] = r.kind
    ident = {(r.cls_a, r.id_a, r.id_b) for r in R[R.kind == "identical"].itertuples()}
    info = D.set_index(["cls", "vid"])
    cluster_lab = D.groupby("c27").label.agg(lambda s: s.value_counts(normalize=True).to_dict())

    def pair_stats(pairs):
        if not len(pairs):
            return {}
        a = info.loc[[(c, x) for c, x, _ in pairs]]
        b = info.loc[[(c, y) for c, _, y in pairs]]
        same = lambda col: (a[col].values == b[col].values)
        fp = np.abs(a.first_pts_s.values - b.first_pts_s.values)
        pk = np.array([(c, min(x, y), max(x, y)) in pk_pairs for c, x, y in pairs])
        idt = np.array([(c, min(x, y), max(x, y)) in ident for c, x, y in pairs])
        return dict(n=len(pairs), same_label=round(float(same("label").mean()), 4),
                    same_c27=round(float(same("c27").mean()), 4), same_c17=round(float(same("c17").mean()), 4),
                    share_packets=round(float(pk.mean()), 4), identical_video_packets=round(float(idt.mean()), 4),
                    same_raw_frames=round(float(same("real_frames").mean()), 4),
                    raw_frames_within_1=round(float((np.abs(a.real_frames.values - b.real_frames.values) <= 1).mean()), 4),
                    same_320p_n_frames=round(float(same("n_frames").mean()), 4),
                    same_first_pts_1ms=round(float((fp < 1e-3).mean()), 4),
                    same_combo=round(float(same("combo").mean()), 4), same_n_audio=round(float(same("n_audio").mean()), 4),
                    same_md5=round(float(same("md5").mean()), 4),
                    first_first_rmse_median=round(float(np.nanmedian(np.array([info.loc[(c, x)][f"f{y - x}"] if 0 < y - x <= K else np.nan for c, x, y in pairs]))), 2))

    prow, prun = [], []
    allpairs = {"partner": [], "same_source": []}
    for (cls, rid), g in D.groupby(["cls", "run"]):
        p = abs(int(g.maj.iloc[0]))
        if p < 2 or len(g) < 10:
            continue
        ids = set(g.vid)
        part = [(cls, n, n + j) for n in sorted(ids) for j in range(1, p) if n + j in ids]
        same = [(cls, n, n + p) for n in sorted(ids) if n + p in ids]
        allpairs["partner"] += [(x, p) for x in part]
        allpairs["same_source"] += [(x, p) for x in same]
        trip = [n for n in sorted(ids) if all(n + j in ids for j in range(p))]
        fr = info.loc[[(cls, n) for n in sorted(ids)]].real_frames
        frd = dict(zip(sorted(ids), fr.values))
        trip_same = [len({frd[n + j] for j in range(p)}) == 1 for n in trip]
        phase = {}
        for ph in range(p):
            tt = [n for n in trip if (n - min(ids)) % p == ph]
            phase[ph] = np.mean([len({frd[n + j] for j in range(p)}) == 1 for n in tt]) if tt else np.nan
        ps, ss = pair_stats(part), pair_stats(same)
        prun.append(dict(cls=cls, id_start=int(g.vid.min()), id_end=int(g.vid.max()), period=int(g.maj.iloc[0]),
                         n_clips=len(g), combo=g.combo.mode().iloc[0] if g.combo.notna().any() else None,
                         partner_share_packets=ps.get("share_packets"), partner_same_c27=ps.get("same_c27"),
                         partner_same_c17=ps.get("same_c17"), partner_same_raw_frames=ps.get("same_raw_frames"),
                         partner_same_first_pts=ps.get("same_first_pts_1ms"), partner_ff_rmse=ps.get("first_first_rmse_median"),
                         same_source_share_packets=ss.get("share_packets"), same_source_same_c27=ss.get("same_c27"),
                         same_source_same_raw_frames=ss.get("same_raw_frames"),
                         tuples_all_same_raw_frames=round(float(np.mean(trip_same)), 3) if trip else None,
                         tuples_best_phase=round(float(np.nanmax(list(phase.values()))), 3) if trip else None,
                         n_partner_pairs=len(part)))
    PR = pd.DataFrame(prun)
    PR.to_csv(T / "partner_tests_by_run.csv", index=False)
    # random same-class pairs and random same-class same-combo pairs
    rnd, rnd_combo = [], []
    byc = {c: D[D.cls == c] for c in CLS}
    for c in CLS:
        g = byc[c]
        a = rng.choice(g.vid.values, 8000)
        b = rng.choice(g.vid.values, 8000)
        rnd += [(c, x, y) for x, y in zip(a, b) if x != y][:6000]
        cmb = dict(zip(g.vid, g.combo))
        rnd_combo += [(c, x, y) for x, y in zip(a, b) if x != y and cmb[x] == cmb[y]][:6000]
    stats = {"random_same_class": pair_stats(rnd), "random_same_class_same_combo": pair_stats(rnd_combo)}
    for p in (2, 3, 4):
        stats[f"partner_p{p}"] = pair_stats([x for x, q in allpairs["partner"] if q == p])
        stats[f"same_source_p{p}"] = pair_stats([x for x, q in allpairs["same_source"] if q == p])
    # adjacent pairs in period-1 runs as another reference
    p1 = [(c, n, n + 1) for (c, rid), g in D.groupby(["cls", "run"]) if abs(int(g.maj.iloc[0])) == 1 and len(g) >= 10
          for n in g.vid if n + 1 in set(g.vid)]
    stats["adjacent_in_period1_runs"] = pair_stats(p1)
    PT = pd.DataFrame(stats).T
    PT.to_csv(T / "partner_tests.csv")
    SUMMARY["partner_tests"] = stats
    # (i') the label is the class folder; the partner's cluster label purity instead
    pur = []
    for x, p in allpairs["partner"]:
        c, n, m_ = x
        ca, cb = info.loc[(c, n)].c27, info.loc[(c, m_)].c27
        la = info.loc[(c, n)].label
        pur.append((cluster_lab[ca].get(la, 0), cluster_lab[cb].get(la, 0)))
    pur = np.array(pur)
    SUMMARY["partner_cluster_label_share"] = {"own_cluster_same_label_share_median": float(np.median(pur[:, 0])) if len(pur) else None,
                                              "partner_cluster_same_label_share_median": float(np.median(pur[:, 1])) if len(pur) else None,
                                              "frac_partner_cluster_100pct_same_label": float((pur[:, 1] == 1).mean()) if len(pur) else None}

    # ---- C. chains
    def build_chains(cut):
        succ_, pred_ = {}, {}
        parent = {}

        def find(x):
            while parent.get(x, x) != x:
                parent[x] = parent.get(parent[x], parent[x])
                x = parent[x]
            return x
        links = []
        Pk = e[e.step.abs() <= K].copy()
        Pk["absstep"] = Pk.step.abs()
        Pk = Pk.sort_values(["off", "absstep"])
        valid = set(zip(D.cls, D.vid))
        for r in Pk.itertuples():
            s, d = (r.cls, r.src), (r.cls, r.dst)
            if s not in valid or d not in valid or s in succ_ or d in pred_ or find(s) == find(d):
                continue
            succ_[s], pred_[d] = d, s
            parent[find(d)] = find(s)
            links.append(dict(cls=r.cls, src=r.src, dst=r.dst, type="packet", link_kind="packet", off=r.off, n_src=r.n_src))
        cand = []
        for r in D.itertuples():
            for k in range(1, K + 1):
                if (r.cls, r.vid, r.vid + k) in pk_pairs or (r.cls, r.vid, r.vid + k) in ident:
                    continue
                if (r.cls, r.vid + k) not in valid:
                    continue
                # both files have packet data but share none: they do not overlap, so only an
                # abutting (frame-exact) junction can be a continuation -> cap the cut at `exact`
                both = r.in_pk_graph and inpk_node.get((r.cls, r.vid + k), False)
                cc = min(cut, exact) if both else cut
                dk, bk = getattr(r, f"d{k}"), getattr(r, f"b{k}")
                if np.isfinite(dk) and dk <= cc:
                    cand.append((dk, r.cls, r.vid, r.vid + k, "rmse_both_pk" if both else "rmse_no_pk"))
                if np.isfinite(bk) and bk <= cc:
                    cand.append((bk, r.cls, r.vid + k, r.vid, "rmse_both_pk" if both else "rmse_no_pk"))
        cand.sort()
        for dval, c, a, b, sub in cand:
            s, d = (c, a), (c, b)
            if s in succ_ or d in pred_ or find(s) == find(d):
                continue
            succ_[s], pred_[d] = d, s
            parent[find(d)] = find(s)
            links.append(dict(cls=c, src=a, dst=b, type="rmse", link_kind=sub, off=np.nan, n_src=np.nan))
        L = pd.DataFrame(links)
        # walk chains
        cid, pos = {}, {}
        k = 0
        for node in sorted(valid):
            if node in pred_:
                continue
            x, i = node, 0
            while True:
                cid[x], pos[x] = k, i
                if x not in succ_:
                    break
                x, i = succ_[x], i + 1
            k += 1
        return L, cid, pos, succ_, pred_

    def junction_d(c, s, d):
        r = info.loc[(c, s)] if d > s else info.loc[(c, d)]
        k = abs(d - s)
        return r[f"d{k}"] if d > s else r[f"b{k}"]

    inpk_node = dict(zip(zip(D.cls, D.vid), D.in_pk_graph))
    chain_res = {}
    for name, cut in [("primary", primary), ("strict", exact), ("intermediate", round(mid, 1))]:
        L, cid, pos, succ_, pred_ = build_chains(cut)
        D[f"chain_{name}"] = [cid[(c, v)] for c, v in zip(D.cls, D.vid)]
        D[f"chainpos_{name}"] = [pos[(c, v)] for c, v in zip(D.cls, D.vid)]
        L["d_junction"] = [junction_d(c, s, d) for c, s, d in zip(L.cls, L.src, L.dst)]
        L["split_src"] = [info.loc[(c, s)].split for c, s in zip(L.cls, L.src)]
        L["split_dst"] = [info.loc[(c, d)].split for c, d in zip(L.cls, L.dst)]
        L["label_src"] = [info.loc[(c, s)].label for c, s in zip(L.cls, L.src)]
        L["label_dst"] = [info.loc[(c, d)].label for c, d in zip(L.cls, L.dst)]
        L["junction_lag"] = L.n_src - 1 - L.off
        g = D.groupby(f"chain_{name}")
        CH = pd.DataFrame({"n_clips": g.size(), "cls": g.cls.first(), "id_min": g.vid.min(), "id_max": g.vid.max(),
                           "dur_sum_s": g.dur_s.sum(), "dur_sum_320p_s": g.n_frames.sum() / 30.0,
                           "n_train": g.split.apply(lambda s: (s == "train").sum()),
                           "n_val": g.split.apply(lambda s: (s == "val").sum()),
                           "n_test": g.split.apply(lambda s: (s == "test").sum()),
                           "n_clusters27": g.c27.nunique()})
        # span: packet links advance by the offset, rmse links by the source duration
        adv = {}
        for r in L.itertuples():
            adv[(r.cls, r.src)] = r.off / 30.0 if r.type == "packet" else info.loc[(r.cls, r.src)].dur_s
        D["_adv"] = [adv.get((c, v), dd) for c, v, dd in zip(D.cls, D.vid, D.dur_s)]
        CH["span_s"] = D.groupby(f"chain_{name}")._adv.sum()
        CH["n_rmse_links"] = L[L.type == "rmse"].assign(ch=[cid[(c, s)] for c, s in zip(L[L.type == "rmse"].cls, L[L.type == "rmse"].src)]).groupby("ch").size() if (L.type == "rmse").any() else 0
        CH["n_rmse_links"] = CH.n_rmse_links.fillna(0).astype(int)
        multi = CH[CH.n_clips >= 2]
        sp = D.split.value_counts(normalize=True)
        res = {"cut": cut, "n_links": int(len(L)), "n_packet_links": int((L.type == "packet").sum()),
               "n_rmse_links": int((L.type == "rmse").sum()),
               "n_links_by_kind": L.link_kind.value_counts().to_dict(),
               "n_chains_ge2": int(len(multi)), "n_singletons": int((CH.n_clips == 1).sum()),
               "chain_len_clips": pct(multi.n_clips, (25, 50, 75, 90, 99)) | {"max": int(multi.n_clips.max())},
               "chain_span_s": pct(multi.span_s, (25, 50, 75, 90, 99)) | {"max": round(float(multi.span_s.max()), 1)},
               "chain_dur_sum_s": pct(multi.dur_sum_s, (25, 50, 75, 90)),
               "share_clips_in_chains_ge5": round(float(CH[CH.n_clips >= 5].n_clips.sum() / len(D)), 4),
               "share_clips_in_chains_ge2": round(float(multi.n_clips.sum() / len(D)), 4),
               "label_changes_along_links": int((L.label_src != L.label_dst).sum()),
               "frac_links_different_split": round(float((L.split_src != L.split_dst).mean()), 4),
               "expected_frac_different_split_if_independent": round(float(1 - (sp ** 2).sum()), 4),
               "frac_links_different_split_by_type": {t: round(float((g.split_src != g.split_dst).mean()), 4) for t, g in L.groupby("type")},
               "chains_ge2_spanning_train_and_test": int(((multi.n_train > 0) & (multi.n_test > 0)).sum()),
               "d_junction_packet": pct(L[L.type == "packet"].d_junction, (5, 25, 50, 75, 95)),
               "d_junction_rmse": pct(L[L.type == "rmse"].d_junction, (5, 25, 50, 75, 95)),
               "d_junction_over_raw_floor": pct(L.d_junction / raw_floor, (25, 50, 75)),
               "packet_junction_lag_frames": pct(L[L.type == "packet"].junction_lag, (5, 25, 50, 75, 95)),
               "packet_link_stride_top": {int(k): round(float(v), 3) for k, v in L[L.type == "packet"].off.value_counts(normalize=True).head(6).items()},
               "packet_link_overlap_frames": pct(L[L.type == "packet"].n_src - L[L.type == "packet"].off, (5, 25, 50, 75, 95)),
               "rmse_links_by_combo": D.set_index(["cls", "vid"]).loc[list(zip(L[L.type == "rmse"].cls, L[L.type == "rmse"].src))].combo.fillna("NA").value_counts().head(6).to_dict() if (L.type == "rmse").any() else {}}
        chain_res[name] = res
        if name == "primary":
            L.to_csv(T / "chain_links.csv", index=False)
            CH.to_csv(T / "chains.csv")
            Lp, CHp = L, CH
            succ_p, pred_p = succ_, pred_
    SUMMARY["chains"] = chain_res
    D.drop(columns=["_adv"], inplace=True)

    # ---- label runs in time: link chain tails to chain heads across all classes (32x32 thumbs, then 128x72)
    c = load_cache(REPO / "outputs" / "review_cache")
    th = c["thumbs"].astype(np.float32)
    rowof = dict(zip(zip(D.cls, D.vid), D.row))
    heads = [n for n in zip(D.cls, D.vid) if n not in pred_p]
    tails = [n for n in zip(D.cls, D.vid) if n not in succ_p]
    Hf = th[[rowof[n] for n in heads], 0].reshape(len(heads), -1)
    Tl = th[[rowof[n] for n in tails], 2].reshape(len(tails), -1)
    d2 = (Tl ** 2).sum(1)[:, None] + (Hf ** 2).sum(1)[None, :] - 2 * Tl @ Hf.T
    dt = np.sqrt(np.maximum(d2, 0) / Tl.shape[1])
    chain_of = {n: D.loc[i, "chain_primary"] for i, n in enumerate(zip(D.cls, D.vid))}
    tc = np.array([chain_of[n] for n in tails])
    hc = np.array([chain_of[n] for n in heads])
    dt[tc[:, None] == hc[None, :]] = np.inf
    parts = [np.load(REPO / "outputs" / "review_cache" / f"frames_{s}_hires.npz", allow_pickle=True) for s in ("train", "val", "test")]
    first = np.concatenate([p["first"] for p in parts])
    last = np.concatenate([p["last"] for p in parts])
    cand = []
    for i in range(len(tails)):
        for j in np.argsort(dt[i])[:3]:
            if not np.isfinite(dt[i, j]):
                continue
            a, b = rowof[tails[i]], rowof[heads[j]]
            dh = float(np.sqrt(((last[a].astype(np.float32) - first[b].astype(np.float32)) ** 2).mean()))
            cand.append((dh, float(dt[i, j]), tails[i], heads[j]))
    del first, last
    cand.sort()
    sup_res = {}
    for cname, cut in [("intermediate", round(mid, 1)), ("strict", exact), ("primary", primary)]:
        used_t, used_h, parent = set(), set(), {}

        def find(x):
            while parent.get(x, x) != x:
                x = parent[x]
            return x
        sl = []
        for dh, d32, tnode, hnode in cand:
            if dh > cut:
                break
            ct, ch_ = chain_of[tnode], chain_of[hnode]
            if tnode in used_t or hnode in used_h or find(ct) == find(ch_):
                continue
            used_t.add(tnode); used_h.add(hnode); parent[find(ch_)] = find(ct)
            sl.append(dict(tail_cls=tnode[0], tail_id=tnode[1], head_cls=hnode[0], head_id=hnode[1], d128=dh, d32=d32,
                           tail_label=info.loc[tnode].label, head_label=info.loc[hnode].label,
                           tail_split=info.loc[tnode].split, head_split=info.loc[hnode].split,
                           id_gap=(hnode[1] - tnode[1]) if hnode[0] == tnode[0] else np.nan))
        SL = pd.DataFrame(sl)
        if cname == "intermediate":
            SL.to_csv(T / "superchain_links.csv", index=False)
        # walk super-chains: order chains, then label runs
        nxt = {chain_of[(r.tail_cls, r.tail_id)]: chain_of[(r.head_cls, r.head_id)] for r in SL.itertuples()}
        prv = set(nxt.values())
        chain_lab = D.groupby("chain_primary").label.first()
        chain_dur = Lp_span = CHp.span_s
        runs_c, runs_s, nchg = [], [], 0
        for ch0 in CHp.index:
            if ch0 in prv:
                continue
            seq = [ch0]
            while seq[-1] in nxt:
                seq.append(nxt[seq[-1]])
            cur_l, cur_n, cur_s = chain_lab[seq[0]], 0, 0.0
            for chx in seq:
                if chain_lab[chx] != cur_l:
                    runs_c.append(cur_n); runs_s.append(cur_s); nchg += 1
                    cur_l, cur_n, cur_s = chain_lab[chx], 0, 0.0
                cur_n += int(CHp.loc[chx, "n_clips"]); cur_s += float(chain_dur[chx])
            runs_c.append(cur_n); runs_s.append(cur_s)
        sup_res[cname] = {"cut": cut, "n_links": int(len(SL)),
                          "n_cross_class_links": int((SL.tail_cls != SL.head_cls).sum()) if len(SL) else 0,
                          "cross_class_pairs": SL[SL.tail_cls != SL.head_cls].groupby(["tail_cls", "head_cls"]).size().to_dict() if len(SL) else {},
                          "same_class_id_gap": pct(SL.id_gap.abs(), (25, 50, 75)) if len(SL) else {},
                          "label_changes": nchg,
                          "label_run_clips": pct(runs_c, (25, 50, 75, 90, 99)) | {"max": int(max(runs_c))},
                          "label_run_s": pct(runs_s, (25, 50, 75, 90, 99)) | {"max": round(float(max(runs_s)), 1)},
                          "n_label_runs": len(runs_c)}
    SUMMARY["superchains"] = {k: {kk: (str(vv) if kk == "cross_class_pairs" else vv) for kk, vv in v.items()} for k, v in sup_res.items()}

    # ---- split leakage along chains / partners / packets
    te = D[D.split == "test"]
    def nb_in_train(n):
        out = []
        for dct in (succ_p, pred_p):
            if n in dct:
                out.append(info.loc[dct[n]].split == "train")
        return any(out)
    part_of = {}
    for (cls, rid), g in D.groupby(["cls", "run"]):
        p = abs(int(g.maj.iloc[0]))
        if p < 2 or len(g) < 10:
            continue
        ids = set(g.vid)
        for n in ids:
            # simultaneous views: the other members of N's time step (same block of p IDs)
            part_of[(cls, n)] = [n + j for j in range(-(p - 1), p) if j != 0 and n + j in ids]
    def partner_in_train(n):
        return any(info.loc[(n[0], m_)].split == "train" for m_ in part_of.get(n, []))
    # strict partner: only the partners of N's own block (phase from the packet links: the block
    # members are those that are not same-source to each other); approximate by |j| < p (both sides)
    tn = [(c, v) for c, v in zip(te.cls, te.vid)]
    temporal = np.array([nb_in_train(n) for n in tn])
    partner = np.array([partner_in_train(n) for n in tn])
    has_partner = np.array([n in part_of for n in tn])
    chain_train = np.array([CHp.loc[chain_of[n], "n_train"] > 0 for n in tn])
    # packets: footage shared with train (any ID distance) and footage entirely in train
    z = np.load(REPO / "outputs" / "research" / "raw_video_packets.npz")
    relz = z["relpath"]
    big_ = z["size"] >= 1024
    fz, hz = z["fidx"][big_], z["hash"][big_]
    splitof = dict(zip(D.rel, D.split))
    fsplit = np.array([splitof.get(r, "NA") for r in relz])
    tr_h = np.unique(hz[fsplit[fz] == "train"])
    rows_cov = []
    for s in ("test", "val"):
        m = fsplit[fz] == s
        inn = np.isin(hz[m], tr_h)
        cov = pd.DataFrame({"f": fz[m], "in": inn}).groupby("f")["in"].agg(["mean", "size"])
        rows_cov.append((s, cov))
    covt = rows_cov[0][1]
    covt.index = relz[covt.index]
    te_cov = te.rel.map(covt["mean"]).fillna(0).values
    leak = {"n_test": int(len(te)),
            "frac_test_pred_or_succ_in_train": round(float(temporal.mean()), 4),
            "frac_test_in_multicam_run": round(float(has_partner.mean()), 4),
            "frac_test_partner_in_train_among_multicam": round(float(partner[has_partner].mean()), 4) if has_partner.any() else None,
            "frac_test_partner_in_train": round(float(partner.mean()), 4),
            "frac_test_temporal_or_partner_in_train": round(float((temporal | partner).mean()), 4),
            "frac_test_chain_contains_train": round(float(chain_train.mean()), 4),
            "frac_test_shares_footage_with_train_packets": round(float((te_cov > 0).mean()), 4),
            "frac_test_footage_entirely_in_train_packets": round(float((te_cov == 1).mean()), 4),
            "frac_test_footage_ge50pct_in_train": round(float((te_cov >= 0.5).mean()), 4),
            "share_of_test_packets_in_train": round(float(np.isin(hz[fsplit[fz] == 'test'], tr_h).mean()), 4),
            "frac_test_temporal_or_partner_or_packet": round(float((temporal | partner | (te_cov > 0)).mean()), 4)}
    covv = rows_cov[1][1]
    covv.index = relz[covv.index]
    va = D[D.split == "val"]
    vcov = va.rel.map(covv["mean"]).fillna(0).values
    leak["frac_val_shares_footage_with_train_packets"] = round(float((vcov > 0).mean()), 4)
    leak["frac_val_footage_entirely_in_train_packets"] = round(float((vcov == 1).mean()), 4)
    leak["by_class_test_shares_footage"] = {c_: round(float((te_cov[(te.cls == c_).values] > 0).mean()), 4) for c_ in CLS}
    leak["by_class_test_temporal_in_train"] = {c_: round(float(temporal[(te.cls == c_).values].mean()), 4) for c_ in CLS}
    SUMMARY["split_leak"] = leak
    D["test_cov_train"] = D.rel.map(covt["mean"])

    keep = ["row", "path", "rel", "cls", "vid", "label", "split", "n_frames", "real_frames", "dur_s", "first_pts_s", "combo",
            "n_audio", "batch", "c27", "c17", "fl", "m", "kstar", "dirstar", "second", "m_fwd", "kstar_fwd", "m_gray",
            "cont_strict", "cont_intermediate", "cont_primary_valley", "lab", "maj", "run", "lab_strict", "maj_strict",
            "lab_mid", "maj_mid", "pk_lab", "pk_maj", "pk_run", "in_pk_graph", "pk_rel_pred", "has_dir_truth",
            "chain_primary", "chainpos_primary", "chain_strict", "chain_intermediate", "test_cov_train"] + \
           [f"{p}{k}" for p in ("d", "b", "f") for k in range(1, K + 1)]
    D[keep].to_csv(T / "clip_temporal.csv", index=False)
    SUMMARY["wall_s"] = round(time.time() - t0, 1)
    with open(T / "summary.json", "w") as fh:
        json.dump(SUMMARY, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(SUMMARY, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:20000])
    print(f"wall {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
