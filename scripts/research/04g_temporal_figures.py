#!/usr/bin/env python
"""Figures for 06_temporal_structure.md (A-C): ID-axis profiles, multi-camera runs, a chain, the
continuation histogram against packet truth, and the audio synchrony test.

Inputs: outputs/research/temporal/{clip_temporal.csv, runs_frames.csv, chain_links.csv, summary.json,
audio_columns.csv, audio_null.csv, ref_frame_lags.csv}, outputs/research/raw_audit_id_runs.csv,
outputs/review_cache/frames_*_hires.npz.
Outputs (docs/research/figures/): temporal_id_axis_{low,mid,high}.png, temporal_triplets.png,
temporal_chain.png, temporal_min_d_hist.png, temporal_audio_sync.png (150 dpi).
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04g_temporal_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
T = REPO / "outputs" / "research" / "temporal"
FIG = REPO / "docs" / "research" / "figures"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SPLIT_C = {"train": PAL[0], "val": PAL[1], "test": PAL[2]}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 150,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


def load_first():
    parts = [np.load(REPO / "outputs" / "review_cache" / f"frames_{s}_hires.npz", allow_pickle=True)
             for s in ("train", "val", "test")]
    return np.concatenate([p["first"] for p in parts])


def period_color(p):
    return PAL[(abs(int(p)) - 1) % len(PAL)] if p != 0 else "#bdbcb6"


def fig_id_axis(C, S, idruns):
    ref = S["reference_scales"]
    lines = [(ref["same_raw_frame_across_clips_lag0"]["p50"], "same raw frame in two clips"),
             (ref["raw_consecutive_frame_step_t3_11_per_clip_median"]["p50"], "consecutive raw frames"),
             (ref["within_clip_first_last_rgb"]["p50"], "first vs last frame of one clip"),
             (ref["random_same_class_first_first_rgb"]["p50"], "random same-class pair")]
    for cls in ("low", "mid", "high"):
        g = C[C.cls == cls].sort_values("vid")
        x = g.vid.values
        fig, ax = plt.subplots(3, 1, figsize=(16, 9.4), sharex=True, gridspec_kw={"height_ratios": [3, 3, 1.6]})
        for a, pre, title in ((ax[0], "d", "forward: RMSE(last(N), first(N+k))  (N+k starts where N ends)"),
                              (ax[1], "b", "backward: RMSE(last(N+k), first(N))  (N+k ends where N starts)")):
            for k in (1, 2, 3):
                y = pd.Series(g[f"{pre}{k}"].values).rolling(15, center=True, min_periods=5).median()
                a.plot(x, y, color=PAL[k - 1], lw=1.4, label=f"k = {k}")
            a.set_yscale("log")
            a.set_ylim(1.5, 130)
            for v, lab in lines:
                a.axhline(v, color=INK2, lw=0.8, ls=(0, (4, 3)))
                a.text(x.max() + 8, v, f"{lab} ({v:.1f})", va="center", fontsize=7.5, color=INK2)
            a.set_ylabel("rolling median RMSE\n(128x72, 0-255)")
            a.set_title(title, loc="left", fontsize=10, fontweight="normal")
            a.legend(loc="lower left", ncol=3, fontsize=8, frameon=True, framealpha=0.9, edgecolor="none")
        # strips: frame period, packet period, raw batch combo
        a = ax[2]
        for yrow, col, lab in ((2, "maj", "period, frames (primary cut)"), (1, "pk_maj", "period, packets (exact)")):
            v = g[col].values
            starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
            ends = np.r_[starts[1:], len(v)]
            for s_, e in zip(starts, ends):
                p = v[s_]
                a.broken_barh([(x[s_] - 0.5, x[e - 1] - x[s_] + 1)], (yrow - 0.4, 0.8), facecolors=period_color(p),
                              hatch="////" if p < 0 else None, edgecolor="white" if p >= 0 else INK2, linewidth=0)
            a.text(x.min() - 8, yrow, lab, ha="right", va="center", fontsize=8, color=INK)
        cb = g.combo.fillna("(unreadable)").map(lambda c: " ".join(c.split("|")[:3]))
        top = cb.value_counts().index[:5].tolist()
        ccol = {c: col for c, col in zip(top, ["#cfcdc6", "#7a5195", "#ffa600", "#003f5c", "#bc5090"])}
        v = cb.values
        starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
        ends = np.r_[starts[1:], len(v)]
        for s_, e in zip(starts, ends):
            a.broken_barh([(x[s_] - 0.5, x[e - 1] - x[s_] + 1)], (-0.4, 0.8), facecolors=ccol.get(v[s_], "#ffffff"),
                          edgecolor=INK2 if v[s_] not in ccol else "none", linewidth=0.3)
        a.text(x.min() - 8, 0, "raw batch (codec, size, audio)", ha="right", va="center", fontsize=8, color=INK)
        a.set_yticks([])
        a.set_ylim(-0.6, 2.6)
        a.grid(False)
        hs = [Patch(facecolor=period_color(p), label=f"|period| {p}") for p in range(1, 7)] + \
             [Patch(facecolor="white", edgecolor=INK2, hatch="////", label="hatched: IDs run backwards"),
              Patch(facecolor=period_color(0), label="no continuation")]
        hs2 = [Patch(facecolor=c, label=k) for k, c in ccol.items()] + [Patch(facecolor="white", edgecolor=INK2, label="other")]
        l1 = a.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=8, fontsize=8, frameon=False)
        a.add_artist(l1)
        a.legend(handles=hs2, loc="upper center", bbox_to_anchor=(0.5, -0.62), ncol=6, fontsize=8, frameon=False,
                 title="raw batch", title_fontsize=8)
        a.set_xlabel(f"clip ID (videos/{cls}/viewN.mp4)")
        # raw-metadata batch boundaries (runs of constant combo of >= 10 clips)
        rr = idruns[idruns.cls == cls].sort_values("id_from")
        for r in rr.itertuples():
            if r.n >= 10:
                for a in ax:
                    a.axvline(r.id_from - 0.5, color="#8c8b85", lw=0.7, alpha=0.8)
        ax[0].set_xlim(x.min() - 2, x.max() + 2)
        fig.suptitle(f"OUC-CGE '{cls}': frame distance to clip N+k along the ID axis "
                     f"(window 15; vertical lines: raw container-metadata batches >= 10 clips)",
                     x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0.08, 0.04, 0.88, 0.97))
        fig.savefig(FIG / f"temporal_id_axis_{cls}.png")
        plt.close(fig)


def tile(ax, img, cap, split=None):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(SPLIT_C.get(split, INK2))
        s.set_linewidth(2.5 if split else 0.5)
    ax.set_title(cap, fontsize=7.5, fontweight="normal", color=INK, pad=2)


def fig_triplets(C, F, A):
    info = C.set_index(["cls", "vid"])
    picks = [("high", 139, 314, 3, 200), ("high", 589, 722, 2, 650)]
    fig = plt.figure(figsize=(15, 9.6))
    sub = fig.subfigures(1, 2, width_ratios=[3, 2], wspace=0.02)
    for sf, (cls, a0, a1, p, start) in zip(sub, picks):
        start = a0 + ((start - a0) // p) * p           # align to the run's column phase
        rows = 12 // p
        axs = sf.subplots(rows, p)
        for t in range(rows):
            for j in range(p):
                v = start + t * p + j
                r = info.loc[(cls, v)]
                tile(axs[t, j], F[int(r.row)], f"{cls} {v} · L{int(r.label)} · {r.split} · c{int(r.c27)}", r.split)
            axs[t, 0].set_ylabel(f"step {t}", fontsize=8)
        for j in range(p):
            axs[0, j].annotate(f"column {j}", xy=(0.5, 1.32), xycoords="axes fraction", ha="center", fontsize=9,
                               fontweight="bold", color=INK)
        q = A[(A.cls == cls) & (A.id_start == a0)]
        sync = "; ".join(f"c{int(r.col_a)}-c{int(r.col_b)}: r={r.score:.2f}, offset {abs(r.lag_s):.1f} s" +
                         (" (sync)" if r.sync else "") for r in q.itertuples())
        sf.suptitle(f"period-{p} run {cls} {a0}-{a1}: IDs N, N+1{', N+2' if p == 3 else ''} = columns\n"
                    f"audio: {sync}", fontsize=9.5, x=0.02, ha="left")
    hs = [Patch(facecolor="white", edgecolor=c, linewidth=2.5, label=s) for s, c in SPLIT_C.items()]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, title="frame colour = published split", fontsize=9)
    fig.savefig(FIG / "temporal_triplets.png", bbox_inches="tight")
    plt.close(fig)


def fig_chain(C, F, L):
    info = C.set_index(["cls", "vid"])
    ch = C.groupby("chain_primary").size().sort_values(ascending=False)
    # longest chain whose middle 12 clips contain train, val and test
    for cid in ch.index[:20]:
        g = C[C.chain_primary == cid].sort_values("chainpos_primary")
        m = len(g) // 2
        w = g.iloc[max(0, m - 6): m + 6]
        if set(w.split) >= {"train", "test"}:
            break
    nxt = {(r.cls, r.src): r for r in L.itertuples()}
    fig, axs = plt.subplots(2, 6, figsize=(16, 5.2))
    for ax, r in zip(axs.ravel(), w.itertuples()):
        lk = nxt.get((r.cls, r.vid))
        if lk is not None:
            how = f"stride {int(lk.off)} fr" if lk.type == "packet" else "frame link"
            j = f"→ {lk.dst}: d={lk.d_junction:.1f}, {how}"
        else:
            j = "(chain end)"
        tile(ax, F[int(r.row)], f"{r.cls} {r.vid} · L{r.label} · {r.split}\n{j}", r.split)
    fig.suptitle(f"12 consecutive clips of chain {cid} ({len(g)} clips, '{g.cls.iloc[0]}', IDs {g.vid.min()}-{g.vid.max()}): "
                 f"first frames in time order; d = RMSE(last(N), first(next)) at 128x72", x=0.01, ha="left", fontsize=11)
    hs = [Patch(facecolor="white", edgecolor=c, linewidth=2.5, label=s) for s, c in SPLIT_C.items()]
    fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(FIG / "temporal_chain.png")
    plt.close(fig)
    return cid, w


def fig_hist(C, S):
    cuts = S["cuts"]
    ref = S["reference_scales"]
    bins = np.logspace(0, np.log10(140), 60)
    cat = np.select([~C.in_pk_graph, C.pk_rel_pred.isna(),
                     ((C.pk_rel_pred == "fwd") & (C.dirstar == 1)) | ((C.pk_rel_pred == "bwd") & (C.dirstar == -1))],
                    ["file shares no packet with any file", "predicted pair shares no packet",
                     "predicted pair overlaps in the predicted direction (packets)"], "predicted pair shares packets, other relation")
    order = ["predicted pair overlaps in the predicted direction (packets)", "predicted pair shares packets, other relation",
             "predicted pair shares no packet", "file shares no packet with any file"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [3, 2]})
    a = ax[0]
    a.hist([C.m[cat == o].dropna() for o in order], bins=bins, stacked=True, color=[PAL[0], PAL[3], PAL[1], "#bdbcb6"],
           label=order, edgecolor="white", linewidth=0.4)
    a.hist(C.m_fwd.dropna(), bins=bins, histtype="step", color=INK, lw=1.2, label="forward-only min_k d_k (all clips)")
    a.set_xscale("log")
    for (name, v), ls in zip(cuts.items(), [":", "--", "-"]):
        a.axvline(v, color=PAL[7], ls=ls, lw=1.3)
        a.text(v * 1.03, a.get_ylim()[1] * 0.97, f"{name.replace('_', ' ')}\n{v:.1f}", color=PAL[7], fontsize=8, va="top")
    a.set_xlabel("m(N) = min over k = 1..8 and both directions of RMSE(last, first)  (128x72, log scale)")
    a.set_ylabel("clips")
    a.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62))
    a.set_title("best temporal neighbour per clip, coloured by packet ground truth", loc="left")
    # junction RMSE against exact frame lag
    b = ax[1]
    jb = ref["junction_lag_bins"]
    labs = list(jb.keys())
    med = [jb[k]["p50"] for k in labs]
    lo = [jb[k]["p25"] for k in labs]
    hi = [jb[k]["p75"] for k in labs]
    xs = np.arange(len(labs))
    b.vlines(xs, lo, hi, color=PAL[0], lw=5, alpha=0.5, label="packet-verified junctions, IQR")
    b.plot(xs, med, "o", color=PAL[0], ms=8, label="median")
    b.axhline(ref["same_raw_frame_across_clips_lag0"]["p50"], color=INK2, ls=(0, (4, 3)), lw=0.8)
    b.axhline(ref["raw_consecutive_frame_step_t3_11_per_clip_median"]["p50"], color=INK2, ls=":", lw=0.8)
    b.text(len(labs) - 0.6, ref["same_raw_frame_across_clips_lag0"]["p50"] * 0.93, "same raw frame", fontsize=7.5, ha="right", va="top", color=INK2)
    b.text(-0.3, ref["raw_consecutive_frame_step_t3_11_per_clip_median"]["p50"] * 1.07, "consecutive raw frames", fontsize=7.5, color=INK2)
    b.set_xticks(xs, [f"{k}\n({jb[k]['n']})" for k in labs], fontsize=8)
    b.set_yscale("log")
    b.set_xlabel("exact lag between last(N) and first(N+k), frames (n pairs)")
    b.set_ylabel("RMSE(last(N), first(N+k))")
    b.set_title("RMSE at the junction vs packet-exact lag", loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "temporal_min_d_hist.png")
    plt.close(fig)


def fig_audio(A, N):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    groups = [("null: columns of unrelated runs", N.score, "#8c8b85"),
              ("column pairs: same camera (shared packets)", A[A.same_camera].score, PAL[3]),
              ("column pairs: different cameras", A[~A.same_camera].score, PAL[0])]
    rng = np.random.default_rng(0)
    for i, (lab, v, c) in enumerate(groups):
        y = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(v, y, s=22, color=c, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(-0.13, i, f"{lab}\n(n={len(v)})", ha="right", va="center", fontsize=8.5, color=INK)
    thr = float(N.score.max())
    ax.axvline(thr, color=PAL[7], ls="--", lw=1.2)
    ax.text(thr + 0.01, 2.45, f"max of null = {thr:.2f}", color=PAL[7], fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.12, 1.0)
    ax.set_ylim(-0.5, 2.6)
    ax.set_xlabel("max correlation of 100 Hz log-loudness envelopes over lags |l| <= 30 s (120-s windows)")
    ax.set_title("Audio synchrony of camera columns in multi-stream runs", loc="left")
    fig.tight_layout(rect=(0.22, 0, 1, 1))
    fig.savefig(FIG / "temporal_audio_sync.png")
    plt.close(fig)


def main():
    C = pd.read_csv(T / "clip_temporal.csv")
    S = json.load(open(T / "summary.json"))
    idruns = pd.read_csv(REPO / "outputs" / "research" / "raw_audit_id_runs.csv")
    A = pd.read_csv(T / "audio_columns.csv")
    N = pd.read_csv(T / "audio_null.csv")
    L = pd.read_csv(T / "chain_links.csv")
    fig_id_axis(C, S, idruns)
    fig_hist(C, S)
    fig_audio(A, N)
    F = load_first()
    fig_triplets(C, F, A)
    cid, w = fig_chain(C, F, L)
    print("chain figure:", cid, w[["cls", "vid", "split"]].values.tolist())


if __name__ == "__main__":
    main()
