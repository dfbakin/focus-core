#!/usr/bin/env python
"""Leak figures for 06_temporal_structure.md (D4-D7), from the 32x32 thumbnails of the review cache.

Clip distance (as in docs/review/01_leakage_analysis.md 2a / 3b): d(i, j) = min over the 3x3
(first/centre/last) frame pairs of the RMSE between 32x32 RGB thumbnails (0-255). The full 7,701 x
7,701 matrix is computed in float32 row blocks (||a||^2 + ||b||^2 - 2ab) and kept in memory (237 MB).

  leak_nn_pairs_hires.png     8 random test clips (default_rng(0)) beside their nearest train clip,
                              128x72 first frames, with d, labels, IDs, clusters (27.5), packet overlap
  leak_clusters_overview.png  the 20 largest clusters at thr 27.5: medoid first frame (128x72), size,
                              label mix, split mix
  leak_tsne.png               PCA(50) -> t-SNE (perplexity 30, random_state 0) of the centre thumbnails,
                              coloured by cluster (12 largest), label, split
  leak_distance_hist.png      test -> nearest-train d under the published split and grp27a/grp27b/grp17,
                              adjacent IDs (N, N+1), random same- and cross-class pairs (20,000 each)
Also writes outputs/research/temporal/leak_nn_by_split.csv and leak_figures_summary.json.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04h_leak_figures.py
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import load_cache  # noqa: E402

T = REPO / "outputs" / "research" / "temporal"
FIG = REPO / "docs" / "research" / "figures"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
SPLIT_C = {"train": PAL[0], "val": PAL[1], "test": PAL[2]}
LAB_C = {0: PAL[0], 1: PAL[1], 2: PAL[2]}
LAB_N = {0: "low (0)", 1: "mid (1)", 2: "high (2)"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
                     "axes.titlesize": 11, "axes.titleweight": "bold"})


def min_rmse_matrix(T3):
    """T3: (N, 3, D) float32. Returns (N, N) float32 min-over-frame-pairs RMSE."""
    N, F, Dd = T3.shape
    sq = (T3 ** 2).sum(2)                                    # (N, 3)
    out = np.empty((N, N), dtype=np.float32)
    for s in range(0, N, 1000):
        blk = None
        for a in range(F):
            A = T3[s:s + 1000, a]
            for b in range(F):
                d2 = sq[s:s + 1000, a][:, None] + sq[:, b][None, :] - 2.0 * (A @ T3[:, b].T)
                blk = d2 if blk is None else np.minimum(blk, d2)
        out[s:s + 1000] = np.sqrt(np.maximum(blk, 0) / Dd)
    out = np.minimum(out, out.T)
    np.fill_diagonal(out, 0)
    return out


def tile(ax, img, cap, edge=None):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(edge or INK2); s.set_linewidth(2.5 if edge else 0.5)
    ax.set_title(cap, fontsize=7.5, fontweight="normal", color=INK, pad=2)


def mix(s, names=None):
    vc = s.value_counts()
    return "/".join(str(vc.get(k, 0)) for k in (names or sorted(vc.index)))


def main():
    t0 = time.time()
    c = load_cache(REPO / "outputs" / "review_cache")
    ok = c["ok"]
    df = pd.DataFrame({"path": c["path"][ok], "label": c["label"][ok], "split": c["split"][ok],
                       "cls": c["cls_folder"][ok], "vid": c["view_id"][ok]})
    thumbs = c["thumbs"][ok]
    parts = [np.load(REPO / "outputs" / "review_cache" / f"frames_{s}_hires.npz", allow_pickle=True) for s in ("train", "val", "test")]
    first = np.concatenate([p["first"] for p in parts])[ok]
    sc = pd.read_csv(REPO / "outputs" / "review_out" / "scene_clusters.csv")
    df = df.merge(sc[["path", "cluster"]], on="path", how="left")
    for name in ("grp27a", "grp27b", "grp17"):
        g = pd.read_csv(REPO / "docs" / "review" / "data" / "grouped" / "splits" / f"split_{name}.csv")
        df = df.merge(g[["path", "split"]].rename(columns={"split": f"split_{name}"}), on="path", how="left")
    R = pd.read_csv(T / "packet_pairs_offsets.csv")
    pk = {(a, b) for a, b in zip(R.rel_a, R.rel_b)} | {(b, a) for a, b in zip(R.rel_a, R.rel_b)}
    df["rel"] = df.path.str.replace("videos/", "", regex=False)
    T3 = thumbs.reshape(len(df), 3, -1).astype(np.float32)
    D = min_rmse_matrix(T3)
    print(f"distance matrix {D.shape} in {time.time() - t0:.1f} s", flush=True)
    rng = np.random.default_rng(0)
    summ = {}

    # ---- nearest train under four splits
    nn_rows = []
    for col in ("split", "split_grp27a", "split_grp27b", "split_grp17"):
        tr = np.flatnonzero(df[col].values == "train")
        te = np.flatnonzero(df[col].values == "test")
        sub = D[np.ix_(te, tr)]
        j = sub.argmin(1)
        nn_rows.append(pd.DataFrame({"split_def": col.replace("split_", "") if col != "split" else "published",
                                     "test_path": df.path.values[te], "nn_train_path": df.path.values[tr[j]],
                                     "d": sub[np.arange(len(te)), j]}))
    NN = pd.concat(nn_rows)
    NN.to_csv(T / "leak_nn_by_split.csv", index=False)
    summ["test_to_nearest_train"] = {k: {**{f"p{q}": round(float(np.percentile(g.d, q)), 2) for q in (5, 25, 50, 75, 95)},
                                         **{f"frac_le_{t}": round(float((g.d <= t).mean()), 4) for t in (10, 17.5, 27.5, 35)},
                                         "n": int(len(g))} for k, g in NN.groupby("split_def")}
    rev = np.load(REPO / "outputs" / "review_out" / "exp3b_nn_test_to_train_dist.npy")
    pub = NN[NN.split_def == "published"].d.values
    summ["check_vs_review_nn"] = {"review_median": round(float(np.median(rev)), 3), "ours_median": round(float(np.median(pub)), 3),
                                  "max_abs_diff_sorted": round(float(np.abs(np.sort(rev) - np.sort(pub)).max()), 4)}

    # ---- figure 4: nn pairs
    te = np.flatnonzero(df.split.values == "test")
    tr = np.flatnonzero(df.split.values == "train")
    sel = rng.choice(te, 8, replace=False)
    fig, axs = plt.subplots(4, 4, figsize=(13, 10.2), gridspec_kw={"hspace": 0.42, "wspace": 0.06})
    for k, a in enumerate(sel):
        b = tr[D[a, tr].argmin()]
        ra, rb = df.iloc[a], df.iloc[b]
        shared = (ra.rel, rb.rel) in pk
        ax1, ax2 = axs[k // 2, (k % 2) * 2], axs[k // 2, (k % 2) * 2 + 1]
        tile(ax1, first[a], f"TEST {ra.cls} {ra.vid} · L{ra.label} · c{ra.cluster}", SPLIT_C["test"])
        tile(ax2, first[b], f"nearest TRAIN {rb.cls} {rb.vid} · L{rb.label} · c{rb.cluster}\n"
                            f"d = {D[a, b]:.1f} · |ID diff| {abs(int(ra.vid) - int(rb.vid)) if ra.cls == rb.cls else 'n/a'}"
                            f"{' · shares packets' if shared else ''}", SPLIT_C["train"])
    fig.suptitle("8 random test clips (seed 0) and their nearest train clip (32x32 min-over-frames RMSE), "
                 "shown as 128x72 first frames", x=0.01, ha="left", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01)
    fig.savefig(FIG / "leak_nn_pairs_hires.png")
    plt.close(fig)

    # ---- figure 5: 20 largest clusters, medoid
    sizes = df.cluster.value_counts()
    fig, axs = plt.subplots(4, 5, figsize=(16, 11.2), gridspec_kw={"hspace": 0.5, "wspace": 0.05})
    meds = []
    for ax, (cid, n) in zip(axs.ravel(), sizes.head(20).items()):
        idx = np.flatnonzero(df.cluster.values == cid)
        m = idx[D[np.ix_(idx, idx)].sum(1).argmin()]
        g = df.iloc[idx]
        meds.append(dict(cluster=int(cid), size=int(n), medoid=df.path.iloc[m], labels=mix(g.label, [0, 1, 2]),
                         splits=mix(g.split, ["train", "val", "test"]), id_ranges=f"{g.cls.mode().iloc[0]}"))
        tile(ax, first[m], f"cluster {cid} · n={n}\nlabels low/mid/high {mix(g.label, [0, 1, 2])}\n"
                           f"train/val/test {mix(g.split, ['train', 'val', 'test'])}")
    fig.suptitle("The 20 largest scene clusters at threshold 27.5 (medoid clip by thumbnail distance, first frame)",
                 x=0.01, ha="left", fontsize=12)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01)
    fig.savefig(FIG / "leak_clusters_overview.png")
    plt.close(fig)
    summ["top20_clusters"] = meds

    # ---- figure 7: distance histograms
    labels = df.label.values
    pairs = {}
    for name, same in (("random same-class pairs", True), ("random cross-class pairs", False)):
        a = rng.integers(0, len(df), 60000); b = rng.integers(0, len(df), 60000)
        m = (a != b) & ((labels[a] == labels[b]) == same)
        pairs[name] = D[a[m][:20000], b[m][:20000]]
    key = {(cl, v): i for i, (cl, v) in enumerate(zip(df.cls, df.vid))}
    adj = np.array([(i, key[(cl, v + 1)]) for i, (cl, v) in enumerate(zip(df.cls, df.vid)) if (cl, v + 1) in key])
    pairs["adjacent IDs (N, N+1)"] = D[adj[:, 0], adj[:, 1]]
    ff = np.sqrt(((T3[adj[:, 0], 0] - T3[adj[:, 1], 0]) ** 2).mean(1))
    pairs["adjacent IDs, first-first frame (f1, 32x32)"] = ff
    summ["reference_pairs"] = {k: {f"p{q}": round(float(np.percentile(v, q)), 2) for q in (1, 5, 25, 50, 75)} for k, v in pairs.items()}
    bins = np.logspace(np.log10(0.3), np.log10(140), 60)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7.8), sharex=True)
    ZERO = 0.05   # below this the thumbnails of some frame pair are identical (float32 noise <= 0.07)

    def h(a, v, col, lab, ls="-"):
        v = np.asarray(v)
        z = (v < ZERO).mean()
        a.hist(v[v >= ZERO], bins=bins, weights=np.full((v >= ZERO).sum(), 1.0 / len(v)), histtype="step", lw=1.8,
               color=col, ls=ls, label=f"{lab}: median {np.median(v):.1f}; identical frame (d=0): {z:.0%}")
    for (k, g), col in zip(NN.groupby("split_def", sort=False), [PAL[7], PAL[0], PAL[2], PAL[6]]):
        h(ax[0], g.d, col, f"{k} (n={len(g)})")
    ax[0].set_title("test clip -> nearest train clip, published split vs scene-grouped splits", loc="left")
    for (k, v), col in zip(pairs.items(), [PAL[3], INK2, PAL[4], PAL[1]]):
        h(ax[1], v, col, k, "--" if "first-first" in k else "-")
    ax[1].set_title("reference pair populations", loc="left")
    for a in ax:
        a.set_xscale("log")
        for t in (10, 17.5, 27.5, 35):
            a.axvline(t, color=INK2, lw=0.9, ls=":")
            a.text(t * 1.02, a.get_ylim()[1] * 0.97, f"{t:g}", fontsize=8, color=INK2, va="top")
        a.set_ylabel("fraction of clips / pairs per bin")
        a.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax[1].set_xlabel("clip distance: min over 3x3 frame pairs of RMSE between 32x32 thumbnails (0-255, log scale)")
    fig.tight_layout()
    fig.savefig(FIG / "leak_distance_hist.png")
    plt.close(fig)

    # ---- figure 6: t-SNE
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X = thumbs[:, 1].reshape(len(df), -1).astype(np.float32) / 255.0
    Z = PCA(50, random_state=0).fit_transform(X)
    t1 = time.time()
    Y = TSNE(n_components=2, perplexity=30, random_state=0, init="pca", learning_rate="auto", n_jobs=8).fit_transform(Z)
    summ["tsne_s"] = round(time.time() - t1, 1)
    np.save(T / "tsne_xy.npy", Y)
    fig, axs = plt.subplots(1, 3, figsize=(18, 6.4))
    big = sizes.head(12).index.tolist()
    a = axs[0]
    rest = ~df.cluster.isin(big).values
    a.scatter(Y[rest, 0], Y[rest, 1], s=3, color="#cfcdc6", lw=0, label="other clusters")
    hs = []
    for i, cid in enumerate(big):
        m = df.cluster.values == cid
        mk = "o" if i < 8 else "^"
        col = PAL[i % 8]
        a.scatter(Y[m, 0], Y[m, 1], s=6 if mk == "o" else 9, color=col, marker=mk, lw=0)
        cx, cy = np.median(Y[m, 0]), np.median(Y[m, 1])
        a.text(cx, cy, str(cid), fontsize=8, fontweight="bold", color=INK,
               bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=1, alpha=0.85))
        hs.append(Line2D([], [], marker=mk, ls="", color=col, label=f"c{cid} (n={int(m.sum())})", ms=6))
    a.legend(handles=hs, fontsize=7, frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.02))
    a.set_title("scene cluster (thr 27.5), 12 largest", loc="left")
    for a, col, cmap, names, title in ((axs[1], "label", LAB_C, LAB_N, "label"),
                                       (axs[2], "split", SPLIT_C, {k: k for k in SPLIT_C}, "published split")):
        order = [0, 1, 2] if col == "label" else ["train", "val", "test"]
        for k in order:
            m = df[col].values == k
            a.scatter(Y[m, 0], Y[m, 1], s=3 if k != "test" else 5, color=cmap[k], lw=0, label=f"{names[k]} (n={int(m.sum())})")
        a.legend(markerscale=4, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.02), ncol=3)
        a.set_title(title, loc="left")
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("t-SNE (perplexity 30, random_state 0) of PCA-50 of the 32x32 centre thumbnails, all 7,701 clips",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "leak_tsne.png")
    plt.close(fig)
    summ["wall_s"] = round(time.time() - t0, 1)
    with open(T / "leak_figures_summary.json", "w") as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != "top20_clusters"}, indent=1))


if __name__ == "__main__":
    main()
