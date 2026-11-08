"""Supervisor check: how coarse does the grouping become when the reconstructed events are united with the
October scene clusters (single linkage at 27.5) and, optionally, with the Tier-2 session candidates?

Rationale: for leak-free evaluation an over-link is harmless (it only merges groups), a missed link is a leak.
The union of all evidence is therefore the conservative grouping. This script reports, for three groupings,
the number of groups, the size distribution, the class composition, and the cross-fold nearest-neighbour
statistics of a StratifiedGroupKFold(5), reusing thumb_distance_matrix() of 05e and nn_stats() of the review.

Groupings:
  G1 event                       = event_id                                  (05e event_disjoint_k5)
  G2 event ∪ scene27.5           = components over clips sharing an event or a scene cluster at 27.5
  G3 event ∪ scene27.5 ∪ tier2   = G2 plus the Tier-2 session candidate links

Outputs: outputs/research/sessions/union_groups.csv (per clip), union_groups_summary.json, union_groups_folds.csv
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "sessions"
sys.path.insert(0, str(REPO / "scripts" / "review"))
sys.path.insert(0, str(REPO / "scripts" / "research"))
sys.dont_write_bytecode = True
g07 = importlib.import_module("07_make_grouped_splits")
e05 = importlib.import_module("05e_table_folds")


def components(n, keys_list):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for keys in keys_list:
        first = {}
        for i, k in enumerate(keys):
            if pd.isna(k):
                continue
            if k in first:
                a, b = find(first[k]), find(i)
                if a != b:
                    parent[a] = b
            else:
                first[k] = i
    roots = [find(i) for i in range(n)]
    uniq = {r: j for j, r in enumerate(sorted(set(roots)))}
    return np.array([uniq[r] for r in roots])


def main():
    t0 = time.time()
    cs = pd.read_csv(OUT / "clip_sessions.csv")
    cs = cs[cs.recording_id.notna() & ~cs.unreadable.astype(bool)].reset_index(drop=True)
    n = len(cs)
    ev_key = cs.event_id.map(lambda v: f"ev{int(v)}" if pd.notna(v) else np.nan)
    sc_key = cs.scene_cluster_27_5.map(lambda v: f"sc{int(v)}" if pd.notna(v) else np.nan)
    se_key = cs.session_candidate_id.map(lambda v: f"se{int(v)}" if pd.notna(v) else np.nan)
    groupings = {
        "G1_event": components(n, [ev_key]),
        "G2_event_scene27": components(n, [ev_key, sc_key]),
        "G3_event_scene27_tier2": components(n, [ev_key, sc_key, se_key]),
    }
    y = cs["class"].map({"low": 0, "mid": 1, "high": 2}).to_numpy()
    print("distance matrix ...", flush=True)
    D = e05.thumb_distance_matrix(cs.path.tolist())
    print(f"D {D.shape} in {time.time() - t0:.0f} s", flush=True)
    summary, fold_rows = {}, []
    for name, g in groupings.items():
        sizes = pd.Series(g).value_counts()
        comp = cs.assign(g=g).groupby("g")["class"].agg(lambda s: ",".join(sorted(set(s))))
        s = dict(n_groups=int(len(sizes)), largest=int(sizes.iloc[0]), largest_share=float(sizes.iloc[0] / n),
                 groups_ge_100=int((sizes >= 100).sum()), groups_single_class=int((comp.str.count(",") == 0).sum()),
                 groups_all_three=int((comp.str.count(",") == 2).sum()),
                 clips_in_multi_class_groups=int(sizes[comp.str.count(",") > 0].sum()))
        sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        f = np.empty(n, int)
        for k, (_, te) in enumerate(sgk.split(np.zeros(n), y, g)):
            f[te] = k
        nn_all = np.empty(n, np.float32)
        per_fold = {}
        for k in range(5):
            q, ref = np.flatnonzero(f == k), np.flatnonzero(f != k)
            st = g07.nn_stats(D, q, ref)
            nn_all[q] = D[np.ix_(q, ref)].min(axis=1)
            cc = cs.iloc[q]["class"].value_counts().to_dict()
            per_fold[k] = dict(n=int(len(q)), classes=cc, nn=st)
            fold_rows.append(dict(grouping=name, fold=k, n=len(q), low=cc.get("low", 0), mid=cc.get("mid", 0),
                                  high=cc.get("high", 0), nn_median=st["median"], nn_p5=st["p5"], nn_min=st["min"],
                                  frac_le_10=st["frac_le_10"], frac_le_17_5=st["frac_le_17_5"], frac_le_27_5=st["frac_le_27_5"]))
        s["cross_fold_nn"] = dict(median=float(np.median(nn_all)), p5=float(np.percentile(nn_all, 5)), min=float(nn_all.min()),
                                  frac_le_10=float((nn_all <= 10).mean()), frac_le_17_5=float((nn_all <= 17.5).mean()),
                                  frac_le_27_5=float((nn_all <= 27.5).mean()))
        s["folds"] = per_fold
        s["size_quantiles"] = {q: int(np.percentile(sizes, q)) for q in (50, 75, 90, 100)}
        summary[name] = s
        cs[name] = g
        cs[name + "_fold"] = f
        print(name, {k: v for k, v in s.items() if k not in ("folds",)}, flush=True)
    # the two residual close pairs of 05e under G2
    for a, b in ((2079, 2081), (2135, 2340)):
        ia = cs.index[cs.path == f"videos/high/view{a}.mp4"][0]
        ib = cs.index[cs.path == f"videos/high/view{b}.mp4"][0]
        print(f"high {a}/{b}: d={D[ia, ib]:.2f}", {k: (int(cs.loc[ia, k]), int(cs.loc[ib, k])) for k in groupings})
    cols = ["path", "class", "label", "split_published", "scene_cluster_27_5", "recording_id", "event_id", "session_candidate_id"]
    cs[cols + list(groupings) + [k + "_fold" for k in groupings]].to_csv(OUT / "union_groups.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "union_groups_folds.csv", index=False)
    (OUT / "union_groups_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
