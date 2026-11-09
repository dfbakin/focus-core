#!/usr/bin/env python
"""Dataset roots for the GPU leak-gradient runs: protocols P1 (G1 events) and P2 (G2 events ∪ scene clusters 27.5).

Plan: docs/research/11_gpu_experiment_plan.md, sections 2 and 6.  For protocol P1 (group column G1_event, fold column
G1_event_fold) and P2 (G2_event_scene27, G2_event_scene27_fold) and every fold k in 0..4:

  test  = clips with fold == k
  pool  = all other clips
  val   = whole groups carved from the pool by a deterministic greedy rule:
          candidates are the pool's groups, minus groups smaller than 1% of the pool and minus every group that holds
          label-conflict clips (event_id in 9, 11, 18; 10_reconstruction_qc_and_protocol.md section 3), under P1 and
          P2 alike, the groups being derived from the data;
          class coverage: V must hold all three classes with at least 30 clips each whenever some admissible choice
          (candidates only, |V| <= 15% of the pool) achieves it.  Whether it does is decided exactly, by a covering
          knapsack over the candidates (smallest total size reaching 30 clips of every class);
          start from V = {} and repeatedly add the candidate g that minimises the L1 distance between the normalised
          class histogram of V ∪ {g} and that of the pool, subject to |V ∪ {g}| <= 15% of the pool and, while V lacks
          coverage that is achievable, only among the g after which coverage is still reachable within the cap
          (same knapsack over the remaining candidates); stop when |V| >= 8% of the pool and V has coverage (or
          coverage is not achievable), or when no candidate fits; ties go to the smaller group id
  train = pool minus val

G1 group ids are NOT event ids: 05h_union_groups.py relabels the connected components, so G1 id 9 is event 6.  The
conflict exclusion therefore takes the groups of the clips whose event_id is 9, 11 or 18, never the ids 9, 11, 18
(the event <-> G1 correspondence is checked against outputs/research/sessions/clip_sessions.csv: one G1 group per
event and one event per G1 group); the excluded groups per protocol, the validation class counts and whether coverage
was achievable and achieved are printed and stored in val_groups.json.

Checks (the script aborts on failure, before and after writing): every clip of union_groups.csv is in exactly one of
train/val/test per fold; no group id is in more than one split; the five test folds partition the clips; the manifests
read back from disk equal what was intended (paths, labels, splits).

Inputs : outputs/research/sessions/union_groups.csv, outputs/research/sessions/clip_sessions.csv,
         data/ouc-cge-320/videos (target of the videos symlink; only its existence is checked)
Outputs: data/p1_f{0..4}/, data/p2_f{0..4}/  (data_labels/{train,val,test}.csv in the project manifest format and a
         relative symlink videos -> ../ouc-cge-320/videos, via scripts/review/common.py::write_dataset_root; a stale
         data_labels/.cache of the dataset class is removed),
         outputs/research/protocols/roots.csv  (per root and split: n, low/mid/high, number of groups, group ids),
         outputs/research/protocols/val_groups.json  (per root: candidates, exclusions, greedy trace, chosen groups)

Usage (CPU, seconds):
  .venv/bin/python scripts/research/06_make_protocol_roots.py
  .venv/bin/python scripts/research/06_make_protocol_roots.py --out-base data --protocols p2
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True  # no __pycache__ under scripts/review
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import SPLITS, load_all_manifests, write_dataset_root  # noqa: E402

PROTOCOLS = {"p1": ("G1_event", "G1_event_fold"), "p2": ("G2_event_scene27", "G2_event_scene27_fold")}
CONFLICT_EVENTS = (9, 11, 18)
N_FOLDS = 5
MIN_GROUP_FRAC = 0.01   # candidates must hold at least 1% of the pool
VAL_MIN_FRAC = 0.08     # stop once |V| >= 8% of the pool
VAL_MAX_FRAC = 0.15     # never let |V| exceed 15% of the pool
MIN_CLASS = 30          # class coverage: at least 30 clips of every class in V when achievable
CLASSES = ["low", "mid", "high"]  # label 0, 1, 2


def die(msg: str) -> None:
    raise SystemExit(f"ABORT: {msg}")


def class_counts(labels) -> np.ndarray:
    return np.bincount(np.asarray(labels, dtype=int), minlength=3).astype(float)


def conflict_g1_ids(u: pd.DataFrame, clip_sessions: Path) -> tuple[list[int], dict]:
    """Map the label-conflict events to G1 group ids and verify the event <-> G1 correspondence."""
    cs = pd.read_csv(clip_sessions, usecols=["path", "event_id"])
    m = u[["path", "event_id", "G1_event"]].merge(cs, on="path", how="left", suffixes=("", "_cs"))
    if m["event_id_cs"].isna().any():
        die(f"{int(m.event_id_cs.isna().sum())} clips of union_groups.csv have no event_id in clip_sessions.csv")
    if not (m["event_id"] == m["event_id_cs"]).all():
        die("event_id differs between union_groups.csv and clip_sessions.csv")
    per_event = m.groupby("event_id")["G1_event"].nunique()
    per_g1 = m.groupby("G1_event")["event_id"].nunique()
    if (per_event != 1).any() or (per_g1 != 1).any():
        die("G1_event is not a one-to-one relabelling of event_id")
    ev2g1 = m.drop_duplicates("event_id").set_index("event_id")["G1_event"].astype(int)
    n_equal = int((m["event_id"].astype(int) == m["G1_event"]).sum())
    ids = []
    for e in CONFLICT_EVENTS:
        if e not in ev2g1.index:
            die(f"conflict event {e} not found in clip_sessions.csv")
        ids.append(int(ev2g1.loc[e]))
    n_conf = int(m["event_id"].isin(CONFLICT_EVENTS).sum())
    info = dict(g1_id_equals_event_id_for_all_clips=bool(n_equal == len(m)), clips_with_g1_id_equal_event_id=n_equal,
                n_clips=int(len(m)), conflict_events=list(CONFLICT_EVENTS),
                conflict_event_to_g1={str(e): g for e, g in zip(CONFLICT_EVENTS, ids)}, conflict_clips=n_conf)
    print(f"G1 id == event id for {n_equal} of {len(m)} clips "
          f"({'identical' if n_equal == len(m) else 'G1 ids are a relabelling of the events'}); "
          f"event_id identical in union_groups.csv and clip_sessions.csv; one G1 group per event and vice versa.")
    print(f"label-conflict events {list(CONFLICT_EVENTS)} -> G1 groups {ids} ({n_conf} clips)")
    return ids, info


def min_cover_size(groups: list, gcounts: dict, need) -> float:
    """Exact covering knapsack: the smallest total size of a subset of `groups` holding at least need[c] clips of
    every class c (need <= MIN_CLASS); inf when no subset does.  f[a, b, c] = smallest size reaching (a, b, c)."""
    need = np.clip(np.asarray(need, dtype=int), 0, MIN_CLASS)
    f = np.full((MIN_CLASS + 1,) * 3, np.inf)
    f[0, 0, 0] = 0.0
    r = np.arange(MIN_CLASS + 1)
    for g in groups:
        x, y, z = (int(v) for v in gcounts[g])
        prev = f[np.ix_(np.maximum(r - x, 0), np.maximum(r - y, 0), np.maximum(r - z, 0))]
        f = np.minimum(f, prev + gcounts[g].sum())
    return float(f[need[0], need[1], need[2]])


def choose_val(pool: pd.DataFrame, gcol: str, exclude: set) -> tuple[list[int], dict]:
    """Greedy whole-group validation set with class coverage (rule in the module docstring)."""
    n_pool = len(pool)
    cap = VAL_MAX_FRAC * n_pool
    target = class_counts(pool["label"])
    target /= target.sum()
    sizes = pool.groupby(gcol).size()
    gcounts = {int(g): class_counts(d["label"]) for g, d in pool.groupby(gcol)}
    small = sorted(int(g) for g in sizes.index if sizes[g] < MIN_GROUP_FRAC * n_pool)
    excluded = sorted(int(g) for g in sizes.index if int(g) in exclude)
    cand = sorted(int(g) for g in sizes.index if int(g) not in small and int(g) not in exclude)
    achievable = min_cover_size(cand, gcounts, [MIN_CLASS] * 3) <= cap
    chosen, counts, trace = [], np.zeros(3), []
    while True:
        covered = bool((counts >= MIN_CLASS).all())
        if counts.sum() >= VAL_MIN_FRAC * n_pool and (covered or not achievable):
            break
        best = None
        for g in cand:  # ascending ids: a strict "<" keeps the smaller id on ties
            if g in chosen:
                continue
            c = counts + gcounts[g]
            if c.sum() > cap:
                continue
            if achievable and not covered:  # keep coverage reachable within the cap
                need = MIN_CLASS - c
                rest = [h for h in cand if h not in chosen and h != g]
                if (need > 0).any() and c.sum() + min_cover_size(rest, gcounts, need) > cap:
                    continue
            d = round(float(np.abs(c / c.sum() - target).sum()), 12)
            if best is None or d < best[0]:
                best = (d, g)
        if best is None:
            break
        chosen.append(best[1])
        counts += gcounts[best[1]]
        trace.append(dict(add=best[1], size=int(sizes[best[1]]), val_size=int(counts.sum()), l1=best[0]))
    info = dict(pool_n=n_pool, pool_counts=class_counts(pool["label"]).astype(int).tolist(),
                min_group_size=MIN_GROUP_FRAC * n_pool, val_min=VAL_MIN_FRAC * n_pool, val_max=VAL_MAX_FRAC * n_pool,
                n_pool_groups=int(len(sizes)), candidates=cand, excluded_small=small, excluded_conflict=excluded,
                trace=trace, val_groups=chosen, val_n=int(counts.sum()), val_counts=counts.astype(int).tolist(),
                coverage_min_per_class=MIN_CLASS, coverage_achievable=bool(achievable),
                coverage_achieved=bool((counts >= MIN_CLASS).all()),
                val_frac=float(counts.sum() / n_pool), reached_min=bool(counts.sum() >= VAL_MIN_FRAC * n_pool),
                l1_final=trace[-1]["l1"] if trace else None)
    return chosen, info


def check_assignment(u: pd.DataFrame, split: pd.Series, gcol: str, name: str) -> None:
    """Every clip in exactly one split, no group crossing splits."""
    if len(split) != len(u) or split.isna().any() or not set(split.unique()) <= set(SPLITS):
        die(f"{name}: not every clip has exactly one of train/val/test")
    crossing = pd.DataFrame({"g": u[gcol].values, "s": split.values}).groupby("g")["s"].nunique()
    if (crossing > 1).any():
        die(f"{name}: groups {sorted(crossing[crossing > 1].index.tolist())} appear in more than one split")
    for s in SPLITS:
        if (split == s).sum() == 0:
            die(f"{name}: empty {s} split")


def check_on_disk(root: Path, intended: pd.DataFrame, gcol: str, u: pd.DataFrame, name: str) -> None:
    disk = load_all_manifests(root)
    if disk["path"].duplicated().any():
        die(f"{name}: duplicated paths in the written manifests")
    a = disk[["path", "label", "split"]].sort_values("path").reset_index(drop=True)
    b = intended[["path", "label", "split"]].sort_values("path").reset_index(drop=True)
    if len(a) != len(b) or not (a.values == b.values).all():
        die(f"{name}: manifests on disk differ from the intended assignment")
    j = disk.merge(u[["path", gcol]], on="path", how="left")
    if j[gcol].isna().any():
        die(f"{name}: manifests hold clips that are not in union_groups.csv")
    if (j.groupby(gcol)["split"].nunique() > 1).any():
        die(f"{name}: a group crosses splits in the written manifests")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--groups", type=Path, default=REPO / "outputs/research/sessions/union_groups.csv")
    ap.add_argument("--clip-sessions", type=Path, default=REPO / "outputs/research/sessions/clip_sessions.csv")
    ap.add_argument("--out-base", type=Path, default=REPO / "data")
    ap.add_argument("--videos-link", default="../ouc-cge-320/videos")
    ap.add_argument("--tables", type=Path, default=REPO / "outputs/research/protocols")
    ap.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=list(PROTOCOLS))
    args = ap.parse_args()

    u = pd.read_csv(args.groups)
    if u["path"].duplicated().any():
        die("duplicated paths in union_groups.csv")
    if not set(u["label"].unique()) <= {0, 1, 2}:
        die("labels outside 0..2 in union_groups.csv")
    print(f"union_groups.csv: {len(u)} clips, class counts {class_counts(u.label).astype(int).tolist()} (low, mid, high)")
    conflict_ids, conflict_info = conflict_g1_ids(u, args.clip_sessions)
    is_conflict = u["event_id"].isin(CONFLICT_EVENTS)
    if not Path(os.path.normpath(args.out_base / "root" / args.videos_link)).exists():  # relative to <out-base>/<root>
        print(f"WARNING: videos target {args.videos_link} (relative to the roots) does not exist here; "
              f"the symlinks are written anyway")

    rule = dict(min_group_frac=MIN_GROUP_FRAC, val_min_frac=VAL_MIN_FRAC, val_max_frac=VAL_MAX_FRAC,
                objective="L1 between normalised class histograms of V ∪ {g} and of the pool",
                ties="smaller group id", min_clips_per_class_in_val=MIN_CLASS,
                excluded="groups holding clips of events 9, 11, 18 (both protocols)", p1_conflict_groups=conflict_ids)
    rows, val_json = [], {"conflict_mapping": conflict_info, "rule": rule, "roots": {}}
    for proto in args.protocols:
        gcol, fcol = PROTOCOLS[proto]
        folds = sorted(u[fcol].unique().tolist())
        if folds != list(range(N_FOLDS)):
            die(f"{fcol}: folds {folds}, expected 0..{N_FOLDS - 1}")
        crossing = u.groupby(gcol)[fcol].nunique()
        if (crossing > 1).any():
            die(f"{gcol}: groups split over several folds in {fcol}")
        tested = pd.Series(0, index=u.index)
        exclude = set(int(g) for g in u.loc[is_conflict, gcol].unique())  # groups holding label-conflict clips
        if proto == "p1" and exclude != set(conflict_ids):
            die(f"P1 conflict groups {sorted(exclude)} differ from the event mapping {conflict_ids}")
        val_json["rule"][f"{proto}_excluded_conflict_groups"] = sorted(exclude)
        print(f"{proto}: groups holding label-conflict clips, excluded from validation: {sorted(exclude)}")
        for k in range(N_FOLDS):
            name = f"{proto}_f{k}"
            test = u[fcol] == k
            pool = u[~test]
            val_ids, info = choose_val(pool, gcol, exclude)
            if not val_ids:
                die(f"{name}: no validation group could be chosen")
            split = pd.Series("train", index=u.index)
            split[test] = "test"
            split[~test & u[gcol].isin(val_ids)] = "val"
            check_assignment(u, split, gcol, name)
            tested += test.astype(int)
            df = pd.DataFrame({"path": u["path"], "label": u["label"], "split": split})
            root = args.out_base / name
            cache = root / "data_labels" / ".cache"
            if cache.exists():  # filtered manifests of OUCCGEDataset from an earlier assignment
                shutil.rmtree(cache)
            write_dataset_root(df, root, videos_link=args.videos_link)
            check_on_disk(root, df, gcol, u, name)
            info["val_groups_conflict_clips"] = int((is_conflict & (split == "val")).sum())
            val_json["roots"][name] = dict(protocol=proto, group_col=gcol, fold_col=fcol, fold=k, **info)
            for s in SPLITS:
                m = split == s
                c = class_counts(u.loc[m, "label"]).astype(int)
                gids = sorted(int(g) for g in u.loc[m, gcol].unique())
                rows.append(dict(root=name, protocol=proto, group_col=gcol, fold=k, split=s, n=int(m.sum()),
                                 low=c[0], mid=c[1], high=c[2], n_groups=len(gids),
                                 conflict_clips=int((is_conflict & m).sum()), group_ids=" ".join(map(str, gids))))
            tr = split == "train"
            print(f"{name}: train {int(tr.sum())} {class_counts(u.loc[tr, 'label']).astype(int).tolist()}  "
                  f"val {info['val_n']} {info['val_counts']} groups {val_ids} ({100 * info['val_frac']:.1f}% of pool, "
                  f"L1 {info['l1_final']:.3f}; coverage >= {MIN_CLASS}/class "
                  f"{'achieved' if info['coverage_achieved'] else 'NOT achieved'}"
                  f"{'' if info['coverage_achievable'] else ' (not achievable)'})  "
                  f"test {int(test.sum())} {class_counts(u.loc[test, 'label']).astype(int).tolist()}")
        if not (tested == 1).all():
            die(f"{proto}: the five test folds do not partition the clips")
        print(f"{proto}: checks passed (each clip in exactly one split per fold, no group crosses splits, "
              f"test folds partition all {len(u)} clips)")

    args.tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.tables / "roots.csv", index=False)
    with open(args.tables / "val_groups.json", "w") as fh:
        json.dump(val_json, fh, indent=1)
    print(f"wrote {args.tables / 'roots.csv'} and {args.tables / 'val_groups.json'}")


if __name__ == "__main__":
    main()
