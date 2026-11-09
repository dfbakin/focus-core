#!/usr/bin/env python
"""Aggregate the leak-gradient runs: pooled and per-fold metrics, label-conflict exclusion, per-G2-group confusion,
seed spread, group-level bootstrap, per-epoch curves, the October reference points and one figure.

Reads every per-clip prediction file <root>/eval/preds_<tag>.csv (path,label,pred,p0,p1,p2) written by
scripts/review/05_eval_checkpoints.py through 06b_run_protocols.sh, or by 06e_probe.py.  Tag convention (06b):
  <tag> = <run>_<ck>_<split>,  <ck> in {best, ep<NN>},  <split> in {test, val}
  <run> = <protocol>[_f<k>]_<recipe>_s<seed>,  protocol in {p0, p1, p2, grp27a, grp27b, grp17},
          recipe in {r50, slow, probe}
Checkpoint rules: "val-selected" = <run>_best_test (best validation accuracy, earliest epoch on ties);
"final" = the highest <run>_ep<NN>_test (when <root>/state/ckpts_<run>.json exists, its final_epoch must be the one
found, else the rule is reported missing for that run); the probe is a single fit, so its "final" is its "best".
Each clip is joined to outputs/research/sessions/union_groups.csv (labels must agree; G1/G2 groups and folds) and to
clip_sessions.csv (event_id).  Label-conflict clips = event_id in {9, 11, 18} (880 clips; under P1 they are the whole
G1 groups 34, 40, 44): every metric is given for all clips and with them excluded ("excl").

Metrics (from 3x3 confusion matrices, cross-checked against sklearn on every pooled set): accuracy, balanced accuracy
(mean recall over the classes present in the truth), macro-F1 (over classes present in truth or prediction), Cohen's
kappa.  Majority baseline = share of the most frequent class in that test set.  Bootstrap: the test groups (16 G2
groups for P2, 45 G1 groups for P1, 42 without the conflict groups) resampled with replacement, 2,000 reps,
numpy default_rng(0) per row, metrics recomputed on the pooled resampled clip set, percentile 95% interval.

Outputs (--results-dir, default outputs/research/protocols/results/): pooled.csv, per_fold.csv, per_g2_group.csv,
seed_spread.csv, bootstrap.csv, epoch_curves.csv, references.csv, leak_points.csv, inventory.csv;
--doc (default docs/research/12_leak_gradient_results.md); --fig (default docs/research/img/leak_gradient.png, a plain
chart: balanced accuracy, val-selected, conflict clips excluded, against the cross-fold nearest-neighbour median of
each protocol, taken from outputs/research/temporal/leak_nn_by_split.csv (published, grp17, grp27b, grp27a; test to
nearest train) and outputs/research/sessions/union_groups_summary.json (G1, G2; cross-fold)).

Usage (CPU, seconds):
  .venv/bin/python scripts/research/06c_aggregate.py
  .venv/bin/python scripts/research/06c_aggregate.py --root <dir with eval/> --results-dir <dir> --doc <md> --fig <png>
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CONFLICT_EVENTS = (9, 11, 18)
RUN_RE = re.compile(r"^(?P<protocol>p0|p1|p2|grp27a|grp27b|grp17)(?:_f(?P<fold>\d+))?_(?P<recipe>r50|slow|probe)_s(?P<seed>\d+)$")
TAG_RE = re.compile(r"^(?P<run>.+)_(?P<ck>best|ep\d+)_(?P<split>test|val)$")
PROTOCOLS = ["p0", "grp17", "p1", "grp27b", "p2", "grp27a"]  # by cross-fold NN distance
PROTO_NAME = {"p0": "P0 published", "p1": "P1 G1 events", "p2": "P2 G2 events+scenes", "grp17": "grp17",
              "grp27b": "grp27b", "grp27a": "grp27a"}
NN_KEY = {"p0": "published", "grp17": "grp17", "grp27b": "grp27b", "grp27a": "grp27a", "p1": "G1_event",
          "p2": "G2_event_scene27"}
GROUP_COL = {"p1": "G1_event", "p2": "G2_event_scene27"}
FOLD_COL = {"p1": "G1_event_fold", "p2": "G2_event_scene27_fold"}
N_FOLDS = {"p1": 5, "p2": 5}
RECIPES = ["r50", "slow", "probe"]
RULES = ["val-selected", "final"]
METRICS = ["acc", "bal_acc", "macro_f1", "kappa"]
N_BOOT, BOOT_SEED = 2000, 0
OCTOBER = {  # tag prefix in outputs/review_grouped/eval -> (protocol, recipe)
    "R1_r50_random": ("p0", "r50"), "R2_r50_grp27a": ("grp27a", "r50"), "R3_r50_grp27b": ("grp27b", "r50"),
    "R4_r50_grp17": ("grp17", "r50"), "R5_slow_grp27a": ("grp27a", "slow")}
PLANNED = ([f"p0_probe_s0"] + [f"p{p}_f{k}_probe_s0" for p in (1, 2) for k in range(5)]
           + [f"{g}_probe_s0" for g in ("grp27a", "grp27b", "grp17")]
           + [f"p1_f{k}_r50_s42" for k in range(5)]
           + [f"p2_f{k}_r50_s{s}" for s in (42, 43, 44) for k in range(5)]
           + [f"p2_f{k}_slow_s42" for k in range(5)])


# ------------------------------------------------------------------ metrics
def confusion(y, p) -> np.ndarray:
    return np.bincount(np.asarray(y, int) * 3 + np.asarray(p, int), minlength=9).reshape(3, 3)


def cm_metrics(C: np.ndarray) -> dict:
    """Metrics of one or many confusion matrices (..., 3, 3); sklearn conventions (see module docstring)."""
    C = np.asarray(C, dtype=float)
    N = C.sum((-2, -1))
    tp = np.diagonal(C, axis1=-2, axis2=-1)
    true, pred = C.sum(-1), C.sum(-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        acc = tp.sum(-1) / N
        present = true > 0
        rec = np.where(present, tp / np.where(present, true, 1), 0.0)
        bal = (rec * present).sum(-1) / present.sum(-1)
        prec = np.where(pred > 0, tp / np.where(pred > 0, pred, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
        lab = (true + pred) > 0
        mf1 = (f1 * lab).sum(-1) / lab.sum(-1)
        pe = (true * pred).sum(-1) / N ** 2
        kappa = np.where(pe < 1, (acc - pe) / np.where(pe < 1, 1 - pe, 1), np.nan)
    return dict(acc=acc, bal_acc=bal, macro_f1=mf1, kappa=kappa)


def sk_check(y, p, m: dict, what: str) -> None:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
    if len(y) == 0:
        return
    ref = dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro"),
               kappa=cohen_kappa_score(y, p) if len(set(y) | set(p)) > 1 else np.nan)
    for k, v in ref.items():
        if not (np.isnan(v) and np.isnan(m[k])) and abs(float(m[k]) - v) > 1e-9:
            raise SystemExit(f"ABORT: metric {k} of {what} differs from sklearn: {m[k]} vs {v}")


def metric_row(df: pd.DataFrame, what: str = "") -> dict:
    """n, metrics, majority baseline, for all clips and without the label-conflict clips."""
    out = {}
    for suffix, d in (("", df), ("_excl", df[~df["conflict"]])):
        y, p = d["label"].to_numpy(int), d["pred"].to_numpy(int)
        out["n" + suffix] = len(d)
        m = cm_metrics(confusion(y, p)) if len(d) else {k: np.nan for k in METRICS}
        if what and len(d):
            sk_check(y, p, m, what + suffix)
        for k in METRICS:
            out[k + suffix] = float(m[k])
        out["majority" + suffix] = np.bincount(y, minlength=3).max() / len(y) if len(y) else np.nan
    return out


def bootstrap(df: pd.DataFrame, gcol: str) -> dict:
    groups = np.sort(df[gcol].unique())
    Cg = np.stack([confusion(d["label"], d["pred"]) for _, d in df.groupby(gcol, sort=True)]).reshape(len(groups), 9)
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    W = np.stack([np.bincount(r, minlength=len(groups)) for r in draws])
    m = cm_metrics((W @ Cg).reshape(N_BOOT, 3, 3))
    out = dict(n_groups=len(groups))
    for k in METRICS:
        out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(m[k], [2.5, 97.5])
    return out


# ------------------------------------------------------------------ inputs
def clip_table(groups: Path, sessions: Path) -> pd.DataFrame:
    u = pd.read_csv(groups)
    cs = pd.read_csv(sessions, usecols=["path", "event_id"]).rename(columns={"event_id": "event_cs"})
    u = u.merge(cs, on="path", how="left")
    if not (u["event_id"] == u["event_cs"]).all():
        raise SystemExit("ABORT: event_id differs between union_groups.csv and clip_sessions.csv")
    u["conflict"] = u["event_id"].isin(CONFLICT_EVENTS)
    return u.drop(columns="event_cs").set_index("path")


def norm_path(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"^.*?(videos/)", r"\1", regex=True)


def read_preds(f: Path, clips: pd.DataFrame, what: str) -> pd.DataFrame:
    df = pd.read_csv(f)
    df["path"] = norm_path(df["path"])
    if df["path"].duplicated().any():
        raise SystemExit(f"ABORT: duplicated clips in {f}")
    miss = ~df["path"].isin(clips.index)
    if miss.any():
        raise SystemExit(f"ABORT: {int(miss.sum())} clips of {f} are not in union_groups.csv")
    j = df.join(clips[["label", "class", "G1_event", "G2_event_scene27", "G1_event_fold", "G2_event_scene27_fold",
                       "event_id", "conflict", "split_published"]], on="path", rsuffix="_u")
    if not (j["label"] == j["label_u"]).all():
        raise SystemExit(f"ABORT: labels of {what} disagree with union_groups.csv")
    return j.drop(columns="label_u")


def ckpt_epoch(summary: Path):
    try:
        m = re.search(r"epoch=(\d+)", json.load(open(summary)).get("ckpt", ""))
        return int(m.group(1)) if m else None
    except Exception:
        return None


def inventory(root: Path, clips: pd.DataFrame):
    """All prediction files under root/eval, parsed; unknown tags are listed, not read."""
    recs, unknown = [], []
    for f in sorted((root / "eval").glob("preds_*.csv")):
        tag = f.stem[len("preds_"):]
        m = TAG_RE.match(tag)
        r = RUN_RE.match(m["run"]) if m else None
        if not r:
            unknown.append(tag)
            continue
        recs.append(dict(tag=tag, run=m["run"], protocol=r["protocol"], fold=int(r["fold"]) if r["fold"] else 0,
                         recipe=r["recipe"], seed=int(r["seed"]), ck=m["ck"], split=m["split"], file=f,
                         best_epoch=ckpt_epoch(root / "eval" / f"summary_{tag}.json") if m["ck"] == "best" else None))
    inv = pd.DataFrame(recs)
    data = {t: read_preds(f, clips, t) for t, f in zip(inv.get("tag", []), inv.get("file", []))}
    final_ep = {}
    for f in (root / "state").glob("ckpts_*.json") if (root / "state").exists() else []:
        try:
            final_ep[f.stem[len("ckpts_"):]] = int(json.load(open(f))["final_epoch"])
        except Exception:
            pass
    return inv, data, unknown, final_ep


def expected_test(protocol: str, fold: int, clips: pd.DataFrame, splits_dir: Path) -> set:
    if protocol in FOLD_COL:
        return set(clips.index[clips[FOLD_COL[protocol]] == fold])
    if protocol == "p0":
        return set(clips.index[clips["split_published"] == "test"])
    s = pd.read_csv(splits_dir / f"split_{protocol}.csv")
    return set(s.loc[s.split == "test", "path"])


def rule_tags(inv: pd.DataFrame, final_ep: dict) -> pd.DataFrame:
    """One row per (run, rule) with the test tag that implements it (or NaN)."""
    rows = []
    test = inv[inv.split == "test"]
    for run, d in test.groupby("run"):
        first = d.iloc[0]
        best = d[d.ck == "best"]
        eps = d[d.ck.str.startswith("ep")].assign(ep=lambda x: x.ck.str[2:].astype(int))
        if len(eps):
            last = eps.sort_values("ep").iloc[-1]
            ok = run not in final_ep or final_ep[run] == last.ep
            final_tag, final_epoch = (last.tag, int(last.ep)) if ok else (np.nan, np.nan)
        elif first.recipe == "probe" and len(best):
            final_tag, final_epoch = best.iloc[0].tag, np.nan
        else:
            final_tag, final_epoch = np.nan, np.nan
        base = dict(run=run, protocol=first.protocol, fold=first.fold, recipe=first.recipe, seed=first.seed,
                    best_epoch=best.iloc[0].best_epoch if len(best) else None)
        rows.append(dict(base, rule="val-selected", tag=best.iloc[0].tag if len(best) else np.nan,
                         epoch=best.iloc[0].best_epoch if len(best) else np.nan))
        rows.append(dict(base, rule="final", tag=final_tag, epoch=final_epoch))
    return pd.DataFrame(rows)


def nn_medians(nn_csv: Path, summary_json: Path) -> dict:
    d = pd.read_csv(nn_csv).groupby("split_def")["d"].median().to_dict()
    s = json.load(open(summary_json))
    out = {}
    for p, key in NN_KEY.items():
        if key in d:
            out[p] = (float(d[key]), f"{nn_csv.name}: median d, {key}")
        elif key in s:
            out[p] = (float(s[key]["cross_fold_nn"]["median"]), f"{summary_json.name}: {key} cross_fold_nn.median")
    return out


# ------------------------------------------------------------------ formatting
def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def no_probe_final(df: pd.DataFrame) -> pd.DataFrame:
    """The probe's "final" rows repeat its "val-selected" rows (one fit); the doc shows them once, the CSVs keep both."""
    return df[~((df["recipe"] == "probe") & (df["rule"] == "final"))] if len(df) else df


def md_table(df: pd.DataFrame, cols: list, heads: list | None = None) -> str:
    heads = heads or cols
    lines = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f3(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO / "outputs/research_gpu/_summary")
    ap.add_argument("--groups", type=Path, default=REPO / "outputs/research/sessions/union_groups.csv")
    ap.add_argument("--sessions", type=Path, default=REPO / "outputs/research/sessions/clip_sessions.csv")
    ap.add_argument("--nn", type=Path, default=REPO / "outputs/research/temporal/leak_nn_by_split.csv")
    ap.add_argument("--groups-summary", type=Path, default=REPO / "outputs/research/sessions/union_groups_summary.json")
    ap.add_argument("--splits-dir", type=Path, default=REPO / "docs/review/data/grouped/splits")
    ap.add_argument("--october-eval", type=Path, default=REPO / "outputs/review_grouped/eval")
    ap.add_argument("--october-epochs", type=Path, default=REPO / "outputs/review_grouped_epochs/epoch_curve.csv")
    ap.add_argument("--results-dir", type=Path, default=REPO / "outputs/research/protocols/results")
    ap.add_argument("--doc", type=Path, default=REPO / "docs/research/12_leak_gradient_results.md")
    ap.add_argument("--fig", type=Path, default=REPO / "docs/research/img/leak_gradient.png")
    args = ap.parse_args()
    t0 = time.time()

    clips = clip_table(args.groups, args.sessions)
    n_conf = int(clips["conflict"].sum())
    inv, data, unknown, final_ep = inventory(args.root, clips)
    print(f"{len(inv)} prediction files under {args.root / 'eval'} ({len(unknown)} with unknown tags)")
    if unknown:
        print("  unknown tags (ignored): " + ", ".join(unknown[:20]))
    rt = rule_tags(inv, final_ep) if len(inv) else pd.DataFrame(
        columns=["run", "protocol", "fold", "recipe", "seed", "best_epoch", "rule", "tag", "epoch"])

    # ---- (2) per fold, with coverage against the expected test set
    fold_rows = []
    for _, r in rt.iterrows():
        if pd.isna(r.tag):
            continue
        d = data[r.tag]
        exp = expected_test(r.protocol, r.fold, clips, args.splits_dir)
        extra = set(d.path) - exp
        if extra:
            raise SystemExit(f"ABORT: {r.tag} holds {len(extra)} clips outside its expected test set")
        row = dict(protocol=r.protocol, recipe=r.recipe, seed=r.seed, rule=r.rule, fold=r.fold, epoch=r.epoch,
                   n_expected=len(exp), **metric_row(d, r.tag))
        vt = f"{r.run}_best_val"
        if r.rule == "val-selected" and vt in data:
            row["val_bal_acc"] = metric_row(data[vt])["bal_acc"]
        fold_rows.append(row)
    per_fold = pd.DataFrame(fold_rows)

    # ---- (1) pooled, (5) bootstrap
    pooled_rows, boot_rows, pooled_sets = [], [], {}
    if len(rt):
        for (proto, rec, seed, rule), g in rt.groupby(["protocol", "recipe", "seed", "rule"]):
            tags = g.tag.dropna().tolist()
            n_exp_folds = N_FOLDS.get(proto, 1)
            if not tags:
                continue
            d = pd.concat([data[t] for t in tags], ignore_index=True)
            if d["path"].duplicated().any():
                raise SystemExit(f"ABORT: a clip is tested twice in {proto} {rec} s{seed} {rule}")
            pooled_sets[(proto, rec, seed, rule)] = d
            n_exp = sum(len(expected_test(proto, f, clips, args.splits_dir)) for f in sorted(g.fold.unique()))
            pooled_rows.append(dict(protocol=proto, recipe=rec, seed=seed, rule=rule,
                                    folds=f"{len(tags)}/{n_exp_folds}", n_expected=n_exp,
                                    **metric_row(d, f"{proto} {rec} s{seed} {rule}")))
            if proto in GROUP_COL:
                for subset, dd in (("all", d), ("excl", d[~d["conflict"]])):
                    b = bootstrap(dd, GROUP_COL[proto])
                    m = metric_row(dd)
                    boot_rows.append(dict(protocol=proto, recipe=rec, seed=seed, rule=rule, subset=subset,
                                          folds=f"{len(tags)}/{n_exp_folds}", n=len(dd), **{k: m[k] for k in METRICS},
                                          **b))
    pooled = pd.DataFrame(pooled_rows)
    boot = pd.DataFrame(boot_rows)

    # ---- (3) per G2 group, P2 runs
    grp_rows = []
    for (proto, rec, seed, rule), d in pooled_sets.items():
        if proto != "p2":
            continue
        for gid, dd in d.groupby("G2_event_scene27"):
            c = np.bincount(dd["label"].to_numpy(int), minlength=3)
            C = confusion(dd["label"], dd["pred"])
            m = metric_row(dd)
            grp_rows.append(dict(recipe=rec, seed=seed, rule=rule, g2_group=int(gid), fold=int(dd["G2_event_scene27_fold"].iloc[0]),
                                 n=len(dd), low=c[0], mid=c[1], high=c[2], n_conflict=int(dd["conflict"].sum()),
                                 acc=m["acc"], bal_acc=m["bal_acc"], bal_acc_excl=m["bal_acc_excl"],
                                 confusion=" / ".join(",".join(map(str, row)) for row in C)))
    per_group = pd.DataFrame(grp_rows)

    # ---- (4) seed spread, r50 on P2
    seed_rows = []
    if len(pooled):
        s = pooled[(pooled.protocol == "p2") & (pooled.recipe == "r50")]
        for rule, g in s.groupby("rule"):
            for subset, suf in (("all", ""), ("excl", "_excl")):
                row = dict(rule=rule, subset=subset, seeds=" ".join(map(str, sorted(g.seed))), n_seeds=len(g),
                           folds=" ".join(g.sort_values("seed").folds))
                for k in METRICS:
                    v = g[k + suf]
                    row.update({f"{k}_mean": v.mean(), f"{k}_min": v.min(), f"{k}_max": v.max()})
                seed_rows.append(row)
    seeds = pd.DataFrame(seed_rows)

    # ---- (6) per-epoch curves
    ep_rows = []
    ep_inv = inv[(inv.split == "test") & inv.ck.str.startswith("ep")] if len(inv) else inv
    for _, r in ep_inv.iterrows():
        ep = int(r.ck[2:])
        best = rt[(rt.run == r.run) & (rt.rule == "val-selected")]["epoch"]
        fin = rt[(rt.run == r.run) & (rt.rule == "final")]["epoch"]
        ep_rows.append(dict(protocol=r.protocol, recipe=r.recipe, seed=r.seed, fold=str(r.fold), epoch=ep,
                            selected_by_val=int(len(best) and best.iloc[0] == ep), final_epoch=int(len(fin) and fin.iloc[0] == ep),
                            **metric_row(data[r.tag])))
    curves = pd.DataFrame(ep_rows)
    if len(curves):
        pooled_curves = []
        for (proto, rec, seed, ep), g in ep_inv.assign(ep=ep_inv.ck.str[2:].astype(int)).groupby(
                ["protocol", "recipe", "seed", "ep"]):
            if len(g) < N_FOLDS.get(proto, 1):
                continue  # pooled only where every fold reached this epoch
            d = pd.concat([data[t] for t in g.tag], ignore_index=True)
            pooled_curves.append(dict(protocol=proto, recipe=rec, seed=seed, fold="pooled", epoch=int(ep),
                                      selected_by_val=0, final_epoch=0, **metric_row(d)))
        curves = pd.concat([curves, pd.DataFrame(pooled_curves)], ignore_index=True)

    # ---- (7) October reference points
    ref_rows = []
    for prefix, (proto, rec) in OCTOBER.items():
        for suffix, rule in (("_test", "val-selected"), ("_last_test", "last.ckpt (later of the 2 best-by-val epochs)")):
            f = args.october_eval / f"preds_{prefix}{suffix}.csv"
            if not f.exists():
                continue
            d = read_preds(f, clips, f.name)
            exp = expected_test(proto, 0, clips, args.splits_dir)
            if set(d.path) - exp:
                raise SystemExit(f"ABORT: {f} holds clips outside the {proto} test set")
            ref_rows.append(dict(ref=prefix.split("_")[0], run=prefix, protocol=proto, recipe=rec, rule=rule,
                                 source=f"outputs/review_grouped/eval/{f.name}", n_expected=len(exp),
                                 **metric_row(d, f.name)))
    if args.october_epochs.exists():
        ec = pd.read_csv(args.october_epochs)
        for run, g in ec.groupby("run"):
            if run not in OCTOBER:
                continue
            proto, rec = OCTOBER[run]
            for flag, rule in (("selected_by_val", "val-selected (epoch-curve repeat run)"),
                               ("final_epoch", "final epoch (epoch-curve repeat run)")):
                r = g[g[flag] == 1].iloc[0]
                ref_rows.append(dict(ref=run.split("_")[0], run=run, protocol=proto, recipe=rec, rule=rule,
                                     source=f"{args.october_epochs.name} epoch {int(r.epoch)}", n=int(r.test_clips),
                                     acc=float(r.test_acc), macro_f1=float(r.test_f1_macro)))
    refs = pd.DataFrame(ref_rows)

    # ---- (8) leak points and figure
    nn = nn_medians(args.nn, args.groups_summary)
    pts = []
    for proto in PROTOCOLS:
        for rec in RECIPES:
            new = pooled[(pooled.protocol == proto) & (pooled.recipe == rec) & (pooled.rule == "val-selected")] \
                if len(pooled) else pooled
            if len(new):
                for _, r in new.iterrows():
                    b = boot[(boot.protocol == proto) & (boot.recipe == rec) & (boot.seed == r.seed) &
                             (boot.rule == "val-selected") & (boot.subset == "excl")] if len(boot) else boot
                    pts.append(dict(protocol=proto, recipe=rec, seed=r.seed, source="research_gpu", nn_median=nn[proto][0],
                                    nn_source=nn[proto][1], bal_acc_excl=r.bal_acc_excl, acc_excl=r.acc_excl,
                                    bal_lo=b.bal_acc_lo.iloc[0] if len(b) else np.nan,
                                    bal_hi=b.bal_acc_hi.iloc[0] if len(b) else np.nan, folds=r.folds))
            elif len(refs):
                o = refs[(refs.protocol == proto) & (refs.recipe == rec) & (refs.rule == "val-selected")]
                for _, r in o.iterrows():
                    pts.append(dict(protocol=proto, recipe=rec, seed=42, source=f"October {r.ref}", nn_median=nn[proto][0],
                                    nn_source=nn[proto][1], bal_acc_excl=r.bal_acc_excl, acc_excl=r.acc_excl,
                                    bal_lo=np.nan, bal_hi=np.nan, folds="1/1"))
    points = pd.DataFrame(pts)
    if len(points):
        make_figure(points, args.fig)

    # ---- write tables
    args.results_dir.mkdir(parents=True, exist_ok=True)
    found = set(inv.run) if len(inv) else set()
    inv_out = pd.DataFrame(dict(run=PLANNED, found=[r in found for r in PLANNED]))
    tables = dict(pooled=pooled, per_fold=per_fold, per_g2_group=per_group, seed_spread=seeds, bootstrap=boot,
                  epoch_curves=curves, references=refs, leak_points=points, inventory=inv_out)
    for name, t in tables.items():
        t.to_csv(args.results_dir / f"{name}.csv", index=False, float_format="%.6f")
    write_doc(args, tables, nn, n_conf, unknown, len(inv), time.time() - t0)
    print(f"wrote {len(tables)} tables to {args.results_dir}, {args.doc}" + (f", {args.fig}" if len(points) else ""))
    if len(pooled):
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(pooled[["protocol", "recipe", "seed", "rule", "folds", "n", "acc", "bal_acc", "macro_f1", "kappa",
                          "n_excl", "acc_excl", "bal_acc_excl"]].round(4).to_string(index=False))
    if len(refs):
        with pd.option_context("display.width", 250, "display.max_columns", 40):
            print(refs[["ref", "protocol", "recipe", "rule", "n", "acc", "macro_f1", "bal_acc", "acc_excl",
                        "bal_acc_excl"]].round(4).to_string(index=False))


def make_figure(points: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    style = {"r50": ("#2a78d6", "o", "ResNet50 single frame"), "slow": ("#eb6834", "s", "SLOW 8x8 R50"),
             "probe": ("#1baf7a", "^", "frozen ResNet50 + logistic regression")}
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    offs = {"r50": -0.6, "slow": 0.6, "probe": 0.0}
    for rec, (col, mk, lab) in style.items():
        d = points[points.recipe == rec]
        if d.empty:
            continue
        main = d[d.seed == d.groupby("protocol").seed.transform("min")]
        x = main.nn_median + offs[rec]
        yerr = np.vstack([main.bal_acc_excl - main.bal_lo, main.bal_hi - main.bal_acc_excl])
        has = ~np.isnan(yerr).any(0)
        if has.any():
            ax.errorbar(x[has], main.bal_acc_excl[has], yerr=yerr[:, has], fmt="none", ecolor=col, elinewidth=1.2,
                        capsize=3, alpha=0.8, zorder=2)
        ax.scatter(x, main.bal_acc_excl, s=46, marker=mk, color=col, edgecolor=surface, linewidth=1.5, zorder=3,
                   label=lab)
        extra = d.drop(main.index)
        if len(extra):
            ax.scatter(extra.nn_median + offs[rec], extra.bal_acc_excl, s=18, marker=mk, facecolor="none",
                       edgecolor=col, linewidth=1.0, zorder=3)
    ax.axhline(1 / 3, color=ink2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(71, 1 / 3 + 0.01, "chance (1/3)", color=ink2, fontsize=8, ha="right", va="bottom")
    names = {"p0": "published", "grp17": "grp17", "p1": "G1 (P1)", "grp27b": "grp27b", "p2": "G2 (P2)",
             "grp27a": "grp27a"}
    xs = points.drop_duplicates("protocol").set_index("protocol").nn_median
    for i, proto in enumerate(sorted(xs.index, key=lambda p: xs[p])):  # staggered in two rows: 31.3/32.9 and 38.9/40.9
        ax.axvline(xs[proto], color=grid, linewidth=0.8, zorder=0)
        ax.text(xs[proto], 0.04 + (0.085 if i % 2 else 0.0), f"{names[proto]}\n{xs[proto]:.1f}", fontsize=7.5,
                color=ink2, ha="center", va="bottom")
    ax.set_xlim(0, 72)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("median distance from a test clip to its nearest training clip (32x32 thumbnails)", color=ink2,
                  fontsize=9)
    ax.set_ylabel("balanced accuracy, val-selected,\nlabel-conflict clips excluded", color=ink2, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(ink2)
    ax.tick_params(colors=ink2, labelsize=8)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    leg = ax.legend(loc="upper right", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(ink)
    ax.set_title("Accuracy along the leak gradient (bars: 95% group bootstrap; hollow: other seeds)", fontsize=10,
                 color=ink, loc="left")
    fig.tight_layout()
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, facecolor=surface)
    plt.close(fig)


def write_doc(args, t: dict, nn: dict, n_conf: int, unknown: list, n_files: int, secs: float) -> None:
    pooled, per_fold, per_group, seeds, boot = t["pooled"], t["per_fold"], t["per_g2_group"], t["seed_spread"], t["bootstrap"]
    curves, refs, inv = t["epoch_curves"], t["references"], t["inventory"]
    rel = lambda p: str(p.resolve().relative_to(REPO)) if p.resolve().is_relative_to(REPO) else str(p)  # noqa: E731
    L = ["# Leak-gradient results: published split, G1 events, G2 events ∪ scenes", "",
         f"*Generated by `scripts/research/06c_aggregate.py` on {time.strftime('%Y-%m-%d %H:%M')} from `{rel(args.root)}` "
         f"({n_files} prediction files, {secs:.0f} s). Do not edit by hand: rerun the script. Plan: "
         "`11_gpu_experiment_plan.md`; protocol: `10_reconstruction_qc_and_protocol.md`.*", "",
         "## 0. Definitions and completeness", "",
         "- **Rules.** *val-selected* = checkpoint with the best validation accuracy (validation = whole groups carved from "
         "the training folds by `06_make_protocol_roots.py`); *final* = last epoch trained. The probe is one fit, so both "
         "rules are the same model.",
         f"- **Label-conflict clips** = events 9, 11, 18 ({n_conf} clips; under P1 the whole G1 groups 34, 40, 44). "
         "Columns marked *excl* leave them out; that is the headline (plan, decision 5).",
         "- **Majority** = share of the most frequent class in that test set (best constant predictor; its balanced "
         "accuracy is 1/3, its κ 0).",
         "- SLOW needs 64 frames, so clips shorter than that drop out of its train, val and test sets (`n` vs `n_expected`).",
         "- Bootstrap: groups resampled with replacement, 2,000 reps, seed 0, pooled metrics recomputed; 95% percentile "
         "interval.", "",
         "Cross-fold nearest-neighbour medians (x axis of the figure):", "",
         "| protocol | NN median | source |", "|---|---|---|"]
    L += [f"| {PROTO_NAME[p]} | {nn[p][0]:.2f} | {nn[p][1]} |" for p in PROTOCOLS if p in nn]
    miss = inv[~inv.found].run.tolist()
    L += ["", f"Planned runs found: {int(inv.found.sum())} of {len(inv)}" +
          (f"; missing: {', '.join(miss)}" if miss else "") + ".",
          ("Unknown tags ignored: " + ", ".join(unknown) + ".") if unknown else "", ""]

    L += ["## 1. Pooled metrics (all test folds together)", ""]
    if len(pooled):
        p = no_probe_final(pooled).sort_values(["protocol", "recipe", "seed", "rule"], key=lambda s: s.map(
            {k: i for i, k in enumerate(PROTOCOLS)}) if s.name == "protocol" else s)
        L += [md_table(p, ["protocol", "recipe", "seed", "rule", "folds", "n", "acc", "bal_acc", "macro_f1", "kappa",
                           "n_excl", "acc_excl", "bal_acc_excl", "macro_f1_excl", "kappa_excl", "majority_excl"],
                       ["protocol", "recipe", "seed", "rule", "folds", "n", "acc", "bal acc", "macro-F1", "κ",
                        "n excl", "acc excl", "bal acc excl", "macro-F1 excl", "κ excl", "majority excl"]), ""]
    else:
        L += ["No results yet.", ""]

    L += ["## 2. Per fold", ""]
    if len(per_fold):
        pf = no_probe_final(per_fold[per_fold.protocol.isin(["p1", "p2"])]).sort_values(["protocol", "recipe", "seed", "rule", "fold"])
        cols = ["protocol", "recipe", "seed", "rule", "fold", "epoch", "n", "majority", "acc", "bal_acc", "macro_f1",
                "kappa", "n_excl", "bal_acc_excl"] + (["val_bal_acc"] if "val_bal_acc" in pf else [])
        L += [md_table(pf.assign(epoch=pf.epoch.map(lambda e: "—" if pd.isna(e) else str(int(e)))), cols), ""]
    else:
        L += ["No results yet.", ""]

    L += ["## 3. Per G2 group (P2)", "",
          "Each G2 group is a test group in exactly one fold. Confusion rows = true low / mid / high, columns = predicted.",
          ""]
    if len(per_group):
        first_seed = per_group.groupby("recipe").seed.transform("min")  # 42 for the deep recipes, 0 for the probe
        for (rec, seed, rule), g in per_group[per_group.seed == first_seed].groupby(["recipe", "seed", "rule"]):
            if rule != "val-selected":
                continue
            L += [f"**{rec}, seed {seed}, {rule}**", "",
                  md_table(g.sort_values("n", ascending=False),
                           ["g2_group", "fold", "n", "low", "mid", "high", "n_conflict", "acc", "bal_acc", "confusion"]),
                  ""]
        wide = no_probe_final(per_group).pivot_table(index="g2_group", columns=["recipe", "seed", "rule"], values="bal_acc")
        wide.columns = [f"{a} s{b} {c}" for a, b, c in wide.columns]
        L += ["Balanced accuracy per group, every P2 run set:", "",
              md_table(wide.reset_index(), ["g2_group"] + list(wide.columns)), ""]
    else:
        L += ["No P2 results yet.", ""]

    L += ["## 4. Seed spread, ResNet50 on P2", ""]
    if len(seeds):
        cols = ["rule", "subset", "seeds"] + [f"{k}_{s}" for k in ("acc", "bal_acc", "kappa") for s in ("mean", "min", "max")]
        L += [md_table(seeds, cols), ""]
    else:
        L += ["No results yet.", ""]

    L += ["## 5. Group-level bootstrap (95% intervals)", ""]
    if len(boot):
        b = no_probe_final(boot).copy()
        for k in METRICS:
            b[k + " [95%]"] = [f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in zip(b[k], b[k + "_lo"], b[k + "_hi"])]
        L += [md_table(b, ["protocol", "recipe", "seed", "rule", "subset", "folds", "n_groups", "n"] +
                       [k + " [95%]" for k in METRICS]), ""]
    else:
        L += ["No P1/P2 results yet.", ""]

    L += ["## 6. Per-epoch test curves", "",
          "Balanced accuracy on the fold's test set for every saved epoch, all clips; `*` = val-selected epoch; the last "
          "column of each row is the final epoch. *pooled* rows only where every fold reached that epoch. Full metrics "
          "(with and without conflict clips) in `epoch_curves.csv`.", ""]
    if len(curves):
        c = curves.copy()
        c["cell"] = [f"{v:.3f}{'*' if s else ''}" for v, s in zip(c.bal_acc, c.selected_by_val)]
        w = c.pivot_table(index=["protocol", "recipe", "seed", "fold"], columns="epoch", values="cell", aggfunc="first")
        w = w.reset_index().fillna("")
        w.columns = [str(x) if not isinstance(x, (int, np.integer)) else f"ep{x}" for x in w.columns]
        L += [md_table(w, list(w.columns)), ""]
    else:
        L += ["No per-epoch evaluations yet.", ""]

    L += ["## 7. October reference points, recomputed from the per-clip predictions", "",
          "R1-R5 of `scripts/review/08_run_grouped_training.sh` (`outputs/review_grouped/eval/preds_*.csv`). Their "
          "`last.ckpt` is not the final epoch (top-k 2); the final epoch exists only in the repeat run of "
          "`11_epoch_curve.sh`, which kept accuracy and macro-F1 but no per-clip predictions.", ""]
    if len(refs):
        r = refs.copy()
        L += [md_table(r, ["ref", "protocol", "recipe", "rule", "n", "acc", "bal_acc", "macro_f1", "kappa", "acc_excl",
                           "bal_acc_excl", "macro_f1_excl", "kappa_excl", "source"]), ""]
    L += ["## 8. Figure", "", f"![leak gradient]({os.path.relpath(args.fig.resolve(), args.doc.resolve().parent)})", "",
          "Balanced accuracy (val-selected, conflict clips excluded) against the median nearest-neighbour distance of "
          "each protocol. Filled markers: seed 42 (or the probe's single fit), with the 95% group bootstrap where the "
          "protocol has groups (P1, P2); hollow markers: seeds 43 and 44. Points at published and grp* without new runs "
          "are the October R1-R5. Data: `leak_points.csv`.", ""]
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
