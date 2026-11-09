#!/usr/bin/env python
"""Scene-collapse table: how a held-out group's clips spread over the predicted classes.

Under P2 every test fold is a handful of whole G2 scenes (under P1, whole G1 events).  For each
held-out group and each run this script measures how concentrated the predictions are on one
class (modal predicted share), which class that is, and whether it is the group's majority
label.  A model that recognises the room and maps it to one class gives a high modal share
whatever the group's true label mix; a model that reads engagement gives a predicted mix close
to the true mix.  Groups are the unit: per-run summaries are unweighted means over groups.

Reads  outputs/research_gpu/_summary/eval/preds_<run>_best_test.csv (probe: its single test file)
       outputs/research/sessions/union_groups.csv
Writes outputs/research/protocols/results/scene_collapse.csv   (one row per run x held-out group)
       docs/research/12b_scene_collapse.md                      (generated; do not edit by hand)
"""
import argparse
import datetime as dt
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CLASSES = ["low", "mid", "high"]
RUNS = {"p2": [("r50", 42), ("r50", 43), ("r50", 44), ("slow", 42), ("probe", 0)],
        "p1": [("r50", 42), ("probe", 0)]}
GROUP_COL = {"p2": "G2_event_scene27", "p1": "G1_event"}
FOLD_COL = {"p2": "G2_event_scene27_fold", "p1": "G1_event_fold"}
CONFLICT_EVENTS = {9, 11, 18}


def find_preds(eval_dir: Path, run: str):
    c = sorted(glob.glob(str(eval_dir / f"preds_{run}_best_test.csv")))
    if not c:
        c = [f for f in sorted(glob.glob(str(eval_dir / f"preds_{run}*test*.csv"))) if "_ep" not in Path(f).name]
    return Path(c[0]) if c else None


def mix(a: np.ndarray) -> np.ndarray:
    return np.bincount(a, minlength=3) / max(len(a), 1)


def collect(args) -> pd.DataFrame:
    g = pd.read_csv(args.groups)
    g["conflict"] = g["event_id"].isin(CONFLICT_EVENTS)
    rows, missing = [], []
    for proto, runs in RUNS.items():
        gcol, fcol = GROUP_COL[proto], FOLD_COL[proto]
        for recipe, seed in runs:
            for k in range(5):
                run = f"{proto}_f{k}_{recipe}_s{seed}"
                f = find_preds(args.eval_dir, run)
                if f is None:
                    missing.append(run)
                    continue
                d = pd.read_csv(f).merge(g[["path", gcol, fcol, "conflict"]], on="path", how="left")
                if d[gcol].isna().any() or not (d[fcol] == k).all():
                    raise SystemExit(f"{run}: test clips outside fold {k} of {gcol}")
                for grp, dd in d.groupby(gcol):
                    t, p = mix(dd.label.values), mix(dd.pred.values)
                    modal = int(p.argmax())
                    present = [c for c in range(3) if (dd.label == c).any()]
                    rec = [((dd.label == c) & (dd.pred == c)).sum() / (dd.label == c).sum() for c in present]
                    rows.append(dict(protocol=proto, recipe=recipe, seed=seed, fold=k, group=int(grp), n=len(dd),
                                     n_conflict=int(dd.conflict.sum()), n_classes_true=len(present),
                                     true_low=t[0], true_mid=t[1], true_high=t[2],
                                     pred_low=p[0], pred_mid=p[1], pred_high=p[2],
                                     true_majority=CLASSES[int(t.argmax())], true_majority_share=float(t.max()),
                                     modal_pred=CLASSES[modal], modal_pred_share=float(p.max()),
                                     true_share_of_modal_pred=float(t[modal]),
                                     scene_decision_correct=bool(modal == int(t.argmax())),
                                     mix_error=float(np.abs(p - t).sum() / 2),
                                     acc=float((dd.label == dd.pred).mean()), bal_acc=float(np.mean(rec))))
    if missing:
        print("missing runs:", " ".join(missing))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (proto, recipe, seed), d in df.groupby(["protocol", "recipe", "seed"], sort=False):
        out.append(dict(protocol=proto, recipe=recipe, seed=seed, groups=len(d),
                        mean_modal_pred_share=d.modal_pred_share.mean(),
                        mean_true_majority_share=d.true_majority_share.mean(),
                        groups_modal_share_ge_060=int((d.modal_pred_share >= 0.60).sum()),
                        scene_decisions_correct=int(d.scene_decision_correct.sum()),
                        mean_mix_error=d.mix_error.mean(),
                        mean_group_acc=d.acc.mean(), mean_group_bal_acc=d.bal_acc.mean()))
    return pd.DataFrame(out)


def fmt_mix(r, prefix):
    return "/".join(f"{getattr(r, f'{prefix}_{c}'):.2f}" for c in CLASSES)


def markdown(df: pd.DataFrame, summ: pd.DataFrame, args) -> str:
    L = ["# Scene collapse: where a held-out group's predictions go", "",
         f"*Generated by `scripts/research/06f_scene_collapse.py` on {dt.datetime.now():%Y-%m-%d %H:%M} from "
         f"`{args.eval_dir.relative_to(REPO)}` and `{args.groups.relative_to(REPO)}`. Do not edit by hand: rerun the "
         f"script. Companion of `12_leak_gradient_results.md`; interpretation in `13_leak_gradient_interpretation.md`.*", "",
         "## 0. Definitions", "",
         "- **Group** = the unit held out by the protocol: a G2 scene under P2 (16 groups), a G1 event under P1 (45 groups). "
         "Every group is a test group in exactly one fold; the validation-selected checkpoint of that fold predicts it "
         "(the probe has one fit per fold).",
         "- **Modal predicted share** = fraction of the group's clips assigned to the most frequent *predicted* class. "
         "**True majority share** = fraction of clips carrying the most frequent *true* label; a model whose predicted mix "
         "matched the true mix would have modal share equal to it.",
         "- **Scene decision correct** = the modal predicted class is the group's majority label. "
         "**Mix error** = total-variation distance between predicted and true class mix (0 = identical, 1 = disjoint).",
         "- Per-run summaries are unweighted over groups. Label-conflict clips (events 9, 11, 18) are included, so groups 9, 10 "
         "and 15 carry labels that disagree between cameras (`n_conflict`).", "",
         "## 1. Per-run summary", "",
         "| protocol | recipe | seed | groups | mean modal pred share | mean true majority share | groups with modal share ≥ 0.60 | scene decisions correct | mean mix error | mean group acc | mean group bal acc |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summ.itertuples():
        L.append(f"| {r.protocol} | {r.recipe} | {r.seed} | {r.groups} | {r.mean_modal_pred_share:.2f} | "
                 f"{r.mean_true_majority_share:.2f} | {r.groups_modal_share_ge_060} | {r.scene_decisions_correct}/{r.groups} | "
                 f"{r.mean_mix_error:.2f} | {r.mean_group_acc:.3f} | {r.mean_group_bal_acc:.3f} |")
    L += ["", "## 2. P2: every G2 group under every run", "",
          "Cell = modal predicted class and its share, then the group's accuracy. True mix is low/mid/high.", ""]
    p2 = df[df.protocol == "p2"]
    runs = [(rc, s) for rc, s in RUNS["p2"] if ((p2.recipe == rc) & (p2.seed == s)).any()]
    head = "| group | fold | n | n_conflict | true low/mid/high | " + " | ".join(f"{rc} s{s}" for rc, s in runs) + " |"
    L += [head, "|" + "---|" * (5 + len(runs))]
    base = p2.drop_duplicates("group").sort_values(["fold", "group"])
    for b in base.itertuples():
        cells = []
        for rc, s in runs:
            r = p2[(p2.recipe == rc) & (p2.seed == s) & (p2.group == b.group)]
            if r.empty:
                cells.append("—")
            else:
                r = r.iloc[0]
                cells.append(f"{r.modal_pred} {r.modal_pred_share:.2f}, acc {r.acc:.2f}")
        L.append(f"| {b.group} | {b.fold} | {b.n} | {b.n_conflict} | {fmt_mix(b, 'true')} | " + " | ".join(cells) + " |")
    L += ["", "## 3. P1: G1 events, ResNet50 seed 42 and probe", "",
          "Same cells; 45 events, sorted by fold then size. The full per-group table for every run is in `scene_collapse.csv`.", ""]
    p1 = df[df.protocol == "p1"]
    runs1 = [(rc, s) for rc, s in RUNS["p1"] if ((p1.recipe == rc) & (p1.seed == s)).any()]
    L += ["| event group | fold | n | n_conflict | true low/mid/high | " + " | ".join(f"{rc} s{s}" for rc, s in runs1) + " |",
          "|" + "---|" * (5 + len(runs1))]
    base1 = p1.drop_duplicates("group").sort_values(["fold", "n"], ascending=[True, False])
    for b in base1.itertuples():
        cells = []
        for rc, s in runs1:
            r = p1[(p1.recipe == rc) & (p1.seed == s) & (p1.group == b.group)]
            cells.append("—" if r.empty else f"{r.iloc[0].modal_pred} {r.iloc[0].modal_pred_share:.2f}, acc {r.iloc[0].acc:.2f}")
        L.append(f"| {b.group} | {b.fold} | {b.n} | {b.n_conflict} | {fmt_mix(b, 'true')} | " + " | ".join(cells) + " |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, default=REPO / "outputs/research_gpu/_summary/eval")
    ap.add_argument("--groups", type=Path, default=REPO / "outputs/research/sessions/union_groups.csv")
    ap.add_argument("--out-csv", type=Path, default=REPO / "outputs/research/protocols/results/scene_collapse.csv")
    ap.add_argument("--out-md", type=Path, default=REPO / "docs/research/12b_scene_collapse.md")
    args = ap.parse_args()
    df = collect(args)
    summ = summarise(df)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    args.out_md.write_text(markdown(df, summ, args))
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print(f"wrote {args.out_csv.relative_to(REPO)} ({len(df)} rows) and {args.out_md.relative_to(REPO)}")


if __name__ == "__main__":
    main()
