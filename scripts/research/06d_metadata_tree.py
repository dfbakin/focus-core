#!/usr/bin/env python
"""Export-batch shortcut check: does a depth-4 decision tree on container metadata alone predict the class, and does
that survive the grouped protocols?

Features, per clip, from the raw release headers (outputs/research/raw_header_audit.csv, joined on
path = "videos/" + relpath): video codec (one-hot), width, height, number of audio tracks, average frame rate,
duration bin (container duration in s, bin edges 4.5, 5.5, 6.5, 9.5, 10.5, 11.5, 15.5, 60) and encoder tag (container
encoder | video-stream encoder, one-hot).  No pixels, no audio content.

Protocols: P0 published split (train -> test, union_groups.csv split_published); P1 and P2 folds 0-4 of
06_make_protocol_roots.py (train -> test of data/p1_f{k}, data/p2_f{k}; the validation groups are not used, so the
tree sees exactly the training clips of the deep runs).  DecisionTreeClassifier(max_depth=4, random_state=0).
Reported per fold and pooled over the five test folds: accuracy, balanced accuracy, macro-F1, Cohen's kappa, the
majority baseline (share of the most frequent class in that test set, the best constant predictor; its balanced
accuracy is 1/3 and its kappa 0), and the three most important features of the tree.

Inputs : outputs/research/raw_header_audit.csv, outputs/research/sessions/union_groups.csv, data/p{1,2}_f{0..4}/data_labels
Outputs: outputs/research/protocols/metadata_tree.csv (and the table on stdout)

Usage (CPU, seconds):
  .venv/bin/python scripts/research/06d_metadata_tree.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.tree import DecisionTreeClassifier

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True  # no __pycache__ under scripts/review
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import load_all_manifests  # noqa: E402

DUR_EDGES = [4.5, 5.5, 6.5, 9.5, 10.5, 11.5, 15.5, 60.0]


def features(audit: Path, paths: pd.Series) -> pd.DataFrame:
    h = pd.read_csv(audit, low_memory=False)
    h["path"] = "videos/" + h["relpath"].astype(str)
    h = h.set_index("path").reindex(paths)
    if h["v_codec"].isna().any():
        missing = h.index[h["v_codec"].isna()].tolist()
        raise SystemExit(f"ABORT: no header metadata for {len(missing)} clips, e.g. {missing[:3]}")
    X = pd.DataFrame(index=h.index)
    X["width"] = h["v_width"].astype(float)
    X["height"] = h["v_height"].astype(float)
    X["audio_tracks"] = h["n_audio_streams"].astype(float)
    X["fps"] = h["v_average_rate"].astype(float)
    X["duration_bin"] = np.digitize(h["container_duration_s"].astype(float), DUR_EDGES)
    enc = h["ctag_encoder"].fillna("none").astype(str) + "|" + h["vtag_encoder"].fillna("none").astype(str)
    X = X.join(pd.get_dummies(h["v_codec"].astype(str), prefix="codec").astype(float))
    X = X.join(pd.get_dummies(enc, prefix="encoder").astype(float))
    if X.isna().any().any():
        raise SystemExit(f"ABORT: missing metadata in columns {X.columns[X.isna().any()].tolist()}")
    return X


def metrics(y, p) -> dict:
    return dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p),
                macro_f1=f1_score(y, p, average="macro"), kappa=cohen_kappa_score(y, p),
                majority_baseline=np.bincount(y, minlength=3).max() / len(y))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", type=Path, default=REPO / "outputs/research/raw_header_audit.csv")
    ap.add_argument("--groups", type=Path, default=REPO / "outputs/research/sessions/union_groups.csv")
    ap.add_argument("--roots-base", type=Path, default=REPO / "data")
    ap.add_argument("--out", type=Path, default=REPO / "outputs/research/protocols/metadata_tree.csv")
    args = ap.parse_args()

    u = pd.read_csv(args.groups)
    X = features(args.audit, u["path"])
    y_all = pd.Series(u["label"].values, index=u["path"].values)
    print(f"{len(X)} clips, {X.shape[1]} metadata features: {', '.join(X.columns)}")

    splits = {("P0", "published"): u.set_index("path")["split_published"]}
    for proto in ("p1", "p2"):
        for k in range(5):
            root = args.roots_base / f"{proto}_f{k}"
            if not (root / "data_labels" / "train.csv").exists():
                raise SystemExit(f"ABORT: missing root {root}; run scripts/research/06_make_protocol_roots.py")
            m = load_all_manifests(root)
            splits[(proto.upper(), k)] = m.set_index("path")["split"]

    rows, pooled = [], {}
    for (proto, fold), sp in splits.items():
        tr, te = sp.index[sp == "train"], sp.index[sp == "test"]
        tree = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X.loc[tr].values, y_all.loc[tr].values)
        p = tree.predict(X.loc[te].values)
        y = y_all.loc[te].values
        imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
        top = "; ".join(f"{k} {v:.2f}" for k, v in imp.head(3).items() if v > 0)
        c = np.bincount(y, minlength=3)
        rows.append(dict(protocol=proto, fold=fold, n_train=len(tr), n_test=len(te), test_low=c[0], test_mid=c[1],
                         test_high=c[2], **metrics(y, p), top_features=top))
        ys, ps = pooled.setdefault(proto, ([], []))
        ys.append(y)
        ps.append(p)
    for proto, (ys, ps) in pooled.items():
        if proto == "P0":
            continue
        y, p = np.concatenate(ys), np.concatenate(ps)
        c = np.bincount(y, minlength=3)
        rows.append(dict(protocol=proto, fold="pooled", n_train=np.nan, n_test=len(y), test_low=c[0], test_mid=c[1],
                         test_high=c[2], **metrics(y, p), top_features=""))
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 80):
        print(df.round(4).to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
