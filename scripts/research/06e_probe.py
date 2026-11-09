#!/usr/bin/env python
"""Frozen-feature probe of the leak gradient: ImageNet ResNet50 centre-frame features + logistic regression.

Stage "extract" (GPU when available, else CPU): the clip universe is exactly the paths of --groups
(outputs/research/sessions/union_groups.csv, 7,701 clips = the files of data/ouc-cge-320; the published manifests
also list 4 raw clips that were never transcoded, which are not used).  For every clip, read from --data-root,
decode the centre frame with PyAV
(sequential decode to index n_frames // 2, the frame the project's resnet50 pathway takes in val/test:
OUCCGEDataset start_point = (total_frames - window + 1) // 2 with window 1), apply the project's own
src.data.transforms.get_val_transforms (Resize 256, CenterCrop 224, ImageNet normalisation) and take the 2048-d
global-pool output of torchvision resnet50 with ResNet50_Weights.IMAGENET1K_V1 (the weights the project's resnet50
backbone starts from).  Decoding runs in --workers spawned processes that import no torch.  A clip whose file is
missing, or that fails to decode or to transform, is logged ("extract: SKIPPED <path>: <reason>") and kept in the
feature file with ok=False; the number and list of skipped clips go into the feature file's meta and into every
summary json of the fit stage (n_skipped_extraction), and the fits run on the clips that were extracted.

Stage "fit" (CPU): StandardScaler + multinomial LogisticRegression(C=1.0, max_iter=2000) fitted on the train split and
applied to val and test, under
  P0       published split                      (split_published of --groups)          run p0_probe_s0
  P1, P2   folds 0-4 of 06_make_protocol_roots  (data/p1_f{k}, data/p2_f{k})            runs p{1,2}_f{k}_probe_s0
  October  grp27a, grp27b, grp17                (docs/review/data/grouped/splits/)     runs grp27a_probe_s0, ...
Per-clip predictions use the format and tag convention of 05_eval_checkpoints.py / 06b_run_protocols.sh, so that
06c_aggregate.py reads them as recipe "probe": preds_<run>_best_{val,test}.csv (path,label,pred,p0,p1,p2) and
summary_<run>_best_{val,test}.json in --out-eval.  "best" is the single fitted model (no checkpoint choice).

Inputs : outputs/research/sessions/union_groups.csv, data/ouc-cge-320/videos,
         data/p1_f*/ and data/p2_f*/data_labels (06_make_protocol_roots.py),
         docs/review/data/grouped/splits/split_{grp27a,grp27b,grp17}.csv
Outputs: --features (default outputs/research_gpu/_summary/probe_features.npz: path, label, ok, n_frames,
         frame_index, feats float32 (n, 2048), meta json), --out-eval (default outputs/research_gpu/_summary/eval)

Usage:
  on the GPU host (06b block "probe"):
    python scripts/research/06e_probe.py --workers 16
  CPU smoke test on the workstation (at most 40 clips):
    CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/06e_probe.py --limit 40 --workers 2 \\
        --threads 3 --features <scratch>/probe_features.npz --out-eval <scratch>/eval
  fit only, on an existing feature file:
    python scripts/research/06e_probe.py --stage fit --features <npz> --out-eval <dir>
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import av
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True  # no __pycache__ under scripts/review
sys.path.insert(0, str(REPO / "scripts" / "review"))
from common import load_all_manifests  # noqa: E402  (numpy/pandas only: safe in the decode workers)

DEFAULT_OUT = REPO / "outputs" / "research_gpu" / "_summary"
OCTOBER = ["grp27a", "grp27b", "grp17"]
SEED = 0


def decode_centre(job):
    """(idx, path) -> centre frame as uint8 RGB.  n_frames from the container header (cv2's CAP_PROP_FRAME_COUNT
    reads the same field); packet count when the header has none."""
    idx, path = job
    out = dict(idx=idx, ok=False, n_frames=0, frame_index=-1, error="")
    if not os.path.isfile(path):
        out["error"] = "missing file"
        return out
    try:
        with av.open(path) as c:
            s = c.streams.video[0]
            n = int(s.frames or 0)
            if n <= 0:
                n = sum(1 for p in c.demux(s) if p.size > 0)
        if n <= 0:
            out["error"] = "no video frames"
            return out
        target, i, frame = n // 2, -1, None
        with av.open(path) as c:
            s = c.streams.video[0]
            for fr in c.decode(s):
                i, frame = i + 1, fr
                if i >= target:
                    break
        if frame is None:
            out["error"] = "no decodable frame"
            return out
        out.update(ok=True, n_frames=n, frame_index=i, rgb=frame.to_ndarray(format="rgb24"))
    except Exception as e:  # missing or unreadable file
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def clip_list(groups: Path, limit: int) -> pd.DataFrame:
    """The clip universe: path, label and published split of every row of union_groups.csv, sorted by path."""
    df = pd.read_csv(groups, usecols=["path", "label", "split_published"]).rename(columns={"split_published": "split"})
    if df["path"].duplicated().any():
        raise SystemExit(f"ABORT: duplicated paths in {groups}")
    df = df.sort_values("path").reset_index(drop=True)
    if limit:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, limit).round().astype(int))].reset_index(drop=True)
    return df


def extract(args) -> None:
    import torch
    import torchvision

    sys.path.insert(0, str(REPO))
    from src.data.transforms import get_val_transforms

    if args.threads:
        torch.set_num_threads(args.threads)
    df = clip_list(args.groups, args.limit)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"extract: {len(df)} clips of {args.groups.name}, videos from {args.data_root}, on {dev} workers={args.workers} batch={args.batch}", flush=True)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model = model.eval().to(dev)
    tf = get_val_transforms()

    n = len(df)
    feats = np.zeros((n, 2048), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    n_frames = np.zeros(n, dtype=np.int64)
    frame_index = np.full(n, -1, dtype=np.int64)
    errors = {}
    buf_x, buf_i = [], []

    def flush():
        if not buf_x:
            return
        with torch.inference_mode():
            f = model(torch.stack(buf_x).to(dev)).float().cpu().numpy()
        feats[buf_i] = f
        buf_x.clear()
        buf_i.clear()

    jobs = [(i, str(args.data_root / p)) for i, p in enumerate(df["path"])]
    t0 = time.time()
    pool = get_context("spawn").Pool(args.workers) if args.workers > 0 else None
    it = pool.imap(decode_centre, jobs, chunksize=4) if pool else map(decode_centre, jobs)
    for k, r in enumerate(it):
        i = r["idx"]
        n_frames[i], frame_index[i] = r["n_frames"], r["frame_index"]
        if not r["ok"]:
            errors[df.path[i]] = r["error"]
            print(f"extract: SKIPPED {df.path[i]}: {r['error']}", flush=True)
            continue
        try:
            x = torch.from_numpy(r["rgb"]).permute(2, 0, 1).float().div(255.0).unsqueeze(0)  # (T=1, C, H, W)
            x = tf(x)[:, 0]                                                                   # (C, T, H, W) -> (C, H, W)
        except Exception as e:
            errors[df.path[i]] = f"transform: {type(e).__name__}: {e}"[:300]
            print(f"extract: SKIPPED {df.path[i]}: {errors[df.path[i]]}", flush=True)
            continue
        ok[i] = True
        buf_x.append(x)
        buf_i.append(i)
        if len(buf_x) == args.batch:
            flush()
        if (k + 1) % 1000 == 0:
            print(f"extract: {k + 1}/{n} in {time.time() - t0:.0f}s", flush=True)
    flush()
    if pool:
        pool.close()
        pool.join()
    dt = time.time() - t0
    meta = dict(groups=str(args.groups), data_root=str(args.data_root), limit=args.limit,
                n_skipped=int(n - ok.sum()), skipped=errors, weights="torchvision ResNet50_Weights.IMAGENET1K_V1",
                transform="src.data.transforms.get_val_transforms (Resize 256, CenterCrop 224, ImageNet norm)",
                frame="PyAV sequential decode, index n_frames // 2", device=str(dev), torch=torch.__version__,
                torchvision=torchvision.__version__, pyav=av.__version__, seconds=round(dt, 1), n=n, n_ok=int(ok.sum()))
    args.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.features, path=np.array([str(p) for p in df.path], dtype=str), label=df.label.values.astype(np.int64),
                        ok=ok, n_frames=n_frames, frame_index=frame_index, feats=feats, meta=json.dumps(meta))
    print(f"features: {int(ok.sum())}/{n} clips ok, {n - int(ok.sum())} skipped, {dt:.0f}s, device {dev} -> "
          f"{args.features}", flush=True)


def protocol_splits(args) -> list[tuple[str, pd.DataFrame]]:
    """(run name, DataFrame path/label/split) for every protocol of the probe."""
    out = [("p0_probe_s0", clip_list(args.groups, 0))]
    for proto in ("p1", "p2"):
        for k in range(5):
            root = args.roots_base / f"{proto}_f{k}"
            if not (root / "data_labels" / "train.csv").exists():
                raise SystemExit(f"ABORT: missing root {root}; run scripts/research/06_make_protocol_roots.py")
            out.append((f"{proto}_f{k}_probe_s0", load_all_manifests(root)[["path", "label", "split"]]))
    for name in OCTOBER:
        f = args.splits_dir / f"split_{name}.csv"
        if not f.exists():
            raise SystemExit(f"ABORT: missing {f}")
        out.append((f"{name}_probe_s0", pd.read_csv(f)[["path", "label", "split"]]))
    return out


def fit(args) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    z = np.load(args.features, allow_pickle=False)
    paths = z["path"].astype(str)
    ok = z["ok"]
    if len(set(paths)) != len(paths):
        raise SystemExit("ABORT: duplicated paths in the feature file")
    row = {p: i for i, p in enumerate(paths) if ok[i]}
    X_all = z["feats"].astype(np.float64)
    meta = json.loads(str(z["meta"]))
    n_skipped = int(meta.get("n_skipped", len(paths) - len(row)))
    print(f"fit: features {args.features} ({len(row)} usable clips of {len(paths)}, {n_skipped} skipped at "
          f"extraction; limit={meta.get('limit')})", flush=True)
    args.out_eval.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for run, sp in protocol_splits(args):
        sp = sp.drop_duplicates("path")
        sp = sp[sp["path"].isin(row)].copy()
        tr = sp[sp.split == "train"]
        ytr = tr["label"].to_numpy(int)
        if len(np.unique(ytr)) < 2 or (sp.split == "test").sum() == 0:
            msg = f"fit {run} SKIPPED (train classes {np.unique(ytr).tolist()}, test {(sp.split == 'test').sum()} clips)"
            print(msg, flush=True)
            if not args.limit:
                n_fail += 1
            continue
        t0 = time.time()
        Xtr = X_all[[row[p] for p in tr.path]]
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(sc.transform(Xtr), ytr)
        n_iter = int(np.max(clf.n_iter_))
        line = [f"fit {run} train={len(tr)} {np.bincount(ytr, minlength=3).tolist()} iter={n_iter}"]
        for split in ("val", "test"):
            ev = sp[sp.split == split]
            if ev.empty:
                continue
            P = np.zeros((len(ev), 3))
            P[:, clf.classes_] = clf.predict_proba(sc.transform(X_all[[row[p] for p in ev.path]]))
            y, pred = ev["label"].to_numpy(int), P.argmax(1)
            df = pd.DataFrame(dict(path=ev.path.values, label=y, pred=pred, p0=P[:, 0], p1=P[:, 1], p2=P[:, 2]))
            tag = f"{run}_best_{split}"
            df.to_csv(args.out_eval / f"preds_{tag}.csv", index=False)
            acc = float(accuracy_score(y, pred))
            summary = dict(tag=tag, model="probe", ckpt=f"logreg C=1.0 max_iter=2000 on {args.features.name}",
                           split=split, mode="identity", evaluated=int(len(df)), n_train=int(len(tr)),
                           n_skipped_extraction=n_skipped,
                           train_class_counts=np.bincount(ytr, minlength=3).tolist(), n_iter=n_iter,
                           converged=bool(n_iter < 2000), sklearn_accuracy=acc,
                           sklearn_balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                           sklearn_f1_macro=float(f1_score(y, pred, average="macro")),
                           confusion_rows_true_low_mid_high=confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
                           features_meta=meta, seconds=round(time.time() - t0, 1))
            with open(args.out_eval / f"summary_{tag}.json", "w") as fh:
                json.dump(summary, fh, indent=2)
            line.append(f"{split} n={len(df)} acc={acc:.4f} bal={summary['sklearn_balanced_accuracy']:.4f}")
        print("  ".join(line) + f"  {time.time() - t0:.0f}s", flush=True)
    if n_fail:
        raise SystemExit(f"ABORT: {n_fail} protocol fit(s) skipped on the full feature set")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["all", "extract", "fit"], default="all")
    ap.add_argument("--groups", type=Path, default=REPO / "outputs" / "research" / "sessions" / "union_groups.csv",
                    help="clip universe (path, label, split_published)")
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "ouc-cge-320", help="where videos/ is read")
    ap.add_argument("--roots-base", type=Path, default=REPO / "data", help="parent of p1_f*/ and p2_f*/")
    ap.add_argument("--splits-dir", type=Path, default=REPO / "docs" / "review" / "data" / "grouped" / "splits")
    ap.add_argument("--features", type=Path, default=DEFAULT_OUT / "probe_features.npz")
    ap.add_argument("--out-eval", type=Path, default=DEFAULT_OUT / "eval")
    ap.add_argument("--workers", type=int, default=16, help="decode processes (0 = decode in the main process)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only N clips, evenly spaced over the sorted paths")
    ap.add_argument("--force-extract", action="store_true", help="re-extract even if --features exists")
    args = ap.parse_args()
    if args.data_root.is_absolute() is False:
        args.data_root = Path.cwd() / args.data_root
    if args.stage in ("all", "extract"):
        reuse = args.features.exists() and not args.force_extract
        if reuse:
            want = set(clip_list(args.groups, args.limit).path)
            try:
                z = np.load(args.features, allow_pickle=False)
                same = set(z["path"].astype(str)) == want
            except (ValueError, KeyError, OSError) as e:  # e.g. an older file with an object-dtype path array
                print(f"extract: cannot read {args.features} ({type(e).__name__}); re-extracting", flush=True)
                same = False
            if not same:
                print(f"extract: {args.features} holds another clip set; re-extracting", flush=True)
                reuse = False
        if reuse:
            print(f"extract: reusing {args.features} (--force-extract to redo)", flush=True)
        else:
            extract(args)
    if args.stage in ("all", "fit"):
        fit(args)


if __name__ == "__main__":
    main()
