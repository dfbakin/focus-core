# Scene-disjoint ("leak-free") OUC-CGE splits: how to reproduce

The published OUC-CGE split is a random shuffle of consecutive 10-second clips cut from 21 recorded
sessions. Almost every test clip therefore has a near-identical training clip, and a model can
score 0.99 by recognising the scene. The scripts in this directory

1. build train / validation / test splits in which **no scene is shared** between the three parts,
2. train the project's own recipes (`python -m src.train`, unchanged presets) on them,
3. compare the result with the published numbers.

Method, per-run details and threats to validity: `docs/review/03_grouped_split_training.md`.
Reference artefacts of the published runs: `docs/review/data/grouped/`.

## Quick start

```bash
# environment: see "Environment" below; from the repository root
bash scripts/review/reproduce.sh
```

This fetches the clips with DVC, writes the three leak-free dataset roots, runs the five trainings
(about 50 minutes on one RTX 5090) and prints the published numbers next to the new ones.
The training needs a GPU; everything before it runs on a CPU.

Only the leak-free data, no training (seconds once the clips are there):

```bash
dvc pull data/ouc-cge-320.dvc                      # 7,701 clips at 320p, 4.4 GB
python scripts/review/09_materialize_splits.py     # data/grp27a, data/grp27b, data/grp17
```

Each root holds `data_labels/{train,val,test}.csv` in the project format and a `videos` symlink to
`../ouc-cge-320/videos`, so `OUCCGEDataset` and every entry point read it through `data.root=`:

```bash
export MLFLOW_ALLOW_FILE_STORE=true                # only needed with MLflow >= 3.17, see below
python -m src.train +experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16 \
    data.root=data/grp27a
```

One of the published runs with its per-clip evaluation:

```bash
RUNS="R2" bash scripts/review/08_run_grouped_training.sh     # R1..R5, see the table below
```

The best two checkpoints by validation accuracy and `last.ckpt` (a copy of the later of the two,
not the final epoch) are written to `outputs/review/<run>/checkpoints/`; summaries, per-clip
predictions and logs to `outputs/review_grouped/`. Both are git-ignored.

The checkpoints of the 2026-11-05 rerun are stored in DVC (`outputs/review.dvc`: the 15 checkpoint
files and the resolved configs of R1-R5, 3.0 GB to download):

```bash
dvc pull outputs/review.dvc
python scripts/review/09_materialize_splits.py      # dataset roots, if not created yet
python scripts/review/05_eval_checkpoints.py --model resnet50 --root data/grp27a --split test \
    --ckpt "outputs/review/R2_r50_grp27a/checkpoints/epoch=02-val_acc=0.6005.ckpt" \
    --out outputs/review_eval --tag R2_test        # 0.5032
```

These are the rerun's files: R1-R4 give the published numbers exactly, R5 gives the rerun's 0.5180
and not the published 0.5088 (see "Rerun on a second host"). Their hashes are listed in
`docs/review/data/grouped/rerun_2026-11-05/checkpoints.sha256`. `08_run_grouped_training.sh` does
not use them: it trains again and writes next to them.

## The three splits

A scene is a connected component of clips linked by near-identical frames: two clips are linked when
the RMSE between 32x32 thumbnails of any of their first / centre / last frames is at most a
threshold (0-255 scale). Whole components go to one split, target 70 / 10 / 20, classes balanced.

| root | threshold | train clips (scenes) | val | test | what it holds out |
|---|---|---|---|---|---|
| `data/grp27a` | 27.5 | 5386 (83) | 771 (9) | 1544 (9) | strict; the test part is dominated by one large mixed-label scene |
| `data/grp27b` | 27.5 | 5187 (84) | 784 (7) | 1730 (10) | strict; different test scenes (2 % overlap with grp27a's test) |
| `data/grp17` | 17.5 | 5391 (437) | 770 (26) | 1540 (67) | looser, every scene has one label; 30 % of test clips still have a train clip within 27.5 |
| `data/ouc-cge-320` | - | 6160 | 770 | 771 | the published random split (control) |

The exact assignment of every clip is `docs/review/data/grouped/splits/split_<name>.csv`
(`path,label,split,cluster`). Four of the 7,705 listed clips failed the 320p transcode
(`data/ouc-cge-320/transcode_failures.txt`) and are in no split.

## Rebuilding the splits instead of using the published files

```bash
STAGES="data rebuild" bash scripts/review/reproduce.sh
```

| step | script | input | output |
|---|---|---|---|
| thumbnails | `01_frame_cache.py --thumbs-only` | the clips | `outputs/review_cache/frames_{train,val,test}.npz` |
| scene clusters | `03b_scene_clusters_thumb.py` | thumbnails | `outputs/review_out/scene_clusters.csv`, `exp3b_clusters_by_thr.csv` |
| check | `10_compare_with_published.py --clusters` | rebuilt and published clusters | same partition or not, per threshold |
| splits | `run_07_splits.sh` (`07_make_grouped_splits.py` x 3) | clusters, thumbnails | dataset roots, `split_<name>.csv`, leakage statistics; compared byte for byte with the published split files |

The split step alone, from the published clusters (CPU, one minute):
`bash scripts/review/run_07_splits.sh`.

## Rerun on a second host (2026-11-05)

Everything above was run on a new, empty vast.ai instance: RTX 5090, `git clone` of commit
203cfd7 plus the files added with this directory, dependencies installed as in the `Dockerfile`,
clips from the DVC remote. Package versions are listed under "Environment".

**Data: exact.** From the raw clips, `STAGES="data rebuild"` produced

- `scene_clusters.csv` and `exp3b_clusters_by_thr.csv` byte-identical to the published files
  (all eleven thresholds),
- `split_grp27a.csv`, `split_grp27b.csv`, `split_grp17.csv` byte-identical to the published files,
- the same leakage statistics (`split_<name>_summary.json` equal value for value): nearest train
  clip of a test clip at median thumbnail RMSE 65.6 / 38.9 / 31.3 for grp27a / grp27b / grp17,
  against 2.7 on the published random split.

`09_materialize_splits.py` writes manifests byte-identical to those of `07_make_grouped_splits.py`.

**Training: exact for ResNet50, close for SLOW.** `bash scripts/review/reproduce.sh` (commit 203cfd7,
seed 42) gave, published / rerun:

| run | best epoch | val acc | test acc | test macro-F1 | same test prediction |
|---|---|---|---|---|---|
| R1 ResNet50, random split | 6 / 6 | 0.9961 / 0.9961 | 0.9935 / 0.9935 | 0.9936 / 0.9936 | 771 of 771 |
| R2 ResNet50, grp27a | 2 / 2 | 0.6005 / 0.6005 | 0.5032 / 0.5032 | 0.4592 / 0.4592 | 1544 of 1544 |
| R3 ResNet50, grp27b | 1 / 1 | 0.6314 / 0.6314 | 0.5266 / 0.5266 | 0.4553 / 0.4553 | 1730 of 1730 |
| R4 ResNet50, grp17 | 5 / 5 | 0.4013 / 0.4013 | 0.6903 / 0.6903 | 0.6309 / 0.6309 | 1540 of 1540 |
| R5 SLOW R50, grp27a | 1 / 1 | 0.5422 / 0.5759 | 0.5088 / 0.5180 | 0.4517 / 0.4467 | 1391 of 1531 |

- R1-R4 are bit-identical: every logged per-epoch metric and every per-clip prediction file
  (softmax included) equals the published one. Both hosts had an RTX 5090; other GPUs were not
  tried.
- R5 is not bit-reproducible (`deterministic: false`; the source of the difference was not
  isolated). Three SLOW runs of the same recipe all select epoch 1 and score 0.509 (published),
  0.518 (rerun) and 0.521 (the per-epoch run below). Treat about 0.01 as the run-to-run noise of
  that number. Early stopping ended the rerun after 11 epochs, the published run after 12.
- The leakage probe of the published split (`run_02_04.sh`, `run_03b_04.sh`, `run_05_06.sh`)
  also repeats exactly: all summary files equal, stored ResNet50 0.9935 and SLOW 0.9987 on the
  published test split, SLOW with shuffled / reversed / single repeated frame 0.9961 / 0.9974 /
  0.9882.

Record of the rerun: `docs/review/data/grouped/rerun_2026-11-05/` (comparison tables, run log,
`pip_freeze.txt`, checkpoint hashes, the R5 files that differ).

**Every epoch, including the final one.** `last.ckpt` is not the final epoch: with Lightning 2.6 and
the project's `save_top_k: 2` it is a copy of the later of the two best-by-validation checkpoints.
`bash scripts/review/11_epoch_curve.sh` repeats R2-R5 with every epoch saved and tests each one:

| run | test acc, min - max over epochs | selected by validation | final epoch |
|---|---|---|---|
| R2 ResNet50, grp27a | 0.371 - 0.503 | 0.503 | 0.395 |
| R3 ResNet50, grp27b | 0.494 - 0.545 | 0.527 | 0.523 |
| R4 ResNet50, grp17 | 0.683 - 0.806 | 0.690 | 0.770 |
| R5 SLOW R50, grp27a | 0.393 - 0.521 | 0.521 | 0.393 |

The ResNet50 rows are the published runs epoch by epoch (identical validation curves). No epoch of
any run is near the 0.99 of the random split. On grp27a the selected epoch is the best one and the
fully trained models sit at the majority-class rate (0.39). On grp17 validation picks one of the
worst epochs, so quote the range there. Per-epoch tables:
`docs/review/data/grouped/rerun_2026-11-05/epochs/epoch_curve.md`.

## Published results

Best checkpoint by validation accuracy, then the test split of the same root
(`docs/review/data/grouped/eval/summary_*.json`):

| run | model | root | best epoch | val acc | test acc | test macro-F1 | test acc of `last.ckpt` (later top-2 epoch) |
|---|---|---|---|---|---|---|---|
| R1 | ResNet50, 1 frame | `data/ouc-cge-320` (random) | 6 | 0.9961 | 0.9935 | 0.9936 | 0.9948 |
| R2 | ResNet50, 1 frame | `data/grp27a` | 2 | 0.6005 | 0.5032 | 0.4592 | 0.4378 |
| R3 | ResNet50, 1 frame | `data/grp27b` | 1 | 0.6314 | 0.5266 | 0.4553 | 0.5451 |
| R4 | ResNet50, 1 frame | `data/grp17` | 5 | 0.4013 | 0.6903 | 0.6309 | 0.7591 |
| R5 | SLOW R50, 8x8 | `data/grp27a` | 1 | 0.5422 | 0.5088 | 0.4517 | 0.4500 |

Recipes: R1-R4 `+experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16`;
R5 `+experiment=baseline_ouc_cge trainer.max_epochs=15 trainer.early_stopping.patience=5
trainer.early_stopping.min_epochs=5 data.batch_size=32 data.num_workers=16
trainer.precision=bf16-mixed`. Seed 42 in both presets.

## Environment

- Python 3.12 and the project dependencies (`requirements.txt`, or the `Dockerfile`, which pins
  torch / torchvision / numpy to the vast.ai PyTorch image and installs the rest unpinned).
  The review scripts add nothing: `scipy` comes with scikit-learn, `dvc` is needed for the clips.
- **MLflow >= 3.17 refuses the file store** that the project's `mlflow.tracking_uri: null` resolves
  to and `MLFlowLogger()` raises. `08_run_grouped_training.sh` exports
  `MLFLOW_ALLOW_FILE_STORE=true`; do the same when calling `python -m src.train` directly.
- `01_frame_cache.py`, `03b_scene_clusters_thumb.py` and the training use a GPU when one is
  visible. On a machine whose GPU must stay idle, run the CPU steps with `CUDA_VISIBLE_DEVICES=""`.
- `run_05_06.sh` re-evaluates the stored random-split checkpoints; fetch them with
  `dvc pull outputs/baseline_ouc_cge.dvc outputs/resnet50_single_frame_best.dvc`. Before commit
  16bcec8 the root `.gitignore` listed `outputs/` and DVC refused these files as git-ignored.

| | published runs, 2026-10-26 | rerun, 2026-11-05 |
|---|---|---|
| GPU | RTX 5090 | RTX 5090, driver 595.91, CUDA 12.8, cuDNN 9.19 |
| Python | 3.12 | 3.12.15 |
| torch / torchvision | not recorded | 2.11.0+cu128 / 0.26.0+cu128 |
| lightning | not recorded | 2.6.6 |
| MLflow | 3.17.2 | 3.18.0 |
| OpenCV, NumPy, pandas, SciPy, scikit-learn | not recorded | 5.0.0, 2.5.4, 3.1.1, 1.18.1, 1.9.1 |
| full list | not recorded | `docs/review/data/grouped/rerun_2026-11-05/pip_freeze.txt` |

The package versions of the first host were not saved before that instance was destroyed.
The rerun reproduces the results with the versions in the right column.

## Scripts

| script | purpose |
|---|---|
| `reproduce.sh` | one entry point: data, splits (or rebuild), train, compare |
| `09_materialize_splits.py` | dataset roots from the published split files; no randomness |
| `08_run_grouped_training.sh` | runs R1-R5 through `python -m src.train`, exports the MLflow metrics, evaluates the best checkpoint and `last.ckpt` per clip |
| `10_compare_with_published.py` | rebuilt clusters and rerun metrics against `docs/review/data/` |
| `11_epoch_curve.sh` | R2-R5 with every epoch saved; test accuracy of each epoch, including the final one |
| `01_frame_cache.py` | first / centre / last frame of every clip: 32x32 thumbnails, optionally ImageNet ResNet50 features |
| `03b_scene_clusters_thumb.py` | thumbnail-distance scene clusters for a sweep of thresholds (uses helpers from `03_scene_clusters.py`) |
| `07_make_grouped_splits.py`, `run_07_splits.sh` | assign whole scene clusters to train / val / test; the driver runs the three published splits and compares |
| `05_eval_checkpoints.py` | evaluate a checkpoint through the project's `test_step`, per-clip predictions, temporal ablations |
| `02_near_duplicates.py`, `03_scene_clusters.py`, `04_group_holdout.py`, `06_report_tables.py`, `run_02_04.sh`, `run_03b_04.sh`, `run_05_06.sh` | the leakage probe of the published split (near-duplicate statistics, feature clusters, frozen-feature group hold-out, stored-checkpoint evaluation); not needed for the splits or the training |
| `common.py` | manifest parsing, frame-cache loading, dataset-root writer |
