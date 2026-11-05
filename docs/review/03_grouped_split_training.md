# Scene-disjoint retraining of the student's recipes (`07_make_grouped_splits.py`, `08_run_grouped_training.sh`)

Written by the host-side ML agent (RTX 5090 on vast.ai), 2026-10-26. Follows `01_leakage_analysis.md`, whose
scene clusters (experiment 3b) and frozen-feature group hold-out (experiment 4) this document turns into the
number the student's report is missing: the accuracy of the student's own training recipes when whole
classroom scenes are held out, next to the same recipe on the published random split.

## Summary

All five runs use the project's own entry point (`python -m src.train`) with the student's presets and the
reviewed working tree (see "Code version"); R1 is the control that reproduces the student's stored
ResNet50 run (10 epochs, batch 16, bf16) on the published split, R2-R5 are the same recipes on the
scene-disjoint splits of the next section. Test numbers are from `trainer.test(ckpt_path="best")` and were
re-computed clip by clip with `05_eval_checkpoints.py` (identical to 4 decimals).

| run | split | model | best val acc (epoch) | **test acc** | test macro-F1 | train acc (last epoch) | epochs | majority-class rate on test | wall |
|---|---|---|---|---|---|---|---|---|---|
| R1 | random (published) | ResNet50, 1 frame | 0.9961 (6) | **0.9935** | 0.9936 | 0.990 | 10 | 0.422 | 6.8 min |
| R2 | G27-A (scenes held out, thr 27.5) | ResNet50, 1 frame | 0.6005 (2) | **0.5032** | 0.4592 | 0.992 | 10 | 0.392 | 6.6 min |
| R3 | G27-B (different scenes, thr 27.5) | ResNet50, 1 frame | 0.6314 (1) | **0.5266** | 0.4553 | 0.991 | 10 | 0.406 | 7.8 min |
| R4 | G17 (label-pure groups, thr 17.5) | ResNet50, 1 frame | 0.4013 (5) | **0.6903** | 0.6309 | 0.990 | 10 | 0.394 | 8.1 min |
| R5 | G27-A | SLOW R50, 8x8 | 0.5422 (1) | **0.5088** | 0.4517 | 0.995 | 12 (early stop) | 0.387 | 18.8 min |

For reference: the frozen-ImageNet-feature linear probe of experiment 4 scores 0.915 (random 5-fold), 0.621
(thr 17.5 groups) and 0.472 (thr 27.5 groups); the student's stored checkpoints score 0.9935 (ResNet50) and
0.9987 (SLOW) on the published test split. Evaluated at every epoch (section "Other epochs instead of the
val-selected one", added 2026-11-05), the grouped runs score 0.37-0.50 (R2), 0.49-0.55 (R3), 0.68-0.81 (R4)
and 0.39-0.52 (R5, SLOW) on test; the final epochs score 0.395 / 0.523 / 0.770 / 0.393. No epoch of any
grouped run comes within 18 points of the random split, so the val-selected numbers above are not an
artefact of checkpoint selection. *Correction:* this paragraph used to quote 0.438 / 0.545 / 0.759 / 0.450
as the "final (`last.ckpt`)" checkpoints; `last.ckpt` is not the final epoch (see that section).

**Verdict.** Trained and selected exactly as the student did, the single-frame ResNet50 reaches 99.4% on the
published random split and 50-53% when whole classroom scenes are held out (69% under the loosest grouping
that merely refuses to train on near-identical frames of the test clips), i.e. 10-30 points above the
majority class and 30-49 points below the published number, with the "medium" class essentially never
recognised on an unseen scene (recall 0.11-0.27); the SLOW video model does no better than the single-frame
model on the same held-out scenes (50.9% vs 50.3%). The student's 99-100% is a measurement of scene
memorisation on a leaked split, and the honest cross-scene number for these recipes is about one half.

## Split construction (`07_make_grouped_splits.py`; `data/grouped/splits/`)

**Universe.** The 7701 usable clips of `data/scene_clusters.csv` (the 320p copy holds 7701 of the 7705
manifest rows; `videos/high/view{2521,2630,2631,2632}.mp4` failed transcoding and the unlabelled
`videos/anonymity/a.mp4` row is dropped by the manifest parser). The published random split restricted to
this universe is 6160 / 770 / 771 clips.

**Groups.** The thumbnail scene clusters of experiment 3b: connected components of the clip distance
d(i, j) = min over the 3x3 first/center/last frame pairs of the 32x32 RGB thumbnail RMSE (0-255 scale),
linked when d <= thr. Single linkage is used on purpose: it is the only linkage under which *no* pair with
d <= thr can straddle two groups, so a group-wise split is near-duplicate free by construction. Two
thresholds, as in experiment 4: **27.5** (101 clusters, the `scene_clusters.csv` choice: below the 1st
percentile of random cross-class distances, largest component 15% of the clips; label purity 0.79 because
transitive merging joins similar tables and because some sessions change label over time) and **17.5**
(530 clusters, every one of them 100% label-pure, i.e. "near-identical frames").

**Assignment.** Whole clusters go to train / val / test with clip targets 70 / 10 / 20 %. Greedy over clusters
in descending size order; each cluster goes to the split that minimises the sum over the nine (split, label)
cells of ((count - target) / target)^2 with target = frac[split] * N_label (the StratifiedGroupKFold idea
with unequal fold sizes). Because the clusters are few, large and label-skewed, the pure size-ordered greedy
is not the best assignment (at thr 27.5 it scores 0.222 with a smallest class share of 0.237), so the script
runs 400 restarts with the size order jittered by a seeded log-normal factor (sigma 0.6), keeps the
candidates in which every split has >= 10% of its clips in every class, and takes the best-scoring one.
Seed 0 and seed 1 differ in the jitter; for seed 1 (G27-B) candidates whose test clips overlap G27-A's test
clusters by more than 34% are rejected first, which is what makes G27-B a genuinely different scene hold-out
(the overlap of the chosen candidate is 2%, one 30-clip cluster). The assignment is deterministic given the
seed (the local dry run and the host run produced identical splits). The manifests are written in the
project's own format under a new dataset root per split (`data/grp27a`, `data/grp27b`, `data/grp17`, with
`videos` a symlink to `../ouc-cge-320/videos`), so `OUCCGEDataset` and `src.train` are used unchanged via
`data.root=...`; a new root also side-steps the stale filtered-manifest cache under
`data/ouc-cge-320/data_labels/.cache/`.

| split | thr | seed / restart | train (clusters) | val (clusters) | test (clusters) | train low/mid/high | val low/mid/high | test low/mid/high | smallest class share |
|---|---|---|---|---|---|---|---|---|---|
| random (published) | - | - | 6160 (96) | 770 (60) | 771 (63) | 2396/1653/2111 (.39/.27/.34) | 307/199/264 | 325/189/257 (.42/.25/.33) | 0.245 |
| **G27-A** | 27.5 | 0 / 396 | 5386 (83) | 771 (9) | 1544 (9) | 2117/1427/1842 (.39/.27/.34) | 306/201/264 | 605/413/526 (.39/.27/.34) | 0.261 |
| **G27-B** | 27.5 | 1 / 122 | 5187 (84) | 784 (7) | 1730 (10) | 2119/1403/1665 (.41/.27/.32) | 303/216/265 | 606/422/702 (.35/.24/.41) | 0.244 |
| **G17** | 17.5 | 0 / 1 | 5391 (437) | 770 (26) | 1540 (67) | 2119/1429/1843 (.39/.27/.34) | 303/204/263 | 606/408/526 (.39/.27/.34) | 0.265 |

In the published split 69 of the 101 clusters span two or more splits (7634 clips, 99.1%); in the grouped
splits no cluster appears in two splits (asserted by the script). 1219 of G27-A's, 1398 of G27-B's and 1215
of G17's test clips were *training* clips in the published split, which is simply what re-partitioning by
scene does.

**Leakage check** (cached 32x32 thumbnails; for every test clip the distance to its nearest train clip; the
random-split row reproduces experiment 2a):

| split | median | p5 | p25 | p75 | p95 | min | <= 10 | <= 17.5 | <= 27.5 | <= 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| random (published) | **2.7** | 0.0 | - | - | 15.3 | 0.0 | 84.0% | 96.0% | 98.7% | - |
| G27-A | **65.6** | 31.4 | 63.5 | 66.6 | 67.7 | 27.6 | 0 | 0 | 0 | 15% |
| G27-B | **38.9** | 33.2 | 36.4 | 42.2 | 47.4 | 27.6 | 0 | 0 | 0 | 14% |
| G17 | **31.3** | 21.6 | 26.6 | 35.9 | 41.3 | 17.5 | 0 | 0 | 30.1% | 70% |

For reference, random pairs of clips from different classes are at median RMSE 87 and their 1st percentile
is 35.0 (experiment 2a / 3b). Validation is scene-disjoint from train as well (nearest train clip for val
clips: median 39.4 / 36.5 / 27.5 for G27-A / G27-B / G17, minimum 27.6 / 27.6 / 18.4) and from test
(test->val median 74.0 / 58.0 / 40.9), so checkpoint selection on val is itself a cross-scene selection.

**Clips under the 64-frame SLOW window** (dropped by `OUCCGEDataset` for the SLOW recipe; ResNet50 uses a
1-frame window and drops nothing): train / val / test = 100 / 0 / 13 (G27-A), 113 / 0 / 0 (G27-B),
106 / 0 / 7 (G17); 98 / 10 / 9 in the published split (4 of the 98 are the missing files).

**Judgement calls, and why.**

1. *Thresholds.* 27.5 is the `scene_clusters.csv` choice of experiment 3b (no near duplicate below the 1st
   percentile of cross-class distances crosses a split) and is the strict, "whole scene" hold-out; 17.5 is
   the loosest grouping at which every cluster is still 100% label-pure, i.e. it only refuses to train on
   near-identical frames of a test clip and still lets 30% of the test clips have a same-scene train clip
   at d <= 27.5. The two bracket what "scene-disjoint" can reasonably mean on this data.
2. *G27-A's test set is 75% one cluster.* Cluster 6 (1163 clips, 554 low / 253 mid / 356 high, purity 0.48)
   is the largest component at 27.5, a transitive merge of several similar tables, and the best-balanced
   assignment puts it in test (it is the one cluster whose label mix matches the dataset's). The
   consequence is visible in the leakage table: G27-A's test clips are at median RMSE 65.6 from their nearest
   train clip, almost as far as unrelated clips (87), so G27-A is the hardest hold-out, "a classroom the
   model has never seen". I kept it because it is the output of the stated rule, not a hand pick, and
   because G27-B is its complement by construction: cluster 6 is in train there and the test set is ten other
   scene clusters (880 / 348 / 312 / 108 / 34 / 30 clips...), at median RMSE 38.9 from train.
3. *G17's test clusters are each single-label* (410 high, 218 low, 213 mid, 179 low, 105 mid, ...), as all
   17.5-clusters are. A model therefore cannot profit from within-cluster label mixing there, but it can
   profit from a same-session train clip at d in (17.5, 27.5] for 30% of the test clips.
4. *Restarts instead of the plain greedy.* The plain size-ordered greedy would have been defensible (smallest
   class share 0.237), but the restart winner is strictly better on the same objective and the choice rule
   (best score among candidates meeting the class-share floor and the overlap cap) is fixed in the script,
   so the splits are reproducible from the two cluster CSVs alone.
5. *Val is scene-disjoint too.* The alternative (val drawn from the train scenes) would make checkpoint
   selection and early stopping blind to over-fitting on scene identity; the student's protocol selects on a
   leaked val split, and the grouped runs should not repeat that.
6. *The SLOW filter.* G27-B has no test clip under 64 frames and G27-A has 13; so for the SLOW run on G27-A
   the ResNet50 and SLOW test sets differ by 13 clips (1544 vs 1531), the same caveat as in experiment 5.

## Runs (`08_run_grouped_training.sh`; logs in `data/grouped/logs/`, metrics in `data/grouped/mlflow/`, predictions in `data/grouped/eval/`)

**Code version.** Every run used the reviewed working tree: the local `src/` (reviewed `ouc_cge.py`,
`train.py`, `evaluate.py`, `training/callbacks.py`), `scripts/` and `configs/` were rsynced over the
student's committed code on the host before step 1 (`configs/` has no diff against `HEAD`; `git status` on
the host shows the four modified `src/` files). The wrapper records the version per run
(`code=reviewed-working-tree` in `data/grouped/runs.log`). No run had to fall back to the committed code.

**Finding: the first R1 attempt failed inside `MLFlowLogger(...)`, not in the reviewed code.** The host's
MLflow (3.17.2; the project pins `mlflow (>=3.0.0)`, the review's `poetry.lock` resolves 3.12.0) refuses
the filesystem tracking backend that the project's default `mlflow.tracking_uri: null` resolves to
(`file:./mlruns`). Traceback, verbatim from
`data/grouped/logs/train_R1_r50_random.FAILED_mlflow_filestore.log` (109 s after launch, i.e. after the
dataset scan, before the first batch):

```
Traceback (most recent call last):
  File "/workspace/focus-core/src/train.py", line 179, in main
    return train(cfg)
  File "/workspace/focus-core/src/train.py", line 104, in train
    mlflow_logger = MLFlowLogger(
  File "/venv/main/lib/python3.12/site-packages/lightning/pytorch/loggers/mlflow.py", line 153, in __init__
    self._mlflow_client = MlflowClient(tracking_uri)
  ...
  File "/venv/main/lib/python3.12/site-packages/mlflow/store/tracking/file_store.py", line 225, in __init__
    raise MlflowException(
mlflow.exceptions.MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode and will not receive further updates. Please migrate to a database backend (e.g., 'sqlite:///mlflow.db') to access the latest MLflow features. The `mlflow migrate-filestore` tool migrates your existing data losslessly. See https://mlflow.org/docs/latest/self-hosting/migrate-from-file-store for migration guidance. If the filesystem backend is required for your workflow, set `MLFLOW_ALLOW_FILE_STORE=true` to opt out of this exception.
```

The student's committed `train.py` builds the same `MLFlowLogger(tracking_uri=None)` (line 104 is the same
call in both versions), so `git stash` would not have helped and the run was not reverted; the wrapper
exports `MLFLOW_ALLOW_FILE_STORE=true` instead and all runs log to `/workspace/focus-core/mlruns`
(experiment `review_grouped`). For the student: with the dependency pin as committed, a fresh environment
gets an MLflow that raises on the project's default configuration; the pulled `mlruns.dvc` store is a file
store, so their own runs were made with an MLflow that still accepted it. Either pin MLflow below the
version that raises, set `mlflow.tracking_uri` to a database URI, or document the environment variable.

**Common settings.** `experiment.seed=42`, AdamW lr 1e-4 / wd 1e-4, cosine schedule over `max_epochs`,
gradient clipping 1.0, bf16-mixed, `ModelCheckpoint(monitor=val/accuracy, save_top_k=2)`, `deterministic:
false`; `trainer.test(ckpt_path="best")` after `fit`. Train transforms: resize 256, random crop 224,
horizontal flip; eval: resize 256, center crop 224, center window. Overrides that do not touch the recipe:
`paths.output_dir=outputs/review/<run>`, `mlflow.experiment_name=review_grouped`,
`hydra.run.dir=outputs/review/<run>/hydra`. Dataset filtering ran fresh for every new root (the
`.cache/<split>_w<window>.csv` files were created by these runs); the "Skipping train sample" warning,
which the review turned into an error for val/test, appears **0 times** in all five training logs. No
duplicate paths in any evaluation (`duplicate_paths_in_eval: 0`). Total wall time of the sequence 51 min
(3072 s), so the budget rule never triggered: R3 was run and R5 got its 15-epoch budget.

### R1, control: ResNet50 on the published split

```
python -m src.train +experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16 data.root=data/ouc-cge-320
```

Train / val / test = 6160 / 770 / 771 clips (window 1, nothing dropped). val/accuracy per epoch: 0.897,
0.966, 0.949, 0.964, 0.987, 0.987, **0.996**, 0.994, 0.995, 0.995; train/accuracy 0.780 -> 0.990. Best
checkpoint epoch 6 (`epoch=06-val_acc=0.9961.ckpt`), the same best value and epoch as the student's stored
checkpoint (`accuracy=0.9961.ckpt`, epoch 6). Test **0.9935**, macro-F1 0.9936, 5 errors, mean max-softmax
0.996. Confusion (rows = true low / mid / high, columns = predicted):

| | low | mid | high |
|---|---|---|---|
| low (325) | 321 | 1 | 3 |
| mid (189) | 0 | 189 | 0 |
| high (257) | 0 | 1 | 256 |

The control reproduces the student's run to the fourth decimal; the pipeline on the host is the student's.

### R2: ResNet50 on G27-A (whole scenes held out, thr 27.5)

```
python -m src.train +experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16 data.root=data/grp27a
```

Train / val / test = 5386 / 771 / 1544. val/accuracy per epoch: 0.327, 0.416, **0.601**, 0.450, 0.350,
0.521, 0.380, 0.471, 0.420, 0.463 (val/loss 2.41 -> 2.16 at epoch 2 -> 3.0-3.7 afterwards); train/accuracy
0.811 -> 0.992. Best checkpoint epoch 2. Test **0.5032**, macro-F1 0.4592, 767 errors, mean max-softmax
0.836 (the model is confident on scenes it has never seen). Majority class (low) 0.392.

| | low | mid | high |
|---|---|---|---|
| low (605) | 389 | 13 | 203 |
| mid (413) | 144 | 74 | 195 |
| high (526) | 199 | 13 | 314 |

Per-class recall 0.64 / 0.18 / 0.60; "mid" is predicted for only 100 of 1544 clips. Per test cluster:
the giant cluster 6 (1163 clips, 554/253/356) 0.437; cluster 5 (214, mid/high) 0.692; cluster 29 (76 high)
0.947; cluster 22 (49 low) 0.796; cluster 38 (30 mid) **0.000** (all predicted low).

### R3: ResNet50 on G27-B (a different set of scenes held out, thr 27.5)

```
python -m src.train +experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16 data.root=data/grp27b
```

Train / val / test = 5187 / 784 / 1730. val/accuracy per epoch: 0.522, **0.631**, 0.579, 0.619, 0.607,
0.461, 0.468, 0.494, 0.468, 0.453; train/accuracy 0.781 -> 0.991. Best checkpoint epoch 1. Test
**0.5266**, macro-F1 0.4553, 819 errors, mean max-softmax 0.867. Majority class (high) 0.406.

| | low | mid | high |
|---|---|---|---|
| low (606) | 534 | 50 | 22 |
| mid (422) | 367 | 45 | 10 |
| high (702) | 325 | 45 | 332 |

The model predicts "low" for 1226 of 1730 clips (recall 0.88 / 0.11 / 0.47). Per test cluster: cluster 0
(880 clips, 178 mid / 702 high, the merged mid/high scene) 0.385; cluster 28 (348, 276 low / 72 mid) 0.658;
cluster 36 (312 low) 0.936; cluster 27 (108 mid) **0.009**; cluster 39 (34 mid) 1.000; cluster 38 (30 mid)
0.000 again. Whether a held-out scene is recognised is a property of the scene, not of the class: two
"mid" scenes of similar size score 100% and 1%.

### R4: ResNet50 on G17 (label-pure near-identical-frame groups held out, thr 17.5)

```
python -m src.train +experiment=resnet50_ouc_cge_baseline trainer.max_epochs=10 data.batch_size=16 data.root=data/grp17
```

Train / val / test = 5391 / 770 / 1540. val/accuracy per epoch: 0.260, 0.169, 0.279, 0.348, 0.338,
**0.401**, 0.366, 0.384, 0.390, 0.379 (val/loss 4.0-5.2 throughout); train/accuracy 0.798 -> 0.990. Best
checkpoint epoch 5. Test **0.6903**, macro-F1 0.6309, 477 errors, mean max-softmax 0.935. Majority class
(low) 0.394.

| | low | mid | high |
|---|---|---|---|
| low (606) | 556 | 9 | 41 |
| mid (408) | 120 | 108 | 180 |
| high (526) | 103 | 24 | 399 |

Unexpected: the val split of this assignment (26 clusters, 303/204/263) is *harder* than its test split:
val never exceeds 0.40 (its confusion is 273/22/8, 168/4/32, 184/47/32, i.e. nearly everything is called
"low") while the selected checkpoint scores 0.69 on test. The 26 val scenes happen to be ones whose mid and
high clips look like train's low clips; the 67 test clusters include large low clusters that transfer well
(218 low: 0.92; 179 low: 0.97) and one 410-clip high cluster at 0.69, while the two large mid clusters are at
0.17 and 0.06. This is what "accuracy on a handful of scenes" looks like: it depends on which scenes.

### R5: SLOW R50 on G27-A

```
python -m src.train +experiment=baseline_ouc_cge trainer.early_stopping.patience=5 trainer.early_stopping.min_epochs=5 data.batch_size=32 data.num_workers=16 trainer.precision=bf16-mixed trainer.max_epochs=15 data.root=data/grp27a
```

Train / val / test = 5286 / 771 / 1531 after the 64-frame filter (100 train and 13 test clips dropped).
Pretrained Kinetics SLOW_8x8_R50 from the PyTorchVideo hub, as in the student's run. val/accuracy per epoch:
0.380, **0.542**, 0.354, 0.353, 0.402, 0.370, 0.445, 0.435, 0.366, 0.377, 0.372, 0.366; train/accuracy 0.805
-> 0.995 (0.947 already at epoch 1). Early stopping fired after epoch 11 (12 epochs run; best epoch 1,
patience 5 counted from epoch 5). Test **0.5088**, macro-F1 0.4517, 752 errors, mean max-softmax 0.839.
Majority class (low) 0.387. 93 s per epoch on this host (the 4 min/epoch estimate was pessimistic).

| | low | mid | high |
|---|---|---|---|
| low (592) | 478 | 7 | 107 |
| mid (413) | 197 | 72 | 144 |
| high (526) | 272 | 25 | 229 |

On the same held-out scenes the 3D model (0.509) is within one point of the 2D single-frame model (0.503)
with the same per-cluster pattern (cluster 6: 0.456 vs 0.437; cluster 38: 0.000 in both). The student's
random-split SLOW run reached val 0.94 after epoch 0 and 1.00 later; here val peaks at 0.54 after epoch 1
and declines while train accuracy goes to 0.995: the extra capacity is spent memorising the training scenes.

### Other epochs instead of the val-selected one

**Correction (2026-11-05).** The first version of this section evaluated `last.ckpt` and called it the final
checkpoint (epoch 9, or 11 for R5). It is not. With Lightning 2.6, `save_last=True` writes `last.ckpt` only
when a top-k checkpoint is saved ("Only save last checkpoint if a checkpoint was actually saved in this
step", `model_checkpoint.py`), so with the project's `save_top_k: 2` it is a byte copy of the later of the two
best-by-validation checkpoints. The downloaded files of the rerun confirm it (same sha256 as the second
top-2 file; `epoch` field 8 / 5 / 3 / 8 for R1-R4), and the logged validation curves give the same epochs for
the published runs. The final-epoch weights were never written. The numbers of the old table are correct;
their epochs are:

| run | split | val-selected checkpoint (epoch): test acc / F1 | `last.ckpt` = later top-2 checkpoint (epoch): test acc / F1 | its confusion (rows true low/mid/high) |
|---|---|---|---|---|
| R1 | random | 0.9935 / 0.9936 (6) | 0.9948 / 0.9944 (8) | [324 0 1] [1 187 1] [0 1 256] |
| R2 | G27-A | 0.5032 / 0.4592 (2) | 0.4378 / 0.3945 (5) | [173 1 431] [85 73 255] [92 4 430] |
| R3 | G27-B | 0.5266 / 0.4553 (1) | 0.5451 / 0.4685 (3) | [313 117 176] [197 58 167] [117 13 572] |
| R4 | G17 | 0.6903 / 0.6309 (5) | 0.7591 / 0.7342 (8) | [546 9 51] [87 194 127] [79 18 429] |
| R5 | G27-A, SLOW | 0.5088 / 0.4517 (1) | 0.4500 / 0.4056 (6) | [109 100 383] [63 125 225] [37 34 455] |

To get the final epoch and every other one, `scripts/review/11_epoch_curve.sh` repeated R2-R5 on the second
host with `trainer.checkpoint.save_top_k=-1` (nothing else changed) and evaluated each epoch on test. The
three ResNet50 runs are bit-deterministic: their validation curves equal the published ones epoch by epoch,
and epochs 2 / 5 (R2), 1 / 3 (R3), 5 / 8 (R4) reproduce the numbers above, so the table below is the
per-epoch test accuracy of the published runs. SLOW does not repeat bit for bit; its row is a third
trajectory of the same recipe (selected epoch 1: test 0.509 published, 0.518 rerun, 0.521 here).

| run | split | test acc by epoch (val-selected epoch in bold) | min - max | val-selected | final epoch |
|---|---|---|---|---|---|
| R2 | G27-A | .385 .382 **.503** .410 .401 .438 .371 .401 .376 .395 | 0.371 - 0.503 | 0.503 | 0.395 |
| R3 | G27-B | .531 **.527** .532 .545 .538 .494 .540 .527 .529 .523 | 0.494 - 0.545 | 0.527 | 0.523 |
| R4 | G17 | .806 .749 .723 .732 .683 **.690** .749 .767 .759 .770 | 0.683 - 0.806 | 0.690 | 0.770 |
| R5 | G27-A, SLOW | .428 **.521** .429 .416 .435 .415 .415 .393 .403 .400 .407 .393 | 0.393 - 0.521 | 0.521 | 0.393 |

What this shows:

- No epoch of any grouped run is near the random split: the best single epoch is 0.81 on the loose grouping
  and 0.55 on the strict ones, against 0.99.
- On G27-A the val-selected epoch is the best of the run for both models. Every other epoch is between 0.37
  and 0.44, and the fully trained models (0.395 ResNet50, 0.393 SLOW) sit at the majority-class rate of that
  test set (0.392 / 0.387). The summary-table numbers for G27-A are therefore the favourable end.
- On G17 it is the other way round: validation accuracy barely reaches the majority rate (0.17-0.40 against
  0.39, 26 scenes) and selects one of the two worst test epochs; the final epoch scores 0.770 and the first
  epoch 0.806. A scene-disjoint validation set this small does not track test, so on G17 the honest
  statement is the range 0.68-0.81, not the single 0.690.
- G27-B is flat (0.49-0.55) whatever the stopping point.

Picking an epoch by its test score would be test-set selection, so the summary table keeps the val-selected
numbers, which is the student's protocol; the ranges above are the uncertainty that protocol hides. Files:
`docs/review/data/grouped/rerun_2026-11-05/epochs/` (`epoch_curve.md`, `epoch_curve.csv`, per-epoch summaries).

### What is consistent across the grouped runs

- Train accuracy reaches 0.99 in every run within 10 epochs, exactly as on the random split; the models fit
  the training scenes equally well either way. The difference is entirely in what they generalise to.
- Val accuracy under a scene-disjoint val split peaks at epoch 1-5 and then drifts down with rising val
  loss; checkpoint selection picks an early, under-trained model because later ones are worse on unseen
  scenes, not better.
- "Mid" is never recognised cross-scene (recall 0.18, 0.11, 0.27, 0.17); "low" transfers best (0.64, 0.88,
  0.92, 0.81). The scripted "low" condition (a slideshow) plausibly has a distinctive scene appearance, which
  is a scene cue, not an engagement cue.
- Mean max-softmax 0.84-0.94 on unseen scenes against 0.996 on the random split: the models are
  over-confident on scenes they have never seen, which matters for the app that thresholds these outputs.

## Threats to validity

1. **Few scenes.** At thr 27.5 the test sets are 9-10 clusters and G27-A's is 75% one cluster; the
   number is in effect an average over a handful of scenes and a different hold-out can move it by several
   points. The two independent 27.5 hold-outs agree (50.3 vs 52.7) and the 17.5 split with 67 test clusters
   gives 69.0, which together with the full threshold sweep of experiment 4 (0.78 at thr 10 down to 0.40 at
   thr 35 for the linear probe) bounds the honest number between roughly 0.5 and 0.7 depending on how
   strictly "unseen scene" is defined. None of these is near 0.99.
2. **Label imbalance across scenes.** Labels are a property of the session (every 17.5-cluster is
   label-pure), so holding out scenes also holds out label-scene pairs; a held-out single-label scene scores
   either ~100% or ~0% (clusters 36 / 39 vs 27 / 38). Accuracy is therefore partly "did the test scenes
   happen to resemble a train scene of the same label", which is what cross-scene engagement recognition
   on this dataset *is*; macro-F1 (0.45-0.63) is the fairer summary and is reported throughout. Majority
   rates differ per split (0.39-0.42) and are listed.
3. **Val-based model selection on few scenes.** Val is 7-26 held-out clusters, so the selected epoch is
   noisy (R4's val maximum is 0.40 for a test score of 0.69). The last-checkpoint comparison above shows
   how much that choice matters; it does not change the conclusion.
4. **Linear probe vs fine-tuning.** Fine-tuning the whole network gains 3-7 points over the frozen
   ImageNet linear probe of experiment 4 on comparable groupings (0.50-0.53 vs 0.47 at thr 27.5; 0.69 vs
   0.62 at thr 17.5), so the probe's numbers were a fair lower bound and the fine-tuned numbers are what the
   student's recipe actually achieves.
5. **Epoch budget.** 10 epochs for ResNet50 is the student's own stored recipe; 15 with early stopping
   for SLOW is shorter than the student's 50-epoch config, but val peaked at epoch 1 and train accuracy was
   0.995 by epoch 11, so more epochs of the same recipe cannot improve the cross-scene number. The runs
   measure the recipe as published, not the best achievable cross-scene model (stronger augmentation,
   lower learning rate, frozen backbone or pose features might do better and are the student's next
   experiment, not this review's).
6. **One seed, non-deterministic training.** `deterministic: false`, bf16 and cuDNN autotuning; the random
   control nevertheless reproduces the stored checkpoint's best val (0.9961 at epoch 6) exactly, so the
   pipeline is the same one. Run-to-run noise on the grouped splits is unknown but is dwarfed by the 40-50
   point gap.
7. **Scene groups are visual, not administrative.** No session or participant metadata was released, so
   "scene" is a thumbnail-similarity component: at 27.5 it may merge distinct tables transitively (cluster 6)
   and at 17.5 a session can be split into several groups, so 30% of G17's test clips still have a
   same-scene train clip at d <= 27.5. The strict and the loose grouping bracket the truth.
8. **Evaluation sets differ slightly between models.** SLOW evaluates 1531 of G27-A's 1544 test clips
   (64-frame filter); ResNet50 evaluates all 1544. Immaterial at this gap.

## Files produced

All under `docs/review/data/grouped/` unless noted; scripts under `scripts/review/`.
How to reproduce the splits and the runs, and the result of doing so on a second, clean host:
`scripts/review/README.md`.

- `07_make_grouped_splits.py` (script), `08_run_grouped_training.sh` (script; also contains the MLflow
  metric export and the per-clip evaluation calls)
- `splits/split_{grp27a,grp27b,grp17}.csv` — path, label, split, cluster for all 7701 clips;
  `splits/split_*_summary.json` — sizes, label counts, clusters, leakage statistics (grouped and random),
  under-64-frame counts
- `eval/preds_R{1..5}_*_{test,val}.csv` — per-clip label, prediction, softmax for the best checkpoint;
  `eval/preds_R*_last_test.csv` — same for `last.ckpt` (the later top-2 checkpoint, not the final epoch); `eval/summary_*.json` — accuracy, macro-F1,
  confusion matrix, evaluated / dropped counts
- `mlflow/R*/{metrics,params,tags}/`, `mlflow/R*/summary.json` — the MLflow run files (per-epoch
  `val/accuracy`, `val/loss`, `train/accuracy`, `test/*`, `best_epoch`, `final_epoch`, checkpoint paths)
- `configs/R*/config.yaml`, `configs/R*/hydra/.hydra/overrides.yaml` — resolved config and overrides per run
- `logs/07_splits.log`, `logs/08_grouped.log`, `logs/train_R*.log`, `logs/eval_R*.log`,
  `logs/last_eval.log`, `logs/train_R1_r50_random.FAILED_mlflow_filestore.log`
- `runs.log` (wrapper record: commands, wall times, code version, best checkpoints, metrics),
  `runs_attempt1_failed.log`
- Host only (that instance no longer exists): dataset roots
  `/workspace/focus-core/data/{grp27a,grp27b,grp17}` (manifests + `videos` symlink; rebuilt byte for
  byte by `scripts/review/09_materialize_splits.py`), checkpoints
  `/workspace/focus-core/outputs/review/R*/checkpoints/` (4.3 GB; not kept. The checkpoints of the
  2026-11-05 rerun are in DVC, `dvc pull outputs/review.dvc`, see `scripts/review/README.md`),
  `/workspace/focus-core/mlruns` (experiment `review_grouped`),
  `/workspace/review_grouped/`
