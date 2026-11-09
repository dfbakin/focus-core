# Leak-gradient experiment: interpretation

*2026-11-08. Hand-written companion to the generated tables in `12_leak_gradient_results.md` (metrics, bootstrap, epoch curves, figure) and `12b_scene_collapse.md` (per held-out group). Plan: `11_gpu_experiment_plan.md`; protocols: `10_reconstruction_qc_and_protocol.md`; what leaks: `07_leak_statement.md`. All 39 planned runs completed (25 training runs in 211 min on one RTX 5090, code 6ecdcda; the frozen probe in 2 min).*

## 1. Headline

Under both leak-controlled protocols every recipe is near chance. Balanced accuracy of the validation-selected checkpoint, label-conflict clips excluded, with the 95% group bootstrap where the protocol has folds:

| protocol (test unit held out) | share of test clips with a same-scene training neighbour (d ≤ 27.5) | ResNet50 single frame | SLOW 8x8 | frozen ResNet50 + logistic regression |
|---|---|---|---|---|
| P0 published random clip split | 98.7% | 0.995 | — | 0.921 |
| grp17 scene clusters, loose (Oct.) | 30.1% | 0.690 | — | 0.662 |
| grp27b scene clusters, strict (Oct.) | 0% | 0.496 | — | 0.507 |
| grp27a scene clusters, strict (Oct.) | 0% | 0.473 | 0.477 | 0.414 |
| P1 events (45 groups, 5 folds) | 15.7% | 0.373 [0.27, 0.47] | — | 0.419 [0.32, 0.52] |
| P2 events ∪ scenes (16 groups, 5 folds) | 0% | 0.405 / 0.397 / 0.388 (seeds 42/43/44) [0.36, 0.52] | 0.420 [0.37, 0.56] | 0.397 [0.35, 0.51] |

Chance is 0.333; a constant predictor reaches 0.403 raw accuracy on the conflict-free set. Cohen's κ is 0.09 (P1) and 0.10 to 0.14 (P2) for the fine-tuned models, 0.15 and 0.12 for the probe; no bootstrap upper bound exceeds 0.31. Final-epoch checkpoints are lower still (P1 0.349; P2 0.366 to 0.381), so about 0.03 of the headline comes from checkpoint selection on one or two validation groups.

The published 99.5% is therefore footage memorisation and nothing else: a frozen ImageNet backbone with a linear head already scores 0.92 on the published split without any training of the backbone, and the same models fall to chance once held-out footage, scenes and moments are all disjoint.

## 2. Reading the gradient

The figure plots these points against the median thumbnail distance from a test clip to its nearest training clip. Accuracy falls monotonically from the published split through grp17 and grp27b to P2 for the fine-tuned ResNet50; grp27a, with the largest median, sits at 0.47 because its single October test set happened to be class-friendly (the probe gives 0.41 there, and no bootstrap exists for a single split).

The median is a crude leak measure. P1 and grp17 have almost the same median (32.9 vs 31.3) but differ by 0.3 in balanced accuracy, because the tail differs: 30.1% of grp17 test clips have a same-scene training neighbour (d ≤ 27.5) against 15.7% under P1 and 0% under P2, grp27a and grp27b. Everything above chance in this dataset tracks that tail.

## 3. Why the folds are uneven and the seeds disagree

### 3.1 The label is a property of the recording, and class is confounded with room

| unit | count | single-class | multi-class |
|---|---|---|---|
| camera recordings | 138 | 138 | 0 |
| events (same moment, all cameras) | 45 | 42 | 3 (the label-conflict events 9, 11, 18) |
| G2 groups (rooms) | 16 | 10 | 6 (5 once the conflict events are removed: 9, 10, 11, 12, 13) |

Every clip of a recording carries the same label, and the cameras of an event agree except in the three conflict events. The release therefore contains about 45 independently labelled moments, not 7,701 labelled clips, and in 10 of the 16 rooms the label never varies. A model trained with one room held out can learn "this room looks like a low room"; it has nothing else to learn from those ten rooms.

### 3.2 Held-out rooms are assigned wholesale to one class

`12b_scene_collapse.md` gives, for every held-out G2 group and run, the predicted class mix. Selected rows (true mix low/mid/high; cell = modal predicted class and its share, then the group's accuracy):

| group | n | true mix | R50 s42 | R50 s43 | R50 s44 | SLOW s42 | probe |
|---|---|---|---|---|---|---|---|
| 11 (mixed) | 1,163 | 0.48/0.22/0.31 | high 0.95, acc 0.33 | high 0.90, 0.30 | high 0.88, 0.29 | high 0.91, 0.35 | low 0.48, 0.46 |
| 13 (mixed) | 745 | 0.31/0.51/0.18 | low 0.70, 0.42 | low 0.48, 0.37 | low 0.56, 0.37 | low 0.43, 0.40 | high 0.42, 0.38 |
| 8 (all low) | 529 | 1/0/0 | mid 0.44, 0.36 | mid 0.47, 0.13 | mid 0.51, 0.26 | mid 0.43, 0.19 | high 0.44, 0.25 |
| 6 (all low) | 49 | 1/0/0 | high 0.57, 0.00 | mid 0.98, 0.00 | mid 0.67, 0.00 | high 0.61, 0.39 | low 0.94, 0.94 |
| 2 (all high) | 308 | 0/0/1 | high 0.91, 0.91 | high 0.91, 0.91 | high 0.87, 0.87 | high 0.95, 0.95 | high 0.85, 0.85 |
| 14 (all mid) | 264 | 0/1/0 | low 0.45, 0.36 | low 0.54, 0.31 | high 0.48, 0.34 | mid 0.49, 0.49 | high 0.64, 0.17 |

Room 11 is the clearest case: its labels are spread over all three classes, yet every fine-tuned model puts 88% to 95% of its clips into `high`. Room 2 is "recognised" as high by every run and room 8 is missed by every run. Rooms 1, 3, 5 and 14 change modal class from seed to seed. Over the 16 rooms the fine-tuned ResNet50 gets the room-level majority right 7, 6 and 8 times (seeds 42, 43, 44), SLOW 10 times, the probe 11 times; the mean total-variation distance between predicted and true class mix is 0.42 to 0.49 for the ResNet50 seeds, 0.37 for SLOW and 0.33 for the probe.

### 3.3 Consequences

- **Uneven folds.** A P2 test fold holds two to five rooms, so its accuracy is the sum of a few near-binary room-level outcomes. Fold 2 is high under every run because rooms 2 and 12 land right; fold 0 is low because room 8 (529 clips) lands wrong.
- **Seed variance is structural.** A seed changes which class a whole room is mapped to, and with it hundreds of clips at once. It would shrink only if the models converged on scene-independent cues, and the training data offers about 45 moments in 16 rooms with class confounded with room to learn them from. More epochs, a bigger backbone or more seeds do not change that; the variance measures the dataset, not the optimiser.
- **Accuracy below the fold majority.** Holding out class-confounded rooms shifts the class prior: folds 0 and 1 test on about 60% `low` while training on 35% to 37% `low`; fold 4 tests on 64% `high` while training on 18% `high`. Models calibrated to the training prior under-predict the test-dominant class. This is why raw accuracy is below the majority rate on four of five P2 folds, and why balanced accuracy and κ are the headline metrics.
- **The right unit of inference is the room (P2) or the event (P1).** The per-clip n of 7,701 overstates the evidence by two orders of magnitude; the group bootstrap intervals in `12_leak_gradient_results.md` §5 are the honest ones.

## 4. Fine-tuning, the video model, epochs and validation

- **The frozen probe matches or beats fine-tuning under P1 and P2** (0.419 vs 0.373; 0.397 vs 0.388 to 0.405) and loses only where footage is shared (published 0.921 vs 0.995, grp17 0.662 vs 0.690). Fine-tuning on this data learns room identity, which does not transfer; frozen features with a linear head cannot overfit the rooms as hard and lose nothing of substance. The probe is also the best room-level decider (11 of 16).
- **SLOW 8x8 (64 frames) is within the ResNet50 seed spread** (0.420 vs 0.388 to 0.405): the video model extracts no additional scene-independent signal. Early stopping (patience 5 after epoch 5) fired at epochs 12 to 14, i.e. almost never; the test curves wander between 0.39 and 0.43 pooled with no trend. 113 clips shorter than 64 frames drop out of its sets (n 7,588).
- **Epoch curves show nothing to converge to.** P1 pooled balanced accuracy peaks at epoch 1 (0.40) and declines to 0.37; P2 seed 42 wanders between 0.34 and 0.41 across epochs 0 to 9.
- **Validation selection is noisy by construction.** Under P2 folds 1 to 4 validate on room 13 alone (the only three-class group that fits the 15% cap once the conflict groups are excluded) and fold 0 on rooms 12 and 2; the fold-level validation balanced accuracy of the probe ranges from 0.20 to 0.51 and tracks the test result poorly. The final-epoch rows are reported alongside for that reason.

## 5. Conflict clips and the metadata shortcut

- Excluding the 880 label-conflict clips lowers P2 balanced accuracy by 0.03 to 0.04 (ResNet50 seed 42: 0.441 → 0.405; SLOW 0.445 → 0.420) and leaves P1 unchanged in substance. The conflict rooms 9, 10 and 15 carry labels that disagree between cameras of the same moment, so agreement with them is partly agreement with a camera choice; the exclusion stands.
- The container-metadata tree (`06d_metadata_tree.py`, depth 4 on codec, resolution, duration bin, audio tracks) reaches 0.476 balanced accuracy under P0, 0.430 pooled under P1 and 0.314 under P2. The export-batch shortcut survives event-disjoint folds and vanishes under P2, which is one more reason P2 is the protocol of record.

## 6. What the dataset can and cannot support

1. **Not supported:** training or evaluating scene-independent recognition of three-level group engagement at clip granularity. The label is a session-level annotation broadcast to every clip of a recording; there are about 45 labelled moments in 16 rooms; class is confounded with room in 10 of them. Under any protocol that holds out rooms, every model tested behaves as a room classifier scored against a room-confounded label, and no recipe exceeds κ 0.17.
2. **Supported as a methodological result:** the leak characterisation (`07_leak_statement.md`), the label-by-camera finding (`10_...` §label by camera; PKMS permanent note), and the room-collapse evidence here. Together they show how a 99.5% benchmark result arises from footage sharing and why the gradient of ever-stricter splits ends at chance.
3. **Weaker questions that remain answerable on this release:** (a) within-room, event-held-out classification in the five mixed rooms, i.e. whether sessions with different labels in a known room are separable from footage (about 33 events; needs a strict near-duplicate check across sessions of the same room); (b) event-level classification with clip aggregation and leave-one-event-out, treating the release as a 45-sample problem and reporting the small-sample uncertainty as such.
4. **A replacement dataset for the recognition task needs:** labels that vary within a scene over time (segment-level annotation), several sessions per room with labels not confounded with the room, verified agreement between synchronised cameras, and a published or reconstructable scene-disjoint split. The survey in `09_alternative_datasets.md` found no public room-camera dataset that meets all four; DIPSER is the nearest.

## 7. Artefacts and caveats

- Commands and timings: `outputs/research_gpu/_summary/runs.log`; per-run Hydra configs: `outputs/research_gpu/<run>/hydra/`; per-clip predictions: `outputs/research_gpu/_summary/eval/preds_*.csv` (261 files); MLflow file store: `outputs/research_gpu/_summary/mlflow/`; result tables: `outputs/research/protocols/results/*.csv`; probe features: `outputs/research_gpu/_summary/probe_features.npz`. Seed-42 checkpoints (best-by-validation and final epoch of every P1/P2 run, 15 runs) are kept under `outputs/research_gpu/<run>/checkpoints/`; seeds 43 and 44 kept predictions only. Everything under `outputs/research_gpu` is DVC-tracked, not in git.
- Caveats: one seed for P1 and SLOW; validation sets of one or two whole groups; the probe uses a single centre frame; the thumbnail nearest-neighbour distance is a proxy for scene similarity, not for identity of people or moment; the October reference points are single splits without bootstrap.
- Reproduce: `python scripts/research/06_make_protocol_roots.py`, then `RUNS="probe r50_p1 r50_p2 slow_p2" MLFLOW_ALLOW_FILE_STORE=true bash scripts/research/06b_run_protocols.sh` on a GPU host, then `python scripts/research/06c_aggregate.py` and `python scripts/research/06f_scene_collapse.py`.
