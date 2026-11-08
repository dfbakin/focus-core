# GPU experiment plan: the leak gradient under three protocols

*Proposed 2026-11-08, awaiting approval. Host: vast.ai RTX 5090 (32 GB), 16 CPUs, 30 GB RAM, 140 GB disk, Python 3.12, CUDA 12.8.*

## 1. Question

How far does accuracy fall for the same recipes when the evaluation unit moves from a random clip (published split) to an
audio-proven moment (G1, event-disjoint) to a room-and-time unit that no measured channel links to training (G2, events ∪ scene
clusters)? The October scene-disjoint runs (grp27a/b, grp17) sit between G1 and G2 and stay as reference points.

## 2. Protocols

| id | grouping | folds | source |
|---|---|---|---|
| P0 | published split | 1 (control; R1 exists, 0.99) | `data/ouc-cge-320` |
| P1 | G1 events (45 groups) | 5, stratified group K-fold, seed 0 | `union_groups.csv`, `G1_event_fold` |
| P2 | G2 events ∪ scene clusters 27.5 (16 groups) | 5, stratified group K-fold, seed 0 | `union_groups.csv`, `G2_event_scene27_fold` |
| ref | October scene-disjoint grp27a, grp27b, grp17 | 1 each (R2-R5 exist) | `docs/review/data/grouped/splits/` |

Every clip is tested exactly once per protocol, so per-group results come free: each G2 group is a test group in one fold.

**Validation set.** Per fold, one whole group is carved from the four training folds (the group whose removal best preserves the
class mix of the remaining training set; never the test group), so validation is group-disjoint from both train and test. It is used
exactly as in the recipes, for checkpoint ranking and early stopping. Because the October runs showed validation can pick a bad
epoch (grp17), every run reports the val-selected checkpoint, the final epoch and the full per-epoch test curve.

**Label-conflict clips.** The 880 clips of events 9, 11 and 18 (`10` section 3) stay in training where the fold puts them; all test
metrics are reported twice, with and without them, from the saved per-clip predictions.

## 3. Recipes (unchanged from October, `scripts/review/08_run_grouped_training.sh`)

- R50: ResNet50, single frame, 10 epochs, batch 16, bf16; about 5.5 min per run on the 5090.
- SLOW: SLOW 8x8 R50, up to 15 epochs, early stopping patience 5 after epoch 5, batch 32; about 12 min per run.
- Probe: frozen ImageNet ResNet50 features (one extraction for all 7,701 clips), multinomial logistic regression per fold on CPU.

## 4. Runs and budget

| block | runs | GPU minutes |
|---|---|---|
| probe, all protocols | 1 extraction + CPU fits | 10 |
| R50 on P1 (seed 42) | 5 | 30 |
| R50 on P2 (seeds 42, 43, 44; variance across seeds is the error bar the 16-group design needs) | 15 | 85 |
| SLOW on P2 (seed 42) | 5 | 60 |
| total | 25 | about 3 h, plus 20 min setup |

Cheap blocks run first; SLOW last, so a budget cut costs only the temporal model.

## 5. Metrics and reporting

Per fold and pooled over folds: accuracy, balanced accuracy, macro-F1, Cohen's κ; majority-class baseline per fold (it differs by fold
under P2, fold 4 is 64% `high`); per-G2-group confusion matrices; group-level bootstrap interval (resample the 16 groups); seed spread
for R50 on P2. A CPU metadata-only tree (codec, resolution, audio tracks) under P1 and P2 tells whether the export-batch shortcut
survives the grouping. Results go to `docs/research/12_leak_gradient_results.md` with one figure: accuracy against cross-fold
nearest-neighbour distance (2.70, 17.5-27.6, 32.9, 40.9).

## 6. Code to add (CPU-tested locally before the host)

- `scripts/research/06_make_protocol_roots.py`: fold roots `data/p1_f{k}`, `data/p2_f{k}` in the project's format
  (`write_dataset_root` of `scripts/review/common.py`), with the validation-group rule above and a check that no group crosses
  train/val/test.
- `scripts/research/06b_run_protocols.sh`: loops the roots through the `run`/`post_run` logic of `08` (same evaluation of best and
  last checkpoints, same logs), with a `RUNS` filter and a budget switch.
- `scripts/research/06c_aggregate.py`: pooled metrics, exclusions, bootstrap, per-group tables, the figure.

## 7. Host workflow, no credentials on the rental

1. `git clone https://github.com/dfbakin/focus-core` (public, read-only) and check out `research/ouc-cge`.
2. `dvc pull data/ouc-cge-320.dvc outputs/research.dvc` from the read-only HTTPS remote `public` (4.4 GB + 177 MB).
3. Run the blocks under `nohup`; monitor from here over ssh.
4. Results (per-clip predictions, summaries, MLflow metrics, best + last checkpoints of the seed-42 runs) come back by `rsync` over
   the host's direct address (`$PUBLIC_IPADDR:$VAST_TCP_PORT_22`, about 7 MB/s against 1 MB/s through the proxy) and are added to
   DVC and pushed from this machine with the keys that live only here.
5. Code changes during the session: edit here, `rsync` to the host or commit, push and `git pull` there.

## 8. Decisions to approve

1. P1 and P2 as the two new protocols, G2 as the primary result.
2. Group-disjoint validation group per fold; report val-selected and final epoch.
3. Three seeds for R50 on P2 only; one seed elsewhere.
4. SLOW on P2 only.
5. Label-conflict clips excluded from test metrics in the headline numbers, included in a second column.
6. Keep best + last checkpoints for the seed-42 runs only; predictions for every run.
