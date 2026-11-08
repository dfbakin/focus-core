# Quality review of the session reconstruction, and the evaluation protocol that follows from it

*Supervisor review (2026-11-08) of `08_session_reconstruction.md`. Two checks were added: `scripts/research/05g_crossclass_hires.py`
(full-resolution frames of the cross-class synchronous events) and `05h_union_groups.py` (events united with the October scene
clusters). Figures with faces stay out of git.*

## 1. Verdict on the reconstruction

The reconstruction is trustworthy where it matters for evaluation, and it is honest about where it is not.

| layer | evidence | coverage | confidence | my reading |
|---|---|---|---|---|
| Tier 0 exact recordings | byte-identical encoded packets, one time axis per unit, residual 1.8e-13 s | 84.2% of clips | certain | no fallback involved |
| Tier 0 probable recordings | frame-junction RMSE at 128x72 under the 95th percentile of 6,478 packet-proven junctions, plus batch, run, column, direction and scene agreement | 15.6% | high; 350 of 355 merges are in the HEVC and 1064x598 batches, where no packets exist, so no stronger evidence is available | the only alternative would be to leave those batches as single clips; the merge median distance 4.0 is below the proven-junction median 11.9 |
| Tier 0 doubtful merges | the 5 merges between two packet units (RMSE 2.5 to 37.2) | 109 clips in the two worst | low (0.06 and 0.10) | both join the same room, class and camera; a wrong merge misplaces the time axis but changes no grouping |
| Tier 1 audio | masked envelope correlation, threshold at the 99.9th percentile of two nulls that cannot be simultaneous, split-half persistence | 74.1% of clips | 0 of 77,294 null pairs pass; fewer than 0.34 expected false links among 8,646 pairs | strongest layer; the null design is correct after the agent dropped the briefed cross-class null (see section 3) |
| Tier 1 footage identity | shared packets (161 clips) or near-identical 32x32 frames (189 clips) | 4.5% | no null exists for the frame rule; all 21 frame links show the same viewpoint | over-linking only; harmless for folds |
| Tier 1 prior fallback | ID adjacency within codec, size and layout | 3 clips (0.04%) | low | negligible |
| Tier 1 structural fallback | interleaving period | 0 clips | not used | the 51 silent column pairs were left unlinked, which is the conservative choice |
| Tier 1 unassigned | 14 recordings alone | 21.3% | these recordings all have 30 s or more of audio, so a partner in the release would have been found | genuine single-camera stretches in the release |
| Tier 2 session candidates | background and clothing colour | 83.3% | not estimable | candidates only, as briefed |

**Memory.** All processes together stayed under 1.3 GB, inside the 2.5 GB budget.

**The asymmetry that governs the review.** For leak-free evaluation an over-link is harmless, because it only merges two groups
that will then be held out together. A missed link is the leak. The doubtful list in `08` section 6.7 is therefore a list of
threats to the time axis, not to the fold design. The residual risk to the folds is measured directly by the cross-fold
nearest-neighbour statistics below.

## 2. Event-disjoint folds are footage-disjoint, not scene-disjoint

`05e` shows that the event-disjoint folds have 0 file pairs sharing packets and 0 synchronous audio pairs across folds. But 15.7%
of clips still have a cross-fold neighbour within the scene-identity radius 27.5, and two pairs sit at 3.09 and 7.85 (high 2079/2081
and 2135/2340; the first is the junction of a singleton recording with its neighbour, a missed merge). Those neighbours are the same
camera position at another time: a different phase of the same session or another session in the same room.

`05h` unites the evidence. Groups are connected components of clips sharing an event, a scene cluster at 27.5 (October single
linkage), or, in the third variant, a Tier-2 candidate.

| grouping | groups | largest group | groups with one class | cross-fold NN median | share ≤ 10 | share ≤ 27.5 | min |
|---|---|---|---|---|---|---|---|
| G1 events (= `event_disjoint_k5`) | 45 | 529 (6.9%) | 42 | 32.9 | 0.05% | 15.7% | 3.09 |
| G2 events ∪ scene clusters 27.5 | 16 | 2,290 (29.7%) | 10 | 40.9 | 0 | 0 | 27.5 |
| G3 G2 ∪ Tier-2 candidates | 10 | 4,209 (54.7%) | 5 | 44.6 | 0 | 0 | 28.2 |
| published split (test vs train) | — | — | — | 2.70 | 84.0% | 98.7% | 0 |

G2 is the coarsest grouping that every measured channel supports: packets, audio and thumbnails all agree that no held-out clip has
a near copy in training. Both residual pairs of G1 fall inside one G2 group. The price is the number and shape of the groups:

| G2 group | clips | low | mid | high | events | scene clusters |
|---|---|---|---|---|---|---|
| 10 | 2,290 | 86 | 476 | 1,728 | 8 | 35 |
| 11 | 1,163 | 554 | 253 | 356 | 6 | 1 |
| 9 | 835 | 501 | 277 | 57 | 7 | 5 |
| 13 | 745 | 233 | 378 | 134 | 10 | 10 |
| 8 | 529 | 529 | 0 | 0 | 1 | 7 |
| 12 | 456 | 276 | 180 | 0 | 2 | 2 |
| 15 | 380 | 167 | 213 | 0 | 1 | 2 |
| 2 | 308 | 0 | 0 | 308 | 1 | 11 |
| 14 | 264 | 0 | 264 | 0 | 1 | 3 |
| 5, 7, 4, 3, 6, 1, 0 | 222, 216, 145, 50, 49, 27, 22 | all single-class | | | | |

One group holds 66% of all `high` clips (1,728 of 2,636). A stratified 5-fold over G2 puts 1,728 `high` clips in one fold and 106 to
356 in the others (`union_groups_folds.csv`). Class is confounded with group: the release has about 16 independent room-and-time
units, and most of them carry a single class. This is a property of the dataset, not of the reconstruction.

## 3. Labels belong to the camera recording, not to the moment

`05c` found four audio-synchronous recording pairs that sit in different class folders (events 9, 11 and 18; r = 0.88 to 0.94 over 2
to 7 minutes). The agent could not tell at 128x72 whether the two cameras film the same group. At full resolution
(`figures/sessions_crossclass_hires.png`, three instants per event) they do:

- event 11: a student in a brown coat sits at the same table in all three views (`low` recording 82, `mid` recordings 124 and 129);
- event 18: the same standing student in black appears at the left of both views at 65 s (`low` 76, `mid` 122);
- event 9: the same group around one round table with a laptop, from three angles (`low` 102, `mid` 120 and 137).

The same group at the same instant is labelled `low` by one camera and `mid` by the others for several minutes. The label is
constant along every recording (`06`), so it was assigned per camera recording, by whatever path, and the two assignments disagree.

| | clips | share of class | published split |
|---|---|---|---|
| clips in the three cross-class events | 880 | — | train 705, val 82, test 93 |
| of which `mid` | 604 | 29.6% of `mid` | |
| of which `low` | 276 | 9.1% of `low` | |
| multi-camera events with a single class | 28 of 31 | | |

For 28 of the 31 multi-camera events the cameras agree, so the labels are mostly phase-consistent. The exceptions are not small: a
third of the `mid` class is simultaneous with footage labelled `low`. No view-invariant model can be right on both sides, which puts
a floor on the `low`/`mid` error and means that part of any `mid` accuracy measures the annotators' choice of camera.

## 4. Protocol recommendation

1. **Evaluation grouping: G2** (`union_groups.csv`, column `G2_event_scene27`), the union of audio-proven events and scene clusters.
   It is the only grouping with no cross-fold neighbour under 27.5 on any channel. Use leave-one-group-out over the 16 groups, or the
   `G2_event_scene27_fold` column (stratified 5-fold, seed 0) when training cost matters; always report the per-group confusion and a
   group-level bootstrap interval, because 16 units is the true sample size.
2. **Report three protocols side by side** on the GPU run: published split, G1 event-disjoint, G2. The gradient from 2.70 to 32.9 to
   40.9 in cross-fold nearest-neighbour distance is the leak gradient, and the accuracy drop along it is the headline result.
3. **Treat events 9, 11 and 18 as label-conflict data.** Exclude their 880 clips from the test metric (or report them separately as a
   noise-floor set); keep them in training only if the model is to learn per-view labels, which is not the stated task.
4. **Chance level per protocol.** Majority class differs by fold under G2 (fold 4 is 64% `high`); report balanced accuracy and κ, not
   accuracy, and compute the majority baseline per fold.
5. **What the dataset can support.** With 16 class-confounded groups the result is a measurement of how far accuracy falls under honest
   evaluation, not a competitive benchmark. That is still the deliverable the field lacks for room-camera group engagement, and it
   transfers to Pri-MCCD, whose release has the same design. DIPSER remains the right second dataset.

## 5. Files added by this review

- `scripts/research/05g_crossclass_hires.py` → `docs/research/figures/sessions_crossclass_hires.png` (faces; untracked),
  `outputs/research/sessions/crossclass_frames.csv`, `05g_crossclass_hires.log`.
- `scripts/research/05h_union_groups.py` → `outputs/research/sessions/union_groups.csv` (per clip: G1, G2, G3 group and fold),
  `union_groups_folds.csv`, `union_groups_summary.json`, `05h_union_groups.log`.
