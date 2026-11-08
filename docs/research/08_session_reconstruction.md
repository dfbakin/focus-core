# Reconstructing the hidden session structure of OUC-CGE

Date: 2026-11-08. Branch `research/ouc-cge`. CPU only (`CUDA_VISIBLE_DEVICES=""`), `.venv/bin/python`, every
python job under `ulimit -v 3000000` and `nice -n 10`, at most 3 worker processes, `OMP_NUM_THREADS=3`; every
random draw seeded (`numpy.default_rng(0)`, sklearn `random_state=0`). Builds on `06_temporal_structure.md` and
`07_leak_statement.md`. The figures show students' faces: internal use only.

Each section separates **observations** (what the numbers show) from **inferences** (what I think they mean).
Section 6 is the quality-monitoring section.

## 0. Scripts, outputs, wall times, memory

All run from the repository root as `CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python <script>`.

| script | what it does | wall | peak RSS |
|---|---|---|---|
| `scripts/research/05b_audio_envelopes.py --workers 3` | 100 Hz log-RMS envelope of audio track 0 of all 7,701 raw clips, frame-wise placement by pts, corrupt AAC packets skipped | 4 min 43 s | 159 MB + 3 workers <= 273 MB |
| `scripts/research/05a_recordings.py` | Tier 0: packet units with exact offsets, probable merges, recordings, flags | 11 s | 376 MB |
| `scripts/research/05c_events_audio.py --workers 3` | Tier 1: envelopes per recording, all-pairs FFT cross-correlation, nulls, events, identity links, fallbacks | 2 min 19 s | 919 MB + 3 workers <= 118 MB |
| `scripts/research/05d_session_candidates.py` | Tier 2: background / people-colour features, cross-class candidate links | 19 s | 416 MB |
| `scripts/research/05e_table_folds.py` | per-clip table, three fold files, verification, doubtful decisions | 52 s | 859 MB |
| `scripts/research/05f_figures.py` | the seven `sessions_*.png` figures | 18 s | 321 MB |

Total about 9 minutes; the largest resident set of all processes together stayed below 1.3 GB (05c with its
workers). Outputs are in `outputs/research/sessions/` (list at the end), figures in `docs/research/figures/`.

## 1. Summary

**Coverage (readable clips, n = 7,701).**

| tier | unit | count | clips covered | evidence |
|---|---|---|---|---|
| 0 recordings | exact (packet-linked, offsets exact) | 90 recordings | 6,483 (84.2%) | shared encoded packets |
| 0 recordings | probable (>= 1 frame-junction merge) | 32 recordings | 1,202 (15.6%) | junction RMSE + structure |
| 0 recordings | single clip, unlinked | 16 recordings | 16 (0.2%) | none |
| 1 events | event of >= 2 recordings | 31 events (124 recordings) | 6,059 (78.7%) | audio 5,706; packets 161; frames 189; prior 3 |
| 1 events | recording alone | 14 recordings | 1,642 (21.3%) | none |
| 2 sessions | candidate session of >= 2 events | 9 candidates (31 events) | 6,415 (83.3%) | visual, CANDIDATE only |

**Observations.**

1. 138 recordings (high 71, low 38, mid 29); 122 have >= 2 clips. The packet offsets of all 16,037 sharing pairs
   agree on a single time axis per unit to 1.8e-13 s.
2. Audio links 132 recordings (6 have no usable audio). With a threshold at the 99.9th percentile of two nulls
   that cannot be simultaneous, plus a split-half persistence test, 94 recording pairs are synchronous; none of
   94,586 null pairs passes the same test. 45 events, 31 with >= 2 recordings; no event has more than 3 distinct
   recordings at the same instant once duplicate footage is counted once.
3. **Surprise 1: four audio-synchronous pairs cross a class folder** (low/mid, r = 0.88-0.94 over 2-7 minutes,
   both halves synchronous). The release therefore contains simultaneous footage of one room under two labels.
   The briefed null "cross-class pairs cannot be simultaneous" is false, so it was replaced (section 3).
4. **Surprise 2: duplicate footage across export batches is common.** 15 audio pairs at r >= 0.95 between
   recordings of the same scene cluster and 21 frame-identity links (32x32 clip distance <= 3) join re-encoded or
   re-cut copies of one camera (h264 vs mpeg4 vs HEVC, distant IDs). 229 clips belong to the smaller copy.
5. **Surprise 3: interleaved ID runs with period 4-6 are not 4-6 cameras of one room.** In each of the 7 such
   runs the column recordings split into 2-3 audio groups (or a group plus a lone column) that are silent with
   each other at the structural lag although all have audio (37 of the 51 silent column pairs are in these runs);
   over all 24 camera runs, 44 column pairs fall in one audio event. Inference: several rooms were recorded in
   the same time slot and exported in 5-s lock-step.
6. None of the 8 recordings of the rectangular-desk ("checkerboard") room has a synchronous partner, although all
   have 260-860 s of audio. Inference: no second camera of that room is in the release.
7. Event-disjoint 5-fold split: 0 file pairs sharing packets across folds, 0 synchronous audio pairs across
   folds, minimum cross-fold 32x32 distance 3.09, cross-fold nearest-neighbour median 32.9 (published split:
   test median 2.70, 84.0% <= 10, 98.7% <= 27.5).

**Inferences.** The release contains 31 multi-camera "moments" plus 14 single-camera stretches. Labels are not a pure function of the
moment: in 3 events (4 synchronous pairs) two cameras of one room at one instant carry different labels (low vs mid). Tier 2
cannot be verified from the release; its candidates are shown for manual checking.

## 2. Tier 0: recordings

**Method** (`05a_recordings.py`).

- *Exact units.* Connected components of the packet graph (pairs sharing a >= 1 KB packet, from
  `packet_pairs_offsets.csv`). Each clip's container origin is solved by breadth-first search over
  `origin(b) = origin(a) + mode(pts_a - pts_b) / 15360`; the residual over all 16,037 pairs is <= 1.8e-13 s. A clip's
  recording time is `origin + container time`, which handles first-pts offsets, sparse files and audio alike.
  Containment and re-cut pairs at distant IDs are included (same encoded footage).
- *Unreliable timing.* 158 clips (115 with <= 5 frames and 43 whose frames are spread over minutes: the broken
  export families of `03` sec. d plus `view1394`, `view1765`, `view1898`) may link only to each other. In a first
  run they bridged unrelated parts of camera columns through frames at 0.07 s and 320 s and produced a 120-s jump
  in the time axis of high 1401-1768, visible as a structural lag of 121.8 s against 0 s in the neighbouring run.
  Their timing is `unknown`; their identity with the column is restored in Tier 1 ("packets" links).
- *Probable merges.* Tail clips of unit A (covering its last 6 s) against head clips of unit B within 26 IDs.
  All criteria must hold: same class; same batch combo; same frame run of 04d and, for runs with |period| >= 2,
  an ID step that is a multiple of the period; step sign equal to the run's time direction (unit's own packet
  direction where the run has period 0); same scene cluster at 27.5; |step| <= 3 max(|p|, 1) + 2; and
  RMSE(last(tail), first(head)) at 128x72 <= 38.57. The cut is the 95th percentile of the 6,478 packet-verified
  junctions of 04d (equal to the median RMSE after a 3-5 s lag, `06` sec. 2): no merge is farther apart than
  95% of real junctions. Greedy by ascending RMSE, one successor and one predecessor per unit, no cycles.
  `merge_conf` = share of packet-verified junctions at least as far apart.

**Observations.**

| | count |
|---|---|
| exact units with >= 2 clips / clips in them | 100 / 7,308 |
| single-clip units (391 without any shared packet) | 393 |
| candidate merges / passing all criteria / accepted | 12,842 / 572 / 355 |
| rejected by exactly one criterion: distance / direction / step / batch / column | 935 / 171 / 151 / 12 / 3 |
| accepted merges with a packetless side (HEVC 178 clips, h264 1064x598, ...) / between two packet units | 350 / 5 |
| recordings / with >= 2 clips | 138 / 122 |
| recordings by tier: exact / probable / single clip | 90 / 32 / 16 |

- Junction RMSE (`sessions_tier0_merges.png`): accepted merges median 4.0 (p95 10.1); exact junctions median
  11.9 (p95 38.6). The packetless HEVC and 1064x598 clips abut at frame boundaries (no overlap), hence the low
  values. The 5 merges between packet units have RMSE 2.5, 7.5, 25.6, 30.7, 37.2 (`merge_conf` 0.91 to 0.06).
- Time axis per clip (`t_source`): packets 6,837; packets with a merge estimate downstream 314; frames 42; frames
  with merge estimate 350; unknown 162 (158 unreliable + 4 unreadable). A merge places the head after the tail
  minus the median packet overlap (9 frames) when a side has no packets, or after a gap read from the
  RMSE-to-lag table of `06` sec. 2.
- Three units contain overlapping clips that share no packet (226 pairs, nearly all in high 722-1107 and low
  2002-2180, mpeg4 1280x720). For the 167 pairs with equal start times the first frames differ by RMSE 0.98-1.75
  at 128x72 (median 1.55), below the same-raw-frame floor of 2.2: re-encoded copies of the same footage, bridged
  by byte-identical files. The time axis holds (`unit_overlaps_without_packets.csv`, column `d_first_aligned`).
- 103 recordings contain both train and test clips; 126 of 355 merges join clips of different published splits.

Per class (clips): exact high 2,184 / low 2,293 / mid 2,006; probable 433 / 734 / 35; single 15 / 1 / 0.

## 3. Tier 1: events

### 3.1 Audio envelopes and scores

- `05b`: audio stream 0 of every raw clip, decoded packet by packet (49 high files in 2010-2620 and `high/view2633`
  contain 1-4 corrupt AAC packets; skipping them keeps the other ~430 frames, whereas whole-stream decoding as in
  04e lost the file); each frame is placed at its pts. 7,645 files decoded; 56 have no audio stream; 166 have a
  single AAC frame (21 ms). The brief's "92 single-frame" does not match this count: all 166 have
  `a0_frames == 1` in `raw_header_audit.csv`; they are the clips that kept source timestamps (first pts >= 1 s).
- Per recording (`05c`): each clip's envelope is placed at `origin + unit_shift + a_start + i/100`, averaged where
  clips overlap, resampled to 20 Hz and high-passed (minus its 5-s running mean). Masked: the first 30 ms of every
  clip (AAC priming dip, repeated at every 5-s cut) and digital-silence dropouts (log10 RMS < -4.5, 317 clips),
  both of which correlate between any two recordings cut on one grid. Clips with < 0.5 s of audio are skipped.
- Score of a pair: masked Pearson correlation at every lag (FFT), maximised over recording-time lags |L| <= 60 s
  (`r60`; L = 0 means both recordings start together) and, separately, over all lags (`rfull`). Only lags with an
  overlap >= 80% of the shorter recording's valid audio (P) are scanned, so that the statistic matches the null
  windows. **Persistence**: at the best lag each half of the overlap must on its own exceed the 99th-percentile
  threshold of its length bin, so a short common sound cannot carry a link. In the first version without the
  overlap rule, several cross-session links were carried by 23-33 s overlaps at lags of 50-58 s.

### 3.2 Nulls and thresholds

The briefed null (cross-class pairs) was tried first. Its upper tail held r60 = 0.90 over 426 s, 0.88 over 254 s
and 0.81 over 115 s at lags of 1.5-13.5 s; the frames show the same people from other angles at the aligned times
(e.g. a person in a brown coat in low/1170 and mid/687). These pairs are real, so cross-class pairs were moved
to the candidates and two nulls that cannot be simultaneous were used instead:

- **B**: two time-disjoint windows of the same recording (P s against P + 120 s, at least P + 60 s apart): same
  room, same microphone, same people, different minutes. 60,002 pairs, about 10,000 per length bin up to 320 s.
- **R**: every recording pair with the partner's envelope reversed in time, both orders: exactly the candidates'
  lengths, gaps and lag scan. 17,292 pairs.

Threshold per bin of P = max of the two nulls' percentiles (`sessions_audio_null.png`, `audio_thresholds.csv`):

| P (s) | null B n | null R n | B q99.9 | R q99.9 | threshold q99 / q99.9 / q99.99 | candidate pairs same / cross class | above q99.9 same / cross |
|---|---|---|---|---|---|---|---|
| 0-10 | 9571 | 506 | 0.926 | 0.773 | 0.877 / 0.926 / 0.949 | 125 / 130 | 2 / 0 |
| 10-20 | 9942 | 3760 | 0.711 | 0.704 | 0.655 / 0.711 / 0.766 | 838 / 1058 | 8 / 2 |
| 20-40 | 10000 | 1278 | 0.557 | 0.442 | 0.488 / 0.557 / 0.594 | 247 / 398 | 6 / 0 |
| 40-80 | 10015 | 813 | 0.422 | 0.330 | 0.363 / 0.422 / 0.447 | 150 / 260 | 7 / 0 |
| 80-160 | 9972 | 5126 | 0.277 | 0.294 | 0.251 / 0.294 / 0.308 | 853 / 1712 | 16 / 3 |
| 160-320 | 9997 | 3710 | 0.189 | 0.205 | 0.183 / 0.205 / 0.217 | 637 / 1218 | 22 / 3 |
| 320-640 | 2 | 1248 | - | 0.149 | 0.136 / 0.149 / 0.153 | 201 / 423 | 11 / 1 |
| > 640 | 0 | 12 | - | - | - | 1 / 5 | 0 / 0 |

Links surviving (8,646 pairs: 3,247 same class, 5,399 cross class):

| | q99 | q99.9 (primary) | q99.99 |
|---|---|---|---|
| above threshold, same / cross class | 88 / 34 | 72 / 9 | 64 / 5 |
| above threshold **and persistent**, same / cross class | 62 / 4 | 61 / 4 | 60 / 4 |
| null pairs above threshold and persistent (B / R) | | 0 / 0 of 60,002 / 17,292 | |
| expected false links from the threshold alone, (1 - q) x pairs | 82.6 | 8.3 | 0.8 |

The persistence test removes 16 pairs that are above the primary threshold (11 same-class, 5 cross-class; median
P 25 s). Estimated false-link rate of the combined rule: 0 of 77,294 null pairs pass, so the rate per pair is
< 3.9e-5 (95% upper bound, rule of three) and the expected number of false links among the 8,646 pairs is < 0.34.

A second pass scans all lags (`rfull`) with its own threshold (99.9th percentile of the reversed null scanned
over all lags, same persistence rule; 0 of 17,292 null pairs pass) for recordings that start more than 60 s
apart: it adds 29 pairs. 16 of them close cycles already built from +-60 s links and agree with them to <= 0.05 s.

### 3.3 Event building

Accepted links (65 at +-60 s, 29 full-lag) are merged by descending null percentile (Kruskal); the event axis
follows the tree. Decisions: 51 tree links at +-60 s (9 of them duplicates), 13 full-lag tree links (6
duplicates), 30 cycle-closing links, all consistent (|residual| <= 0.05 s, median 0), no conflict.

- **Duplicates.** A pair at r >= 0.95 in the same scene cluster is a copy of one camera's footage (all accepted
  pairs at r >= 0.95 are same-cluster; the highest different-cluster pair is 0.938). It joins the event and is not
  counted as a second camera; the copy with fewer clips is flagged `reencoded_duplicate_recording`.
- **Footage identity** (exact or near-exact, offset unknown): 6 recording pairs share packets through the broken
  families (05a kept them out of the time axes); 21 further recording pairs have a clip pair at 32x32 clip distance
  <= 3.0 (34 more such pairs were already in one event). Both join events (tiers `packets`, `frames_duplicate`).
- **Structure (ii).** For the 95 column pairs of the 24 runs typed "cameras", the structural lag (median
  start-time difference of partner clips N, N + j) was used to test audio in a +-10 s window. 44 pairs were
  already in one audio event (structural and audio offsets differ by median 5.7 s, p75 14 s, max 187 s); the
  other 51 are silent at the structural lag although both have audio, so they were **not** linked. No
  structural fallback was needed: every recording that audio left alone has >= 30 s of audio.
- **Prior (iii)**, fallback: 2 recordings (3 clips) with < 30 s of audio attached by ID adjacency within the same
  codec and frame size and the same layout; 14 recordings with audio were not eligible.

**Observations.**

| | |
|---|---|
| events / with >= 2 recordings | 45 / 31 |
| recordings per event | 1: 14; 2: 13; 3: 9; 5: 3; 6: 1; 7: 1; 8: 2; 10: 1; 17: 1 |
| events by class set | high 10, low 10, mid 8, low+mid 3 (plus 14 single-recording events) |
| recordings by event tier | audio 96, frames_duplicate 19, packets 7, prior 2, none 14 |
| clips by event tier | audio 5,706 (74.1%), frames_duplicate 189 (2.5%), packets 161 (2.1%), prior 3, none 1,642 (21.3%) |
| clips in cross-class events | 880 |
| alone (single-recording events), by class | high 8 rec. / 510 clips; low 4 / 879; mid 2 / 253 |
| alone, by audio | all 14 have >= 30 s of valid audio; 0 of the 222 clips without usable audio are alone |

The 14 recordings alone: the 8 checkerboard recordings (high 2227-2480, 2318-2347, 2481-2629; low 543-2399,
544-592, 593-2883; mid 12-203, 13-1223), low 267-542 (the 23-min chain), the fourth columns of the period-4 runs
high 1107-1222 and 1397-1510 (high 1111-1223, 1403-1511), high 499-589, 786-802 and 2061-2079.

## 4. Tier 2: session candidates

Visual evidence only (`05d`): per recording, the per-pixel median of up to 60 first/last frames (128x72) is the
background; pixels with gray MAD < 6 are static, MAD >= 12 moving (people). `view_d` = background RMSE over pixels
static in both; `cloth_s` = intersection of HSV histograms (8x4x4) of the moving pixels. Calibration: first vs
second half of one recording (same camera, same people): view_d p95 28.6, cloth_s p5 0.73, median 0.89;
recordings of different events in one class (mostly other sessions): cloth_s median 0.64, p95 0.83, and 7.9%
of them pass the view cut. Different cameras of one event (same people, other angle): cloth_s median 0.66, so
the colour signature is only comparable within one viewpoint. A cross-class event pair is a candidate if some
recording pair passes view_d <= 28.6, has the same layout and cloth_s >= 0.73; score = share of reference pairs
with lower cloth_s; only mutual best candidates per other class are linked.

**Observations.** 698 cross-class event pairs; 167 view-matched; 123 candidates; 28 mutual-best links (score
0.78-1.00, median 0.995). 32 of 45 events have a candidate partner (23 of the 31 multi-recording events). 22
session candidates, 9 with >= 2 events holding 6,415 clips; 6 contain all three levels, 2 high+low, 1 low+mid.
In `sessions_candidates_sheet.png` all eight random links show the same camera position, and in most rows the
same people are recognisable by eye (the person in an orange top in events 12/27, 14/24 and 24/29; a brown coat in
15/42); two rows (events 10/21, 5/7) are not decidable at this resolution. The lesson-plan check "every session
contains all three levels" holds for 6 of 9 multi-event candidates.

**Inference.** Tier 2 is plausible where a camera stayed in place between phases and people kept their clothes;
it cannot separate two sessions of the same group in the same room on different days if clothes were similar.
Not verifiable from the release.

## 5. Table and folds

`clip_sessions.csv` (7,705 rows; the 4 unreadable clips have no ids and `unreadable` set): path, class, label,
split_published, scene_cluster_27_5, recording_id, recording_tier (exact / probable / singleton), exact_unit_id
and chain_04d (pre-merge ids, so merges can be undone), camera_column, t_start_s, t_end_s, t_source (packets /
frames / unknown, `+merge_estimate` when a probable merge sits upstream) and t_source_detail, event_id,
event_tier (audio / packets / frames_duplicate / structural / prior / none), event_confidence (`r=...;
null_pct=...` of the recording's best accepted link, else NA), event_audio_r, event_audio_null_pct,
event_offset_s, t_event_s, event_size_recordings, event_classes, duplicate_of_recording, layout,
session_candidate_id, session_candidate_size_events, session_score, flags (semicolon list) and one boolean per
flag: byte_duplicate 299, fragment_le5_frames 115, padded_320p 241, recut_duplicate_footage 1,691 (fully
contained in another, non-identical file; of an identical-packet pair only the higher ID), no_audio 222 (no
stream or < 0.5 s), hevc_no_packets 183, unreliable_timing 158, reencoded_duplicate_recording 229,
no_shared_packets 391, audio_single_frame 166, unreadable 4; plus byte_duplicate_redundant 171, readable and
clean_subset. **clean_subset** = readable and none of fragment, redundant byte copy, recut duplicate, padded
320p, unreliable timing, re-encoded duplicate recording: 5,692 clips (high 2,077, low 2,141, mid 1,474).

Folds (readable clips; `outputs/research/sessions/folds/`):

1. `event_disjoint_k5.csv`: `StratifiedGroupKFold(5, shuffle=True, random_state=0)`, groups = event_id (45 groups;
   a recording alone is its own group).
2. `loeo.csv`: leave-one-event-out for the 10 largest events (289-529 clips). Column `loeo_k` is `test` for the
   event, `train_session_candidate` for clips of the same Tier-2 candidate (exclude them for a strict test), else
   `train`.
3. `session_candidate_k5.csv`: same splitter, groups = session_candidate_id where the candidate has >= 2 events,
   else event_id (22 groups). Marked as candidate-based in a column.

Verification (`fold_verification.csv` / `.json`). Leakage statistics use `nn_stats()` imported from
`scripts/review/07_make_grouped_splits.py` on the review's clip distance (min over the 3x3 first/centre/last
pairs of the 32x32 RMSE), computed blockwise in float64; it reproduces the published-split numbers of `07`
(test median 2.70, 84.0% <= 10, 96.0% <= 17.5, 98.7% <= 27.5).

| scheme | fold | n | low / mid / high | NN median | NN min | <= 10 | <= 17.5 | <= 27.5 |
|---|---|---|---|---|---|---|---|---|
| event_disjoint_k5 | 0 | 1667 | 696 / 398 / 573 | 34.8 | 19.8 | 0 | 0 | 7.9% |
| | 1 | 1489 | 597 / 386 / 506 | 33.5 | 3.1 | 0.1% | 0.2% | 17.4% |
| | 2 | 1475 | 566 / 391 / 518 | 32.6 | 19.2 | 0 | 0 | 8.6% |
| | 3 | 1524 | 571 / 436 / 517 | 30.7 | 20.0 | 0 | 0 | 30.0% |
| | 4 | 1546 | 598 / 430 / 518 | 32.1 | 3.1 | 0.1% | 0.1% | 15.1% |
| session_candidate_k5 | 0 | 2242 | 1043 / 477 / 722 | 35.8 | 12.6 | 0 | 0 | 1.9% |
| | 1 | 1140 | 421 / 350 / 369 | 37.7 | 3.1 | 0.2% | 0.2% | 10.7% |
| | 2 | 1560 | 407 / 476 / 677 | 34.5 | 23.3 | 0 | 0 | 1.7% |
| | 3 | 1288 | 536 / 382 / 370 | 30.2 | 3.1 | 0.2% | 0.2% | 29.3% |
| | 4 | 1471 | 621 / 356 / 494 | 29.8 | 19.6 | 0 | 0 | 30.2% |
| published split | val | 770 | 307 / 199 / 264 | 2.9 | 0 | 83.2% | 94.5% | 99.2% |
| | test | 771 | 325 / 189 / 257 | 2.7 | 0 | 84.0% | 96.0% | 98.7% |

| scheme | file pairs sharing a packet across folds | accepted audio links across folds | min cross-fold distance | cross-fold NN median / <= 10 / <= 27.5 |
|---|---|---|---|---|
| event_disjoint_k5 | **0** | **0** | 3.09 | 32.9 / 0.05% / 15.7% |
| session_candidate_k5 | **0** | **0** | 3.09 | 33.8 / 0.05% / 13.1% |
| published split | 5,025 | 94 | 0 | |

LOEO (10 events): 0 packet pairs across in every fold; test-to-train NN median 29.8-38.4, minimum 7.8-28.3;
removing the `train_session_candidate` clips moves the medians by <= 0.8. Four clips are within 10 of another
fold in both k-fold schemes: high 2079 / 2081 (d 3.09, ID-adjacent across a run boundary, recordings 52 and 54) and
high 2135 / 2340 (d 7.85, HEVC clips of recordings 56 and 64; 2135 has no audio).

## 6. Quality monitoring

### 6.1 Coverage per tier

| | high | low | mid | all |
|---|---|---|---|---|
| Tier 0 exact | 2,184 | 2,293 | 2,006 | 6,483 (84.2%) |
| Tier 0 probable | 433 | 734 | 35 | 1,202 (15.6%) |
| Tier 0 single clip | 15 | 1 | 0 | 16 (0.2%) |
| Tier 1 audio | 1,872 | 2,048 | 1,786 | 5,706 (74.1%) |
| Tier 1 packets / frames_duplicate | 151 / 98 | 10 / 91 | 0 / 0 | 161 / 189 (4.5%) |
| Tier 1 prior (fallback) | 1 | 0 | 2 | 3 (0.04%) |
| Tier 1 none (recording alone) | 510 | 879 | 253 | 1,642 (21.3%) |
| Tier 2 in a multi-event candidate | 2,213 | 2,331 | 1,871 | 6,415 (83.3%), candidate only |

### 6.2 Evidence mix (links)

| evidence | links | clips it places |
|---|---|---|
| packets, within units (04d chain links + all sharing pairs) | 16,037 pairs -> 100 units | 7,308 |
| frames, Tier-0 junction merges | 355 | 1,202 clips in probable recordings |
| audio +-60 s (tree / cycle) | 65 (51 / 14) | 5,706 clips with tier audio, with the full-lag links |
| audio full lag (tree / cycle) | 29 (13 / 16) | |
| footage identity: packets / frames | 6 / 21 | 161 / 189 |
| structure (fallback) | 0 (44 column pairs agree with audio, 51 silent) | 0 |
| prior (fallback) | 2 | 3 |
| Tier 2 visual candidates | 28 mutual-best links | 6,415 (candidate) |

### 6.3 Confidence distributions with the null

- Audio: accepted +-60 s links have r60 median 0.88 (IQR 0.83-0.91, min 0.53) at P median 174 s; the null medians
  in the bins from 40 s up are 0.09-0.23 (`sessions_audio_null.png`). For 85 of the 94 accepted links the weaker
  half of the overlap lies above every null sample of its half-length bin; the lowest is at the 96.9th percentile
  (a 10-s single-clip recording). The doubtful list uses this percentile as the audio confidence.
- Tier 0: accepted merges against exact junctions: median 4.0 vs 11.9, `merge_conf` median 0.84 (p5 0.57)
  (`sessions_tier0_merges.png`).
- Tier 2: candidate scores 0.78-1.00 against a reference distribution that cannot exclude same-session pairs.

### 6.4 Estimated false-link rates

- Audio, combined rule: 0 of 77,294 null pairs (+-60 s) and 0 of 17,292 (full lag) pass; < 3.9e-5 per pair,
  < 0.34 expected false links among 8,646 pairs. Threshold alone at q99 / q99.9 / q99.99: 82.6 / 8.3 / 0.8
  expected.
- Frame identity (d32 <= 3): no null; risk is a static scene seen by one camera at two times. 21 links.
- Tier 0 merges: precision of the frame-junction rule against packets was 0.94 at RMSE 16.5 and 0.87 at 76.9
  in 04d; merges here also require batch, run, column, direction and scene agreement, and 95% have RMSE <= 10.1.
- Tier 2: not estimable.

### 6.5 Fallbacks

| fallback | used | clips affected |
|---|---|---|
| frame-based time axis (no packets) | 42 clips + 350 with merge estimate | 392 |
| merge-estimated offsets downstream of a probable merge (packet units) | 314 clips | 314 |
| unknown timing (broken families, sparse files) | 158 clips | 158 |
| structural event link | 0 | 0 |
| prior event link | 2 recordings | 3 |
| event offset unknown (identity or prior attachment, or an audio group joined to its event by identity) | 34 recordings | see `events.csv` (`event_offset_s` NaN) |
| full-lag audio pass (beyond the briefed +-60 s) | 13 tree links | |
| room layout = agent's visual label of 138 frames | all recordings | 8 checkerboard |

### 6.6 Consistency checks

| check | result |
|---|---|
| <= 3 distinct recordings at any instant of an event (duplicates counted once; 0.5 s junction tolerance) | **pass**: max 3 (13 events), 2 (16), 1 (2). Before duplicates were recognised, 6 events reached 4 |
| recordings of one event share no packets | **pass for distinct cameras**; 159 clip pairs share packets inside events, all between the 6 recording pairs joined because they are the same footage (broken families and their columns) |
| one room layout per event | **pass**: 0 mixed events |
| cycle consistency of audio offsets | **pass**: 30 cycles, |residual| <= 0.05 s |
| agreement with the 28 synchronous column pairs of 04f | **26 / 28 in the same event**. Offsets of the 26: 9 agree within 0.02 s, 11 within 0.9-5.0 s (04f stitched columns by clip durations where packets were missing), 4 differ by 155-448 s because the 04f column's first clip is not in the recording (its offset cannot be translated), 2 not comparable (offset unknown). The two misses (low 1852-1997 cols 2/5, mid 1745-1768 cols 0/2) have r60 = 0.12 here; their 04f scores were 0.44 and 0.42, just above its threshold 0.41 |
| structural lag vs audio offset, same-event column pairs | median 5.7 s, p75 14 s: structure is a coarse prior only |
| unassigned recordings | 14 recordings (1,642 clips), all with >= 30 s of audio: high 8, low 4, mid 2; 0 by missing audio |
| "every session contains all three levels" (Tier 2) | 6 of 9 multi-event candidates |

### 6.7 The 15 most doubtful decisions

`doubt = (1 - confidence) x clips on the smaller side` (removing the link splits a recording or an event; links
confirmed by another path affect no clip and are not doubtful). Shown in `sessions_doubtful_sheet.png`; full list
in `doubtful_decisions.csv` (409 bridge decisions: 355 merges, 25 audio, 21 frame identity, 6 packets, 2 prior).

| # | decision | evidence | value | confidence | clips | reason |
|---|---|---|---|---|---|---|
| 1 | merge low 993 -> 991 (units 342 -> 346) | junction RMSE | 37.2 | 0.06 | 69 | at the cut; checkerboard room, two packet units |
| 2 | merge high 2520 -> 2522 (327 -> 328) | junction RMSE | 30.7 | 0.10 | 40 | two packet units, gap of unknown length |
| 3 | rec 21 - rec 38 (high) | frame identity | 6 pairs, d32 1.36 | 0.77 | 118 | could be a static view at two times |
| 4 | rec 5 - rec 19 (high) | frame identity | 1 pair, d32 2.75 | 0.54 | 46 | single pair near the cut |
| 5 | rec 40 - rec 55 (high) | frame identity | 57 pairs, d32 1.56 | 0.74 | 60 | broken 2-frame family vs HEVC clips |
| 6 | rec 73 - rec 100 (low) | frame identity | 8 pairs, d32 1.27 | 0.79 | 44 | re-encoded copy |
| 7 | merge high 2134 -> 2136 (213 -> 215) | junction RMSE | 33.2 | 0.08 | 10 | HEVC, near the cut |
| 8 | rec 73 - rec 98 (low) | frame identity | 1 pair, d32 1.40 | 0.77 | 37 | single pair |
| 9 | merge low 2219 -> 2217 (383 -> 382) | junction RMSE | 10.7 | 0.54 | 18 | h264 1064x598 chain, no packets |
| 10 | merge high 2212 -> 2213 (266 -> 267) | junction RMSE | 22.4 | 0.21 | 10 | HEVC |
| 11 | merge low 2320 -> 2318 (434 -> 433) | junction RMSE | 8.9 | 0.61 | 20 | 1064x598 chain |
| 12 | merge low 2310 -> 2308 (429 -> 428) | junction RMSE | 9.2 | 0.60 | 19 | 1064x598 chain |
| 13 | rec 7 - rec 45 (high) | frame identity | 141 pairs, d32 1.26 | 0.79 | 33 | mpeg4 854x480 vs HEVC copy |
| 14 | merge low 2304 -> 2302 (426 -> 425) | junction RMSE | 8.9 | 0.61 | 16 | 1064x598 chain |
| 15 | merge low 2296 -> 2294 (422 -> 421) | junction RMSE | 11.4 | 0.52 | 12 | 1064x598 chain |

No audio link is among them: the 25 audio bridges have their weaker half above the null of its bin. The frames of
all 15 show the same viewpoint on both sides (`sessions_doubtful_sheet.png`).

### 6.8 Known limits

- **Tier 2 cannot be verified from the release.** No packet, frame or audio link crosses phases; the candidates
  rest on viewpoint and clothing colour, which do not separate two sessions of one group in one room.
- Cross-class simultaneity (4 pairs) shows that "class folder = lesson phase" does not hold everywhere; whether
  the two cameras film different groups or the same group labelled differently cannot be told at 128x72.
- Recordings starting > 60 s apart are found only by the full-lag pass; recordings with < 30 s of audio rest on
  identity or prior evidence; the 56 clips without an audio stream and 166 single-frame clips carry no audio.
- Event offsets are unknown for 34 recordings (identity or prior attachments); the 3-camera check ignores them.
- The duplicate rule (r >= 0.95 and same scene cluster) and the frame-identity cut (3.0) are set from the data,
  without ground truth; two clips at 3.09 and 7.85 sit just above it and cross folds.
- Room layout is my visual label of one frame per recording (`sessions_recordings_sheet.png`); only two layouts
  were distinguished.
- Probable merges inside HEVC and 1064x598 chains rely on frames alone (no packets exist there).
- Labels are constant per recording by construction of the class folders; nothing here locates label changes.

## Files

- Scripts: `scripts/research/05a_recordings.py`, `05b_audio_envelopes.py`, `05c_events_audio.py`,
  `05d_session_candidates.py`, `05e_table_folds.py`, `05f_figures.py`.
- `outputs/research/sessions/`:
  - table: `clip_sessions.csv`;
  - folds: `folds/event_disjoint_k5.csv`, `folds/loeo.csv`, `folds/session_candidate_k5.csv`;
  - Tier 0: `clip_recordings.csv`, `units.csv`, `recordings.csv`, `merges.csv` (every candidate and the reason),
    `unit_overlaps_without_packets.csv`, `tier0_summary.json`;
  - Tier 1: `audio_envelopes.npz`, `audio_pairs.csv` (every recording pair, both scores, halves, decision),
    `audio_null.csv`, `audio_thresholds.csv`, `events.csv`, `event_table.csv`, `structural_links.csv`,
    `identity_links.csv`, `prior_links.csv`, `compare_04f_column_pairs.csv`, `tier1_summary.json`;
  - Tier 2: `recording_features.npz`, `session_pairs.csv`, `session_links.csv`, `session_candidates.csv`,
    `tier2_summary.json`;
  - verification: `fold_verification.csv`, `fold_verification.json`, `doubtful_decisions.csv`,
    `table_folds_summary.json`; logs `05*.log`.
- Figures (`docs/research/figures/`): `sessions_events_sheet.png`, `sessions_doubtful_sheet.png`,
  `sessions_candidates_sheet.png`, `sessions_timeline.png`, `sessions_recordings_sheet.png`,
  `sessions_audio_null.png`, `sessions_tier0_merges.png`.
