# Temporal and camera structure along the OUC-CGE clip IDs

Date: 2026-11-08. Branch `research/ouc-cge`. CPU only (`CUDA_VISIBLE_DEVICES=""`, 8 processes at most),
`.venv/bin/python`, every random draw seeded (`numpy.default_rng(0)`, sklearn `random_state=0`).
This report follows `05_id_structure_probe.md` (3 x 61 clips) and covers all 7,701 readable clips.
It combines three independent sources of evidence:

- **Frames**: the first and last frame of every 320p clip at 128x72 (`frames_*_hires.npz`) and the 32x32
  thumbnails of the October review.
- **Packets** (ground truth for temporal overlap): the hash of every encoded video packet of every raw
  file (`raw_video_packets.npz`). The clips were cut by stream copy, so two files that contain a
  byte-identical packet hold the same encoded frame. The position of that packet in each file gives
  the exact frame offset between the two files.
- **Audio**: the loudness envelope of the raw audio track. It is used to test whether different camera
  streams were recorded at the same time.

The figures show students' faces. They are for internal use and should not go into the public repository.

Each section separates **observations** (what the numbers show) from **inferences** (what I think they mean).

## 0. Scripts, outputs, wall times

All commands are run from the repository root as `CUDA_VISIBLE_DEVICES="" .venv/bin/python <script>`.

| script | what it does | wall |
|---|---|---|
| `scripts/research/04a_reference_scales.py` | decodes frames 0..150 of 300 random 320p clips: consecutive-frame and lag RMSE at 128x72 | 38 s |
| `scripts/research/04b_packet_offsets.py` | for every file pair sharing packets: exact frame offset, overlap and relation (16,037 pairs) | 33 s |
| `scripts/research/04c_successor_distances.py` | d_k, b_k, f_k (k = 1..8), RGB and gray, per clip; random reference pairs | 43 s |
| `scripts/research/04d_temporal_analysis.py` | A (cuts, packet validation, segmentation, runs), B (partner tests), C (chains, label runs, split leak) | 27 s |
| `scripts/research/04e_audio_sync.py` | audio-envelope correlation of individual partner pairs within ±1 s | 1 min 40 s |
| `scripts/research/04f_audio_columns.py` | audio synchrony of whole camera columns (±30 s); simultaneous views; leak union | 1 min 11 s |
| `scripts/research/04g_temporal_figures.py` | figures D1-D3, plus two supporting figures | 9 s |
| `scripts/research/04h_leak_figures.py` | full 7,701² 32x32 distance matrix; figures D4-D7 | 51 s |
| `scripts/research/04i_report_tables.py` | markdown tables (`outputs/research/temporal/report_tables.md`) | 1 s |

Total about 7 minutes. Outputs are in `outputs/research/temporal/` (CSV and JSON, listed at the end) and figures in
`docs/research/figures/`.

## 1. Method

**Distances.** All distances in A-C are RMSE between 128x72 RGB frames on the 0-255 scale. All distances in the
leak figures use the review's clip distance: the minimum over the 3x3 first/centre/last pairs of the RMSE
between 32x32 thumbnails. The two scales are not interchangeable.

**Both directions.** The brief defines only the forward distance d_k(N) = RMSE(last(N), first(N+k)). The
packets show that in most of `low` a higher ID starts *earlier* in time (section 3). There, the continuation
of N is the clip *before* it in ID, and only the backward distance b_k(N) = RMSE(last(N+k), first(N)) can find
it. For every clip I therefore take k*, dir* = argmin over the 16 candidates {d_1..d_8, b_1..b_8} and call the
minimum m(N). The forward-only version is reported alongside for comparison.

**Signed period.** lab(N) = dir* x k*, the ID step per forward step in time (−1 means "IDs run backwards with
period 1"). The ID axis is segmented by the majority label in a centred window of 15 clips. The same
segmentation is run on the exact packet successor of each clip.

**RGB or gray.** I used RGB. Gray gives the same argmin for 99.6% of clips (median m 11.6 gray vs 11.7 RGB).

## 2. Reference scales at 128x72 (A)

| quantity | median | spread |
|---|---|---|
| 320p frame 0 vs frame 1 | **0.28** | p5-p95 0.14-2.15 |
| real consecutive raw frames (steps t >= 3, per-clip median, 298 clips) | **2.79** | p25-p75 1.46-4.65, p5-p95 0.34-8.34 |
| the same raw frame in two different clips (packet-verified junctions with lag 0, n = 335) | **2.22** | p5-p95 1.91-2.53 |
| frame 0 vs frame L (L = 4 / 8 / 15 / 30 / 60 / 150) | 4.9 / 11.1 / 16.5 / 20.9 / 27.5 / 35.8 | |
| first vs last frame of one clip (all 7,701) | **36.8** | p5-p95 14.9-67.4 |
| random same-class pairs, first vs first (20,000) | **95.9** | p0.1 25.4, p1 40.9 |
| random cross-class pairs, first vs first (20,000) | 96.5 | p0.1 41.8, p1 47.0 |

Observation: frames 0, 1 and 2 of most 320p clips are the same raw frame. ffmpeg pads the start up to the raw
first pts, which is about 0.067 s. Their RMSE (0.28, 0.24) is therefore the re-encoding floor of one frame, not
a step in time. The first real step is between frames 2 and 3 (median 2.1; 2.8 from t = 3 on). The probe's
"consecutive-frame floor 0.2-0.5" measured this padding.

The supporting figure `temporal_min_d_hist.png` (right panel) calibrates RMSE against time. It plots the RMSE at
the junction of 7,366 packet-verified neighbour pairs against their exact lag in frames:

| lag | 0 | 1-2 | 3-5 | 6-9 | 10-14 | 15-29 | 30-89 | 90-159 frames |
|---|---|---|---|---|---|---|---|---|
| median RMSE | 2.2 | 4.5 | 8.3 | 12.6 | 15.8 | 21.1 | 29.9 | 38.6 |

## 3. Exact strides from packets

From `packet_offsets_by_iddiff.csv` and `summary.json` → `packet_successor_stride`.

**Observations.**

- In every file pair, every shared packet gives the same frame offset (100% of pairs, both in demux order and
  in pts order). The offsets concentrate at 150 and 300 frames.
- The immediate packet successor of a clip (partial overlap, later start, later end) starts:
  - **150 frames (5.0 s)** later in 72% of `high`, 53% of `low` and 60% of `mid` clips;
  - 300 frames (10 s) later in 14% / 16% / 29%;
  - 60 frames (2 s) later in 12.5% of `low`.
- The two segments overlap at the junction by a median of **9-10 frames (0.3 s)**; p95 is 35 (high), 103 (low)
  and 26 (mid) frames.
- Time direction of ID-adjacent pairs that share packets:

  | class | higher ID starts later | higher ID starts earlier | same start (containment) |
  |---|---|---|---|
  | high | 78% | 0% | 22% |
  | mid | 97% | | |
  | low | 10% | **79%** | 11% |

  The signed ID step to the packet successor is −1 for 61% of `low` clips and −2 for 18%.

**Inference.** The raw sources were cut into segments of about 5.3 s every 5 s, or about 10.3 s every 10 s,
with a few frames of overlap. Within an export batch, the IDs of one source follow time forwards in `mid` and
`high` and **backwards** in most of `low`.

## 4. Continuation detection and the cut (A)

**Histogram.** The log-histogram of m (both directions) has three modes (Gaussian KDE, Scott bandwidth 0.066
in log10):

| mode | where | what it is |
|---|---|---|
| 1 | m = 2.34 | frame-exact junction: the same raw frame, see the 2.22 floor above |
| 2 | m = 15.7 | the main mode: a junction a few frames inside an overlap |
| 3 | m = 92 | a different room |

The valleys between them are at **3.31** and **76.9** (`temporal_min_d_hist.png`, left). The forward-only
histogram has a mode at 2.4 and a main mode at 34 (the backward-running `low` clips are pushed up), and no
upper valley.

**Cuts.**

| cut | value | meaning |
|---|---|---|
| primary | **76.9** | upper valley: the best neighbour is in the same scene, not a different room |
| strict | **3.31** | lower valley: frame-exact junctions only |
| intermediate | **16.5** | median RMSE after 0.5 s of motion |

**Validation against packets.** The truth is "N and N+k* overlap in the predicted direction". It is evaluated
on the 7,310 clips whose raw file shares packets with some other file; 6,706 of them have a directional packet
neighbour within k <= 8.

| cut | continuations | precision (direction) | precision (any shared packet) | recall | forward-only precision / recall |
|---|---|---|---|---|---|
| strict 3.31 | 12.5% | 0.993 | 0.993 | 0.135 | 0.996 / 0.077 |
| 5 | 19.5% | 0.990 | 0.990 | 0.211 | 0.987 / 0.125 |
| intermediate 16.5 | 63.9% | 0.936 | 0.939 | 0.652 | 0.886 / 0.397 |
| 30 | 87.6% | 0.911 | 0.919 | 0.870 | 0.752 / 0.543 |
| **primary 76.9** | 98.5% | **0.867** | 0.883 | **0.931** | 0.551 / 0.590 |

**Observations.**

- The frame argmin over both directions is a reliable temporal-neighbour detector. Even at the loose primary
  cut, 87% of predicted neighbours are packet-verified overlaps in the right direction, and recall is 93%.
- Forward-only detection, as specified in the brief, fails in `low`: precision 0.55 at the primary cut.

**Where frames and packets disagree** (primary cut; `disagreements.csv`).

- **839 clips**: the frames predict a continuation, but the predicted pair shares no packet although both files
  have packet data. The median m is 25.6, i.e. the same scene but not adjacent in time. They are spread over
  batches: h264 a4 277, h264 a2 171, mpeg4 291 (three batches); k* = 3: 235, k* = 2: 164.
- **117 clips**: the predicted pair shares packets but is a containment rather than a successor. These are the
  10-s/15-s re-cuts.
- **464 clips**: a packet neighbour exists but the frames pick another k. The packet junction lag for these is
  long (median 51 frames, p75 118): 10-s clips cut every 5 s, where last(N) and first(N+1) are 5 s apart.
- **391 readable clips** share no packet with any file (155 hevc, 71 h264 a4, 90 1064x598, 46 a2, ...). For
  them the frames are the only evidence.

## 5. Runs along the ID axis (A)

Full table: `runs_frames.csv` (272 runs); packet-based segmentation: `runs_packets.csv`.

**Observations.**

- At the primary cut the frame segmentation agrees with the packet segmentation on **89.5%** of the clips that
  have a packet period. At the intermediate cut the agreement is 64%; at the strict cut it is 3%, because 96%
  of clips are then "no continuation".
- Share of clips by majority period:

  | class | periods |
  |---|---|
  | high | +1 40%, +3 35%, +2 12.5%, +4 12% |
  | mid | +1 51%, +2 20%, +3 10%, +5 9%, +4 8% |
  | low | −1 47%, −2 28%, +1 9%, −4 7%, −6 5%, −3 4% |

- 64 runs of >= 15 clips cover 7,206 clips (93.6%).

**Figures.** `temporal_id_axis_{low,mid,high}.png` show, per class, the rolling median (window 15) of d_1, d_2,
d_3 (top panel) and b_1, b_2, b_3 (middle panel). The bottom strips give three things along the ID axis: the
frame period, the packet period, and the raw batch (codec, resolution, audio streams). Vertical lines mark raw
batches of >= 10 clips.

Runs of >= 50 clips:

- *cont.* = share of continuations;
- *k\*=period* = share of clips whose own k* equals the run period;
- *packets agree* = share whose packet period equals it;
- *stride / overlap* = median packet start offset and overlap of (N, N+|p|) pairs, in frames;
- *run type* comes from section 6.

| class | IDs | period | clips | cont. | k*=period | packets agree | top clusters (27.5) | train/val/test | raw batch (share) | raw batches | stride / overlap (fr) | run type |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | 1-54 | +2 | 54 | 1.00 | 0.50 | 0.00 | 16:100% | 45/4/5 | mpeg4 1092x614 a1 (100%) | 1 | 300 / 143 | recut |
| high | 59-138 | +4 | 80 | 1.00 | 0.91 | 0.95 | 48:25%; 35:25%; 16:24% | 64/8/8 | h264 1920x1080 a4 (99%) | 3 | 150 / 9 | cameras, audio-sync |
| high | 139-314 | +3 | 176 | 0.99 | 0.98 | 0.99 | 19:34%; 16:33%; 43:18% | 145/16/15 | h264 1920x1080 a4 (100%) | 1 | 150 / 9 | cameras, audio-sync |
| high | 493-588 | +3 | 96 | 0.99 | 0.93 | 0.96 | 25:32%; 35:31%; 37:18% | 80/6/10 | h264 1920x1080 a4 (96%) | 2 | 150 / 8 | cameras, audio-sync |
| high | 589-722 | +2 | 134 | 0.99 | 0.95 | 0.98 | 25:49%; 52:13%; 53:12% | 108/16/10 | h264 1920x1080 a4 (99%) | 2 | 150 / 8 | cameras, audio-sync |
| high | 723-805 | +1 | 83 | 0.98 | 0.71 | 0.58 | 21:59%; 11:20%; 13:16% | 63/9/11 | mpeg4 1280x720 a1 (59%) | 3 | 300 / 18 | |
| high | 806-917 | +3 | 112 | 1.00 | 0.96 | 0.98 | 0:33%; 33:32%; 12:24% | 86/11/15 | h264 1920x1080 a4 (98%) | 2 | 150 / 9 | cameras, audio-sync |
| high | 918-1020 | +2 | 103 | 0.98 | 0.95 | 0.98 | 0:50%; 5:27%; 31:11% | 86/10/7 | h264 1920x1080 a4 (100%) | 1 | 150 / 9 | cameras, audio-sync |
| high | 1021-1106 | +1 | 86 | 0.99 | 0.71 | 0.57 | 21:58%; 2:22%; 0:19% | 72/7/7 | mpeg4 1280x720 a1 (58%) | 3 | 300 / 18 | |
| high | 1107-1222 | +4 | 116 | 1.00 | 0.97 | 0.99 | 0:25%; 25:25%; 1:24% | 96/12/8 | h264 1920x1080 a4 (74%) | 59 | 150 / 9 | cameras, audio-sync |
| high | 1223-1396 | +3 | 174 | 1.00 | 0.94 | 0.79 | 0:34%; 29:33%; 25:30% | 139/20/15 | h264 1920x1080 a4 (67%) | 115 | 150 / 8 | cameras, audio-sync |
| high | 1397-1510 | +4 | 114 | 1.00 | 0.96 | 0.97 | 0:27%; 10:25%; 2:25% | 92/9/13 | h264 1920x1080 a4 (75%) | 57 | 150 / 7 | cameras, audio-sync |
| high | 1511-1768 | +3 | 258 | 1.00 | 0.94 | 0.75 | 0:34%; 2:31%; 5:21% | 201/29/28 | h264 1920x1080 a4 (67%) | 168 | 150 / 8 | cameras, audio-sync |
| high | 1769-2059 | +1 | 291 | 0.98 | 0.94 | 0.54 | 0:57%; 44:40%; 21:3% | 234/30/27 | h264 1920x1080 a4 (42%) | 10 | 150 / 8 | |
| high | 2080-2635 | +1 | 552 | 0.98 | 0.95 | 0.77 | 6:64%; 0:10%; 44:8% | 438/55/59 | h264 1920x1080 a4 (78%) | 11 | 150 / 8 | |
| low | 99-264 | -2 | 166 | 0.99 | 0.86 | 0.06 | 7:99%; 36:1% | 124/22/20 | h264 1920x1080 a4 (100%) | 1 | 150 / 8 | recut |
| low | 265-548 | -1 | 284 | 0.99 | 0.96 | 0.99 | 28:97%; 22:2%; 7:1% | 235/16/33 | h264 1920x1080 a4 (98%) | 2 | 150 / 8 | |
| low | 586-1318 | -1 | 733 | 0.99 | 0.94 | 1.00 | 6:63%; 1:17%; 26:14% | 569/73/91 | h264 1920x1080 a4 (99%) | 2 | 150 / 8 | |
| low | 1319-1406 | -2 | 88 | 0.99 | 0.94 | 0.98 | 1:49%; 67:15%; 60:14% | 70/9/9 | h264 1920x1080 a4 (100%) | 1 | 150 / 4 | cameras, no audio sync |
| low | 1407-1507 | -3 | 101 | 1.00 | 0.72 | 0.43 | 11:66%; 51:18%; 40:14% | 79/11/11 | h264 1920x1080 a4 (100%) | 1 | 150 / 15 | recut |
| low | 1508-1583 | -1 | 76 | 0.99 | 0.92 | 0.97 | 26:96%; 11:3%; 51:1% | 60/7/9 | h264 1920x1080 a4 (100%) | 1 | 150 / 7 | |
| low | 1584-1686 | -4 | 103 | 1.00 | 0.82 | 0.52 | 11:50%; 26:25%; 35:24% | 84/9/10 | h264 1920x1080 a4 (100%) | 1 | 150 / 12 | recut |
| low | 1687-1775 | -2 | 89 | 0.98 | 0.95 | 0.97 | 42:48%; 9:46%; 35:1% | 69/12/8 | h264 1920x1080 a4 (100%) | 1 | 150 / 8 | cameras, audio-sync |
| low | 1776-1850 | -4 | 75 | 1.00 | 0.95 | 0.96 | 23:25%; 9:24%; 54:24% | 64/3/8 | h264 1920x1080 a4 (75%) | 39 | 150 / 9 | cameras, audio-sync |
| low | 1852-1997 | -6 | 146 | 1.00 | 0.90 | 0.90 | 54:17%; 23:17%; 35:16% | 113/17/16 | h264 1920x1080 a4 (83%) | 50 | 150 / 9 | cameras, audio-sync |
| low | 2174-2364 | -2 | 191 | 0.99 | 0.52 | 0.45 | 36:94%; 17:4%; 14:2% | 151/26/14 | mpeg4 1064x598 a1 (47%) | 180 | 300 / 69 | recut |
| low | 2496-2595 | -1 | 100 | 0.99 | 0.92 | 0.99 | 47:90%; 4:10% | 78/12/10 | h264 1920x1080 a4 (90%) | 2 | 150 / 7 | |
| low | 2596-2645 | +1 | 50 | 0.94 | 0.92 | 0.96 | 26:52%; 6:44%; 18:4% | 40/6/4 | h264 1920x1080 a4 (96%) | 3 | 300 / 12 | |
| low | 2717-2864 | -2 | 148 | 0.99 | 0.97 | 0.99 | 32:51%; 3:49% | 120/19/9 | h264 1920x1080 a4 (100%) | 1 | 150 / 8 | cameras, audio-sync |
| low | 2931-3028 | -1 | 98 | 0.98 | 0.95 | 0.99 | 36:93%; 41:7% | 76/11/11 | h264 1920x1080 a4 (93%) | 2 | 150 / 8 | |
| mid | 2-203 | +1 | 202 | 0.99 | 0.96 | 0.99 | 6:95%; 35:5% | 162/17/23 | h264 1920x1080 a4 (95%) | 2 | 150 / 7 | |
| mid | 204-350 | +2 | 147 | 0.99 | 0.96 | 0.97 | 27:50%; 28:49%; 35:1% | 116/16/15 | h264 1920x1080 a4 (99%) | 2 | 150 / 8 | cameras, audio-sync |
| mid | 499-1145 | +1 | 647 | 0.98 | 0.58 | 0.61 | 20:23%; 8:16%; 10:11% | 542/60/45 | h264 1920x1080 a4 (97%) | 13 | 300 / 10 | |
| mid | 1232-1310 | +3 | 79 | 1.00 | 0.89 | 0.92 | 23:34%; 38:32%; 1:30% | 61/9/9 | h264 1920x1080 a4 (63%) | 54 | 150 / 5 | cameras, no audio sync |
| mid | 1321-1492 | +5 | 172 | 1.00 | 0.97 | 0.98 | 8:20%; 35:20%; 11:20% | 135/17/20 | h264 1920x1080 a4 (80%) | 70 | 150 / 7 | cameras, audio-sync |
| mid | 1493-1543 | +2 | 51 | 0.96 | 0.90 | 0.90 | 8:49%; 26:47%; 11:2% | 40/5/6 | h264 1920x1080 a4 (98%) | 3 | 150 / 8 | cameras, no audio sync |
| mid | 1593-1744 | +4 | 152 | 1.00 | 0.95 | 1.00 | 33:25%; 5:25%; 0:25% | 121/15/16 | h264 1920x1080 a4 (100%) | 1 | 150 / 10 | cameras, audio-sync |
| mid | 1831-1921 | +3 | 91 | 1.00 | 0.98 | 1.00 | 0:34%; 1:33%; 10:33% | 75/7/9 | h264 1920x1080 a4 (100%) | 1 | 150 / 6 | cameras, audio-sync |
| mid | 1922-2031 | +2 | 110 | 0.99 | 0.96 | 0.99 | 10:50%; 0:48%; 1:1% | 86/12/12 | h264 1920x1080 a4 (100%) | 1 | 150 / 9 | cameras, audio-sync |

Labels are not a column of the table: every run lies inside one class folder, so its label is constant.

**Relation to the raw-metadata batches.**

Observations:

- Several period boundaries coincide with raw-batch boundaries. Examples:
  - high 1-54 is the mpeg4 1092x614 batch;
  - the mpeg4 854x480 batch (high 328-496) contains the period-2/3 recut runs high 355-492;
  - mid 499-1145, period +1 with stride 300, is the batch of 10-s h264 clips.
- Several multi-camera runs span dozens of raw "batches": high 1107-1222 spans 59, 1223-1396 spans 115 and
  1511-1768 spans 168. In each of them one camera column has a different audio-stream count (a2 vs a4) from
  the others, so the raw combo alternates every p IDs. Per-column metadata is in `report_tables.md`; examples:
  - high 1107-1222, column 3: a2;
  - high 1223-1396, column 0: a2, 9-frame broken clips;
  - high 1511-1768, column 2: a2, 4-frame broken clips;
  - low 1776-1850 and 1852-1997, column 1: a2;
  - mid 1321-1492, column 1: a2.
- The 225 clips with first pts >= 1 s (`frac_first_pts_ge_1s`) are the series of `raw_audit_tables.md` b. In
  the runs table they appear inside high 139-333 (14-63% of clips), high 455-492, high 1769-2059, low 1-98,
  low 2174-2364 and low 2931-3028. They do not form run boundaries of their own.

Inference: many of the 1-clip raw "runs" in `raw_audit_id_runs.csv` are camera columns, not separate export
batches.

## 6. Interleaved partners = cameras? (B)

Partners are (N, N+j) with j < |p|, inside the 35 runs of >= 15 clips with |period| >= 2 (3,553 clips).
Same-source pairs are (N, N+|p|). The baselines are random same-class pairs, and random same-class pairs from
the same raw batch.

**(i) Labels.** Labels are the class folder, and IDs are numbered within a class, so partners always carry the
same label: 100% by construction. No packet is shared across classes either (0 of 16,037 pairs). The
informative substitute is whether the columns are distinct, stable sources. Each column of a run has its own
dominant scene cluster, often 95-100% pure (`report_tables.md`). For example:

- high 139-314: c43 / c19 / c16;
- mid 1593-1744: c33 / c5 / c0 / c11, each 100%.

The same column clusters recur across classes: c11, c35, c0 and c5 appear as columns in low, mid and high runs.

**(ii)-(iv) Pair statistics by run type** (pair-weighted; `partner_tests_by_run_typed.csv`). A run is "recut"
when its columns share video packets (>= 5 cross-column clip pairs).

| pairs | runs / clips | share packets | same cluster 27.5 | same cluster 17.5 | identical raw frame count | same first pts (1 ms) | first-first RMSE |
|---|---|---|---|---|---|---|---|
| partners, camera runs with audio sync | 20 / 2,450 | **0.4%** | **0.7%** | 0.4% | 5.0% | 19.4% | 99 |
| partners, camera runs without audio sync | 4 / 328 | 0.5% | 1.0% | 0.7% | 4.2% | 18.0% | 98 |
| partners, recut runs | 11 / 775 | 49.8% | 65.8% | 64.1% | 2.4% | 19.7% | 53 |
| same source (N, N+\|p\|), camera runs with audio sync | | **94.2%** | **96.6%** | | 21.3% | | 36-38 (all multi-stream runs) |
| random same-class, same raw batch (14,790) | | 0.16% | 9.7% | 4.2% | 3.2% | 10.3% | 58 |
| random same-class (18,000) | | 0.11% | 8.2% | 3.5% | 2.6% | 8.4% | 68 |

Lock-step durations: across the camera runs with audio sync, the share of aligned tuples (N, ..., N+p−1) whose
members all have the same raw frame count is:

| period | share of tuples (best phase) |
|---|---|
| 2 | 3.8% (4.6%) |
| 3 | 1.2% (2.1%) |
| 4 | 0.2% |

This is about the random level. In the triplet probe, IDs 141/142 and 144/145 had equal lengths, but that does
not generalise.

**Audio** (`04e`, `04f`, `temporal_audio_sync.png`). Each raw file has its own audio track: partner files share
no audio packet. Per-pair envelope correlation within ±1 s gives no evidence of synchrony (partner median 0.32,
random 0.33), apart from correlation ≈ 1 at lag 0 in the recut pairs.

The column test stitches each camera column into a continuous loudness envelope of up to 120 s, using the
packet strides. It then correlates column pairs over lags up to ±30 s. The null is 105 pairs of columns from
unrelated runs: median 0.17, p99 0.40, **max 0.41**.

| column pairs (110) | count |
|---|---|
| synchronous, different cameras (no shared packets) | **28** |
| synchronous, same camera (shared packets) | 10 |
| not synchronous | 72 |

- 20 of the 24 camera runs contain at least one synchronous different-camera pair; 16 do when the threshold is
  0.6.
- Synchronous scores reach 0.95 (e.g. high 1107-1222, columns 1-2-3: 0.95 / 0.92 / 0.92).
- The time offsets between synchronous camera columns are 0.1-24 s (quartiles 2.0 / 5.4 / 8.6 s).
- Positive control: two audio tracks of the same file correlate at 0.89-0.98 at lag 0.

**Visual check** (`temporal_triplets.png`).

- Period-3 run high 139-314: one group of students seen from three angles (a wide overhead camera and two side
  cameras); the same person in a black jacket with a pink print is visible in all three.
- Period-2 run high 589-722: one table from two angles (overhead and side).
- low 1319-1406 (period −2, no audio sync, no shared packets): two *different* groups seen from the same camera
  angle (scratch check; not in the figure).

**Conclusion (inference).**

- Interleaving by ID has three different causes.
  - **(a) Multi-camera recordings of one session (24 runs, 2,778 clips).** Each column is one camera:
    - consecutive same-camera clips share footage (94%);
    - partners never share footage and fall in different scene clusters (0.7% same cluster vs 9.7% for random
      same-batch pairs);
    - in 20 of 24 runs at least two columns carry the same sound at a fixed offset.
  - **(b) Re-cuts of one camera (11 runs, 775 clips).** The columns share packets: the same footage was cut at
    two lengths (10 s and 15 s in the mpeg4 batches) or with shifted windows, and interleaved by ID.
  - **(c) Possibly two different groups exported alternately** (low 1319-1406).
- The cameras were exported per 5-s step in an interleaved order, but **not cut in lock-step at the same
  instant**:
  - partner durations and first pts agree only at about the random rate;
  - the audio offsets between camera columns are seconds, up to 24 s.

  A partner clip is a view of the same session from about the same time, overlapping the clip by 1-6.6 s
  (median 3.2 s) when the offset is known.
- The "broken byte-identical clips at every third ID" are one whole camera column of a 3-camera run: column 0
  of high 1223-1396 (9 frames, a2) and column 2 of high 1511-1768 (4 frames, a2).

## 7. Chains (C)

**Construction.** Each clip gets at most one successor and one predecessor, and cycles are rejected.

1. **Packet links first.** Pairs with |ID diff| <= 8 and partial overlap, oriented by time and taken in
   ascending start offset.
2. **Frame links** for pairs without shared packets, in ascending d. If both files have packet data but share
   none, they do not overlap, so only an abutting, frame-exact junction (d <= 3.31) is accepted.

Containment pairs (re-cuts) are not links.

| | primary 76.9 | intermediate 16.5 | strict 3.31 |
|---|---|---|---|
| links: packet / frame (one file lacks packet data) / frame (both have packet data) | **6,478 / 379 / 2** | 6,478 / 355 / 2 | 6,478 / 130 / 2 |
| chains of >= 2 clips / singletons | 571 / 271 | 586 / 280 | 633 / 458 |
| share of clips in chains of >= 2 / >= 5 clips | 96.5% / **85.0%** | 96.4% / 84.5% | 94.1% / 80.5% |
| chains with both train and test clips | 186 | 190 | 191 |
| links whose two clips are in different splits | **33.9%** | 33.9% | 34.0% |
| expected if the split ignored time (1 − Σ p_split²) | 34.0% | | |

**Chain length.** Per chain (primary cut):

| chains of >= 2 clips | p25 | median | p75 | p90 | max |
|---|---|---|---|---|---|
| clips | 2 | 2 | 4.5 | 39 | 276 |
| span (s) | 16 | 18 | 40 | 260 | 1,399 (23.3 min; low 267-542, `temporal_chain.png`) |

Seen from a random clip (clip-weighted, singletons included):

| | p25 | median | p75 | p90 |
|---|---|---|---|---|
| clips in its chain | 25 | **72** | 97 | 118 |
| span (s) | 150 | **368** | 472 | 584 |

The span counts packet offsets for packet links and the clip duration for frame links. The sum of clip
durations (overlaps counted twice) has a clip-weighted median of 388 s.

**Gap at the junction.**

- Packet links: the junction lag is a median of **8 frames** (p5 1, p95 60), with an overlap of 9 frames.
  Strides: 150 frames for 61% of links, 300 for 19%, 60 for 5%.
- The RMSE at a packet junction has a median of 11.9, which is 4.3x the raw consecutive-frame step (2.79) and
  5.4x the same-frame floor (2.22).
- 335 junctions are frame-exact (lag 0, RMSE 2.2).
- Frame links: median d 4.2 (p95 41.8), mostly in hevc and 1064x598 batches without packets.

**Inference.** Consecutive segments are not frame-contiguous at the cut. They overlap by about 0.3 s, so a
segment begins about 8 frames before its predecessor ends.

**Label runs.**

- Observation: label changes along the 6,859 links: **0**, by construction, because every chain stays inside one
  class folder.
- To look for label changes in time, every chain tail was matched to the nearest chain head in *any* class
  (32x32 search, then verified at 128x72).
  - At d <= 16.5: 189 tail→head links, **none across classes** (same-class ID gap median 9).
  - At d <= 3.31: 40 links, none across classes.
  - Only at the loose 76.9 cut do 41 cross-class links appear (the same-scene regime, not continuations).
- Inference: no recording can be followed across a class folder at the frame level. Label runs last at least as
  long as the chains: half of all clips sit in a constant-label stretch of >= 6 min, and a quarter in one of
  >= 7.9 min. The same camera views do recur in other classes (clusters 11, 35, 0, 5), at times that cannot be
  linked.

**Split along chains.**

- 33.9% of consecutive pairs are in different splits, identical to the 34.0% expected if the split ignored time.
- 186 of 571 chains contain both train and test clips.
- Inference: the published split was drawn independently of time.

## 8. Leak figures (D4-D7)

| figure | shows |
|---|---|
| `leak_nn_pairs_hires.png` | 8 random test clips (seed 0) beside their nearest train clip, at 128x72 |
| `leak_clusters_overview.png` | medoid first frame, size, label mix and split mix of the 20 largest clusters at 27.5 |
| `leak_tsne.png` | PCA-50 → t-SNE of the centre thumbnails, coloured by cluster, label and split |
| `leak_distance_hist.png` | test→nearest-train distance under four splits, plus reference pair populations |

Notes on the leak figures:

- **Nearest train clip.** In `leak_nn_pairs_hires.png`, 5 of the 8 pairs are 1-2 IDs apart and share packets.
  Two are more than 1,000 IDs apart and still share packets: the same footage cut twice. One (test high 222,
  train high 50, d = 1.3) is the same footage in an h264 and an mpeg4 batch, with no shared packet: a
  re-encoded copy that the packet analysis cannot see.
- **Re-encoded copies, overall.** 43 test clips have a near-identical (d <= 2) train clip without a shared
  packet; 16 of them are across codec batches.
- **t-SNE.** Labels form islands that follow scenes, and train/val/test are completely interleaved within every
  island.
- **Distance histogram**, test → nearest train (32x32 clip distance; exact float64 check against the review:
  max deviation 0.07):

  | split | median | share <= 10 | <= 17.5 | <= 27.5 | <= 35 |
  |---|---|---|---|---|---|
  | published | **2.7** | 84.1% | 96.0% | 98.7% | 99.9% |
  | grp17 | 31.3 | 0 | 0 | 0 | |
  | grp27b | 38.9 | 0 | 0 | 0 | 14.4% |
  | grp27a | 65.6 | 0 | 0 | 0 | 14.8% |

  120 published test clips (15.6%) share an *identical* 32x32 thumbnail frame with a train clip (exact d = 0,
  `compare_03b_exact_nn.txt`). The October doc's floor of 0.4 was float32 cdist noise. The legend's 13% counts
  d < 0.05 in float32.
- **Reference populations** (32x32 clip distance): adjacent IDs median 21.4 (25% <= 4.4, 5% exactly 0); random
  same-class 86.3; random cross-class 86.9 (p1 35.0).

## 9. Answers in plain terms

**Are consecutive clips near-identical?** Mostly yes, but only once IDs are read correctly.

- Observations:
  - Clips have a temporal neighbour within 1-8 IDs: for 87% of clips (6,706) the packets prove it, and 98% of
    clips have a frame neighbour in the same scene.
  - The neighbour is often *not* N+1. In `low` it is usually N−1 (IDs run backwards); in multi-camera runs it
    is N±p with p up to 6.
  - Consecutive segments overlap by about 9 frames (0.3 s). The frame at the junction is 4x the
    consecutive-frame noise away from its predecessor (median 11.9 vs 2.8), and exactly the same frame in 335
    junctions.
  - Within a clip, first vs last differs by 36.8, so neighbours differ from each other by less than a clip does
    from itself.

**How long are label runs in time?**

- Observation: a label never changes along a chain.
- Inference: for a random clip, the constant-label stretch it belongs to spans a median of about 6 min (368 s;
  IQR 2.5-7.9 min) and up to 23 min. No frame-level continuation crosses a class folder, so where a label
  changes inside a session cannot be located from this data.

**Does the random split put temporal neighbours and simultaneous views on both sides?**

- Observation: yes, at the chance rate. 33.9% of temporal links and 32.9% of simultaneous-view pairs cross
  splits.
- Inference: the split was drawn independently of time and camera.

**What fraction of test clips has a temporal neighbour or a simultaneous view in train?** Of the 771 test clips:

| test clips with ... in train | share |
|---|---|
| a temporal neighbour (chain predecessor or successor) | **90.7%** (low 91.1%, mid 92.1%, high 89.1%) |
| an audio-confirmed simultaneous view from another camera (overlap >= 1 s) | **20.4%** (17.3% at audio r > 0.6) |
| any interleaved partner, structural upper bound | 45.3% |
| a temporal neighbour or a simultaneous view | 91.8% |
| shared encoded frames (packets) | **91.8%** |
| all footage in train | **32.6%** (251 clips; >= 50% of footage: 43.8%; 49.0% of all test packets occur in train) |
| any of the three (temporal neighbour, simultaneous view, shared packets) | **97.8%** (only 17 test clips have none) |

21.4% of test clips have a simultaneous view at all. Validation looks the same: 92.2% share footage with train
and 35.5% are entirely in train.

## 10. What contradicts or corrects `05_id_structure_probe.md`

1. **The floor was wrong.** The probe's "consecutive-frame floor 0.2-0.5" was the ffmpeg padding duplicate at
   the start of the 320p copies (frames 0-2 are the same raw frame). The real step between consecutive frames
   is 2.8 at 128x72, and the same raw frame in two clips gives 2.2. With the right floor, frame-exact
   continuations do exist: 335 packet-verified junctions at lag 0, and the 2.3 mode of the histogram. So
   "nowhere does last(N) → first(N+k) reach the floor" no longer holds.
2. **`low` 500-560 is not "a single source with a gap between segments".** IDs run *backwards* in time there:
   N+1 starts 150 frames before N, and 79% of |ID diff| = 1 packet pairs in `low` run backwards. The
   continuation is last(N+1) → first(N), which the forward-only probe could not measure. The segments overlap by
   about 8 frames; they have no gap.
3. **"Three cameras exported in lock-step" holds only in part.**
   - Confirmed:
     - three- and more-camera interleaving exists (periods 3, 4, 5, 6 as well as 2);
     - the every-third-ID broken clips are one camera column;
     - the columns are cameras of one session (no shared footage, distinct viewpoints, synchronous audio).
   - Not confirmed: "lock-step". Partner clips are not cut at the same instant. Durations and first pts agree
     only at the random rate, and the camera timelines are offset by 0.1-24 s (median 5.4 s).
4. **Not all two-way alternation is cameras.**
   - 11 runs (775 clips) alternate two cuts of the *same* camera footage. Examples: high 1-54 (10-s and 15-s
     cuts, one contained in the other), the mpeg4 854x480 runs high 355-492, and low 99-264 and 2174-2364.
   - At least one run (low 1319-1406) alternates two different groups.
5. **Periods beyond 3** (4, 5 and 6) occur in 8 runs with 958 clips.

## Files

- `outputs/research/temporal/`:
  - `succ_dist.csv`: d_k, b_k, f_k RGB and gray per clip;
  - `ref_frame_lags.csv`, `ref_pairs.csv`: reference scales;
  - `packet_pairs_offsets.csv`: exact offsets for all 16,037 sharing pairs;
  - `clip_temporal.csv`: one row per clip with k*, dir*, m, flags at the 3 cuts, period labels, runs, chains,
    test packet coverage;
  - `runs_frames.csv`, `runs_packets.csv`;
  - `pr_sweep.csv`, `disagreements.csv`;
  - `packet_offsets_by_iddiff.csv`, `packet_offsets_by_period.csv`;
  - `partner_tests.csv`, `partner_tests_by_run.csv`, `partner_tests_by_run_typed.csv`;
  - `chains.csv`, `chain_links.csv`, `superchain_links.csv`;
  - `audio_sync_pairs.csv`, `audio_sync_summary.json`, `audio_columns.csv`, `audio_null.csv`,
    `simultaneous_views.csv`, `audio_columns_summary.json`;
  - `leak_nn_by_split.csv`, `leak_figures_summary.json`, `tsne_xy.npy`;
  - `summary.json`, `report_tables.md`;
  - logs `04*.log`.
- `docs/research/figures/`:
  - `temporal_id_axis_{low,mid,high}.png`, `temporal_triplets.png`, `temporal_chain.png`;
  - `temporal_min_d_hist.png`, `temporal_audio_sync.png`;
  - `leak_nn_pairs_hires.png`, `leak_clusters_overview.png`, `leak_tsne.png`, `leak_distance_hist.png`.
