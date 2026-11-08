# Raw OUC-CGE release: header audit

Record of a subagent run (Claude Opus 5.5, 2026-11-07, report written 2026-11-08 after the coordinating session
crashed; no number was recomputed for the report). The brief was a header audit of every raw file of the release
(`data/ouc-cge/videos/{high,mid,low}/viewN.mp4`, the project's DVC copy of the OSF data). It followed up two
earlier findings: the release is heterogeneous, and container timestamps carry information (see
`02_authors_code_audit.md`, section 4 and "Verification on the released raw clips"). Section f (shared encoded
frames) goes beyond the brief and was added because the first-PTS analysis pointed to it. Each section separates
observations from inferences; the closing list does so again.

All tables below are copied from `outputs/research/raw_audit_tables.md` (generated, 874 lines), which holds the
full versions. Per-file data: `outputs/research/raw_audit_joined.csv`.

## Headline

- **The release is not what the paper describes.** It mixes 3 codecs (h264 6674, mpeg4 844, hevc 183), 5 frame
  sizes and 0, 1, 2 or 4 audio streams. Only 355 of 7701 readable clips are 1280x720; 6886 are 1920x1080. Video
  totals **13.910 h** (13 h 54.6 min) by header against the paper's 12 h 50 min. The median clip is **5.40 s**;
  1167 clips (15.2%) are 9.5-10.5 s.
- **Broken and duplicate files.** 4 files are unreadable (zero-filled). 115 files hold <= 5 video frames and 155
  hold < 1 s. 299 files form 128 groups of byte-identical files: 171 redundant copies, and **26 test files are
  byte-identical to a train file**.
- **Frame-level overlap between splits (section f).** Every video packet was hashed without decoding.
  **708 of 771 test files (91.8%) contain at least one encoded frame that also occurs, byte for byte, in a train
  file.** 251 test files (32.6%) are fully contained in train, and 49.0% of all test frames (packets >= 1 KB) occur
  in train. Five sampled shared keyframes decode to pixel-identical images in both files. No frame is shared
  across classes.
- **Two segmentation scales.** Most clips are stream-copy cuts of longer recordings, segmented on a ~5 s and a
  ~10 s grid; neighbouring cuts overlap by a few frames, and many ~5 s clips lie entirely inside a ~10 s clip of
  the same recording. Distinct footage is at most **9.05 h** (977,626 distinct encoded frames of 1,477,863).
- **Container metadata is a weak class predictor** (5-fold tree 47.5% vs. majority 39.3%) and is independent of
  the split (chi-square p >= 0.14). Some values, however, are pure: all 183 HEVC clips and all 164 clips with
  44.1 kHz audio are `high`; all 180 clips at 1064x598 are `low`.
- **320p copies.** 241 of them are longer than the raw stream by > 1 s, and for 7487/7701 clips the excess equals
  the raw first PTS within one frame. The 320p total is 23.184 h against 13.910 h raw.

## Method

**Files.** 7706 files under `data/ouc-cge/videos/`: 7705 mp4 (low 3028, mid 2041, high 2636) and one
`.DS_Store`. The DVC pull finished at 21:32:32 (log `.../scratchpad/dvc_pull_raw.log`: "1 file modified and 7530
files fetched"). Files were opened read-only; nothing under `data/` was written.

**Header pass** (`scripts/research/raw_header_audit.py`; PyAV 17.0.1, `Pool(8)`, `imap`, incremental JSON lines).
Per file it records:
- file size;
- container format, duration, bit rate and all `container.metadata` tags;
- for every stream, its metadata tags;
- for the video stream: codec, profile, size, pix_fmt, average/guessed/base rate, time_base, header frame count,
  duration in ticks and seconds, bit rate, start_time;
- for each audio stream: codec, profile, sample rate, layout, channels, frames and duration;
- for the first 16 video packets (demux only): pts, dts, keyframe flag, the pts step and whether pts are
  reordered.

Nothing is decoded except clips whose header shows <= 5 frames, < 1 s of video or an average rate < 5 fps. Those
**159 files** were decoded fully, one frame in memory at a time, to count real frames and record the first and last
decoded pts. The < 5 fps condition was added after the first run showed clips whose few frames are spread over
minutes. Exceptions are recorded in an `error` column. Output: `outputs/research/raw_header_audit.csv` (7705 rows,
132 columns) and `.jsonl`.

**md5 pass** (`--md5`): md5 of the file bytes of all 7705 files, written to `outputs/research/raw_md5.csv`. It
equals DVC's recorded md5 (`outputs/research/raw_videos_dvc_dir.json`) for **7705/7705** files.

**Packet pass** (`--packets`, beyond the brief): demuxes every video packet of every readable file without
decoding it. It stores pts, dts, size, keyframe flag and a 64-bit blake2b hash of the packet bytes in
`outputs/research/raw_video_packets.npz` (1,477,863 packets).

**Analysis** (`scripts/research/raw_header_audit_tables.py`; pandas, scipy, sklearn): writes
`outputs/research/raw_audit_tables.md`, the CSVs listed at the end, and the figures in `docs/research/figures/raw_*.png`.
The only decoding in the analysis is the pixel check in section f: one shared keyframe in each of 5 file pairs.
The 320p numbers come from the other agent's `outputs/research/clip_audit.csv` (frames counted by decoding with
cv2).

**Wall times** (8 processes, NVMe):

| pass | files | wall |
|---|---|---|
| header (first run) | 7705 | 50.1 s (52.1 s incl. start-up) |
| header (rerun with the < 5 fps decode condition; the numbers here) | 7705 | 48.1 s (50.0 s) |
| md5 of all files, 28.98 GB | 7705 | 24.3 s (about 1.2 GB/s) |
| packet hashes | 7705 | 66.7 s |
| tables and figures (incl. pixel check) | - | 60-150 s |

**Durations.** "Duration" means the video stream's header duration (`stream.duration * time_base`, i.e. the sum
of the frame durations), unless stated otherwise. Every readable stream has `time_base = 1/15360`, so one frame at
30 fps is 512 ticks. For the broken clips of section d the header duration understates the pts span: a 9-frame
clip lasts "0.3 s" but its frames sit at 0.074-0.274 s and 175.71-175.74 s.

## 0. Inventory and unreadable files

- Readable by PyAV: 7701; unreadable: 4.
- Manifests: 7705 labelled rows (train 6164, val 770, test 771) plus one unlabelled row, `videos/anonymity/a.mp4`
  (test.csv line 772).
- Every raw file has exactly one labelled row, and the class directory always equals the label.

| file | split | bytes | non-zero bytes | content |
|---|---|---|---|---|
| high/view2521.mp4 | train | 4,498,060 | 0 | all zero |
| high/view2632.mp4 | train | 4,827,243 | 0 | all zero |
| high/view2630.mp4 | train | 3,170,251 | 510,154 | valid `ftyp`/`mdat` start; zero from byte 524,288 on, no `moov` |
| high/view2631.mp4 | train | 3,605,167 | 509,103 | same; zero from byte 524,288 on |

These 4 are the 4 clips without a 320p copy. Their md5 equals DVC's record, so they were already corrupt when the
student added the data to DVC. Whether the OSF originals are also corrupt cannot be told from here.

## a. Distributions (7701 readable files)

**Constant everywhere:**
- container: mov/mp4, `major_brand isom`, `minor_version 512`, `handler_name VideoHandler`/`SoundHandler`;
- video: pix_fmt yuv420p, time_base 1/15360, guessed and base rate 30;
- audio: AAC (whenever present), stereo;
- the first packet is a keyframe in every file.

No container carries a `creation_time` tag. A single file, `high/view2135.mp4`, has a stream-level
`creation_time`: its streams are video and data, and the data stream has handler_name `SkundHanflEr` and
creation_time 1972-03-11.

**Varying properties:**

| property | values (clips) |
|---|---|
| video codec | h264 6674; mpeg4 844; hevc 183 |
| profile | High 6371; Simple 844; Constrained Baseline 303; Main 183 |
| resolution | 1920x1080 6886; **1280x720 355**; 854x480 224; 1064x598 180; 1092x614 56 |
| audio streams | 4: 6033; 1: 1061; 2: 551; 0: 56 |
| audio[0] sample rate | 48 kHz 7481; 44.1 kHz 164; none 56 |
| container `encoder` tag | Lavf58.47.100 7518; Lavf61.7.100 182; none 1 |
| video-stream `encoder` tag | none 7518; `Lavc61.19.100 hevc_qsv` 183 |
| compatible_brands | isomiso2avc1iso6mp41 6468; isomiso2mp41 1027; isomiso2avc1mp41 206 |
| average rate (frames / duration) | exactly 30: 6903; 30-30.001: 344; 29-30: 125; 5-29: 325; < 5: 4 |

By class:

| | low | mid | high |
|---|---|---|---|
| h264 / mpeg4 / hevc | 2541 / 487 / 0 | 2012 / 29 / 0 | 2121 / 328 / 183 |
| 1920x1080 / 1280x720 / 854x480 / 1064x598 / 1092x614 | 2550 / 243 / 55 / 180 / 0 | 2041 / 0 / 0 / 0 / 0 | 2295 / 112 / 169 / 0 / 56 |
| audio streams 4 / 2 / 1 / 0 | 2373 / 78 / 540 / 37 | 1887 / 125 / 29 / 0 | 1773 / 348 / 492 / 19 |

The batch combo (codec | size | audio streams | encoder tag) takes 13 values. 6033 clips are
`h264|1920x1080|a4|Lavf58.47.100`; the full table is in `raw_audit_tables.md`, section a.

**File size.** 28.977 GB in total. Per clip: median 3.195 MB, p1 1.027 MB, p99 11.62 MB, max 16.79 MB.

**Video duration (s), header:**

| | n | min | p1 | p5 | p25 | p50 | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 7701 | 0.0667 | 0.133 | 2.40 | 5.17 | 5.40 | 7.50 | 10.67 | 14.5 | 323.2 | 6.502 |
| low | 3028 | 0.367 | 2.07 | 2.27 | 5.17 | 5.37 | 5.83 | 11.13 | 13.82 | 15.13 | 6.06 |
| mid | 2041 | 4.5 | 5.03 | 5.07 | 5.23 | 5.47 | 10.2 | 10.57 | 11.07 | 15.23 | 7.04 |
| high | 2632 | 0.0667 | 0.0667 | 0.3 | 5.17 | 5.37 | 9.77 | 10.87 | 14.87 | 323.2 | 6.60 |

**Duration counts:**

| duration | clips |
|---|---|
| < 1 s | 155 |
| < 5 s | 622 |
| 5 to < 9.5 s | 5236 |
| 9.5-10.5 s | **1167** |
| > 10.5 s | 676 |
| > 30 s | 3 |
| > 60 s | 2 |
| exactly 300 frames | 167 |

**Totals against the paper:**
- Header durations add up to 50,074.2 s = **13.910 h**; frame counts / 30 give 13.686 h. The paper states
  12 h 50 min (12.833 h, i.e. 5.996 s per clip) and "about 10 seconds" per clip (7705 x 10 s would be 21.40 h).
- Container durations add up to 26.93 h. They are inflated by pts offsets and by the audio of the broken clips,
  so they are not a content measure.

**Audio:**
- 56 clips have no audio stream.
- 92 clips carry a single AAC frame (0.0213 s) of audio; these are the clips with a large first PTS (section b).
- 349 clips have audio[0] and video differing by > 1 s.

**Frame rate.** All clips are nominally 30 fps. 454 clips have an average rate below 30, i.e. timestamp gaps
(variable frame rate). In 4 of them the average rate is below 5 fps: 131, 81, 42 and 83 frames spread over 180 s,
323 s, 1.4 s (starting at 320.6 s) and 47 s (section d).

Figure: `figures/raw_duration_hist.png`.

## b. First-PTS structure

The first video packet in demux order is always a keyframe. Its pts equals the stream `start_time` and the
smallest pts of the first 16 packets in 7701/7701 files.

| first pts | low | mid | high | total |
|---|---|---|---|---|
| = 0 | 9 | 1 | 29 | 39 |
| (0, 0.1) s | 2920 | 2036 | 2461 | **7417** |
| [0.1, 1) s | 8 | 4 | 8 | 20 |
| >= 1 s | 91 | 0 | 134 | **225** |

- Codec mix of the 39 clips starting at exactly 0: 22 hevc and 17 mpeg4. h264 clips never start at 0.
- In the first packet, pts - dts is 1024 ticks (a two-frame B-frame delay) in 6777 clips and 0 in 844 (mpeg4).
- 58 clips have pts - dts = 4,923,387 ticks; they are the broken 2-frame family and `view1888`, section d.

**Clips with first pts >= 1 s (225).** These are offsets into longer recordings, on regular grids. The table
groups them into series of the same class, batch combo and sub-second start offset (full table:
`raw_audit_tables.md`; per clip: `outputs/research/raw_audit_pts_series.csv`):

| series (class, codec, size) | clips | IDs | start times | grid step | clips per start | median length | next start - end | Spearman(ID, pts) |
|---|---|---|---|---|---|---|---|---|
| high h264 1920x1080 a4 | 37 | 240-327 (every 3rd up to 315, then consecutive) | 2.35-184.3 s | 5.0 s (32x), 5.5 s (4x) | 1 | 5.33 s | -0.27 s | **+1** |
| low h264 1920x1080 a4 | 37 | 2938-2974 | 1.61-184.1 s | 5.0 s (31x), 5.5 s (5x) | 1 | 5.33 s | -0.27 s | **-1** |
| high mpeg4 854x480 | 36 | 461-496 | 4.77-174.8 s | 10.0 s | **2** | 11.3 s | -3.8 s | +0.999 |
| low mpeg4 1064x598 | 35 | 2182-2250 (even IDs) | 8.04-178.0 s | 10.0 s | 1-2 | 10.1 s | -3.5 s | -0.998 |
| low h264 1920x1080 a4 | 14 | 693, 881, 3, 4, 42-44, 52-57, 67 | 4.61-134.6 s | 10.0 s | 1 | 10.1 s | -0.1 s | +0.26 |
| high h264 1920x1080 a2 | 58 | 1831-1888 | all 320.55 s | - | 58 | 0.067 s | - | - |

- Observed: within a series, start times step by exactly 5 or 10 s with a fixed sub-second offset.
- Observed: successive clips overlap in source time (negative "next start - end").
- Observed: in the two mpeg4 series, the clips come in pairs that share a start time (to within 1 ms) and differ
  in length (high: about 8 s and 13 s; low: about 6 s and 11 s). The shorter clip of a pair is a byte-exact prefix
  of the longer one (section f: `high/view461` is the first 241 packets of `view462`).
- Observed: in these series, IDs run forwards or backwards in source time (Spearman +1 or -1), or out of order
  (the third low series starts at IDs 693 and 881, then continues at 3, 4, 42 ...).
- Observed: in `high/236-250`, every third ID has a large first PTS and 0.02 s of audio, while the two IDs in
  between start near 0 with full audio.
- Observed: the low 5 s series (2938-2974) lies inside the low 10 s series (3 ... 881). Each 5 s clip is either
  contained in one 10 s clip or split across two (shared packets), and both series form one component of section f.
- Inference: these 225 clips kept the timestamps of the recording they were cut from. The other clips were cut
  the same way but had their timestamps rebased, so their offsets are lost; section f recovers their overlaps
  from the packet bytes instead.

**Relation to clip ID.** The first PTS hardly correlates with ID over a whole class (Spearman: low -0.217, mid
0.008, p = 0.72; high 0.081; all -0.097), because 97% of clips start below 0.1 s. Batch membership, by
contrast, runs strongly along the ID axis. Fraction of ID pairs k apart with identical batch combo (codec, size,
audio-stream count, encoder tag), against a within-class permutation baseline (1000 draws; no draw reached the
observed value, p = 0.001 for all):

| class | lag 1 | lag 2 | lag 3 | permutation mean |
|---|---|---|---|---|
| low | 0.901 | 0.950 | 0.888 | 0.624 |
| mid | 0.895 | 0.899 | 0.915 | 0.859 |
| high | 0.836 | 0.826 | **0.921** | 0.479 |

- IDs run 1..n in each class with no gaps.
- Runs of identical combo: low 302 runs (longest 1184); mid 216 (longest 543); high 433 (longest 225).
- Runs of length >= 50 cover 2295/3028 low, 1493/2041 mid and 1668/2636 high clips.
- Full list: `outputs/research/raw_audit_id_runs.csv`.

Figures: `figures/raw_first_pts_hist.png`; `figures/raw_first_pts_vs_id.png` (first PTS against ID per class, with
codec, size and audio-count strips underneath).

## c. Container metadata x label and x split

Readable, labelled clips: n = 7701; majority class (low) 39.3%.

| property | values | chi2 p (class) | Cramér's V (class) | chi2 p (split) | Cramér's V (split) | majority-rule acc. |
|---|---|---|---|---|---|---|
| v_codec | 3 | 9.7e-139 | 0.205 | 0.507 | 0.015 | 0.417 |
| v_profile | 4 | 8.1e-153 | 0.217 | 0.584 | 0.018 | 0.417 |
| resolution | 5 | 2.2e-162 | 0.225 | 0.863 | 0.016 | 0.415 |
| n_audio | 4 | 8.2e-138 | 0.206 | 0.249 | 0.023 | 0.428 |
| container encoder | 3 | 7.3e-77 | 0.153 | 0.535 | 0.014 | 0.417 |
| video encoder | 2 | 4.0e-79 | 0.217 | 0.249 | 0.019 | 0.417 |
| compatible_brands | 3 | 1.1e-99 | 0.174 | 0.912 | 0.008 | 0.400 |
| average-rate bin | 5 | 1.3e-15 | 0.076 | 0.180 | 0.027 | 0.399 |
| audio[0] sample rate | 3 | 4.8e-74 | 0.150 | 0.142 | 0.021 | 0.414 |
| batch combo | 13 | 1.3e-304 | 0.313 | 0.561 | 0.038 | 0.474 |

pix_fmt is constant, and `creation_time` is absent from every container, so neither can be tested.

**Class-pure values** (>= 95% one class, n >= 20; all are 100%):

| value | clips | class | share of that class |
|---|---|---|---|
| hevc / Main profile / `Lavc61.19.100 hevc_qsv` | 183 | high | 7.0% |
| encoder tag Lavf61.7.100 | 182 | high | 6.9% |
| audio 44.1 kHz | 164 | high | 6.2% |
| 1064x598 | 180 | low | 5.9% |
| 1092x614 | 56 | high | 2.1% |

Mid is 1920x1080 / 48 kHz throughout.

**Predicting class from metadata.** A `DecisionTreeClassifier(max_depth=4)` on one-hot categorical metadata (11
properties, 35 columns), 5-fold `StratifiedKFold(shuffle=True, random_state=0)`:

| model | accuracy |
|---|---|
| majority baseline | 0.393 |
| tree, categorical metadata | **0.475** (folds 0.466-0.484) |
| tree, + numeric header values (duration, first pts, size, bit rate, audio duration) | 0.533 |
| official split, train -> test, tree, categorical | 0.512 (majority 0.422) |
| official split, train -> test, tree, + numeric | 0.537 |

- Observed: no property alone identifies a class with high accuracy. The best single property, the 13-value
  batch combo, reaches 47.4% in sample. About 7% of `high` and 6% of `low` clips carry a pure value.
- Observed: no property depends on the split (all p >= 0.14), as expected of a random clip-level split.
- Inference: a model never sees the container, so none of this is a shortcut a model could use. It does show that
  parts of `high` (the HEVC / Lavf61 batch, the 44.1 kHz audio, 1092x614) and of `low` (1064x598) were exported
  separately from the rest.

## d. Broken files and exact duplicates

**Real frame count <= 5: 115 files**, all in `high`. With < 1 s of video: 155. Add the 4 unreadable files. In all
159 decoded files, the decoded frame count equals the header frame count and the packet count.

| family | files | IDs | frames | where the frames sit (pts) | audio[0] | identical bytes | split |
|---|---|---|---|---|---|---|---|
| high, 2 frames | 57 | 1831-1887 (consecutive) | 2 | 320.551, 320.584 s | 34.41 s growing to 205.95 s by 5.035 s per ID; last 22 equal | 22 identical (1866-1887); 36 distinct md5 | train 53, val 3, test 1 |
| high, 4 frames | 58 | 1591-1762 (every 3rd ID) | 4 | 0.070, 0.103, 320.603, 320.703 s | 32.0 -> 214.85 s by 5.01 s per member; last 21 equal | 21 identical (1702-1762); 38 distinct | train 48, test 7, val 3 |
| high, 9 frames | 30 | 1304-1391 (every 3rd ID) | 9 | seven at 0.074-0.274 s, two at 175.707/175.740 s | 28.46 -> 149.12 s by 5.06 s; last 6 equal | 6 identical (1376-1391); 25 distinct | train 25, val 4, test 1 |
| low, 11 frames | 10 | 13, 14, 33, 35-41 | 11 | nine at 0.076-0.343 s, two at 137.709/137.743 s | 137.5 -> 47.6 s, falling by 10.0 s per ID | 10 distinct | train 9, val 1 |

- Observed: consecutive members of a family share the same first pts. The 2-frame family starts at 4,923,663
  ticks (320.55 s) with a dts of 276 ticks, so the offset lives in the composition offset (pts - dts = 4,923,387),
  not in the dts.
- Observed: the next member after three of these families holds the missing frames, sparse in time:
  - `high/view1888`: 42 frames from 320.55 s;
  - `high/view1394`: 131 frames over 180 s;
  - `high/view1765`: 81 frames over 323 s.
- Observed: the files are 1.06-6.69 MB, almost entirely audio.
- Full lists: `outputs/research/raw_audit_broken.csv` and `raw_audit_tables.md`, section d.
- Figure: `figures/raw_broken_audio.png`.

**Exact duplicates** (md5 of file bytes, all 7705 files):
- **299 files in 128 groups** (171 redundant copies; 7534 distinct files). Group sizes: 125 pairs plus one group
  each of 22, 21 and 6, the three broken high families above.
- By class: low 102 groups, mid 23, high 3. No group spans two classes.
- **low/1060-1131 = low/1512-1583**: 72 pairs at ID + 452.
- low/1132-1158 = 27 IDs between 1586 and 1689: the offset grows 454, 457, ..., 531.
- low pairs 134/135, 1405/1406 and 1511/2001.
- mid/818-840 = mid/841-863 at ID + 22..24: 19 pairs.
- mid/423-429 = 436-443: 4 pairs.
- 46 groups span more than one split: **26 test files are byte-identical to a train file**, 20 val files to a train
  file, and 3 test files to a val file.
- Of the 299 files, 43 are broken (<= 5 frames) and 250 have >= 30 frames.
- Full list: `outputs/research/raw_audit_duplicate_groups.csv`.

## e. Raw vs 320p copies

Joined on class + basename with `outputs/research/clip_audit.csv`. 7701 raw clips have a 320p copy; the 4 without
one are the unreadable files.

- 320p minus raw duration: median 0.067 s, p95 0.1 s, p99 158.6 s, max 320.6 s.
- **320p longer than raw by > 1 s: 241 clips**; by > 0.1 s: 763; within one frame: 1127.
- 320p frames minus raw frames: +2 in 4904 clips; +3 in 1000; 0 in 758; +1 in 361; +9617 in 58.
- **The 320p excess equals the raw first PTS within one frame for 7487/7701 clips.** Among the 241 clips with
  > 1 s excess, Pearson r(excess, raw first PTS) = 0.99964 and the median |excess - first PTS| is 0.0066 s.
- Of the 214 exceptions, most received no start padding; a few gained extra frames (up to +5.78 s).
- Totals: raw 13.910 h, 320p 23.184 h (+9.274 h). Clips over 60 s: 175 in 320p, 2 in raw.

Largest differences (full top 20 in `raw_audit_tables.md`):

| clip | raw frames | raw duration | raw first pts | 320p frames | 320p duration | difference |
|---|---|---|---|---|---|---|
| high/view1831..1887 (57 files, all equal) | 2 | 0.067 s | 320.55 s | 9619 | 320.63 s | +320.6 s |
| high/view1765 | 81 | 323.2 s | 0.070 s | **4** | 0.133 s | **-323.1 s** |
| high/view1394 | 131 | 180.1 s | 0.074 s | **9** | 0.300 s | **-179.8 s** |

- Observed: the transcode in `scripts/preprocess_resize.py` (ffmpeg, `-an`, no `-vsync`) filled the span from 0
  to the first PTS with repeated frames. This is the source of the 320p audit's long clips and of its 23.2 h total.
- Observed: the two sparse clips went the other way. Their 320p copies keep only the first 4 or 9 frames.
- The 320p copies cannot stand in for the raw durations.
- Figure: `figures/raw_vs_320p_excess.png`.

## f. Shared encoded frames (byte-identical video packets) [beyond the brief]

**Method.**
- Two different files containing the same encoded packet contain the same encoded frame. They were therefore cut
  by stream copy from one source and overlap in time, whatever their timestamps.
- Packets < 1024 B (25,044 of 1,477,863) are ignored so that tiny identical packets cannot link unrelated files.
- Only demuxing is needed. Results: `outputs/research/raw_audit_shared_packet_pairs.csv` (all pairs) and
  `raw_audit_overlap_pairs_normal.csv` (pairs where both clips have >= 30 frames, with overlap type).

**Distinct footage.**
- 977,626 distinct packets out of 1,477,863 (66.2%), i.e. at most **9.05 h of distinct footage at 30 fps** against
  13.68 h of frames.
- This is an upper bound: re-encoded copies of the same footage cannot be detected by hashing. The 183 HEVC clips
  share no packet with anything.

**Pairs and components.**
- 16,037 file pairs share >= 1 packet: 581 exact-duplicate pairs and 15,456 pairs of different files. Different
  pairs share a median of 10 packets (p90 153, max 380).
- **No pair crosses classes.**
- 7310 of 7705 files share packets with another file. Per class: low 2937 of 3028; mid 2041 of 2041; high 2332
  of 2636.
- The sharing graph has **96 connected components** (low 36, mid 30, high 30). Every component contains a single
  batch combo and a single class. **89 components have members in both train and test**, and they hold 7239
  files.
- The 395 files that share nothing: 183 HEVC, 90 h264 at 1064x598, 71 + 46 other h264, 1 mpeg4, and the 4
  unreadable files. By class: high 304, low 91.

**Split leakage.**

| | test files with >= 1 frame also in train | ... with >= 30 frames | ... and not byte-identical to any file | fully contained in train |
|---|---|---|---|---|
| packets >= 1 KB | **708/771 (91.8%)** | 699/762 (91.7%) | 675/738 (91.5%) | 221 |
| keyframes >= 1 KB | 706/771 (91.6%) | 697/762 (91.5%) | 673/738 (91.2%) | 264 |
| packets >= 16 KB | 706/771 (91.6%) | 697/762 (91.5%) | 673/738 (91.2%) | 242 |

- Over all 771 test files, **251 (32.6%) are fully contained in train** (every packet >= 1 KB occurs in a train
  file) and 338 have >= 50% of their frames in train.
- **49.0% of all test packets occur in train.**
- The same holds for val against train: 710/770 (92.2%) share frames, 273 are fully contained, and the packet
  share is 51.1%.
- Test against train ∪ val: 722/771 (93.6%) share frames, 289 are fully contained, and the packet share is 52.7%.
- Different-file pairs by split: train-train 10,277; train-val 2389; train-test 2252; val-test 251; val-val 155;
  test-test 132.

**Pixel check.** For 5 random test/train pairs (seed 0), one shared keyframe (>= 16 KB) was decoded from both
files with PyAV. All 5 are **pixel-identical** (yuv420p arrays equal):

| test clip | at | train clip | at |
|---|---|---|---|
| low/view1371 | 0.073 s | low/view1373 | 5.086 s |
| mid/view561 | 0.080 s | mid/view1438 | 2.577 s |
| high/view83 | 5.080 s | high/view87 | 0.066 s |
| mid/view1143 | 0.081 s | mid/view1141 | 10.087 s |
| low/view2872 | 0.069 s | low/view2871 | 10.075 s |

**Overlap patterns.** These cover the 11,940 different-file pairs where both clips have >= 30 frames.

- **Contained** (every packet of the shorter clip occurs in the other), 1824 pairs:
  - median 160 shared packets;
  - median |ID difference| 484; 63.7% are > 100 IDs apart;
  - by length: a 4-6 s clip inside a 9-12 s clip in 1134 pairs, a < 4 s clip inside a 9-12 s clip in 158, a 6-9 s
    clip inside a 9-12 s clip in 149.
- **Partial**, 10,116 pairs:
  - median 12 shared packets (about 0.4 s);
  - median |ID difference| 3; 59.9% are within 3 IDs;
  - mostly 4-6 s with 4-6 s (5100) and 9-12 s with 9-12 s (1675).
- Of the 7155 clips that overlap another clip, the nearest overlapping clip is within 3 IDs for 83.9%, and *all*
  overlapping clips are within 3 IDs for 49.7%. The median number of partners is 2 (max 10).
- In the ID ranges probed in `05_id_structure_probe.md`, overlapping pairs fall at ID + 2 in `mid/300-360` (50 of
  63 pairs), at ID + 1 in `low/500-560` (57 of 64), and at both in `high/1000-1060` (39 and 37).
- Figure: `figures/raw_shared_packet_pairs.png`. Clip ID is plotted against clip ID. The diagonal holds adjacent
  partial overlaps; the off-diagonal segments are ~5 s clips contained in ~10 s clips of a second ID range.

## What this means

**Observations** (measured on the files):
1. The release mixes:
   - 3 codecs, 5 frame sizes, 0-4 audio streams and two FFmpeg muxer versions;
   - only 355 clips at the paper's 1280x720;
   - a median length of 5.4 s, with 15% of clips at 9.5-10.5 s;
   - 13.91 h of video against the paper's 12 h 50 min.
2. Four files are unreadable (zero-filled), and 155 hold under 1 s of video (115 hold <= 5 frames). Their md5
   matches DVC's record.
3. 299 files form 128 groups of byte-identical files. 26 test files are byte-identical to train files.
4. 225 clips keep source timestamps on 5 s and 10 s grids; successive clips overlap, and some pairs share a start.
   In these series, IDs run forwards, backwards, or out of order.
5. 91.8% of test clips share at least one byte-identical encoded frame (in practice a keyframe of >= 16 KB) with a
   train clip, 32.6% are fully contained in train, and 49% of test frames occur in train. Shared keyframes decode
   to identical pixels.
6. No encoded frame is shared across classes. The 96 overlap components are each single-class and single-batch.
7. Container metadata predicts class only weakly (47.5% vs 39.3%), and is independent of the split. HEVC, 44.1 kHz
   audio and 1092x614 occur only in `high`; 1064x598 occurs only in `low`.
8. Most 320p copies are padded by exactly the raw first PTS. For the two sparse clips the 320p copy is truncated
   instead.

**Inferences** (consistent with the above, not proven by it):
1. The clips were cut by stream copy from longer recordings, in at least two segmentation passes: a ~5 s and a
   ~10 s grid, with cuts that overlap their neighbours by a few frames. Both passes were released, so many 5 s
   clips are subsets of 10 s clips. This would explain the median of 5.4 s against the paper's "about 10 seconds".
2. The random clip-level split puts overlapping frames of the same recording on both sides for > 90% of test
   clips. The leakage is therefore not only same-scene: it is frame-identical. Any frame-matching method can score
   high, which agrees with the review's 1-NN on thumbnails (96.1%) and single-frame models near 99%.
3. Clip IDs follow cutting order locally, with 2- or 3-way interleaving of sources in places and reversed order in
   some series. The same recording reappears in distant ID ranges (the second segmentation and the +452 / +23
   duplicate blocks), so ID proximity is a partial proxy for recording identity, not a complete one. The overlap
   components are a direct proxy for recording identity, except for the clips that share no packet (the HEVC
   and h264 1064x598 clips, among others).
4. The broken families look like failed export runs. In each one, every file holds the same few frames from the
   start of a source and from near its end, plus audio that grows by the grid step; the run's last file holds the
   real tail. Which tool produced them cannot be determined from the files.
5. The HEVC / Lavf61.7.100 / `hevc_qsv` batch (183 `high` clips) was re-encoded, by a newer FFmpeg with hardware
   encoding. Whether it duplicates footage present elsewhere cannot be checked by hashing.
6. Container metadata is useless to a model, which never sees it. The class-pure batches do show that some
   classes were exported in separate runs, so the batch structure is confounded with class for about 6-7% of
   `high` and `low`.
7. Every one of the 96 components is single-class. It is therefore possible that whole recordings (or the parts
   of them that were cut) were assigned a single class. That would make class partly a recording identity, but the
   files cannot tell this apart from cutting the class segments without overlap at their boundaries.

## Files

- Scripts:
  - `scripts/research/raw_header_audit.py` (header pass; `--md5`; `--packets`);
  - `scripts/research/raw_header_audit_tables.py` (tables and figures).
- Data, all in `outputs/research/`:
  - per-file headers and timing: `raw_header_audit.csv` and `.jsonl`, `raw_header_audit_timing.txt`;
  - hashes: `raw_md5.csv`, `raw_video_packets.npz`;
  - tables: `raw_audit_tables.md`, `raw_audit_joined.csv`, `raw_audit_id_runs.csv`, `raw_audit_pts_series.csv`;
  - broken files and duplicates: `raw_audit_broken.csv`, `raw_audit_duplicate_groups.csv`;
  - overlaps: `raw_audit_shared_packet_pairs.csv`, `raw_audit_overlap_pairs_normal.csv`.
- Figures, all in `docs/research/figures/`: `raw_duration_hist.png`, `raw_first_pts_hist.png`,
  `raw_first_pts_vs_id.png`, `raw_broken_audio.png`, `raw_vs_320p_excess.png`, `raw_shared_packet_pairs.png`.
- Reproduce, CPU only, from the repo root:
  ```bash
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit.py --fresh
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit.py --md5 --fresh
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit.py --packets
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit_tables.py
  ```
