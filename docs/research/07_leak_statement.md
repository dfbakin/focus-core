# What leaks in OUC-CGE, stated carefully

Claude Fable 5.1 for Denis Bakin, 2026-11-08. Synthesis of `docs/review/01_leakage_analysis.md` (October),
`docs/research/01`-`06` (this week) and the paper (`papers/ouc_cge.pdf`). Every number below is in one of
those documents; the CSVs behind them are under `outputs/research/`. Figures: `docs/research/figures/`.

## Verdict in one paragraph

The published train/val/test split of OUC-CGE is a random draw over clips that were cut, with overlap, from
about sixty camera recordings whose label is constant over whole recordings. For 97.8% of the 771 test clips
the training set contains at least one of: the same footage (byte-identical encoded frames, 91.8%), the
temporally adjacent segment of the same recording (90.7%), or the same instant seen by another camera (about
20%); 98.7% have a train clip from the same scene within thumbnail distance 27.5. A classifier only has to
recognise which recording a clip belongs to, which a single frame does. The 93-98% accuracies of the paper,
the student's 99.4-99.9%, and the 96.1% of a learning-free 1-NN on 32x32 thumbnails all measure that. The
hypothesis "the next clip is very close to the previous one and carries the same label" is right in
substance, and the data show something stronger: the clips are not merely similar, a third of the test set is
literally contained in training footage, and the label is a property of the recording, not of the clip.

## 1. How the release was produced (reconstructed from the files)

Observations, with the document that holds them:

- 21 sessions of about 70 minutes, 3 cameras per room, labels assigned per clip but adjudicated toward the
  scripted level of the lesson phase (paper, `02_paper_protocol.md`).
- Each camera recording was cut by stream copy into segments of about 5.3 s every 5 s, or 10.3 s every 10 s,
  so that consecutive segments overlap by a median of 9-10 frames (0.3 s); 335 junctions share the identical
  frame. Part of the footage was cut twice, on both grids and once on a 15-s grid, and both cuts were
  released under distant IDs (`06_temporal_structure.md` §3, `03_raw_release_audit.md` §f). Of 13.7 h of
  frames in the release, at most 9.05 h are distinct.
- Clips were exported per class in batches with different tools: three codecs, five resolutions, 0/1/2/4 audio
  tracks; every property depends on the class (Cramér's V up to 0.31) and none on the split (`03` §c).
- File names are sequential per class within each batch: forwards in time for `mid` and `high`, **backwards**
  for most of `low`; cameras of one session are interleaved by ID with periods 2 to 6, but their timelines are
  offset by 0.1-24 s, so partners are not the same instant cut in lock-step (`06` §3, §6). The paper's
  "randomly generated filenames" are not random, and the shuffle of the CSV rows is the only randomisation.
- Along a reconstructed recording (a chain of overlapping segments) the label never changes, and no frame-level
  continuation crosses a class folder. A random clip sits in a constant-label stretch of median 6 minutes (IQR
  2.5-7.9 min, up to 23 min) (`06` §7, §9).
- Integrity: median clip 5.4 s (paper: "about 10 seconds"); 13.9 h of video (paper: 12 h 50 min); 115 files
  with at most 5 frames; 4 unreadable; 128 groups of byte-identical files (`01`, `03`).

Inference: a recording is cut into a few hundred overlapping windows; every window inherits the recording's
label; the split then draws windows at random. Nothing in the release lets a user undo this without
reconstructing the recordings from pixels or packets, as done here.

## 2. The four layers of the leak, measured on the published test set (n = 771)

| layer | what is in train for a test clip | share of test clips | source |
|---|---|---|---|
| footage | a byte-identical encoded frame | **91.8%** (708) | `03` §f |
| footage | all of the clip's frames | **32.6%** (251) | `03` §f |
| footage | a byte-identical file | 3.4% (26) | `03` §d |
| time | the preceding or following segment of the same recording | **90.7%** | `06` §7 |
| camera | the same instant from another camera, audio-confirmed | 20.4% (structural upper bound 45%) | `06` §6 |
| scene | a clip of the same scene, thumbnail RMSE ≤ 27.5 | **98.7%** | `01_leakage_analysis.md` |
| any of footage / time / camera | | **97.8%**; 17 clips have none | `06` §9 |

Validation is the same (92.2% share footage with train; 35.5% entirely). Links between consecutive segments
cross the train/test boundary at 33.9%, the chance rate for a split that ignores time (`06` §7).

## 3. How close the nearest training frames are

Distances are RMSE between RGB frames on the 0-255 scale.

| comparison | 32x32 thumbnails | 128x72 frames |
|---|---|---|
| two consecutive raw frames (1/30 s) | 0.2-0.5 | 2.8 (IQR 1.5-4.7) |
| the same raw frame in two clips | 0 | 2.2 |
| junction between consecutive segments | | 11.9 (median) |
| about 0.5 s of motion | | 16.5 |
| first vs last frame of one clip (5.5 s) | | 36.8 |
| **test clip → nearest train clip, published split** | **2.7 median, 15.3 p95; 13% exactly 0; 84% ≤ 10** | |
| test clip → nearest train clip, grp17 / grp27b / grp27a | 31.3 / 38.9 / 65.6 (none ≤ 27.5) | |
| random pair of clips | 86-87 | 96 |

Reading: the nearest training clip of a typical test clip is closer to it than the first frame of the test
clip is to its own last frame. For 13% of test clips the nearest training thumbnail is identical to the pixel,
for 84% it is the same frame up to compression. Figures: `leak_nn_pairs_hires.png` (eight random test clips
beside their nearest train clip, d = 0.0-18.8, most sharing packets), `leak_distance_hist.png` (the
published-split distribution sits two decades left of the random-pair distribution; the scene-disjoint splits
sit between them, above 27.5).

## 4. How the clustering looks

Scenes are single-linkage components of the min-over-frame-pairs thumbnail distance (`01` §3b, reproduced
byte for byte on this machine, `04_cache_rebuild.md`). At threshold 27.5 there are 101 scenes; the 20 largest
hold 6,150 of 7,701 clips.

- `leak_tsne.png`: t-SNE of the centre thumbnails. The islands are the scenes; the label panel shows that
  labels are organised by island, and the split panel shows train, val and test interleaved inside every
  island. This is the leak in one picture.
- `leak_clusters_overview.png`: one medoid frame per cluster with label and split mix. Most clusters are one
  table seen from one camera and are single-label (e.g. cluster 0: 880 clips, 0/178/702; cluster 36: 312 clips,
  all low; cluster 25: 204, all high). The largest, cluster 6 (1,163 clips, 554/253/356), is the checkerboard
  room seen from the side with a standing teacher: one room whose phases changed label, merged by single
  linkage through near-identical backgrounds. It is 75% of grp27a's test set, which is why that split is the
  strictest "never seen this classroom" test.
- `temporal_triplets.png`: a period-3 ID run (high 139-314) laid out as three camera columns of one group,
  with audio synchrony between two of them, and the published split assigning train, val and test within the
  same step. `temporal_chain.png`: twelve consecutive segments of a 276-clip, 23-minute chain with IDs running
  backwards and splits mixed.
- Purity: at thresholds up to 17.5 (near-identical frames) every one of 530 clusters is 100% label-pure; at 27.5
  clip-weighted purity is 0.79 because of merges like cluster 6 (`01` §3b).

## 5. Verdict on the hypothesis, point by point

"They took continuous videos, cut them into clips of about 10 s, annotated each" — yes, with corrections: the
cut is 5 s for most of the data, the windows overlap, some footage was cut twice, and the annotation was
adjudicated toward the phase's scripted level, which makes the label constant over each recording.

"The leak comes from the next clip being very close to the previous one and very likely to have the same
label" — yes, and three things are stronger than that:

1. It is not similarity but identity. The neighbouring segment shares frames with the test clip, and for a
   third of the test set the entire clip is inside a training clip.
2. It does not depend on adjacency in the ID or in time. Any clip of the same recording anywhere in train
   suffices, and 98.7% of test clips have one within the same-scene distance. Shuffled file names change
   nothing.
3. "Very likely the same label" is "always the same label": no frame-level continuation crosses a class
   folder, so the label is constant along every reconstructed recording.

Two further points bear on the paper's numbers. The authors' PySlowFast patch passes frame indices as
timestamp ticks and, with the released files, returns one or two frames per clip; their six "video" models
were single-frame classifiers (`02_authors_code_audit.md`). And their accuracy (97.8%) exceeds the agreement
between their own two annotators (κ 0.75), which is only possible when the label is predictable from
something simpler than engagement, here the recording.

## 6. What this means for the research line

- Any protocol that puts clips of one recording on both sides is invalid for this dataset. The scene-disjoint
  splits already built (`grp27a`, `grp27b`, `grp17`) are also footage-disjoint: their minimum test-to-train
  distance is above 17.5, far above the 0-2 range of shared frames. On them the student's recipes score
  0.50-0.53 (strict) and 0.68-0.81 (loose) against a majority rate of 0.39-0.41 (`docs/review/03`).
- The effective sample size for cross-scene learning is about sixty camera recordings (21 sessions × ≤ 3
  cameras), not 7,705 clips; leave-one-session-out is the right unit, and the reconstructed chains plus the
  audio synchrony give the session structure the release withholds.
- Label changes inside a session cannot be located from the release, because the class folders cut every
  recording into single-label pieces. Behaviour-grounded labels would have to be re-annotated.
- The container metadata co-varies with the class (HEVC and 1064x598 are single-class), which a video model
  never sees but which shows that classes were exported in separate batches; any future pre-processing must
  not reintroduce such a batch signature (e.g. through resolution-dependent cropping).

## Addendum 2026-11-08: labels are a property of the camera recording

The session reconstruction (`08`, reviewed in `10`) found three audio-synchronous events whose cameras sit in different class
folders. At full resolution the cameras film the same group at the same table at the same instant; one camera's clips are
labelled `low` and the others' `mid` for several minutes (880 clips, 29.6% of `mid`, 9.1% of `low`). The label is constant along
each recording, so it was assigned per camera recording, not per moment. This does not change the leak verdict above; it adds a
label-consistency floor for the `low`/`mid` boundary that any honest evaluation has to state.
