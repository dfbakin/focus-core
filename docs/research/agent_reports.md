# Agent reports (verbatim hand-backs)

Each entry is the final report of a subagent (Claude Opus 5.5) as delivered to the supervising session
(Claude Fable 5.1), kept verbatim so that the reasoning behind the numbers in the numbered research
documents stays on record. Observations and inferences are the agents' own; the numbered documents and
the PKMS project note hold the reviewed conclusions.

## 2026-11-07 — 320p clip audit and rebuild attempt (brief: audit every 320p clip; rebuild the thumbnail cache and scene clusters on this CPU-only machine)

### Summary
- **Task 1 (clip audit): done.**
- **Task 2 (rebuild): not run.** Its thumbnail step needs far more than the 4 GB limit on these clips, even with one worker, so I stopped there as instructed. Instead I ran the split step from the published clusters: all three split files are byte-identical to the published ones.
- **Task 3:** the `review_cache` npz layout is described below. It comes from the code plus a real 30-clip run of `01_frame_cache.py`, because the full cache was never produced.
- **Main anomalies:**
  - 279 redundant byte-identical 320p files; 31 test clips have a byte-identical copy in train.
  - 202 clips are longer than 30 s, up to 322 s.
  - The median clip is 5.5 s, not 10 s.

Nothing was committed and nothing under `scripts/review/` or `docs/review/` was modified. No GPU was used and nothing was installed (PyAV 17.0.1 was already in the venv).

### Task 1: clip audit (`docs/research/01_clip_audit.md`, `outputs/research/clip_audit.csv`)

**Method.** cv2 sequential `grab()` until it fails gives the exact frame count (1 decode thread, pool of 8). PyAV (demux only) gives streams, audio, packet counts, pts and encoder tags. Each file's md5 is computed. All 7,706 manifest records are covered: the 7,705 labelled ones plus the unlabelled `videos/anonymity/a.mp4` row (test.csv line 772).

**Counts.** Labelled records per split and class:

| split | low | medium | high | total |
|---|---|---|---|---|
| train | 2396 | 1653 | 2115 | 6164 |
| val | 307 | 199 | 264 | 770 |
| test | 325 | 189 | 257 | 771 |
| all | 3028 | 2041 | 2636 | 7705 |

- 7,701 clips are readable.
- Labels always match the folder. The folder is named `mid`, not `medium`.
- The labelled manifest paths are exactly the same set as the raw `.mp4` files in the raw-video DVC manifest. That manifest also contains a `.DS_Store`.

**Unreadable clips.** `videos/high/view2521.mp4`, `view2630`, `view2631` and `view2632` (all train, label 2) are absent from the 320p directory. They are exactly the `fail` lines of `transcode_failures.txt`, and they do exist in the raw manifest. The md5s of the 7,701 local files match the 320p DVC manifest (0 mismatches).

**Duration** (n_frames / 30), over 7,701 clips:

| percentile | p1 | p5 | p25 | p50 | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| seconds | 0.300 | 2.567 | 5.267 | 5.500 | 10.000 | 11.833 | 168.933 |

- Min 0.133 s (4 frames), max 321.967 s (9,659 frames), mean 10.84 s.
- 9.5–10.5 s: 1,069 clips (13.9 %).
- Below 9.5 s: 5,692 (73.9 %).
- Above 10.5 s: 940 (12.2 %).
- Above 30 s: 202 (2.6 %); above 60 s: 175.
- Two modes: 150–170 frames (4,505 clips, about 5.0–5.7 s) and 300–302 frames (271 clips).
- The histogram is `docs/research/figures/clip_duration_hist.png`.

**Total hours.**
- All readable clips: **23.18 h** (train 19.15, val 2.10, test 1.93).
- The 202 clips over 30 s account for 9.55 h.
- The clips of 30 s or less total 13.64 h (mean 6.55 s).
- Distinct file contents only: 17.97 h.

The paper's 12 h 50 min is 46,200 s, a mean of 6.0 s per clip; 7,705 × 10 s would be 21.40 h. The 320p data support neither statement as written. "About 10 s" fails because the median is 5.5 s and only 13.9 % of clips are 9.5–10.5 s. The 12 h 50 min total also fails, because the observed total is 23.2 h. Raw durations could not be checked.

**fps, resolution, codec.**
- fps is 30.0 for every clip, by `CAP_PROP_FPS`, PyAV average rate and PyAV base rate.
- `CAP_PROP_FRAME_COUNT`, the PyAV header count, the packet count and the decoded count are equal for every clip.
- Resolution is 568×320 for 7,241 clips and 570×320 for 460. The 570-wide ones are high IDs 1–56 and 328–496, and low IDs 2181–2360, 2596 and 2645–2698.
- Every file is h264 yuv420p with a single stream, encoder tag `Lavc60.31.102 libx264`.
- My audit gives the same frame percentiles as the October Experiment 1 (9/77/158/165/300/355/5068). Its 117 clips under 64 frames equal my 113 readable ones plus the 4 unreadable.

**Audio.** None of the 7,701 320p files has an audio stream (one video stream each, per PyAV). That is by construction, since the transcode uses `-an`. Whether the raw clips have audio is **NA**: the raw files are not local.

**What `preprocess_resize.py` does.** Only one version exists in git (b2c0b5c, 2026-06-05). Per clip it runs:

`ffmpeg -y -hide_banner -loglevel error -threads N [-t max_seconds] -i SRC -vf scale='if(gt(iw,ih),-2,320)':'if(gt(iw,ih),320,-2)' -c:v ENC {-preset p4 -cq 24 | -preset veryfast -crf 24} -g 15 -pix_fmt yuv420p -an DST`

- **Audio:** `-an` drops it.
- **Duration:** `-t` before `-i`, with `--max-seconds` defaulting to 30, would truncate clips to 30 s. Yet 202 files exceed 30 s, so the copies were not made by this code with its default; either `--max-seconds 0` or a run without `-t` was used.
- **Frame rate:** there is no `-r`, `fps` or `-fps_mode` option.
- **Resolution:** the short side becomes 320, aspect kept, long side rounded to even, which gives the 568 vs 570 widths.
- **Encoder:** the default is h264_nvenc, but the file tags say libx264, so it was run with `--encoder libx264`.
- **Script comment:** it says p99 is about 14.5 s; the observed 320p p99 is 168.9 s. Its "up to 323 s" matches the observed 322 s maximum.

### Task 2: rebuild, not run

**Why.** `scripts/review/01_frame_cache.py` `decode()` keeps every decoded BGR frame of a clip in a list, so one clip needs n_frames × 320 × 568 × 3 bytes.

| per-clip memory | value |
|---|---|
| median | 90 MB |
| p99 | 2.77 GB |
| maximum | 5.27 GB (view1888) |
| clips over 1 GB / 2 GB / 4 GB | 175 / 120 / 58 |

- The 58 clips over 4 GB are the 57 identical 9,619-frame copies plus view1888.
- A simulation of `Pool(8)`, chunksize 4, gives a peak of about 23.5 GB, and that is optimistic.
- This machine has 7.6 GiB in total and about 3.4 GiB was available during the session.

**Problems found in `reproduce.sh` before running it:**
- `python` is not on PATH here, so it would need `VENV=.venv`.
- The default `OUT=outputs/review_grouped` would overwrite the 2026-11-05 rerun's `comparison_clusters.md` and `splits/` that are stored there.

**Files not produced:** `outputs/review_cache/*.npz`, `outputs/review_out/scene_clusters.csv` and `exp3b_clusters_by_thr.csv`. No thumbnail, `03b` or `10 --clusters` comparison was run.

**Split step from the published clusters.** I ran:

`VENV=.venv REPORT=outputs/research/splits_from_published_clusters OUT_BASE=outputs/research/split_roots bash scripts/review/run_07_splits.sh`

It took 21 s and exited 0. Output:
```
split_grp27a.csv: identical to docs/review/data/grouped/splits/split_grp27a.csv
split_grp27b.csv: identical to docs/review/data/grouped/splits/split_grp27b.csv
split_grp17.csv: identical to docs/review/data/grouped/splits/split_grp17.csv
```
The md5s are beb50d45…, 81d1f382… and 296bfb1a…, the same as the published files, with pandas 2.3.3 and numpy 2.2.6. Without a cache, the summary JSONs omit the leakage statistics. The `videos` symlinks under `outputs/research/split_roots/*` are dangling.

**Decision for you.** Getting the full rebuild needs one of these:
- Run the rebuild stage on a host with at least 32 GB RAM.
- Allow a memory-light copy of `01_frame_cache.py` in `scripts/research/` that keeps only the first, centre and last frames. It could be checked byte for byte against the original on the 30-clip sample.
- Allow an edit to `scripts/review/01_frame_cache.py`.

### Task 3: what `outputs/review_cache` would contain

These are the `--thumbs-only` npz files, verified on a real run: `01_frame_cache.py --limit 10 --workers 2` into `outputs/research/frame_cache_limit10/`, 8 s, values equal to the audit. There is one file per split, `frames_{train,val,test}.npz`, with N = 6164 / 770 / 771 rows in manifest order. The malformed row is dropped, and the 4 failed clips stay in train with `ok=False` and zero thumbnails.

| key | dtype | shape | notes |
|---|---|---|---|
| `path` | `<U24` | (N,) | |
| `label` | int64 | (N,) | |
| `split` | `<U5` | (N,) | |
| `cls_folder` | `<U4` | (N,) | |
| `view_id` | int64 | (N,) | |
| `ok` | bool | (N,) | |
| `n_frames` | int64 | (N,) | |
| `n_reported` | int64 | (N,) | |
| `fps` | float32 | (N,) | |
| `w`, `h` | int64 | (N,) | |
| `thumbs` | uint8 | (N, 3, 32, 32, 3) | frames first, n//2 and last; RGB; INTER_AREA |
| `feats` | float16 | (N, 3, 2048) | only without `--thumbs-only` |

- Uncompressed, the thumbnails are 9,216 bytes per clip (56.8 MB for train).
- With pandas 2.3.3 the files load without `allow_pickle`. `common.load_cache` concatenates the splits in train, val, test order.

### Wall times
- 200-clip audit sample: 31 s.
- Full audit: **23 min 19 s** (21:14:26–21:37:45, 8 workers, peak RSS 140 MB). Other agents' jobs were running at the same time, which may have slowed it.
- Summary script: about 10 s.
- 30-clip frame-cache sample: 8 s.
- Split step: 21 s.
- Rebuild: not run.

### Anomalies

**Exact duplicate files.** These are observations; causes are not determined.

| | 320p copies | raw (DVC md5) |
|---|---|---|
| duplicate groups | 130 | 128 |
| files in those groups | 409 | 299 |
| redundant copies | 279 | 171 |
| groups mixing labels | 0 | 0 |
| test clips with a byte-identical copy in train | 31 | 26 |

The large 320p groups:
- 59 × high, IDs 1591–1765 (every 3rd), 4 frames each.
- 57 × high, IDs 1831–1887, 9,619 frames (320.6 s) each. They make up 5.08 h of the total.
- 31 × high, IDs 1304–1394 (every 3rd), 9 frames each.
- 10 × low, 11 frames each.

Five of these 320p groups come from raw files that are *not* identical: 39, 36, 26, 10 and 2 distinct raw md5s. The ordinary-length pairs follow ID blocks: low 1060–1158 duplicates 1512–1583 and then roughly every 4th ID up to 1689; mid 818–840 duplicates 841–863. 100 of the 113 clips with fewer than 64 frames belong to the three short groups.

**Other anomalies:**
- Long clips over 30 s occur only in high (125) and low (77).
- There are two output widths, 568 and 570.
- There is the stray `anonymity/a.mp4` manifest row, and a `.DS_Store` in the raw directory.

### Files I created
- `scripts/research/01_clip_audit.py`, `scripts/research/01b_clip_audit_summary.py`
- `docs/research/01_clip_audit.md`, `docs/research/figures/clip_duration_hist.png`
- `outputs/research/clip_audit.csv`, `clip_audit_sample200.csv`, `clip_audit.log`, `clip_audit_summary.json`, `clip_audit_summary.log`, `clip_audit_duplicates.csv`, `raw_videos_dvc_dir.json`, `timings.txt`
- `outputs/research/frame_cache_limit10/frames_{train,val,test}.npz`
- `outputs/research/splits_from_published_clusters/split_{grp27a,grp27b,grp17}.csv` and `split_*_summary.json`, `splits_from_published_clusters.log`, `outputs/research/split_roots/{grp27a,grp27b,grp17}/`

Supervisor's decision on "Task 2": option 2, a memory-light streaming copy in `scripts/research/` verified byte for byte against the original (delegated as a separate run, see `04_cache_rebuild.md`).


## 2026-11-08 — Raw release header audit (brief: PyAV header audit of all 7,705 raw files; first-PTS structure; metadata x label; broken files and duplicates; raw vs 320p; report `03_raw_release_audit.md`)

**The headline finding goes beyond the brief: the split leaks at the frame level, not just by scene.** **708 of 771 test clips (91.8%) contain at least one encoded frame that is byte-identical to a frame in a train clip.** 251 test clips (32.6%) are entirely contained in train clips, and 49.0% of all test frames also occur in train. I checked this by hashing every video packet (demux only, no decoding). Five randomly chosen shared keyframes decode to pixel-identical images in both files. The rate barely moves when I keep only keyframes ≥ 16 KB (91.6%) or drop broken and duplicate files (673/738, 91.2%). Val against train is the same (710/770 share frames, 273 fully contained). No frame is shared across classes.

Both deliverables are done. All artefacts on disk were complete (`raw_audit_tables.md` is 874 lines and ends with the final table), so I recomputed nothing; the only extra step was a read of an existing CSV for one cross-check. GPU stayed hidden throughout, no `dvc` commands, nothing written under `data/`, no commits.

### a. Distributions (7701 of 7705 files readable)
- **Mixed formats:** 3 codecs (h264 6674, mpeg4 844, hevc 183), 5 resolutions (1920x1080 6886, **1280x720 only 355**, 854x480 224, 1064x598 180, 1092x614 56), and 0/1/2/4 audio streams (56/1061/551/6033).
- **Constant everywhere:** time_base 1/15360, pix_fmt yuv420p, nominal 30 fps. 454 clips have timestamp gaps (average rate below 30).
- **Tags:** two FFmpeg muxer versions (Lavf58.47.100 on 7518 files, Lavf61.7.100 on 182). No container has a `creation_time` tag.
- **Durations (video header):** median 5.40 s, mean 6.502 s, p1 0.133, p5 2.40, p25 5.17, p75 7.50, p95 10.67, p99 14.5, max 323.2 s.
  - Counts: 1167 clips in 9.5–10.5 s; 622 under 5 s; 676 over 10.5 s; 3 over 30 s.
- **Total footage:**
  - 13.910 h (13 h 54.6 min) by header; 13.686 h by frame count.
  - Paper: 12 h 50 min and "about 10 seconds" per clip.
  - At most 9.05 h is distinct footage (977,626 distinct encoded frames out of 1,477,863).
- File sizes total 28.977 GB.

### b. First-PTS structure
- **Start times:** 39 clips start at exactly 0, 7417 between 0 and 0.1 s, 20 between 0.1 and 1 s, and 225 at 1 s or later.
- **The 225 late starts sit on regular 5 s or 10 s grids,** consistent with offsets into longer recordings.
  - Neighbouring clips overlap in source time by 0.1 to 0.5 s.
  - In two mpeg4 series the clips come in pairs that share a start time; the shorter one is a byte-exact prefix of the longer.
  - Within these series, ID order matches time order exactly (+1), runs exactly backwards (−1), or is scrambled (0.26).
- **Correlation with ID over a whole class is weak:** Spearman low −0.217, mid 0.008, high 0.081.
- **IDs largely follow cutting order, at least locally.** Fraction of neighbouring-ID pairs with the same codec/size/audio/encoder, against a shuffled baseline (p = 0.001 throughout):

| class | lag 1 | lag 3 | shuffled |
|---|---|---|---|
| low | 0.901 | – | 0.624 |
| mid | 0.895 | – | 0.859 |
| high | 0.836 | 0.921 | 0.479 |

### c. Metadata vs label and split
- **Every property depends on class** (chi-square p from 1e-15 to 1e-304) **and none depends on the split** (p ≥ 0.14).
- **As a predictor it is weak:** a depth-4 decision tree reaches 0.475 in 5-fold CV against a 0.393 majority baseline. With numeric header values it reaches 0.533; trained on official train and scored on test, 0.512 (majority 0.422).
- **No property identifies a class with high accuracy, but a few values are 100% one class:**
  - high: hevc / Lavf61 / `hevc_qsv` (183 clips), 44.1 kHz audio (164), 1092x614 (56);
  - low: 1064x598 (180).
- A model never sees the container, so this is not a usable shortcut. It does show some batches were exported separately per class.

### d. Broken files and duplicates
- **4 unreadable files**, all in high and train:
  - `high/view2521` and `view2632` are entirely zero bytes;
  - `view2630` and `view2631` hold data only in their first 512 KiB, then zeros.
  - Their md5 matches DVC's record, so they were already corrupt when added; whether the OSF originals are also corrupt is unknown.
- **115 files have ≤ 5 real frames; 155 have under 1 s of video.** They come in four families:

| family | IDs | frames | first pts | audio length |
|---|---|---|---|---|
| high | 1831–1887 (consecutive) | 2 | 320.55 s | 34 s, growing 5.03 s per ID to 206 s |
| high | 1591–1762 (every 3rd ID) | 4 | – | grows 5 s per member to 215 s |
| high | 1304–1391 (every 3rd ID) | 9 | – | grows 5 s per member to 149 s |
| low | 13–41 | 11 | – | falls 10 s per ID |

  - Within a family, consecutive IDs share the same first pts.
  - Each file holds a few frames from the start of a source plus two frames near its end (175.7, 320.6 or 137.7 s).
  - The header duration understates where these frames actually sit in time.
- **Exact duplicates:** md5 over all files took 24.3 s, and all 7705 hashes equal DVC's.
  - 299 files in 128 byte-identical groups (171 redundant copies). Examples: low 1060–1131 repeat as 1512–1583 (ID + 452); mid 818–840 repeat as 841–863.
  - **26 test files are byte-identical to a train file**; 20 val files to train; 3 test files to val.

### e. Raw vs 320p copies
- **241 copies are longer than the raw stream by more than 1 s** (763 by more than 0.1 s).
- For 7487 of 7701 clips the extra length equals the raw first PTS to within one frame (r = 0.99964 among the 241).
- The 320p set totals 23.184 h against 13.910 h raw. It has 175 clips over 60 s; the raw set has 2.
- **Two clips went the other way:** `high/view1765` and `high/view1394` were cut from 81 to 4 and from 131 to 9 frames (−323 s and −180 s).

### f. Shared encoded frames (beyond the brief)
- 7310 of 7705 files share packets with another file. They form 96 groups, each with a single class and a single format batch. 89 groups have members in both train and test (7239 files).
- **Two kinds of overlap:**
  - *Contained*, 1824 pairs: typically a 4–6 s clip lying inside a 9–12 s clip. Median ID distance is 484.
  - *Partial*, 10,116 pairs: about 12 frames shared, usually between near-neighbour IDs (median distance 3).
- **What I infer:** the same recordings were cut twice, on a ~5 s grid and a ~10 s grid, with slightly overlapping cuts, and both cuts were released. This would explain the 5.4 s median against the paper's "about 10 seconds".
- This agrees with `docs/research/05_id_structure_probe.md`: in mid 300–360, overlaps fall at ID + 2 (50 of 63 pairs); in low 500–560, at ID + 1 (57 of 64).

### Wall times (8 processes)
| step | time |
|---|---|
| header pass | 50.1 s (rerun with extra decoding of sparse clips: 48.1 s) |
| md5 of 28.98 GB | 24.3 s |
| packet hashing (1,477,863 packets) | 66.7 s |
| tables and figures | 60–150 s |

The dvc pull finished at 21:32:32.

### Unexpected
- The 4 zero-filled files above.
- `high/view2135.mp4` has a data stream with garbled tags (handler `SkundHanflEr`, `creation_time` 1972-03-11).
- 92 clips carry a single AAC audio frame (0.021 s); 56 have no audio at all.
- The 183 HEVC clips share no packets with anything, so any overlap they have with other footage can't be seen this way.
- `.gitignore` contains unresolved merge-conflict markers (`<<<<<<< HEAD` / `=======`).
- The dvc log reports "M data/ouc-cge/videos/".
- `test.csv` line 772 is `videos/anonymity/a.mp4` with no label.

The report keeps observations and inferences in separate lists and cites the CSVs throughout.

### Files created
`docs/research/03_raw_release_audit.md`; `docs/research/figures/raw_{duration_hist,first_pts_hist,first_pts_vs_id,broken_audio,vs_320p_excess,shared_packet_pairs}.png`; `scripts/research/raw_header_audit.py`, `raw_header_audit_tables.py`; `outputs/research/raw_header_audit.{csv,jsonl}`, `raw_header_audit_timing.txt`, `raw_md5.csv`, `raw_video_packets.npz`, `raw_audit_tables.md`, `raw_audit_{joined,id_runs,pts_series,broken,duplicate_groups,shared_packet_pairs,overlap_pairs_normal}.csv`.

Supervisor's notes: the `.gitignore` conflict markers were committed on `master` by the merge of the review branch; resolved on `research/ouc-cge` on 2026-11-08 (both sides kept). `dvc status data/ouc-cge/videos.dvc` reports up to date, so the "M" line was transient.


## 2026-11-08 — Streaming frame cache, scene clusters and splits rebuilt on CPU (brief: memory-light re-implementation of `01_frame_cache.py --thumbs-only` verified byte for byte, then 03b, 10 --clusters and run_07_splits; report `04_cache_rebuild.md`)

The streaming cache matches the original script on every clip it was compared on. The thr 27.5 clusters and all three grouped splits reproduced exactly. One cluster threshold (12.5) differs on 6 clips because of float32 rounding on one boundary pair, not because of the data.

Every command ran with `CUDA_VISIBLE_DEVICES=""`. Nothing under `scripts/review/`, `docs/review/`, `src/`, `data/` or `outputs/review_grouped/` was modified, and nothing was committed. The full write-up is `docs/research/04_cache_rebuild.md`.

### Problems
- **My long-clip check ran out of memory on its first attempt** (killed with exit 137, nothing written). The bug was mine: it read `z[key][i]` from the npz inside a per-clip loop, so each access decompressed a whole array and kept it alive. Inference: this memory pressure may be what crashed your session. Fixed by reading each key once and copying only the needed rows; the rerun used 4 workers under `ulimit -v 4500000`.
- **The video links in `outputs/research/split_roots_rebuilt/*/videos` are dangling.** They point to `../ouc-cge-320/videos`, because 07's default link assumes `OUT_BASE=data`. The earlier `outputs/research/split_roots/` has the same problem. The manifests in them are correct.
- **`.gitignore` was modified by someone else**, not me, at 2026-11-08 01:03: merge-conflict markers were resolved. It is uncommitted.
- 03b ran without `--chains`, because the chains file needs the ResNet features. Only the `n_chains` column (-1 here) and the `frac_chains_in_single_cluster` key are affected; the clusters are not.

### 1. Streaming script and its verification (observations)
`scripts/research/02_frame_cache_stream.py` uses the same calls as the original 01 script: default `cv2.VideoCapture(path)`, `cv2.setNumThreads(1)`, `read()` until it fails, BGR→RGB, then a 32x32 INTER_AREA resize.
- **What it keeps:** only frame 0, the frame at `n_hint//2` and the latest frame read. I checked in 01's code that `n = len(frames)` and the indices are `(0, n//2, n-1)`.
- **Where `n_hint` comes from:** `n_frames` in the audit CSV, used only when the file size matches. Otherwise a counting pass with `grab()`.
- **Why the hint can't change the result:** if the real `read()` count differs from the hint, the clip is read again with `n//2`.
- **Extra outputs:** `frames_{split}_hires.npz`, with keys `path, view_id, label, split, ok, first, last`; `first`/`last` are (N,72,128,3) uint8, zero for unreadable clips. `frames_stream_meta.csv`, with per-clip hint source and pass count.

The comparison tool (`02b_compare_frame_caches.py`) checks key order, dtype, shape and `np.array_equal` for every key, plus the hires keys and shapes.

- **(a) First 10 clips per split vs `outputs/research/frame_cache_limit10`:** all 12 keys OK in all 3 splits, hires OK, `ALL IDENTICAL`. Saved in `outputs/research/fc_check/compare_stream_limit10.txt`.
- **(b) 30 random clips under 15 s:** 10 per split, seed 0, 0.13–11.47 s and 4–344 frames, in `outputs/research/tmp_root/` (manifests plus a `videos` link, built with `common.write_dataset_root`). I ran the original `01 --thumbs-only` and my script on this root in three hint modes; all 12 keys `array_equal=True` in every split, hires keys and shapes OK, `ALL IDENTICAL` (full text in `fc_check/compare_stream_audit_tmp_root.txt` and in the doc). The other two modes also end `ALL IDENTICAL` with exit 0: `--no-audit` (30 counting passes) and deliberately wrong hints (30 clips needing a second pass).
- **(c) Edge-case root `tmp_root_edge`:** the 4 missing files, the malformed `anonymity` row, six 4-frame clips and 2 ordinary clips per split. Identical in both hint modes, with the same 4 `BAD` lines as the original.
- **(d) Long padded clips, where the original cannot run here:** `02d_long_clip_check.py` decodes every frame but stores only its 32x32 thumbnail, picks `[0]`, `[n//2]`, `[n-1]` afterwards with no hint, and compares with the full cache. It checked all 58 clips over 9,000 frames plus 40 random clips: `meta_equal=58/58 thumbs_equal=58/58 hires_first_equal=58/58 hires_last_equal=58/58` (random: 40/40 each), `ALL IDENTICAL`. It took 4 min 28 s with 296 MB peak memory. This checks against a reimplementation of the original's logic, not the original itself.

### 2. Full run into `outputs/review_cache`
- **Time:** 11 min 25 s wall (decoding 665 s), 683% CPU while another agent was also loading the machine.
- **Memory:** 99 MB peak per worker. The main process peaked at 1,020 MB because it holds the hires arrays of all clips.
- **Clips:** 7,701 hints came from the audit and 4 from counting passes; every clip needed only one read pass, so the audit counts equal the `read()` counts. Results: 4 BAD clips (`high/view2521,2630,2631,2632`), train 6164/ok 6160, val 770/770, test 771/771, 117 clips under 64 frames.

### 3. Scene clusters (03b)
Estimated 1.5–2 GB, measured 2.0 GB peak, 23 s, threshold 27.5 chosen. `10_compare_with_published.py --clusters outputs/review_out` (exit 1): `scene_clusters.csv` byte-identical (101 / 101 clusters); thresholds 10, 15, 17.5, 20, 22.5, 25, 27.5, 30, 35, 40 same partition; **thr 12.5 DIFFERS (6 clips)**: 1127 published / 1126 rebuilt clusters.

**The thr 12.5 difference (observations):** in the rebuild `high/view1211` joins published cluster 35 (`view1191, 1195, 1199, 1203, 1207`). The linking pair is `view1207` (last frame) to `view1211` (first frame). Its exact float64 distance is 12.4986848, 0.0013 below the threshold; the CPU float32 run gives 12.4986982.

**Accuracy of the published distances (observations):** against exact float64 nearest-train distances for the 771 test clips: published errors median 7.1e-3, max 0.805, falling steadily with distance to 9e-5 relative above 20 (the float32 cancellation pattern); rebuilt (CPU) errors median 2e-4, max 0.072. An earlier GPU rebuild on an RTX 5090 host (`outputs/review_grouped/host_logs`) reproduced every threshold byte for byte.

**Inference:** the published distances were computed in float32 on a GPU, and the 12.5 flip is that rounding error on a boundary pair, not a difference in the thumbnails. The evidence for identical thumbnails is the byte-identical thr 27.5 file, the 10 matching partitions, this error pattern, and the leakage statistics in section 4.

**Finding about the published split:** 120 of the 771 published test clips have a train clip with an identical 32x32 thumbnail (exact distance 0). The published 03b reports them as 0.35–0.81 because of float32 noise; the 07 float64 statistics show `min 0.0`.

### 4. Grouped splits (`run_07_splits.sh` with `CACHE`)
2 min 52 s, 2.6 GB peak. All three split files identical to the published ones (`SPLITS_DONE`). Median test-to-nearest-train RMSE, published vs rebuilt: grp27a 65.598969 / 65.598969; grp27b 38.900162 / 38.900162; grp17 31.284698 / 31.284698; random split 2.696509 / 2.696509. All medians, p5 values, minima and fractions under 10/17.5/27.5 exactly equal, including val→train and test→val. The only differences are four p95 values, off by 3.8e-6 to 7.6e-6 (1–2 float32 ulps), and the `avoided` path string. Inference: numpy percentile interpolation or BLAS last-bit differences between the two hosts.

### Paths created
Scripts `scripts/research/02_frame_cache_stream.py`, `02b_compare_frame_caches.py`, `02c_make_tmp_roots.py`, `02d_long_clip_check.py`; cache `outputs/review_cache/` (`frames_{train,val,test}.npz`, `frames_{train,val,test}_hires.npz`, `frames_stream_meta.csv`); clusters `outputs/review_out/` (16 files); splits `outputs/research/splits_rebuilt/`, `outputs/research/split_roots_rebuilt/`; verification `outputs/research/frame_cache_stream_limit10/`, `tmp_root/`, `tmp_root_edge/`, `fc_check/`, `long_clip_check.csv`; logs and comparisons `frame_cache_stream.log`, `scene_clusters_03b.log`, `compare_clusters_published.txt`, `compare_03b_numeric.txt`, `compare_03b_exact_nn.txt`, `exp3b_nn_test_to_train_dist_exact_float64.npy`, `splits_rebuilt.log`, `compare_split_summaries.txt`, `long_clip_check.log`; report `docs/research/04_cache_rebuild.md`.

Supervisor's note: the OOM (exit 137) of the first long-clip check coincides with the IDE crash of 2026-11-07 evening; memory-heavy steps on this 7 GB machine now run under `ulimit -v` with at most 4 workers.


## 2026-11-08 — Temporal and camera structure, leak figures (brief: successor structure along the ID axis from 128x72 first/last frames, interleaved partners as cameras, chains, figures; addendum: use the packet-sharing graph as exact ground truth; report `06_temporal_structure.md`)

All of A–E is done, and the packet ground truth is used throughout. The main results: in `low` the clip IDs mostly run **backwards** in time; interleaving by ID has three different causes, and only one of them is cameras; the cameras are **not cut in lock-step** (their timelines are offset by up to 24 s). Of the 771 test clips, **97.8%** have a temporal neighbour, a simultaneous view or shared footage in train. Everything ran on CPU, seeded, in about 7 minutes in total, with nothing changed under the protected directories and no commits.

**Method change.** The forward-only d_k from the brief fails because of the backward `low` IDs (precision 0.55 against packets). I added the backward distance b_k(N) = RMSE(last(N+k), first(N)) and take k*, direction = argmin over the 16 candidates. Distances are RGB at 128x72; gray gives the same argmin for 99.6% of clips.

**Reference scales (128x72).**
- Frames 0–2 of most 320p clips are ffmpeg padding duplicates (RMSE 0.28), not a step in time.
- Real consecutive raw frames: 2.79 (IQR 1.46–4.65).
- The same raw frame appearing in two clips (packet lag 0, n=335): 2.22.
- About 0.5 s of motion: 16.5. First vs last frame of one clip: 36.8. Random same-class pairs: 95.9.

**A. Continuations and periods**
- **Cuts.** The KDE of the per-clip minimum distance m has modes at 2.34, 15.7 and 92 and valleys at 3.31 and 76.9. Primary: 76.9. Strict: 3.31 (frame-exact junctions only). Intermediate: 16.5.

  | cut | precision vs packets | recall |
  |---|---|---|
  | 3.31 | 0.993 | 0.135 |
  | 16.5 | 0.936 | 0.652 |
  | 76.9 | 0.867 | 0.931 |

- **Exact strides from packets.** Every shared packet in a pair gives the same offset. The next segment starts 150 frames (5 s) later for 53–72% of clips and 300 frames for 14–29%; consecutive segments overlap by a median of 9 frames. In `low`, 79% of ID-adjacent overlapping pairs have the higher ID earlier.
- **Periods.** The frame segmentation agrees with the packet segmentation on 89.5% of clips; 64 runs of ≥15 clips cover 93.6% of clips. high: +1 40%, +3 35%, +2 12.5%, +4 12%; mid: +1 51%, +2 20%, +3 10%, +5 9%; low: −1 47%, −2 28%.
- **Disagreements at the primary cut:** 839 clips whose frame neighbour shares no packet (same scene, not adjacent in time); 464 clips with long overlaps where frames pick another k; 391 clips without packet data, where frames are the only evidence.

**B. Partners = cameras?** This covers 35 runs with |period| ≥ 2 and 3,553 clips. Labels always agree by construction (IDs are per class, and no packet crosses classes).
- **Cameras, 24 runs / 2,778 clips.** Partners: share packets 0.4%, same scene cluster at 27.5 0.7% (random same-batch pairs: 9.7%). The same camera's next segment, N+p: shares packets 94%, same cluster 97%. Durations: identical frame counts 5.0% (random 3.2%), all-equal triples 1.2%, same first pts 19% (random 10%), so **not lock-step**.
- **Audio, as a direct test of simultaneity.** I stitched each column into a 120-s loudness envelope and correlated column pairs over ±30 s. The null (unrelated columns) has a maximum of 0.41. 28 different-camera column pairs are synchronous, up to r=0.95. 20 of the 24 camera runs have at least one such pair. Offsets between cameras are 0.1–24 s (median 5.4 s).
- **Re-cuts, 11 runs / 775 clips.** The interleaved columns are the same footage cut twice (10-s and 15-s mpeg4 cuts, or shifted windows): they share packets (50%) and their audio correlates ≈1.
- **Two different groups.** `low` 1319–1406 alternates two groups of students, with no audio sync.
- **The broken byte-identical clips at every third ID** are one whole camera column (with 2 audio streams instead of 4): column 0 of high 1223–1396 (9-frame clips) and column 2 of high 1511–1768 (4-frame clips). Per-camera audio-stream counts also explain many of the 1-clip runs in `raw_audit_id_runs.csv`.

**C. Chains.** Links: 6,478 from packets, 379 from frames where one file has no packet data, and 2 frame-exact frame links between files that share no packets.
- 571 chains plus 271 singletons; 85% of clips are in chains of ≥5.
- For a random clip, its chain has a median of 72 clips spanning 368 s (IQR 150–472 s); the longest is 276 clips (23 min).
- At the junction: median lag 8 frames, median d 11.9 (4.3× the consecutive-frame floor).
- Label changes along chains: 0, by construction. Linking chain ends across all classes found **no** cross-class continuation at d ≤ 16.5 (189 links) or ≤ 3.31.
- 33.9% of links cross splits, against 34.0% expected if the split ignored time; 186 chains contain both train and test.
- Results at the strict and intermediate cuts are close (80.5% and 84.5% of clips in chains ≥5).

**Test-clip leak (n=771)**

| test clips with … in train | share |
|---|---|
| a chain predecessor or successor | 90.7% |
| an audio-confirmed simultaneous view from another camera | 20.4% (17.3% at r>0.6) |
| any interleaved partner (structural upper bound) | 45.3% |
| shared footage (packets) | 91.8% |
| all footage | 32.6% (251 clips) |
| any of the above | 97.8% (only 17 clips have none) |

Validation clips: 92.2% share footage with train, 35.5% entirely. 43 test clips have a near-identical (d ≤ 2) train clip with no shared packet; 16 of those cross codec batches (re-encoded copies the packets cannot see). Nearest-train medians: published 2.7 (98.7% of test clips ≤ 27.5); grp17 31.3, grp27b 38.9, grp27a 65.6 (0% ≤ 27.5).

**Contradictions to `05_id_structure_probe.md`**
1. The probe's floor of 0.2–0.5 was the padding duplicate. With the right floor, frame-exact continuations exist (335 junctions).
2. `low` 500–560 has no gap between segments: its IDs run backwards and the segments overlap by about 8 frames.
3. Multi-camera interleaving is confirmed, but lock-step cutting is not.
4. Not every alternation is cameras: there are re-cut runs and one run of two different groups.
5. Periods 4, 5 and 6 also exist (8 runs, 958 clips).
6. The October doc's nearest-train floor of 0.4 was float32 noise: 120 test clips share an identical 32x32 frame with train.

**Beyond the brief.** Two audio scripts (`04e`, `04f`) that decode raw audio, a table script (`04i`), and two supporting figures (`temporal_min_d_hist.png`, `temporal_audio_sync.png`).

**Figures** (`docs/research/figures/`, 150 dpi): `temporal_id_axis_{low,mid,high}.png` (rolling median of forward d_1–3 and backward b_1–3 along the IDs, with strips for frame period, packet period and raw batch); `temporal_triplets.png` (a period-3 run, 3 cameras of one group, and a period-2 run, 2 cameras, with ID, label, split, cluster and the audio offsets between columns); `temporal_chain.png` (12 consecutive clips of the 276-clip chain, with the distance to the next clip and the packet stride; IDs running backwards and the split mixing); `temporal_min_d_hist.png`; `temporal_audio_sync.png`; `leak_nn_pairs_hires.png` (8 test clips beside their nearest train clip, with d, ID gap and packet sharing); `leak_clusters_overview.png` (medoid frame and label/split mix of the 20 largest clusters at 27.5); `leak_tsne.png` (t-SNE coloured by cluster, label and split; splits fully interleaved); `leak_distance_hist.png` (test→nearest-train distances under the four splits, plus reference pair populations).

**Wall times:** 04a 38 s, 04b 33 s, 04c 43 s, 04d 27 s, 04e 1 min 40 s, 04f 1 min 11 s, 04g 9 s, 04h 51 s, 04i 1 s.

Files: report `docs/research/06_temporal_structure.md`; scripts `scripts/research/04a_reference_scales.py` … `04i_report_tables.py`; outputs `outputs/research/temporal/` (`succ_dist.csv`, `ref_frame_lags.csv`, `ref_pairs.csv`, `packet_pairs_offsets.csv`, `clip_temporal.csv`, `runs_frames.csv`, `runs_packets.csv`, `pr_sweep.csv`, `disagreements.csv`, `packet_offsets_by_iddiff.csv`, `packet_offsets_by_period.csv`, `partner_tests*.csv`, `chains.csv`, `chain_links.csv`, `superchain_links.csv`, `audio_sync_pairs.csv`, `audio_sync_summary.json`, `audio_columns.csv`, `audio_null.csv`, `simultaneous_views.csv`, `audio_columns_summary.json`, `leak_nn_by_split.csv`, `leak_figures_summary.json`, `tsne_xy.npy`, `summary.json`, `report_tables.md`, logs `04*.log`).


## 2026-11-08 — Survey of alternative classroom datasets (web only). Kept in full as `docs/research/09_alternative_datasets.md`; verdict: no public dataset is clearly better suited for group engagement from room cameras; DIPSER is the only open alternative (per-student labels, 668 GB of frames), TVD/GTI has the best real footage but no usable engagement label and contract-bound access.


## 2026-11-08 — Session reconstruction with quality monitoring (Opus 5.5 agent, verbatim hand-back)

Brief: reconstruct recordings (tier 0), events (tier 1, audio cross-correlation with a null-calibrated threshold) and session candidates (tier 2, scored only); produce per-clip table, fold files and `docs/research/08_session_reconstruction.md` with a mandatory quality-monitoring section. CPU only, ≤2.5 GB RSS. Supervisor notes follow the report.

### Session reconstruction for OUC-CGE: per-clip table, three fold files and quality report

Everything you asked for is built: the per-clip table, three fold files, seven figures and the report `docs/research/08_session_reconstruction.md` (section 6 is quality monitoring). The pipeline was rerun end to end on CPU. Nothing was committed and nothing in the protected directories was changed. One import of the review's fold code left a `.pyc` file in `scripts/review/__pycache__`; I deleted it, and the script no longer writes one.

The main problem: the null you briefed (pairs from different class folders cannot be simultaneous) turned out to be false, so I replaced it (Surprise 1 below).

#### Coverage (7,701 readable clips)
| tier | result | clips |
|---|---|---|
| 0 exact recordings | 90 recordings | 6,483 (84.2%) |
| 0 probable recordings | 32 recordings | 1,202 (15.6%) |
| 0 single clip, unlinked | 16 recordings | 16 (0.2%) |
| 1 events with ≥ 2 recordings | 31 events, 124 recordings | 6,059 (78.7%) |
| 1 recording alone | 14 recordings | 1,642 (21.3%) |
| 2 session candidates with ≥ 2 events | 9 candidates (22 groups in all) | 6,415 (83.3%), not verified |

- **Recordings:** 138 in total (high 71, low 38, mid 29).
- **Events:** 45 in total. By evidence, clips sit in events via audio 5,706, shared packets 161, near-identical frames 189 and the prior fallback 3.

#### Tier 0 (observations)
- **Exact:** packet offsets agree on one time axis per unit (to 1.8e-13 s over all 16,037 sharing pairs).
- **Probable merges:** 355 accepted out of 12,842 candidates. The distance cut is 38.6 (128x72 RMSE), the 95th percentile of the 6,478 junctions proven by packets.
- **Merge distances:** median 4.0 versus 11.9 for the exact junctions.
- **Where merges are:** 350 involve clips with no shared packets (HEVC, 1064x598); 5 join two packet-linked units.
- **Fix to the earlier chains:** I kept 158 broken-family or sparse clips out of the time axes. They had bridged parts of camera columns and produced a 120-s jump.

#### Tier 1: audio threshold and false-link estimate
- **Pairs scored:** all 8,646 recording pairs, same class and cross class.
- **Score:** masked correlation of the loudness envelopes, lags within ±60 s, the overlap must cover at least 80% of the shorter recording's audio. Each half of the overlap must also pass the 99th-percentile threshold on its own (persistence test).
- **Nulls:** same recording at two disjoint times (60,002 pairs) and the partner reversed in time (17,292). The threshold is the larger of the two at q99.9, per length bin.
- **Links surviving (same / cross class):**

| | q99 | q99.9 | q99.99 |
|---|---|---|---|
| above threshold | 88 / 34 | 72 / 9 | 64 / 5 |
| above threshold and persistent | 62 / 4 | 61 / 4 | 60 / 4 |

- **False-link estimate:** no null pair passes the combined rule. That puts the rate below 3.9e-5 per pair, or under 0.34 expected false links. The threshold alone would give 8.3 at q99.9.
- **Second pass over all lags:** added 29 links (13 new tree links, 16 that close existing cycles). All 30 cycles agree to within 0.05 s.

#### Consistency checks
- **At most 3 recordings present at once:** pass in all events once duplicate copies count once. Before duplicates were recognised, 6 events reached 4.
- **Same room layout within an event:** pass, 0 mixed.
- **No shared packets between cameras in an event:** pass. The 159 clip pairs that do share packets are the same footage (broken export families joined to their camera columns).
- **Previous agent's 28 synchronous pairs:** 26 land in the same event; 9 offsets agree within 0.02 s and 11 within 0.9–5 s. The other 4 cannot be compared directly and 2 have no offset.
- **Singletons:** all 14 have at least 30 s of audio (high 8 / 510 clips, low 4 / 879, mid 2 / 253). Every clip without audio got into an event some other way.
- **Note:** I count 166 clips with a single AAC frame, not the 92 in the brief; 56 have no audio stream.

#### Folds (`StratifiedGroupKFold`, `random_state=0`)
| | event_disjoint_k5 (45 groups) | session_candidate_k5 (22 groups) | published split |
|---|---|---|---|
| file pairs sharing packets across folds | 0 | 0 | 5,025 |
| synchronous audio pairs across folds | 0 | 0 | 94 |
| minimum cross-fold thumbnail distance | 3.09 | 3.09 | 0 |
| nearest-neighbour median, cross fold | 32.9 | 33.8 | 2.70 (test) |
| share ≤ 10 | 0.05% | 0.05% | 84.0% |
| share ≤ 27.5 | 15.7% | 13.1% | 98.7% |

- **Class counts:** per fold are in `fold_verification.csv`; the event-disjoint folds hold 566–696 low, 386–436 mid and 506–573 high clips each.
- **Leave-one-event-out (10 largest events):** 0 shared packets in every fold; test-to-train nearest-neighbour median 29.8–38.4.
- **Leakage statistics:** computed with `nn_stats()` imported from `07_make_grouped_splits.py`; it reproduces the published-split numbers.
- **Residual close pairs across folds:** high 2079/2081 (d 3.09) and high 2135/2340 (d 7.85).

#### Surprises
1. **Four audio-synchronous pairs cross a class folder** (low and mid, r 0.88–0.94 over 2–7 min). The frames show the same scene from other angles at the same instant. So 3 events hold one room at one moment under two labels. These pairs became candidates, and the nulls are built only from pairs that cannot be simultaneous.
2. **Duplicate footage across export batches.** 15 audio pairs at r ≥ 0.95 within one scene cluster, plus 21 near-identical-frame links, join re-encoded or re-cut copies (h264, mpeg4, HEVC; distant IDs). 229 clips are flagged as the smaller copy.
3. **Interleaved runs with period 4–6 are not one room.** All 7 split into 2–3 audio groups that are silent with each other at the structural lag. Inference: several rooms were filmed in the same time slot and exported in lock-step. As a result the structural fallback was used 0 times; 51 column pairs were left unlinked.
4. **The rectangular-desk room has no second camera.** None of its 8 recordings has a synchronous partner, although all have long audio.

#### Fallbacks
- **Prior fallback:** 2 recordings, 3 clips.
- **Time axis:** frame-based for 392 clips; downstream of a probable merge for 314 packet-timed clips; unknown for 158.
- **Event offset:** unknown for 34 recordings.
- **Room layout:** my own visual label of one frame per recording (8 rectangular-desk, the rest round table). Please check it against `sessions_recordings_sheet.png`.

#### Doubtful decisions (15 listed in report §6.7)
Ten are Tier-0 merges and five are near-identical-frame links; no audio link is among them. The worst is low 993→991 (distance 37.2, confidence 0.06, 69 clips). All 15 show the same viewpoint on both sides in the sheet.

#### Tier 2 (candidate only)
- **Method:** visual only — the background must match within 28.6 RMSE and the clothing-colour similarity must be at least 0.73; only mutual best pairs are linked.
- **Result:** 28 links (scores 0.78–1.0). Six of the 9 multi-event candidates contain all three levels.
- **Limit:** the method cannot tell two sessions of one group in one room apart. It is not verifiable from the release.

#### Wall times and memory
Total about 9.5 min:

| script | wall | peak RSS |
|---|---|---|
| 05b | 4:43 | 3 workers ≤ 273 MB each |
| 05a | 0:09 | 428 MB |
| 05c | 2:19 | 919 MB + 3 workers ≤ 118 MB |
| 05d | 0:19 | 416 MB |
| 05e | 0:52 | 859 MB |
| 05f | 0:18 | 321 MB |

All processes together stayed under 1.3 GB.

#### Files
In `/home/dfbakin/Documents/focus-core/`:
- **Report:** `docs/research/08_session_reconstruction.md`
- **Scripts** (`scripts/research/`): `05a_recordings.py`, `05b_audio_envelopes.py`, `05c_events_audio.py`, `05d_session_candidates.py`, `05e_table_folds.py`, `05f_figures.py`
- **Table and folds** (`outputs/research/sessions/`): `clip_sessions.csv`, `folds/event_disjoint_k5.csv`, `folds/loeo.csv`, `folds/session_candidate_k5.csv`
- **Supporting outputs** (same folder, 49 MB): `clip_recordings.csv`, `units.csv`, `recordings.csv`, `merges.csv`, `unit_overlaps_without_packets.csv`, `audio_envelopes.npz`, `audio_pairs.csv`, `audio_null.csv`, `audio_thresholds.csv`, `events.csv`, `event_table.csv`, `structural_links.csv`, `identity_links.csv`, `prior_links.csv`, `compare_04f_column_pairs.csv`, `recording_features.npz`, `session_pairs.csv`, `session_links.csv`, `session_candidates.csv`, `fold_verification.csv`, `fold_verification.json`, `doubtful_decisions.csv`, `tier0_summary.json`, `tier1_summary.json`, `tier2_summary.json`, `table_folds_summary.json`, and logs `05a`–`05f*.log`
- **Figures** (`docs/research/figures/`, faces visible, internal only): `sessions_events_sheet.png`, `sessions_doubtful_sheet.png`, `sessions_candidates_sheet.png`, `sessions_timeline.png`, `sessions_recordings_sheet.png`, `sessions_audio_null.png`, `sessions_tier0_merges.png`

Agent usage: 163 tool calls, ~579k tokens, 80 min wall.

**Supervisor notes (2026-11-08).** Reviewed the QC section and all seven figures. Verdict in `docs/research/10_reconstruction_qc_and_protocol.md`. Tiers 0 and 1 are sound; fallbacks touch 3 clips; the structural fallback was never used; the 14 unassigned recordings all have enough audio to have been linked if a partner existed. Two additions of my own: (1) `05g` extracted full-resolution frames of the three cross-class synchronous events and shows the same group at the same instant under `low` and `mid` labels (880 clips, 30% of `mid`); the agent had left this open at 128x72. (2) `05h` united events with the October scene clusters: 16 groups, no cross-fold neighbour under 27.5 on any channel, but one group holds 66% of `high`. The event-disjoint folds alone leave 15.7% of clips with a same-scene neighbour across folds and two missed-merge pairs at 3.09 and 7.85. The layout labels in `sessions_recordings_sheet.png` were checked by eye: the 8 rectangular-desk recordings are correctly marked. The brief's count of 92 single-frame audio clips was wrong; the agent's 166 is right (it matches `a0_frames == 1` in the raw audit).


## 2026-11-08 — GPU experiment code A–E (Opus 5.5 agent, verbatim hand-back, first round)

Brief: implement section 6 of `docs/research/11_gpu_experiment_plan.md` (roots materialiser, host runner, aggregator, metadata tree, frozen probe), CPU-tested locally, no host access. Supervisor notes follow.

### Report: GPU experiment code for the OUC-CGE leak gradient (A–E)

All five scripts are written and CPU-tested here. Nothing touched the GPU, and there were no dvc, git commit/push or ssh calls.

**Most important finding: G1 group ids are not event ids.** The brief assumed they were. `05h_union_groups.py` relabels the connected components, so the ids differ for all 7,701 clips (0 of 7,701 match).
- I checked this against `clip_sessions.csv`: `event_id` is identical in both tables, and the mapping is one-to-one (45 events ↔ 45 G1 groups).
- Events 9, 11 and 18 are **G1 groups 34, 40 and 44**. These are the only three multi-class G1 groups, with 345 + 155 + 380 = 880 clips.
- G1 ids 9, 11 and 18 are single-class events 6, 0 and 23. Excluding them by number would have been wrong.
- So 06 maps the events through `event_id` and excludes groups 34, 40 and 44 from the P1 validation candidates. The mapping is printed and stored in `val_groups.json`.

#### A. Roots (`06_make_protocol_roots.py`)
Runs in 1.1 s with 119 MB peak RSS. All checks passed: each clip is in exactly one split per fold, no group crosses splits, and the five test folds partition the 7,701 clips. The manifests are re-read from disk and compared with the intended split, and a rerun gives byte-identical files.

Counts are low/mid/high. Validation groups are listed in the order the greedy rule added them.

| root | train | val | val groups | val % of pool, L1 | test |
|---|---|---|---|---|---|
| p1_f0 | 5517 (2147/1458/1912) | 517 (185/185/147) | 18, 7, 33, 19, 39 | 8.6%, 0.171 | 1667 (696/398/573) |
| p1_f1 | 5670 (2221/1470/1979) | 542 (210/185/147) | 19, 7, 39, 21, 33 | 8.7%, 0.150 | 1489 (597/386/506) |
| p1_f2 | 5709 (2277/1465/1967) | 517 (185/185/147) | 18, 7, 39, 19, 33 | 8.3%, 0.186 | 1475 (566/391/518) |
| p1_f3 | 5574 (2220/1386/1968) | 603 (237/219/147) | 18, 7, 33, 21, 35 | 9.8%, 0.207 | 1524 (571/436/517) |
| p1_f4 | 5587 (2193/1489/1905) | 568 (237/122/209) | 18, 5, 35, 21 | 9.2%, 0.094 | 1546 (598/430/518) |
| p2_f0 | 5592 (1765/1386/2441) | 835 (501/277/57) | 9 | 13.0%, 0.641 | 1274 (762/378/134) |
| p2_f1 | 6072 (2294/1386/2392) | 745 (233/378/134) | 13 | 10.9%, 0.497 | 884 (501/277/106) |
| p2_f2 | 5663 (2254/1219/2190) | 745 (233/378/134) | 13 | 11.6%, 0.516 | 1293 (541/444/308) |
| p2_f3 | 5413 (2074/1197/2142) | 745 (233/378/134) | 13 | 12.1%, 0.503 | 1543 (721/466/356) |
| p2_f4 | 4538 (2249/1385/904) | 456 (276/180/0) | 12 | 9.1%, 0.362 | 2707 (503/476/1728) |

Label-conflict clips per split:
- **P1:** f0 train 500, test 380; f1 train 880; f2 train 380, test 500; f3 and f4 train 880.
- **P2:** f0 train 725, val 155; f1 train 725, test 155; f2 train 880; f3 train 500, test 380; f4 train 535, test 345.

I also added `/data/p1_f*/` and `/data/p2_f*/` to `.gitignore` after the `/data/grp*/` line.

#### B. Runner (`06b_run_protocols.sh`), DRY=1 summary
The DRY run prints 26 blocks (probe plus 25 runs), each with its training, evaluation and cleanup commands:
```
DRY SUMMARY: to run: probe=1 r50=20 slow=5 (already done, skipped: 0); dropped by budget: none
DRY SUMMARY: projected wall time 260 min (estimates: probe 10, r50 8, slow 18 min per run incl. evaluation)
DRY SUMMARY: all needed roots and inputs exist
```
- **Budget:** `BUDGET_MIN=150` drops slow_p2 and then r50_p2 seed 44 (projected 130 min). 200 and 250 drop slow_p2 only; 300 drops nothing. The drop set is recomputed at every decision point from measured run times, so a block dropped on early estimates can come back.
- **Other checks:** the RUNS filter works for run names and block names. In a copy of the layout with no data, DRY lists every missing root and exits 2. A non-DRY call with `CUDA_VISIBLE_DEVICES=""` aborts with exit 3 before writing anything.
- **Post-training logic:** tested in a scratch layout with stub training, evaluator, torch and nvidia-smi:
  - Ties go to the earliest epoch.
  - Seed 42 keeps the best and final epoch plus `last.ckpt`; seeds 43/44 delete all checkpoints.
  - A failed epoch evaluation keeps the checkpoints and writes no done marker. A rerun resumes without retraining and evaluates only the missing epoch.
  - A failed training run aborts with its exit code; FORCE=1 retrains and re-evaluates.
- **Hydra overrides:** composed through Hydra's compose API. They give seed 43, `save_top_k=-1`, the epoch counts, early stopping (15,10) for r50 and (5,5) for slow, the paths and `mlflow.experiment_name=research_protocols`.

#### C. Aggregator (`06c_aggregate.py`), what it reproduced
I copied the October prediction files under the new tag names into a scratch layout and ran C on it. It reproduced the October summaries' test accuracies (val-selected checkpoint):

| run | accuracy |
|---|---|
| p0 r50 | 0.9935 |
| grp27a r50 | 0.5032 |
| grp27b r50 | 0.5266 |
| grp17 r50 | 0.6903 |
| grp27a slow | 0.5180 |

R2's macro-F1 is 0.4592, matching October's 0.459223. Every metric is cross-checked against sklearn inside the script, which aborts on any mismatch.

The reference section (built from the original October files) reproduces:
- `last.ckpt` accuracies: 0.9948, 0.4378, 0.5451, 0.7591, 0.3991.
- Final epochs from `epoch_curve.csv`: R2 0.3951, R3 0.5225, R4 0.7701, R5 0.3926; the repeat run's val-selected R5 is 0.5206.

New numbers from the October per-clip predictions (val-selected), all clips → conflict clips excluded:

| run | balanced accuracy | accuracy |
|---|---|---|
| R1 published | 0.9946 → 0.9952 | — |
| R2 grp27a | 0.4730 (no conflict clips in test) | — |
| R3 grp27b | 0.4869 → 0.4958 | 0.5266 → 0.5556 |
| R4 grp17 | 0.6469 → 0.6903 | 0.6903 → 0.7833 (243 conflict clips in its test set) |
| R5 grp27a slow | 0.4770 | — |

To exercise the P1/P2 paths I also built synthetic files: the scratch probe fit described under E, plus perturbed per-epoch and per-seed copies of it, with one fold missing, one inconsistent final epoch and one unknown tag. C reported 4/5 folds where expected, flagged the inconsistent final, ignored the unknown tag, and produced every table and the figure. Runtime was 7.9 s for 330 files at 304 MB.

#### D. Metadata tree (`06d_metadata_tree.py`)
Features come from the raw-release container headers; the tree is trained on each root's train split and scored on test.

| protocol | fold | n test | acc | bal acc | κ | majority |
|---|---|---|---|---|---|---|
| P0 | published | 771 | 0.534 | 0.476 | 0.235 | 0.422 |
| P1 | 0 / 1 / 2 / 3 / 4 | 1667 / 1489 / 1475 / 1524 / 1546 | 0.386 / 0.567 / 0.544 / 0.322 / 0.486 | 0.367 / 0.542 / 0.514 / 0.300 / 0.450 | 0.014 / 0.312 / 0.279 / −0.056 / 0.193 | 0.418 / 0.401 / 0.384 / 0.375 / 0.387 |
| P1 | pooled | 7701 | 0.459 | 0.430 | 0.144 | 0.393 |
| P2 | 0 / 1 / 2 / 3 / 4 | 1274 / 884 / 1293 / 1543 / 2707 | 0.276 / 0.258 / 0.531 / 0.345 / 0.226 | 0.252 / 0.278 / 0.449 / 0.407 / 0.411 | −0.043 / −0.021 / 0.215 / 0.094 / 0.052 | 0.598 / 0.567 / 0.418 / 0.467 / 0.638 |
| P2 | pooled | 7701 | 0.313 | 0.305 | −0.064 | 0.393 |

The top features are duration bin, width/height and audio tracks. The export-batch shortcut partly survives P1 and is gone under P2. These are raw-release headers; the models see uniform 320p re-encodes, so this measures batch structure in the release, not a feature the network can read directly.

#### E. Probe (`06e_probe.py`), smoke test
- 40 clips, evenly spaced over the sorted paths, all decoded with PyAV on CPU with 2 workers.
- The frame index is n_frames // 2 (300 → 150), the same frame the project's resnet50 evaluation takes.
- 2048-d features came out in 5–6 s, then 14 fits (P0, P1 f0–4, P2 f0–4, grp27a/b, grp17) wrote 56 files under the 06b tag names.
- Wall time 12–15 s; peak RSS of the whole process tree 1.54–1.68 GB.

I also ran the fit stage at full size on the October 32×32 centre-frame thumbnails as a test input (not the planned probe): 14 fits in 87 s at 719 MB. Balanced accuracy, all clips (excl. in brackets): published 0.949, P1 0.405 (0.380), P2 0.399 (0.391), grp17 0.729, grp27b 0.440, grp27a 0.394.

#### Wall time and memory
- A 1.1 s / 119 MB; D 2.1 s / 203 MB; C 4–8 s / ≤ 304 MB; E smoke 12–15 s / ≤ 1.68 GB; thumbnail fit test 87 s / 719 MB.
- At most about 0.93 GB was in use at once.

#### Deviations and open points
1. **Memory limit for E:** `ulimit -v 3000000` makes torch impossible to import here (the CUDA build maps 4.6 GB of address space at import). I ran E under `ulimit -v 6000000` and measured the real RSS instead. A, C and D ran under the 3 GB limit.
2. **More than one validation group:** the brief's rule picks 4–5 G1 groups under P1, not the plan's "one whole group". Under P2 it picks one group:
   - P2 folds 1–3 all share G2 group 13.
   - Fold 4's validation group (12) has no `high` clips.
   - Fold 0's validation group (9) contains 155 label-conflict clips; the exclusion was P1-only as briefed.
3. **P1 validation may not select well:** the thumbnail fit scored 0.09–0.25 balanced accuracy on the P1 validation groups, below chance, against 0.28–0.53 on test. The val-selected checkpoint under P1 may carry little signal; the final epoch and the full epoch curve are reported alongside it.
4. **`last.ckpt` is kept for seed 42:** it is Lightning's second save of the final epoch, not byte-identical to the epoch file, so each seed-42 run keeps 3 files. Use `rsync --exclude last.ckpt` to avoid transferring it twice.
5. **Train split only:** the probe and the metadata tree train on the root's train split, not train+val, so they see the same clips as the deep runs. The probe's "final" rule is the same single fit as its "best".
6. **October references:** a "final epoch" for R2–R5 exists only as accuracy and macro-F1 in `epoch_curve.csv`; there are no per-clip predictions behind it, so no balanced accuracy, κ or exclusion.
7. **Runtime on the host:** about 260 min with per-epoch evaluation, against the plan's 3 h. The R50, SLOW and probe time estimates can be overridden.
8. **Extra summary folder:** 06b adds `$OUT/state/` for the trained/done markers and the best/final epoch json.
9. **Host setup:** the roots are gitignored. Run `python scripts/research/06_make_protocol_roots.py` on the host after `dvc pull` and before 06b. The committed DVC pointer lists 210 files, matching the local `outputs/research`, so the session tables should be in it. `DRY=1` will catch any missing root.
10. **DVC status:** writing `outputs/research/protocols/` inside the DVC-tracked `outputs/research` will make `dvc status` report it as modified.
11. **Results doc not yet written:** I did not generate `docs/research/12_leak_gradient_results.md` or `docs/research/img/leak_gradient.png`, since there are no GPU results yet. Running `06c_aggregate.py` with its defaults after the rsync creates both. A preview from the synthetic test is in the scratchpad.

#### Files created or changed
Created: `scripts/research/06_make_protocol_roots.py`, `06b_run_protocols.sh`, `06c_aggregate.py`, `06d_metadata_tree.py`, `06e_probe.py`; `data/p1_f0..4`, `data/p2_f0..4` (gitignored); `outputs/research/protocols/{roots.csv,val_groups.json,metadata_tree.csv}`; torchvision ResNet50 weights in `~/.cache/torch/hub/checkpoints/`. Changed: `.gitignore` (2 lines).

**Supervisor notes (2026-11-08).** Sent back for three fixes before launch: (1) exclude label-conflict groups from validation under both protocols and require all three classes (≥ 30 clips each) in every validation set, because P2 fold 4 had no `high` clips in validation and fold 0's validation group held 155 conflict clips; (2) per-epoch test evaluation for seed-42 runs only (switch `EPOCH_CURVE`), seeds 43/44 evaluate best-by-val and final only; (3) rerun roots and DRY. The thumbnail-probe result under P1 validation (below chance) is noted: P1 validation sets are a handful of small single-class events, so the val-selected checkpoint is reported next to the final epoch, as planned. The metadata tree already answers one plan question before any GPU time: the export-batch shortcut survives P1 (pooled balanced accuracy 0.43 vs 0.39 majority) and vanishes under P2 (0.31).


## 2026-11-08 — GPU experiment code, second round (same agent, verbatim hand-back)

### Report: the three pre-launch changes

All three changes are in, and the roots and checks pass. P2 validation now excludes the conflict groups and has all three classes in every fold, fold 4 included. The DRY run projects 250 min, down from 260.

#### 1. Validation rule in `06_make_protocol_roots.py`
- **Conflict exclusion, both protocols:** the excluded groups are now taken from the data — every group holding a clip with event_id 9, 11 or 18.
  - P1: G1 groups 34, 40, 44. The script checks this equals the event mapping.
  - P2: G2 groups 9, 10, 15.
- **Class coverage (at least 30 clips of each class):**
  - Whether coverage is possible within the 15% cap is decided exactly, by a small search for the smallest group set that covers all three classes. I checked that search against brute force on 300 random cases with 0 mismatches.
  - While coverage is not yet met but still possible, the L1 greedy only picks groups after which coverage stays reachable within the cap.
  - It stops at 8% of the pool once coverage is met (or if coverage is impossible). The 1% floor, the 15% cap and the smaller-id tie-break are unchanged.
- **Stored per fold** in `val_groups.json`: validation class counts, whether coverage was achievable, whether it was achieved, and the excluded groups. Each fold's line also prints them.
- **Coverage:** achievable and achieved in all ten folds.
- **What changed on disk:** P1 is identical to the previous version, every manifest byte for byte. Only `p2_f0` and `p2_f4` train/val changed; all test sets are unchanged.

Counts are low/mid/high; validation groups are in the order the greedy added them:

| root | train | val | val groups | val % of pool, L1 | coverage | test |
|---|---|---|---|---|---|---|
| p1_f0 | 5517 (2147/1458/1912) | 517 (185/185/147) | 18, 7, 33, 19, 39 | 8.6%, 0.171 | yes | 1667 (696/398/573) |
| p1_f1 | 5670 (2221/1470/1979) | 542 (210/185/147) | 19, 7, 39, 21, 33 | 8.7%, 0.150 | yes | 1489 (597/386/506) |
| p1_f2 | 5709 (2277/1465/1967) | 517 (185/185/147) | 18, 7, 39, 19, 33 | 8.3%, 0.186 | yes | 1475 (566/391/518) |
| p1_f3 | 5574 (2220/1386/1968) | 603 (237/219/147) | 18, 7, 33, 21, 35 | 9.8%, 0.207 | yes | 1524 (571/436/517) |
| p1_f4 | 5587 (2193/1489/1905) | 568 (237/122/209) | 18, 5, 35, 21 | 9.2%, 0.094 | yes | 1546 (598/430/518) |
| p2_f0 | 5663 (1990/1483/2190) | 764 (276/180/308) | 12, 2 | 11.9%, 0.046 | yes | 1274 (762/378/134) |
| p2_f1 | 6072 (2294/1386/2392) | 745 (233/378/134) | 13 | 10.9%, 0.497 | yes | 884 (501/277/106) |
| p2_f2 | 5663 (2254/1219/2190) | 745 (233/378/134) | 13 | 11.6%, 0.516 | yes | 1293 (541/444/308) |
| p2_f3 | 5413 (2074/1197/2142) | 745 (233/378/134) | 13 | 12.1%, 0.503 | yes | 1543 (721/466/356) |
| p2_f4 | 4249 (2292/1187/770) | 745 (233/378/134) | 13 | 14.9% (cap 749.1), 0.388 | yes | 2707 (503/476/1728) |

P2 fold 4 now has 134 `high` clips in validation (group 13: 745 clips against a cap of 749.1).

Label-conflict clips per split. Validation now holds none in every fold:

| root | train | val | test |
|---|---|---|---|
| p1_f0 | 500 | 0 | 380 |
| p1_f1 | 880 | 0 | 0 |
| p1_f2 | 380 | 0 | 500 |
| p1_f3 | 880 | 0 | 0 |
| p1_f4 | 880 | 0 | 0 |
| p2_f0 | 880 | 0 | 0 |
| p2_f1 | 725 | 0 | 155 |
| p2_f2 | 880 | 0 | 0 |
| p2_f3 | 500 | 0 | 380 |
| p2_f4 | 535 | 0 | 345 |

Things you should know:
- **Same validation group in four folds:** P2 folds 1–4 all validate on G2 group 13. Fold 0 uses groups 12 + 2.
- **Fold 4 training set:** group 12 (456 clips, low/mid) went back into fold 4's training set, so fold 4 trains on 4,249 clips with only 770 `high`.
- **Metadata tree rerun:** `metadata_tree.csv` was computed on the old P2 roots, so I reran the unchanged 06d to keep it consistent.
  - P1 rows are unchanged.
  - P2 pooled is now accuracy 0.321, balanced accuracy 0.314, κ −0.057 (was 0.313 / 0.305 / −0.064).
  - Fold 4 is now accuracy 0.300, balanced accuracy 0.467, κ 0.112.

#### 2. `EPOCH_CURVE` switch in `06b_run_protocols.sh` (default `seed42`; also `all`, `none`)
- **With the curve:** every epoch is evaluated on test, as before. By default this covers r50 on P1, r50 on P2 seed 42, and SLOW on P2.
- **Without it (default: seeds 43 and 44):** the best-by-validation checkpoint is evaluated on val and test, and only the final epoch on test (tag `<run>_ep<NN>_test`, which 06c already reads as the final epoch). Then all checkpoints are deleted, as before.
- **Safety and budget:** the completeness check before cleanup counts only the evaluations that mode expects. An invalid value aborts with exit 4. The budget now tracks measured time separately for each recipe and mode.
- **New estimates without the curve** (overridable): r50 7 min (`R50_NOCURVE_EST_MIN`), SLOW 13 min (`SLOW_NOCURVE_EST_MIN`). With the curve they stay at r50 8 and SLOW 18.
- **Stub tests:**
  - Seed 42 gets all epochs, then keeps best, final and `last.ckpt`.
  - Seed 44 gets best plus final only, then all its checkpoints are deleted.
  - With `EPOCH_CURVE=none`, a failed final-epoch evaluation keeps the checkpoints, and the rerun resumes and finishes.

#### 3. DRY=1 on the regenerated roots
Exit code 0, 131 lines, 26 blocks. Nothing was written (`outputs/research_gpu` does not exist).
```
DRY SUMMARY: to run: probe=1 r50=20 slow=5 (already done, skipped: 0); dropped by budget: none
DRY SUMMARY: EPOCH_CURVE=seed42: 15 of 25 training runs get every epoch evaluated on test, the others best (val, test) + final epoch (test)
DRY SUMMARY: projected wall time 250 min (estimates per run incl. evaluation: probe 10; r50 8 with curve, 7 without; slow 18 with curve, 13 without)
DRY SUMMARY: all needed roots and inputs exist
```
The 250 min breaks down as probe 10 + r50 P1 40 + r50 P2 seed 42 40 + seeds 43/44 70 + SLOW 90. With `BUDGET_MIN` set to 200 or 230 the SLOW block is dropped (160 min); at 250 nothing is dropped.

#### Runtime and memory
06 took 2.3 s at 114 MB peak; 06d took about 2 s; the DRY run under 1 s.

#### Files changed this round
- `scripts/research/06_make_protocol_roots.py`, `scripts/research/06b_run_protocols.sh`; regenerated `data/p2_f0`, `data/p2_f4` (train/val), `outputs/research/protocols/{roots.csv,val_groups.json,metadata_tree.csv}`.

No dvc, git or ssh.

**Supervisor notes.** Accepted; scripts synced to the host over the direct address and launched with `/workspace/host_launch.sh` (roots rebuilt on the host, DRY check, then the runner under nohup). Known compromise: P2 folds 1–4 share validation group 13, the only three-class group that fits under the 15% cap once the conflict groups are excluded; validation is used for epoch ranking only and the final epoch is reported alongside.


## 2026-11-08 — Probe fix for clips missing from the 320p set (same agent, verbatim hand-back)

### Report: 06e_probe.py fix for clips missing from the 320p set

The fix is in and the CPU test passes: a missing file and a corrupt file are logged and skipped, the run finishes with exit 0, and the skip count reaches the summaries. Only `scripts/research/06e_probe.py` changed.

**The host failure may have another cause; check before relaunching.** Here, the previous 06e did not abort on these clips: its decode worker caught the error (`decode_centre` on `videos/high/view2521.mp4` returned ok=False with "FileNotFoundError …"). It printed `extract: FAILED <path>` and carried on, and the fit stage dropped those clips. If the probe exited non-zero on the host, the real reason is probably further down `outputs/research_gpu/_summary/logs/probe.log`; its last lines before the exit are worth reading before rerunning with `RUNS=probe`.

#### What changed
About 60 changed lines; nothing else touched.
- **(a) Clip universe:** a new `--groups` option, default `outputs/research/sessions/union_groups.csv`. The clip list is exactly its paths (path, label, split_published), sorted by path; `--limit` takes evenly spaced rows from it. P0 now uses `split_published` from that file instead of the published manifests, so the 4 untranscoded paths never appear. `--data-root` is now only where `videos/` is read. 06b's call is unchanged and still valid.
- **(b) Skip and log:** the decode worker checks the file exists first ("missing file"); decode errors are caught as before, and the transform step has its own guard. Each skipped clip prints `extract: SKIPPED <path>: <reason>` and stays in the feature file with ok=False. `n_skipped` and the list of skipped clips go into the feature file's meta, and `n_skipped_extraction` into every fit `summary_*.json`.
- **(c) Fits:** they run on the clips that were extracted.
- **Feature reuse:** a feature file left by the old 7,705-clip universe holds a different set of paths, so it is re-extracted, not reused.

#### Test output (CPU, `CUDA_VISIBLE_DEVICES=""`, nice, `ulimit -v 6000000`)
Scratch copy of union_groups.csv: 40 real rows plus `videos/high/view0.mp4` (no file) and `videos/low/view0.mp4` (random bytes). `--limit 42 --workers 2 --threads 3`: both fakes SKIPPED (missing file; InvalidDataError), `features: 40/42 clips ok, 2 skipped, 6s`, all 14 fits ran, `PEAKRSS tree_peak_rss_mb=1682 wall_s=12.6 rc=0`. Feature meta n 42 / n_ok 40 / n_skipped 2; `summary_p0_probe_s0_best_test.json` has `n_skipped_extraction` = 2; neither fake path appears in any prediction file. Two runs against the defaults: `--limit 40` gives 40/40 ok; a second call reuses the feature file.

**Supervisor notes.** The agent was right that the missing clips were not the cause. The host log shows the extraction completed (7,701 of 7,705 ok in 47 s on the GPU) and the fit stage then crashed loading the feature file with `allow_pickle=False` because the old version saved the path array with object dtype. The new version saves unicode paths; I added a try/except around the reuse check so the stale host file cannot crash it again, moved that file aside on the host, and synced the script. The probe block reruns with `RUNS=probe` after the training blocks.

**Supervisor addendum (after the training blocks).** The relaunched probe failed the same way with a freshly extracted feature file: on the host pandas is 3.1.1, where `Series.values.astype(str)` on a string column still yields an object array, so `np.savez` stored object paths and `np.load(allow_pickle=False)` refused them. Locally (pandas 2.3.3) the same line gives a unicode array, which is why the agent's CPU test passed. Fixed by building the path array explicitly, `np.array([str(p) for p in df.path], dtype=str)`; the third run completed (extraction 46 s on the GPU, 14 fits in 83 s). The 25 training runs finished before that with no failures (211 min on the RTX 5090); results are aggregated in `12_leak_gradient_results.md`, the per-group behaviour in `12b_scene_collapse.md` (new script `06f_scene_collapse.py`, written by the supervisor), interpretation in `13_leak_gradient_interpretation.md`.
