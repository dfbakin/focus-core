# OUC-CGE frame cache, scene clusters and grouped splits rebuilt on the CPU workstation

Date: 2026-11-07/08. Host: local workstation, 8 CPU cores, 7.6 GiB RAM, no GPU used
(`CUDA_VISIBLE_DEVICES=""` on every command). Branch `research/ouc-cge`. Data: the 320p copies
`data/ouc-cge-320/videos` (7,701 files) and the manifests `data_labels/{train,val,test}.csv`
(identical bytes in `data/ouc-cge/` and `data/ouc-cge-320/`). Nothing under `scripts/review/`,
`docs/review/`, `src/`, `data/` or `outputs/review_grouped/` was modified. Nothing was committed.

Sections 1-6 report what the commands printed. Inferences are marked **Inference**.

## Summary

| step | result | wall time | peak RSS |
|---|---|---|---|
| streaming frame cache vs `01_frame_cache.py --thumbs-only`, 10 clips/split sample | all 12 keys identical in dtype, shape and values | 3.9 s | 126 MB |
| same, 30 random clips < 15 s (temporary root), 3 hint modes | all identical | about 2 s each | original 267 MB, streaming 123 MB |
| same, edge root (4 missing files, malformed row, six 4-frame clips) | all identical | about 1 s | |
| independent long-clip check (58 padded clips + 40 random) vs the cache | all identical (meta, thumbs, hires first/last) | 4 min 28 s | 296 MB |
| full cache `outputs/review_cache` (7,705 records, 7,701 decodable) | written; 1 read pass per clip, 0 second passes | **11 min 25 s** | **1,020 MB** main process, **99 MB** per worker |
| `03b_scene_clusters_thumb.py` (torch CPU) | `scene_clusters.csv` **byte-identical**; 10 of 11 thresholds same partition; thr 12.5 differs on 6 clips | 23 s | 2.0 GB |
| `run_07_splits.sh` with `CACHE` | `split_grp27a/27b/17.csv` **identical**; leakage statistics equal up to 1-2 float32 ulp | 2 min 52 s | 2.6 GB |

## 1. Why a streaming cache

`scripts/review/01_frame_cache.py::decode()` appends every decoded BGR frame of a clip to a
Python list and only then picks three frames. The 320p copies contain 57 padded clips of 9,619
frames (`high/view1831`-`view1887`) and one of 9,659 frames (`high/view1888`). At 568x320x3 bytes
per frame, each is about 5.2 GB of decoded frames, which does not fit in 7 GB of RAM.

## 2. What `scripts/research/02_frame_cache_stream.py` does

Read from the code of `01_frame_cache.py` (lines 44-72, 109-161) and `common.py`:

- Manifests: `common.load_all_manifests(root)`, the same function. It drops the malformed
  `videos/anonymity/a.mp4` row (label NaN), which leaves 7,705 records (6,164 / 770 / 771).
- Decode: `cv2.VideoCapture(path)` with default arguments, `cv2.setNumThreads(1)`,
  `CAP_PROP_FRAME_COUNT` → `n_reported`, `CAP_PROP_FPS` → `fps`, then `cap.read()` until it fails.
  `n = len(frames)` is the number of successful `read()` calls. The frames taken are
  `for i in (0, n // 2, n - 1)`. Each is converted with `cv2.cvtColor(f, cv2.COLOR_BGR2RGB)` and
  resized with `cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)`, giving uint8
  `(3, 32, 32, 3)`. `h, w` come from `frames[0].shape`.
- Unreadable clips: if the file does not open, the record keeps `ok=False`, `n_reported=0`,
  `fps=0.0`, `w=h=0` and zero thumbnails. If it opens but yields no frame, `n_reported` and `fps`
  are kept.
- Saved per split as `frames_{split}.npz` (`np.savez_compressed`), with the keys in this order:
  `path` (`<U24`), `label` int64, `split` (`<U5/<U3/<U4`), `cls_folder` (`<U4`), `view_id` int64,
  `ok` bool, `n_frames` int64, `n_reported` int64, `fps` float32, `w` int64, `h` int64,
  `thumbs` uint8 (N, 3, 32, 32, 3). With `--thumbs-only` there is no `feats` key.

The streaming script uses the same calls in the same order. It keeps only frame 0, the frame at
index `c = n_hint // 2`, and the most recent frame, which is frame `n - 1` when the loop ends.

- `n_hint`, fast path: `n_frames` from `outputs/research/clip_audit.csv` (an exact sequential
  `grab()` count of the same 320p files). It is used only if the audit row's `file_bytes` equals
  the file size on disk.
- `n_hint`, fallback: a counting pass with `cap.grab()`.
- Exactness does not depend on the hint. After the `read()` pass the actual count `n` is known. If
  `n != n_hint`, the clip is read a second time with `c = n // 2`, and the script raises an error
  if the second count differs from the first. The centre frame is therefore always frame `n // 2`
  of the `read()` count, as in the original.
- Jobs run longest first (by hint) with `chunksize=1`. The arrays and the npz files are built with
  the same expressions as in `01`, including `meta = np.array([...])` from Python scalars and the
  `.astype(...)` casts.
- Extra output, `frames_{split}_hires.npz`: keys `path, view_id, label, split, ok, first, last`.
  `first` and `last` are the first and last decoded frames, BGR→RGB, `cv2.resize(.., (128, 72),
  INTER_AREA)`, uint8 `(N, 72, 128, 3)`, zero for unreadable clips. These are for a later
  temporal-reconstruction analysis.
- Extra output, `frames_stream_meta.csv`: per clip the hint source, `n_hint`, number of passes,
  worker peak RSS and `n_frames`.
- CLI: `--root --out --workers --limit` as in `01`. `--thumbs-only` is accepted and has no effect
  (always on). `--audit PATH`, `--no-audit` (always count with `grab()`), `--no-hires`.

## 3. Verification against the original script

Comparison tool: `scripts/research/02b_compare_frame_caches.py`. For each split it checks that
the key lists are equal (same order) and, for every key, dtype equality, shape equality and
`np.array_equal`. With `--hires` it also checks the hires file: its keys, that its
`path/view_id/label/split/ok` equal the reference main file, the dtype and shape of `first`/`last`,
and zero rows for `ok=False`.

### 3.1 First 10 clips per split vs `outputs/research/frame_cache_limit10/` (made by the original)

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02_frame_cache_stream.py --root data/ouc-cge-320 \
    --out outputs/research/frame_cache_stream_limit10 --workers 8 --limit 10      # 3.88 s, max RSS 128,568 KB
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02b_compare_frame_caches.py \
    --a outputs/research/frame_cache_limit10 --b outputs/research/frame_cache_stream_limit10 --hires
```

Output, verbatim (`outputs/research/fc_check/compare_stream_limit10.txt`):

```
[train] keys equal (same order): True  ['path', 'label', 'split', 'cls_folder', 'view_id', 'ok', 'n_frames', 'n_reported', 'fps', 'w', 'h', 'thumbs']
  path       dtype <U24   /<U24    shape (10,)               /(10,)                array_equal=True -> OK
  label      dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  split      dtype <U5    /<U5     shape (10,)               /(10,)                array_equal=True -> OK
  cls_folder dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  view_id    dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  ok         dtype bool   /bool    shape (10,)               /(10,)                array_equal=True -> OK
  n_frames   dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  n_reported dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  fps        dtype float32/float32 shape (10,)               /(10,)                array_equal=True -> OK
  w          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  h          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  thumbs     dtype uint8  /uint8   shape (10, 3, 32, 32, 3)  /(10, 3, 32, 32, 3)   array_equal=True -> OK
  hires: keys=['path', 'view_id', 'label', 'split', 'ok', 'first', 'last'] OK; path=OK; view_id=OK; label=OK; split=OK; ok=OK; first uint8 (10, 72, 128, 3) OK zero-for-bad=True; last uint8 (10, 72, 128, 3) OK zero-for-bad=True
[val] keys equal (same order): True  ['path', 'label', 'split', 'cls_folder', 'view_id', 'ok', 'n_frames', 'n_reported', 'fps', 'w', 'h', 'thumbs']
  path       dtype <U24   /<U24    shape (10,)               /(10,)                array_equal=True -> OK
  label      dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  split      dtype <U3    /<U3     shape (10,)               /(10,)                array_equal=True -> OK
  cls_folder dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  view_id    dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  ok         dtype bool   /bool    shape (10,)               /(10,)                array_equal=True -> OK
  n_frames   dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  n_reported dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  fps        dtype float32/float32 shape (10,)               /(10,)                array_equal=True -> OK
  w          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  h          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  thumbs     dtype uint8  /uint8   shape (10, 3, 32, 32, 3)  /(10, 3, 32, 32, 3)   array_equal=True -> OK
  hires: keys=['path', 'view_id', 'label', 'split', 'ok', 'first', 'last'] OK; path=OK; view_id=OK; label=OK; split=OK; ok=OK; first uint8 (10, 72, 128, 3) OK zero-for-bad=True; last uint8 (10, 72, 128, 3) OK zero-for-bad=True
[test] keys equal (same order): True  ['path', 'label', 'split', 'cls_folder', 'view_id', 'ok', 'n_frames', 'n_reported', 'fps', 'w', 'h', 'thumbs']
  path       dtype <U24   /<U24    shape (10,)               /(10,)                array_equal=True -> OK
  label      dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  split      dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  cls_folder dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  view_id    dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  ok         dtype bool   /bool    shape (10,)               /(10,)                array_equal=True -> OK
  n_frames   dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  n_reported dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  fps        dtype float32/float32 shape (10,)               /(10,)                array_equal=True -> OK
  w          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  h          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  thumbs     dtype uint8  /uint8   shape (10, 3, 32, 32, 3)  /(10, 3, 32, 32, 3)   array_equal=True -> OK
  hires: keys=['path', 'view_id', 'label', 'split', 'ok', 'first', 'last'] OK; path=OK; view_id=OK; label=OK; split=OK; ok=OK; first uint8 (10, 72, 128, 3) OK zero-for-bad=True; last uint8 (10, 72, 128, 3) OK zero-for-bad=True
ALL IDENTICAL
```

### 3.2 30 random clips shorter than 15 s, original and streaming script on the same root

`01_frame_cache.py` can restrict the clip set only with `--limit`, so
`scripts/research/02c_make_tmp_roots.py` builds a small dataset root with
`common.write_dataset_root`: copied manifests in the project format plus a relative `videos`
symlink to `data/ouc-cge-320/videos`, as `09_materialize_splits.py` does. The root is
`outputs/research/tmp_root/`. It holds 10 clips per published split, drawn with numpy seed 0
from the clips with audit `duration_s < 15`. Durations range from 0.13 to 11.47 s and frame
counts from 4 to 344; the list is in `tmp_root/selection.csv`.

```
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/review/01_frame_cache.py --root outputs/research/tmp_root \
    --out outputs/research/fc_check/orig_tmp_root --thumbs-only --workers 8           # 1.98 s, max RSS 273,232 KB
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02_frame_cache_stream.py --root outputs/research/tmp_root \
    --out outputs/research/fc_check/stream_audit_tmp_root --workers 8                 # 2.07 s, max RSS 125,668 KB
  ... --out outputs/research/fc_check/stream_grab_tmp_root --no-audit                 # grab() counting pass
  ... --out outputs/research/fc_check/stream_badhints_tmp_root \
      --audit outputs/research/tmp_root/bad_hints_audit.csv                           # every hint wrong by -5/+1/+7/-1
```

The three streaming modes exercise the three code paths. The run logs report: audit hints,
"30 clips x1"; `--no-audit`, "0 from (audit disabled), 30 by a grab() counting pass … 30 clips
x1"; wrong hints, "0 clips x1, 30 clips x2 (hint != read() count)".

Output of `02b_compare_frame_caches.py --a fc_check/orig_tmp_root --b fc_check/stream_audit_tmp_root --hires`,
verbatim (`outputs/research/fc_check/compare_stream_audit_tmp_root.txt`):

```
[train] keys equal (same order): True  ['path', 'label', 'split', 'cls_folder', 'view_id', 'ok', 'n_frames', 'n_reported', 'fps', 'w', 'h', 'thumbs']
  path       dtype <U24   /<U24    shape (10,)               /(10,)                array_equal=True -> OK
  label      dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  split      dtype <U5    /<U5     shape (10,)               /(10,)                array_equal=True -> OK
  cls_folder dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  view_id    dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  ok         dtype bool   /bool    shape (10,)               /(10,)                array_equal=True -> OK
  n_frames   dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  n_reported dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  fps        dtype float32/float32 shape (10,)               /(10,)                array_equal=True -> OK
  w          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  h          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  thumbs     dtype uint8  /uint8   shape (10, 3, 32, 32, 3)  /(10, 3, 32, 32, 3)   array_equal=True -> OK
  hires: keys=['path', 'view_id', 'label', 'split', 'ok', 'first', 'last'] OK; path=OK; view_id=OK; label=OK; split=OK; ok=OK; first uint8 (10, 72, 128, 3) OK zero-for-bad=True; last uint8 (10, 72, 128, 3) OK zero-for-bad=True
[val] keys equal (same order): True  ['path', 'label', 'split', 'cls_folder', 'view_id', 'ok', 'n_frames', 'n_reported', 'fps', 'w', 'h', 'thumbs']
  path       dtype <U24   /<U24    shape (10,)               /(10,)                array_equal=True -> OK
  label      dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  split      dtype <U3    /<U3     shape (10,)               /(10,)                array_equal=True -> OK
  cls_folder dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  view_id    dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  ok         dtype bool   /bool    shape (10,)               /(10,)                array_equal=True -> OK
  n_frames   dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  n_reported dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  fps        dtype float32/float32 shape (10,)               /(10,)                array_equal=True -> OK
  w          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  h          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  thumbs     dtype uint8  /uint8   shape (10, 3, 32, 32, 3)  /(10, 3, 32, 32, 3)   array_equal=True -> OK
  hires: keys=['path', 'view_id', 'label', 'split', 'ok', 'first', 'last'] OK; path=OK; view_id=OK; label=OK; split=OK; ok=OK; first uint8 (10, 72, 128, 3) OK zero-for-bad=True; last uint8 (10, 72, 128, 3) OK zero-for-bad=True
[test] keys equal (same order): True  ['path', 'label', 'split', 'cls_folder', 'view_id', 'ok', 'n_frames', 'n_reported', 'fps', 'w', 'h', 'thumbs']
  path       dtype <U24   /<U24    shape (10,)               /(10,)                array_equal=True -> OK
  label      dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  split      dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  cls_folder dtype <U4    /<U4     shape (10,)               /(10,)                array_equal=True -> OK
  view_id    dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  ok         dtype bool   /bool    shape (10,)               /(10,)                array_equal=True -> OK
  n_frames   dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  n_reported dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  fps        dtype float32/float32 shape (10,)               /(10,)                array_equal=True -> OK
  w          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  h          dtype int64  /int64   shape (10,)               /(10,)                array_equal=True -> OK
  thumbs     dtype uint8  /uint8   shape (10, 3, 32, 32, 3)  /(10, 3, 32, 32, 3)   array_equal=True -> OK
  hires: keys=['path', 'view_id', 'label', 'split', 'ok', 'first', 'last'] OK; path=OK; view_id=OK; label=OK; split=OK; ok=OK; first uint8 (10, 72, 128, 3) OK zero-for-bad=True; last uint8 (10, 72, 128, 3) OK zero-for-bad=True
ALL IDENTICAL
```

The other two modes print the same per-key lines (every key `OK`) and end in `ALL IDENTICAL`, with
exit code 0 (`fc_check/compare_stream_grab_tmp_root.txt`,
`fc_check/compare_stream_badhints_tmp_root.txt`).

### 3.3 Edge cases (`outputs/research/tmp_root_edge/`)

This root holds the 4 clips missing from the 320p copies (`high/view2521, 2630, 2631, 2632`), the
6 shortest readable clips (4 frames each, so `n // 2 = 2`), 2 ordinary clips per split, and the
malformed `videos/anonymity/a.mp4` line appended to `test.csv`. Both scripts printed the same 4
`BAD … train` lines, "saved train: 12 clips, ok=8" and the same percentile lines. The comparison
ended in `ALL IDENTICAL` for audit hints and for `--no-audit`
(`fc_check/compare_stream_audit_tmp_root_edge.txt`, `fc_check/compare_stream_grab_tmp_root_edge.txt`).
It also confirms that the malformed row is dropped and that unreadable clips keep `ok=False`,
`n_reported=0`, `fps=0`, `w=h=0` and zero thumbnail and hires rows.

### 3.4 Long padded clips (the original cannot run on them here)

`01_frame_cache.py` would need about 5 GB per padded clip, so `scripts/research/02d_long_clip_check.py`
re-implements its index logic without storing full frames:

- It calls `cv2.VideoCapture(path)` and `cap.read()` until failure, but appends the 32x32 RGB
  INTER_AREA thumbnail of every frame (about 3 KB) instead of the frame itself, and keeps the
  full first and last frame.
- After the loop it takes `thumbs[0], thumbs[n // 2], thumbs[n - 1]` with `n = len(list)`. No
  frame-count hint is involved.
- It compares `n_frames, n_reported, fps, w, h, thumbs` with `outputs/review_cache/frames_*.npz`,
  and the 128x72 first and last frames with `frames_*_hires.npz`.
- Clips checked: all 58 clips with audit `n_frames > 9000` (57 × 9,619 and 1 × 9,659 frames),
  plus 40 random readable clips (seed 0).

The first attempt was OOM-killed (see section 7) and wrote nothing. The rerun after the fix:

```
(ulimit -v 4500000; CUDA_VISIBLE_DEVICES="" /usr/bin/time -v .venv/bin/python scripts/research/02d_long_clip_check.py \
    --root data/ouc-cge-320 --cache outputs/review_cache --workers 4 > outputs/research/long_clip_check.log 2>&1)
```

```
checking 58 long clips (n_frames>9000) + 40 random clips
random               n= 40 meta_equal=40/40 thumbs_equal=40/40 hires_first_equal=40/40 hires_last_equal=40/40
long (>9000 frames)  n= 58 meta_equal=58/58 thumbs_equal=58/58 hires_first_equal=58/58 hires_last_equal=58/58
ALL IDENTICAL
	Elapsed (wall clock) time (h:mm:ss or m:ss): 4:28.02
	Maximum resident set size (kbytes): 295580
	Exit status: 0
```

The per-clip results are in `outputs/research/long_clip_check.csv`. This checks the streaming
cache against a reimplementation of `01`'s index logic, not against `01` itself.

## 4. Full run

```
CUDA_VISIBLE_DEVICES="" nohup /usr/bin/time -v .venv/bin/python scripts/research/02_frame_cache_stream.py \
    --root data/ouc-cge-320 --out outputs/review_cache --workers 8 > outputs/research/frame_cache_stream.log 2>&1 &
```

From `outputs/research/frame_cache_stream.log`:

```
manifest rows: 7705 train=6164 val=770 test=771
frame-count hints: 7701 from /home/dfbakin/Documents/focus-core/outputs/research/clip_audit.csv, 4 by a grab() counting pass
decoded 500/7705 in 284s
...
decode done in 665s
read passes: 7705 clips x1, 0 clips x2 (hint != read() count); worker peak RSS max 99 MB
unreadable clips: 4
  BAD videos/high/view2631.mp4 train
  BAD videos/high/view2521.mp4 train
  BAD videos/high/view2632.mp4 train
  BAD videos/high/view2630.mp4 train
saved train: 6164 clips, ok=6160
saved val: 770 clips, ok=770
saved test: 771 clips, ok=771
n_frames percentiles (1,5,25,50,75,95,99): [   9.   77.  158.  165.  300.  355. 5068.]
clips with reported frame count < 64 (dropped by OUCCGEDataset window filter): 117 of 7705
reported != actual frame count: 0
total wall 684s; main process peak RSS 1020 MB
	Elapsed (wall clock) time (h:mm:ss or m:ss): 11:25.48
	User time (seconds): 4234.49   System time (seconds): 451.29   Percent of CPU this job got: 683%
	Maximum resident set size (kbytes): 1044372
	Exit status: 0
```

- Wall time: 11 min 25 s. The 58 long clips were scheduled first and were done after about 280 s.
  Another agent was using the CPU at the same time (load average about 30).
- Peak RSS: 99 MB per worker; 1,020 MB for the main process. The main process figure comes from
  the hires arrays (426 MB for all clips) plus the boolean-mask copies made while saving the
  train split.
- No clip needed a second read pass, so the audit's `grab()` count equals the `read()` count for
  all 7,701 decodable files.
- The 117 clips under 64 frames match the number in the review README.

Output files are in `outputs/review_cache/`:

- `frames_train.npz` (51.7 MB), `frames_val.npz` (6.5 MB), `frames_test.npz` (6.5 MB);
- `frames_train_hires.npz` (277 MB), `frames_val_hires.npz` (34.7 MB), `frames_test_hires.npz` (34.7 MB);
- `frames_stream_meta.csv`.

## 5. Scene clusters (`03b_scene_clusters_thumb.py`)

**Memory estimate before running.** No GPU is visible, so `pairwise_min_rmse` runs
`torch.cdist` on the CPU. The thumbnails as float32 take 284 MB, each (N, N) float32 matrix
237 MB, and at most about four such matrices are alive at once. With the contiguous copies of
the inputs this comes to about 1.5-2 GB, below the 4 GB limit, so it was run. Measured peak:
2.0 GB.

```
CUDA_VISIBLE_DEVICES="" /usr/bin/time -v .venv/bin/python scripts/review/03b_scene_clusters_thumb.py \
    --cache outputs/review_cache --out outputs/review_out > outputs/research/scene_clusters_03b.log 2>&1
#   Elapsed 0:23.20, Maximum resident set size 2,079,952 KB, exit 0; chosen thr=27.5
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/review/10_compare_with_published.py --clusters outputs/review_out
```

It was run without `--chains`, because `exp2_chains.csv` needs the ResNet features. This affects
only the `n_chains` column of the cluster table and the `frac_chains_in_single_cluster` summary
key; the clusters do not depend on it.

Output of `10_compare_with_published.py`, verbatim (exit code 1;
`outputs/research/compare_clusters_published.txt`):

```
## Scene clusters

| file / threshold | clips | clusters (published / rebuilt) | result |
|---|---|---|---|
| scene_clusters.csv | 7701 | 101 / 101 | byte-identical file |
| thr 10 | 7701 | 1626 / 1626 | same partition |
| thr 12.5 | 7701 | 1127 / 1126 | DIFFERS (6 clips) |
| thr 15 | 7701 | 784 / 784 | same partition |
| thr 17.5 | 7701 | 530 / 530 | same partition |
| thr 20 | 7701 | 370 / 370 | same partition |
| thr 22.5 | 7701 | 250 / 250 | same partition |
| thr 25 | 7701 | 165 / 165 | same partition |
| thr 27.5 | 7701 | 101 / 101 | same partition |
| thr 30 | 7701 | 54 / 54 | same partition |
| thr 35 | 7701 | 19 / 19 | same partition |
| thr 40 | 7701 | 8 / 8 | same partition |
```

### 5.1 The thr 12.5 difference

The 6 clips are `high/view1191, 1195, 1199, 1203, 1207` (published cluster 35) and
`high/view1211` (published cluster 533, a singleton). In the rebuild `view1211` joins cluster 35.
The only cross pair below 23 is `view1207` (last frame) to `view1211` (first frame):

| computation | d(view1207, view1211) |
|---|---|
| exact, float64 direct differences on the rebuilt thumbnails | **12.4986848** |
| torch CPU float32 `cdist`, matmul form (what 03b uses here) | 12.4986982 |
| torch CPU float32 `cdist`, `donot_use_mm` | 12.4986849 |
| published run | > 12.5 (implied by the singleton; the value is not stored) |

The pair lies 0.0013 below the threshold. The CPU run links it, which is the correct result for
the exact distance. The published run did not link it. All other thresholds, including 27.5, give
the same partition.

### 5.2 Size of the float32 error in the published and rebuilt distances

Output of the exact-distance check (`outputs/research/compare_03b_exact_nn.txt`). It compares the
nearest-train distance of each test clip, published `exp3b_nn_test_to_train_dist.npy` versus
rebuilt, against an exact float64 computation from the rebuilt thumbnails. The exact values use
the float64 matmul form, with direct differences for the 40 smallest; they are saved as
`outputs/research/exp3b_nn_test_to_train_dist_exact_float64.npy`.

```
published (03b, GPU? float32 cdist)    |x - exact|: median 7.09e-03  p99 7.20e-01  max 0.805
    exact in [  0,  1): n=130 max|err|=8.05e-01 max rel err=8.05e+02
    exact in [  1,  5): n=368 max|err|=1.42e-01 max rel err=1.26e-01
    exact in [  5, 20): n=253 max|err|=1.58e-02 max rel err=1.55e-03
    exact in [ 20,200): n= 20 max|err|=1.85e-03 max rel err=9.15e-05
rebuilt (03b, CPU float32 cdist)       |x - exact|: median 1.95e-04  p99 5.10e-02  max 0.072
    exact in [  0,  1): n=130 max|err|=7.22e-02 max rel err=7.22e+01
    exact in [  1,  5): n=368 max|err|=3.22e-03 max rel err=2.77e-03
    exact in [  5, 20): n=253 max|err|=7.55e-04 max rel err=1.45e-04
    exact in [ 20,200): n= 20 max|err|=1.26e-04 max rel err=5.51e-06
largest published/rebuilt gap: test clip videos/mid/view1987.mp4 published 0.8053 rebuilt 0.0000 exact 0.000000
median nn test->train: published 2.7080 rebuilt 2.6974 exact 2.6965
test clips with exact float64 nearest-train distance == 0 (identical 32x32 thumbnails in some frame pair): 120 of 771
  published values for them: min 0.350 median 0.592 max 0.805
  rebuilt CPU values for them: min 0.000 median 0.000 max 0.072
exact < 0.5: 121  published < 0.5: 17  rebuilt < 0.5: 121
```

Other 03b files (`outputs/research/compare_03b_numeric.txt`):

- `exp3b_threshold_sweep.csv` differs only in the thr 12.5 row (n_clusters 1127 → 1126,
  n_singletons 490 → 489).
- `exp3b_random_pair_pcts.csv` differs by at most 0.57. The largest gaps are in the near-zero
  percentiles: adjacent-pair p0.1/p1/p5 are 0.37/0.45/0.62 published and 0.00/0.00/0.05 rebuilt.
- `p1_random_cross` is 34.962337 published and 34.962265 rebuilt. Threshold 27.5 is chosen in
  both runs.
- `exp3b_cluster_table.csv` is equal apart from `n_chains` (-1 here, because no chains file was
  given).

Observations:

- The published errors fall steadily with distance, to a relative 9e-5 above 20, which is the
  pattern of cancellation in the `|x|² + |y|² − 2x·y` form at float32. They are about 30 times
  larger than the CPU float32 errors.
- The rebuilt values are closer to the exact float64 values everywhere.
- An earlier rebuild on an RTX 5090 host (`outputs/review_grouped/host_logs/reproduce_rebuild.log`,
  `setup.log`: torch 2.11 cu128, cv2 5.0.0, `cuda True`) reproduced every threshold byte for byte
  (`outputs/review_grouped/comparison_clusters.md`).

**Inference.** The published 03b distances were computed on a GPU, whose float32 matmul (possibly
TF32 or a different accumulation order) is less accurate here. The thr 12.5 difference is that
error on a pair 0.0013 below the threshold. It is not a difference in the thumbnails.

**Inference.** The published "near-duplicate floor" of about 0.35-0.8 is float32 noise. 120 of
the 771 published test clips have a train clip with an identical 32x32 thumbnail in some frame
pair. The `07` leakage statistics (float64) already show this as random-split test→train `min 0.0`
and `p5 0.0`.

**Caveat.** The thumbnails were not compared directly with the published cache, which is not on
this machine. Their identity is inferred from the byte-identical thr 27.5 file, the 10 other
identical partitions, the float32-shaped error pattern above, and the float64 leakage statistics
in section 6.

## 6. Grouped splits (`run_07_splits.sh`)

```
CUDA_VISIBLE_DEVICES="" VENV=.venv CLUSTERS=outputs/review_out CACHE=outputs/review_cache \
    REPORT=outputs/research/splits_rebuilt OUT_BASE=outputs/research/split_roots_rebuilt \
    /usr/bin/time -v bash scripts/review/run_07_splits.sh > outputs/research/splits_rebuilt.log 2>&1
#   Elapsed 2:52.08, Maximum resident set size 2,693,244 KB, exit 0
```

Its final lines (`outputs/research/splits_rebuilt.log`):

```
split_grp27a.csv: identical to docs/review/data/grouped/splits/split_grp27a.csv
split_grp27b.csv: identical to docs/review/data/grouped/splits/split_grp27b.csv
split_grp17.csv: identical to docs/review/data/grouped/splits/split_grp17.csv
SPLITS_DONE
```

The manifests in `outputs/research/split_roots_rebuilt/{grp27a,grp27b,grp17}/data_labels/` are
byte-identical to those in `outputs/research/split_roots/` (built earlier from the published
clusters).

Leakage statistics, published `docs/review/data/grouped/splits/split_*_summary.json` versus
rebuilt `outputs/research/splits_rebuilt/split_*_summary.json`. Both use 07's float64 distance.
The medians are nearest-train (or nearest-val) RMSE on the 0-255 scale.
(`outputs/research/compare_split_summaries.txt`)

| split | scheme | pair | median published | median rebuilt | p5 pub / reb | p95 pub / reb | min pub / reb | frac ≤10/17.5/27.5 pub = reb |
|---|---|---|---|---|---|---|---|---|
| grp27a | grouped | test→train | 65.598969 | 65.598969 | 31.4267 / 31.4267 | 67.7471 / 67.7471 | 27.5719 / 27.5719 | 0/0/0 |
| grp27a | grouped | val→train | 39.434212 | 39.434212 | 31.0383 / 31.0383 | 44.9033 / 44.9033 | 27.5775 / 27.5775 | 0/0/0 |
| grp27a | grouped | test→val | 74.028534 | 74.028534 | 35.6009 / 35.6009 | 77.1984 / 77.1984 | 27.7131 / 27.7131 | 0/0/0 |
| grp27b | grouped | test→train | 38.900162 | 38.900162 | 33.2351 / 33.2351 | 47.3903 / 47.3903 | 27.5775 / 27.5775 | 0/0/0 |
| grp27b | grouped | val→train | 36.517624 | 36.517624 | 30.0016 / 30.0016 | 44.6908 / 44.6908 | 27.5719 / 27.5719 | 0/0/0 |
| grp27b | grouped | test→val | 58.011570 | 58.011570 | 39.1184 / 39.1184 | 66.1384 / 66.1384 | 28.9601 / 28.9601 | 0/0/0 |
| grp17 | grouped | test→train | 31.284698 | 31.284698 | 21.6035 / 21.6035 | 41.2737 / 41.2737 | 17.5462 / 17.5462 | 0/0/0.301 |
| grp17 | grouped | val→train | 27.508018 | 27.508018 | 22.3942 / 22.3942 | 38.4406 / 38.4406 | 18.3730 / 18.3730 | 0/0/0.497 |
| grp17 | grouped | test→val | 40.888432 | 40.888432 | 28.7719 / 28.7719 | 53.0747 / 53.0747 | 19.0238 / 19.0238 | 0/0/0.023 |
| all three | random | test→train | 2.696509 | 2.696509 | 0.0000 / 0.0000 | 15.3339 / 15.3339 | 0.0000 / 0.0000 | 0.840/0.960/0.987 |
| all three | random | val→train | 2.876075 | 2.876075 | 0.0000 / 0.0000 | 17.7684 / 17.7684 | 0.0000 / 0.0000 | 0.832/0.945/0.992 |
| all three | random | test→val | 22.392706 | 22.392706 | 1.2492 / 1.2492 | 39.3704 / 39.3704 | 0.0000 / 0.0000 | 0.241/0.396/0.651 |

Of the 66 leakage numbers per summary, all are exactly equal except these:

```
grp27a leakage.random.val_vs_train.p95: published 17.768362045288086 rebuilt 17.76835823059082  diff=-3.81e-06  float32 ulp at this value=1.91e-06
grp27b avoided: published '/workspace/review_grouped/splits/split_grp27a.csv' rebuilt 'outputs/research/splits_rebuilt/split_grp27a.csv'
grp27b leakage.grouped.test_vs_train.p95: published 47.390323638916016 rebuilt 47.390316009521484  diff=-7.63e-06  float32 ulp at this value=3.81e-06
grp27b leakage.grouped.val_vs_train.p95: published 44.69084930419922 rebuilt 44.69084548950195  diff=-3.81e-06  float32 ulp at this value=3.81e-06
grp27b leakage.random.val_vs_train.p95: published 17.768362045288086 rebuilt 17.76835823059082  diff=-3.81e-06  float32 ulp at this value=1.91e-06
grp17 leakage.grouped.test_vs_train.p95: published 41.273704528808594 rebuilt 41.27370071411133  diff=-3.81e-06  float32 ulp at this value=3.81e-06
grp17 leakage.random.val_vs_train.p95: published 17.768362045288086 rebuilt 17.76835823059082  diff=-3.81e-06  float32 ulp at this value=1.91e-06
```

All other fields are equal. The summary files are not byte-identical, because of these values and
the `avoided` path.

**Inference.** Only interpolated p95 values differ, by 1-2 float32 ulps, while medians, p5, minima
and every fraction are exact. The likely cause is `np.percentile` interpolation on a float32 array
under numpy 2.5.4 (GPU host) versus 2.2.6 here, or last-bit BLAS differences. It is not a data
difference.

## 7. Problems and caveats

- **The first run of `02d_long_clip_check.py` was OOM-killed (exit 137) and wrote nothing.** The
  bug was mine. It read the cache with `z[key][i]` inside a per-clip loop. Every `NpzFile[key]`
  access decompresses the whole array, and the stored row is a view that keeps that whole array
  alive, so memory grew without bound. **Inference:** the memory pressure from this may be what
  crashed the parent session. Fixed: each key is now decompressed once and only the 98 needed rows
  are copied. The rerun used 4 workers under `ulimit -v 4500000`.
- The dataset roots in `outputs/research/split_roots_rebuilt/*/` have a dangling `videos` symlink
  (`../ouc-cge-320/videos`). `07`'s default `--videos-link` assumes `OUT_BASE=data`. The same
  holds for the earlier `outputs/research/split_roots/`. The manifests are correct. To train from
  these roots, fix the link or use `09_materialize_splits.py`.
- The main process of the streaming cache peaks at about 1 GB, because it holds the hires arrays
  of all clips. Workers stay at 99 MB.
- `outputs/review_out/` contains only the 03b outputs. 02/03/04 need the ResNet features, which
  `--thumbs-only` does not compute.
- The hires `first`/`last` frames have no counterpart in the original script. Their only
  independent check is `02d` (section 3.4).

## 8. Paths

Scripts (new):

- `scripts/research/02_frame_cache_stream.py`: the streaming cache.
- `scripts/research/02b_compare_frame_caches.py`: key-by-key cache comparison.
- `scripts/research/02c_make_tmp_roots.py`: the temporary roots and the wrong-hint audit.
- `scripts/research/02d_long_clip_check.py`: the independent long-clip check.

Cache and clusters:

- `outputs/review_cache/frames_{train,val,test}.npz`, `frames_{train,val,test}_hires.npz`,
  `frames_stream_meta.csv`.
- `outputs/review_out/`: `scene_clusters.csv`, `exp3b_clusters_by_thr.csv`,
  `exp3b_threshold_sweep.csv`, `exp3b_random_pair_pcts.csv`, `exp3b_cluster_table.csv`,
  `exp3b_summary.json`, `exp3b_nn_test_to_train_dist.npy`, `exp3b_test_nn_pairs.png`,
  `exp3b_cluster_{0..7}_id*_n*.png`.

Splits:

- `outputs/research/splits_rebuilt/split_{grp27a,grp27b,grp17}.csv` and `_summary.json`.
- `outputs/research/split_roots_rebuilt/{grp27a,grp27b,grp17}/data_labels/{train,val,test}.csv`
  and the `videos` symlink.

Verification:

- `outputs/research/frame_cache_stream_limit10/`: streaming cache, first 10 clips per split.
- `outputs/research/tmp_root/`: `data_labels/`, `videos` symlink, `selection.csv`,
  `bad_hints_audit.csv`.
- `outputs/research/tmp_root_edge/`: `data_labels/`, `videos` symlink, `selection.csv`.
- `outputs/research/fc_check/`: the original caches `orig_tmp_root/` and `orig_tmp_root_edge/`;
  the streaming caches `stream_audit_tmp_root/`, `stream_grab_tmp_root/`,
  `stream_badhints_tmp_root/`, `stream_audit_tmp_root_edge/`, `stream_grab_tmp_root_edge/`; and
  the `compare_*.txt` outputs.
- `outputs/research/long_clip_check.csv`, `outputs/research/long_clip_check.log`.

Logs and comparisons:

- `outputs/research/frame_cache_stream.log`, `outputs/research/frame_cache_stream.start`
  (start epoch).
- `outputs/research/scene_clusters_03b.log`.
- `outputs/research/compare_clusters_published.txt`, `outputs/research/compare_03b_numeric.txt`,
  `outputs/research/compare_03b_exact_nn.txt`,
  `outputs/research/exp3b_nn_test_to_train_dist_exact_float64.npy`.
- `outputs/research/splits_rebuilt.log`, `outputs/research/compare_split_summaries.txt`.
- This report: `docs/research/04_cache_rebuild.md`.
