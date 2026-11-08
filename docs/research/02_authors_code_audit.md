# Audit of the OUC-CGE authors' code (github.com/oucyy361/OUC-CGE)

Record of a subagent run (Claude Opus 5.5, 2026-11-07, supervised by Claude Fable 5.1). The brief: clone the
authors' repository read-only and report, with file paths and verbatim excerpts, how the data were split,
loaded and evaluated. Nothing from the repository was executed. The report is kept as delivered, apart from
this preface; "your" in the text refers to this repository. Observations and inferences are separated by the
author of the report; the inferences in sections 2, 4 and 5 (unstratified shuffle, consecutive IDs as cutting
order, the PTS decoding bug) are hypotheses to be checked, not findings.

Supersedes nothing; complements `docs/review/02_paper_protocol.md` (web research, 2026-10-26), which it
confirms on the split and extends with the diff against upstream PySlowFast, the checkpoint epochs actually
tested and the decoder patch.

---

Nothing in the repo or on OSF records session, camera or time. There is no code that creates the split, and no result files to recompute accuracy from. The shipped split matches an unstratified random clip-level shuffle, and the authors' one real code change probably makes every model see a single repeated frame per clip (one file header check confirms or rules this out). Details follow.

**The URL in the brief has a typo.** `github.com/oucy361/OUC-CGE` returns 404 and that user does not exist. The paper's Code Availability section gives `https://github.com/oucyy361/OUC-CGE.git` (two y's). I cloned that with `--depth 1` into a scratch directory. Nothing from the repo was executed.

## 1. Repository overview (observed)
- **History:** 11 commits between 2025-03-20 and 2025-03-24. HEAD is `4afc0f5`, "Update README.md", 2025-03-24 15:19 +0800. Nothing has been pushed since.
- **Size:** 2.0 MB working tree, 139 files; GitHub reports 353 KB. 10 stars, 2 forks, 3 open issues.
- **License:** there is no LICENSE file and the API reports `license: null`, although the paper says MIT.
- **Layout:** `README.md`, `cal_metrics.py`, `train.csv`, `val.csv`, `test.csv`, `configs/Kinetics/**` (59 yaml), `slowfast/**`, `tools/**`.
- **Framework:** an unmodified copy of PySlowFast (facebookresearch/SlowFast). I cloned upstream and diffed. The closest commit is `bac7b67` (2024-06-06). Against it, only these differ:
  - **New files:** `cal_metrics.py`, `tools/out_results.py`, the three CSVs.
  - **Modified code:** `slowfast/datasets/decoder.py` (PyAV patch, see §5), `slowfast/datasets/kinetics.py` (commented-out prints only), `tools/train_net.py` (top-k=3 fix), `tools/test_net.py` (dumps predictions).
  - **Modified configs:** 13 yaml files.
  - **No file under `slowfast/models/` differs from upstream.**
- **README:** says the 8:1:1 split is "applied … to ensure balanced data representation". It lists the accuracies (SlowFast 93.2, C2D 94.3, I3D 93.3, X3D 96.8, SLOW-NLN 97.4, SLOW 97.8) and links the upstream SlowFast repo. It has no setup or run instructions; issue #3 asks for exactly that.

## 2. Split protocol
**Observed: no split-generation code exists.** Grepping for `train_test_split`, shuffle, sample, `to_csv`, Stratified or Group finds nothing except `out_results.py`'s `to_csv`. There is no seed for a split. `RNG_SEED: 0` in the configs affects training only.

**The CSVs are shipped**, CRLF, space-separated, no header:

| file | rows | low | mid | high |
|---|---|---|---|---|
| `train.csv` | 6164 | 2396 | 1653 | 2115 |
| `val.csv` | 770 | 307 | 199 | 264 |
| `test.csv` | 772 (771 labelled + 1 junk) | 325 | 189 | 257 |

- **First lines:**
  - train: `videos/high/view2015.mp4 2`, `videos/low/view1216.mp4 0`
  - val: `videos/high/view1232.mp4 2`, `videos/mid/view1211.mp4 1`
  - test: `videos/high/view1298.mp4 2`, `videos/low/view2463.mp4 0`
- **Path form:** every labelled row matches `videos/(low|mid|high)/view\d+.mp4`. The directory is `mid`, not `medium` as the paper says. The directory always agrees with the label (0=low, 1=mid, 2=high).
- **No duplicates:** none within a split and none between splits. There are 7705 unique paths.
- **Unlabelled last line:** `test.csv` line 772 is `videos/anonymity/a.mp4` with no label. `kinetics.py:135-136` parses one-field rows as label 0, so the test set would contain one fake "Low" sample. This file is not on OSF; the transcode log `data/ouc-cge-320/transcode_failures.txt` records it as `missing_src`.
  - In test mode a missing file is retried at the same index 100 times. `__getitem__` then only logs a warning and returns None (`kinetics.py:469-472`), which would break collation.
  - Inference: either the authors had such a file when they evaluated, or the line was added to the CSV later.
- **Not stratified:**
  - Totals are exactly 0.8N / ⌊0.1N⌋ / rest (6164/770/771). Per-class fractions drift: low test 10.7%, mid test 9.3%.
  - A class×split χ² test gives 3.65 with 4 degrees of freedom, p=0.46. Exact stratification would keep every cell within about ±1 clip of proportional. Low-test is +22 clips.
  - Inference: this is an unstratified random 80/10/10 shuffle of all clips, which contradicts the paper's "stratified distribution across all splits".
- **Not random filenames:** IDs are contiguous `1..N` within each class with no gaps (high 1–2636, low 1–3028, mid 1–2041). This contradicts the paper's "randomly generated video filenames … alphanumeric strings". The OSF archives are cut on the same ranges, e.g. `high-2636（1-1000）.7z`. Inference: these are per-class sequential indices, possibly in cutting order.

## 3. Session, camera or time structure
**Absent everywhere I could look.** The repo has no metadata, mapping, source-video list or timestamps.

OSF (`api.osf.io/v2/nodes/brd2c`) has no child components and a single storage provider. It holds only three folders, `high`, `mid` and `Low`, with 16 archive files in total:
- each folder has a `* sample.zip` dated 2025-01-11;
- the class archives were uploaded 2025-06-26 to 2025-11-28, after the April 2025 publication.

There is no raw data (despite "We provide raw and pre-processed data"), no `result` folder, no CSVs and no metadata. The only remaining trace of structure is the per-class sequential ID.

## 4. Clip generation
**Observed: no cutting, frame-extraction or offline resizing script exists in the repo.** The paper credits Boilsoft Video Splitter, manual and automatic segmentation, and says segments were "shortened" to remove uninformative parts. There is no evidence about overlap or stride.

**Clip lengths, measured** on the local copy `data/ouc-cge-320` (PyAV headers, 7701 clips). Stream duration matches frames ÷ 30 fps in every case. `scripts/preprocess_resize.py` has no `-r`, and its optional `-t` cap was evidently not applied since clips up to 322 s are present, so these lengths should reflect the source files.
- Median 5.5 s.
- Quantiles: 5% 2.6 s, 25% 5.3 s, 75% 10.0 s, 95% 11.8 s, 99% 168.9 s.
- 552 clips are under 5 s, 5140 are 5–9.5 s, 1069 are 9.5–10.5 s, 940 are over 10.5 s, and 175 are over 60 s.
- The longest are around 320 s: `high/view1888` is 322 s, and `high/view1831` through `view1834` are each exactly 320.63 s, four consecutive IDs.
- Total 23.2 h, which is neither "12 h 50 min" nor 7705×10 s = 21.4 h.
- Inference: clips are not uniformly 10 s. The four identical 320 s clips with consecutive IDs may be duplicated or overlapping material and are worth hashing.

Aside on this repository's own code: the comment in `scripts/preprocess_resize.py` says "p99 ~14.5 s" and that a 30 s cap leaves ">99%" of clips untouched. I measure p99 = 169 s, and 202 clips (2.6%) are over 30 s.

## 5. Training and evaluation
**Clip sampling in the shipped configs:** all models use `DECODING_BACKEND: "pyav"` and `NUM_CLASSES: 3`.

| model | frames × rate | window per sample |
|---|---|---|
| SLOW | 8×8 | about 1.9 s |
| C2D, I3D | 8×8 | about 1.9 s |
| SLOW-NLN (4x16) | 4×16 | about 1.6 s |
| SlowFast | 32×2, α=4 | about 2.1 s |
| X3D_M | 16×5 | about 2.5 s |

**Other training settings:**
- Batch 16 (X3D 12), SGD, learning rate 0.1, cosine schedule. Warmup is 34 epochs (X3D 35) inside `MAX_EPOCH: 50`, a value carried over from the 196-epoch Kinetics schedules.
- `EVAL_PERIOD: 10`, `CHECKPOINT_PERIOD: 1` (X3D 10), `AUTO_RESUME: True`.
- **No pretrained weights:** `TRAIN.CHECKPOINT_FILE_PATH` is not set, and the default is `""` (`defaults.py:155`).
- Augmentation is upstream default: random short-side jitter to 256–320, random 224 crop, horizontal flip, mean 0.45 / std 0.225, no mixup or AUG.

**Test-time settings:** `NUM_ENSEMBLE_VIEWS: 1` and `NUM_SPATIAL_CROPS: 1`. The short side is resized to 256 and a 256×256 centre crop is taken, which keeps about 56% of a 16:9 frame's width. The temporal window is the one at clip index 0 of 1 with `USE_OFFSET_SAMPLING=False`: `start_idx = max_start * clip_idx / num_clips_uniform` (`decoder.py:137`) gives frame 0. So each clip is judged on its first ≤2.5 s only, and accuracy is clip level (one row = one clip = one view).

**Which split is evaluated:**
- `test_net.py:226` uses `loader.construct_loader(cfg, "test")`, which reads `test.csv`.
- The val split is used only by `eval_epoch` to log top-1 error when `(cur_epoch+1) % 10 == 0` or at the final epoch (`misc.py:211-221`).
- No code uses validation to select a checkpoint or stop training.
- **There is no early stopping:** grep for early stop, patience or best finds nothing. The paper claims "early stopping with patience of 10 epochs monitored on the validation loss".

**Checkpoints actually tested** (`TEST.CHECKPOINT_FILE_PATH`; file N is saved after epoch N, `checkpoint.py:143`):
- **Final epoch 50:** SLOW (`exp5`), I3D (`exp4`), SLOW_NLN_4x16 (`exp6`) and X3D_M (`exp9`).
- **C2D:** `exp3/.../checkpoint_epoch_00053.pyth` with `MAX_EPOCH: 196`. Epoch 53 is not a validation epoch, so the code logs no validation metric that could have chosen it.
- **SlowFast:** `exp2/.../checkpoint_epoch_00075.pyth` while the shipped config says `MAX_EPOCH: 50`, so the shipped config is not the one that was run.
- Inference: C2D and SlowFast checkpoints were chosen by a criterion that is not in the repo, possibly test accuracy. Saving every epoch makes that kind of post-hoc choice possible.

**Config defects:**
- `SLOW_NLN_8x8_R50.yaml:31` has `  SOLVER:` indented under `BN:`, so yacs would raise KeyError. Its `OUTPUT_DIR` is `./checkpoints/exp2`, the same as SlowFast's.
- With `AUTO_RESUME` on, other output directories are also shared: `exp6` (SLOW_NLN_4x16 and SLOWFAST_NLN_4x16), `exp5` (SLOW and MVIT_B), `exp7` (X3D_L and X3D_XS).

**Prediction dump** (`test_net.py:50,129-135`), verbatim:
```
out_results = OutResults(os.path.join(cfg.OUTPUT_DIR, "results.xlsx"))
preds_max_, pre_class_ = torch.max(preds, dim=-1)
pred_dict = {f"preds_vector_{c}":preds[:,c].numpy()for c in range(preds.shape[-1])}
add_dict = {"pre_conf": preds_max_, "pre_class": pre_class_, "labels": labels, "video_path": test_loader.dataset._path_to_videos[video_idx[0]:video_idx[-1]+1]}
```
This writes 7 columns for the test split only. Producing train or val results would require swapping CSV files by hand. The paper's 4-column CSVs per split cannot come from this code.

**Metrics** (`cal_metrics.py`): it reads every `*.xlsx` in `ROOT = r"./results path"`, `Sheet1`.
- Lines 104-107: `accuracy_score` and macro `f1`/`recall`/`precision`.
- Line 110: row-normalised confusion matrix.
- Lines 41-79: macro one-vs-rest AUC as the standard sklearn recipe (interpolate each class's ROC curve at the union of FPRs, average, then `auc`) on the `preds_vector_*` softmax scores.
- There is no code for the paper's 95% CI on AUC.
- `read_csv` (line 37) has a bug, `data[f"{conf_titles}i"]`, but it is unused.
- The `train_net`/`test_net` changes make the "Top5" log line report top-3 accuracy, which is trivially 100% with three classes.

**The probable frame-sampling bug (inference, unverified).** Upstream's PyAV path is broken for list-valued `sampling_rate`, which is why the authors patched it; every model config therefore runs through their patch. In `decoder.py:462-468`:
```
start_idx = start_end_delta_time[k, 0]
end_idx = start_end_delta_time[k, 1]
timebase = duration / frames_length
video_start_pts = int(start_idx)
video_end_pts = int(end_idx)
```
Upstream computes `int(start_idx * timebase)`. Here frame indices are passed as PTS ticks, and `timebase` is computed but never used.
- This is only correct if the stream has one tick per frame (time_base 1/30).
- With a typical MP4 time base (e.g. 1/15360, 512 ticks per frame, or 1/90000), the window [start, start+55] ticks falls inside the first frame interval. `pyav_decode_stream` then returns 1–2 frames, and `temporal_sampling` repeats them T times.
- The effect: every training and test sample would be a still frame from the first ~1–2 frames of the clip, making all six "video" models effectively image classifiers.
- **Check:** read `stream.time_base` from one original clip, e.g. `av.open(p).streams.video[0].time_base`. The original-resolution clips are not in the local DVC cache, and the 320 px copy (1/15360) was re-encoded by ffmpeg, so it says nothing about the originals.

## 6. Model result CSVs
**Absent** from both the repo and OSF, so per-split accuracy cannot be computed. As noted in §5, the code produces `results.xlsx` for the test split only.

## 7. Other points bearing on validity
- **Paper claims absent from the code:** "attention-guided spatial cropping to focus on facial regions" and "adaptive fusion weighting". Nothing in `slowfast/models/` differs from upstream, and there is no wavelet code.
- **Corrupted or black clips:**
  - Issue #1 (2025-05-12) asks whether the mid-2041 archive is corrupt.
  - Issue #2 (2025-06-07) asks why many videos are a black screen. The author replied on 2025-07-23: "可能是解压缩时出现了问题，我已经重新上传了" ("maybe a decompression problem; I've re-uploaded").
  - The transcode log lists 4 undecodable high clips (view2521, 2630, 2631, 2632).
  - Training mode silently swaps undecodable videos for a random other index (`kinetics.py:277-290, 355-361`).
- **Paper count typo:** the paper says "7,057 clips" in two places versus 7,705.
- **Leakage-relevant facts, observed:** a clip-level random split, contiguous per-class IDs, no grouping metadata, no pretraining, and very high accuracy from scratch. These are consistent with the scene leakage already established.

Key files: `slowfast/datasets/decoder.py:373-490`, `slowfast/datasets/kinetics.py:85-148, 221-343`, `tools/test_net.py:50-171`, `tools/train_net.py:227-228, 377-378`, `cal_metrics.py`, `configs/Kinetics/{SLOW_8x8_R50,C2D_8x8_R50,I3D_8x8_R50,X3D_M,SLOW_NLN_4x16_R50,SLOWFAST_8x8_R50,SLOW_NLN_8x8_R50}.yaml`, `test.csv:772`.

---

## Verification on the released raw clips (Claude Fable 5.1, 2026-11-07)

Five raw clips were fetched from the project's DVC copy of the OSF release (`data/ouc-cge/videos/`,
29 GB in total; the full set is being pulled for a header audit) and read with PyAV 17:

| clip | video stream | audio | frames | duration | first PTS (ticks of 1/15360 s) |
|---|---|---|---|---|---|
| `high/view1.mp4` | mpeg4, **1092x614**, 30 fps | aac 48 kHz | 300 | 10.00 s | 0 |
| `low/view1.mp4` | h264, 1920x1080, 30 fps | **two** aac 48 kHz streams | 300 | 10.00 s | 1183 (0.077 s) |
| `mid/view1.mp4` | mpeg4, 1920x1080, 30 fps | aac 48 kHz | 431 | 14.37 s | 200 (0.013 s) |
| `high/view1831.mp4` | h264, 1920x1080, 30 fps | two aac streams | **2** | 0.07 s | 4923663 (**320.55 s**) |
| `high/view1832.mp4` | h264, 1920x1080, 30 fps | two aac streams | **2** | 0.07 s | 4923663 (320.55 s) |

Observations.
- Every stream has `time_base = 1/15360`, i.e. 512 ticks per frame at 30 fps. None of the five has
  one tick per frame, which is the only case in which the patched decoder is correct.
- The release is heterogeneous: two codecs (mpeg4 and h264), two resolutions (1092x614 and
  1920x1080), one or two audio streams. The paper states 1280x720 for the cameras. Which of these
  properties co-vary with session, camera or label is unknown until the full header audit is in.
- `high/view1831` and `view1832` contain two frames whose timestamps start at 320.55 s. The 320p
  copies of the same clips are 320.63 s long (9,619 frames): the student's `ffmpeg` transcode
  (`scripts/preprocess_resize.py`, `-vf scale ... -an`, no `-vsync`) padded the gap from 0 to the first
  timestamp with duplicated frames. The ">60 s clips" of the 320p audit are therefore artefacts of
  broken two-frame source files, not long recordings; the raw durations are the ones to report.

Simulation of the shipped decoder (own re-implementation of `pyav_decode_stream` and
`temporal_sampling`, nothing from the repository executed): the authors' patch passes frame indices
to `pyav_decode_stream` as PTS ticks; with `buffer_size=0` that function also appends the first frame
beyond `end_pts` before breaking. For the SLOW/C2D/I3D configs (8x8, 56-frame window):

| clip | test window (`clip_idx=0`: ticks 0-55) | five random training windows |
|---|---|---|
| `high/view1` | 2 frames, t = 0.000 and 0.033 s | 1 frame each (t = 0.033 s) |
| `low/view1` | 1 frame, t = 0.077 s | 1 frame each (the same frame) |
| `mid/view1` | 1 frame, t = 0.013 s | 1 or 2 frames (t = 0.013 / 0.046 s) |
| `high/view1831` | 1 frame, t = 320.55 s | 1 frame each (the same frame) |

Whereas the upstream code (`int(start_idx * timebase)`) returns 53-56 frames for the same windows.
`temporal_sampling` then indexes the 1-2 decoded frames with `linspace` clamped to their count, so the
network input is one frame, or two adjacent frames, repeated 8 (SLOW, C2D, I3D), 4 (SLOW-NLN), 16 (X3D)
or 32 (SlowFast) times.

Conclusion (conditional on the released files being the files the authors trained on; their local
copies could in principle have had a 1/30 time base, which nothing in the release suggests): all six
baselines of Table 4 were trained and tested on the first 0.03-0.08 s of each clip, i.e. on a single
frame. This agrees with everything measured in the review: C2D (2D) at 93.3% in the paper, a
single-frame ResNet50 at 99.35% in the student's run, SLOW unchanged under frame shuffling and under a
static repeated frame, and 1-NN on 32x32 thumbnails at 96.1%. The paper's discussion of why SLOW's
"4 fps sampling" suits "low-frequency" classroom dynamics describes a model that never saw more than
two consecutive frames.
