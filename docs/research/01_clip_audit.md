# OUC-CGE clip audit (320p copies)

Date: 2026-11-07. Host: local workstation, 8 CPU cores, 7.6 GiB RAM, no GPU used
(`CUDA_VISIBLE_DEVICES=""` everywhere). Branch `research/ouc-cge`.

Every record of `data/ouc-cge/data_labels/{train,val,test}.csv` was checked against
`data/ouc-cge-320/videos/`. The raw videos (`data/ouc-cge/videos`) are not on this machine, so
everything below describes the **320p copies**. The only raw-side information used is the DVC
directory manifest of the raw videos: file names and md5 hashes, no durations.

Sections 1-8 report what the files show. Inferences are marked **Inference**.

## Method

| step | command (repository root) | wall time |
|---|---|---|
| audit, 200-clip timing sample | `CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/01_clip_audit.py --sample 200 --out outputs/research/clip_audit_sample200.csv` | 31 s |
| audit, all 7,706 records | `CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/01_clip_audit.py --out outputs/research/clip_audit.csv` | **23 min 19 s** (8 workers, peak RSS 140 MB) |
| summary, duplicates, figure | `CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/01b_clip_audit_summary.py` | about 10 s |

For each clip, `01_clip_audit.py` does three things:

- **cv2 4.13** (FFmpeg backend, one decode thread): `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FPS`, and
  the exact frame count by sequential `grab()` until it fails. `grab()` decodes every frame; only
  the first frame is `retrieve()`d, to read its shape.
- **PyAV 17.0.1**, demux only: streams, codecs, header frame count, packet count, pts range,
  container duration, `encoder` tags.
- **md5** of the file bytes.

`duration_s = n_frames / fps`. PyAV was already in the venv, so nothing was installed.

The raw-side duplicate check reads the DVC directory manifest of `data/ouc-cge/videos`
(`md5 3486e78a…dir`, referenced by `data/ouc-cge/videos.dvc`). It was downloaded from the
project's public DVC remote and saved as `outputs/research/raw_videos_dvc_dir.json`.

## 1. Manifests and file presence

The manifests hold 7,706 records: 6,164 train, 770 val and 772 test. One test record (line 772 of
`test.csv`) is `videos/anonymity/a.mp4` with no label; it is not in the raw video set either.
That leaves **7,705 labelled records**. Each path appears once, and the label always matches the
class folder (low → 0, mid → 1, high → 2). The class folder is named `mid`, not `medium`.

| split | low (0) | medium (1, folder `mid`) | high (2) | total | readable 320p |
|---|---|---|---|---|---|
| train | 2,396 | 1,653 | 2,115 | 6,164 | 6,160 |
| val | 307 | 199 | 264 | 770 | 770 |
| test | 325 | 189 | 257 | 771 | 771 |
| all | 3,028 | 2,041 | 2,636 | **7,705** | **7,701** |

Every labelled path is in the raw DVC manifest and every raw `.mp4` is in a manifest. The raw
manifest has 7,706 entries: 7,705 `.mp4` (low 3,028, mid 2,041, high 2,636) plus one `.DS_Store`.

## 2. Unreadable clips

Four clips are missing from `data/ouc-cge-320/videos`:

- `videos/high/view2521.mp4`
- `videos/high/view2630.mp4`
- `videos/high/view2631.mp4`
- `videos/high/view2632.mp4`

All four are train, label 2 (high). They are exactly the four `fail` lines of
`data/ouc-cge-320/transcode_failures.txt`. The fifth line of that file is
`missing_src videos/anonymity/a.mp4`. The four clips are present in the raw manifest. All other
7,701 files open and decode in both cv2 and PyAV. Their md5 hashes equal the md5s in the
320p DVC manifest (`data/ouc-cge-320.dvc`), with 0 mismatches.

## 3. Duration

All 7,701 readable clips. Seconds = frames / 30.

| | p1 | p5 | p25 | p50 | p75 | p95 | p99 | min | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| duration (s) | 0.300 | 2.567 | 5.267 | **5.500** | 10.000 | 11.833 | **168.933** | 0.133 | **321.967** | 10.838 |
| frames | 9 | 77 | 158 | 165 | 300 | 355 | 5,068 | 4 | 9,659 | 325 |

![clip duration histogram](figures/clip_duration_hist.png)

`figures/clip_duration_hist.png`. Left: 1/3-s bins up to 30 s; the last bar collects all clips
longer than 30 s. Right: full range, both axes logarithmic.

| duration bin | clips | share | train / val / test |
|---|---|---|---|
| 9.5-10.5 s | **1,069** | 13.9 % | 878 / 98 / 93 |
| < 9.5 s | **5,692** | 73.9 % | 4,519 / 586 / 587 |
| > 10.5 s | **940** | 12.2 % | 763 / 86 / 91 |
| > 30 s | **202** | 2.6 % | 169 / 15 / 18 |
| > 60 s | 175 | 2.3 % | |
| < 1 s | 100 | 1.3 % | |
| < 64 frames (2.13 s) | 113 | 1.5 % | 94 / 10 / 9 |

The distribution has two modes:

- 150-170 frames (5.0-5.7 s): 4,505 clips. 4,751 clips are between 5.0 and 6.0 s.
- 300-302 frames (10.0-10.07 s): 271 clips; 220 of them have exactly 302 frames.

The tail above 30 s holds 202 clips. Only the high (125) and low (77) classes have such clips;
medium has none. Their IDs fall in runs:

- high: 255-327, 465-496, 1768, 1831-1889, 1898
- low: 3-4, 42-67, 2182-2244 (even IDs), 2596, 2938-2969

By class:

| class | clips | median (s) | mean (s) | > 30 s | hours |
|---|---|---|---|---|---|
| low | 3,028 | 5.47 | 8.69 | 77 | 7.31 |
| medium | 2,041 | 5.53 | 7.11 | 0 | 4.03 |
| high | 2,632 | 5.47 | 16.19 | 125 | 11.84 |

### Total duration against the paper's statements

| quantity | value |
|---|---|
| total of the 7,701 readable 320p clips | **23.18 h (23 h 11 min)**: train 19.15 h, val 2.10 h, test 1.93 h |
| of which the 202 clips > 30 s | 9.55 h |
| total of the 7,499 clips ≤ 30 s | 13.64 h (mean 6.55 s per clip) |
| total with every clip capped at 30 s | 15.32 h |
| total over distinct file contents (7,422 unique md5, see section 6) | 17.97 h |
| paper: "12 h 50 min" | 46,200 s, which is a mean of 6.00 s over 7,705 clips |
| paper: "about 10 seconds" x 7,705 | 21.40 h |

The 320p data support **neither** statement as written:

- "About 10 seconds" does not describe most clips. The median is 5.5 s, 64.8 % of clips are
  4.5-6.5 s long, and only 13.9 % are within 9.5-10.5 s.
- The total is 23.2 h, not 12 h 50 min. It is close to the 21.4 h of the "10 s" reading only
  because of the 202 long clips.

The paper's 12 h 50 min implies a mean of 6.0 s per clip. The median here is 5.5 s, and the mean
of the clips up to 30 s is 6.55 s, giving 13.6 h in total. **Inference:** the 12 h 50 min figure
is closer to the bulk of the clips than to the long tail. Whether the raw clips have the same
durations as the 320p copies cannot be checked here (section 9).

## 4. Frame rate, resolution, codec

- **fps:** 30.0 for all 7,701 clips. This holds for `CAP_PROP_FPS`, PyAV `average_rate` and
  PyAV `base_rate`. The first pts is 0 in every file.
- **Frame-count consistency:** `CAP_PROP_FRAME_COUNT`, the PyAV header frame count, the number
  of demuxed packets and the sequentially decoded count are equal for every clip. The container
  duration equals n_frames / 30 within 3.3e-7 s. **Inference:** for these files the reported
  frame count is reliable.
- **Resolution:** 568x320 for 7,241 clips and **570x320 for 460**. The 570-wide clips are
  high IDs 1-56 and 328-496, and low IDs 2181-2360, 2596 and 2645-2698. Split: 371 train,
  52 val, 37 test.
- **Codec:** h264, yuv420p, a single stream per file. The stream `encoder` tag is
  `Lavc60.31.102 libx264` on every file; the container tag is `Lavf60.16.100`.
- **Size:** 4.42 GB in total. The median bitrate is 57 kB/s.

## 5. Audio

None of the 7,701 320p files has an audio stream: `n_streams = 1` (video) and
`n_audio_streams = 0`, by PyAV. In `clip_audit.csv`, `has_audio` is False for every readable
clip and NA for the four missing ones; `audio_codec` is empty.

This follows from how the copies were made: `scripts/preprocess_resize.py` passes `-an` (section
9). Whether the **raw** clips carry audio cannot be determined on this machine: **NA**. Answering
it needs the raw files, or at least one of them.

## 6. Exact duplicate files (anomaly)

Duplicates were found by grouping the md5 of the file bytes.

| | 320p copies (local md5) | raw videos (DVC manifest md5) |
|---|---|---|
| groups with more than one file | 130 | 128 |
| files in those groups | 409 | 299 |
| redundant copies (files minus groups) | **279** | **171** |
| group sizes | 126 x 2, 10, 31, 57, 59 | 125 x 2, 6, 21, 22 |
| groups with more than one label | **0** | 0 |
| groups spanning more than one split | 49 | 46 |
| test clips with a byte-identical copy in train | **31** | **26** |

Every raw duplicate group is also a 320p duplicate group. Five 320p groups merge raw files that
differ:

| 320p group | IDs | frames | splits | distinct raw md5 |
|---|---|---|---|---|
| 59 clips | high 1591-1765 (every 3rd ID) | 4 frames each (0.13 s) | 49 / 3 / 7 | 39 |
| 57 clips | high 1831-1887 | 9,619 frames each (320.6 s) | 53 / 3 / 1 | 36 |
| 31 clips | high 1304-1394 (every 3rd ID) | 9 frames each (0.3 s) | 25 / 5 / 1 | 26 |
| 10 clips | low 13, 14, 33, 35-41 | 11 frames each (0.37 s) | 9 / 1 / 0 | 10 |
| 2 clips | low 1584, 1585 | 170 frames | | 2 |

**Observation:** these different raw files decode, after the student's transcode, to
byte-identical 320p files. The cause cannot be determined without the raw files.

The 126 duplicate pairs are ordinary-length clips (146-341 frames): 103 low and 23 mid. They
follow ID blocks:

- low 1060-1158 duplicates low 1512-1583 and then about every 4th ID from 1586 to 1689.
- mid 818-840 duplicates mid 841-863.
- a few pairs are adjacent IDs, for example low 134/135, 1405/1406 and 1584/1585.

Split membership of the pairs: train/train 80, test/train 22, train/val 20, test/val 3,
val/val 1.

**Observation:** 100 of the 113 clips with fewer than 64 frames belong to the 59-, 31- and
10-clip groups. The other 13 have 63 frames each.

The 57 copies of the 320.6 s clip make up 5.08 h of the 23.18 h total.

All groups, one row per file: `outputs/research/clip_audit_duplicates.csv` (with the raw md5).

## 7. Other anomalies

- **Very long clips.** 202 clips last more than 30 s, 175 more than 60 s; the maximum is 322 s
  (high/view1888, 9,659 frames). They hold 146 distinct contents, because 57 of them are the
  identical 320.6 s file.
- **Very short clips.** 100 clips are under 1 s and 113 have fewer than 64 frames; the minimum is
  4 frames.
- **Two output widths**, 568 and 570 (section 4).
- **Stray manifest row and raw file:** the unlabelled `videos/anonymity/a.mp4` row in `test.csv`,
  and a `.DS_Store` in the raw video directory.
- **Encoder:** the files were encoded with libx264, while the default of
  `preprocess_resize.py --encoder` is `h264_nvenc` (section 9).

## 8. Agreement with the October review (Experiment 1 of `docs/review/01_leakage_analysis.md`)

- Same frame-count percentiles, 1/5/25/50/75/95/99: 9 / 77 / 158 / 165 / 300 / 355 / 5,068.
- Same four unreadable clips.
- The October count of 117 clips under 64 frames (98 / 10 / 9) is this audit's 113 readable
  clips (94 / 10 / 9) plus the 4 unreadable ones, which `01_frame_cache.py` counts with a
  reported frame count of 0.
- The October run also found reported equal to actual frame counts.

## 9. What `scripts/preprocess_resize.py` does, and what it could have changed

The script has a single version in git (commit b2c0b5c, 2026-06-05). The 320p data were added to
DVC in commit 24e3dec (2026-10-26). It builds one ffmpeg call per clip:

```
ffmpeg -y -hide_banner -loglevel error -threads {threads} [-t {max_seconds}] -i SRC
       -vf scale='if(gt(iw,ih),-2,320)':'if(gt(iw,ih),320,-2)'
       -c:v {encoder} {-preset p4 -cq 24 | -preset veryfast -crf 24}
       -g 15 -pix_fmt yuv420p -an DST
```

- **Audio:** `-an` drops it. The 320p copies cannot carry audio, and their lack of audio says
  nothing about the raw clips.
- **Duration:** `-t {max_seconds}` is an input option placed before `-i`. The docstring says it
  "stops *decoding* after max_seconds", and `--max-seconds` defaults to 30. The comment next to
  it reads: "the few very long clips (up to 323 s) … a 30 s cap leaves >99% of clips (p99 ~14.5 s)
  untouched".
  - **Observation:** 202 of the 320p files are longer than 30 s, up to 322 s. They were therefore
    not produced by this code with `--max-seconds 30`: either a run with `--max-seconds 0` or a
    run without the `-t` option.
  - **Observation:** the comment's p99 of about 14.5 s does not match the 320p p99 of 168.9 s.
    Its maximum ("up to 323 s") does match the 320p maximum of 322 s.
- **Frame rate:** there is no `-r`, `fps=` filter, `-vsync` or `-fps_mode` option, so the output
  frame timing follows ffmpeg's defaults for the input. The docstring claims "the full frame
  sequence and fps are preserved". Every output is constant 30 fps. This cannot be compared with
  the raw files here: their fps and frame counts are not local.
- **Resolution:** the short side becomes 320, the aspect ratio is kept, and the long side is
  rounded to an even number (`-2`). This yields 568 or 570 depending on the source aspect. The
  docstring calls the sources "mostly 1080p HEVC"; this could not be checked here.
- **Encoder:** the default is `h264_nvenc` with `-cq 24`. The files carry the tag
  `Lavc60.31.102 libx264`, so the copies were made with `--encoder libx264`, that is
  `-preset veryfast -crf {cq}`. The cq value cannot be read from the files.
- **GOP and pixel format:** `-g 15` and `-pix_fmt yuv420p` change only the encoding, not the
  number of frames.

## 10. Memory needed by `scripts/review/01_frame_cache.py` on these clips

This matters for the scene-cluster rebuild. `decode()` appends every decoded BGR frame of a clip
to a Python list before keeping three of them, so one clip needs
n_frames x 320 x 568 (or 570) x 3 bytes.

- Median 90 MB, p99 2.77 GB, maximum **5.27 GB** (high/view1888).
- 175 clips need more than 1 GB, 120 more than 2 GB, and **58 more than 4 GB**: the 57 identical
  9,619-frame copies plus view1888.
- A simulation of its `Pool(8).imap_unordered(..., chunksize=4)` in manifest order gives a peak of
  about 23.5 GB. The simulation assumes decode time proportional to n_frames and memory released
  at the end of each clip, so it is optimistic.

On this 7.6 GiB machine, with about 3.4 GiB available during the session, the thumbnail step of
`reproduce.sh` cannot run within the 4 GB limit, even with one worker. On the October host it ran
with 1 TB of RAM.

## Files

| file | content |
|---|---|
| `scripts/research/01_clip_audit.py` | the audit |
| `scripts/research/01b_clip_audit_summary.py` | statistics, duplicate groups, memory estimate, figure |
| `outputs/research/clip_audit.csv` | one row per manifest record (7,706): `path, label, split, exists, readable, n_frames, fps, width, height, duration_s, file_bytes, has_audio, audio_codec`, plus `manifest_line, n_reported, md5, n_audio_streams, n_streams, container_format, container_duration_s, video_codec, pix_fmt, av_*, container_encoder, video_encoder, cv2_error, av_error, probe_s` |
| `outputs/research/clip_audit_sample200.csv` | the 200-clip timing sample |
| `outputs/research/clip_audit_summary.json` | every number in this report |
| `outputs/research/clip_audit_duplicates.csv` | the 409 files in 320p duplicate groups, with raw md5 |
| `outputs/research/raw_videos_dvc_dir.json` | DVC directory manifest of the raw videos (relpath, md5) |
| `outputs/research/clip_audit.log`, `clip_audit_summary.log`, `timings.txt` | logs and wall times |
| `docs/research/figures/clip_duration_hist.png` | duration histogram |
