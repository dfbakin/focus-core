# Probe: do clip IDs carry temporal order, and are they interleaved over cameras?

> **Corrected by `06_temporal_structure.md` §10 (2026-11-08).** The "consecutive-frame floor" of 0.2-0.5 measured here
> is the ffmpeg padding duplicate at the start of the 320p copies, not a real frame step (the real step is 2.8 at
> 128x72). `low/view500-560` has no gap: its IDs run backwards in time and the segments overlap by about 8 frames.
> Multi-camera interleaving is confirmed, lock-step cutting is not; some alternations are re-cuts of the same
> footage or two different groups; periods up to 6 occur. The probe is kept as the record of the hypothesis.

Claude Fable 5.1, 2026-11-07. Script: `scripts/research/03_id_offset_probe.py` (CPU, 3 x 61 clips of the
320p copies, 32x32 RGB thumbnails, RMSE on the 0-255 scale). Motivation: the clip audit found groups of
byte-identical broken clips at **every third ID** (`high/view1591..1765`, `view1304..1394`), which would be
the signature of three cameras exported in lock-step; and the October probe only compared `viewN` with
`viewN+1`.

For each contiguous ID range the table gives the median RMSE between the first frame of clip N and the
first frame of clip N+k, and between the **last** frame of N and the first frame of N+k, for k = 1..6.
Reference scales: the median RMSE between two consecutive frames inside a clip (the floor), and between
the first and last frame of a clip (5.5 s of motion).

```
high ids 1000-1060: n=61, median n_frames=167; consecutive-frame floor 0.5; first->last within clip 41.9
  offset k | first(N)->first(N+k) | last(N)->first(N+k)
         1 |       45.5 |       39.1
         2 |       44.2 |       19.2
         3 |       51.1 |       48.1
         4 |       47.2 |       44.4
         5 |       54.5 |       51.4
         6 |       48.3 |       49.2

low ids 500-560: n=61, median n_frames=165; consecutive-frame floor 0.2; first->last within clip 23.8
         1 |       24.1 |       29.4
         2 |       29.5 |       33.0
         3 |       33.1 |       35.4
         4 |       35.8 |       36.6
         5 |       37.2 |       38.1
         6 |       38.0 |       38.1

mid ids 300-360: n=61, median n_frames=163; consecutive-frame floor 0.2; first->last within clip 17.3
         1 |       88.7 |       88.7
         2 |       17.2 |        2.6
         3 |       88.6 |       88.6
         4 |       24.9 |       18.9
         5 |       88.7 |       88.6
         6 |       29.4 |       28.0
```

Observations.
- `mid/view300-360`: odd offsets land in a different room (88.7, the random-pair level); even offsets stay
  in the scene, and `last(N) -> first(N+2)` is 2.6, within a few frames' worth of motion. Two sources
  alternate ID by ID, and within each source consecutive IDs are consecutive in time.
- `high/view1000-1060`: the same two-way alternation, weaker: `last(N) -> first(N+2)` 19.2 against 39-51
  for the other offsets, in a scene with a lot of motion (first-to-last within a clip 41.9).
- `low/view500-560`: no alternation; similarity decays smoothly with the ID distance (24 -> 38), which is a
  single source in ID order. `last(N) -> first(N+1)` (29.4) is larger than `first(N) -> first(N+1)`
  (24.1), so adjacent clips are not frame-contiguous: there is a gap between segments, or the first and
  last frames of the 320p copies are not the segment boundaries.
- Nowhere does `last(N) -> first(N+k)` reach the consecutive-frame floor (0.2-0.5), so no pair of clips
  in these ranges is a frame-exact continuation.

Inferences (to be tested on all 7,701 clips with the 128x72 first/last-frame cache).
- The ID axis is a concatenation of export batches with different interleaving: one source, two
  sources alternating, and (from the duplicate groups) three sources in lock-step. Within a batch the IDs
  of one source follow time. This is the structure the temporal reconstruction should recover: for every
  clip the best successor offset k and the run structure along the ID axis.
- "Sources" that alternate are most likely the cameras of one session, which is the three-camera
  question of research question (a): the same instant would then sit at IDs N, N+1 (and N+2) and a
  random split puts them on different sides.
