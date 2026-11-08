#!/usr/bin/env python
"""Tables and figures for the raw OUC-CGE header audit.

Inputs (all read-only):
  outputs/research/raw_header_audit.csv     scripts/research/raw_header_audit.py (header pass)
  outputs/research/raw_md5.csv              same script, --md5 pass
  outputs/research/raw_video_packets.npz    same script, --packets pass (optional, section f)
  outputs/research/raw_videos_dvc_dir.json  DVC's per-file md5 listing of the raw release (cross-check)
  data/ouc-cge/data_labels/{train,val,test}.csv
  outputs/research/clip_audit.csv           320p copies (decoded frame counts; other agent's audit)
  data/ouc-cge/videos/high/view*.mp4        only the unreadable files, to count zero bytes

Outputs:
  outputs/research/raw_audit_tables.md
  outputs/research/raw_audit_joined.csv            one row per raw clip (+ label, split, md5, 320p columns)
  outputs/research/raw_audit_id_runs.csv           runs of identical batch along the ID axis
  outputs/research/raw_audit_pts_series.csv        clips with first pts >= 1 s, with their series key
  outputs/research/raw_audit_broken.csv            clips decoded fully (<= 5 frames, < 1 s, < 5 fps)
  outputs/research/raw_audit_duplicate_groups.csv  exact (md5) duplicate groups
  outputs/research/raw_audit_shared_packet_pairs.csv  file pairs sharing byte-identical video packets
  docs/research/figures/raw_*.png

  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/raw_header_audit_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/research"
FIG = ROOT / "docs/research/figures"
LABELS = {0: "low", 1: "mid", 2: "high"}
CLS_ORDER = ["low", "mid", "high"]
SPLITS = ["train", "val", "test"]
PAPER_TOTAL_S = 12 * 3600 + 50 * 60
TB = 15360  # every readable stream has time_base 1/15360 (asserted in main)
MIN_PKT = 1024  # packets smaller than this are ignored in the shared-packet analysis

# figure style: reference palette, light surface (same as 01b_clip_audit_summary.py)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CLS_COL = {"low": CAT[0], "mid": CAT[1], "high": CAT[2]}

md: list[str] = []


def w(s: str = "") -> None:
    md.append(s)


def vc(s: pd.Series, n: int | None = None) -> str:
    v = s.fillna("(missing)").astype(str).value_counts()
    if n:
        v = v.head(n)
    return "; ".join(f"{k}: {c}" for k, c in v.items())


def table(df: pd.DataFrame, index: bool = True, floatfmt: str = "{:.4g}") -> None:
    d = df.reset_index() if index else df.copy()
    cols = [str(c) for c in d.columns]
    w("| " + " | ".join(cols) + " |")
    w("|" + "|".join("---" for _ in cols) + "|")
    for _, r in d.iterrows():
        cells = []
        for v in r.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        w("| " + " | ".join(cells) + " |")
    w()


def read_manifests() -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        with open(ROOT / f"data/ouc-cge/data_labels/{split}.csv") as fh:
            for ln, line in enumerate(fh, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                lab = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
                p = parts[0].split("/")
                rows.append(dict(m_path=parts[0], cls=p[1], basename=p[2], label=lab, split=split, m_line=ln))
    return pd.DataFrame(rows)


def pct_table(s: pd.Series, qs=(1, 5, 25, 50, 75, 95, 99)) -> pd.DataFrame:
    s = s.dropna()
    row = {"n": len(s), "min": s.min()}
    for q in qs:
        row[f"p{q}"] = np.percentile(s, q)
    row["max"] = s.max()
    row["mean"] = s.mean()
    return pd.DataFrame([row])


def chi2_v(ct: pd.DataFrame) -> tuple[float, float]:
    ct = ct.loc[ct.sum(1) > 0, ct.sum(0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan"), float("nan")
    chi2, p, _, _ = stats.chi2_contingency(ct.values)
    return p, float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))


def rate_bin(r: float) -> str:
    if pd.isna(r):
        return "(missing)"
    if r == 30:
        return "30 exactly"
    if 30 < r <= 30.001:
        return "30 < r <= 30.001"
    if 29 <= r < 30:
        return "29 <= r < 30"
    if 5 <= r < 29:
        return "5 <= r < 29"
    return "< 5"


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(OUT / "raw_header_audit.csv", low_memory=False)
    ok = df.error.isna()
    df["readable"] = ok
    df["resolution"] = np.where(
        ok, df.v_width.astype("Int64").astype(str) + "x" + df.v_height.astype("Int64").astype(str), None)
    df["fps"] = df.v_average_rate
    df["rate_bin"] = df.v_average_rate.map(rate_bin)
    df["first_pts"] = df.pkt0_pts
    df["first_pts_s"] = df.pkt0_pts_s
    df["pts_minus_dts0"] = df.pkt0_pts - df.pkt0_dts
    fd = df.full_decoded.fillna(False).astype(bool)
    df["real_frames"] = np.where(fd, df.dec_n_frames, df.v_frames)
    df["n_audio"] = df.n_audio_streams.astype("Int64")
    df["encoder"] = df.ctag_encoder.fillna("(none)")
    df["video_encoder"] = df.vtag_encoder.fillna("(none)") if "vtag_encoder" in df else "(none)"
    df["combo"] = np.where(
        ok, df.v_codec.astype(str) + "|" + df.resolution.astype(str) + "|a" + df.n_audio.astype(str) + "|" + df.encoder,
        "(unreadable)")
    df["end_s"] = df.first_pts_s + df.v_duration_s
    man = read_manifests()
    lab = man[man.label.notna()]
    df = df.merge(lab[["cls", "basename", "label", "split", "m_path"]], on=["cls", "basename"], how="left")
    m5 = pd.read_csv(OUT / "raw_md5.csv")
    df = df.merge(m5[["relpath", "md5"]], on="relpath", how="left")
    dvc = OUT / "raw_videos_dvc_dir.json"
    if dvc.exists():
        dv = pd.DataFrame(json.load(open(dvc))).rename(columns={"md5": "dvc_md5"})
        df = df.merge(dv, on="relpath", how="left")
    df = df.sort_values(["cls", "clip_id"]).reset_index(drop=True)
    return df, man


# ----------------------------------------------------------------------------------------------
def sec0(df: pd.DataFrame, man: pd.DataFrame) -> None:
    w("# Raw OUC-CGE release: header audit tables")
    w()
    w(f"Generated by `scripts/research/raw_header_audit_tables.py`. Source: `outputs/research/raw_header_audit.csv` "
      f"({len(df)} mp4 files, PyAV 17.0.1, headers and the first 16 video packets; full decode only of clips with "
      f"<= 5 header frames, < 1 s video or < 5 fps average rate). Durations are video-stream header durations "
      f"(`stream.duration * time_base`) unless stated. time_base is 1/{TB} s (one frame at 30 fps = 512 ticks).")
    w()
    w("## 0. Inventory, joins, unreadable files")
    w()
    lab = man[man.label.notna()]
    w(f"- mp4 files: {len(df)} (" + ", ".join(f"{c} {int((df.cls == c).sum())}" for c in CLS_ORDER)
      + "); one non-mp4 file `.DS_Store` at the top of `videos/`")
    w(f"- readable by PyAV: {int(df.readable.sum())}; unreadable: {int((~df.readable).sum())}")
    w(f"- labelled manifest rows: {len(lab)} (train {int((lab.split == 'train').sum())}, val "
      f"{int((lab.split == 'val').sum())}, test {int((lab.split == 'test').sum())}); unlabelled row: "
      f"{', '.join(man[man.label.isna()].m_path)} (test.csv line {', '.join(str(x) for x in man[man.label.isna()].m_line)})")
    miss = lab.merge(df[["cls", "basename"]], on=["cls", "basename"], how="left", indicator=True)
    w(f"- raw files without a manifest row: {int(df.label.isna().sum())}; manifest rows without a raw file: "
      f"{int((miss._merge == 'left_only').sum())}; class directory != label: "
      f"{int((df.label.notna() & (df.label.map(LABELS) != df.cls)).sum())}")
    if "dvc_md5" in df:
        w(f"- md5 of file bytes (own pass) equals DVC's recorded md5 for {int((df.md5 == df.dvc_md5).sum())}/{len(df)} files")
    w()
    bad = df[~df.readable]
    rows = []
    for _, r in bad.iterrows():
        a = np.fromfile(ROOT / "data/ouc-cge/videos" / r.relpath, dtype=np.uint8)
        nz = np.flatnonzero(a)
        rows.append(dict(relpath=r.relpath, split=r.split, bytes=a.size, nonzero_bytes=nz.size,
                         last_nonzero_offset=int(nz[-1]) if nz.size else -1,
                         head=a[4:8].tobytes().decode("latin1") if nz.size else "(all zero)", error=r.error[:40]))
    if rows:
        w("Unreadable files (PyAV `av.open` raises). Byte counts read with numpy; `head` = bytes 4-8 (an mp4 starts with `ftyp`).")
        w()
        table(pd.DataFrame(rows), index=False)


def sec_a(df: pd.DataFrame) -> None:
    h = df[df.readable]
    w("## a. Distributions (readable files)")
    w()
    w(f"n = {len(h)}")
    w()
    props = [("v_codec", "video codec"), ("v_profile", "video profile"), ("resolution", "resolution"),
             ("v_pix_fmt", "pix_fmt"), ("rate_bin", "average_rate (frames / duration)"),
             ("v_guessed_rate", "guessed_rate"), ("v_base_rate", "r_frame_rate (base_rate)"),
             ("v_time_base", "video time_base"), ("n_audio", "audio stream count"), ("audio_codecs", "audio codecs"),
             ("a0_sample_rate", "audio[0] sample rate"), ("a0_layout", "audio[0] layout"),
             ("format_name", "container format"), ("ctag_major_brand", "major_brand"),
             ("ctag_minor_version", "minor_version"), ("ctag_compatible_brands", "compatible_brands"),
             ("encoder", "container `encoder` tag"), ("video_encoder", "video-stream `encoder` tag"),
             ("vtag_handler_name", "video handler_name"), ("a0_tag_handler_name", "audio handler_name"),
             ("stream_types", "stream layout"), ("pkt0_key", "first packet is keyframe")]
    for col, name in props:
        if col in h:
            w(f"- **{name}** (`{col}`): {vc(h[col])}")
    tags = sorted(c for c in df.columns if "tag_" in c)
    w(f"- metadata tag columns seen in any file: {', '.join(f'`{c}`' for c in tags)}")
    if "o0tag_creation_time" in h:
        oc = h[h.o0tag_creation_time.notna()]
        w("- no container-level `creation_time` tag in any file; stream-level `creation_time` only in: " + "; ".join(
            f"`{r.relpath}` (streams {r.stream_types}; the non-A/V stream has handler_name `{r.o0tag_handler_name}` and "
            f"creation_time {r.o0tag_creation_time})" for _, r in oc.iterrows()))
    w()
    w("Per class (counts):")
    w()
    for col in ["v_codec", "resolution", "n_audio", "encoder", "a0_sample_rate", "rate_bin", "combo"]:
        ct = pd.crosstab(h[col].fillna("(missing)").astype(str), h.cls).reindex(columns=CLS_ORDER, fill_value=0)
        ct["total"] = ct.sum(1)
        w(f"`{col}`" + (" = codec | WxH | audio streams | container encoder tag" if col == "combo" else ""))
        w()
        table(ct.sort_values("total", ascending=False))
    fs = h.file_bytes / 1e6
    w(f"**File size**: total {df.file_bytes.sum() / 1e9:.3f} GB ({df.file_bytes.sum() / 2**30:.3f} GiB) over all "
      f"{len(df)} files. MB, readable files:")
    w()
    table(pct_table(fs), index=False)
    d = h.v_duration_s
    w("**Video stream duration (s)**, header, readable files:")
    w()
    table(pct_table(d), index=False)
    w("by class:")
    w()
    table(pd.concat([pct_table(d[h.cls == c]).assign(cls=c) for c in CLS_ORDER]).set_index("cls"))
    cnt = {"< 1 s": (d < 1).sum(), "< 5 s": (d < 5).sum(), "5 to < 9.5 s": ((d >= 5) & (d < 9.5)).sum(),
           "9.5-10.5 s": d.between(9.5, 10.5).sum(), "> 10.5 s": (d > 10.5).sum(), "> 30 s": (d > 30).sum(),
           "> 60 s": (d > 60).sum(), "exactly 300 frames": (h.v_frames == 300).sum()}
    w("Counts: " + "; ".join(f"{k}: {int(v)}" for k, v in cnt.items()))
    w()
    tot, totf, totc = d.sum(), (h.real_frames / 30).sum(), h.container_duration_s.sum()
    w(f"- total video duration (header): {tot:.1f} s = {tot / 3600:.3f} h = {int(tot // 3600)} h {tot % 3600 / 60:.1f} min")
    w(f"- total real frames / 30: {h.real_frames.sum():.0f} frames = {totf / 3600:.3f} h")
    w(f"- total container duration (max end over streams, includes pts offsets and audio): {totc / 3600:.3f} h")
    w(f"- paper: 12 h 50 min = {PAPER_TOTAL_S} s = {PAPER_TOTAL_S / 3600:.3f} h, i.e. {PAPER_TOTAL_S / 7705:.3f} s per clip; "
      f"7705 x 10 s = {7705 * 10 / 3600:.3f} h. Measured mean {d.mean():.3f} s, median {d.median():.3f} s per clip")
    w(f"- clips at 1280x720 (paper's stated resolution): {int((h.resolution == '1280x720').sum())}; at 1920x1080: "
      f"{int((h.resolution == '1920x1080').sum())}")
    a0 = h.a0_duration_s
    w(f"- audio[0] duration: total {a0.sum() / 3600:.3f} h; files with no audio stream: {int((h.n_audio == 0).sum())}; "
      f"audio[0] of one AAC frame (1024 samples, 0.0213 s at 48 kHz): {int((a0.round(4) == 0.0213).sum())}; "
      f"|audio[0] - video| > 1 s: {int(((a0 - d).abs() > 1).sum())}")
    w(f"- ticks between the first two video packets: {vc(h.pkt1_minus_pkt0_pts.astype('Int64'))}; pts reordered within "
      f"the first 16 packets (B-frames): {vc(h.probe_pts_reordered)}")
    dens = h.v_frames / (h.v_duration_s * 30)
    w(f"- header frames / (duration x 30): < 0.5: {int((dens < 0.5).sum())}; 0.5-0.9: "
      f"{int(((dens >= 0.5) & (dens < 0.9)).sum())}; 0.9-0.999: {int(((dens >= 0.9) & (dens < 0.999)).sum())}; "
      f">= 0.999: {int((dens >= 0.999).sum())}")
    w()
    sp = h[h.v_average_rate < 5]
    if len(sp):
        w("Clips with average rate < 5 fps (header frames spread over a long span; decoded fully):")
        w()
        table(sp[["relpath", "v_frames", "dec_n_frames", "v_duration_s", "v_average_rate", "first_pts_s",
                  "dec_first_pts_s", "dec_last_pts_s", "a0_duration_s", "split"]], index=False)


def sec_b(df: pd.DataFrame) -> None:
    h = df[df.readable].copy()
    fp = h.first_pts_s
    w("## b. First-PTS structure")
    w()
    w(f"First video packet in demux order. Its pts equals the stream `start_time` in "
      f"{int((h.pkt0_pts == h.v_start_time_ticks).sum())}/{len(h)} files and the smallest pts among the first 16 packets "
      f"in {int((h.pkt0_pts == h.probe_min_pts).sum())}/{len(h)}; it is a keyframe in {int(h.pkt0_key.sum())}.")
    w()
    cut = pd.cut(fp, [-np.inf, -1e-12, 1e-12, 0.1, 1, np.inf], right=False,
                 labels=["< 0", "= 0", "(0, 0.1) s", "[0.1, 1) s", ">= 1 s"])
    ct = pd.crosstab(cut, h.cls).reindex(columns=CLS_ORDER, fill_value=0)
    ct["total"] = ct.sum(1)
    table(ct)
    table(pd.crosstab(cut, h.v_codec))
    w(f"- first pts modulo 512 ticks (position within a frame interval), most common: {vc((h.first_pts % 512).astype(int), 8)}")
    w(f"- pts - dts of the first packet (ticks; 1024 = two-frame B-frame delay): {vc(h.pts_minus_dts0.astype(int), 12)}")
    w()
    big = h[fp >= 1].copy()
    big["residue"] = (big.first_pts % 7680).astype(int)  # position within a 0.5 s grid
    # cluster residues within (class, combo) with a 64-tick (4 ms) tolerance: start times jitter by ~1 ms
    big = big.sort_values(["cls", "combo", "residue"])
    newc = big.groupby(["cls", "combo"]).residue.diff().fillna(1e9) > 64
    big["series_id"] = newc.cumsum()
    # ... and split where successive IDs (sorted) differ by more than 1000
    big = big.sort_values(["series_id", "clip_id"])
    sub = (big.groupby("series_id").clip_id.diff().fillna(0) > 1000).astype(int).groupby(big.series_id).cumsum()
    big["series_id"] = big.series_id.astype(str) + "_" + sub.astype(str)
    big["series"] = (big.cls + "|" + big.combo + "|r" + big.groupby("series_id").residue.transform("min").astype(str)
                     + "|id" + big.groupby("series_id").clip_id.transform("min").astype(str))
    rows = []
    for key, g in big.groupby("series"):
        g = g.sort_values("first_pts")
        starts = np.unique(g.first_pts_s.round(3))
        steps = np.diff(starts)
        rho = stats.spearmanr(g.clip_id, g.first_pts_s)[0] if len(g) >= 3 and g.first_pts_s.nunique() > 1 else np.nan
        nxt = g.groupby(g.first_pts_s.round(3)).end_s.max()
        gaps = (nxt.index[1:].values - nxt.values[:-1]) if len(nxt) > 1 else np.array([])
        rows.append(dict(series=key, n=len(g), ids=f"{int(g.clip_id.min())}-{int(g.clip_id.max())}",
                         id_steps=vc(pd.Series(np.diff(np.sort(g.clip_id.values))), 3),
                         start_min_s=starts.min(), start_max_s=starts.max(), n_starts=len(starts),
                         start_steps_s=vc(pd.Series(steps.round(2)), 3),
                         clips_per_start=vc(g.first_pts_s.round(3).value_counts(), 2),
                         median_dur_s=g.v_duration_s.median(),
                         next_start_minus_end_s=np.median(gaps) if len(gaps) else np.nan,
                         rho_id_pts=rho, frames=vc(g.real_frames.astype(int), 2),
                         audio0_s=vc(g.a0_duration_s.round(3), 2), splits=vc(g.split)))
    ser = pd.DataFrame(rows).sort_values("n", ascending=False)
    big.to_csv(OUT / "raw_audit_pts_series.csv", index=False)
    w(f"### Clips with first pts >= 1 s: {len(big)}")
    w()
    w("Grouped into series by (class, batch combo, first pts modulo 0.5 s, clustered with a 4 ms tolerance, split where successive IDs differ by > 1000). `start_steps_s` = differences between "
      "successive distinct start times; `next_start_minus_end_s` (median) < 0 means successive clips overlap in source "
      "time; `clips_per_start` = how many clips share one start time; `rho_id_pts` = Spearman(ID, first pts) within the "
      "series (+1: IDs follow source time, -1: IDs run backwards).")
    w()
    table(ser, index=False)
    w("### Relation to clip ID")
    w()
    rows = []
    for c in CLS_ORDER + ["all"]:
        s = h if c == "all" else h[h.cls == c]
        r, p = stats.spearmanr(s.clip_id, s.first_pts_s)
        r2, p2 = stats.spearmanr(s.clip_id, s.v_duration_s)
        rows.append(dict(cls=c, n=len(s), rho_firstpts_id=r, p=p, rho_duration_id=r2, p_dur=p2))
    w("Spearman correlation over all readable clips of a class:")
    w()
    table(pd.DataFrame(rows), index=False)
    rng = np.random.default_rng(0)
    rows = []
    for c in CLS_ORDER:
        s = df[df.cls == c].sort_values("clip_id")  # unreadable files keep their slot as '(unreadable)'
        combo = s.combo.values
        for lag in (1, 2, 3):
            same = combo[lag:] == combo[:-lag]
            perm = []
            for _ in range(1000):
                pc = rng.permutation(combo)
                perm.append((pc[lag:] == pc[:-lag]).mean())
            perm = np.array(perm)
            rows.append(dict(cls=c, lag=lag, pairs=len(same), frac_identical=same.mean(), perm_mean=perm.mean(),
                             perm_max=perm.max(), perm_p=(1 + (perm >= same.mean()).sum()) / 1001))
    gaps = {c: int((np.diff(np.sort(df[df.cls == c].clip_id.values)) != 1).sum()) for c in CLS_ORDER}
    w("Fraction of ID pairs (k apart after sorting by ID within class) with identical batch combo "
      "(codec, width, height, audio-stream count, container encoder tag). Permutation baseline: IDs shuffled "
      f"within class, 1000 draws, seed 0. Gaps in the ID sequence per class: {gaps}; IDs run 1..n.")
    w()
    table(pd.DataFrame(rows), index=False)
    runs = []
    for c in CLS_ORDER:
        s = df[df.cls == c].sort_values("clip_id")
        brk = (s.combo != s.combo.shift()).cumsum()
        for _, rr in s.groupby(brk):
            runs.append(dict(cls=c, id_from=int(rr.clip_id.min()), id_to=int(rr.clip_id.max()), n=len(rr),
                             combo=rr.combo.iloc[0], median_dur_s=rr.v_duration_s.median(),
                             n_first_pts_ge_1s=int((rr.first_pts_s >= 1).sum())))
    runs = pd.DataFrame(runs)
    runs.to_csv(OUT / "raw_audit_id_runs.csv", index=False)
    w("Runs of identical batch combo along the ID axis:")
    w()
    for c in CLS_ORDER:
        r = runs[runs.cls == c]
        w(f"- {c}: {len(r)} runs; longest {int(r.n.max())}; runs of length 1: {int((r.n == 1).sum())}; "
          f"runs of length >= 50: {int((r.n >= 50).sum())} covering {int(r.n[r.n >= 50].sum())} of "
          f"{int((df.cls == c).sum())} clips")
    w()
    w("Runs of length >= 20 (full list in `outputs/research/raw_audit_id_runs.csv`):")
    w()
    table(runs[runs.n >= 20], index=False)


def sec_c(df: pd.DataFrame) -> None:
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.tree import DecisionTreeClassifier, export_text

    lab = df[df.readable & df.label.notna()].copy()
    w("## c. Container metadata x label and x split")
    w()
    w(f"Readable, labelled clips: n = {len(lab)}.")
    w()
    props = ["v_codec", "v_profile", "resolution", "n_audio", "encoder", "video_encoder", "v_pix_fmt",
             "ctag_compatible_brands", "rate_bin", "a0_sample_rate", "combo"]
    rows = []
    for p_ in props:
        x = lab[p_].fillna("(missing)").astype(str)
        p1, v1 = chi2_v(pd.crosstab(x, lab.cls))
        p2, v2 = chi2_v(pd.crosstab(x, lab.split))
        maj = pd.crosstab(x, lab.cls).max(1).sum() / len(lab)
        rows.append(dict(property=p_, n_values=x.nunique(), chi2_p_class=p1, cramers_v_class=v1, chi2_p_split=p2,
                         cramers_v_split=v2, majority_rule_acc=maj))
    w("Chi-square test of independence (`scipy.stats.chi2_contingency`), Cramér's V, and the in-sample accuracy of "
      "predicting the class by the majority class of each property value (majority baseline = "
      f"{(lab.cls.value_counts().max() / len(lab)):.4f}). `creation_time` is absent from every container, so it is "
      "not tested.")
    w()
    table(pd.DataFrame(rows), index=False, floatfmt="{:.3g}")
    w("Property values that are (nearly) exclusive to one class (>= 95% of their clips in one class, n >= 20):")
    w()
    pur = []
    for p_ in props[:-1]:
        x = lab[p_].fillna("(missing)").astype(str)
        ct = pd.crosstab(x, lab.cls)
        for val, r in ct.iterrows():
            n = r.sum()
            if n >= 20 and r.max() / n >= 0.95 and n < len(lab):
                pur.append(dict(property=p_, value=val, n=int(n), cls=r.idxmax(), purity=r.max() / n,
                                share_of_that_class=r.max() / (lab.cls == r.idxmax()).sum()))
    if pur:
        table(pd.DataFrame(pur), index=False)
    for p_ in ["v_codec", "resolution", "n_audio", "encoder"]:
        x = lab[p_].fillna("(missing)").astype(str)
        w(f"`{p_}` x split")
        w()
        table(pd.crosstab(x, lab.split).reindex(columns=SPLITS, fill_value=0))
    feats = ["v_codec", "v_profile", "resolution", "n_audio", "encoder", "video_encoder", "v_pix_fmt",
             "ctag_compatible_brands", "rate_bin", "a0_sample_rate", "a1_sample_rate"]
    X = pd.get_dummies(lab[feats].fillna("(missing)").astype(str))
    y = lab.label.astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    res = []
    for name, model in [("majority baseline", DummyClassifier(strategy="most_frequent")),
                        ("DecisionTree(max_depth=4), categorical metadata",
                         DecisionTreeClassifier(max_depth=4, random_state=0))]:
        a = cross_val_score(model, X, y, cv=cv)
        res.append(dict(model=name, acc=a.mean(), std=a.std(), folds=" ".join(f"{v:.3f}" for v in a)))
    num = ["v_duration_s", "first_pts_s", "file_bytes", "v_bit_rate", "a0_duration_s"]
    Xn = pd.concat([X, lab[num].fillna(-1)], axis=1)
    a = cross_val_score(DecisionTreeClassifier(max_depth=4, random_state=0), Xn, y, cv=cv)
    res.append(dict(model="DecisionTree(max_depth=4), + numeric header values", acc=a.mean(), std=a.std(),
                    folds=" ".join(f"{v:.3f}" for v in a)))
    tr, te = (lab.split == "train").values, (lab.split == "test").values
    for name, XX in [("categorical metadata", X), ("+ numeric header values", Xn)]:
        clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(XX[tr], y[tr])
        res.append(dict(model=f"official train -> test, tree, {name}", acc=clf.score(XX[te], y[te]), std=np.nan, folds=""))
    res.append(dict(model="official train -> test, majority", acc=(y[te] == np.bincount(y[tr]).argmax()).mean(),
                    std=np.nan, folds=""))
    w("### Predicting the class from container metadata alone")
    w()
    w(f"One-hot of {len(feats)} categorical header properties ({', '.join(feats)}; {X.shape[1]} columns). "
      f"5-fold `StratifiedKFold(shuffle=True, random_state=0)` unless stated. The numeric variant adds "
      f"{', '.join(num)}.")
    w()
    table(pd.DataFrame(res), index=False)
    clf = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    w("Tree fitted on all labelled readable clips (categorical metadata); importances: "
      + "; ".join(f"`{k}` {v:.3f}" for k, v in imp[imp > 0].items()))
    w()
    w("```")
    w(export_text(clf, feature_names=list(X.columns), class_names=["low", "mid", "high"]))
    w("```")
    w()


def sec_d(df: pd.DataFrame) -> None:
    w("## d. Broken files and exact duplicates")
    w()
    h = df[df.readable]
    brk = h[h.full_decoded.fillna(False).astype(bool)].copy()
    brk.to_csv(OUT / "raw_audit_broken.csv", index=False)
    w(f"Decoded fully: {len(brk)} files (header frames <= 5: {int((h.v_frames <= 5).sum())}; video < 1 s: "
      f"{int((h.v_duration_s < 1).sum())}; average rate < 5 fps: {int((h.v_average_rate < 5).sum())}). Decoded frame "
      f"count equals the header frame count in {int((brk.dec_n_frames == brk.v_frames).sum())}/{len(brk)}; "
      f"decoded frames == packets in {int((brk.dec_n_frames == brk.dec_n_packets).sum())}/{len(brk)}.")
    w()
    w(f"- real frame count <= 5: **{int((h.real_frames <= 5).sum())}** files; <= 11: {int((h.real_frames <= 11).sum())}; "
      f"plus {len(df) - len(h)} unreadable files")
    w()
    sub = brk[brk.v_duration_s < 1].copy()
    sub["family"] = sub.cls + "|" + sub.v_frames.astype(int).astype(str) + "fr|pts" + sub.first_pts.astype(int).astype(str)
    fam = []
    for k, g in sub.groupby("family"):
        g = g.sort_values("clip_id")
        a0 = g.a0_duration_s.values
        sl = np.polyfit(np.arange(len(g)), a0, 1)[0] if len(g) > 2 else np.nan
        fam.append(dict(family=k, n=len(g), ids=f"{int(g.clip_id.min())}-{int(g.clip_id.max())}",
                        id_step=vc(pd.Series(np.diff(g.clip_id.values)), 2), real_frames=vc(g.real_frames.astype(int)),
                        first_pts_s=g.first_pts_s.iloc[0], video_s=vc(g.v_duration_s.round(3), 2),
                        audio0_min_s=a0.min(), audio0_max_s=a0.max(), audio0_step_s=np.median(np.abs(np.diff(a0))),
                        audio0_fit_slope=sl, n_at_max_audio=int((np.abs(a0 - a0.max()) < 1e-6).sum()),
                        bytes_min=int(g.file_bytes.min()), bytes_max=int(g.file_bytes.max()),
                        distinct_md5=g.md5.nunique(), splits=vc(g.split)))
    w("Families of clips with < 1 s of video (class | header frames | first pts in ticks). `audio0_step_s` = median "
      "absolute change of audio[0] duration between successive members (by ID).")
    w()
    table(pd.DataFrame(fam), index=False)
    w("All files with real frame count <= 5:")
    w()
    b5 = brk[brk.real_frames <= 5].sort_values(["cls", "clip_id"])
    table(b5[["relpath", "real_frames", "first_pts", "first_pts_s", "dec_last_pts_s", "a0_duration_s", "file_bytes",
              "split"]].assign(md5=b5.md5.str[:10]), index=False)
    others = brk[brk.real_frames > 5]
    if len(others):
        w("Other fully decoded files (6-11 frames, or < 5 fps):")
        w()
        table(others[["relpath", "v_frames", "real_frames", "first_pts_s", "v_duration_s", "dec_first_pts_s",
                      "dec_last_pts_s", "a0_duration_s", "file_bytes", "split"]], index=False)
    w("### Exact duplicates (md5 of the file bytes)")
    w()
    dup = df[df.md5.duplicated(keep=False)].copy()
    groups = []
    for hsh, gg in dup.groupby("md5"):
        gg = gg.sort_values(["cls", "clip_id"])
        ids = gg.clip_id.values
        groups.append(dict(md5=hsh[:10], n=len(gg), classes=",".join(sorted(gg.cls.unique())),
                           members=" ".join(f"{a}/{b}" for a, b in zip(gg.cls, gg.clip_id)),
                           id_diffs=",".join(str(int(x)) for x in np.unique(np.diff(ids))) if gg.cls.nunique() == 1 else "",
                           splits=vc(gg.split), real_frames=vc(gg.real_frames.fillna(0).astype(int)),
                           video_s=gg.v_duration_s.iloc[0], first_pts_s=gg.first_pts_s.iloc[0],
                           bytes=int(gg.file_bytes.iloc[0]), readable=bool(gg.readable.iloc[0])))
    g = pd.DataFrame(groups).sort_values(["n", "members"], ascending=[False, True])
    g.to_csv(OUT / "raw_audit_duplicate_groups.csv", index=False)
    in_tr = set(dup[dup.split == "train"].md5)
    in_va = set(dup[dup.split == "val"].md5)
    w(f"- files in a duplicate group: **{len(dup)}** in **{len(g)}** groups; redundant copies: {len(dup) - len(g)}; "
      f"distinct contents among {len(df)} files: {df.md5.nunique()}")
    w(f"- group sizes: {vc(g.n)}")
    w(f"- groups spanning two classes: {int(g.classes.str.contains(',').sum())}")
    w(f"- groups by class: {vc(g.classes)}")
    w(f"- ID difference between successive members (same-class groups): "
      f"{vc(g.id_diffs.str.split(',').explode(), 12)}")
    w(f"- groups whose members are in more than one split: {int((g.splits.str.count(':') > 1).sum())}")
    w(f"- test files with a byte-identical copy in train: **{int(((dup.split == 'test') & dup.md5.isin(in_tr)).sum())}** "
      f"of {int((df.split == 'test').sum())}; val files with a copy in train: "
      f"{int(((dup.split == 'val') & dup.md5.isin(in_tr)).sum())}; test files with a copy in val: "
      f"{int(((dup.split == 'test') & dup.md5.isin(in_va)).sum())}")
    w(f"- duplicate files that are broken (<= 5 real frames): {int((dup.real_frames <= 5).sum())}; unreadable: "
      f"{int((~dup.readable).sum())}; with >= 30 frames: {int((dup.real_frames >= 30).sum())}")
    w()
    table(g, index=False)


def sec_e(df: pd.DataFrame) -> pd.DataFrame:
    w("## e. Raw vs 320p copies")
    w()
    ca = pd.read_csv(OUT / "clip_audit.csv", low_memory=False)
    ca = ca[ca.path.str.match(r"^videos/(?:low|mid|medium|high)/view\d+\.mp4$")].copy()
    ca["cls"] = ca.path.str.split("/").str[1].replace({"medium": "mid"})
    ca["basename"] = ca.path.str.split("/").str[2]
    ca = ca.rename(columns={"n_frames": "p320_frames", "duration_s": "p320_duration_s", "fps": "p320_fps",
                            "exists": "p320_exists", "av_first_pts_s": "p320_first_pts_s"})
    df = df.merge(ca[["cls", "basename", "p320_exists", "p320_frames", "p320_fps", "p320_duration_s", "p320_first_pts_s"]],
                  on=["cls", "basename"], how="left")
    has = df.p320_exists.fillna(False).astype(bool)
    jj = df[df.readable & has].copy()
    jj["diff_s"] = jj.p320_duration_s - jj.v_duration_s
    jj["diff_frames"] = jj.p320_frames - jj.real_frames
    df["p320_minus_raw_s"] = df.p320_duration_s - df.v_duration_s
    dd = jj.diff_s
    w("320p source: `outputs/research/clip_audit.csv` (frames counted by decoding every frame with cv2; duration = "
      "frames / fps). Joined on class + basename.")
    w()
    w(f"- raw files with a 320p copy: {len(jj)}; without: {', '.join(df[~has].relpath)}")
    w(f"- 320p duration - raw video duration (s): " + "; ".join(
        f"{k} {v:.4g}" for k, v in dd.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).items()))
    w(f"- 320p longer by > 1 s: **{int((dd > 1).sum())}**; by > 0.1 s: {int((dd > 0.1).sum())}; within one frame "
      f"(|diff| <= 1/30 s): {int((dd.abs() <= 1 / 30 + 1e-6).sum())}; shorter by > 0.1 s: {int((dd < -0.1).sum())}; "
      f"shorter by > 1 s: {int((dd < -1).sum())}")
    w(f"- 320p frames - raw frames: {vc(jj.diff_frames.astype(int), 12)}")
    lng = jj[dd > 1]
    if len(lng):
        e = (lng.diff_s - lng.first_pts_s).abs()
        w(f"- among the {len(lng)} clips longer by > 1 s: |excess - raw first pts| median {e.median():.4f} s, max "
          f"{e.max():.4f} s; Pearson r(excess, raw first pts) = {np.corrcoef(lng.diff_s, lng.first_pts_s)[0, 1]:.5f}")
    allx = (jj.diff_s - jj.first_pts_s).abs() <= 1 / 30 + 1e-6
    w(f"- clips whose 320p excess equals the raw first pts within one frame: {int(allx.sum())}/{len(jj)}")
    w(f"- total duration over these clips: raw {jj.v_duration_s.sum() / 3600:.3f} h; 320p "
      f"{jj.p320_duration_s.sum() / 3600:.3f} h; difference {(jj.p320_duration_s.sum() - jj.v_duration_s.sum()) / 3600:.3f} h")
    w(f"- clips over 60 s: 320p {int((jj.p320_duration_s > 60).sum())}; raw {int((jj.v_duration_s > 60).sum())}")
    w()
    cols = ["relpath", "real_frames", "v_duration_s", "first_pts_s", "p320_frames", "p320_duration_s", "diff_s", "split"]
    w("20 largest positive differences (320p longer):")
    w()
    table(jj.sort_values("diff_s", ascending=False).head(20)[cols], index=False)
    w("10 most negative differences (320p shorter):")
    w()
    table(jj.sort_values("diff_s").head(10)[cols], index=False)
    return df


def sec_f(df: pd.DataFrame) -> pd.DataFrame | None:
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components

    npz = OUT / "raw_video_packets.npz"
    if not npz.exists():
        return None
    z = np.load(npz)
    rel = z["relpath"]
    P = pd.DataFrame({"f": z["fidx"], "size": z["size"], "key": z["key"], "hash": z["hash"]})
    w("## f. Shared encoded frames (byte-identical video packets) [beyond the brief]")
    w()
    w(f"`raw_header_audit.py --packets` demuxed (did not decode) every video packet of every readable file and hashed "
      f"its bytes (64-bit blake2b): {len(P)} packets. Two different files that contain the same encoded packet hold "
      f"the same encoded frame, i.e. were cut by stream copy from one source and overlap in time, whatever their "
      f"timestamps. Packets under {MIN_PKT} bytes are ignored ({int((P['size'] < MIN_PKT).sum())} packets) so that "
      f"tiny identical packets (e.g. all-skip frames) cannot link unrelated files. Exact md5 duplicates are in d.")
    w()
    allh = z["hash"]
    n_u = len(np.unique(allh))
    n_u1k = len(np.unique(allh[z["size"] >= MIN_PKT]))
    w(f"- distinct encoded frames: {n_u} of {len(allh)} packets ({n_u / len(allh):.1%}) = {n_u / 30 / 3600:.3f} h at "
      f"30 fps, against {len(allh) / 30 / 3600:.3f} h of frames in all files; packets >= {MIN_PKT} B: {n_u1k} distinct of "
      f"{int((z['size'] >= MIN_PKT).sum())}. Re-encoded copies of the same footage (if any) are not detectable this way, "
      f"so the distinct count is an upper bound on distinct footage.")
    w()
    meta = df.set_index("relpath")
    fi = pd.DataFrame({"relpath": rel, "f": np.arange(len(rel))})
    fi = fi.join(meta[["cls", "clip_id", "split", "md5", "real_frames"]], on="relpath")
    P = P[P["size"] >= MIN_PKT].drop_duplicates(["f", "hash"])
    fi["n_pk"] = fi.f.map(P.groupby("f").size()).fillna(0).astype(int)
    S = P[P.groupby("hash").f.transform("size") > 1]
    pairs = S.merge(S, on="hash", suffixes=("_a", "_b"))
    pairs = pairs[pairs.f_a < pairs.f_b]
    pr = pairs.groupby(["f_a", "f_b"]).agg(shared=("hash", "size"), shared_key=("key_a", "sum")).reset_index()
    for s in ("a", "b"):
        pr = pr.merge(fi.rename(columns={c: f"{c}_{s}" for c in fi.columns}), on=f"f_{s}")
    pr["exact_dup"] = pr.md5_a == pr.md5_b
    pr["frac_of_smaller"] = pr.shared / np.minimum(pr.n_pk_a, pr.n_pk_b)
    pr["same_class"] = pr.cls_a == pr.cls_b
    pr["id_diff"] = np.where(pr.same_class, (pr.clip_id_b - pr.clip_id_a).abs(), np.nan)
    pr["splits"] = [",".join(sorted([a, b])) for a, b in zip(pr.split_a.fillna("NA"), pr.split_b.fillna("NA"))]
    pr.drop(columns=[c for c in pr.columns if c.startswith("md5_")]).to_csv(
        OUT / "raw_audit_shared_packet_pairs.csv", index=False)
    part = pr[~pr.exact_dup].copy()
    w(f"- file pairs sharing >= 1 packet: {len(pr)} ({int(pr.exact_dup.sum())} exact-duplicate pairs, {len(part)} pairs of "
      f"different files)")
    w(f"- different-file pairs, shared packets per pair: " + "; ".join(
        f"{k} {v:.4g}" for k, v in part.shared.describe(percentiles=[0.5, 0.9]).items()))
    w(f"- different-file pairs where every packet (>= {MIN_PKT} B) of the smaller file occurs in the other: "
      f"{int((part.frac_of_smaller >= 1).sum())}; >= 50%: {int((part.frac_of_smaller >= 0.5).sum())}")
    w(f"- pairs across classes: {int((~pr.same_class).sum())}")
    w(f"- same-class different-file pairs by ID difference: {vc(part.id_diff.dropna().astype(int), 15)}")
    w(f"- different-file pairs by split combination: {vc(part.splits)}")
    linked = pd.Index(pd.concat([pr.f_a, pr.f_b]).unique())
    plinked = pd.Index(pd.concat([part.f_a, part.f_b]).unique())
    w(f"- files sharing >= 1 packet with another file: {len(linked)} of {len(fi)} ({len(plinked)} with a non-identical "
      f"file); per class: " + "; ".join(
        f"{c} {int(fi[fi.f.isin(linked) & (fi.cls == c)].shape[0])}" for c in CLS_ORDER))
    n = len(fi)
    A = sp.coo_matrix((np.ones(len(pr)), (pr.f_a.values, pr.f_b.values)), shape=(n, n))
    _, labc = connected_components(A, directed=False)
    fi["comp"] = labc
    sizes = fi.comp.value_counts()
    fi["comp_size"] = fi.comp.map(sizes)
    multi = sizes[sizes >= 2]
    w(f"- connected components with >= 2 files: {len(multi)}, holding {int(multi.sum())} files; sizes: {vc(multi, 12)}")
    test_f = fi[fi.split == "test"].f
    train_h = set(P[P.f.isin(set(fi[fi.split == "train"].f))].hash)
    Pt = P[P.f.isin(set(test_f))]
    hit = Pt.assign(in_train=Pt.hash.isin(train_h)).groupby("f").in_train.agg(["sum", "size"])
    hit["frac"] = hit["sum"] / hit["size"]
    hit = hit.join(fi.set_index("f")[["relpath", "cls", "md5"]])
    tr_md5 = set(fi[fi.split == "train"].md5)
    any_ = hit["sum"] > 0
    nt = int((df.split == "test").sum())
    w(f"- **test files containing >= 1 packet that also occurs in a train file: {int(any_.sum())} of {nt} "
      f"({any_.sum() / nt:.2%})**; of these, byte-identical to a train file: {int((any_ & hit.md5.isin(tr_md5)).sum())}; "
      f"by class: {vc(hit[any_].cls)}")
    w(f"- for those test files, fraction of their packets found in train: " + "; ".join(
        f"{k} {v:.3g}" for k, v in hit.frac[any_].describe(percentiles=[0.25, 0.5, 0.75]).items()))
    w(f"- test files with >= 50% of their packets in train: {int((hit.frac >= 0.5).sum())}; 100%: "
      f"{int((hit.frac >= 1).sum())}; share of all test packets (>= {MIN_PKT} B) that occur in train: "
      f"{hit['sum'].sum() / hit['size'].sum():.4f}")
    w()
    # robustness of the test-in-train count
    dupmd5 = set(df.md5[df.md5.duplicated(keep=False)])
    Pall = pd.DataFrame({"f": z["fidx"], "size": z["size"], "key": z["key"], "hash": z["hash"]})
    subsets = {"all test files": fi[fi.split == "test"],
               "test files with >= 30 frames": fi[(fi.split == "test") & (fi.real_frames >= 30)],
               "... and not byte-identical to any other file": fi[(fi.split == "test") & (fi.real_frames >= 30)
                                                                   & ~fi.md5.isin(dupmd5)]}
    rows = []
    for mn in (MIN_PKT, 4096, 16384):
        for keyonly in (False, True):
            Q = Pall[Pall["size"] >= mn]
            if keyonly:
                Q = Q[Q.key]
            Q = Q.drop_duplicates(["f", "hash"])
            trh = set(Q[Q.f.isin(set(fi[fi.split == "train"].f))].hash)
            row = {"min_packet_bytes": mn, "keyframes_only": keyonly}
            for name, te in subsets.items():
                Qt = Q[Q.f.isin(set(te.f))]
                g = Qt.assign(t=Qt.hash.isin(trh)).groupby("f").t.agg(["any", "mean"])
                row[name] = f"{int(g['any'].sum())}/{len(te)} ({g['any'].sum() / len(te):.1%})"
                if name.startswith("..."):
                    row["... fully contained (all packets in train)"] = int((g["mean"] >= 1).sum())
            rows.append(row)
    w("Robustness: test files that share >= 1 packet with a train file, by packet-size threshold, keyframes only, "
      "and excluding broken / exact-duplicate files:")
    w()
    table(pd.DataFrame(rows), index=False)
    # pixel check: decode one shared keyframe from both files of 5 random test/train pairs
    K = Pall[Pall.key & (Pall["size"] >= 16384)].merge(fi[["f", "split", "real_frames", "md5"]], on="f")
    K = K[K.real_frames >= 30]
    mm = K[K.split == "test"].merge(K[K.split == "train"], on="hash", suffixes=("_te", "_tr"))
    mm = mm[mm.md5_te != mm.md5_tr].drop_duplicates("f_te")
    sel = mm.iloc[np.random.default_rng(0).choice(len(mm), size=min(5, len(mm)), replace=False)]
    chk = []
    for _, r in sel.iterrows():
        a, b = rel[r.f_te], rel[r.f_tr]
        pa, A = _decode_frame_with_hash(a, int(r.hash))
        pb, B = _decode_frame_with_hash(b, int(r.hash))
        chk.append(dict(test_file=a, pts_in_test_s=pa / TB if pa is not None else None, train_file=b,
                        pts_in_train_s=pb / TB if pb is not None else None,
                        pixel_identical=bool(A is not None and B is not None and np.array_equal(A, B))))
    w("Pixel check: one shared keyframe (>= 16 KB) of 5 random test/train pairs (seed 0) decoded from both files "
      "with PyAV and compared as yuv420p arrays:")
    w()
    table(pd.DataFrame(chk), index=False)
    # overlap patterns among normal clips
    nrm = part[(part.real_frames_a >= 30) & (part.real_frames_b >= 30)].copy()
    dur = df.set_index("relpath").v_duration_s
    nrm["short_s"] = np.minimum(nrm.relpath_a.map(dur), nrm.relpath_b.map(dur))
    nrm["long_s"] = np.maximum(nrm.relpath_a.map(dur), nrm.relpath_b.map(dur))
    nrm["kind"] = np.where(nrm.frac_of_smaller >= 0.99, "contained", "partial")
    bins = [0, 4, 6, 9, 12, 20, 400]
    labs = ["<4", "4-6", "6-9", "9-12", "12-20", ">20"]
    w(f"Overlap patterns among different-file pairs where both files have >= 30 frames ({len(nrm)} pairs; "
      f"`contained` = every packet of the shorter file occurs in the other):")
    w()
    for kind in ("contained", "partial"):
        q = nrm[nrm.kind == kind]
        w(f"- {kind}: {len(q)} pairs; shared packets median {q.shared.median():.0f}; |ID difference| <= 3: "
          f"{(q.id_diff <= 3).mean():.1%}, > 100: {(q.id_diff > 100).mean():.1%}; median |ID difference| {q.id_diff.median():.0f}")
    w()
    w("Contained pairs by duration of the shorter (rows) and longer (columns) file, seconds:")
    w()
    q = nrm[nrm.kind == "contained"]
    table(pd.crosstab(pd.cut(q.short_s, bins, labels=labs), pd.cut(q.long_s, bins, labels=labs)))
    w("Partial-overlap pairs, same layout:")
    w()
    q = nrm[nrm.kind == "partial"]
    table(pd.crosstab(pd.cut(q.short_s, bins, labels=labs), pd.cut(q.long_s, bins, labels=labs)))
    lng = pd.concat([nrm[["relpath_a", "id_diff"]].rename(columns={"relpath_a": "r"}),
                     nrm[["relpath_b", "id_diff"]].rename(columns={"relpath_b": "r"})]).groupby("r").id_diff
    w(f"- files (>= 30 frames) overlapping another such file: {lng.ngroups}; nearest overlapping file within 3 IDs: "
      f"{(lng.min() <= 3).mean():.1%}; all overlapping files within 3 IDs: {(lng.max() <= 3).mean():.1%}; "
      f"overlapping partners per file: median {lng.size().median():.0f}, max {lng.size().max()}")
    iso = fi[fi.comp_size < 2].join(df.set_index("relpath")[["combo"]], on="relpath")
    w(f"- files sharing no packet with any other file: {len(iso)}; by class: {vc(iso.cls)}; by batch combo: {vc(iso.combo, 8)}")
    cc = fi[fi.comp_size >= 2].join(df.set_index("relpath")[["combo"]], on="relpath").groupby("comp").agg(
        cls=("cls", "first"), n=("f", "size"), n_combo=("combo", "nunique"), n_cls=("cls", "nunique"))
    w(f"- components per class: {vc(cc.cls)}; components with more than one batch combo: {int((cc.n_combo > 1).sum())}; "
      f"with more than one class: {int((cc.n_cls > 1).sum())}")
    w()
    nrm.to_csv(OUT / "raw_audit_overlap_pairs_normal.csv", index=False)
    w("Different-file pairs by class pair x split pair:")
    w()
    part["classes"] = [",".join(sorted([a, b])) for a, b in zip(part.cls_a, part.cls_b)]
    table(pd.crosstab(part.classes, part.splits))
    comp_rows = []
    for cid, g in fi[fi.comp_size >= 2].groupby("comp"):
        comp_rows.append(dict(size=len(g), classes=vc(g.cls), ids=f"{int(g.clip_id.min())}-{int(g.clip_id.max())}",
                              splits=vc(g.split), distinct_md5=g.md5.nunique()))
    ct = pd.DataFrame(comp_rows).sort_values("size", ascending=False)
    both = ct.splits.str.contains("train") & ct.splits.str.contains("test")
    w(f"Components with members in both train and test: {int(both.sum())} of {len(ct)}, holding "
      f"{int(ct[both]['size'].sum())} files. Largest 15 components:")
    w()
    table(ct.head(15), index=False)
    df = df.merge(fi[["relpath", "comp", "comp_size"]], on="relpath", how="left")
    df["test_frac_packets_in_train"] = df.relpath.map(hit.set_index("relpath").frac)
    return df


def _decode_frame_with_hash(relpath: str, h: int):
    """Decode a file until the frame whose packet has blake2b-64 hash h; return (pts, yuv420p ndarray)."""
    import hashlib

    import av

    with av.open(str(ROOT / "data/ouc-cge/videos" / relpath), mode="r") as c:
        v = c.streams.video[0]
        v.codec_context.thread_count = 1
        target = None
        for pkt in c.demux(v):
            if pkt.size and target is None:
                if int.from_bytes(hashlib.blake2b(bytes(pkt), digest_size=8).digest(), "little") == h:
                    target = pkt.pts
            for fr in pkt.decode():
                if target is not None and fr.pts == target:
                    return fr.pts, fr.to_ndarray()
    return None, None


# ----------------------------------------------------------------------------------------------
def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
                         "legend.labelcolor": INK, "text.color": INK})
    return plt


def figures(df: pd.DataFrame) -> None:
    plt = setup_mpl()
    FIG.mkdir(parents=True, exist_ok=True)
    h = df[df.readable]

    # 1. durations
    d = h.v_duration_s
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.4, 1]})
    bins = np.arange(0, 30.01, 0.5)
    ax.hist(np.minimum(d, 30.2), bins=np.append(bins, 30.6), color=CAT[0], edgecolor=SURF, linewidth=0.6)
    ax.axvspan(9.5, 10.5, color=GRID, zorder=0)
    ax.text(10.7, 0.97, "shaded: 9.5-10.5 s", color=INK2, fontsize=9, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlabel("video stream duration (s, header); last bin = over 30 s")
    ax.set_ylabel("clips")
    ax.set_title(f"Raw clip durations (n = {len(d)} readable files)", loc="left")
    lb = np.logspace(np.log10(d.min() * 0.9), np.log10(d.max() * 1.1), 50)
    ax2.hist(d, bins=lb, color=CAT[0], edgecolor=SURF, linewidth=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("video stream duration (s), log")
    ax2.set_title("same, log-log", loc="left")
    for a_ in (ax, ax2):
        a_.grid(axis="y", color=GRID, linewidth=0.6)
        a_.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "raw_duration_hist.png")
    plt.close(fig)

    # 2. first-pts histogram, stacked by class
    fp = h.first_pts_s
    pos = h[fp > 0]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    lb = np.logspace(np.log10(pos.first_pts_s.min() * 0.9), np.log10(pos.first_pts_s.max() * 1.1), 60)
    ax.hist([pos.first_pts_s[pos.cls == c] for c in CLS_ORDER], bins=lb, stacked=True,
            color=[CLS_COL[c] for c in CLS_ORDER], label=CLS_ORDER, edgecolor=SURF, linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("first video packet pts (s), log")
    ax.set_ylabel("clips (log)")
    ax.set_title(f"First PTS of raw clips: {int((fp == 0).sum())} start at exactly 0 (not drawn), {len(pos)} later",
                 loc="left")
    for x, t in [(0.1, "0.1 s"), (1, "1 s"), (10, "10 s"), (100, "100 s")]:
        ax.axvline(x, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(x * 1.08, 0.97, t, color=INK2, fontsize=9, va="top", transform=ax.get_xaxis_transform())
    ax.legend(frameon=False, title="class", loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "raw_first_pts_hist.png")
    plt.close(fig)

    # 3. first pts vs id + property strips
    strips = [("v_codec", "codec"), ("resolution", "WxH"), ("n_audio", "audio streams")]
    cats = {}
    for col, _ in strips:
        vals = h[col].astype(str).value_counts().index.tolist()
        cats[col] = {v: CAT[i] for i, v in enumerate(vals[:8])}
    fig, axes = plt.subplots(4, 3, figsize=(14, 6.8), gridspec_kw={"height_ratios": [5, 0.6, 0.6, 0.6]}, sharex="col")
    for k, c in enumerate(CLS_ORDER):
        s = df[df.cls == c].sort_values("clip_id")
        sr = s[s.readable]
        ax = axes[0, k]
        ax.scatter(sr.clip_id, sr.first_pts_s.clip(lower=1e-3), s=5, color=CLS_COL[c], linewidths=0)
        ub = s[~s.readable]
        if len(ub):
            ax.scatter(ub.clip_id, np.full(len(ub), 1e-3), marker="x", s=25, color=INK, linewidths=1, label="unreadable")
            ax.legend(frameon=False, loc="upper left", fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"{c} (n = {len(s)})", loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        if k == 0:
            ax.set_ylabel("first pts (s), log; 0 drawn at 1 ms")
        for j, (col, name) in enumerate(strips, start=1):
            a2 = axes[j, k]
            for v, colr in cats[col].items():
                m = sr[col].astype(str) == v
                if m.any():
                    a2.bar(sr.clip_id[m], 1, width=1.0, color=colr, linewidth=0)
            a2.set_yticks([])
            a2.set_ylim(0, 1)
            a2.spines["left"].set_visible(False)
            if k == 0:
                a2.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        axes[-1, k].set_xlabel("clip ID within class")
    handles, labels = [], []
    for col, name in strips:
        for v, colr in cats[col].items():
            handles.append(plt.Rectangle((0, 0), 1, 1, color=colr))
            labels.append(f"{name}: {v}")
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG / "raw_first_pts_vs_id.png")
    plt.close(fig)

    # 4. raw vs 320p excess
    if "p320_duration_s" in df:
        jj = h[h.p320_duration_s.notna()]
        dd = jj.p320_duration_s - jj.v_duration_s
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = jj.first_pts_s.clip(lower=1e-3)
        yv = dd.clip(lower=1e-3)
        for c in CLS_ORDER:
            m = jj.cls == c
            ax.scatter(x[m], yv[m], s=9, color=CLS_COL[c], label=c, linewidths=0, alpha=0.85)
        lim = [1e-3, max(x.max(), yv.max()) * 1.5]
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(30, 60, "y = x", color=INK2, fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("raw first pts (s), log; <= 1 ms drawn at 1 ms")
        ax.set_ylabel("320p duration - raw duration (s), log; <= 1 ms at 1 ms")
        ax.set_title("320p copies are longer by the raw first PTS", loc="left")
        ax.legend(frameon=False, title="class", loc="upper left")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(FIG / "raw_vs_320p_excess.png")
        plt.close(fig)

    # 5. broken families: audio length vs clip id
    b = h[h.full_decoded.fillna(False).astype(bool) & (h.v_duration_s < 1)].copy()
    if len(b):
        b["family"] = b.cls + ", " + b.v_frames.astype(int).astype(str) + "-frame clips"
        fams = b.family.value_counts().index.tolist()
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for i, f in enumerate(fams):
            g = b[b.family == f].sort_values("clip_id")
            ax.plot(g.clip_id, g.a0_duration_s, marker="o", markersize=3.5, linewidth=1.6, color=CAT[i],
                    label=f"{f}: IDs {int(g.clip_id.min())}-{int(g.clip_id.max())} (n = {len(g)})")
        ax.set_xlabel("clip ID within class")
        ax.set_ylabel("audio[0] duration (s)")
        ax.set_title("Clips with < 1 s of video carry 28-214 s of audio", loc="left")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(FIG / "raw_broken_audio.png")
        plt.close(fig)
    # 6. overlapping pairs: ID vs ID per class
    op = OUT / "raw_audit_overlap_pairs_normal.csv"
    if op.exists():
        nrm = pd.read_csv(op)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
        for k, c in enumerate(CLS_ORDER):
            ax = axes[k]
            q = nrm[nrm.cls_a == c]
            for kind, colr in (("partial", CAT[1]), ("contained", CAT[0])):
                m = q.kind == kind
                lo = np.minimum(q.clip_id_a[m], q.clip_id_b[m])
                hi = np.maximum(q.clip_id_a[m], q.clip_id_b[m])
                ax.scatter(lo, hi, s=3, color=colr, linewidths=0, label=f"{kind} ({int(m.sum())})")
            ax.set_title(f"{c}: {len(q)} overlapping pairs", loc="left")
            ax.set_xlabel("smaller clip ID")
            if k == 0:
                ax.set_ylabel("larger clip ID")
            ax.grid(color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8, loc="lower right", markerscale=3)
        fig.suptitle("Pairs of clips (>= 30 frames) sharing byte-identical video packets", x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(FIG / "raw_shared_packet_pairs.png")
        plt.close(fig)
    print(f"figures in {FIG}")


def main() -> int:
    global OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--fig-dir", default=str(FIG))
    a = ap.parse_args()
    OUT, FIG = Path(a.out_dir), Path(a.fig_dir)
    df, man = load()
    assert (df[df.readable].v_time_base == f"1/{TB}").all()
    sec0(df, man)
    sec_a(df)
    sec_b(df)
    sec_c(df)
    sec_d(df)
    df = sec_e(df)
    df2 = sec_f(df)
    if df2 is not None:
        df = df2
    df.to_csv(OUT / "raw_audit_joined.csv", index=False)
    (OUT / "raw_audit_tables.md").write_text("\n".join(md) + "\n")
    print(f"wrote {OUT / 'raw_audit_tables.md'} ({len(md)} lines) and raw_audit_joined.csv")
    figures(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
