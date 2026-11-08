#!/usr/bin/env python
"""Summary statistics, duplicate check and duration histogram for outputs/research/clip_audit.csv.

Also estimates the RAM that scripts/review/01_frame_cache.py would need on these clips: its decode()
keeps every decoded BGR frame of a clip in a Python list (n_frames x height x width x 3 bytes), and
Pool(workers) runs `workers` such clips at once (imap_unordered, chunksize 4, manifest order).

Usage (repository root):
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/01b_clip_audit_summary.py \
      --audit outputs/research/clip_audit.csv --raw-dir-manifest outputs/research/raw_videos_dvc_dir.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABEL_NAME = {0: "low", 1: "medium", 2: "high"}
FOLDER_LABEL = {"low": 0, "mid": 1, "high": 2}


def simulate_frame_cache_peak(df_manifest_order: pd.DataFrame, workers=8, chunksize=4,
                              sec_per_frame=1.0 / 1000, overhead_s=0.05):
    """Discrete-event simulation of Pool(workers).imap_unordered(decode, jobs, chunksize).
    Memory of a clip grows linearly while it is decoded and is released when decode returns
    (optimistic: assumes the allocator gives the memory back). Returns peak bytes, time series."""
    mem = df_manifest_order["frame_bytes"].values.astype(float)
    dur = overhead_s + df_manifest_order["n_frames"].values * sec_per_frame
    chunks = [list(range(i, min(i + chunksize, len(mem)))) for i in range(0, len(mem), chunksize)]
    # each worker: queue of (start, end, mem) intervals
    free_at = np.zeros(workers)
    intervals = []
    ci = 0
    while ci < len(chunks):
        w = int(np.argmin(free_at))
        t = free_at[w]
        for j in chunks[ci]:
            intervals.append((t, t + dur[j], mem[j]))
            t += dur[j]
        free_at[w] = t
        ci += 1
    iv = np.array(intervals)
    # evaluate memory on a grid of event times (ends of intervals = peaks of each clip)
    ends = iv[:, 1]
    order = np.argsort(ends)
    peak, peak_t = 0.0, 0.0
    starts, stops, m = iv[:, 0], iv[:, 1], iv[:, 2]
    # memory at time t = sum over active clips of m * (t - start) / (stop - start); peaks occur just before ends
    for k in order[::max(1, len(order) // 20000)] if len(order) > 20000 else order:
        t = ends[k] - 1e-9
        act = (starts <= t) & (stops > t)
        cur = float((m[act] * (t - starts[act]) / (stops[act] - starts[act])).sum())
        if cur > peak:
            peak, peak_t = cur, t
    return peak, peak_t, float(free_at.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=Path("outputs/research/clip_audit.csv"))
    ap.add_argument("--raw-dir-manifest", type=Path, default=Path("outputs/research/raw_videos_dvc_dir.json"))
    ap.add_argument("--dvc320-dir-manifest", type=Path,
                    default=Path(".dvc/cache/files/md5/f9/476986840e39fdcd1f43b266ad3d4c.dir"))
    ap.add_argument("--fig", type=Path, default=Path("docs/research/figures/clip_duration_hist.png"))
    ap.add_argument("--out-json", type=Path, default=Path("outputs/research/clip_audit_summary.json"))
    ap.add_argument("--out-dups", type=Path, default=Path("outputs/research/clip_audit_duplicates.csv"))
    args = ap.parse_args()

    a = pd.read_csv(args.audit, dtype={"md5": str})
    a["label"] = a["label"].astype("Int64")
    S = {}
    S["records"] = int(len(a))
    S["records_without_label"] = a.loc[a.label.isna(), ["path", "split"]].to_dict("records")
    lab = a[a.label.notna()].copy()
    S["labelled_records"] = int(len(lab))
    S["unique_labelled_paths"] = int(lab.path.nunique())
    S["paths_in_more_than_one_manifest_row"] = lab.path[lab.path.duplicated(keep=False)].tolist()
    lab["folder"] = lab.path.str.extract(r"videos/(\w+)/")[0]
    lab["label_s"] = lab.label.astype(int).astype(str)
    S["label_vs_folder_mismatch"] = int((lab.folder.map(FOLDER_LABEL) != lab.label).sum())
    ct = pd.crosstab(lab.split, lab.label.map(LABEL_NAME), margins=True)
    S["counts_split_by_class"] = ct.to_dict()
    S["exists"] = int(lab.exists.sum())
    S["readable"] = int(lab.readable.sum())
    bad = lab[~lab.readable]
    S["unreadable"] = bad[["path", "label", "split", "exists", "file_bytes", "cv2_error", "av_error",
                           "av_stream_frames", "container_duration_s"]].astype(str).to_dict("records")
    r = lab[lab.readable].copy()
    S["readable_by_split"] = r.split.value_counts().to_dict()

    d = r.duration_s.values
    pcts = [1, 5, 25, 50, 75, 95, 99]
    S["duration_s_percentiles"] = {f"p{p}": round(float(np.percentile(d, p)), 3) for p in pcts}
    S["duration_s_min"], S["duration_s_max"] = float(d.min()), float(d.max())
    S["duration_s_mean"] = float(d.mean())
    S["n_frames_percentiles"] = {f"p{p}": float(np.percentile(r.n_frames, p)) for p in pcts}
    S["n_frames_min_max"] = [int(r.n_frames.min()), int(r.n_frames.max())]
    S["dur_bins"] = {
        "within_9.5_10.5": int(((d >= 9.5) & (d <= 10.5)).sum()),
        "below_9.5": int((d < 9.5).sum()),
        "above_10.5": int((d > 10.5).sum()),
        "above_30": int((d > 30).sum()),
        "above_60": int((d > 60).sum()),
        "below_2.1333(64 frames at 30fps)": int((r.n_frames < 64).sum()),
        "5.0_6.0": int(((d >= 5.0) & (d < 6.0)).sum()),
        "below_1": int((d < 1).sum()),
    }
    S["dur_bins_by_split"] = {s: {"n": int((r.split == s).sum()),
                                  "within_9.5_10.5": int(((g.duration_s >= 9.5) & (g.duration_s <= 10.5)).sum()),
                                  "below_9.5": int((g.duration_s < 9.5).sum()),
                                  "above_10.5": int((g.duration_s > 10.5).sum()),
                                  "above_30": int((g.duration_s > 30).sum()),
                                  "median_s": float(g.duration_s.median()),
                                  "hours": float(g.duration_s.sum() / 3600)}
                              for s, g in r.groupby("split")}
    S["dur_by_class"] = {LABEL_NAME[int(k)]: {"n": int(len(g)), "median_s": float(g.duration_s.median()),
                                              "mean_s": float(g.duration_s.mean()),
                                              "above_30": int((g.duration_s > 30).sum()),
                                              "hours": float(g.duration_s.sum() / 3600)}
                         for k, g in r.groupby("label")}
    tot = float(d.sum())
    S["total_hours_readable"] = tot / 3600
    S["total_hours_readable_clips_capped_at_30s"] = float(np.minimum(d, 30).sum() / 3600)
    S["total_hours_if_all_7705_were_10s"] = 7705 * 10 / 3600
    S["paper_12h50_in_s"] = 12 * 3600 + 50 * 60
    S["paper_12h50_mean_s_per_clip_7705"] = (12 * 3600 + 50 * 60) / 7705
    S["top_n_frames_values"] = r.n_frames.value_counts().head(15).to_dict()
    S["longest_20"] = r.sort_values("duration_s", ascending=False).head(20)[
        ["path", "label", "split", "n_frames", "duration_s", "file_bytes"]].to_dict("records")
    S["shortest_20"] = r.sort_values("duration_s").head(20)[
        ["path", "label", "split", "n_frames", "duration_s", "file_bytes"]].to_dict("records")

    S["fps_counts"] = {str(k): int(v) for k, v in r.fps.value_counts().items()}
    S["av_avg_rate_counts"] = {str(k): int(v) for k, v in r.av_avg_rate.value_counts().items()}
    S["resolution_counts"] = {f"{int(w)}x{int(h)}": int(v) for (w, h), v in
                              r.groupby(["width", "height"]).size().sort_values(ascending=False).items()}
    S["video_codec_counts"] = r.video_codec.value_counts().to_dict()
    S["pix_fmt_counts"] = r.pix_fmt.value_counts().to_dict()
    S["video_encoder_counts"] = r.video_encoder.fillna("").value_counts().to_dict()
    S["container_encoder_counts"] = r.container_encoder.fillna("").value_counts().to_dict()
    S["has_audio_counts"] = {str(k): int(v) for k, v in lab.has_audio.value_counts(dropna=False).items()}
    S["n_streams_counts"] = {str(k): int(v) for k, v in lab.n_streams.value_counts(dropna=False).items()}
    S["audio_codec_counts"] = lab.audio_codec.fillna("").value_counts().to_dict()
    # consistency of frame counts / durations
    S["n_reported_ne_n_frames"] = int((r.n_reported != r.n_frames).sum())
    S["av_packets_ne_n_frames"] = int((r.av_n_video_packets != r.n_frames).sum())
    S["av_stream_frames_ne_n_frames"] = int((r.av_stream_frames != r.n_frames).sum())
    S["container_duration_minus_n_over_fps_absmax_s"] = float((r.container_duration_s - r.duration_s).abs().max())
    S["first_pts_nonzero"] = int((r.av_first_pts_s.abs() > 1e-9).sum())
    S["file_bytes_total_GB"] = float(lab.file_bytes.sum() / 1e9)
    S["bytes_per_second_percentiles"] = {f"p{p}": float(np.percentile(r.file_bytes / r.duration_s, p)) for p in (1, 50, 99)}

    # exact duplicate files (md5 of the 320p bytes)
    g = r.groupby("md5")
    sizes = g.path.transform("size")
    dup = r[sizes > 1].copy()
    dup["group_size"] = sizes[sizes > 1]
    agg = dup.groupby("md5").agg(n=("path", "size"), labels=("label_s", lambda x: ",".join(sorted(set(x)))),
                                 splits=("split", lambda x: ",".join(sorted(set(x)))),
                                 n_frames=("n_frames", "first"), file_bytes=("file_bytes", "first"),
                                 paths=("path", lambda x: " ".join(sorted(x))))
    S["dup320_groups"] = int(len(agg))
    S["dup320_files_in_groups"] = int(agg.n.sum())
    S["dup320_redundant_copies"] = int(agg.n.sum() - len(agg))
    S["dup320_groups_mixed_label"] = int((agg.labels.str.contains(",")).sum())
    S["dup320_files_in_mixed_label_groups"] = int(agg.loc[agg.labels.str.contains(","), "n"].sum())
    S["dup320_groups_spanning_splits"] = int((agg.splits.str.contains(",")).sum())
    S["dup320_groups_train_and_test"] = int(agg.splits.apply(lambda s: "train" in s and "test" in s).sum())
    S["dup320_test_clips_with_exact_copy_in_train"] = int(sum(
        ((grp.split == "test").sum() if (grp.split == "train").any() else 0) for _, grp in dup.groupby("md5")))
    S["dup320_group_size_counts"] = agg.n.value_counts().sort_index().to_dict()
    S["dup320_n_frames_of_dups_percentiles"] = {f"p{p}": float(np.percentile(dup.n_frames, p)) for p in (0, 25, 50, 75, 100)}

    # local file md5 against the 320p DVC directory manifest
    if args.dvc320_dir_manifest.exists():
        man = pd.DataFrame(json.loads(args.dvc320_dir_manifest.read_text()))
        man["path"] = man.relpath
        m = lab[lab.exists].merge(man[["path", "md5"]], on="path", how="left", suffixes=("", "_dvc"))
        S["md5_vs_dvc320_manifest"] = {"compared": int(m.md5_dvc.notna().sum()),
                                       "missing_in_manifest": int(m.md5_dvc.isna().sum()),
                                       "mismatch": int((m.md5_dvc.notna() & (m.md5 != m.md5_dvc)).sum())}
    # duplicates among the raw videos (DVC directory manifest of data/ouc-cge/videos, md5 of raw bytes)
    if args.raw_dir_manifest.exists():
        raw = pd.DataFrame(json.loads(args.raw_dir_manifest.read_text()))
        raw["path"] = "videos/" + raw.relpath
        S["raw_manifest_entries"] = int(len(raw))
        S["raw_non_mp4_entries"] = raw.loc[~raw.relpath.str.endswith(".mp4"), "relpath"].tolist()
        rawv = raw[raw.relpath.str.endswith(".mp4")]
        S["raw_mp4_by_folder"] = rawv.relpath.str.split("/").str[0].value_counts().to_dict()
        S["raw_mp4_not_in_manifests"] = sorted(set(rawv.path) - set(lab.path))
        S["manifest_paths_not_in_raw"] = sorted(set(lab.path) - set(rawv.path))
        j = lab.merge(rawv[["path", "md5"]].rename(columns={"md5": "raw_md5"}), on="path", how="left")
        rg = j.groupby("raw_md5")
        rs = rg.path.transform("size")
        rd = j[rs > 1]
        ragg = rd.groupby("raw_md5").agg(n=("path", "size"), labels=("label_s", lambda x: ",".join(sorted(set(x)))),
                                         splits=("split", lambda x: ",".join(sorted(set(x)))))
        S["dupraw_groups"] = int(len(ragg))
        S["dupraw_files_in_groups"] = int(ragg.n.sum())
        S["dupraw_redundant_copies"] = int(ragg.n.sum() - len(ragg))
        S["dupraw_groups_mixed_label"] = int(ragg.labels.str.contains(",").sum())
        S["dupraw_files_in_mixed_label_groups"] = int(ragg.loc[ragg.labels.str.contains(","), "n"].sum())
        S["dupraw_groups_spanning_splits"] = int(ragg.splits.str.contains(",").sum())
        S["dupraw_test_clips_with_exact_copy_in_train"] = int(sum(
            ((grp.split == "test").sum() if (grp.split == "train").any() else 0) for _, grp in rd.groupby("raw_md5")))
        S["dupraw_group_size_counts"] = ragg.n.value_counts().sort_index().to_dict()
        # are raw duplicates also 320p duplicates, and vice versa
        j2 = j[j.readable]
        same_raw = j2.groupby("raw_md5").md5.nunique()
        S["raw_dup_groups_whose_320p_copies_differ"] = int((same_raw[same_raw.index.isin(ragg.index)] > 1).sum())
        same_320 = j2.groupby("md5").raw_md5.nunique()
        S["dup320_groups_from_distinct_raw_files"] = int((same_320[same_320.index.isin(agg.index)] > 1).sum())
        dup = dup.merge(j[["path", "raw_md5"]], on="path", how="left")
    dup.sort_values(["group_size", "md5", "path"], ascending=[False, True, True])[
        ["md5", "group_size", "path", "label", "split", "n_frames", "duration_s", "file_bytes"] +
        (["raw_md5"] if "raw_md5" in dup else [])].to_csv(args.out_dups, index=False)

    # RAM of scripts/review/01_frame_cache.py decode(): all frames of one clip in memory
    r["frame_bytes"] = r.n_frames * r.width * r.height * 3
    S["frame_cache_clip_bytes_max_GB"] = float(r.frame_bytes.max() / 1e9)
    S["frame_cache_clip_bytes_percentiles_MB"] = {f"p{p}": float(np.percentile(r.frame_bytes, p) / 1e6) for p in (50, 95, 99, 100)}
    S["frame_cache_clips_over_1GB"] = int((r.frame_bytes > 1e9).sum())
    S["frame_cache_clips_over_2GB"] = int((r.frame_bytes > 2e9).sum())
    S["frame_cache_clips_over_4GB"] = int((r.frame_bytes > 4e9).sum())
    S["frame_cache_top8_sum_GB"] = float(r.frame_bytes.nlargest(8).sum() / 1e9)
    # the manifest order used by 01_frame_cache.py (common.load_all_manifests: train, val, test, malformed row dropped)
    order = lab.copy()
    order = order.merge(r[["path", "split", "frame_bytes"]], on=["path", "split"], how="left")
    order["frame_bytes"] = order.frame_bytes.fillna(0)
    order["n_frames"] = order.n_frames.fillna(0)
    peak, peak_t, makespan = simulate_frame_cache_peak(order)
    S["frame_cache_simulated_peak_GB_8workers"] = peak / 1e9
    S["frame_cache_simulation_note"] = ("Pool(8), chunksize 4, manifest order, decode time ~ n_frames, memory of a clip "
                                        "released when its decode returns; optimistic (assumes the allocator returns memory)")
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(S, indent=2, default=str))
    print(json.dumps(S, indent=1, default=str)[:20000])

    # histogram figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    surf, ink, ink2, blue, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#e4e3df"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2,
                         "ytick.color": ink2, "axes.titlecolor": ink, "figure.facecolor": surf, "axes.facecolor": surf})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ax = axes[0]
    bins = np.arange(0, 30.0001, 1 / 3)
    ax.hist(np.minimum(d, 30.2), bins=np.append(bins, 30.6), color=blue, edgecolor=surf, linewidth=0.6)
    ax.axvspan(9.5, 10.5, color=grid, zorder=0)
    ax.text(10.7, 0.97, "shaded: 9.5-10.5 s", color=ink2, fontsize=9, va="top", transform=ax.get_xaxis_transform())
    ax.set_xlim(0, 30.7)
    ax.set_xlabel("clip duration (s), n_frames / fps; last bar = clips > 30 s")
    ax.set_ylabel("clips")
    ax.set_title(f"OUC-CGE 320p, {len(d)} readable clips, 1/3-s bins", loc="left", fontsize=10)
    ax2 = axes[1]
    lb = np.logspace(np.log10(d.min() * 0.9), np.log10(d.max() * 1.1), 60)
    ax2.hist(d, bins=lb, color=blue, edgecolor=surf, linewidth=0.6)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    for x, t in ((10, "10 s"), (30, "30 s")):
        ax2.axvline(x, color=ink2, linewidth=1, linestyle=(0, (3, 3)))
        ax2.text(x * 1.06, 0.97, t, color=ink2, fontsize=9, va="top", transform=ax2.get_xaxis_transform())
    ax2.set_xlabel("clip duration (s), log scale")
    ax2.set_ylabel("clips (log scale)")
    ax2.set_title(f"full range {d.min():.2f}-{d.max():.1f} s, log-spaced bins", loc="left", fontsize=10)
    for a_ in axes:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(axis="y", color=grid, linewidth=0.6)
        a_.set_axisbelow(True)
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.fig, dpi=130, facecolor=surf)
    print("figure ->", args.fig)


if __name__ == "__main__":
    sys.exit(main())
