#!/usr/bin/env python
"""Markdown tables for 06_temporal_structure.md from the outputs of 04b-04h (no new measurements
except simple aggregations). Writes outputs/research/temporal/report_tables.md and
partner_tests_by_run_typed.csv.

Run types (multi-stream runs of the frame segmentation, |period| >= 2):
  recut   - the columns share video packets (>= 5 clip pairs across two columns, 04f): the same
            camera's footage cut twice / with overlapping windows, interleaved by ID;
  cameras - columns share no footage; "audio-synchronous" if at least one column pair passes the
            audio test of 04f.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04i_report_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
T = REPO / "outputs" / "research" / "temporal"


def md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else (f"{v:.{floatfmt}g}" if abs(v) < 1000 else f"{v:.0f}"))
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values) + "\n"


def main():
    out = []
    C = pd.read_csv(T / "clip_temporal.csv")
    P = pd.read_csv(T / "partner_tests_by_run.csv")
    A = pd.read_csv(T / "audio_columns.csv")
    k = A.groupby(["cls", "id_start"]).agg(max_cross_pk=("n_cross_pk", "max"),
                                           sync_diff=("kind", lambda s: (s == "sync_different_camera").any()),
                                           best_audio=("score", "max")).reset_index()
    P = P.merge(k, on=["cls", "id_start"], how="left")
    P["type"] = np.where(P.max_cross_pk >= 5, "recut", np.where(P.sync_diff, "cameras, audio-sync", "cameras, no audio sync"))
    P.loc[P.max_cross_pk.isna(), "type"] = "short run (<15 clips)"
    P.to_csv(T / "partner_tests_by_run_typed.csv", index=False)
    cols = ["partner_share_packets", "partner_same_c27", "partner_same_c17", "partner_same_raw_frames",
            "partner_same_first_pts", "partner_ff_rmse", "same_source_share_packets", "same_source_same_c27",
            "same_source_same_raw_frames"]
    agg = []
    for t, g in P.groupby("type"):
        w = g.n_partner_pairs
        agg.append(dict(run_type=t, runs=len(g), clips=int(g.n_clips.sum()), partner_pairs=int(w.sum()),
                        **{c: float(np.average(g[c].fillna(0), weights=w)) for c in cols}))
    out.append("## Partner tests by run type (pair-weighted means)\n\n" + md(pd.DataFrame(agg)))
    pt = pd.read_csv(T / "partner_tests.csv", index_col=0)
    out.append("## Pair statistics (04d, primary cut)\n\n" + md(pt.reset_index().rename(columns={"index": "pairs"})[
        ["pairs", "n", "same_c27", "same_c17", "share_packets", "same_raw_frames", "same_320p_n_frames",
         "same_first_pts_1ms", "same_combo", "first_first_rmse_median"]]))
    # per-column metadata
    RT = pd.read_csv(T / "runs_frames.csv")
    rows = []
    for r in RT[(RT.period.abs() >= 2) & (RT.n_clips >= 15)].itertuples():
        g = C[(C.cls == r.cls) & C.vid.between(r.id_start, r.id_end)].copy()
        p = abs(r.period)
        g["col"] = (g.vid - r.id_start) % p
        desc = []
        for cidx, x in g.groupby("col"):
            cb = x.combo.fillna("NA").value_counts()
            cl = x.c27.value_counts()
            desc.append(f"{cidx}: c{cl.index[0]} ({cl.iloc[0] / len(x):.0%}), {cb.index[0].split('|')[0]} "
                        f"{cb.index[0].split('|')[2]}, {x.real_frames.median():.0f} fr")
        t = P[(P.cls == r.cls) & (P.id_start == r.id_start)].type
        rows.append(dict(cls=r.cls, ids=f"{r.id_start}-{r.id_end}", period=r.period, clips=r.n_clips,
                         type=t.iloc[0] if len(t) else "", columns="; ".join(desc)))
    out.append("## Columns of the multi-stream runs (>= 15 clips): dominant cluster (share), codec, audio streams, median raw frames\n\n" + md(pd.DataFrame(rows)))
    sync = A[A.sync].copy()
    out.append("## Audio-synchronous column pairs\n\n" + md(sync[["cls", "id_start", "id_end", "period", "col_a", "col_b", "score", "lag_s", "n_cross_pk", "kind"]]))
    # runs table abridged
    rt = RT[RT.n_clips >= 50][["cls", "id_start", "id_end", "period", "n_clips", "frac_cont", "frac_lab_eq_period",
                               "frac_pk_lab_eq_period", "clusters27", "labels", "splits", "combo", "n_raw_batches",
                               "frac_first_pts_ge_1s", "median_stride_frames", "median_overlap_frames"]]
    out.append("## Runs of constant majority period, >= 50 clips (primary cut)\n\n" + md(rt))
    off = pd.read_csv(T / "packet_offsets_by_iddiff.csv")
    out.append("## Packet offsets by ID difference\n\n" + md(off))
    offp = pd.read_csv(T / "packet_offsets_by_period.csv")
    offp = offp[offp.n_pairs >= 20]
    out.append("## Packet offsets by run period (pairs inside one frame run, >= 20 pairs)\n\n" + md(offp))
    # re-encoded duplicates: nearest train clip near-identical but no shared packet
    NN = pd.read_csv(T / "leak_nn_by_split.csv")
    NN = NN[NN.split_def == "published"].copy()
    R = pd.read_csv(T / "packet_pairs_offsets.csv")
    pk = {(a, b) for a, b in zip(R.rel_a, R.rel_b)} | {(b, a) for a, b in zip(R.rel_a, R.rel_b)}
    cmb = dict(zip(C.rel, C.combo))
    NN["ta"] = NN.test_path.str.replace("videos/", "", regex=False)
    NN["tb"] = NN.nn_train_path.str.replace("videos/", "", regex=False)
    NN["pk"] = [(a, b) in pk for a, b in zip(NN.ta, NN.tb)]
    rr = []
    for t in (2, 5, 10):
        q = NN[NN.d <= t]
        rr.append(dict(nn_d_le=t, test_clips=len(q), nn_shares_packets=int(q.pk.sum()), nn_no_shared_packet=int((~q.pk).sum()),
                       of_which_different_codec_batch=int(((~q.pk) & (q.ta.map(cmb) != q.tb.map(cmb))).sum())))
    out.append("## Test clips whose nearest train clip (32x32) is near-identical, with and without shared packets\n\n" + md(pd.DataFrame(rr)))
    CH = pd.read_csv(T / "chains.csv")

    def wpct(v, w, q):
        o = np.argsort(v)
        cw = np.cumsum(w[o]) / w.sum()
        return float(v[o][np.searchsorted(cw, q / 100)])
    rows = []
    for col in ("n_clips", "span_s", "dur_sum_s"):
        v, w = CH[col].values.astype(float), CH.n_clips.values.astype(float)
        rows.append(dict(quantity=col, **{f"clip_weighted_p{q}": round(wpct(v, w, q), 1) for q in (10, 25, 50, 75, 90)}))
    out.append("## Chain length seen from a random clip (clip-weighted percentiles, primary cut, singletons included)\n\n" + md(pd.DataFrame(rows)))
    sw = pd.read_csv(T / "pr_sweep.csv")
    out.append("## Continuation cut sweep against packet truth (clips in the packet graph)\n\n" + md(sw))
    (T / "report_tables.md").write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
