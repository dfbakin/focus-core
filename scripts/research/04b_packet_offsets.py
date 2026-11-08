#!/usr/bin/env python
"""Exact temporal offsets between raw files that share byte-identical video packets.

Input: outputs/research/raw_video_packets.npz from raw_header_audit.py --packets (one row per
video packet of every readable raw file, in demux order per file: fidx, pts, dts, size, key,
64-bit blake2b hash). Same filter as raw_header_audit_tables.py section f: packets < 1024 bytes are
ignored, and repeated hashes inside one file keep their first occurrence.

For every pair of different files (a, b) of the same class that share >= 1 packet, oriented so that
id_a < id_b (or by relpath for exact duplicates with the same ID; never happens), and every shared
packet p:
  * demux offset   pos_a(p) - pos_b(p), pos = index of the packet in the file's demux order;
  * pts-rank offset rank_a(p) - rank_b(p), rank = index in presentation (pts) order.
The pair offset is the mode over shared packets; *_frac is the share of shared packets at the mode
(1.0 = a single consistent frame offset, as expected for stream-copied cuts of one source).
With o = mode pts-rank offset, b's first frame is a's frame o, so b starts o frames (o / 30 s) after
a (o < 0: b starts before a). Overlap and the relation of the two files follow from o and the
packet counts n_a, n_b (all packets, in frames):
  contained_b_in_a: 0 <= o and o + n_b <= n_a;  contained_a_in_b: o <= 0 and n_a <= o + n_b;
  identical: both;  partial: otherwise.

Output: outputs/research/temporal/packet_pairs_offsets.csv (one row per pair) and a printed summary.
Usage: CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/04b_packet_offsets.py
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "temporal"
MIN_PKT = 1024


def mode_and_frac(g):
    vc = g.value_counts()
    return vc.index[0], vc.iloc[0] / len(g)


def main():
    t0 = time.time()
    z = np.load(REPO / "outputs" / "research" / "raw_video_packets.npz")
    rel = z["relpath"]
    fidx = z["fidx"]
    pts = z["pts"]
    size = z["size"]
    key = z["key"]
    h = z["hash"]
    n = len(fidx)
    # position in demux order: rows of one file are contiguous and in demux order
    starts = np.r_[0, np.flatnonzero(np.diff(fidx)) + 1]
    pos = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    # rank in presentation order within the file
    order = np.lexsort((pts, fidx))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n) - np.repeat(starts, np.diff(np.r_[starts, n]))
    n_pk_all = np.bincount(fidx, minlength=len(rel))
    P = pd.DataFrame({"f": fidx, "pos": pos, "rank": rank, "size": size, "key": key, "hash": h, "pts": pts})
    P = P[P["size"] >= MIN_PKT].drop_duplicates(["f", "hash"])
    S = P[P.groupby("hash").f.transform("size") > 1]
    pairs = S.merge(S, on="hash", suffixes=("_a", "_b"))
    pairs = pairs[pairs.f_a != pairs.f_b]
    m = re.compile(r"(low|mid|high)/view(\d+)\.mp4")
    cls = np.array([m.match(r).group(1) for r in rel])
    cid = np.array([int(m.match(r).group(2)) for r in rel])
    ka = cid[pairs.f_a.values] * 10 + 0
    kb = cid[pairs.f_b.values] * 10 + 0
    keep = (ka < kb) | ((ka == kb) & (pairs.f_a.values < pairs.f_b.values))
    pairs = pairs[keep]
    pairs["d_demux"] = pairs.pos_a - pairs.pos_b
    pairs["d_rank"] = pairs.rank_a - pairs.rank_b
    pairs["d_pts"] = pairs.pts_a - pairs.pts_b
    print(f"{len(pairs)} shared-packet rows over {pairs[['f_a', 'f_b']].drop_duplicates().shape[0]} pairs "
          f"({time.time() - t0:.1f} s)")
    rows = []
    for (fa, fb), g in pairs.groupby(["f_a", "f_b"], sort=False):
        od, fd = mode_and_frac(g.d_demux)
        orr, fr = mode_and_frac(g.d_rank)
        op, fp = mode_and_frac(g.d_pts)
        rows.append((fa, fb, len(g), int(g.key_a.sum()), od, fd, orr, fr, op, fp,
                     int(g.pos_a.min()), int(g.pos_a.max()), int(g.pos_b.min()), int(g.pos_b.max())))
    R = pd.DataFrame(rows, columns=["f_a", "f_b", "shared", "shared_key", "off_demux", "off_demux_frac", "off_rank",
                                    "off_rank_frac", "off_pts_ticks", "off_pts_frac", "pos_a_min", "pos_a_max",
                                    "pos_b_min", "pos_b_max"])
    R["rel_a"] = rel[R.f_a]
    R["rel_b"] = rel[R.f_b]
    R["cls_a"] = cls[R.f_a]
    R["cls_b"] = cls[R.f_b]
    R["id_a"] = cid[R.f_a]
    R["id_b"] = cid[R.f_b]
    R["id_diff"] = R.id_b - R.id_a
    R["n_a"] = n_pk_all[R.f_a]
    R["n_b"] = n_pk_all[R.f_b]
    o = R.off_rank.values
    na, nb = R.n_a.values, R.n_b.values
    in_a = (o >= 0) & (o + nb <= na)
    in_b = (o <= 0) & (na <= o + nb)
    R["kind"] = np.select([in_a & in_b, in_a, in_b], ["identical", "contained_b_in_a", "contained_a_in_b"], "partial")
    R["start_b_minus_a"] = o                                  # frames
    R["end_b_minus_a"] = o + nb - na                          # frames
    R["overlap_frames"] = np.minimum(na, o + nb) - np.maximum(0, o)
    R.to_csv(OUT / "packet_pairs_offsets.csv", index=False)
    print(f"wrote {len(R)} pairs in {time.time() - t0:.1f} s")
    print("cross-class pairs:", int((R.cls_a != R.cls_b).sum()))
    print("offset consistency (share of pairs whose mode covers 100% of shared packets): demux "
          f"{(R.off_demux_frac == 1).mean():.4f}, pts-rank {(R.off_rank_frac == 1).mean():.4f}, "
          f"pts {(R.off_pts_frac == 1).mean():.4f}; demux == rank offset: {(R.off_demux == R.off_rank).mean():.4f}")
    print(R.kind.value_counts().to_string())


if __name__ == "__main__":
    main()
