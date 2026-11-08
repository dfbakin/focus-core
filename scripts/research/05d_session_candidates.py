#!/usr/bin/env python
"""Tier 2 of the session reconstruction: CANDIDATE links between events of different class folders.

Nothing in the release links two class folders at the frame or packet level, and events of different phases
are different times, so audio cannot link them either (the cross-class audio links of 05c are simultaneous
footage and are already inside Tier-1 events). The evidence here is visual and circumstantial:

  viewpoint  background of a recording = per-pixel median of up to 60 first/last frames (128x72) of clips
             spread over the recording; "static" pixels = per-pixel median absolute deviation (gray) < 6.
             view_d(r1, r2) = RMSE of the two backgrounds over pixels static in both (>= 25% of the image).
  people     pixels that move (MAD >= 12) in a recording are the people region; their colours over all sampled
             frames form an HSV histogram (8 hue x 4 saturation x 4 value). cloth_s = histogram intersection.
  layout     the agent's visual label (round / checkerboard), from 05c.
  batch      codec | frame size of the matched recordings (reported, not scored).

Calibration (no ground truth exists):
  same viewpoint, same people  first half vs second half of the same recording (one camera, one group);
  different viewpoint, same people  pairs of recordings in one audio event (different cameras, same moment);
  reference   all pairs of recordings from different events of the same class (mostly different sessions).
A pair of events (E1 in class A, E2 in class B) is a view match if some recording pair has view_d below the
95th percentile of the same-recording halves. Its session_score = the share of reference pairs whose cloth_s
is below the pair's best cloth_s among view-matched recording pairs (0-1). A CANDIDATE link needs a view match,
the same layout, and cloth_s above the 5th percentile of the same-recording halves; each event keeps its best
candidate per other class, and only mutual best candidates are linked. session_candidate_id = connected
components of events under these links. These links are candidates for manual checking, not established.

Outputs (outputs/research/sessions/): recording_features.npz, session_pairs.csv (every cross-class event pair),
session_links.csv (accepted candidate links), session_candidates.csv (event -> session_candidate_id),
tier2_summary.json.
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/05d_session_candidates.py
"""
import json
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "sessions"
CACHE = REPO / "outputs" / "review_cache"
N_CLIPS = 30          # clips sampled per recording (first + last frame each)
STATIC_MAD, MOVE_MAD = 6.0, 12.0
MIN_STATIC = 0.25


def sample_rows(C, rng):
    """cache rows to read per recording, split into halves by time."""
    want = {}
    for rec, g in C.groupby("recording"):
        g = g.sort_values("t_start_s")
        idx = np.unique(np.linspace(0, len(g) - 1, min(N_CLIPS, len(g))).round().astype(int))
        rows = g.row.values[idx]
        half = (np.arange(len(idx)) >= len(idx) / 2).astype(int)
        want[rec] = list(zip(rows, half))
    return want


def load_frames(want):
    """{rec: [(half, frame uint8 72x128x3), ...]} reading one npz array at a time."""
    need = {r for v in want.values() for r, _ in v}
    got = {}
    base = 0
    for s in ("train", "val", "test"):
        z = np.load(CACHE / f"frames_{s}_hires.npz", allow_pickle=True)
        n = len(z["view_id"])
        for name in ("first", "last"):
            arr = z[name]
            for r in need:
                if base <= r < base + n:
                    got[(r, name)] = arr[r - base].copy()
            del arr
        base += n
        del z
    out = {}
    for rec, v in want.items():
        out[rec] = [(h, got[(r, nm)]) for r, h in v for nm in ("first", "last") if (r, nm) in got]
    return out


def features(frames):
    """background (72,128,3) float32, static mask, moving mask, people HSV histogram (128,)"""
    X = np.stack(frames).astype(np.float32)
    bg = np.median(X, axis=0)
    gray = X.mean(-1)
    mad = np.median(np.abs(gray - np.median(gray, axis=0)), axis=0)
    static = mad < STATIC_MAD
    moving = mad >= MOVE_MAD
    hist = np.zeros(128, np.float64)
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        px = hsv[moving]
        if len(px):
            h = (px[:, 0].astype(int) * 8 // 180).clip(0, 7)
            sat = (px[:, 1].astype(int) * 4 // 256).clip(0, 3)
            val = (px[:, 2].astype(int) * 4 // 256).clip(0, 3)
            np.add.at(hist, h * 16 + sat * 4 + val, 1)
    if hist.sum() > 0:
        hist /= hist.sum()
    return bg, static, moving, hist


def view_d(f1, f2):
    m = f1[1] & f2[1]
    if m.mean() < MIN_STATIC:
        return np.nan
    return float(np.sqrt(((f1[0] - f2[0]) ** 2)[m].mean()))


def cloth_s(h1, h2):
    if h1.sum() == 0 or h2.sum() == 0:
        return np.nan
    return float(np.minimum(h1, h2).sum())


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    C = pd.read_csv(OUT / "clip_recordings.csv")
    E = pd.read_csv(OUT / "events.csv").set_index("recording")
    want = sample_rows(C, rng)
    frames = load_frames(want)
    F, Fh = {}, {}
    for rec, fr in frames.items():
        F[rec] = features([f for _, f in fr])
        for h in (0, 1):
            sub = [f for hh, f in fr if hh == h]
            if len(sub) >= 6:
                Fh[(rec, h)] = features(sub)
    del frames
    print(f"features for {len(F)} recordings, {time.time() - t0:.1f} s", flush=True)
    np.savez_compressed(OUT / "recording_features.npz", rec=np.array(sorted(F)),
                        bg=np.stack([F[r][0] for r in sorted(F)]).astype(np.uint8),
                        static=np.stack([F[r][1] for r in sorted(F)]), moving=np.stack([F[r][2] for r in sorted(F)]),
                        hist=np.stack([F[r][3] for r in sorted(F)]))

    # ---- calibration populations
    halves = [dict(rec=r, view_d=view_d(Fh[(r, 0)], Fh[(r, 1)]), cloth_s=cloth_s(Fh[(r, 0)][3], Fh[(r, 1)][3]))
              for r in F if (r, 0) in Fh and (r, 1) in Fh]
    H = pd.DataFrame(halves)
    recs = sorted(F)
    pair_rows = []
    for i, a in enumerate(recs):
        for b in recs[i + 1:]:
            pair_rows.append(dict(a=a, b=b, cls_a=E.cls[a], cls_b=E.cls[b], ev_a=E.event[a], ev_b=E.event[b],
                                  view_d=view_d(F[a], F[b]), cloth_s=cloth_s(F[a][3], F[b][3]),
                                  same_layout=E.layout[a] == E.layout[b]))
    P = pd.DataFrame(pair_rows)
    same_event = P[P.ev_a == P.ev_b]
    ref = P[(P.ev_a != P.ev_b) & (P.cls_a == P.cls_b)]
    VIEW_CUT = float(np.nanpercentile(H.view_d, 95))
    CLOTH_MIN = float(np.nanpercentile(H.cloth_s, 5))
    ref_cloth = np.sort(ref.cloth_s.dropna().values)
    print(f"view cut {VIEW_CUT:.2f}, cloth min {CLOTH_MIN:.3f}", flush=True)

    # ---- event pairs across classes
    ev_recs = E.groupby("event").apply(lambda g: list(g.index), include_groups=False).to_dict()
    ev_cls = E.groupby("event").cls.agg(lambda s: ",".join(sorted(set(s)))).to_dict()
    ev_layout = E.groupby("event").layout.first().to_dict()
    ev_nclips = E.groupby("event").n_clips.sum().to_dict()
    combo = E.combo.to_dict()
    Pk = {(r.a, r.b): r for r in P.itertuples()}
    rows = []
    evs = sorted(ev_recs)
    for i, e1 in enumerate(evs):
        for e2 in evs[i + 1:]:
            c1, c2 = set(ev_cls[e1].split(",")), set(ev_cls[e2].split(","))
            if c1 & c2 and c1 == c2:
                continue                       # same class set: not a cross-class candidate
            if not (c1 ^ c2):
                continue
            best_v, best_pair, best_c_matched = np.inf, None, np.nan
            cl_all = []
            for a in ev_recs[e1]:
                for b in ev_recs[e2]:
                    if E.cls[a] == E.cls[b]:
                        continue
                    r = Pk.get((min(a, b), max(a, b)))
                    if r is None:
                        continue
                    cl_all.append(r.cloth_s)
                    if np.isfinite(r.view_d) and r.view_d < best_v:
                        best_v, best_pair = r.view_d, (a, b)
                    if np.isfinite(r.view_d) and r.view_d <= VIEW_CUT and np.isfinite(r.cloth_s):
                        best_c_matched = np.nanmax([best_c_matched, r.cloth_s])
            if best_pair is None and not cl_all:
                continue
            view_match = np.isfinite(best_c_matched)
            score = float(np.searchsorted(ref_cloth, best_c_matched) / len(ref_cloth)) if view_match else np.nan
            a, b = best_pair if best_pair else (np.nan, np.nan)
            rows.append(dict(event_a=e1, event_b=e2, classes_a=ev_cls[e1], classes_b=ev_cls[e2],
                             n_clips_a=ev_nclips[e1], n_clips_b=ev_nclips[e2], layout_a=ev_layout[e1], layout_b=ev_layout[e2],
                             best_view_d=best_v if np.isfinite(best_v) else np.nan, best_view_rec_a=a, best_view_rec_b=b,
                             view_match=view_match, cloth_s_view_matched=best_c_matched,
                             cloth_s_max_any=float(np.nanmax(cl_all)) if cl_all else np.nan,
                             same_batch_family="|".join(str(combo.get(a, "")).split("|")[:2]) == "|".join(str(combo.get(b, "")).split("|")[:2]) if best_pair else False,
                             session_score=score))
    SP = pd.DataFrame(rows)
    SP["candidate"] = SP.view_match & (SP.layout_a == SP.layout_b) & (SP.cloth_s_view_matched >= CLOTH_MIN)
    # mutual best per other class
    def other_classes(x):
        return x
    best_for = {}
    for r in SP[SP.candidate].itertuples():
        for e, other, ocls in ((r.event_a, r.event_b, r.classes_b), (r.event_b, r.event_a, r.classes_a)):
            k = (e, ocls)
            if k not in best_for or r.session_score > best_for[k][1]:
                best_for[k] = (other, r.session_score)
    SP["mutual_best"] = [bool(r.candidate and best_for.get((r.event_a, r.classes_b), (None,))[0] == r.event_b
                              and best_for.get((r.event_b, r.classes_a), (None,))[0] == r.event_a) for r in SP.itertuples()]
    SP.to_csv(OUT / "session_pairs.csv", index=False)
    LK = SP[SP.mutual_best].copy()
    LK.to_csv(OUT / "session_links.csv", index=False)
    parent = {e: e for e in evs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in LK.itertuples():
        parent[find(r.event_b)] = find(r.event_a)
    comp = {e: find(e) for e in evs}
    roots = sorted(set(comp.values()), key=lambda k: (-sum(ev_nclips[e] for e in evs if comp[e] == k), k))
    sid = {k: i for i, k in enumerate(roots)}
    best_score = {}
    for r in LK.itertuples():
        for e in (r.event_a, r.event_b):
            best_score[e] = max(best_score.get(e, 0), r.session_score)
    SC = pd.DataFrame(dict(event=evs, session_candidate_id=[sid[comp[e]] for e in evs],
                           classes=[ev_cls[e] for e in evs], n_clips=[ev_nclips[e] for e in evs],
                           session_score=[best_score.get(e, np.nan) for e in evs]))
    SC["session_size_events"] = SC.session_candidate_id.map(SC.session_candidate_id.value_counts())
    SC["session_classes"] = SC.session_candidate_id.map(SC.groupby("session_candidate_id").classes.agg(lambda s: ",".join(sorted(set(",".join(s).split(","))))))
    SC.to_csv(OUT / "session_candidates.csv", index=False)
    multi = SC[SC.session_size_events > 1]
    S = {
        "n_recordings_with_features": len(F),
        "halves_view_d": H.view_d.describe(percentiles=[.05, .5, .95]).round(2).to_dict(),
        "halves_cloth_s": H.cloth_s.describe(percentiles=[.05, .5, .95]).round(3).to_dict(),
        "same_event_pairs_view_d": same_event.view_d.describe(percentiles=[.05, .5, .95]).round(2).to_dict(),
        "same_event_pairs_cloth_s": same_event.cloth_s.describe(percentiles=[.05, .5, .95]).round(3).to_dict(),
        "reference_pairs_view_d": ref.view_d.describe(percentiles=[.05, .5, .95]).round(2).to_dict(),
        "reference_pairs_cloth_s": ref.cloth_s.describe(percentiles=[.05, .5, .95]).round(3).to_dict(),
        "reference_frac_view_match": float((ref.view_d <= VIEW_CUT).mean()),
        "view_cut": VIEW_CUT, "cloth_min": CLOTH_MIN,
        "n_cross_class_event_pairs": int(len(SP)), "n_view_match": int(SP.view_match.sum()),
        "n_candidates": int(SP.candidate.sum()), "n_mutual_best_links": int(len(LK)),
        "link_score": LK.session_score.describe().round(3).to_dict() if len(LK) else {},
        "events_with_candidate_partner": int(len(set(LK.event_a) | set(LK.event_b))),
        "events_ge2_rec_with_candidate_partner": int(len((set(LK.event_a) | set(LK.event_b)) & set(E.groupby("event").size()[lambda s: s > 1].index))),
        "n_events": len(evs), "n_session_candidates": int(SC.session_candidate_id.nunique()),
        "n_session_candidates_multi_event": int(multi.session_candidate_id.nunique()),
        "clips_in_multi_event_sessions": int(multi.n_clips.sum()),
        "sessions_with_all_three_classes": int(SC.drop_duplicates("session_candidate_id").session_classes.eq("high,low,mid").sum()),
        "session_class_sets": SC.drop_duplicates("session_candidate_id")[lambda d: d.session_size_events > 1].session_classes.value_counts().to_dict(),
        "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
    }
    with open(OUT / "tier2_summary.json", "w") as fh:
        json.dump(S, fh, indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(LK.sort_values("session_score", ascending=False).to_string())


if __name__ == "__main__":
    main()
