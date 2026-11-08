#!/usr/bin/env python
"""Figures for manual checking of the session reconstruction (docs/research/figures/sessions_*.png, 150 dpi).

  sessions_events_sheet.png      10 random events (default_rng(0)) with >= 2 recordings and audio evidence: one row
                                 each, the first frame of the clip of every recording that covers a common instant,
                                 with the recording's best audio r and its offset on the event axis
  sessions_doubtful_sheet.png    the 15 most doubtful decisions of doubtful_decisions.csv, shown as the frames the
                                 decision joins
  sessions_candidates_sheet.png  8 random (seed 0) cross-class session candidate links: two frames of the
                                 best-matching recording of each event
  sessions_timeline.png          the 6 largest events: every clip as a bar on the event time axis, by recording,
                                 coloured by the published split
  sessions_recordings_sheet.png  one frame per recording (the source of the agent's room-layout labels)
  sessions_audio_null.png        r60 of candidate pairs against the two nulls, per length bin, with thresholds
  sessions_tier0_merges.png      junction RMSE: packet-verified junctions, accepted probable merges, rejected candidates
The figures show students' faces: internal use only.
Usage: CUDA_VISIBLE_DEVICES="" nice -n 10 .venv/bin/python scripts/research/05f_figures.py
"""
import resource
import textwrap
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "sessions"
FIG = REPO / "docs" / "research" / "figures"
CACHE = REPO / "outputs" / "review_cache"
T = REPO / "outputs" / "research" / "temporal"
DPI = 150
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4 (dataviz reference palette)
INK, MUTED = "#222222", "#777777"
SPLIT_COL = {"train": SER[0], "val": SER[1], "test": SER[2]}
plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def load_rows(rows):
    rows = set(int(r) for r in rows)
    out = {}
    base = 0
    for s in ("train", "val", "test"):
        z = np.load(CACHE / f"frames_{s}_hires.npz", allow_pickle=True)
        n = len(z["view_id"])
        for name in ("first", "last"):
            want = [r for r in rows if base <= r < base + n]
            if want:
                arr = z[name]
                for r in want:
                    out[(r, name)] = arr[r - base].copy()
                del arr
        base += n
        del z
    return out


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def common_instant(E, recs):
    """event time covered by the most recordings (offsets known)."""
    iv = [(E.event_offset_s[r], E.event_offset_s[r] + E.span_s[r]) for r in recs if np.isfinite(E.event_offset_s[r])]
    pts = sorted({a + 0.5 for a, _ in iv})
    best = max(pts, key=lambda t: sum(a <= t <= b for a, b in iv)) if pts else 0.0
    return best


def clip_at(C, rec, t_rec):
    g = C[(C.recording == rec)]
    cov = g[(g.t_start_s <= t_rec) & (g.t_end_s >= t_rec)]
    if not len(cov):
        cov = g.iloc[[(g.t_start_s - t_rec).abs().values.argmin()]]
    return cov.iloc[(cov.t_start_s - t_rec).abs().values.argmin()]


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    C = pd.read_csv(OUT / "clip_recordings.csv")
    E = pd.read_csv(OUT / "events.csv").set_index("recording")
    ET = pd.read_csv(OUT / "event_table.csv").set_index("event")
    PR = pd.read_csv(OUT / "audio_pairs.csv")
    DD = pd.read_csv(OUT / "doubtful_decisions.csv")
    SL = pd.read_csv(OUT / "session_links.csv")
    MG = pd.read_csv(OUT / "merges.csv")
    TH = pd.read_csv(OUT / "audio_thresholds.csv")
    NULL = pd.read_csv(OUT / "audio_null.csv")
    CS = pd.read_csv(OUT / "clip_sessions.csv")
    split_of = dict(zip(CS.path, CS.split_published))

    # ---------------- plan all frames needed, then read them once
    plan = []   # (figure, row, col, cache_row, which, label)
    # events sheet
    ev_ok = ET[(ET.n_recordings >= 2) & ET.tiers.str.contains("audio")].index.values
    ev_pick = sorted(rng.choice(ev_ok, size=min(10, len(ev_ok)), replace=False))
    for i, ev in enumerate(ev_pick):
        recs = [r for r in E.index[E.event == ev] if np.isfinite(E.event_offset_s[r])]
        recs = sorted(recs, key=lambda r: (E.duplicate_of_recording.notna()[r], -E.n_clips[r]))[:6]
        tE = common_instant(E, recs)
        for j, r in enumerate(recs):
            c = clip_at(C, r, tE - E.event_offset_s[r])
            dt = c.t_start_s + E.event_offset_s[r] - tE
            rr = E.best_audio_r[r]
            lab = (f"ev{ev} rec{r} {E.cls[r]} c{int(E.c27[r])}\n{c.rel.split('/')[1][:-4]} dt={dt:+.1f}s "
                   f"r={rr:.2f}" if np.isfinite(rr) else f"ev{ev} rec{r} {E.cls[r]}\n{c.rel.split('/')[1][:-4]} {E.event_tier[r]}")
            if E.duplicate_of_recording.notna()[r]:
                lab += " dup"
            plan.append(("events", i, j, int(c.row), "first", lab))
    # doubtful sheet
    top = DD.head(15)
    units = C.groupby("unit")
    for i, d in enumerate(top.itertuples()):
        if d.kind == "merge":
            ga, gb = units.get_group(d.a), units.get_group(d.b)
            ta = ga.loc[ga.t1.idxmax()]
            hb = gb.loc[gb.t0.idxmin()]
            plan.append(("doubt", i, 0, int(ta.row), "first", f"unit {d.a} tail {ta.rel.split('/')[1][:-4]} (first)"))
            plan.append(("doubt", i, 1, int(ta.row), "last", f"unit {d.a} tail (last frame)"))
            plan.append(("doubt", i, 2, int(hb.row), "first", f"unit {d.b} head {hb.rel.split('/')[1][:-4]} (first)"))
            plan.append(("doubt", i, 3, int(hb.row), "last", f"unit {d.b} head (last)"))
        else:
            a, b = int(d.a), int(d.b)
            if d.kind == "audio" and np.isfinite(E.event_offset_s[a]) and np.isfinite(E.event_offset_s[b]):
                tE = common_instant(E, [a, b])
                ts = [tE, tE + min(E.span_s[a], E.span_s[b]) * 0.4]
                for j, (r, tt) in enumerate([(a, ts[0]), (b, ts[0]), (a, ts[1]), (b, ts[1])]):
                    c = clip_at(C, r, tt - E.event_offset_s[r])
                    dt = c.t_start_s + E.event_offset_s[r] - tt
                    plan.append(("doubt", i, j, int(c.row), "first", f"rec{r} {E.cls[r]} {c.rel.split('/')[1][:-4]} dt={dt:+.1f}"))
            else:
                for j, r in enumerate([a, b]):
                    g = C[C.recording == r].sort_values("t_start_s")
                    for k, q in enumerate([0.25, 0.75]):
                        c = g.iloc[int(q * (len(g) - 1))]
                        plan.append(("doubt", i, 2 * j + k, int(c.row), "first", f"rec{r} {E.cls[r]} {c.rel.split('/')[1][:-4]}"))
    # candidates sheet
    cand_pick = SL.iloc[sorted(rng.choice(len(SL), size=min(8, len(SL)), replace=False))] if len(SL) else SL
    for i, s in enumerate(cand_pick.itertuples()):
        for j, r in enumerate([int(s.best_view_rec_a), int(s.best_view_rec_b)]):
            g = C[C.recording == r].sort_values("t_start_s")
            for k, q in enumerate([0.3, 0.7]):
                c = g.iloc[int(q * (len(g) - 1))]
                plan.append(("cand", i, 2 * j + k, int(c.row), "first", f"ev{E.event[r]} rec{r} {E.cls[r]} {c.rel.split('/')[1][:-4]}"))
    # recordings sheet
    for k, r in enumerate(E.index):
        g = C[C.recording == r].sort_values("t_start_s")
        c = g.iloc[len(g) // 2]
        plan.append(("recs", k // 12, k % 12, int(c.row), "first",
                     f"rec{r} {E.cls[r]} n{E.n_clips[r]} c{int(E.c27[r])}\nev{E.event[r]} {E.layout[r]}"))
    FR = load_rows([p[3] for p in plan])
    print(f"{len(FR)} frames loaded, {time.time() - t0:.1f} s", flush=True)

    def sheet(name, nrows, ncols, title, row_titles=None, fname=None, w=1.9, h=1.25):
        fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * w, nrows * h + 0.6), squeeze=False)
        for ax in axs.flat:
            blank(ax)
        for p in plan:
            if p[0] != name:
                continue
            ax = axs[p[1], p[2]]
            ax.imshow(FR[(p[3], p[4])])
            ax.set_title(p[5], fontsize=5.5, color=INK, pad=2)
        if row_titles:
            for i, t in enumerate(row_titles):
                axs[i, 0].set_ylabel(t, fontsize=5.5, rotation=0, ha="right", va="center", color=INK, labelpad=4)
        fig.suptitle(textwrap.fill(title, int(ncols * w * 13)), fontsize=8, color=INK, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (nrows * h + 0.6)))
        fig.savefig(FIG / fname, dpi=DPI)
        plt.close(fig)

    sheet("events", len(ev_pick), 6, "Ten random audio events (seed 0): one frame per recording at a common instant of the event axis "
          "(dt = clip start minus that instant; r = recording's best accepted audio link)", fname="sessions_events_sheet.png")
    rt = [f"#{i + 1} {d.tier}\n{d.decision[:38]}\n{d.value[:40]}\nconf={d.confidence:.2f} clips={d.clips_affected}" for i, d in enumerate(top.itertuples())]
    sheet("doubt", len(top), 4, "The 15 most doubtful decisions (doubt = (1 - confidence) x clips on the smaller side); "
          "merges: tail clip first/last, head clip first/last; audio: both recordings at two common instants",
          row_titles=rt, fname="sessions_doubtful_sheet.png", w=2.1, h=1.3)
    rt = [f"ev{s.event_a} ({s.classes_a}) - ev{s.event_b} ({s.classes_b})\nview_d={s.best_view_d:.1f} cloth={s.cloth_s_view_matched:.2f}\nscore={s.session_score:.3f}"
          for s in cand_pick.itertuples()]
    sheet("cand", len(cand_pick), 4, "Eight random cross-class session CANDIDATES (seed 0): two frames of the best view-matched "
          "recording of each event (left pair = first event, right pair = second)", row_titles=rt,
          fname="sessions_candidates_sheet.png", w=2.1, h=1.3)
    sheet("recs", int(np.ceil(len(E) / 12)), 12, "One frame per recording (median clip): source of the room-layout labels "
          "(checkerboard = rectangular desks and blackboard)", fname="sessions_recordings_sheet.png", w=1.6, h=1.15)

    # ---------------- timeline of the 6 largest events
    big = ET.sort_values("n_clips", ascending=False).head(6).index
    fig, axs = plt.subplots(6, 1, figsize=(11, 13))
    for ax, ev in zip(axs, big):
        recs = [r for r in E.index[E.event == ev]]
        recs = sorted(recs, key=lambda r: (not np.isfinite(E.event_offset_s[r]), E.event_offset_s[r] if np.isfinite(E.event_offset_s[r]) else 0))
        for y, r in enumerate(recs):
            off = E.event_offset_s[r]
            unknown = not np.isfinite(off)
            g = C[C.recording == r].sort_values("t_start_s")
            for k, c in enumerate(g.itertuples()):
                x0 = (0 if unknown else off) + c.t_start_s
                lane = y + (0.18 if k % 2 else -0.18)
                ax.plot([x0, x0 + (c.t_end_s - c.t_start_s)], [lane, lane], color=SPLIT_COL[split_of.get(c.path, "train")],
                        lw=3.2, solid_capstyle="butt", alpha=0.35 if unknown else 1.0)
            tag = f"rec{r} {E.cls[r]} c{int(E.c27[r])} {E.event_tier[r]}" + (" dup" if pd.notna(E.duplicate_of_recording[r]) else "") + \
                  (" (offset unknown)" if unknown else "")
            ax.text(-4, y, tag, ha="right", va="center", fontsize=6, color=INK)
        ax.set_yticks([])
        ax.set_ylim(-0.8, len(recs) - 0.2)
        ax.set_xlim(-2, None)
        ax.grid(axis="x", color="#e5e5e5", lw=0.6)
        ax.set_title(f"event {ev}: {ET.n_recordings[ev]} recordings, {ET.n_clips[ev]} clips, classes {ET.classes[ev]}, "
                     f"max concurrent {ET.max_concurrent[ev]}", fontsize=7.5, color=INK, loc="left")
        ax.spines["left"].set_visible(False)
    axs[-1].set_xlabel("event time (s); every clip is a bar, alternating lanes show the clip boundaries")
    handles = [plt.Line2D([0], [0], color=SPLIT_COL[s], lw=3.2, label=s) for s in ("train", "val", "test")]
    fig.legend(handles=handles, loc="upper right", frameon=False, title="published split", fontsize=7, title_fontsize=7)
    fig.suptitle("The six largest events on a common time axis", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0.12, 0, 1, 0.97))
    fig.savefig(FIG / "sessions_timeline.png", dpi=DPI)
    plt.close(fig)

    # ---------------- audio null vs candidates, per length bin
    fig, axs = plt.subplots(2, 4, figsize=(12, 5.2), sharex=True)
    bins = np.linspace(-0.3, 1.0, 66)
    PR["bin"] = PR.P.map(lambda p: int(np.searchsorted([0, 10, 20, 40, 80, 160, 320, 640, np.inf], p, side="right") - 1))
    for bi, ax in enumerate(axs.flat):
        sets = [("null B: same recording, disjoint", NULL[(NULL.kind == "null_B_same_recording") & (NULL.bin == bi)].r60, SER[0]),
                ("null R: partner reversed", NULL[(NULL.kind == "null_R_reversed") & (NULL.bin == bi)].r60, SER[1]),
                ("same-class pairs", PR[(PR.kind == "same_class") & (PR.bin == bi)].r60, SER[2]),
                ("cross-class pairs", PR[(PR.kind == "cross_class") & (PR.bin == bi)].r60, SER[3])]
        for lab, v, col in sets:
            v = v.dropna()
            if len(v):
                ax.hist(v, bins=bins, density=True, histtype="step", lw=1.4, color=col, label=f"{lab} (n={len(v)})")
        thr = TH.loc[TH.bin == bi, "thr_q99.9"]
        if len(thr) and np.isfinite(thr.iloc[0]):
            ax.axvline(thr.iloc[0], color=INK, lw=0.8, ls="--")
        ax.set_yscale("log")
        row = TH[TH.bin == bi].iloc[0]
        ax.set_title(f"P {row.P_lo:g}-{row.P_hi:g} s", fontsize=7.5, color=INK, loc="left")
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, [x.split(" (n=")[0] for x in l_], loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.95))
    for ax in axs[1]:
        ax.set_xlabel("r60 (best masked correlation, lags within 60 s)")
    for ax in axs[:, 0]:
        ax.set_ylabel("density (log)")
    fig.suptitle("Audio synchrony: candidate recording pairs against two nulls, by valid audio length P of the shorter "
                 "recording (dashed: primary threshold, 99.9th percentile of the larger null)", fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIG / "sessions_audio_null.png", dpi=DPI)
    plt.close(fig)

    # ---------------- Tier-0 junction distances
    L = pd.read_csv(T / "chain_links.csv")
    jd = np.sort(L[L.type == "packet"].d_junction.dropna().values)
    acc = np.sort(MG[MG.status == "accepted"].d128.values)
    rej_struct = np.sort(MG[(MG.status != "accepted") & MG.same_c27 & MG.same_run & MG.direction_ok].d128.values)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for v, lab, col in ((jd, f"packet-verified junctions (exact, n={len(jd)})", SER[0]),
                        (acc, f"accepted probable merges (n={len(acc)})", SER[1]),
                        (rej_struct, f"rejected candidates, same scene/run/direction (n={len(rej_struct)})", SER[2])):
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, lw=1.6, label=lab)
    cut = float(np.percentile(jd, 95))
    ax.axvline(cut, color=INK, lw=0.8, ls="--")
    ax.text(cut + 1, 0.05, f"cut {cut:.1f} = p95 of exact junctions", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("RMSE(last frame of tail, first frame of head), 128x72")
    ax.set_ylabel("cumulative share")
    ax.grid(color="#eeeeee", lw=0.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("Tier 0: junction distance of probable merges against exact junctions", fontsize=8, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "sessions_tier0_merges.png", dpi=DPI)
    plt.close(fig)
    print(f"wall {time.time() - t0:.1f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
