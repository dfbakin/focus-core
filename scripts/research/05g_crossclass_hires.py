"""Supervisor spot-check of the cross-class audio-synchronous pairs (05c, Surprise 1).

For every accepted audio link whose two recordings sit in different class folders, take instants inside the
synchronous overlap on the event axis, extract one full-resolution frame per recording of that event at each
instant (PyAV, raw release files) and compose a sheet. The sheet shows faces and stays out of git.

Inputs : outputs/research/sessions/{audio_pairs,events,clip_sessions}.csv, data/ouc-cge/videos/
Outputs: docs/research/figures/sessions_crossclass_hires.png, outputs/research/sessions/crossclass_frames.csv
"""
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "research" / "sessions"
RAW = REPO / "data" / "ouc-cge"
FIG = REPO / "docs" / "research" / "figures"
TMP = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "crossclass_tmp"
TMP.mkdir(parents=True, exist_ok=True)
W = 640
N_INSTANTS = 3


def frame_at(path: Path, off_s: float, dst: Path):
    """Decode sequentially (clips are <= 10 s) and keep the first frame at or after off_s from the first frame."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        tb = float(s.time_base)
        first = None
        last = None
        for fr in c.decode(s):
            if first is None:
                first = fr.pts
            last = fr
            if (fr.pts - first) * tb >= off_s - 1e-6:
                break
        im = last.to_image()
    im = im.resize((W, round(im.height * W / im.width)))
    im.save(dst)


def main():
    ap = pd.read_csv(OUT / "audio_pairs.csv")
    ev = pd.read_csv(OUT / "events.csv").set_index("recording")
    cs = pd.read_csv(OUT / "clip_sessions.csv")
    cs = cs[cs.recording_id.notna()].copy()
    cs["dur"] = cs.t_end_s - cs.t_start_s
    acc = ap[(ap["accept_q99.9"] == True) | (ap["accept_full"] == True)]  # noqa: E712
    cross = acc[acc.cls_a != acc.cls_b]
    events = sorted(set(ev.loc[cross.a, "event"]) | set(ev.loc[cross.b, "event"]))
    rows, meta = [], []
    for e in events:
        recs = ev[ev.event == e]
        recs = recs[recs.duplicate_of_recording.isna()] if "duplicate_of_recording" in recs else recs
        recs = recs[recs.event_offset_s.notna()].sort_values("cls")
        # synchronous overlap on the event axis = intersection of the recordings' spans
        spans = []
        for rec in recs.index:
            c = cs[cs.recording_id == rec]
            spans.append((c.t_event_s.min(), (c.t_event_s + c.dur).max()))
        lo, hi = max(s[0] for s in spans), min(s[1] for s in spans)
        instants = np.linspace(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo), N_INSTANTS)
        for t in instants:
            row = []
            for rec in recs.index:
                c = cs[cs.recording_id == rec]
                hit = c[(c.t_event_s <= t) & (c.t_event_s + c.dur > t)]
                if hit.empty:  # gap between clips of this recording: nearest clip start
                    hit = c.iloc[[(c.t_event_s - t).abs().argmin()]]
                h = hit.iloc[0]
                off = float(np.clip(t - h.t_event_s, 0, max(h.dur - 0.1, 0)))
                dst = TMP / f"ev{e}_rec{rec}_{Path(h.path).stem}_{off:.2f}.png"
                if not dst.exists():
                    frame_at(RAW / h.path, off, dst)
                label = f"ev{e} t={t:.1f}s | rec{rec} {h['class']} {Path(h.path).stem} +{off:.1f}s | split {h.split_published}"
                row.append((dst, label))
                meta.append(dict(event=e, t_event_s=round(float(t), 2), recording=rec, cls=h["class"], clip=h.path,
                                 offset_in_clip_s=round(off, 2), split=h.split_published))
            rows.append((e, t, row))
    pd.DataFrame(meta).to_csv(OUT / "crossclass_frames.csv", index=False)
    ncol = max(len(r[2]) for r in rows)
    im0 = Image.open(rows[0][2][0][0])
    h = im0.height
    pad, head = 22, 60
    sheet = Image.new("RGB", (ncol * (W + 8) + 8, head + len(rows) * (h + pad + 8)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 8), "Cross-class audio-synchronous events (05c, Surprise 1): one full-resolution frame per recording at a "
                    "common instant of the event axis. Same room and instant, different class folder.", fill="black")
    d.text((10, 26), f"Events {events}; {N_INSTANTS} instants each, inside the synchronous overlap. Faces visible: internal only.",
           fill="black")
    for i, (e, t, row) in enumerate(rows):
        y = head + i * (h + pad + 8)
        for j, (p, label) in enumerate(row):
            x = 8 + j * (W + 8)
            d.text((x, y), label, fill="black")
            sheet.paste(Image.open(p), (x, y + pad))
    FIG.mkdir(exist_ok=True)
    sheet.save(FIG / "sessions_crossclass_hires.png", optimize=True)
    print("events", events, "rows", len(rows), "->", FIG / "sessions_crossclass_hires.png")
    for m in meta:
        print(m)


if __name__ == "__main__":
    main()
