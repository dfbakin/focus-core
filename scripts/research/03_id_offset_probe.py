# Probe: are clip IDs interleaved over cameras?  For contiguous ID ranges, compare first/last frames of
# clip N with the first frame of clip N+k (k = 1..6) on 32x32 thumbnails (RMSE, 0-255).  If IDs cycle
# over 3 cameras, offset 3 should be much closer than offsets 1 and 2, and last(N) -> first(N+3) should
# sit near the consecutive-frame noise floor when segments are contiguous in time.
import cv2, numpy as np, sys, os
root = "data/ouc-cge-320/videos"
def thumbs(path):
    cap = cv2.VideoCapture(path); first = None; last = None; prev = None; n = 0; second = None
    while True:
        ok, fr = cap.read()
        if not ok: break
        n += 1
        if first is None: first = fr
        elif second is None: second = fr
        last = fr
    cap.release()
    f = lambda x: cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (f(first), f(second if second is not None else first), f(last), n) if first is not None else None
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
ranges = {"high": range(1000, 1061), "low": range(500, 561), "mid": range(300, 361)}
for cls, ids in ranges.items():
    T = {}
    for i in ids:
        p = f"{root}/{cls}/view{i}.mp4"
        if os.path.exists(p):
            t = thumbs(p)
            if t: T[i] = t
    floor = np.median([rmse(T[i][0], T[i][1]) for i in T])            # frame t -> t+1 inside a clip
    within = np.median([rmse(T[i][0], T[i][2]) for i in T])           # first -> last inside a clip
    print(f"\n{cls} ids {ids.start}-{ids.stop-1}: n={len(T)}, median n_frames={np.median([T[i][3] for i in T]):.0f}; "
          f"consecutive-frame floor {floor:.1f}; first->last within clip {within:.1f}")
    print("  offset k | median first(N)->first(N+k) | median last(N)->first(N+k) | share of pairs with last->first < 2*floor")
    for k in range(1, 7):
        ff = [rmse(T[i][0], T[i+k][0]) for i in T if i+k in T]
        lf = [rmse(T[i][2], T[i+k][0]) for i in T if i+k in T]
        close = np.mean([x < 2*floor for x in lf]) if lf else float('nan')
        print(f"  {k:8d} | {np.median(ff):10.1f} | {np.median(lf):10.1f} | {close:5.2f}")
