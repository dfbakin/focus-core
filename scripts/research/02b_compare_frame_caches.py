#!/usr/bin/env python
"""Compare two frame-cache directories key by key (np.array_equal, dtype, shape).

Compares frames_{train,val,test}.npz of --a and --b. With --hires-ref, also checks that
frames_{split}_hires.npz in --b has the expected keys / dtypes / shapes and that its
path/view_id/label/split/ok equal the main file of --a.  Exit code 1 on any difference.

Usage:
  CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/research/02b_compare_frame_caches.py \
      --a outputs/research/frame_cache_limit10 --b outputs/research/frame_cache_stream_limit10 --hires
"""
import argparse
import sys
from pathlib import Path

import numpy as np

SPLITS = ["train", "val", "test"]
HIRES_KEYS = ["path", "view_id", "label", "split", "ok", "first", "last"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True, help="reference (01_frame_cache.py) cache dir")
    ap.add_argument("--b", type=Path, required=True, help="cache dir to check (02_frame_cache_stream.py)")
    ap.add_argument("--hires", action="store_true", help="also check frames_{split}_hires.npz of --b")
    args = ap.parse_args()
    all_ok = True
    for s in SPLITS:
        A = np.load(args.a / f"frames_{s}.npz", allow_pickle=False)
        B = np.load(args.b / f"frames_{s}.npz", allow_pickle=False)
        ka, kb = list(A.files), list(B.files)
        same_keys = ka == kb
        all_ok &= same_keys
        print(f"[{s}] keys equal (same order): {same_keys}  {ka if same_keys else (ka, kb)}")
        for k in ka:
            if k not in kb:
                print(f"  {k:10s} MISSING in b"); all_ok = False; continue
            a, b = A[k], B[k]
            eq = a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
            all_ok &= eq
            print(f"  {k:10s} dtype {str(a.dtype):7s}/{str(b.dtype):7s} shape {str(a.shape):20s}/{str(b.shape):20s} "
                  f"array_equal={np.array_equal(a, b)} -> {'OK' if eq else 'DIFF'}")
        if args.hires:
            H = np.load(args.b / f"frames_{s}_hires.npz", allow_pickle=False)
            hk_ok = list(H.files) == HIRES_KEYS
            all_ok &= hk_ok
            n = len(A["path"])
            msgs = [f"keys={list(H.files)} {'OK' if hk_ok else 'DIFF'}"]
            for k in ("path", "view_id", "label", "split", "ok"):
                e = H[k].dtype == A[k].dtype and np.array_equal(H[k], A[k])
                all_ok &= e
                msgs.append(f"{k}={'OK' if e else 'DIFF'}")
            for k in ("first", "last"):
                e = H[k].dtype == np.uint8 and H[k].shape == (n, 72, 128, 3)
                z = bool((H[k][~A["ok"]] == 0).all())
                all_ok &= e and z
                msgs.append(f"{k} {H[k].dtype} {H[k].shape} {'OK' if e else 'DIFF'} zero-for-bad={z}")
            print(f"  hires: " + "; ".join(msgs))
    print("ALL IDENTICAL" if all_ok else "DIFFERENCES FOUND")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
