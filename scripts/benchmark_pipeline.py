"""Benchmark the data pipeline vs GPU throughput, and probe VRAM headroom.

Answers three questions without launching real training:

  1. How fast is on-the-fly clip decoding (single worker)?
  2. With a DataLoader (N workers), can the input pipeline feed the GPU,
     i.e. is decoding the bottleneck?
  3. What batch size fits in the available VRAM for a fwd+bwd step?

The dataset's __init__ normally scans the *entire* split (slow), so here we
subclass it to clean only a small subset for benchmarking.

Usage:
    poetry run python scripts/benchmark_pipeline.py
    poetry run python scripts/benchmark_pipeline.py --subset 64 --batch-size 8 \
        --num-workers 4 --model slow_r50
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader

from src.data.ouc_cge import OUCCGEDataset
from src.data.transforms import get_train_transforms
from src.models.backbones import *  # noqa: F401,F403 — register models
from src.models.factory import create_model


class SubsetOUCCGE(OUCCGEDataset):
    """OUC-CGE that validates only enough clips to gather `bench_n` valid ones.

    Avoids the full-split scan (slow) while still skipping short/corrupt clips
    so the benchmark measures realistic decode cost on usable clips.
    """

    bench_n = 64

    def _cleaning_dataset(self, window_size: int = 64, use_cache: bool = False):
        import cv2
        keep = []
        for idx in range(len(self.sample)):
            if len(keep) >= self.bench_n:
                break
            vp = self.root / str(self.sample.iloc[idx, 0])
            cap = cv2.VideoCapture(str(vp))
            ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= window_size
            cap.release()
            if ok:
                keep.append(idx)
        self.sample = self.sample.iloc[keep].reset_index(drop=True)


def fmt(x):
    return f"{x:,.1f}"


def benchmark_raw(ds, n):
    """Single-worker raw __getitem__ throughput."""
    n = min(n, len(ds))
    t0 = time.perf_counter()
    n_frames = 0
    for i in range(n):
        sample = ds[i]
        for k, v in sample.items():
            if k.startswith("flow_num_"):
                n_frames += v.shape[1] if v.ndim == 4 else v.shape[0]
    dt = time.perf_counter() - t0
    print(f"[raw __getitem__]   {n} samples in {fmt(dt)}s -> "
          f"{fmt(n / dt)} samples/s, {fmt(n_frames / dt)} decoded-frames/s "
          f"(single worker)")
    return n / dt


def benchmark_loader(ds, batch_size, num_workers, max_batches):
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    it = iter(loader)
    # warm up one batch (spawns workers / fills prefetch)
    next(it)
    t0 = time.perf_counter()
    n = 0
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        n += batch["flow_num_0"].shape[0]
    dt = time.perf_counter() - t0
    sps = n / dt if dt > 0 else 0
    print(f"[DataLoader x{num_workers}] {n} samples in {fmt(dt)}s -> "
          f"{fmt(sps)} samples/s (batch_size={batch_size})")
    del loader, it
    return sps


def benchmark_gpu(model_name, batch_sizes, num_frames, device):
    """Fwd+bwd throughput + peak VRAM per batch size."""
    if device == "cuda" and not torch.cuda.is_available():
        print("[gpu] CUDA not available, skipping.")
        return
    model = create_model(name=model_name, num_classes=3, pretrained=False).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"\n[gpu] model={model_name} frames={num_frames} device={device}")
    for bs in batch_sizes:
        try:
            if device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, 3, num_frames, 224, 224, device=device)
            y = torch.randint(0, 3, (bs,), device=device)
            # warmup
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            iters = 5
            for _ in range(iters):
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            sps = bs * iters / dt
            peak = (torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0
            print(f"   bs={bs:>2}: {fmt(sps)} samples/s  "
                  f"step={fmt(1000*dt/iters)} ms  peak_VRAM={peak:.2f} GB  -> OK")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"   bs={bs:>2}: OOM ✗")
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
    del model, opt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/ouc-cge")
    p.add_argument("--subset", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--model", default="slow_r50")
    p.add_argument("--num-frames", type=int, default=8, help="frames for GPU bench")
    args = p.parse_args()

    SubsetOUCCGE.bench_n = args.subset
    print(f"Building subset dataset ({args.subset} clips) ...")
    ds = SubsetOUCCGE(root=args.root, split="train", transform=get_train_transforms())
    print(f"Dataset ready: {len(ds)} clips "
          f"(decodes {len(ds.num_frames)} pathways: num_frames={ds.num_frames}, "
          f"rates={ds.list_of_rates})\n")

    raw_sps = benchmark_raw(ds, n=min(24, len(ds)))
    print()
    for nw in sorted({0, args.num_workers}):
        benchmark_loader(ds, args.batch_size, nw, args.max_batches)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark_gpu(args.model, [2, 4, 6, 8, 12], args.num_frames, dev)


if __name__ == "__main__":
    main()
