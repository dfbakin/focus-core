"""Compute per-channel mean and std for a video dataset.

Usage:
    python scripts/compute_dataset_stats.py --root data/ouc_cge --num_samples 500
"""

import argparse
from torchinfo import summary
import src.models.backbones.pytorchvideo
from src.models.factory import create_model

def main():
    parser = argparse.ArgumentParser(description="Compute video dataset statistics")
    parser.add_argument("--model", type=str, default="slow_r50")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--frames_count", type=int, default=8)
    args = parser.parse_args()

    model = create_model(name=args.model)
    summary(model, input_size=(args.batch_size, 3, args.frames_count, 224, 224))

if __name__ == "__main__":
    main()
