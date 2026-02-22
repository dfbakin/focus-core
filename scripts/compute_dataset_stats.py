"""Compute per-channel mean and std for a video dataset.

Usage:
    python scripts/compute_dataset_stats.py --root data/ouc_cge --num_samples 500
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute video dataset statistics")
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--num_samples", type=int, default=500)
    args = parser.parse_args()

    raise NotImplementedError(
        "Implement video frame sampling and per-channel mean/std computation."
    )


if __name__ == "__main__":
    main()
