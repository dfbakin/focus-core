"""Engagement model inference.

Loads the TorchScript SLOW R50 model and, for a clip of NUM_FRAMES frames,
returns the probabilities of three engagement classes: low, medium, high.

PyTorch does not implement 3D max pooling for the Apple GPU (MPS) yet, and
the model has one such layer. PYTORCH_ENABLE_MPS_FALLBACK lets that single
layer run on the CPU while the rest of the model runs on the GPU. PyTorch
reads the variable only when it is imported, so it is set before the import.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402

MODEL_PATH = Path(__file__).parent.parent / "models" / "engagement_slow_r50.ts"

NUM_FRAMES = 8
RESIZE_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_WEIGHTS = torch.tensor([0.0, 50.0, 100.0])


def candidate_devices() -> list[str]:
    """Return the devices to try, fastest first.

    On Apple Silicon the GPU is available through MPS and is usually several
    times faster than the CPU; the CPU is always the last resort.
    """
    if torch.backends.mps.is_available():
        return ["mps", "cpu"]
    return ["cpu"]


class EngagementModel:
    """Wrapper around the TorchScript model that accepts raw OpenCV frames.

    Attributes:
        device: Device the model actually runs on.
    """

    def __init__(self, model_path: Path = MODEL_PATH, device: str | None = None) -> None:
        """Load the model and make sure it runs.

        Without an explicit `device` the fastest available one is tried first.
        If the model cannot run there, for example an operation is not
        supported by MPS, the next device is used.

        Raises:
            RuntimeError: The model could not run on any device.
        """
        devices = [device] if device else candidate_devices()
        error: RuntimeError | None = None
        for name in devices:
            try:
                self._load(model_path, name)
                return
            except RuntimeError as exc:
                error = exc
        raise error

    def _load(self, model_path: Path, device: str) -> None:
        """Load the weights onto `device`, remapping GPU-trained weights if needed."""
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        self._warm_up()

    def _warm_up(self) -> None:
        """Run one empty clip.

        The first run is much slower than the next ones, so it is better to
        pay for it before the lecture starts. It also reveals early that the
        device cannot run the model.
        """
        empty = torch.zeros(1, 3, NUM_FRAMES, CROP_SIZE, CROP_SIZE, device=self.device)
        with torch.no_grad():
            self.model(empty)

    def _preprocess_clip(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """Convert OpenCV frames (BGR, HxWx3, uint8) into a [1, C, T, H, W] tensor.

        The transforms must match the ones used during training.
        """
        if len(frames_bgr) != NUM_FRAMES:
            raise ValueError(f"Expected {NUM_FRAMES} frames, got {len(frames_bgr)}")

        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        clip = np.ascontiguousarray(np.stack(rgb))

        t = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 255.0
        t = TF.resize(t, RESIZE_SIZE, antialias=True)
        t = TF.center_crop(t, [CROP_SIZE, CROP_SIZE])
        t = TF.normalize(t, MEAN, STD)
        t = t.permute(1, 0, 2, 3).unsqueeze(0)
        return t.to(self.device)

    def predict(self, frames_bgr: list[np.ndarray]) -> torch.Tensor:
        """Return class probabilities [low, mid, high] as a tensor of shape [3]."""
        clip = self._preprocess_clip(frames_bgr)
        with torch.no_grad():
            logits = self.model(clip)
        return torch.softmax(logits, dim=1)[0].cpu()

    def engagement_score(self, frames_bgr: list[np.ndarray]) -> float:
        """Return an engagement score 0..100: the expectation over CLASS_WEIGHTS."""
        probs = self.predict(frames_bgr)
        return float((probs * CLASS_WEIGHTS).sum())
