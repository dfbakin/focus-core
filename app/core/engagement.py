"""Engagement model inference.

Loads the TorchScript SLOW R50 model and, for a clip of NUM_FRAMES frames,
returns the probabilities of three engagement classes: low, medium, high.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

MODEL_PATH = Path(__file__).parent.parent / "models" / "engagement_slow_r50.ts"

NUM_FRAMES = 8
RESIZE_SIZE = 256
CROP_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_WEIGHTS = torch.tensor([0.0, 50.0, 100.0])


class EngagementModel:
    """Wrapper around the TorchScript model that accepts raw OpenCV frames."""

    def __init__(self, model_path: Path = MODEL_PATH, device: str = "cpu") -> None:
        """Load the model onto `device`, remapping GPU-trained weights if needed."""
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

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
