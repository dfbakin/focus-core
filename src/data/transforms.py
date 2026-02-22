"""Video transform pipelines.

Provides spatial and temporal transforms for video clips.
Uses torchvision transforms; can also integrate PyTorchVideo transforms.
"""

from torchvision.transforms import Compose, Normalize, Resize


def get_train_transforms(image_size: int = 224) -> Compose:
    """Spatial transforms applied to each frame during training."""
    raise NotImplementedError


def get_val_transforms(image_size: int = 224) -> Compose:
    """Spatial transforms applied to each frame during validation/test."""
    raise NotImplementedError
