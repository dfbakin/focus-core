"""Video transform pipelines.

Provides spatial and temporal transforms for video clips.
Uses torchvision transforms; can also integrate PyTorchVideo transforms.
"""

from torchvision.transforms import Compose
from torchvision.transforms import v2


def get_train_transforms(image_size: int = 224) -> Compose:
    """Spatial transforms applied to each frame during training."""
    return Compose([
        v2.Resize(256),
        v2.RandomCrop(size=[image_size, image_size]),
        v2.RandomHorizontalFlip(0.5),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 


def get_val_transforms(image_size: int = 224) -> Compose:
    """Spatial transforms applied to each frame during validation/test."""
    return Compose([
        v2.Resize(256),
        v2.CenterCrop(size=[image_size, image_size]),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
