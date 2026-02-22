"""PyTorchVideo model wrappers.

Loads pretrained video models from the PyTorchVideo model hub
and replaces the classification head for the target num_classes.

To add a new PyTorchVideo model:
    1. Write a factory function that returns nn.Module
    2. Decorate it with @register_model("your_model_name")
    3. Add a matching config dataclass in src/conf/model.py
    4. Register the config in src/conf/config.py
"""

import torch.nn as nn

from src.models.factory import register_model


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classification head with the target num_classes.

    PyTorchVideo models from torch.hub typically have their head at
    model.blocks[-1].proj or similar. Adjust as needed per architecture.
    """
    raise NotImplementedError(
        "Implement head replacement logic for your specific PyTorchVideo model. "
        "See pytorchvideo.models documentation for the head structure."
    )


@register_model("slow_r50")
def create_slow_r50(num_classes: int = 3, pretrained: bool = True, **kwargs) -> nn.Module:
    """SLOW ResNet-50 (4fps pathway from SlowFast)."""
    raise NotImplementedError


@register_model("slowfast_r50")
def create_slowfast_r50(num_classes: int = 3, pretrained: bool = True, **kwargs) -> nn.Module:
    """SlowFast ResNet-50 dual-pathway model."""
    raise NotImplementedError


