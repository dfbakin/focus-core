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
import torch 
from src.models.factory import register_model


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Replace the final classification head with the target num_classes.

    PyTorchVideo models from torch.hub typically have their head at
    model.blocks[-1].proj or similar. Adjust as needed per architecture.
    """
    new_head = nn.Linear(in_features=model.blocks[-1].proj.in_features, out_features=num_classes, bias=True)
    model.blocks[-1].proj = new_head
    return model


@register_model("slow_r50")
def create_slow_r50(num_classes: int = 3, pretrained: bool = True, **kwargs) -> nn.Module:
    """SLOW ResNet-50 (4fps pathway from SlowFast)."""
    model = _replace_head(model=torch.hub.load(repo_or_dir='facebookresearch/pytorchvideo', model='slow_r50', pretrained=pretrained), num_classes=num_classes)
    return model
    


@register_model("slowfast_r50")
def create_slowfast_r50(num_classes: int = 3, pretrained: bool = True, **kwargs) -> nn.Module:
    """SlowFast ResNet-50 dual-pathway model."""
    raise NotImplementedError


if __name__ == "__main__":
    model = create_slow_r50()
    print(model.blocks[-1].proj)