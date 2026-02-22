"""Model factory with registry pattern.

Register new models by adding them to MODEL_REGISTRY.
"""

from typing import Callable

import torch.nn as nn

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model constructor."""

    def wrapper(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        MODEL_REGISTRY[name] = fn
        return fn

    return wrapper


def create_model(name: str, num_classes: int = 3, pretrained: bool = True, **kwargs) -> nn.Module:
    """Create a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys()) or "(none registered)"
        raise ValueError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained, **kwargs)
