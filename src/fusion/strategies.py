"""Multimodal fusion strategies (Phase 4).

Combine video, pose, and landmark modalities.
Strategies: early fusion, late fusion, attention-based fusion.
"""

import torch.nn as nn


class EarlyFusion(nn.Module):
    """Concatenate features before temporal modeling."""

    def __init__(self):
        raise NotImplementedError

    def forward(self, **modality_features):
        raise NotImplementedError


class LateFusion(nn.Module):
    """Combine predictions from independent per-modality models."""

    def __init__(self):
        raise NotImplementedError

    def forward(self, **modality_logits):
        raise NotImplementedError


class AttentionFusion(nn.Module):
    """Learned attention-based modality weighting."""

    def __init__(self):
        raise NotImplementedError

    def forward(self, **modality_features):
        raise NotImplementedError
