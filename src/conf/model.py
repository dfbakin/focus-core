from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Base model configuration."""

    name: str = "base"
    num_classes: int = 3
    pretrained: bool = True
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


@dataclass
class SlowConfig(ModelConfig):
    name: str = "slow_r50"
    sample_rate: int = 8
    num_frames: int = 8


@dataclass
class SlowFastConfig(ModelConfig):
    name: str = "slowfast_r50"
    slow_sample_rate: int = 8
    fast_sample_rate: int = 2
    slow_num_frames: int = 8
    fast_num_frames: int = 32


