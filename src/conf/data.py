from dataclasses import dataclass


@dataclass
class DataConfig:
    """Base data configuration for video datasets."""

    name: str = "base"
    root: str = "data"
    num_classes: int = 3
    clip_duration_sec: float = 10.0
    fps: int = 30
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class OUCCGEDataConfig(DataConfig):
    name: str = "ouc_cge"
    root: str = "data/ouc-cge"
    num_classes: int = 3
    clip_duration_sec: float = 10.0
    fps: int = 30


@dataclass
class DIPSERDataConfig(DataConfig):
    name: str = "dipser"
    root: str = "data/dipser"
    num_classes: int = 3
    clip_duration_sec: float = 10.0
    fps: int = 9
