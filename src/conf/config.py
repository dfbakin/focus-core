from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.conf.data import OUCCGEDataConfig, DIPSERDataConfig
from src.conf.model import SlowConfig, SlowFastConfig, ResNet50Config
from src.conf.trainer import TrainerConfig


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    seed: int = 42


@dataclass
class MLflowConfig:
    tracking_uri: Optional[str] = None
    experiment_name: str = "${experiment.name}"


@dataclass
class PathsConfig:
    output_dir: str = "outputs/${experiment.name}/${now:%Y-%m-%d_%H-%M-%S}"
    checkpoints: str = "${paths.output_dir}/checkpoints"
    logs: str = "${paths.output_dir}/logs"


@dataclass
class Config:
    """Root configuration."""

    defaults: list[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"data": "ouc_cge"},
            {"model": "slow"},
            {"trainer": "default"},
        ]
    )

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    data: Any = MISSING
    model: Any = MISSING
    trainer: Any = MISSING

    checkpoint_path: Optional[str] = None


def register_configs() -> None:
    """Register all structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()

    cs.store(name="config", node=Config)

    cs.store(group="data", name="ouc_cge", node=OUCCGEDataConfig)
    cs.store(group="data", name="dipser", node=DIPSERDataConfig)

    cs.store(group="model", name="slow", node=SlowConfig)
    cs.store(group="model", name="slowfast", node=SlowFastConfig)
    cs.store(group="model", name="resnet50", node=ResNet50Config)

    cs.store(group="trainer", name="default", node=TrainerConfig)
