from dataclasses import dataclass, field


@dataclass
class CheckpointConfig:
    monitor: str = "val/accuracy"
    mode: str = "max"
    save_top_k: int = 2
    save_last: bool = True


@dataclass
class EarlyStoppingConfig:
    monitor: str = "val/accuracy"
    patience: int = 15
    mode: str = "max"
    min_epochs: int = 10


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    min_epochs: int = 5

    accelerator: str = "auto"
    devices: str = "auto"
    strategy: str = "auto"
    precision: str = "32"

    accumulate_grad_batches: int = 1

    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    log_every_n_steps: int = 10
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1

    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = "norm"

    deterministic: bool = False
