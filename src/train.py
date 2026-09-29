"""Training entry point.

Examples:
    python -m src.train                                # defaults (SLOW on OUC-CGE)
    python -m src.train model=slowfast                 # SlowFast
    python -m src.train data=dipser model=i3d trainer.max_epochs=30
    python -m src.train --multirun model=slow,slowfast,i3d,x3d
"""

import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from src.data.ouc_cge import OUCCGEDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.datamodule import VideoDataModule
from src.conf.config import Config, register_configs
from src.models import VideoClassificationModule, create_model
from src.models.backbones import *  # noqa: F401,F403 — trigger model registration
from src.training.callbacks import EarlyStoppingWithPatience, MLflowModelCheckpoint

log = logging.getLogger(__name__)

register_configs()


def _pathways_for_model(cfg: Config) -> tuple[list[float], list[int]]:
    """Map the model config to (fps, num_frames) temporal pathways.

    SlowFast needs two pathways (slow + fast); single-pathway models (SLOW,
    I3D, X3D, C2D, ...) need one. Decoding only the required pathways avoids
    wasting I/O on frames the model never sees.
    """
    name = cfg.model.name
    if name == "slowfast_r50":
        # slow: 8 frames @ stride 8 (~3.75 fps); fast: 32 frames @ stride 2 (~15 fps)
        return [3.75, 15.0], [8, 32]
    if name == "resnet50":
        return [30], [1] #chosing 1 frame from video randomly (just our startpoint)
    # Default single SLOW pathway: 8 frames @ stride 8.
    return [3.75], [8]


def build_datamodule(cfg: Config):
    """Build the Lightning DataModule from config."""
    fps, num_frames = _pathways_for_model(cfg)

    train_ds = OUCCGEDataset(
        root=cfg.data.root, split="train", fps=fps, num_frames=num_frames,
        transform=get_train_transforms(),
    )
    val_ds = OUCCGEDataset(
        root=cfg.data.root, split="val", fps=fps, num_frames=num_frames,
        transform=get_val_transforms(),
    )
    test_ds = OUCCGEDataset(
        root=cfg.data.root, split="test", fps=fps, num_frames=num_frames,
        transform=get_val_transforms(),
    )

    return VideoDataModule(
        train_dataset=train_ds, val_dataset=val_ds, test_dataset=test_ds,
        batch_size=cfg.data.batch_size, num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )


def train(cfg: Config) -> float:
    """Run training. Returns best val accuracy."""
    L.seed_everything(cfg.experiment.seed, workers=True)
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = build_datamodule(cfg)

    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.data.num_classes,
        pretrained=cfg.model.pretrained,
    )

    module = VideoClassificationModule(
        model=model_backbone,
        num_classes=cfg.data.num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    model_choice = HydraConfig.get().runtime.choices.get("model", cfg.model.name)
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.mlflow.experiment_name,
        tracking_uri=cfg.mlflow.tracking_uri,
        run_name=f"{model_choice}_{cfg.data.name}_{cfg.experiment.seed}",
        tags={
            "model": cfg.model.name,
            "dataset": cfg.data.name,
            "num_classes": str(cfg.data.num_classes),
        },
    )

    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    mlflow_logger.experiment.log_artifact(mlflow_logger.run_id, str(config_path))

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.paths.checkpoints,
            filename="{epoch:02d}-{val/accuracy:.4f}",
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
            save_top_k=cfg.trainer.checkpoint.save_top_k,
            save_last=cfg.trainer.checkpoint.save_last,
        ),
        EarlyStoppingWithPatience(
            monitor=cfg.trainer.early_stopping.monitor,
            patience=cfg.trainer.early_stopping.patience,
            mode=cfg.trainer.early_stopping.mode,
            min_epochs=cfg.trainer.early_stopping.min_epochs,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        MLflowModelCheckpoint(
            monitor=cfg.trainer.checkpoint.monitor,
            mode=cfg.trainer.checkpoint.mode,
        ),
    ]

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        deterministic=cfg.trainer.deterministic,
        limit_train_batches=cfg.trainer.get("limit_train_batches", None),
        logger=mlflow_logger,
        callbacks=callbacks,
        default_root_dir=str(output_dir),
    )

    log.info("Starting training...")
    trainer.fit(module, datamodule=datamodule)

    log.info("Running test evaluation...")
    trainer.test(module, datamodule=datamodule, ckpt_path="best")

    return float(trainer.callback_metrics.get("val/accuracy", torch.tensor(0.0)))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    return train(cfg)


if __name__ == "__main__":
    main()
