"""Evaluation entry point.

Examples:
    python -m src.evaluate +checkpoint_path=outputs/.../best.ckpt model=slow
"""

import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from src.conf.config import Config, register_configs
from src.models import VideoClassificationModule, create_model
from src.models.backbones import *  # noqa: F401,F403

log = logging.getLogger(__name__)

register_configs()


def evaluate(cfg: Config) -> dict[str, float]:
    """Evaluate checkpoint on test set. Returns test metrics dict."""
    L.seed_everything(cfg.experiment.seed, workers=True)

    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided via +checkpoint_path=...")

    log.info(f"Loading checkpoint: {checkpoint_path}")

    model_backbone = create_model(
        name=cfg.model.name,
        num_classes=cfg.data.num_classes,
        pretrained=False,
    )

    module = VideoClassificationModule.load_from_checkpoint(
        checkpoint_path, model=model_backbone
    )

    # TODO: build datamodule from config (same as in train.py)
    datamodule = None
    raise NotImplementedError("Wire up dataset loading before evaluating.")

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
    )

    results = trainer.test(module, datamodule=datamodule)
    return results[0] if results else {}


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig) -> None:
    results = evaluate(cfg)
    log.info(f"Test Results:\n{OmegaConf.to_yaml(results)}")


if __name__ == "__main__":
    main()
