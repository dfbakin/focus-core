"""Base LightningModule for video classification."""

from typing import Any

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class VideoClassificationModule(L.LightningModule):
    """Lightning module for video classification.

    Wraps any nn.Module backbone that accepts video tensors (B, C, T, H, W)
    and returns logits (B, num_classes).
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        metrics_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.val_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.val_f1 = torchmetrics.F1Score(**metrics_kwargs, average="macro")
        self.test_acc = torchmetrics.Accuracy(**metrics_kwargs)
        self.test_f1 = torchmetrics.F1Score(**metrics_kwargs, average="macro")

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video, labels = batch["flow_num_0"], batch["label"]
        logits = self(video)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy", self.val_acc, prog_bar=True, on_epoch=True)
        self.log("val/f1_macro", self.val_f1, on_epoch=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test/loss", loss, on_epoch=True)
        self.log("test/accuracy", self.test_acc, on_epoch=True)
        self.log("test/f1_macro", self.test_f1, on_epoch=True)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 50,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
