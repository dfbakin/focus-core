"""Custom Lightning callbacks for MLflow logging and early stopping."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import Callback


class MLflowModelCheckpoint(Callback):
    """Logs checkpoint paths and best metrics to MLflow.

    Only logs paths (not the heavy checkpoint files themselves).
    Runs only on rank 0 for multi-GPU compatibility.
    """

    def __init__(self, monitor: str = "val/accuracy", mode: str = "max") -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")

    def _get_mlflow_logger(self, trainer: L.Trainer):
        from lightning.pytorch.loggers import MLFlowLogger

        if trainer.logger is None:
            return None
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        if hasattr(trainer.logger, "_loggers"):
            for logger in trainer.logger._loggers:
                if isinstance(logger, MLFlowLogger):
                    return logger
        return None

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return

        from mlflow import MlflowClient

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_best = (self.mode == "max" and current > self.best_value) or (
            self.mode == "min" and current < self.best_value
        )
        if not is_best:
            return

        self.best_value = current.item()

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return

        client = MlflowClient(tracking_uri=mlflow_logger._tracking_uri)
        client.log_metric(mlflow_logger.run_id, f"best_{self.monitor}", self.best_value)
        client.log_metric(mlflow_logger.run_id, "best_epoch", trainer.current_epoch)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        import mlflow
        from mlflow import MlflowClient

        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None or mlflow_logger.run_id is None:
            return

        run_id = mlflow_logger.run_id
        client = MlflowClient(tracking_uri=mlflow_logger._tracking_uri)

        client.log_metric(run_id, "final_epoch", trainer.current_epoch)

        ckpt_callback = trainer.checkpoint_callback
        if ckpt_callback is not None:
            if ckpt_callback.best_model_path:
                client.log_param(
                    run_id, "best_checkpoint_path", ckpt_callback.best_model_path
                )
            if ckpt_callback.last_model_path:
                client.log_param(
                    run_id, "last_checkpoint_path", ckpt_callback.last_model_path
                )

        client.log_param(run_id, "output_dir", str(trainer.default_root_dir))

        with mlflow.start_run(run_id=run_id):
            hydra_dir = Path(trainer.default_root_dir) / ".hydra"
            if hydra_dir.exists():
                mlflow.log_artifacts(str(hydra_dir), artifact_path="hydra_config")


class EarlyStoppingWithPatience(Callback):
    """Early stopping that waits for min_epochs before activating."""

    def __init__(
        self,
        monitor: str = "val/accuracy",
        patience: int = 10,
        min_epochs: int = 10,
        mode: str = "max",
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_epochs = min_epochs
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.wait_count = 0

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if trainer.current_epoch < self.min_epochs:
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        is_improvement = (self.mode == "max" and current > self.best_value) or (
            self.mode == "min" and current < self.best_value
        )

        if is_improvement:
            self.best_value = current.item()
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                trainer.should_stop = True
