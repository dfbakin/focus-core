"""Lightning DataModule for video classification datasets."""

from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from src.data.video_dataset import VideoDataset


class VideoDataModule(L.LightningDataModule):
    """Lightning DataModule wrapping a VideoDataset.

    Handles train/val/test split loading and DataLoader creation.
    """

    def __init__(
        self,
        train_dataset: Optional[VideoDataset] = None,
        val_dataset: Optional[VideoDataset] = None,
        test_dataset: Optional[VideoDataset] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def train_dataloader(self) -> DataLoader:
        assert self._train_dataset is not None
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_dataset is not None
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        assert self._test_dataset is not None
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
