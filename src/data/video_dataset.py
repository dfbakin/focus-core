"""Base video dataset interface.

All dataset implementations (OUC-CGE, DIPSER) should subclass VideoDataset
and implement the abstract methods.
"""

from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class VideoDataset(Dataset, ABC):
    """Abstract base class for video clip datasets.

    Each sample is a dict with at least:
        - "video": Tensor of shape (C, T, H, W)
        - "label": int engagement level
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict: ...

    @property
    @abstractmethod
    def num_classes(self) -> int: ...

    @property
    @abstractmethod
    def class_names(self) -> list[str]: ...
