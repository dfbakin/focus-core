"""OUC-CGE dataset: classroom group engagement recognition.

Dataset: https://osf.io/brd2c/
Paper: "A Video Dataset for Classroom Group Engagement Recognition" (Lu et al., 2025)

3 engagement levels: Low (0), Medium (1), High (2)
7,705 segments of 10-second clips, 1280x720, 30fps
"""

from src.data.video_dataset import VideoDataset

ENGAGEMENT_CLASSES = ["low", "medium", "high"]


class OUCCGEDataset(VideoDataset):
    """OUC-CGE video clip dataset."""

    def __init__(self, root: str, split: str = "train", transform=None):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        return 3

    @property
    def class_names(self) -> list[str]:
        return ENGAGEMENT_CLASSES
