"""DIPSER dataset: in-person student engagement recognition.

Paper: "DIPSER: A Dataset for In-Person Student Engagement Recognition in the Wild" (2025)

5-level individual attention labels -> mapped to 3-level group engagement.
51.3h of video, context cameras at 1280x720 ~9fps.
Pre-extracted pose/landmarks available.
"""

from src.data.video_dataset import VideoDataset

ENGAGEMENT_CLASSES = ["low", "medium", "high"]


class DIPSERDataset(VideoDataset):
    """DIPSER video clip dataset (context cameras)."""

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
