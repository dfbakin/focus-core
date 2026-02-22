"""Body pose extraction pipeline.

Uses MediaPipe BlazePose (33 landmarks) to extract body pose keypoints
from video frames. Output can be used as input to GCN or sequence models.

DIPSER provides pre-extracted pose data that can bypass this pipeline.
"""


class PoseExtractor:
    """Extracts body pose keypoints from video frames."""

    def __init__(self, model_complexity: int = 1):
        raise NotImplementedError

    def extract_frame(self, frame):
        """Extract pose landmarks from a single RGB frame.

        Returns array of shape (33, 3) for x, y, visibility per landmark,
        or None if no person detected.
        """
        raise NotImplementedError

    def extract_clip(self, frames):
        """Extract pose landmarks from a sequence of frames.

        Returns array of shape (T, num_persons, 33, 3).
        """
        raise NotImplementedError
