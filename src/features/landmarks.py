"""Facial landmark extraction pipeline.

Uses MediaPipe Face Mesh (468 landmarks) to extract facial features.
Derived features: head pose, Eye Aspect Ratio (EAR), gaze direction.

DIPSER provides pre-extracted facial landmarks.
"""


class LandmarkExtractor:
    """Extracts facial landmarks and derived features from video frames."""

    def __init__(self, max_faces: int = 10):
        raise NotImplementedError

    def extract_frame(self, frame):
        """Extract facial landmarks from a single RGB frame.

        Returns list of detected faces, each with 468 landmarks.
        """
        raise NotImplementedError

    def extract_clip(self, frames):
        """Extract facial landmarks from a sequence of frames."""
        raise NotImplementedError
