"""End-to-end inference pipeline (Phase 6).

Processes raw video input through the full pipeline:
    Video -> clip extraction -> (optional: pose/landmark extraction) -> model -> predictions
"""


class InferencePipeline:
    """Run engagement prediction on raw video files."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        raise NotImplementedError

    def predict_video(self, video_path: str) -> list[dict]:
        """Predict engagement levels for all clips in a video.

        Returns list of dicts with "start_sec", "end_sec", "label", "confidence".
        """
        raise NotImplementedError
