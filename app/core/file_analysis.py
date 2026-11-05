"""Analysis of a whole video file without the interface.

Used by the command line tool and the Docker image: reads the file as fast
as possible, estimates engagement every `interval_sec` of video time,
smooths the series and finds critical moments, exactly like a lecture
started from a video file in the application. The model is passed in as a
function, so the pipeline is tested without torch; for the same reason the
clip constants repeat those of app.core.engagement instead of importing it.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.analysis import ExponentialMovingAverage
from app.core.offline import iter_clips, usable_fps
from app.core.report import SessionReport

NUM_FRAMES = 8
FRAME_STRIDE = 8
ANALYSIS_FRAME_HEIGHT = 320

ScoreFunction = Callable[[list[np.ndarray]], float]


class VideoUnreadableError(Exception):
    """The file is missing or cannot be decoded as video."""


@dataclass(frozen=True)
class FileAnalysis:
    """Smoothed engagement series of a video file and its length."""

    points: list[tuple[float, float]]
    duration_sec: float


def analyze_video_file(
    path: str | Path,
    score: ScoreFunction,
    interval_sec: float,
    smoothing_window: int,
    on_progress: Callable[[float, float], None] | None = None,
) -> FileAnalysis:
    """Estimate engagement over the whole file.

    Args:
        path: Video file.
        score: Function that returns an engagement score 0..100 for a clip
            of NUM_FRAMES frames, normally EngagementModel.engagement_score.
        interval_sec: Video time between two estimates.
        smoothing_window: Span of the exponential smoothing.
        on_progress: Called with (position_sec, duration_sec) after every estimate.

    Raises:
        VideoUnreadableError: The file cannot be opened.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoUnreadableError(f"Cannot open video: {path}")
    try:
        fps = usable_fps(capture.get(cv2.CAP_PROP_FPS))
        duration_sec = max(0.0, capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        smoother = ExponentialMovingAverage(smoothing_window)
        points = []
        for offset, clip in iter_clips(
            _read_frames(capture), fps, interval_sec, NUM_FRAMES, FRAME_STRIDE
        ):
            points.append((offset, smoother.add(score(clip))))
            if on_progress is not None:
                on_progress(offset, max(duration_sec, offset))
    finally:
        capture.release()
    return FileAnalysis(points=points, duration_sec=duration_sec)


def build_report(
    analysis: FileAnalysis, threshold_pct: int, started_at: datetime | None = None
) -> SessionReport:
    """Wrap the series into a report, as if the lecture started at `started_at`."""
    return SessionReport(
        session_id=f"file-{datetime.now():%Y%m%d%H%M%S}",
        started_at=started_at or datetime.now(),
        duration_sec=round(analysis.duration_sec),
        threshold_pct=threshold_pct,
        points=analysis.points,
    )


def _read_frames(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
    """Yield frames scaled down to the height the model was trained on."""
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        height, width = frame.shape[:2]
        if height > ANALYSIS_FRAME_HEIGHT:
            scale = ANALYSIS_FRAME_HEIGHT / height
            frame = cv2.resize(frame, (int(width * scale), ANALYSIS_FRAME_HEIGHT))
        yield frame
