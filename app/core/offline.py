"""Choosing clips from a video file that is analysed without real-time playback.

A recorded video does not have to be played at its own speed: frames are
read as fast as the computer allows, and estimates are spaced by video time
instead of wall-clock time. The module is pure Python, so the sampling rule
is tested without OpenCV, Qt or the model.
"""

import math
from collections import deque
from collections.abc import Iterable, Iterator
from typing import TypeVar

DEFAULT_FPS = 25.0
MIN_FRACTION_FOR_FORECAST = 0.02
MIN_SECONDS_FOR_FORECAST = 2.0
COUNTDOWN_TOLERANCE_SEC = 5.0

Frame = TypeVar("Frame")


def usable_fps(reported_fps: float) -> float:
    """Return the frame rate of a file, or DEFAULT_FPS if it is unknown.

    Some containers do not store the frame rate, and OpenCV then reports
    0 or NaN.
    """
    if math.isnan(reported_fps) or reported_fps <= 0:
        return DEFAULT_FPS
    return reported_fps


def frames_between_estimates(interval_sec: float, fps: float) -> int:
    """Return how many frames of video pass between two estimates, at least one."""
    return max(1, round(interval_sec * fps))


def iter_clips(
    frames: Iterable[Frame],
    fps: float,
    interval_sec: float,
    num_frames: int,
    stride: int,
) -> Iterator[tuple[float, list[Frame]]]:
    """Yield (offset_sec, clip) pairs every `interval_sec` of video time.

    A clip is every `stride`-th frame of the last `num_frames * stride`
    frames, the same sampling as during training. The offset is the video
    time at the end of that window. The first clip is yielded as soon as
    the window is filled.
    """
    window: deque[Frame] = deque(maxlen=num_frames * stride)
    step = frames_between_estimates(interval_sec, fps)
    since_last = step
    for index, frame in enumerate(frames):
        window.append(frame)
        since_last += 1
        if len(window) == window.maxlen and since_last >= step:
            since_last = 0
            yield (index + 1) / fps, list(window)[::stride]


def forecast_processing_time(
    elapsed_sec: float, fraction_done: float
) -> tuple[float, float] | None:
    """Predict the total and the remaining time of processing a file, in seconds.

    Assumes the rest of the file is processed at the same speed as the part
    already done. Returns None while too little has been processed for the
    forecast to be stable.
    """
    if fraction_done < MIN_FRACTION_FOR_FORECAST or elapsed_sec < MIN_SECONDS_FOR_FORECAST:
        return None
    total = elapsed_sec / min(fraction_done, 1.0)
    return total, max(0.0, total - elapsed_sec)


def next_countdown_value(shown: float | None, forecast: float) -> float:
    """Return the remaining time to display, so that the countdown goes down.

    The forecast jitters a little from second to second. A higher forecast
    replaces the shown value only if it is higher by more than
    COUNTDOWN_TOLERANCE_SEC, which means processing really slowed down;
    smaller jumps up are ignored.
    """
    if shown is None or forecast <= shown or forecast - shown > COUNTDOWN_TOLERANCE_SEC:
        return forecast
    return shown
