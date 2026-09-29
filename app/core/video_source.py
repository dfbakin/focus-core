"""Video sources: camera discovery and signal check.

Uses only OpenCV, so the logic can be tested without the GUI.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from cv2_enumerate_cameras import enumerate_cameras

CAMERA = "camera"
FILE = "file"
CAMERA_WARMUP_FRAMES = 5


class VideoSourceError(Exception):
    """Base error for a video source that cannot deliver frames."""


class CameraUnavailableError(VideoSourceError):
    """The camera cannot be opened or returns no frames, e.g. it is busy."""


class FileUnreadableError(VideoSourceError):
    """The file is missing, corrupted or has an unsupported format."""


@dataclass(frozen=True)
class VideoSource:
    """Where frames come from: a camera or a path to a video file.

    For a camera, `camera_index` is what OpenCV needs to open it and
    `camera_name` is what the user sees and what identifies it reliably,
    because indices shift when devices are plugged in or out.
    """

    kind: str
    camera_index: int = 0
    camera_name: str = ""
    file_path: str = ""

    @property
    def display_name(self) -> str:
        """Name shown to the user: the camera name or the file name."""
        if self.kind == CAMERA:
            return self.camera_name or f"Камера {self.camera_index}"
        return Path(self.file_path).name

    def open(self) -> cv2.VideoCapture:
        """Return an OpenCV capture for this source; it may fail to open."""
        if self.kind == CAMERA:
            return cv2.VideoCapture(self.camera_index)
        return cv2.VideoCapture(self.file_path)


@dataclass(frozen=True)
class CameraDevice:
    """A camera known to the operating system."""

    index: int
    name: str


@dataclass(frozen=True)
class SignalInfo:
    """Result of a successful signal check."""

    frame: np.ndarray
    width: int
    height: int
    fps: float


def _camera_backend() -> int:
    """Return the OpenCV camera backend native to the current OS."""
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    if sys.platform.startswith("win"):
        return cv2.CAP_MSMF
    return cv2.CAP_V4L2


def find_cameras() -> list[CameraDevice]:
    """Return the cameras registered in the operating system with their names.

    The returned index already encodes the backend, so it can be passed
    directly to cv2.VideoCapture.
    """
    return [
        CameraDevice(index=info.index, name=info.name)
        for info in enumerate_cameras(_camera_backend())
    ]


def check_signal(source: VideoSource) -> SignalInfo:
    """Open the source, read one frame and describe the stream.

    Raises:
        CameraUnavailableError: The camera does not deliver frames.
        FileUnreadableError: The file cannot be decoded as video.
    """
    is_camera = source.kind == CAMERA
    error = CameraUnavailableError if is_camera else FileUnreadableError

    if not is_camera and not Path(source.file_path).is_file():
        raise FileUnreadableError(f"File not found: {source.file_path}")

    capture = source.open()
    try:
        if not capture.isOpened():
            raise error(f"Cannot open source: {source}")
        attempts = CAMERA_WARMUP_FRAMES if is_camera else 1
        frame = _read_last_frame(capture, attempts)
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if frame is None:
        raise error(f"Source returned no frames: {source}")
    height, width = frame.shape[:2]
    return SignalInfo(frame=frame, width=width, height=height, fps=fps)


def _read_last_frame(capture: cv2.VideoCapture, attempts: int) -> np.ndarray | None:
    """Read up to `attempts` frames and return the last successful one.

    Cameras often return dark frames right after opening, so several frames
    are read to let the exposure settle.
    """
    frame = None
    for _ in range(attempts):
        ok, current = capture.read()
        if ok:
            frame = current
    return frame
