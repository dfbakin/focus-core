"""Background threads that read video without freezing the interface."""

import random
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from app.core.engagement import NUM_FRAMES, EngagementModel
from app.core.video_source import FILE, VideoSource

MAX_FAILED_READS = 20
RETRY_DELAY_MS = 100
STOP_TIMEOUT_MS = 1000
PREVIEW_MAX_WIDTH = 960
ANALYSIS_FRAME_HEIGHT = 320
FRAME_STRIDE = 8
ANALYSIS_STOP_TIMEOUT_MS = 3000
PAUSE_SLEEP_MS = 100


class PreviewWorker(QThread):
    """Continuously reads frames from a source and emits them for display.

    A video file is played at its own frame rate and restarts from the
    beginning when it ends. If the source stops delivering frames, for
    example a camera is disconnected, `failed` is emitted and reading stops.

    Signals:
        frame_ready: Emitted with every new BGR frame (numpy array).
        failed: Emitted once when the source stops delivering frames.
    """

    frame_ready = pyqtSignal(object)
    failed = pyqtSignal()

    def __init__(self, source: VideoSource, parent=None) -> None:
        super().__init__(parent)
        self._source = source
        self._running = True

    def stop(self) -> None:
        """Ask the loop to finish and wait until the source is released."""
        self._running = False
        self.wait(STOP_TIMEOUT_MS)

    def run(self) -> None:
        """Thread body: read frames until stopped or the source fails."""
        capture = self._source.open()
        is_file = self._source.kind == FILE
        fps = capture.get(cv2.CAP_PROP_FPS)
        delay_ms = int(1000 / fps) if is_file and fps > 0 else 0
        failed_reads = 0
        try:
            while self._running:
                ok, frame = capture.read()
                if ok:
                    failed_reads = 0
                    self.frame_ready.emit(_shrink(frame))
                    self.msleep(delay_ms)
                    continue

                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    self.failed.emit()
                    return
                if is_file:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    self.msleep(RETRY_DELAY_MS)
        finally:
            capture.release()


def _shrink(frame: np.ndarray) -> np.ndarray:
    """Downscale large frames so that displaying them stays cheap."""
    height, width = frame.shape[:2]
    if width <= PREVIEW_MAX_WIDTH:
        return frame
    scale = PREVIEW_MAX_WIDTH / width
    return cv2.resize(frame, (PREVIEW_MAX_WIDTH, int(height * scale)))


class AnalysisWorker(QThread):
    """Reads the source during a lecture and periodically runs the model.

    Keeps the last NUM_FRAMES * FRAME_STRIDE frames in a buffer. Every
    `interval_sec` seconds it takes every FRAME_STRIDE-th of them, which is
    the same sampling as during training, and emits the engagement score.
    Every frame is also emitted for the live view.

    The model runs in a separate helper thread, so frames keep flowing
    while an estimate is being computed and the live view does not freeze.

    Signals:
        model_ready: Emitted once the model is loaded and the source is open.
        frame_ready: Emitted with every processed BGR frame (numpy array).
        estimate_ready: Emitted with every engagement score 0..100.
        estimate_failed: Emitted when the model could not produce a score.
        source_ended: Emitted when a video file reaches its end.
        failed: Emitted with a message if the model or the source fails.
    """

    model_ready = pyqtSignal()
    frame_ready = pyqtSignal(object)
    estimate_ready = pyqtSignal(float)
    estimate_failed = pyqtSignal()
    source_ended = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        source: VideoSource,
        interval_sec: float,
        simulated_error_rate: float = 0.0,
        parent=None,
    ) -> None:
        """Create the worker.

        Args:
            source: Camera or file to analyse.
            interval_sec: Time between two estimates.
            simulated_error_rate: Share of estimates (0..1) failed on purpose
                to test and demonstrate how gaps are handled.
        """
        super().__init__(parent)
        self._source = source
        self._interval_sec = interval_sec
        self._simulated_error_rate = simulated_error_rate
        self._running = True
        self._paused = False

    def set_paused(self, paused: bool) -> None:
        """Pause or resume frame processing and inference."""
        self._paused = paused

    def stop(self) -> None:
        """Ask the loop to finish and wait until the source is released.

        The wait is longer than for the preview, because an inference
        that has already started must complete first.
        """
        self._running = False
        self.wait(ANALYSIS_STOP_TIMEOUT_MS)

    def run(self) -> None:
        """Thread body: load the model, then read frames and estimate."""
        try:
            model = EngagementModel()
        except (RuntimeError, OSError, ValueError):
            self.failed.emit("Не удалось загрузить модель анализа.")
            return

        capture = self._source.open()
        if not capture.isOpened():
            capture.release()
            self.failed.emit("Не удалось открыть источник видео.")
            return
        self.model_ready.emit()

        executor = ThreadPoolExecutor(max_workers=1)
        try:
            self._process(capture, model, executor)
        finally:
            executor.shutdown(wait=True)
            capture.release()

    def _process(
        self,
        capture: cv2.VideoCapture,
        model: EngagementModel,
        executor: ThreadPoolExecutor,
    ) -> None:
        """Main loop: buffer frames and start an estimate every interval."""
        is_file = self._source.kind == FILE
        fps = capture.get(cv2.CAP_PROP_FPS)
        delay_ms = int(1000 / fps) if is_file and fps > 0 else 0

        frames: deque[np.ndarray] = deque(maxlen=NUM_FRAMES * FRAME_STRIDE)
        next_estimate_at = time.monotonic()
        failed_reads = 0
        pending: Future | None = None

        while self._running:
            if pending is not None and pending.done():
                self._emit_result(pending)
                pending = None

            if self._paused:
                frames.clear()
                if is_file:
                    self.msleep(PAUSE_SLEEP_MS)
                else:
                    capture.read()
                continue

            ok, frame = capture.read()
            if not ok:
                if is_file:
                    if pending is not None:
                        self._emit_result(pending)
                    self.source_ended.emit()
                    return
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    self.failed.emit("Источник видео перестал передавать изображение.")
                    return
                self.msleep(RETRY_DELAY_MS)
                continue

            failed_reads = 0
            small = _resize_to_height(frame, ANALYSIS_FRAME_HEIGHT)
            frames.append(small)
            self.frame_ready.emit(small)

            buffer_full = len(frames) == frames.maxlen
            if buffer_full and pending is None and time.monotonic() >= next_estimate_at:
                clip = list(frames)[::FRAME_STRIDE]
                pending = executor.submit(model.engagement_score, clip)
                next_estimate_at = time.monotonic() + self._interval_sec

            if delay_ms:
                self.msleep(delay_ms)

    def _emit_result(self, finished: Future) -> None:
        """Emit the score of a finished estimate, or report that it failed."""
        if self._paused:
            return
        try:
            score = finished.result()
        except RuntimeError:
            self.estimate_failed.emit()
            return
        if random.random() < self._simulated_error_rate:
            self.estimate_failed.emit()
            return
        self.estimate_ready.emit(score)


def _resize_to_height(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Downscale a frame to the given height keeping its aspect ratio.

    The model was trained on 320p video, so larger frames only cost memory.
    """
    height, width = frame.shape[:2]
    if height <= target_height:
        return frame
    scale = target_height / height
    return cv2.resize(frame, (int(width * scale), target_height))


class TaskWorker(QThread):
    """Runs one function in the background and reports its result.

    Used for requests to the cloud, so the window stays responsive while
    waiting for the server. Any exception is caught and passed to the
    interface, because an exception raised inside a thread would otherwise
    be lost silently.

    Signals:
        succeeded: Emitted with the function's return value.
        failed: Emitted with the exception the function raised.
    """

    succeeded = pyqtSignal(object)
    failed = pyqtSignal(object)

    def __init__(self, function: Callable[[], object], parent=None) -> None:
        super().__init__(parent)
        self._function = function

    def run(self) -> None:
        """Thread body: call the function and emit the outcome."""
        try:
            result = self._function()
        except Exception as exc:
            self.failed.emit(exc)
        else:
            self.succeeded.emit(result)
