"""Background threads that read video without freezing the interface."""

import random
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtBoundSignal, pyqtSignal

from app.core.engagement import NUM_FRAMES, EngagementModel
from app.core.offline import iter_clips, usable_fps
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


class CameraAnalysisWorker(QThread):
    """Reads a camera during a lecture and periodically runs the model.

    Keeps the last NUM_FRAMES * FRAME_STRIDE frames in a buffer. Every
    `interval_sec` seconds it takes every FRAME_STRIDE-th of them, which is
    the same sampling as during training, and emits the engagement score.
    Every frame is also emitted for the live view.

    The model runs in a separate helper thread, so frames keep flowing
    while an estimate is being computed and the live view does not freeze.
    If an estimate takes longer than the interval, the next one starts
    right after it, so the real rate is limited by the model speed.

    Signals:
        model_ready: Emitted once the model is loaded and the camera is open.
        frame_ready: Emitted with every processed BGR frame (numpy array).
        estimate_ready: Emitted with every engagement score 0..100.
        estimate_failed: Emitted when the model could not produce a score.
        failed: Emitted with a message if the model or the camera fails.
    """

    model_ready = pyqtSignal()
    frame_ready = pyqtSignal(object)
    estimate_ready = pyqtSignal(float)
    estimate_failed = pyqtSignal()
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
            source: Camera to analyse.
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
        """Ask the loop to finish and wait until the camera is released.

        The wait is longer than for the preview, because an inference
        that has already started must complete first.
        """
        self._running = False
        self.wait(ANALYSIS_STOP_TIMEOUT_MS)

    def run(self) -> None:
        """Thread body: load the model, then read frames and estimate."""
        model = _load_model(self.failed)
        if model is None:
            return
        capture = _open_capture(self._source, self.failed)
        if capture is None:
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
                capture.read()
                continue

            ok, frame = capture.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= MAX_FAILED_READS:
                    self.failed.emit("Камера перестала передавать изображение.")
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

    def _emit_result(self, finished: Future) -> None:
        """Emit the score of a finished estimate, or report that it failed."""
        if self._paused:
            return
        try:
            score = finished.result()
        except RuntimeError:
            self.estimate_failed.emit()
            return
        if _simulate_failure(self._simulated_error_rate):
            self.estimate_failed.emit()
            return
        self.estimate_ready.emit(score)


class FileAnalysisWorker(QThread):
    """Analyses a whole video file as fast as the computer allows.

    Frames are read without waiting for the video's own frame rate, and
    estimates are spaced by video time (see app.core.offline), so the speed
    depends only on the model and the chosen interval, and the report is
    ready right after. Only the frame of each estimate is emitted for
    display, so the interface is not flooded with frames.

    Signals:
        model_ready: Emitted once the model is loaded and the file is open.
        frame_ready: Emitted with the last frame of every analysed clip.
        estimate_ready: Emitted with (offset_sec, score) for every estimate;
            the offset is the position in the video.
        estimate_failed: Emitted with offset_sec when an estimate failed.
        progress: Emitted with (position_sec, duration_sec) of the video.
        source_ended: Emitted when the whole file has been analysed.
        failed: Emitted with a message if the model or the file fails.
    """

    model_ready = pyqtSignal()
    frame_ready = pyqtSignal(object)
    estimate_ready = pyqtSignal(float, float)
    estimate_failed = pyqtSignal(float)
    progress = pyqtSignal(float, float)
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
            source: Video file to analyse.
            interval_sec: Video time between two estimates.
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
        """Pause or resume reading the file."""
        self._paused = paused

    def stop(self) -> None:
        """Ask the loop to finish and wait until the file is released."""
        self._running = False
        self.wait(ANALYSIS_STOP_TIMEOUT_MS)

    def run(self) -> None:
        """Thread body: load the model, then analyse the file to its end."""
        model = _load_model(self.failed)
        if model is None:
            return
        capture = _open_capture(self._source, self.failed)
        if capture is None:
            return
        self.model_ready.emit()

        try:
            finished = self._process(capture, model)
        finally:
            capture.release()
        if finished:
            self.source_ended.emit()

    def _process(self, capture: cv2.VideoCapture, model: EngagementModel) -> bool:
        """Estimate every clip of the file; return False if stopped early."""
        fps = usable_fps(capture.get(cv2.CAP_PROP_FPS))
        duration_sec = max(0.0, capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
        clips = iter_clips(
            self._read_frames(capture), fps, self._interval_sec, NUM_FRAMES, FRAME_STRIDE
        )
        for offset, clip in clips:
            if not self._running:
                return False
            self.frame_ready.emit(clip[-1])
            try:
                score = model.engagement_score(clip)
            except RuntimeError:
                self.estimate_failed.emit(offset)
            else:
                if _simulate_failure(self._simulated_error_rate):
                    self.estimate_failed.emit(offset)
                else:
                    self.estimate_ready.emit(offset, score)
            self.progress.emit(offset, max(duration_sec, offset))
        return self._running

    def _read_frames(self, capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
        """Yield downscaled frames until the file ends or the worker stops."""
        while self._running:
            if self._paused:
                self.msleep(PAUSE_SLEEP_MS)
                continue
            ok, frame = capture.read()
            if not ok:
                return
            yield _resize_to_height(frame, ANALYSIS_FRAME_HEIGHT)


def _load_model(failed: pyqtBoundSignal) -> EngagementModel | None:
    """Load the model, or emit `failed` with a message and return None."""
    try:
        return EngagementModel()
    except (RuntimeError, OSError, ValueError):
        failed.emit("Не удалось загрузить модель анализа.")
        return None


def _open_capture(source: VideoSource, failed: pyqtBoundSignal) -> cv2.VideoCapture | None:
    """Open the source, or emit `failed` with a message and return None."""
    capture = source.open()
    if capture.isOpened():
        return capture
    capture.release()
    failed.emit("Не удалось открыть источник видео.")
    return None


def _simulate_failure(rate: float) -> bool:
    """Decide at random whether to fail this estimate on purpose."""
    return random.random() < rate


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
