"""Lecture page: recording and live engagement estimation.

Covers user scenario 3 (start, pause, finish with the model running in the
background) and scenario 9 (RAM check before the start). A camera is analysed
in real time; a video file is analysed as fast as possible, and the report
opens as soon as the whole file is processed. Audio recording is added in the
next stage.
"""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.analysis import ExponentialMovingAverage, level_for_score, summarize
from app.core.diagnostics import MemoryStatus, check_memory
from app.core.formatting import format_clock, format_seconds
from app.core.offline import forecast_processing_time, next_countdown_value
from app.core.session import SessionClock, SessionState
from app.core.video_source import FILE, VideoSource, VideoSourceError, check_signal
from app.cloud.config import load_cloud_config
from app.db.account import load_account
from app.db.local import DB_PATH
from app.db.sessions import (
    create_session,
    finish_session,
    save_engagement_points,
    save_engagement_summary,
)
from app.db.settings import (
    load_min_free_ram_mb,
    load_settings,
    load_simulated_error_rate,
    load_video_source,
)
from app.ui.image import frame_to_pixmap
from app.ui.indicator import EngagementIndicator
from app.ui.memory_dialog import MemoryDialog
from app.ui.theme import page_title, set_role
from app.ui.workers import CameraAnalysisWorker, FileAnalysisWorker

TIMER_INTERVAL_MS = 500
FORECAST_UPDATE_SEC = 1.0
CAMERA_VIEW_WIDTH = 384
CAMERA_VIEW_HEIGHT = 216

STATUS_IDLE = "Готово к началу лекции"
STATUS_PREPARING = "Подготовка анализа…"
STATUS_RUNNING = "Анализ идёт"
STATUS_PROCESSING = "Обработка видеофайла"
FORECAST_PENDING = "Оценка оставшегося времени…"
FORECAST_TEXT = "Осталось ≈ {remaining}"
STATUS_PAUSED = "Пауза"
NOT_SIGNED_IN_TEXT = "Войдите в учётную запись в разделе «Профиль», чтобы начать лекцию."
NO_SOURCE_LABEL = "Источник видео не выбран"
NO_SOURCE_TEXT = "Сначала выберите источник видео в разделе «Источник видео»."
SOURCE_NOT_READY_TEXT = (
    "Источник видео не отвечает. Проверьте подключение камеры или файл "
    "в разделе «Источник видео»."
)

STATUS_ROLES = {
    SessionState.RUNNING: "statusRunning",
    SessionState.PAUSED: "statusPaused",
}
DEFAULT_STATUS_ROLE = "statusIdle"


class LecturePage(QWidget):
    """Main screen: start button, then timer, indicator, pause and finish.

    Signals:
        lecture_started: Emitted when a session starts.
        session_finished: Emitted with the session id after it is saved.
        frame_ready: Re-emits every analysed frame for live views elsewhere.
        point_recorded: Emitted with (offset_sec, value, is_gap) for every
            new point of the engagement series. For a video file the offset
            is the position in the video.
        change_source_requested: Emitted when the user clicks «Изменить».
    """

    lecture_started = pyqtSignal()
    session_finished = pyqtSignal(str)
    frame_ready = pyqtSignal(object)
    point_recorded = pyqtSignal(float, float, bool)
    change_source_requested = pyqtSignal()

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._session_id: str | None = None
        self._clock = SessionClock()
        self._worker: CameraAnalysisWorker | FileAnalysisWorker | None = None
        self._smoother = ExponentialMovingAverage(1)
        self._points: list[tuple[float, float]] = []
        self._is_file = False
        self._video_position = 0.0
        self._video_duration = 0.0
        self._processing_started_at: float | None = None
        self._forecast_updated_at = 0.0
        self._shown_remaining: float | None = None

        self._timer = QTimer(self)
        self._timer.setInterval(TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._update_timer)

        self._build_widgets()
        self._build_layout()
        self._render(STATUS_IDLE)

    def _build_widgets(self) -> None:
        """Create the source line, the indicator, the timer and the buttons."""
        self.source_label = QLabel()
        set_role(self.source_label, "hint")
        self.change_source_link = QLabel('<a href="#">Изменить</a>')
        self.change_source_link.linkActivated.connect(
            lambda _: self.change_source_requested.emit()
        )

        self.status_label = QLabel()
        self.timer_label = QLabel(format_clock(0))
        set_role(self.timer_label, "timer")
        self.indicator = EngagementIndicator()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p %")
        self.progress_bar.setFixedWidth(CAMERA_VIEW_WIDTH)
        self.forecast_label = QLabel()
        set_role(self.forecast_label, "hint")
        self.progress_row = QWidget()
        progress_layout = QVBoxLayout(self.progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(
            self.forecast_label, alignment=Qt.AlignmentFlag.AlignHCenter
        )

        self.camera_view = QLabel()
        self.camera_view.setFixedSize(CAMERA_VIEW_WIDTH, CAMERA_VIEW_HEIGHT)
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_role(self.camera_view, "preview")
        self.camera_view.hide()

        self.start_button = QPushButton("Начать лекцию")
        set_role(self.start_button, "large")
        self.start_button.clicked.connect(self._on_start_clicked)

        self.pause_button = QPushButton("Пауза")
        self.pause_button.setMinimumSize(160, 48)
        self.pause_button.clicked.connect(self._on_pause_clicked)

        self.finish_button = QPushButton("Завершить лекцию")
        self.finish_button.setMinimumSize(160, 48)
        set_role(self.finish_button, "danger")
        self.finish_button.clicked.connect(self._on_finish_clicked)

        self.camera_button = QPushButton("Показать видео")
        self.camera_button.setCheckable(True)
        self.camera_button.toggled.connect(self._on_camera_toggled)

    def _build_layout(self) -> None:
        """Center all elements of the page."""
        title = page_title("Лекция")
        center = Qt.AlignmentFlag.AlignHCenter

        source_row = QHBoxLayout()
        source_row.addStretch()
        source_row.addWidget(self.source_label)
        source_row.addWidget(self.change_source_link)
        source_row.addStretch()

        controls = QHBoxLayout()
        controls.addStretch()
        controls.addWidget(self.pause_button)
        controls.addWidget(self.finish_button)
        controls.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addStretch()
        layout.addLayout(source_row)
        layout.addWidget(self.status_label, alignment=center)
        layout.addWidget(self.timer_label, alignment=center)
        layout.addWidget(self.progress_row, alignment=center)
        layout.addWidget(self.indicator, alignment=center)
        layout.addWidget(self.camera_view, alignment=center)
        layout.addSpacing(12)
        layout.addWidget(self.start_button, alignment=center)
        layout.addLayout(controls)
        layout.addWidget(self.camera_button, alignment=center)
        layout.addStretch()

    def showEvent(self, event: QShowEvent) -> None:
        """Refresh the source name every time the page is opened."""
        super().showEvent(event)
        self._update_source_label()

    def is_lecture_running(self) -> bool:
        """True while a session is running or paused."""
        return self._clock.state in (SessionState.RUNNING, SessionState.PAUSED)

    def shutdown(self) -> None:
        """Stop the analysis thread; called when the window closes."""
        self._stop_worker()

    def _update_source_label(self) -> None:
        """Show which source the next lecture will use."""
        source = load_video_source(self._db_path)
        if source is None:
            self.source_label.setText(NO_SOURCE_LABEL)
        else:
            self.source_label.setText(f"Источник: {source.display_name}")

    def _on_start_clicked(self) -> None:
        """Check the prerequisites and start a session if they are met."""
        if load_cloud_config() is not None and load_account(self._db_path) is None:
            QMessageBox.warning(self, "Требуется вход", NOT_SIGNED_IN_TEXT)
            return
        source = load_video_source(self._db_path)
        if source is None:
            QMessageBox.information(self, "Источник не выбран", NO_SOURCE_TEXT)
            return
        if not self._source_is_ready(source):
            QMessageBox.warning(self, "Источник недоступен", SOURCE_NOT_READY_TEXT)
            return
        if not self._memory_is_sufficient():
            return
        self._start_session(source)

    def _source_is_ready(self, source: VideoSource) -> bool:
        """Try to read a frame from the source before the lecture starts."""
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            check_signal(source)
        except VideoSourceError:
            return False
        finally:
            QApplication.restoreOverrideCursor()
        return True

    def _memory_is_sufficient(self) -> bool:
        """Check free RAM; while it is short, keep the user in MemoryDialog."""
        status = self._check_memory()
        if status.is_enough:
            return True
        dialog = MemoryDialog(status, self._check_memory, self)
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _check_memory(self) -> MemoryStatus:
        """Compare free RAM with the requirement stored in settings."""
        return check_memory(load_min_free_ram_mb(self._db_path))

    def _start_session(self, source: VideoSource) -> None:
        """Create a session record, start the timer and the analysis thread."""
        settings = load_settings(self._db_path)
        self._session_id = create_session(
            source.kind, settings.threshold_pct, self._db_path
        )
        self._smoother = ExponentialMovingAverage(settings.smoothing_window)
        self._points = []
        self._is_file = source.kind == FILE
        self._video_position = 0.0
        self._video_duration = 0.0
        self._processing_started_at = None
        self._forecast_updated_at = 0.0
        self._shown_remaining = None
        self.progress_bar.setValue(0)
        self.forecast_label.setText(FORECAST_PENDING)

        self._clock = SessionClock()
        self._clock.start()
        self._timer.start()

        error_rate = load_simulated_error_rate(self._db_path)
        if self._is_file:
            self._worker = self._create_file_worker(source, settings.interval_sec, error_rate)
        else:
            self._worker = self._create_camera_worker(source, settings.interval_sec, error_rate)
        self._worker.model_ready.connect(self._on_model_ready)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()

        self._render(STATUS_PREPARING)
        self.indicator.show_neutral("Ожидание первой оценки…")
        self.lecture_started.emit()

    def _create_camera_worker(
        self, source: VideoSource, interval_sec: float, error_rate: float
    ) -> CameraAnalysisWorker:
        """Real-time analysis: points are placed by the lecture clock."""
        worker = CameraAnalysisWorker(source, interval_sec, error_rate, self)
        worker.estimate_ready.connect(self._on_camera_estimate)
        worker.estimate_failed.connect(self._on_camera_estimate_failed)
        return worker

    def _create_file_worker(
        self, source: VideoSource, interval_sec: float, error_rate: float
    ) -> FileAnalysisWorker:
        """Fast analysis of a file: points are placed by the video position."""
        worker = FileAnalysisWorker(source, interval_sec, error_rate, self)
        worker.estimate_ready.connect(self._record_estimate)
        worker.estimate_failed.connect(self._record_gap)
        worker.progress.connect(self._on_file_progress)
        worker.source_ended.connect(self._on_source_ended)
        return worker

    def _on_model_ready(self) -> None:
        """The model is loaded and frames are flowing.

        For a file the processing time is counted from this moment, so that
        loading the model does not distort the forecast.
        """
        self._processing_started_at = self._clock.elapsed_exact()
        if self._clock.state == SessionState.RUNNING:
            self._render(self._running_status())

    def _on_frame(self, frame: np.ndarray) -> None:
        """Show the frame in the camera view and pass it on."""
        if self.camera_view.isVisible():
            pixmap = frame_to_pixmap(frame).scaled(
                self.camera_view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.camera_view.setPixmap(pixmap)
        self.frame_ready.emit(frame)

    def _on_camera_estimate(self, score: float) -> None:
        """Place a camera estimate at the current moment of the lecture."""
        self._record_estimate(self._clock.elapsed_exact(), score)

    def _on_camera_estimate_failed(self) -> None:
        """Mark a failed camera estimate at the current moment of the lecture."""
        self._record_gap(self._clock.elapsed_exact())

    def _record_estimate(self, offset: float, score: float) -> None:
        """Smooth the new score, remember it and update the indicator."""
        if self._clock.state != SessionState.RUNNING:
            return
        smoothed = self._smoother.add(score)
        self._points.append((offset, smoothed))
        self.indicator.show_level(level_for_score(smoothed))
        self.point_recorded.emit(offset, smoothed, False)

    def _record_gap(self, offset: float) -> None:
        """Repeat the previous value and mark the gap (user scenario 4, step 5).

        Before the first successful estimate there is nothing to repeat,
        so the failure is ignored.
        """
        if self._clock.state != SessionState.RUNNING or not self._points:
            return
        previous = self._points[-1][1]
        self._points.append((offset, previous))
        self.point_recorded.emit(offset, previous, True)

    def _on_file_progress(self, position_sec: float, duration_sec: float) -> None:
        """Show how much of the video file has been analysed."""
        self._video_position = position_sec
        self._video_duration = duration_sec
        self.progress_bar.setValue(self._progress_percent())

    def _progress_percent(self) -> int:
        """Share of the video file analysed so far, 0..100."""
        return round(100 * self._fraction_done())

    def _fraction_done(self) -> float:
        """Share of the video file analysed so far, 0..1."""
        if self._video_duration <= 0:
            return 0.0
        return min(1.0, self._video_position / self._video_duration)

    def _update_forecast(self) -> None:
        """Show how much processing time is left, counting down every second.

        Paused time is not counted, because the session clock stops during
        a pause.
        """
        if self._processing_started_at is None:
            return
        now = self._clock.elapsed_exact()
        if now - self._forecast_updated_at < FORECAST_UPDATE_SEC:
            return
        self._forecast_updated_at = now
        forecast = forecast_processing_time(
            now - self._processing_started_at, self._fraction_done()
        )
        if forecast is None:
            self.forecast_label.setText(FORECAST_PENDING)
            return
        _, remaining = forecast
        self._shown_remaining = next_countdown_value(self._shown_remaining, remaining)
        self.forecast_label.setText(
            FORECAST_TEXT.format(remaining=format_seconds(round(self._shown_remaining)))
        )

    def _running_status(self) -> str:
        """Status text while analysis is running."""
        return STATUS_PROCESSING if self._is_file else STATUS_RUNNING

    def _on_pause_clicked(self) -> None:
        """Toggle between pause and running, pausing the analysis as well."""
        if self._clock.state == SessionState.RUNNING:
            self._clock.pause()
            self._set_worker_paused(True)
            self.indicator.show_neutral("Анализ приостановлен")
            self._render(STATUS_PAUSED)
        else:
            self._clock.resume()
            self._set_worker_paused(False)
            self.indicator.show_neutral("Ожидание оценки…")
            self._render(self._running_status())

    def _on_finish_clicked(self) -> None:
        """Finish the lecture on the user's request."""
        self._finish_session()

    def _on_source_ended(self) -> None:
        """The whole video file is analysed: finish and open the report."""
        self._video_position = max(self._video_position, self._video_duration)
        self._finish_session()

    def _on_worker_failed(self, message: str) -> None:
        """Show why analysis stopped; the user can still finish the lecture."""
        self._stop_worker()
        self.status_label.setText(message)
        set_role(self.status_label, "statusError")
        self.indicator.show_neutral("Нет данных")

    def _finish_session(self) -> None:
        """Stop analysis, save the series and the summary, return to idle."""
        if self._session_id is None:
            return
        self._stop_worker()
        self._timer.stop()
        duration = self._clock.finish()
        if self._is_file:
            duration = round(self._video_position)
        values = [value for _, value in self._points]
        summary = summarize(values)

        finish_session(self._session_id, self._db_path, duration_sec=duration)
        save_engagement_points(self._session_id, self._points, self._db_path)
        if summary is not None:
            save_engagement_summary(self._session_id, *summary, self._db_path)

        finished_id = self._session_id
        self._session_id = None
        self._reset_to_idle()
        self.session_finished.emit(finished_id)

    def _reset_to_idle(self) -> None:
        """Return the page to its initial look; the results are shown in the report."""
        self._clock = SessionClock()
        self.camera_button.setChecked(False)
        self.indicator.show_neutral("")
        self._render(STATUS_IDLE)

    def _on_camera_toggled(self, visible: bool) -> None:
        """Show or hide the live camera view under the indicator."""
        self.camera_view.setVisible(visible)
        self.camera_view.clear()
        self.camera_button.setText("Скрыть видео" if visible else "Показать видео")

    def _set_worker_paused(self, paused: bool) -> None:
        """Forward pause state to the analysis thread if it is running."""
        if self._worker is not None:
            self._worker.set_paused(paused)

    def _stop_worker(self) -> None:
        """Stop the analysis thread if it is running."""
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

    def _update_timer(self) -> None:
        """Show the active time of a camera session or the forecast for a file."""
        if self._is_file and self.is_lecture_running():
            self._update_forecast()
            return
        self.timer_label.setText(format_clock(self._clock.elapsed()))

    def _render(self, status: str) -> None:
        """Show the buttons and the status that match the session state."""
        state = self._clock.state
        in_progress = self.is_lecture_running()

        self.status_label.setText(status)
        set_role(self.status_label, STATUS_ROLES.get(state, DEFAULT_STATUS_ROLE))
        self.change_source_link.setVisible(not in_progress)
        self.start_button.setVisible(not in_progress)
        self.pause_button.setVisible(in_progress)
        self.finish_button.setVisible(in_progress)
        self.camera_button.setVisible(in_progress)
        self.indicator.setVisible(in_progress)
        processing_file = in_progress and self._is_file
        self.timer_label.setVisible(not processing_file)
        self.progress_row.setVisible(processing_file)
        self.pause_button.setText("Продолжить" if state == SessionState.PAUSED else "Пауза")
        self._update_timer()
