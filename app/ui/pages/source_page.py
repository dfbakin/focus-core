"""Video source page (user scenario 1)."""

import sqlite3
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
import numpy as np
from PyQt6.QtGui import QHideEvent, QShowEvent
from PyQt6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.core.video_source import (
    CAMERA,
    FILE,
    CameraUnavailableError,
    FileUnreadableError,
    VideoSource,
    check_signal,
    find_cameras,
)
from app.db.local import DB_PATH
from app.db.settings import load_video_source, save_video_source
from app.ui.image import frame_to_pixmap
from app.ui.theme import page_title, set_role
from app.ui.workers import PreviewWorker

PREVIEW_WIDTH = 480
PREVIEW_HEIGHT = 270
PREVIEW_PLACEHOLDER = "Здесь появится изображение с источника"
FIRST_FRAME_TIMEOUT_MS = 5000
STALL_TIMEOUT_MS = 2000
VIDEO_FILTER = "Видео (*.mp4 *.avi *.mov *.mkv);;Все файлы (*)"

NOTHING_SELECTED_ERROR = "Выберите камеру или видеофайл."
CAMERA_ERROR = (
    "Не удалось получить изображение с камеры. Возможно, она занята другим "
    "приложением. Закройте его и повторите проверку."
)
FILE_ERROR = (
    "Не удалось прочитать видео. Файл повреждён или имеет неподдерживаемый формат."
)
LECTURE_NOTE = "Идёт лекция: источник нельзя изменить до её завершения."


class SourcePage(QWidget):
    """Choice of a camera or a video file with a signal check and live preview.

    Signals:
        applied: Emitted with the VideoSource after it is saved.
    """

    applied = pyqtSignal(object)

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._checked_source: VideoSource | None = None
        self._preview_worker: PreviewWorker | None = None
        self._saved_camera_name = ""
        self._cameras_scanned = False
        self._lecture_mode = False

        self._stall_timer = QTimer(self)
        self._stall_timer.setSingleShot(True)
        self._stall_timer.timeout.connect(self._on_preview_failed)

        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._load_saved_source()

    def _build_widgets(self) -> None:
        """Create all widgets of the page."""
        self.camera_radio = QRadioButton("Камера")
        self.file_radio = QRadioButton("Видеофайл")
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.camera_radio)
        self.mode_group.addButton(self.file_radio)

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(220)
        self.refresh_button = QPushButton("Обновить список")

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("Файл не выбран")
        self.browse_button = QPushButton("Выбрать файл…")

        self.check_button = QPushButton("Проверка сигнала")

        self.preview = QLabel(PREVIEW_PLACEHOLDER)
        self.preview.setFixedSize(PREVIEW_WIDTH, PREVIEW_HEIGHT)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_role(self.preview, "preview")

        self.info_label = QLabel()
        set_role(self.info_label, "hint")
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        set_role(self.error_label, "error")

        self.apply_button = QPushButton("Применить и сохранить")
        set_role(self.apply_button, "primary")
        self.apply_button.setEnabled(False)

        self.lecture_note = QLabel(LECTURE_NOTE)
        set_role(self.lecture_note, "warning")
        self.lecture_note.hide()

    def _build_layout(self) -> None:
        """Arrange the widgets from top to bottom in the order of the scenario."""
        title = page_title("Источник видео")

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.camera_radio)
        mode_row.addWidget(self.file_radio)
        mode_row.addStretch()

        self.camera_controls = QWidget()
        camera_row = QHBoxLayout(self.camera_controls)
        camera_row.setContentsMargins(0, 0, 0, 0)
        camera_row.addWidget(self.camera_combo)
        camera_row.addWidget(self.refresh_button)
        camera_row.addStretch()

        self.file_controls = QWidget()
        file_row = QHBoxLayout(self.file_controls)
        file_row.setContentsMargins(0, 0, 0, 0)
        file_row.addWidget(self.file_path_edit)
        file_row.addWidget(self.browse_button)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.lecture_note)
        layout.addLayout(mode_row)
        layout.addWidget(self.camera_controls)
        layout.addWidget(self.file_controls)
        layout.addWidget(self.check_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.preview)
        layout.addWidget(self.info_label)
        layout.addWidget(self.error_label)
        layout.addWidget(self.apply_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch()

    def _connect_signals(self) -> None:
        """Wire widget events to their handlers."""
        self.camera_radio.toggled.connect(self._on_mode_changed)
        self.refresh_button.clicked.connect(self._scan_cameras)
        self.camera_combo.currentIndexChanged.connect(self._reset_check)
        self.browse_button.clicked.connect(self._choose_file)
        self.check_button.clicked.connect(self._on_check_clicked)
        self.apply_button.clicked.connect(self._on_apply_clicked)

    def set_lecture_mode(self, active: bool) -> None:
        """Lock the page during a lecture and show the lecture's own video.

        The lecture already reads the source, so the page must not open it
        a second time; it displays frames passed to show_lecture_frame.
        """
        self._lecture_mode = active
        self._reset_check()
        self.lecture_note.setVisible(active)
        for widget in (
            self.camera_radio,
            self.file_radio,
            self.camera_controls,
            self.file_controls,
            self.check_button,
        ):
            widget.setEnabled(not active)

    def show_lecture_frame(self, frame: np.ndarray) -> None:
        """Display a frame coming from the running lecture."""
        if self._lecture_mode and self.isVisible():
            self._show_frame(frame)

    def showEvent(self, event: QShowEvent) -> None:
        """Scan cameras the first time the page is opened.

        Scanning takes a moment, so it is postponed until the user actually
        needs the list instead of slowing down the application start.
        """
        super().showEvent(event)
        if self._lecture_mode:
            return
        if self.camera_radio.isChecked() and not self._cameras_scanned:
            self._scan_cameras()
        elif self.file_radio.isChecked() and self.file_path_edit.text():
            self._on_check_clicked()

    def hideEvent(self, event: QHideEvent) -> None:
        """Release the camera when the user leaves the page."""
        super().hideEvent(event)
        self._reset_check()

    def _load_saved_source(self) -> None:
        """Restore the previously saved source in the form."""
        source = load_video_source(self._db_path)
        if source is not None and source.kind == FILE:
            self.file_radio.setChecked(True)
            self.file_path_edit.setText(source.file_path)
        else:
            self.camera_radio.setChecked(True)
            if source is not None:
                self._saved_camera_name = source.camera_name
        self._on_mode_changed(self.camera_radio.isChecked())

    def _on_mode_changed(self, camera_mode: bool) -> None:
        """Show only the controls of the chosen mode.

        A file is checked right after it is chosen, so the check button
        is needed only for cameras.
        """
        self.camera_controls.setVisible(camera_mode)
        self.file_controls.setVisible(not camera_mode)
        self.check_button.setVisible(camera_mode)
        if camera_mode and not self._cameras_scanned and self.isVisible():
            self._scan_cameras()
        self._reset_check()
        if not camera_mode and self.file_path_edit.text() and self.isVisible():
            self._on_check_clicked()

    def _scan_cameras(self) -> None:
        """Find connected cameras and fill the drop-down list with their names.

        The previously saved camera is selected by name, not by index,
        because indices change when devices are plugged in or out.
        """
        self._reset_check()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            cameras = find_cameras()
        finally:
            QApplication.restoreOverrideCursor()
        self._cameras_scanned = True

        self.camera_combo.clear()
        for camera in cameras:
            self.camera_combo.addItem(camera.name, camera.index)
        if not cameras:
            self.camera_combo.addItem("Камеры не найдены")
        self.camera_combo.setEnabled(bool(cameras))

        saved_position = self.camera_combo.findText(self._saved_camera_name)
        if saved_position >= 0:
            self.camera_combo.setCurrentIndex(saved_position)

    def _choose_file(self) -> None:
        """Open a file dialog and immediately check and play the chosen video."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Выберите видеофайл", "", VIDEO_FILTER
        )
        if path:
            self.file_path_edit.setText(path)
            self._on_check_clicked()

    def _selected_source(self) -> VideoSource | None:
        """Build a VideoSource from the form, or None if nothing is chosen."""
        if self.camera_radio.isChecked():
            index = self.camera_combo.currentData()
            if index is None:
                return None
            return VideoSource(
                CAMERA, camera_index=index, camera_name=self.camera_combo.currentText()
            )
        path = self.file_path_edit.text()
        return VideoSource(FILE, file_path=path) if path else None

    def _on_check_clicked(self) -> None:
        """Check the source, show its parameters and start the live preview."""
        source = self._selected_source()
        if source is None:
            self._show_error(NOTHING_SELECTED_ERROR)
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            info = check_signal(source)
        except CameraUnavailableError:
            self._show_error(CAMERA_ERROR)
            return
        except FileUnreadableError:
            self._show_error(FILE_ERROR)
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._show_frame(info.frame)
        fps_text = f"{info.fps:.0f} кадров/с" if info.fps > 0 else "не определена"
        self.info_label.setText(
            f"Разрешение: {info.width}×{info.height}    Частота кадров: {fps_text}"
        )
        self.error_label.clear()
        self._checked_source = source
        self.apply_button.setEnabled(True)
        self._start_preview(source)

    def _start_preview(self, source: VideoSource) -> None:
        """Start reading frames from the source in a background thread."""
        self._stop_preview()
        self._preview_worker = PreviewWorker(source, self)
        self._preview_worker.frame_ready.connect(self._on_preview_frame)
        self._preview_worker.failed.connect(self._on_preview_failed)
        self._preview_worker.start()
        self._stall_timer.start(FIRST_FRAME_TIMEOUT_MS)

    def stop_preview(self) -> None:
        """Stop the live preview and release the source; safe to call anytime."""
        self._stop_preview()

    def _stop_preview(self) -> None:
        """Stop the worker if it is running."""
        self._stall_timer.stop()
        if self._preview_worker is not None:
            self._preview_worker.stop()
            self._preview_worker = None

    def _on_preview_frame(self, frame: np.ndarray) -> None:
        """Show a frame from the worker, ignoring late frames of a stopped one.

        Every frame restarts the stall timer: if no frame arrives within
        STALL_TIMEOUT_MS, the source is considered lost. The first frame gets
        more time, because some cameras need a moment to start.
        """
        if self.sender() is self._preview_worker:
            self._stall_timer.start(STALL_TIMEOUT_MS)
            self._show_frame(frame)

    def _on_preview_failed(self) -> None:
        """Report that the source stopped delivering frames during the preview.

        Triggered either by the worker or by the stall timer.
        """
        if self._preview_worker is None or self._checked_source is None:
            return
        is_camera = self._checked_source.kind == CAMERA
        self._show_error(CAMERA_ERROR if is_camera else FILE_ERROR)

    def _show_frame(self, frame: np.ndarray) -> None:
        """Scale a frame to the preview area and display it."""
        pixmap = frame_to_pixmap(frame).scaled(
            self.preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def _on_apply_clicked(self) -> None:
        """Save the checked source and emit `applied`."""
        if self._checked_source is None:
            return
        try:
            save_video_source(self._checked_source, self._db_path)
        except sqlite3.Error as exc:
            QMessageBox.critical(
                self, "Ошибка сохранения", f"Не удалось сохранить источник:\n{exc}"
            )
            return
        self.applied.emit(self._checked_source)

    def _reset_check(self) -> None:
        """Stop the preview and forget the last check result."""
        self._stop_preview()
        self._checked_source = None
        self.apply_button.setEnabled(False)
        self.preview.clear()
        self.preview.setText(PREVIEW_PLACEHOLDER)
        self.info_label.clear()
        self.error_label.clear()

    def _show_error(self, text: str) -> None:
        """Reset the previous result and display an error message."""
        self._reset_check()
        self.error_label.setText(text)
