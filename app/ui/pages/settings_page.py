"""Analysis settings page (user scenario 2)."""

import sqlite3
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.core import config
from app.core.config import AnalysisSettings
from app.db.local import DB_PATH
from app.db.settings import load_settings, save_settings
from app.ui.theme import page_title, set_role

THRESHOLD_WARNING = (
    "При пороге 0 % или 100 % критические моменты фиксироваться не будут."
)
INTERVAL_WARNING = (
    "Модель не успеет обработать кадры чаще, чем раз в секунду. "
    "Увеличьте интервал до 1 с или больше."
)
SMOOTHING_PRESETS: list[tuple[str, int]] = [
    ("Подробно: видны даже короткие спады", 1),
    ("Сбалансированно", 3),
    ("Общая тенденция: без случайных скачков", 6),
]
SAVED_MESSAGE = "Параметры сохранены и будут применены к следующей лекции."


class SettingsPage(QWidget):
    """Form for editing the analysis settings.

    Signals:
        saved: Emitted with the new AnalysisSettings after they are stored.
    """

    saved = pyqtSignal(object)

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._set_form_values(load_settings(db_path))
        self.status_label.clear()

    def _build_widgets(self) -> None:
        """Create all form widgets with their ranges and texts."""
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(config.THRESHOLD_MIN, config.THRESHOLD_MAX)
        self.threshold_slider.setPageStep(5)
        self.threshold_slider.setMinimumWidth(320)
        self.threshold_value = QLabel()
        self.threshold_value.setFixedWidth(
            self.threshold_value.fontMetrics().horizontalAdvance("100 %") + 8
        )
        self.threshold_warning = self._warning_label(THRESHOLD_WARNING)

        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(config.INTERVAL_MIN_SEC, config.INTERVAL_MAX_SEC)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setSuffix(" с")
        self.interval_warning = self._warning_label(INTERVAL_WARNING)

        self.smoothing_combo = QComboBox()
        for text, window in SMOOTHING_PRESETS:
            self.smoothing_combo.addItem(text, window)
        self.smoothing_hint = QLabel()
        set_role(self.smoothing_hint, "hint")

        self.save_button = QPushButton("Сохранить параметры")
        set_role(self.save_button, "primary")
        self.status_label = QLabel()
        set_role(self.status_label, "success")

    def _build_layout(self) -> None:
        """Arrange the widgets into a titled form."""
        title = page_title("Параметры анализа")

        threshold_row = QHBoxLayout()
        threshold_row.addWidget(self.threshold_slider)
        threshold_row.addWidget(self.threshold_value)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(10)
        form.addRow("Критический порог вовлечённости:", threshold_row)
        form.addRow("", self.threshold_warning)
        form.addRow("Периодичность оценки:", self.interval_spin)
        form.addRow("", self.interval_warning)
        form.addRow("Детализация графика:", self.smoothing_combo)
        form.addRow("", self.smoothing_hint)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(form)
        layout.addSpacing(12)
        layout.addWidget(self.save_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def _connect_signals(self) -> None:
        """Wire widget events to their handlers."""
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        self.interval_spin.valueChanged.connect(self._on_interval_changed)
        self.smoothing_combo.currentIndexChanged.connect(self._on_smoothing_changed)
        self.save_button.clicked.connect(self._on_save_clicked)

    def _set_form_values(self, settings: AnalysisSettings) -> None:
        """Fill the form from settings and refresh the dependent labels."""
        self.threshold_slider.setValue(settings.threshold_pct)
        self.interval_spin.setValue(settings.interval_sec)
        self._select_smoothing(settings.smoothing_window)
        self._on_threshold_changed(settings.threshold_pct)
        self._on_interval_changed(settings.interval_sec)

    def _current_settings(self) -> AnalysisSettings:
        """Build settings from the current state of the form."""
        return AnalysisSettings(
            threshold_pct=self.threshold_slider.value(),
            interval_sec=round(self.interval_spin.value(), 1),
            smoothing_window=self.smoothing_combo.currentData(),
        )

    def _on_threshold_changed(self, value: int) -> None:
        """Show the new threshold and warn about degenerate values."""
        self.threshold_value.setText(f"{value} %")
        self.threshold_warning.setVisible(config.is_threshold_degenerate(value))
        self.status_label.clear()

    def _on_interval_changed(self, value: float) -> None:
        """Warn about a too short interval and block saving while it persists."""
        too_short = config.is_interval_too_short(value)
        self.interval_warning.setVisible(too_short)
        self.save_button.setEnabled(not too_short)
        self._update_smoothing_hint()
        self.status_label.clear()

    def _on_smoothing_changed(self) -> None:
        """Refresh the hint after another smoothing preset is chosen."""
        self._update_smoothing_hint()
        self.status_label.clear()

    def _select_smoothing(self, window: int) -> None:
        """Select the preset with this window, or the default one if none matches."""
        index = self.smoothing_combo.findData(window)
        if index < 0:
            index = self.smoothing_combo.findData(AnalysisSettings().smoothing_window)
        self.smoothing_combo.setCurrentIndex(index)

    def _update_smoothing_hint(self) -> None:
        """Explain the chosen preset in seconds, based on the current interval."""
        window = self.smoothing_combo.currentData()
        if window is None:
            return
        if window == 1:
            self.smoothing_hint.setText("Каждая оценка показывается без усреднения.")
            return
        seconds = f"{window * self.interval_spin.value():g}".replace(".", ",")
        self.smoothing_hint.setText(f"Значение усредняется за последние {seconds} с.")

    def _on_save_clicked(self) -> None:
        """Store the settings, confirm to the user and emit `saved`."""
        settings = self._current_settings()
        try:
            save_settings(settings, self._db_path)
        except sqlite3.Error as exc:
            QMessageBox.critical(
                self, "Ошибка сохранения", f"Не удалось сохранить параметры:\n{exc}"
            )
            return
        self.status_label.setText(SAVED_MESSAGE)
        self.saved.emit(settings)

    @staticmethod
    def _warning_label(text: str) -> QLabel:
        """Create a hidden warning label that keeps its space in the layout.

        Reserving the space prevents the form from shifting under the cursor
        when the warning appears while the user is dragging the slider.
        """
        label = QLabel(text)
        label.setWordWrap(True)
        set_role(label, "warning")
        policy = label.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        label.setSizePolicy(policy)
        label.hide()
        return label
