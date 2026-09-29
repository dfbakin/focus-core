"""Live engagement chart (user scenario 4)."""

from pathlib import Path

from PyQt6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.core.analysis import level_for_score
from app.core.formatting import format_engagement
from app.db.local import DB_PATH
from app.db.settings import load_settings
from app.ui.chart import EngagementChart
from app.ui.indicator import LEVEL_COLORS, LEVEL_TEXTS
from app.ui.theme import page_title, set_role

STATUS_WAITING = "Начните лекцию, чтобы увидеть график вовлечённости."
STATUS_LIVE = "График обновляется по мере поступления оценок."
STATUS_FINISHED = "Лекция завершена. Итоговая кривая вовлечённости за всё занятие."
GAP_NOTE = "✕ — пропуск: оценка не получена, повторено предыдущее значение"


class MonitoringPage(QWidget):
    """Chart of the smoothed engagement score since the lecture started."""

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._points: list[tuple[int, float]] = []
        self._gaps: list[tuple[int, float]] = []

        self._build_widgets()
        self._build_layout()
        self.status_label.setText(STATUS_WAITING)

    def _build_widgets(self) -> None:
        """Create the current value line, the status and the chart."""
        self.current_value = QLabel("—")
        set_role(self.current_value, "timer")
        self.current_level = QLabel()
        set_role(self.current_level, "indicatorCaption")
        self.status_label = QLabel()
        set_role(self.status_label, "hint")
        self.gap_note = QLabel(GAP_NOTE)
        set_role(self.gap_note, "hint")
        self.gap_note.hide()
        self.chart = EngagementChart()

    def _build_layout(self) -> None:
        """Put the current value above the chart."""
        value_row = QHBoxLayout()
        value_row.addWidget(self.current_value)
        value_row.addSpacing(16)
        value_row.addWidget(self.current_level)
        value_row.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(page_title("Мониторинг"))
        layout.addWidget(self.status_label)
        layout.addLayout(value_row)
        layout.addWidget(self.chart, stretch=1)
        layout.addWidget(self.gap_note)

    def start_session(self) -> None:
        """Clear the chart and show the threshold of the new lecture."""
        self._points = []
        self._gaps = []
        self.chart.set_series([])
        self.chart.set_threshold(load_settings(self._db_path).threshold_pct)
        self.gap_note.hide()
        self.current_value.setText("—")
        self.current_level.clear()
        self.status_label.setText(STATUS_LIVE)

    def add_point(self, offset_sec: int, value: float, is_gap: bool) -> None:
        """Append a point; a gap is additionally marked with a red cross."""
        self._points.append((offset_sec, value))
        if is_gap:
            self._gaps.append((offset_sec, value))
            self.gap_note.show()
        self.chart.set_series(self._points, self._gaps)

        level = level_for_score(value)
        self.current_value.setText(format_engagement(value))
        self.current_level.setText(LEVEL_TEXTS[level])
        self.current_level.setStyleSheet(f"color: {LEVEL_COLORS[level]};")

    def clear(self) -> None:
        """Remove the curve of the previous lecture, e.g. after signing out."""
        self._points = []
        self._gaps = []
        self.chart.set_series([])
        self.gap_note.hide()
        self.current_value.setText("—")
        self.current_level.clear()
        self.status_label.setText(STATUS_WAITING)

    def finish_session(self) -> None:
        """Stop updating and keep the final curve of the whole lecture."""
        self.status_label.setText(STATUS_FINISHED)
