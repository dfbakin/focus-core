"""Reports page: summary, timeline and PDF export (user scenario 5)."""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QShowEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.formatting import format_datetime, format_duration, format_engagement
from app.core.report import SessionReport
from app.core.report_pdf import export_report_pdf
from app.db.local import DB_PATH
from app.db.sessions import list_sessions, load_session_report
from app.ui.chart import EngagementChart
from app.ui.theme import page_title, set_role

NO_SESSIONS_TEXT = "Проведённых лекций пока нет."
CHOOSE_SESSION_TEXT = "Выберите сессию из списка слева."
NO_DATA_TEXT = "Недостаточно данных для генерации отчёта."
NO_WRITE_ACCESS_TEXT = "Нет прав на запись в выбранную папку. Выберите другую папку."
SESSION_LIST_WIDTH = 240


class StatTile(QFrame):
    """A card with a large value and a caption under it."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_role(self, "card")
        self.value = QLabel("—")
        set_role(self.value, "statValue")
        label = QLabel(caption)
        set_role(label, "hint")
        layout = QVBoxLayout(self)
        layout.addWidget(self.value)
        layout.addWidget(label)


class ReportsPage(QWidget):
    """List of sessions on the left, the report of the chosen one on the right."""

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._report: SessionReport | None = None
        self._pending_session_id: str | None = None
        self._last_export_dir = Path.home() / "Documents"

        self._build_widgets()
        self._build_layout()
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        self.export_button.clicked.connect(self._on_export_clicked)

    def _build_widgets(self) -> None:
        """Create the session list, summary tiles, chart and export button."""
        self.session_list = QListWidget()
        self.session_list.setObjectName("sessionList")
        self.session_list.setFixedWidth(SESSION_LIST_WIDTH)

        self.message_label = QLabel()
        set_role(self.message_label, "emptyState")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.title_label = QLabel()
        set_role(self.title_label, "indicatorCaption")
        self.average_tile = StatTile("Средняя вовлечённость")
        self.minimum_tile = StatTile("Минимум")
        self.moments_tile = StatTile("Критические моменты")
        self.duration_tile = StatTile("Длительность")
        self.chart = EngagementChart()

        self.export_button = QPushButton("Экспорт в PDF")
        set_role(self.export_button, "primary")

    def _build_layout(self) -> None:
        """Place the list and the report side by side."""
        tiles = QHBoxLayout()
        for tile in (self.average_tile, self.minimum_tile, self.moments_tile, self.duration_tile):
            tiles.addWidget(tile)

        self.report_view = QWidget()
        report_layout = QVBoxLayout(self.report_view)
        report_layout.setContentsMargins(0, 0, 0, 0)
        report_layout.addWidget(self.title_label)
        report_layout.addLayout(tiles)
        report_layout.addWidget(self.chart, stretch=1)
        report_layout.addWidget(self.export_button, alignment=Qt.AlignmentFlag.AlignLeft)

        content = QHBoxLayout()
        content.addWidget(self.session_list)
        content.addSpacing(16)
        content.addWidget(self.report_view, stretch=1)
        content.addWidget(self.message_label, stretch=1)

        layout = QVBoxLayout(self)
        layout.addWidget(page_title("Отчёты"))
        layout.addLayout(content, stretch=1)

    def showEvent(self, event: QShowEvent) -> None:
        """Refresh the list every time the page is opened."""
        super().showEvent(event)
        self._reload()

    def show_session(self, session_id: str) -> None:
        """Open the report of a specific session, e.g. the one just finished."""
        self._pending_session_id = session_id
        if self.isVisible():
            self._reload()

    def _reload(self) -> None:
        """Fill the list with finished sessions and restore the selection."""
        selected_id = self._pending_session_id or self._selected_session_id()
        self._pending_session_id = None

        self.session_list.blockSignals(True)
        self.session_list.clear()
        for session in list_sessions(self._db_path):
            text = f"{format_datetime(session.started_at)}\n{format_duration(session.duration_sec)}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, session.id)
            self.session_list.addItem(item)
        self.session_list.blockSignals(False)

        if self.session_list.count() == 0:
            self._show_message(NO_SESSIONS_TEXT)
            return
        row = self._row_of(selected_id)
        if row < 0:
            self._show_message(CHOOSE_SESSION_TEXT)
            return
        self.session_list.setCurrentRow(row)
        self._on_session_selected(self.session_list.item(row))

    def _row_of(self, session_id: str | None) -> int:
        """Return the list row of a session, or -1."""
        for row in range(self.session_list.count()):
            if self.session_list.item(row).data(Qt.ItemDataRole.UserRole) == session_id:
                return row
        return -1

    def _selected_session_id(self) -> str | None:
        """Id of the session currently selected in the list."""
        item = self.session_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_session_selected(self, item: QListWidgetItem | None) -> None:
        """Build and display the report of the selected session."""
        if item is None:
            return
        self._report = load_session_report(item.data(Qt.ItemDataRole.UserRole), self._db_path)
        if self._report is None or not self._report.has_data:
            self._show_message(NO_DATA_TEXT)
            return
        self._render(self._report)

    def _render(self, report: SessionReport) -> None:
        """Fill the tiles and the chart."""
        self.title_label.setText(f"Занятие {format_datetime(report.started_at)}")
        self.average_tile.value.setText(format_engagement(report.average))
        self.minimum_tile.value.setText(format_engagement(report.minimum))
        self.moments_tile.value.setText(str(len(report.critical_moments)))
        self.duration_tile.value.setText(format_duration(report.duration_sec))
        self.chart.set_threshold(report.threshold_pct)
        self.chart.set_series(report.points)
        self.chart.highlight_moments(report.critical_moments)
        self.message_label.hide()
        self.report_view.show()

    def _show_message(self, text: str) -> None:
        """Replace the report with an explanatory message."""
        self._report = None
        self.message_label.setText(text)
        self.report_view.hide()
        self.message_label.show()

    def _on_export_clicked(self) -> None:
        """Ask where to save the PDF until it is written or the user cancels."""
        if self._report is None:
            return
        while True:
            path = self._ask_export_path(self._report)
            if path is None:
                return
            if not os.access(path.parent, os.W_OK):
                if not self._offer_another_folder():
                    return
                continue
            try:
                export_report_pdf(self._report, path)
            except PermissionError:
                if not self._offer_another_folder():
                    return
                continue
            except OSError as exc:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить отчёт:\n{exc}")
                return
            self._last_export_dir = path.parent
            self._offer_to_open(path)
            return

    def _ask_export_path(self, report: SessionReport) -> Path | None:
        """Show the save dialog with a suggested file name."""
        suggested = report.started_at.strftime("focuscore_report_%Y-%m-%d_%H-%M.pdf")
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить отчёт", str(self._last_export_dir / suggested), "PDF (*.pdf)"
        )
        if not path:
            return None
        path = Path(path)
        return path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")

    def _offer_another_folder(self) -> bool:
        """Report missing write access; True if the user wants to pick again."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Нет доступа к папке")
        box.setText(NO_WRITE_ACCESS_TEXT)
        retry_button = box.addButton("Выбрать другую папку", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        return box.clickedButton() is retry_button

    def _offer_to_open(self, path: Path) -> None:
        """Confirm the export and offer to open the file."""
        box = QMessageBox(self)
        box.setWindowTitle("Отчёт сохранён")
        box.setText(f"Отчёт сохранён:\n{path}")
        open_button = box.addButton("Открыть", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Закрыть", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
