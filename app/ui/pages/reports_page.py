"""Reports page: summary, timeline, critical moments and PDF export.

Covers user scenario 5 (report and PDF) and scenario 6 (critical moments
with transcripts and comments).
"""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QShowEvent
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.formatting import (
    format_datetime,
    format_duration,
    format_engagement,
    format_time,
)
from app.core.report import CriticalMoment, SessionReport
from app.core.report_pdf import export_report_pdf
from app.db.local import DB_PATH
from app.db.moments import save_comment
from app.db.sessions import list_sessions, load_session_report
from app.ui.chart import EngagementChart
from app.ui.theme import set_role

NO_SESSIONS_TEXT = "Проведённых лекций пока нет."
CHOOSE_SESSION_TEXT = "Выберите сессию из списка слева."
NO_DATA_TEXT = "Недостаточно данных для генерации отчёта."
NO_WRITE_ACCESS_TEXT = "Нет прав на запись в выбранную папку. Выберите другую папку."
NO_MOMENTS_TEXT = "Критических моментов не зафиксировано."
TRANSCRIBING_TEXT = "Идёт расшифровка речи в критических моментах…"
NO_TRANSCRIPT_TEXT = "Расшифровки нет: для этой лекции звук не расшифровывался."
NO_COMMENT_TEXT = "Комментария пока нет."
MOSTLY_LOW_TEXT = (
    "Вовлечённость была ниже порога большую часть занятия: возможно, порог "
    "стоит понизить в разделе «Параметры анализа»."
)
SESSION_LIST_WIDTH = 200
MOMENTS_HEIGHT = 190


class StatTile(QFrame):
    """A card with a large value and a caption under it."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        set_role(self, "card")
        self.value = QLabel("—")
        set_role(self.value, "statValue")
        label = QLabel(caption)
        label.setWordWrap(True)
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
        self._transcription_messages: dict[str, str] = {}

        self._build_widgets()
        self._build_moment_widgets()
        self._build_layout()
        self.session_list.currentItemChanged.connect(self._on_session_selected)
        self.moments_list.currentRowChanged.connect(self._on_moment_selected)
        self.comment_button.clicked.connect(self._on_comment_clicked)
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

    def _build_moment_widgets(self) -> None:
        """Create the list of critical moments and the panel with their details."""
        self.moments_title = QLabel("Критические моменты")
        set_role(self.moments_title, "indicatorCaption")
        self.transcription_status = QLabel()
        self.transcription_status.setWordWrap(True)
        set_role(self.transcription_status, "hint")
        self.no_moments_label = QLabel(NO_MOMENTS_TEXT)
        self.threshold_hint = QLabel(MOSTLY_LOW_TEXT)
        self.threshold_hint.setWordWrap(True)
        set_role(self.threshold_hint, "warning")
        set_role(self.no_moments_label, "hint")

        self.moments_list = QListWidget()
        self.moments_list.setFixedSize(SESSION_LIST_WIDTH, MOMENTS_HEIGHT)
        self.transcript_label = QLabel()
        self.transcript_label.setWordWrap(True)
        self.transcript_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.comment_label = QLabel()
        self.comment_label.setWordWrap(True)
        set_role(self.comment_label, "hint")
        self.comment_button = QPushButton("Добавить комментарий")

        self.moment_details = QWidget()
        details = QVBoxLayout(self.moment_details)
        details.setContentsMargins(0, 0, 0, 0)
        details.addWidget(self.transcript_label)
        details.addWidget(self.comment_label)
        details.addWidget(self.comment_button, alignment=Qt.AlignmentFlag.AlignLeft)
        details.addStretch()

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
        report_layout.addWidget(self.moments_title)
        report_layout.addWidget(self.transcription_status)
        report_layout.addWidget(self.threshold_hint)
        report_layout.addWidget(self.no_moments_label)
        moments_row = QHBoxLayout()
        moments_row.addWidget(self.moments_list)
        moments_row.addSpacing(16)
        moments_row.addWidget(self.moment_details, stretch=1)
        report_layout.addLayout(moments_row)
        report_layout.addWidget(self.export_button, alignment=Qt.AlignmentFlag.AlignLeft)

        content = QHBoxLayout()
        content.addWidget(self.session_list)
        content.addSpacing(16)
        content.addWidget(self.report_view, stretch=1)
        content.addWidget(self.message_label, stretch=1)

        layout = QVBoxLayout(self)
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
        self._render_moments(report)
        self.message_label.hide()
        self.report_view.show()

    def on_transcription_started(self, session_id: str) -> None:
        """Note that the moments of a session are being transcribed."""
        self._transcription_messages[session_id] = TRANSCRIBING_TEXT
        self._refresh_if_shown(session_id)

    def on_transcription_finished(self, session_id: str, message: str) -> None:
        """Show the saved transcripts, or why they could not be made."""
        if message:
            self._transcription_messages[session_id] = message
        else:
            self._transcription_messages.pop(session_id, None)
        self._refresh_if_shown(session_id)

    def _refresh_if_shown(self, session_id: str) -> None:
        """Reload the report if it is the one on screen, keeping the selected moment."""
        if self._report is None or self._report.session_id != session_id:
            return
        row = self.moments_list.currentRow()
        self._report = load_session_report(session_id, self._db_path)
        if self._report is not None:
            self._render_moments(self._report, max(row, 0))

    def _render_moments(self, report: SessionReport, row: int = 0) -> None:
        """Fill the list of critical moments and select one (scenario 6, steps 3-4)."""
        moments = report.critical_moments
        status = self._transcription_messages.get(report.session_id, "")
        self.transcription_status.setText(status)
        self.transcription_status.setVisible(bool(status))
        self.no_moments_label.setVisible(not moments)
        self.threshold_hint.setVisible(report.mostly_below_threshold)
        self.moments_list.setVisible(bool(moments))
        self.moment_details.setVisible(bool(moments))

        self.moments_list.blockSignals(True)
        self.moments_list.clear()
        for moment in moments:
            self.moments_list.addItem(
                f"{self._moment_range(report, moment)}\n"
                f"минимум {format_engagement(moment.min_value)}"
            )
        self.moments_list.blockSignals(False)

        if not moments:
            self.chart.highlight_moments([])
            return
        row = min(row, len(moments) - 1)
        self.moments_list.setCurrentRow(row)
        self._on_moment_selected(row)

    def _on_moment_selected(self, row: int) -> None:
        """Highlight the moment on the chart and show its transcript and comment."""
        if self._report is None or row < 0:
            return
        moments = self._report.critical_moments
        note = self._report.note_for(moments[row])
        self.chart.highlight_moments(moments, selected=row)
        if note.transcript_parts:
            self.transcript_label.setText(
                "\n\n".join(f"«{part}»" for part in note.transcript_parts)
            )
        else:
            self.transcript_label.setText(
                self._transcription_messages.get(self._report.session_id, NO_TRANSCRIPT_TEXT)
            )
        self.comment_label.setText(
            f"Комментарий: {note.comment}" if note.comment else NO_COMMENT_TEXT
        )
        self.comment_button.setText(
            "Изменить комментарий" if note.comment else "Добавить комментарий"
        )

    def _on_comment_clicked(self) -> None:
        """Ask for a comment to the selected moment and save it (scenario 6, steps 7-8)."""
        row = self.moments_list.currentRow()
        if self._report is None or row < 0:
            return
        moment = self._report.critical_moments[row]
        text, accepted = QInputDialog.getMultiLineText(
            self,
            "Комментарий",
            f"Комментарий к моменту {self._moment_range(self._report, moment)}:",
            self._report.note_for(moment).comment or "",
        )
        if not accepted:
            return
        save_comment(self._report.session_id, moment, text, self._db_path)
        self._refresh_if_shown(self._report.session_id)

    @staticmethod
    def _moment_range(report: SessionReport, moment: CriticalMoment) -> str:
        """Wall-clock time span of a moment, e.g. '14:20:05 – 14:20:30'."""
        start = format_time(report.clock_time(moment.start_sec))
        end = format_time(report.clock_time(moment.end_sec))
        return f"{start} – {end}"

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
        # Append rather than Path.with_suffix: a name like "Лекция 26.09" must
        # become "Лекция 26.09.pdf", not "Лекция 26.pdf".
        return path if path.suffix.lower() == ".pdf" else path.with_name(path.name + ".pdf")

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
