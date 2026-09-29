"""Session history page (user scenario 7)."""

import sqlite3
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.formatting import (
    date_matches,
    format_datetime,
    format_duration,
    format_engagement,
)
from app.db.local import DB_PATH
from app.db.sessions import SessionSummary, delete_sessions, list_sessions
from app.ui.theme import page_title, set_role

COLUMNS = ["", "Дата", "Длительность", "Средняя вовлечённость"]
CHECK_COLUMN = 0
CONFIRM_TEXT = "Вы уверены, что хотите безвозвратно удалить эти данные?"
NOT_FOUND_TEXT = "Записи не найдены."
NO_SESSIONS_TEXT = "Проведённых лекций пока нет."


class HistoryPage(QWidget):
    """Table of past sessions with a date search and deletion.

    Signals:
        sessions_deleted: Emitted after sessions were deleted locally, so that
            the deletion can be sent to the server.
    """

    sessions_deleted = pyqtSignal()

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._sessions: list[SessionSummary] = []

        self._build_widgets()
        self._build_layout()
        self._connect_signals()

    def _build_widgets(self) -> None:
        """Create the search field, the table and the delete button."""
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Поиск по дате, например 26.09.2026 или 09.2026")
        self.search_edit.setClearButtonEnabled(True)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(CHECK_COLUMN, QHeaderView.ResizeMode.ResizeToContents)

        self.empty_label = QLabel()
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        set_role(self.empty_label, "emptyState")

        self.delete_button = QPushButton("Удалить выбранное")
        set_role(self.delete_button, "danger")
        self.delete_button.setEnabled(False)

    def _build_layout(self) -> None:
        """Arrange the widgets from top to bottom."""
        title = page_title("История сессий")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.search_edit)
        layout.addWidget(self.table)
        layout.addWidget(self.empty_label)
        layout.addWidget(self.delete_button, alignment=Qt.AlignmentFlag.AlignLeft)

    def _connect_signals(self) -> None:
        """Wire widget events to their handlers."""
        self.search_edit.textChanged.connect(self._apply_filter)
        self.table.itemChanged.connect(self._update_delete_button)
        self.delete_button.clicked.connect(self._on_delete_clicked)

    def showEvent(self, event: QShowEvent) -> None:
        """Reload the data every time the page is opened, to include new sessions."""
        super().showEvent(event)
        self.reload()

    def reload(self) -> None:
        """Read sessions from the database and redraw the table."""
        self._sessions = list_sessions(self._db_path)
        self._apply_filter()

    def _apply_filter(self) -> None:
        """Show only the sessions whose date matches the search query."""
        query = self.search_edit.text()
        visible = [s for s in self._sessions if date_matches(s.started_at, query)]
        self._fill_table(visible)

        has_rows = bool(visible)
        self.table.setVisible(has_rows)
        self.empty_label.setVisible(not has_rows)
        self.empty_label.setText(NOT_FOUND_TEXT if self._sessions else NO_SESSIONS_TEXT)
        self._update_delete_button()

    def _fill_table(self, sessions: list[SessionSummary]) -> None:
        """Put one row per session; the session id is kept in the checkbox cell."""
        self.table.blockSignals(True)
        self.table.setRowCount(len(sessions))
        for row, session in enumerate(sessions):
            check_item = QTableWidgetItem()
            check_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check_item.setCheckState(Qt.CheckState.Unchecked)
            check_item.setData(Qt.ItemDataRole.UserRole, session.id)

            self.table.setItem(row, CHECK_COLUMN, check_item)
            self.table.setItem(row, 1, QTableWidgetItem(format_datetime(session.started_at)))
            self.table.setItem(row, 2, QTableWidgetItem(format_duration(session.duration_sec)))
            self.table.setItem(row, 3, QTableWidgetItem(format_engagement(session.avg_engagement)))
        self.table.blockSignals(False)

    def _checked_ids(self) -> list[str]:
        """Return ids of the sessions ticked in the table."""
        ids = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, CHECK_COLUMN)
            if item.checkState() == Qt.CheckState.Checked:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _update_delete_button(self) -> None:
        """Allow deletion only when at least one session is ticked."""
        self.delete_button.setEnabled(bool(self._checked_ids()))

    def _on_delete_clicked(self) -> None:
        """Ask for confirmation and delete the ticked sessions."""
        ids = self._checked_ids()
        if not ids or not self._confirm_deletion(len(ids)):
            return
        try:
            delete_sessions(ids, self._db_path)
        except sqlite3.Error as exc:
            QMessageBox.critical(self, "Ошибка удаления", f"Не удалось удалить данные:\n{exc}")
            return
        self.reload()
        self.sessions_deleted.emit()

    def _confirm_deletion(self, count: int) -> bool:
        """Return True only if the user explicitly pressed «Удалить»."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Удаление сессий")
        box.setText(CONFIRM_TEXT)
        box.setInformativeText(f"Будет удалено сессий: {count}")
        delete_button = box.addButton("Удалить", QMessageBox.ButtonRole.DestructiveRole)
        cancel_button = box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_button)
        box.exec()
        return box.clickedButton() is delete_button
