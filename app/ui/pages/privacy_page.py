"""Data and privacy page (user scenario 8)."""

import sqlite3
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.formatting import format_days, format_size, plural
from app.core.retention import RETENTION_CHOICES_DAYS, RetentionPolicy
from app.db.local import DB_PATH
from app.db.maintenance import delete_expired_sessions, storage_status
from app.db.settings import load_retention_policy, save_retention_policy
from app.ui.theme import page_title, set_role

PRIVACY_TEXT = (
    "Видео- и аудиопоток не сохраняются на диск: кадры обрабатываются "
    "в оперативной памяти и сразу удаляются. Хранятся только числовые оценки "
    "вовлечённости, длительность занятий и ваши комментарии."
)
IRREVERSIBLE_WARNING = "Удалённые данные нельзя будет восстановить."
NOTHING_TO_DELETE = "Устаревших записей не найдено, удалять нечего."
POLICY_SAVED = "Политика хранения сохранена."


class PrivacyPage(QWidget):
    """Retention policy settings, manual cleanup and storage status.

    Signals:
        sessions_deleted: Emitted after expired sessions were deleted.
    """

    sessions_deleted = pyqtSignal()

    def __init__(self, db_path: Path = DB_PATH, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._db_path = db_path
        self._build_widgets()
        self._build_layout()
        self._connect_signals()
        self._set_form_values(load_retention_policy(db_path))

    def _build_widgets(self) -> None:
        """Create the privacy statement, the policy form and the status."""
        self.privacy_text = QLabel(PRIVACY_TEXT)
        self.privacy_text.setWordWrap(True)

        self.auto_delete_check = QCheckBox("Автоматическое удаление старых сессий")
        self.days_combo = QComboBox()
        for days in RETENTION_CHOICES_DAYS:
            self.days_combo.addItem(f"Удалять через {format_days(days)}", days)

        self.warning_label = QLabel(IRREVERSIBLE_WARNING)
        set_role(self.warning_label, "warning")

        self.clean_button = QPushButton("Очистить сейчас")
        set_role(self.clean_button, "danger")
        self.save_button = QPushButton("Сохранить политику хранения")
        set_role(self.save_button, "primary")

        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.storage_label = QLabel()
        set_role(self.storage_label, "hint")

    def _build_layout(self) -> None:
        """Arrange the page in the order of the scenario."""
        section = QLabel("Хранение данных")
        set_role(section, "indicatorCaption")

        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.clean_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(page_title("Данные и приватность"))
        layout.addWidget(self.privacy_text)
        layout.addSpacing(16)
        layout.addWidget(section)
        layout.addWidget(self.auto_delete_check)
        layout.addWidget(self.days_combo, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.warning_label)
        layout.addSpacing(8)
        layout.addLayout(buttons)
        layout.addWidget(self.result_label)
        layout.addSpacing(8)
        layout.addWidget(self.storage_label)
        layout.addStretch()

    def _connect_signals(self) -> None:
        """Wire widget events to their handlers."""
        self.auto_delete_check.toggled.connect(self._on_policy_edited)
        self.days_combo.currentIndexChanged.connect(self._on_policy_edited)
        self.save_button.clicked.connect(self._on_save_clicked)
        self.clean_button.clicked.connect(self._on_clean_clicked)

    def showEvent(self, event: QShowEvent) -> None:
        """Refresh the storage status every time the page is opened."""
        super().showEvent(event)
        self._update_storage_status()

    def _set_form_values(self, policy: RetentionPolicy) -> None:
        """Fill the form from the stored policy."""
        self.auto_delete_check.setChecked(policy.auto_delete)
        index = self.days_combo.findData(policy.days)
        self.days_combo.setCurrentIndex(max(index, 0))
        self.result_label.clear()

    def _current_policy(self) -> RetentionPolicy:
        """Build a policy from the form."""
        return RetentionPolicy(
            auto_delete=self.auto_delete_check.isChecked(),
            days=self.days_combo.currentData(),
        )

    def _on_policy_edited(self) -> None:
        """Clear the previous message when the user changes the policy."""
        self.result_label.clear()

    def _on_save_clicked(self) -> None:
        """Store the retention policy and confirm it."""
        try:
            save_retention_policy(self._current_policy(), self._db_path)
        except sqlite3.Error as exc:
            self._show_db_error(exc)
            return
        self._show_result(POLICY_SAVED, "success")

    def _on_clean_clicked(self) -> None:
        """Delete sessions older than the chosen period and report the result."""
        days = self.days_combo.currentData()
        try:
            result = delete_expired_sessions(days, self._db_path)
        except sqlite3.Error as exc:
            self._show_db_error(exc)
            return

        if result.deleted_count == 0:
            self._show_result(NOTHING_TO_DELETE, "hint")
            return
        count = result.deleted_count
        word = plural(count, "сессия", "сессии", "сессий")
        self._show_result(
            f"Удалено: {count} {word} старше {format_days(days)}. "
            f"Освобождено {format_size(result.freed_bytes)}.",
            "success",
        )
        self._update_storage_status()
        self.sessions_deleted.emit()

    def _update_storage_status(self) -> None:
        """Show how many sessions are stored and how much space they take."""
        status = storage_status(self._db_path)
        word = plural(status.session_count, "сессия", "сессии", "сессий")
        self.storage_label.setText(
            f"Сейчас хранится: {status.session_count} {word}, "
            f"размер базы данных {format_size(status.size_bytes)}."
        )

    def _show_result(self, text: str, role: str) -> None:
        """Show the outcome of the last action."""
        self.result_label.setText(text)
        set_role(self.result_label, role)

    def _show_db_error(self, exc: sqlite3.Error) -> None:
        """Report a database failure."""
        QMessageBox.critical(self, "Ошибка базы данных", f"Операция не выполнена:\n{exc}")
