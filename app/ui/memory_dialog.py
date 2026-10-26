"""Modal warning about insufficient RAM (user scenario 9)."""

from collections.abc import Callable
from datetime import datetime

from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.diagnostics import MemoryStatus
from app.core.formatting import format_memory
from app.ui.theme import set_role

WARNING_TEXT = (
    "Критическая нехватка ОЗУ. Пожалуйста, закройте неиспользуемые программы."
)


class MemoryDialog(QDialog):
    """Shows free and required RAM and lets the user repeat the check.

    The dialog is accepted as soon as a repeated check succeeds and rejected
    if the user cancels. It does not know how memory is measured: it calls
    the `recheck` function it was given.
    """

    def __init__(
        self,
        status: MemoryStatus,
        recheck: Callable[[], MemoryStatus],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._recheck = recheck
        self.setWindowTitle("Недостаточно памяти")
        self.setModal(True)

        message = QLabel(WARNING_TEXT)
        message.setWordWrap(True)
        set_role(message, "statusError")
        self.numbers_label = QLabel()
        self.retry_note = QLabel()
        set_role(self.retry_note, "hint")

        retry_button = QPushButton("Повторить проверку")
        retry_button.setDefault(True)
        set_role(retry_button, "primary")
        cancel_button = QPushButton("Отмена")
        retry_button.clicked.connect(self._on_retry_clicked)
        cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(retry_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(message)
        layout.addWidget(self.numbers_label)
        layout.addWidget(self.retry_note)
        layout.addLayout(buttons)

        self._show_status(status)

    def _show_status(self, status: MemoryStatus) -> None:
        """Display free and required memory."""
        self.numbers_label.setText(
            f"Свободно: {format_memory(status.available_mb)}\n"
            f"Необходимо: {format_memory(status.required_mb)}"
        )

    def _on_retry_clicked(self) -> None:
        """Measure memory again: close on success, otherwise update the numbers."""
        status = self._recheck()
        if status.is_enough:
            self.accept()
            return
        self._show_status(status)
        checked_at = datetime.now().strftime("%H:%M:%S")
        self.retry_note.setText(
            f"Проверка повторена в {checked_at}: памяти по-прежнему недостаточно."
        )
