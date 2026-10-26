"""Dialog for changing the password (user scenario 10, step 8)."""

from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.core.validation import new_password_error
from app.ui.theme import set_role


def password_field() -> QLineEdit:
    """A line edit that hides the typed characters."""
    field = QLineEdit()
    field.setEchoMode(QLineEdit.EchoMode.Password)
    return field


class ChangePasswordDialog(QDialog):
    """Asks for the current password and the new one twice.

    The dialog only checks the form; the password is changed on the server
    by the caller after the dialog is accepted.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Смена пароля")
        self.setModal(True)
        self.current_field = password_field()
        self.new_field = password_field()
        self.repeat_field = password_field()
        self.error_label = QLabel()
        set_role(self.error_label, "error")

        form = QFormLayout()
        form.addRow("Текущий пароль:", self.current_field)
        form.addRow("Новый пароль:", self.new_field)
        form.addRow("Повторите новый пароль:", self.repeat_field)

        save_button = QPushButton("Сохранить")
        set_role(save_button, "primary")
        save_button.setDefault(True)
        cancel_button = QPushButton("Отмена")
        save_button.clicked.connect(self._on_save_clicked)
        cancel_button.clicked.connect(self.reject)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(cancel_button)
        buttons.addWidget(save_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addLayout(form)
        layout.addWidget(self.error_label)
        layout.addLayout(buttons)

    @property
    def current_password(self) -> str:
        """Password the user has now."""
        return self.current_field.text()

    @property
    def new_password(self) -> str:
        """Password the user wants to set."""
        return self.new_field.text()

    def _on_save_clicked(self) -> None:
        """Accept only a complete and consistent form."""
        if not self.current_password:
            self.error_label.setText("Введите текущий пароль.")
            return
        error = new_password_error(self.new_password, self.repeat_field.text())
        if error:
            self.error_label.setText(error)
            return
        self.accept()
