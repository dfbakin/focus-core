"""Sign-in, registration and profile page (user scenario 10)."""

from collections.abc import Callable
from datetime import datetime

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.cloud.account import AccountService
from app.cloud.client import (
    CloudError,
    CloudUnavailableError,
    EmailConfirmationRequiredError,
    InvalidCredentialsError,
    RateLimitError,
    UserAlreadyExistsError,
    WeakPasswordError,
)
from app.cloud.sync import SyncResult
from app.core.formatting import plural
from app.core.validation import is_valid_email, registration_error
from app.ui.password_dialog import ChangePasswordDialog, password_field
from app.ui.theme import page_title, set_role
from app.ui.workers import TaskWorker

OFFLINE_TEXT = (
    "Сервер недоступен. Приложение работает в автономном режиме: "
    "открыты ранее сохранённые данные."
)
NOT_CONFIGURED_TEXT = (
    "Облачное хранилище не настроено. Приложение работает в автономном режиме."
)
WRONG_CREDENTIALS_TEXT = "Неверный адрес электронной почты или пароль. Попробуйте ещё раз."
ALREADY_REGISTERED_TEXT = "Этот адрес уже зарегистрирован. Выполните вход."
WEAK_PASSWORD_TEXT = "Сервер отклонил пароль как слишком простой. Придумайте другой."
CONFIRM_EMAIL_TEXT = "Подтвердите адрес по ссылке из письма, затем выполните вход."
SESSION_EXPIRED_TEXT = "Срок входа истёк. Войдите снова."
WRONG_CURRENT_PASSWORD_TEXT = "Текущий пароль введён неверно."
RATE_LIMIT_TEXT = "Слишком много попыток за короткое время. Подождите несколько минут и повторите."
SERVER_ERROR_TEXT = "Сервер не смог выполнить запрос. Попробуйте ещё раз позже."

LOGIN_VIEW, REGISTER_VIEW, PROFILE_VIEW = 0, 1, 2
FORM_WIDTH = 420
SHUTDOWN_WAIT_MS = 3000


def sessions_text(count: int) -> str:
    """'3 сессии' with the right word form."""
    return f"{count} {plural(count, 'сессия', 'сессии', 'сессий')}"


class ProfilePage(QWidget):
    """Switches between the sign-in form, the registration form and the profile.

    All server requests run in a TaskWorker, one at a time.

    Signals:
        data_synced: Emitted after sessions were exchanged with the server.
        status_message: Emitted with a short text for the window's status bar.
        signed_in: Emitted after a successful sign-in or registration.
        signed_out: Emitted after the user signed out or the sign-in expired.
    """

    data_synced = pyqtSignal()
    status_message = pyqtSignal(str)
    signed_in = pyqtSignal()
    signed_out = pyqtSignal()

    def __init__(self, service: AccountService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._service = service
        self._task: TaskWorker | None = None
        self._sync_requested = False

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_login_view())
        self.stack.addWidget(self._build_register_view())
        self.stack.addWidget(self._build_profile_view())
        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.title_label = page_title("Профиль")
        self.welcome_label = QLabel("Войдите или зарегистрируйтесь, чтобы начать работу.")
        set_role(self.welcome_label, "hint")

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.welcome_label)
        layout.addWidget(self.stack, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.message_label)
        layout.addStretch()
        self._show_account_or_login()

    def _build_login_view(self) -> QWidget:
        """Email, password, «Войти» and the link to registration."""
        self.login_email = QLineEdit()
        self.login_email.setPlaceholderText("name@example.com")
        self.login_password = password_field()
        login_button = QPushButton("Войти")
        set_role(login_button, "primary")
        login_button.clicked.connect(self._on_login_clicked)
        self.login_password.returnPressed.connect(self._on_login_clicked)
        to_register = self._link("Нет учётной записи? Зарегистрироваться", REGISTER_VIEW)

        form = QFormLayout()
        form.addRow("Электронная почта:", self.login_email)
        form.addRow("Пароль:", self.login_password)
        return self._form_view("Вход", form, login_button, to_register)

    def _build_register_view(self) -> QWidget:
        """Name, surname, email, password twice and the link back to sign-in."""
        self.reg_first_name = QLineEdit()
        self.reg_last_name = QLineEdit()
        self.reg_email = QLineEdit()
        self.reg_email.setPlaceholderText("name@example.com")
        self.reg_password = password_field()
        self.reg_repeat = password_field()
        register_button = QPushButton("Зарегистрироваться")
        set_role(register_button, "primary")
        register_button.clicked.connect(self._on_register_clicked)
        to_login = self._link("Уже есть учётная запись? Войти", LOGIN_VIEW)

        form = QFormLayout()
        form.addRow("Имя:", self.reg_first_name)
        form.addRow("Фамилия:", self.reg_last_name)
        form.addRow("Электронная почта:", self.reg_email)
        form.addRow("Пароль:", self.reg_password)
        form.addRow("Повторите пароль:", self.reg_repeat)
        return self._form_view("Регистрация", form, register_button, to_login)

    def _build_profile_view(self) -> QWidget:
        """Account email, editable name, password, sync and sign-out buttons."""
        self.account_label = QLabel()
        set_role(self.account_label, "hint")
        self.profile_first_name = QLineEdit()
        self.profile_last_name = QLineEdit()
        self.sync_label = QLabel()
        set_role(self.sync_label, "hint")

        save_button = QPushButton("Сохранить изменения")
        set_role(save_button, "primary")
        save_button.clicked.connect(self._on_save_profile_clicked)
        password_button = QPushButton("Сменить пароль")
        password_button.clicked.connect(self._on_change_password_clicked)
        sync_button = QPushButton("Синхронизировать сейчас")
        sync_button.clicked.connect(self.sync_in_background)
        logout_button = QPushButton("Выйти")
        set_role(logout_button, "danger")
        logout_button.clicked.connect(self._on_logout_clicked)

        form = QFormLayout()
        form.addRow("Имя:", self.profile_first_name)
        form.addRow("Фамилия:", self.profile_last_name)

        buttons = QHBoxLayout()
        for button in (save_button, password_button, sync_button, logout_button):
            buttons.addWidget(button)

        view = QWidget()
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.account_label)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addLayout(buttons)
        layout.addWidget(self.sync_label)
        return view

    def _form_view(self, title: str, form: QFormLayout, button: QPushButton, link: QLabel) -> QWidget:
        """Common layout of the sign-in and registration forms."""
        heading = QLabel(title)
        set_role(heading, "indicatorCaption")
        view = QWidget()
        view.setFixedWidth(FORM_WIDTH)
        layout = QVBoxLayout(view)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addLayout(form)
        layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(link)
        return view

    def _link(self, text: str, view_index: int) -> QLabel:
        """A clickable text that switches to another form."""
        link = QLabel(f'<a href="#">{text}</a>')
        link.linkActivated.connect(lambda _: self._switch_view(view_index))
        return link

    def shutdown(self) -> None:
        """Wait briefly for a running request before the window closes."""
        if self._task is not None:
            self._task.wait(SHUTDOWN_WAIT_MS)

    @property
    def requires_sign_in(self) -> bool:
        """True if the cloud is set up and nobody is signed in on this device."""
        return self._service.is_configured and self._service.account is None

    def restore_on_startup(self) -> None:
        """Sign in again with the saved token and synchronize, in the background."""
        if self._service.account is None:
            return
        self._run(self._service.restore, self._on_synced, "Синхронизация с сервером…")

    def sync_in_background(self) -> None:
        """Exchange sessions with the server if signed in; queued if busy."""
        if self._service.account is None:
            return
        if self._task is not None:
            self._sync_requested = True
            return
        self._run(self._service.synchronize, self._on_synced, "Синхронизация с сервером…")

    def _on_login_clicked(self) -> None:
        """Validate the form and sign in (steps 1, 4, 5)."""
        email = self.login_email.text().strip()
        password = self.login_password.text()
        if not is_valid_email(email) or not password:
            self._show_message("Введите адрес электронной почты и пароль.", "error")
            return
        self._run(
            lambda: self._service.sign_in(email, password),
            self._on_signed_in,
            "Выполняется вход…",
        )

    def _on_register_clicked(self) -> None:
        """Validate the form and create an account (steps 2, 3)."""
        first = self.reg_first_name.text().strip()
        last = self.reg_last_name.text().strip()
        email = self.reg_email.text().strip()
        password = self.reg_password.text()
        error = registration_error(first, last, email, password, self.reg_repeat.text())
        if error:
            self._show_message(error, "error")
            return
        self._run(
            lambda: self._service.register(first, last, email, password),
            self._on_signed_in,
            "Создание учётной записи…",
        )

    def _on_save_profile_clicked(self) -> None:
        """Send the new name to the server (steps 7, 9)."""
        first = self.profile_first_name.text().strip()
        last = self.profile_last_name.text().strip()
        if not first or not last:
            self._show_message("Укажите имя и фамилию.", "error")
            return
        self._run(
            lambda: self._service.update_profile(first, last),
            lambda _: self._on_profile_updated("Данные профиля обновлены и синхронизированы с сервером."),
            "Сохранение…",
        )

    def _on_change_password_clicked(self) -> None:
        """Ask for the passwords and change them on the server (steps 8, 9)."""
        dialog = ChangePasswordDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        current, new = dialog.current_password, dialog.new_password
        self._run(
            lambda: self._service.change_password(current, new),
            lambda _: self._on_profile_updated("Пароль изменён и синхронизирован с сервером."),
            "Смена пароля…",
            wrong_credentials_text=WRONG_CURRENT_PASSWORD_TEXT,
        )

    def _on_logout_clicked(self) -> None:
        """Forget the account on this device; local data remains."""
        self._service.sign_out()
        self._switch_view(LOGIN_VIEW)
        self._show_message(
            "Вы вышли из учётной записи. Её лекции скрыты и снова появятся после входа.",
            "hint",
        )
        self.signed_out.emit()

    def _run(
        self,
        function: Callable[[], object],
        on_success: Callable[[object], None],
        busy_text: str,
        wrong_credentials_text: str = WRONG_CREDENTIALS_TEXT,
    ) -> None:
        """Start a server request in the background and lock the forms meanwhile."""
        if self._task is not None:
            return
        self.stack.setEnabled(False)
        self._show_message(busy_text, "hint")
        task = TaskWorker(function, self)
        task.succeeded.connect(lambda result: self._on_task_done(on_success, result))
        task.failed.connect(lambda exc: self._on_task_failed(exc, wrong_credentials_text))
        task.finished.connect(task.deleteLater)
        self._task = task
        task.start()

    def _on_task_done(self, on_success: Callable[[object], None], result: object) -> None:
        """Unlock the forms, handle the result and run a queued synchronization."""
        self._task = None
        self.stack.setEnabled(True)
        on_success(result)
        if self._sync_requested:
            self._sync_requested = False
            self.sync_in_background()

    def _on_task_failed(self, exc: Exception, wrong_credentials_text: str) -> None:
        """Explain the failure in terms of the scenario."""
        self._task = None
        self._sync_requested = False
        self.stack.setEnabled(True)
        if isinstance(exc, CloudUnavailableError):
            text = OFFLINE_TEXT if self._service.is_configured else NOT_CONFIGURED_TEXT
            self._show_account_or_login()
            self._show_message(text, "warning")
            self.status_message.emit("Автономный режим")
        elif isinstance(exc, UserAlreadyExistsError):
            self.login_email.setText(self.reg_email.text().strip())
            self._switch_view(LOGIN_VIEW)
            self._show_message(ALREADY_REGISTERED_TEXT, "warning")
        elif isinstance(exc, InvalidCredentialsError):
            had_account = self.stack.currentIndex() == PROFILE_VIEW
            self._show_account_or_login()
            if had_account and self._service.account is None:
                self.signed_out.emit()
            expired = self.stack.currentIndex() == LOGIN_VIEW and not self.login_password.text()
            self.login_password.clear()
            self._show_message(SESSION_EXPIRED_TEXT if expired else wrong_credentials_text, "error")
        elif isinstance(exc, WeakPasswordError):
            self._show_message(WEAK_PASSWORD_TEXT, "error")
        elif isinstance(exc, RateLimitError):
            self._show_message(RATE_LIMIT_TEXT, "warning")
        elif isinstance(exc, EmailConfirmationRequiredError):
            self._switch_view(LOGIN_VIEW)
            self._show_message(CONFIRM_EMAIL_TEXT, "warning")
        elif isinstance(exc, CloudError):
            self._show_message(SERVER_ERROR_TEXT, "error")
            self.message_label.setToolTip(str(exc))
        else:
            self._show_message("Непредвиденная ошибка. Попробуйте ещё раз.", "error")
            self.message_label.setToolTip(str(exc))

    def _on_signed_in(self, result: SyncResult) -> None:
        """Show the profile and report what was synchronized (step 5)."""
        self.login_password.clear()
        self.reg_password.clear()
        self.reg_repeat.clear()
        self._show_account_or_login()
        self._show_message(
            f"Вход выполнен. Загружено с сервера: {sessions_text(result.downloaded)}, "
            f"отправлено: {sessions_text(result.uploaded)}.",
            "success",
        )
        self._update_sync_label()
        self.signed_in.emit()
        self.data_synced.emit()

    def _on_synced(self, result: SyncResult | None) -> None:
        """Report a background synchronization."""
        self._show_account_or_login()
        self.message_label.clear()
        self._update_sync_label()
        if result is not None and (result.downloaded or result.uploaded):
            self.data_synced.emit()
        self.status_message.emit("Данные синхронизированы с сервером")

    def _on_profile_updated(self, text: str) -> None:
        """Confirm a profile or password change (step 9)."""
        self._show_account_or_login()
        self._show_message(text, "success")

    def _show_account_or_login(self) -> None:
        """Show the profile if an account is remembered, otherwise the sign-in form."""
        account = self._service.account
        self.title_label.setText("Профиль" if account else "Добро пожаловать в FocusCore")
        self.welcome_label.setVisible(account is None)
        if account is None:
            if self.stack.currentIndex() == PROFILE_VIEW:
                self._switch_view(LOGIN_VIEW)
            return
        self.account_label.setText(f"Вы вошли как {account.email}")
        self.profile_first_name.setText(account.first_name)
        self.profile_last_name.setText(account.last_name)
        self.stack.setCurrentIndex(PROFILE_VIEW)

    def _update_sync_label(self) -> None:
        """Show when the data was last exchanged with the server."""
        self.sync_label.setText(
            f"Последняя синхронизация: {datetime.now().strftime('%H:%M:%S')}"
        )

    def _switch_view(self, index: int) -> None:
        """Show another form and clear the old message."""
        self.stack.setCurrentIndex(index)
        self.message_label.clear()

    def _show_message(self, text: str, role: str) -> None:
        """Show a message under the form with the given style role.

        Technical details of unexpected errors stay in the tooltip, so the
        interface text is always in Russian.
        """
        self.message_label.setText(text)
        self.message_label.setToolTip("")
        set_role(self.message_label, role)
