"""Main window: a sidebar with the sections on the left, the selected section on the right."""

from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QKeySequence
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QScrollArea,
    QStackedWidget,
    QToolButton,
    QWidget,
)

from app.cloud.account import create_account_service
from app.ui.pages.help_page import HelpPage
from app.ui.pages.history_page import HistoryPage
from app.ui.pages.lecture_page import LecturePage
from app.ui.pages.monitoring_page import MonitoringPage
from app.ui.pages.privacy_page import PrivacyPage
from app.ui.pages.profile_page import ProfilePage
from app.ui.pages.reports_page import ReportsPage
from app.ui.pages.settings_page import SettingsPage
from app.ui.pages.source_page import SourcePage

LECTURE = "Лекция"
MONITORING = "Мониторинг"
SOURCE = "Источник видео"
SETTINGS = "Параметры анализа"
HISTORY = "История"
PRIVACY = "Данные и приватность"
REPORTS = "Отчёты"
HELP = "Помощь"
PROFILE = "Профиль"

SECTIONS: list[str] = [
    LECTURE,
    MONITORING,
    SOURCE,
    SETTINGS,
    HISTORY,
    REPORTS,
    PRIVACY,
    PROFILE,
    HELP,
]
STATUS_MESSAGE_MS = 5000
MENU_ROW_HEIGHT = 34
SIDEBAR_WIDTH = 240
MINIMUM_SIZE = QSize(900, 600)
SIDEBAR_SHORTCUT = "Ctrl+Meta+S"


class MainWindow(QMainWindow):
    """Application shell that switches between section pages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FocusCore")
        self.resize(1100, 740)
        self.setMinimumSize(MINIMUM_SIZE)

        self.menu = QListWidget()
        self.menu.setObjectName("menu")
        self.stack = QStackedWidget()

        account_service = create_account_service()
        self.lecture_page = LecturePage(account_service)
        self.monitoring_page = MonitoringPage()
        self.source_page = SourcePage()
        self.settings_page = SettingsPage()
        self.history_page = HistoryPage()
        self.privacy_page = PrivacyPage()
        self.reports_page = ReportsPage()
        self.help_page = HelpPage()
        self.profile_page = ProfilePage(account_service)
        pages: dict[str, QWidget] = {
            LECTURE: self.lecture_page,
            MONITORING: self.monitoring_page,
            SOURCE: self.source_page,
            SETTINGS: self.settings_page,
            HISTORY: self.history_page,
            PRIVACY: self.privacy_page,
            REPORTS: self.reports_page,
            HELP: self.help_page,
            PROFILE: self.profile_page,
        }
        for title in SECTIONS:
            self._add_section(title, pages[title])

        self._build_view_menu()
        self._build_layout()
        self._connect_pages()
        self._apply_access()
        QTimer.singleShot(0, self.profile_page.restore_on_startup)

    def _add_section(self, title: str, page: QWidget) -> None:
        """Add a menu item and its page together so their indices always match.

        The row height is set on the item, because Qt sizes list rows before
        the stylesheet padding is applied. Every page is placed in a scroll
        area, otherwise the stack would be as large as its largest page and
        the window could not be made smaller than that; a page that does
        not fit a small window scrolls instead.
        """
        item = QListWidgetItem(title)
        item.setSizeHint(QSize(0, MENU_ROW_HEIGHT))
        self.menu.addItem(item)
        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)

    def _build_view_menu(self) -> None:
        """Add the «Вид» menu with the command that hides the sidebar.

        Apple HIG suggests letting people hide the sidebar with ⌘⌃S; on
        macOS Qt maps «Ctrl» to ⌘ and «Meta» to ⌃.
        """
        self.sidebar_action = QAction("Боковая панель", self)
        self.sidebar_action.setCheckable(True)
        self.sidebar_action.setChecked(True)
        self.sidebar_action.setShortcut(QKeySequence(SIDEBAR_SHORTCUT))
        self.sidebar_action.toggled.connect(self._show_sidebar)
        self.menuBar().addMenu("Вид").addAction(self.sidebar_action)

    def _show_sidebar(self, visible: bool) -> None:
        """Show or hide the sidebar; the page then takes the whole width."""
        self.sidebar.setVisible(visible)

    def _build_layout(self) -> None:
        """Place the sidebar with the menu and the page area side by side."""
        self.sidebar = QWidget()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 12)
        sidebar_layout.addWidget(self.menu)

        self.sidebar_toggle = QToolButton()
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setText("☰")
        self.sidebar_toggle.setToolTip("Показать или скрыть боковую панель (⌘⌃S)")
        self.sidebar_toggle.clicked.connect(self.sidebar_action.trigger)

        content = QWidget()
        content.setObjectName("content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 10, 32, 24)
        content_layout.addWidget(self.sidebar_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        content_layout.addWidget(self.stack)
        self.stack.setContentsMargins(20, 0, 0, 0)

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addWidget(content)
        self.setCentralWidget(container)

        self.menu.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.menu.setCurrentRow(SECTIONS.index(LECTURE))

    def _connect_pages(self) -> None:
        """React to events of the pages that affect the whole window."""
        self.source_page.applied.connect(self._on_source_applied)
        self.lecture_page.change_source_requested.connect(
            lambda: self.menu.setCurrentRow(SECTIONS.index(SOURCE))
        )
        self.lecture_page.lecture_started.connect(
            lambda: self.source_page.set_lecture_mode(True)
        )
        self.lecture_page.session_finished.connect(
            lambda _: self.source_page.set_lecture_mode(False)
        )
        self.lecture_page.frame_ready.connect(self.source_page.show_lecture_frame)
        self.lecture_page.lecture_started.connect(self.monitoring_page.start_session)
        self.lecture_page.point_recorded.connect(self.monitoring_page.add_point)
        self.lecture_page.session_finished.connect(
            lambda _: self.monitoring_page.finish_session()
        )
        self.lecture_page.session_finished.connect(self._on_session_finished)
        self.lecture_page.transcription_started.connect(
            self.reports_page.on_transcription_started
        )
        self.lecture_page.transcription_finished.connect(
            self.reports_page.on_transcription_finished
        )
        self.history_page.sessions_deleted.connect(self.profile_page.sync_in_background)
        self.privacy_page.sessions_deleted.connect(self.profile_page.sync_in_background)
        self.profile_page.status_message.connect(self._show_status)
        self.profile_page.data_synced.connect(self._on_data_synced)
        self.profile_page.signed_out.connect(self.monitoring_page.clear)
        self.profile_page.signed_out.connect(self._apply_access)
        self.profile_page.signed_in.connect(self._on_signed_in)
        self.settings_page.saved.connect(
            lambda _: self._show_status("Параметры анализа сохранены")
        )

    def _on_source_applied(self) -> None:
        """Confirm the saved source and return to the main screen (Lecture)."""
        self.menu.setCurrentRow(SECTIONS.index(LECTURE))
        self._show_status("Источник видео сохранён")

    def _apply_access(self) -> None:
        """Without a signed-in account show only the sign-in screen.

        The sidebar and its toggle are hidden, so no section, and therefore
        no lecture, can be opened until the user signs in or registers.
        """
        locked = self.profile_page.requires_sign_in
        self.sidebar_action.setEnabled(not locked)
        self.sidebar_toggle.setVisible(not locked)
        self.sidebar.setVisible(not locked and self.sidebar_action.isChecked())
        if locked:
            self.menu.setCurrentRow(SECTIONS.index(PROFILE))

    def _on_signed_in(self) -> None:
        """Unlock the application and open the main screen."""
        self._apply_access()
        self.menu.setCurrentRow(SECTIONS.index(LECTURE))

    def _on_session_finished(self, session_id: str) -> None:
        """Open the report of the lecture that has just finished (scenario 3, step 9)."""
        self.reports_page.show_session(session_id)
        self.menu.setCurrentRow(SECTIONS.index(REPORTS))
        self._show_status("Лекция сохранена")
        self.profile_page.sync_in_background()

    def _on_data_synced(self) -> None:
        """Redraw the history if new sessions arrived while it is open."""
        if self.history_page.isVisible():
            self.history_page.reload()

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background threads before the window closes."""
        self.source_page.stop_preview()
        self.lecture_page.shutdown()
        self.profile_page.shutdown()
        super().closeEvent(event)

    def _show_status(self, text: str) -> None:
        """Show a short confirmation in the status bar at the bottom."""
        self.statusBar().showMessage(text, STATUS_MESSAGE_MS)
