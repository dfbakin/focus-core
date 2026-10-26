"""FocusCore desktop application entry point."""

import sys

from PyQt6.QtCore import QLibraryInfo, QLocale, QTranslator
from PyQt6.QtWidgets import QApplication

from app.db.local import init_db
from app.db.maintenance import apply_retention_policy
from app.db.settings import load_retention_policy
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme


def install_russian_translation(app: QApplication) -> None:
    """Translate Qt's standard buttons and dialogs (OK, Cancel, ...) into Russian.

    The application is passed as the translator's parent, so Qt keeps the
    translator alive for as long as the application runs.
    """
    translator = QTranslator(app)
    folder = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale(QLocale.Language.Russian), "qtbase", "_", folder):
        app.installTranslator(translator)


def main() -> None:
    """Prepare the database, apply the retention policy and run the Qt event loop."""
    init_db()
    apply_retention_policy(load_retention_policy())
    app = QApplication(sys.argv)
    install_russian_translation(app)
    apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
