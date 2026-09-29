"""Visual theme of the application: colors, widget roles and the stylesheet.

Widgets get a "role" (a Qt dynamic property) instead of inline styles,
and the stylesheet below decides how every role looks. The accent color is
chosen by ACCENT_PRESET; everything else is a dark graphite palette.
"""

from pathlib import Path

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

ASSETS_DIR = Path(__file__).parent / "assets"
CHEVRON_DOWN = (ASSETS_DIR / "chevron_down.svg").as_posix()
CHEVRON_UP = (ASSETS_DIR / "chevron_up.svg").as_posix()

BACKGROUND = "#141416"
SURFACE = "#1c1c1f"
FIELD = "#232327"
BORDER = "#34343a"
HOVER = "#2b2b30"
TEXT = "#f2f2f3"
MUTED = "#a1a1aa"
DISABLED = "#5c5c63"
PREVIEW_BACKGROUND = "#0b0b0c"

SUCCESS = "#34c38f"
WARNING = "#e0a846"
DANGER = "#f06a6a"
DANGER_SOFT = "#3a2124"

ACCENT_PRESETS = {
    "emerald": {"base": "#1f9d74", "hover": "#25b385", "pressed": "#187a5a",
                "soft": "#17362c", "on": "#ffffff"},
    "ochre": {"base": "#c9962b", "hover": "#d9a73c", "pressed": "#a67a1f",
              "soft": "#3a3020", "on": "#141416"},
    "burgundy": {"base": "#9b2c42", "hover": "#b0354e", "pressed": "#7c2234",
                 "soft": "#3a1f26", "on": "#ffffff"},
}
ACCENT_PRESET = "emerald"

_accent = ACCENT_PRESETS[ACCENT_PRESET]
ACCENT = _accent["base"]
ACCENT_HOVER = _accent["hover"]
ACCENT_PRESSED = _accent["pressed"]
ACCENT_SOFT = _accent["soft"]
ON_ACCENT = _accent["on"]

STYLESHEET = f"""
QWidget {{
    font-size: 14px;
    color: {TEXT};
}}
QMainWindow, QDialog, QMessageBox {{
    background: {BACKGROUND};
}}

QWidget#sidebar {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
QLabel#brand {{
    font-size: 20px;
    font-weight: 700;
    color: {ACCENT};
    padding: 4px 8px 12px 8px;
}}
QListWidget#menu {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget#menu::item {{
    padding: 10px 12px;
    margin: 2px 0;
    border-radius: 8px;
    color: {MUTED};
}}
QListWidget#menu::item:hover {{
    background: {HOVER};
    color: {TEXT};
}}
QListWidget#menu::item:selected {{
    background: {ACCENT_SOFT};
    color: {TEXT};
    font-weight: 600;
}}

QLabel[role="pageTitle"] {{
    font-size: 24px;
    font-weight: 700;
    padding-bottom: 8px;
}}
QLabel[role="hint"] {{ color: {MUTED}; }}
QLabel[role="warning"] {{ color: {WARNING}; }}
QLabel[role="error"] {{ color: {DANGER}; }}
QLabel[role="success"] {{ color: {SUCCESS}; }}
QLabel[role="emptyState"] {{ color: {MUTED}; font-size: 16px; padding: 40px; }}
QLabel[role="timer"] {{ font-size: 44px; font-weight: 700; }}
QLabel[role="statusIdle"] {{ font-size: 16px; color: {MUTED}; }}
QLabel[role="statusRunning"] {{ font-size: 16px; font-weight: 600; color: {SUCCESS}; }}
QLabel[role="statusPaused"] {{ font-size: 16px; font-weight: 600; color: {WARNING}; }}
QLabel[role="statusError"] {{ font-size: 16px; font-weight: 600; color: {DANGER}; }}
QLabel[role="indicatorCaption"] {{ font-size: 17px; font-weight: 600; }}
QLabel[role="preview"] {{
    background: {PREVIEW_BACKGROUND};
    color: {MUTED};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}

QPushButton {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 8px 18px;
}}
QPushButton:hover {{ background: {HOVER}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {DISABLED}; background: {SURFACE}; border-color: {HOVER}; }}

QPushButton[role="primary"], QPushButton[role="large"] {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
    color: {ON_ACCENT};
    font-weight: 600;
}}
QPushButton[role="primary"]:hover, QPushButton[role="large"]:hover {{
    background: {ACCENT_HOVER};
    border-color: {ACCENT_HOVER};
}}
QPushButton[role="primary"]:pressed, QPushButton[role="large"]:pressed {{
    background: {ACCENT_PRESSED};
}}
QPushButton[role="primary"]:disabled {{
    background: {ACCENT_SOFT};
    border-color: {ACCENT_SOFT};
    color: {DISABLED};
}}
QPushButton[role="large"] {{
    font-size: 18px;
    border-radius: 12px;
    padding: 16px 40px;
}}
QPushButton[role="danger"] {{
    background: transparent;
    color: {DANGER};
    border-color: #6b3036;
}}
QPushButton[role="danger"]:hover {{ background: {DANGER_SOFT}; }}
QPushButton[role="danger"]:disabled {{ color: {DISABLED}; border-color: {HOVER}; }}

QLineEdit, QComboBox, QAbstractSpinBox {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 10px;
    min-height: 22px;
}}
QLineEdit:focus, QComboBox:focus, QAbstractSpinBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {{
    color: {DISABLED};
}}
QComboBox {{ padding-right: 28px; }}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 28px;
    border: none;
}}
QComboBox::down-arrow {{ image: url({CHEVRON_DOWN}); width: 12px; height: 12px; }}
QComboBox QAbstractItemView {{
    background: {FIELD};
    border: 1px solid {BORDER};
    outline: none;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
}}

QAbstractSpinBox {{ padding-right: 26px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
    subcontrol-origin: border;
    width: 24px;
    border: none;
    background: transparent;
}}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {HOVER};
    border-radius: 6px;
}}
QAbstractSpinBox::up-arrow {{ image: url({CHEVRON_UP}); width: 10px; height: 10px; }}
QAbstractSpinBox::down-arrow {{ image: url({CHEVRON_DOWN}); width: 10px; height: 10px; }}

QSlider {{ min-height: 26px; }}
QSlider::groove:horizontal {{
    height: 6px;
    background: {BORDER};
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: {TEXT};
    border: 3px solid {ACCENT};
    width: 12px;
    height: 12px;
    margin: -6px 0;
    border-radius: 9px;
}}

QRadioButton {{ spacing: 8px; padding: 4px 12px 4px 0; }}

QTableWidget {{
    background: {SURFACE};
    alternate-background-color: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 10px;
}}
QTableWidget::item {{ padding: 6px; }}
QHeaderView::section {{
    background: {SURFACE};
    color: {MUTED};
    font-weight: 600;
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 8px;
}}

QFrame[role="card"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
}}
QLabel[role="statValue"] {{ font-size: 26px; font-weight: 700; }}

QListWidget#sessionList {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    outline: none;
    padding: 4px;
}}
QListWidget#sessionList::item {{
    padding: 8px 10px;
    border-radius: 8px;
    color: {MUTED};
}}
QListWidget#sessionList::item:hover {{ background: {HOVER}; color: {TEXT}; }}
QListWidget#sessionList::item:selected {{ background: {ACCENT_SOFT}; color: {TEXT}; }}

QTextBrowser {{
    background: transparent;
    border: none;
}}

QStatusBar {{ color: {MUTED}; background: {SURFACE}; border-top: 1px solid {BORDER}; }}
QToolTip {{ background: {FIELD}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the theme to the whole application.

    The Fusion style draws widgets identically on every OS and fully obeys
    the stylesheet; the palette colors the parts the stylesheet does not
    cover, such as dialogs and checkboxes.
    """
    app.setStyle("Fusion")
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: BACKGROUND,
        QPalette.ColorRole.WindowText: TEXT,
        QPalette.ColorRole.Base: FIELD,
        QPalette.ColorRole.AlternateBase: SURFACE,
        QPalette.ColorRole.Text: TEXT,
        QPalette.ColorRole.Button: FIELD,
        QPalette.ColorRole.ButtonText: TEXT,
        QPalette.ColorRole.Highlight: ACCENT,
        QPalette.ColorRole.HighlightedText: ON_ACCENT,
        QPalette.ColorRole.PlaceholderText: DISABLED,
        QPalette.ColorRole.Link: ACCENT_HOVER,
        QPalette.ColorRole.ToolTipBase: FIELD,
        QPalette.ColorRole.ToolTipText: TEXT,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


def set_role(widget: QWidget, role: str) -> None:
    """Assign a style role and redraw the widget with it.

    Qt does not restyle a widget automatically when a property changes,
    hence the unpolish/polish pair.
    """
    widget.setProperty("role", role)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def page_title(text: str) -> QLabel:
    """Create the large title shown at the top of every page."""
    label = QLabel(text)
    set_role(label, "pageTitle")
    return label
