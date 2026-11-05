"""Visual theme: the native macOS style in a dark graphite look with one blue accent.

The approach follows two public sources:
    Apple Human Interface Guidelines, «Sidebars» and «Color»: a sidebar
        with a rounded selection, dark neutral grays and one accent color;
    «How to Use System Default Colors in Qt Stylesheets» (pythonguis.com):
        colors are defined once, given to the application palette and
        reused in the stylesheet.

Lists, tables and drop-down lists keep the «macos» style of Qt, which
draws them with the system's own controls; the application asks macOS for
the dark appearance. The stylesheet repaints what would otherwise show the
light system grays or the bright system blue (buttons, text fields,
sliders, scroll bars, the progress bar) and describes what macOS has no ready-made
widget for: the sidebar, cards, the big start button and status texts.
Widgets get a "role" (a Qt dynamic property), and the stylesheet decides
how every role looks.
"""

from dataclasses import dataclass

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QFrame,
    QLabel,
    QStyleFactory,
    QVBoxLayout,
    QWidget,
)

NATIVE_STYLE = "macos"
FALLBACK_STYLE = "Fusion"
CARD_WIDTH = 760
FONT_FAMILY = "Helvetica Neue"
FONT_SIZE = 14


@dataclass(frozen=True)
class Colors:
    """Colors of the application, as '#rrggbb' strings.

    A near-black background, dark gray buttons, one navy accent with a
    lighter periwinkle tone of the same blue for lines and the selected
    section, and muted status colors: emerald, dark amber and burgundy
    instead of the bright system green, orange and red.
    """

    background: str = "#0c0c0f"
    sidebar: str = "#111114"
    surface: str = "#15151a"
    raised: str = "#1d1d23"
    separator: str = "#26262d"
    text: str = "#e6e6ec"
    secondary: str = "#8b8b95"
    accent: str = "#233a73"
    accent_pressed: str = "#1b2e5c"
    accent_light: str = "#6d8cf0"
    selection: str = "#18203a"
    selection_border: str = "#2c3c6e"
    on_accent: str = "#ffffff"
    green: str = "#2e9e6b"
    yellow: str = "#c9a227"
    orange: str = "#c8741e"
    red: str = "#a8293d"


COLORS = Colors()


def colors() -> Colors:
    """Colors of the application."""
    return COLORS


def build_stylesheet(c: Colors) -> str:
    """Describe the window background and the custom roles."""
    return f"""
QMainWindow, QWidget#content, QStatusBar {{ background: {c.background}; }}
QStatusBar {{ color: {c.secondary}; }}
QScrollArea#pageScroll, QScrollArea#pageScroll > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {c.separator}; border-radius: 3px; }}
QScrollBar::handle:hover {{ background: {c.secondary}; }}
QScrollBar::handle:vertical {{ min-height: 32px; }}
QScrollBar::handle:horizontal {{ min-width: 32px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QWidget#sidebar {{ background: {c.sidebar}; border-right: 1px solid {c.separator}; }}
QListWidget#menu {{ background: transparent; border: none; outline: none; }}
QListWidget#menu {{ font-size: 15px; }}
QListWidget#menu::item {{ padding: 0 10px; border: 1px solid transparent; border-radius: 8px; }}
QListWidget#menu::item:selected {{
    background: {c.selection};
    border: 1px solid {c.selection_border};
    color: {c.accent_light};
}}
QToolButton#sidebarToggle {{ border: none; font-size: 16px; padding: 2px 6px; color: {c.secondary}; }}

QLabel[role="pageTitle"] {{ font-size: 24px; font-weight: 700; padding-bottom: 8px; }}
QLabel[role="hint"] {{ color: {c.secondary}; }}
QLabel[role="warning"] {{ color: {c.orange}; }}
QLabel[role="error"] {{ color: {c.red}; }}
QLabel[role="success"] {{ color: {c.green}; }}
QLabel[role="emptyState"] {{ color: {c.secondary}; font-size: 15px; padding: 40px; }}
QLabel[role="timer"] {{ font-size: 44px; font-weight: 600; }}
QLabel[role="statusIdle"] {{ font-size: 15px; color: {c.secondary}; }}
QLabel[role="statusRunning"] {{ font-size: 15px; font-weight: 600; color: {c.green}; }}
QLabel[role="statusPaused"] {{ font-size: 15px; font-weight: 600; color: {c.orange}; }}
QLabel[role="statusError"] {{ font-size: 15px; font-weight: 600; color: {c.red}; }}
QLabel[role="indicatorCaption"] {{ font-size: 16px; font-weight: 600; }}
QLabel[role="statValue"] {{ font-size: 24px; font-weight: 700; }}
QLabel[role="preview"] {{ background: black; color: {c.secondary}; border-radius: 10px; }}

QLineEdit {{
    background: {c.background};
    border: 1px solid {c.separator};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {c.accent};
}}
QLineEdit:focus {{ border: 1px solid {c.accent_light}; }}

QSlider::groove:horizontal {{ height: 4px; background: {c.separator}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {c.accent_light}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {c.text};
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
}}
QProgressBar {{
    background: {c.separator};
    border: none;
    border-radius: 6px;
    min-height: 18px;
    max-height: 18px;
    text-align: center;
    color: {c.text};
}}
QProgressBar::chunk {{ background: {c.accent}; border-radius: 6px; }}

QFrame[role="card"] {{
    background: {c.surface};
    border: 1px solid {c.separator};
    border-radius: 10px;
}}

QPushButton {{
    background: {c.raised};
    color: {c.text};
    border: 1px solid {c.separator};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:pressed {{ background: {c.surface}; }}
QPushButton:disabled {{ color: {c.secondary}; }}

QPushButton[role="primary"], QPushButton[role="large"] {{
    background: {c.accent};
    color: {c.on_accent};
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 600;
}}
QPushButton[role="primary"]:pressed, QPushButton[role="large"]:pressed {{
    background: {c.accent_pressed};
}}
QPushButton[role="primary"]:disabled {{ background: {c.separator}; color: {c.secondary}; }}
QPushButton[role="large"] {{ font-size: 17px; border-radius: 10px; padding: 14px 40px; }}
QPushButton[role="danger"] {{ color: {c.red}; }}
"""


def apply_theme(app: QApplication) -> None:
    """Use the native macOS style in the dark appearance with the application colors and font.

    FONT_FAMILY empty means the system font of macOS (SF Pro).

    On other systems, where the macOS style does not exist, Fusion is used.
    The dark appearance is requested from the system where Qt supports it
    (Qt 6.8 and newer); the palette makes Qt-drawn parts, such as list
    selections and links, use the same colors.
    """
    available = [name.lower() for name in QStyleFactory.keys()]
    app.setStyle(NATIVE_STYLE if NATIVE_STYLE in available else FALLBACK_STYLE)
    hints = app.styleHints()
    if hasattr(hints, "setColorScheme"):
        hints.setColorScheme(Qt.ColorScheme.Dark)
    app.setPalette(_palette(COLORS))
    font = QFont(FONT_FAMILY) if FONT_FAMILY else app.font()
    font.setPixelSize(FONT_SIZE)
    app.setFont(font)
    app.setStyleSheet(build_stylesheet(COLORS))


def _palette(c: Colors) -> QPalette:
    """Palette with the application colors for the parts Qt draws itself."""
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: c.background,
        QPalette.ColorRole.WindowText: c.text,
        QPalette.ColorRole.Base: c.surface,
        QPalette.ColorRole.AlternateBase: c.sidebar,
        QPalette.ColorRole.Text: c.text,
        QPalette.ColorRole.Button: c.raised,
        QPalette.ColorRole.ButtonText: c.text,
        QPalette.ColorRole.PlaceholderText: c.secondary,
        QPalette.ColorRole.Highlight: c.accent,
        QPalette.ColorRole.HighlightedText: c.on_accent,
        QPalette.ColorRole.Accent: c.accent,
        QPalette.ColorRole.Link: c.accent_light,
        QPalette.ColorRole.Mid: c.separator,
    }
    for role, color in roles.items():
        palette.setColor(role, QColor(color))
    return palette


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


def form_layout() -> QFormLayout:
    """Form with labels on the left, aligned to the left edge of the page.

    On macOS a QFormLayout is centered and right-aligns its labels by
    default, which looks out of place on a page whose title is on the left.
    """
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    form.setHorizontalSpacing(20)
    form.setVerticalSpacing(14)
    return form


def card(heading: str = "") -> tuple[QFrame, QVBoxLayout]:
    """Create a rounded panel that groups related settings, like in System Settings.

    Returns the panel and its layout, into which the caller adds widgets.
    The width is limited so that fields do not stretch across a wide window.
    """
    frame = QFrame()
    set_role(frame, "card")
    frame.setMaximumWidth(CARD_WIDTH)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(28, 24, 28, 24)
    layout.setSpacing(14)
    if heading:
        title = QLabel(heading)
        set_role(title, "indicatorCaption")
        layout.addWidget(title)
    return frame, layout
