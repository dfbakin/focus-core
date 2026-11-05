"""Help section describing every user action (criterion: UX)."""

from html import escape

from PyQt6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from app.ui.help_content import FAQ, HELP_SECTIONS


def build_help_html() -> str:
    """Render the help texts as simple HTML for QTextBrowser."""
    parts = []
    for title, items in HELP_SECTIONS:
        parts.append(f"<h3>{escape(title)}</h3><ul>")
        parts.extend(f"<li>{escape(item)}</li>" for item in items)
        parts.append("</ul>")
    parts.append("<h3>Частые вопросы</h3>")
    for question, answer in FAQ:
        parts.append(f"<p><b>{escape(question)}</b><br>{escape(answer)}</p>")
    style = "<style>h3 { margin-top: 18px; } li, p { margin-bottom: 6px; }</style>"
    return style + "".join(parts)


class HelpPage(QWidget):
    """Scrollable description of all sections and frequent questions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        browser = QTextBrowser()
        browser.setHtml(build_help_html())
        layout = QVBoxLayout(self)
        layout.addWidget(browser)
