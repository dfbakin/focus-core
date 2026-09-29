"""Help section describing every user action (criterion: UX)."""

from html import escape

from PyQt6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from app.ui import theme
from app.ui.help_content import FAQ, HELP_SECTIONS
from app.ui.theme import page_title


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
    style = (
        f"<style>h3 {{ color: {theme.TEXT}; margin-top: 18px; }}"
        f"li, p {{ color: {theme.MUTED}; margin-bottom: 6px; }}"
        f"b {{ color: {theme.TEXT}; }}</style>"
    )
    return style + "".join(parts)


class HelpPage(QWidget):
    """Scrollable description of all sections and frequent questions."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        browser = QTextBrowser()
        browser.setHtml(build_help_html())
        layout = QVBoxLayout(self)
        layout.addWidget(page_title("Помощь"))
        layout.addWidget(browser)
