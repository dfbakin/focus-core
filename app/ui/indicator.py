"""Large colored engagement indicator for the lecture screen."""

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.core.analysis import EngagementLevel
from app.ui import theme
from app.ui.theme import set_role

CIRCLE_SIZE = 160
LEVEL_COLOR_NAMES = {
    EngagementLevel.HIGH: "green",
    EngagementLevel.MEDIUM: "yellow",
    EngagementLevel.LOW: "red",
}
LEVEL_TEXTS = {
    EngagementLevel.HIGH: "Высокая вовлечённость",
    EngagementLevel.MEDIUM: "Средняя вовлечённость",
    EngagementLevel.LOW: "Низкая вовлечённость",
}


def level_color(level: EngagementLevel) -> str:
    """Apple system color of a level in the current appearance."""
    return getattr(theme.colors(), LEVEL_COLOR_NAMES[level])


class EngagementIndicator(QWidget):
    """A colored circle with a caption: green, yellow, red or neutral grey."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._level: EngagementLevel | None = None
        self.circle = QLabel()
        self.circle.setFixedSize(CIRCLE_SIZE, CIRCLE_SIZE)
        self.caption = QLabel()
        set_role(self.caption, "indicatorCaption")

        layout = QVBoxLayout(self)
        layout.addWidget(self.circle, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.caption, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.show_neutral("")

    def show_level(self, level: EngagementLevel) -> None:
        """Paint the circle in the color of the level and name the level."""
        self._level = level
        self._paint()
        self.caption.setText(LEVEL_TEXTS[level])

    def show_neutral(self, caption: str) -> None:
        """Grey circle for states without an estimate: waiting, pause."""
        self._level = None
        self._paint()
        self.caption.setText(caption)

    def changeEvent(self, event: QEvent) -> None:
        """Repaint when the system switches between light and dark appearance."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "circle"):
            self._paint()

    def _paint(self) -> None:
        """Fill the circle with the color of the current level, or grey."""
        if self._level is None:
            color = theme.colors().separator
        else:
            color = level_color(self._level)
        radius = CIRCLE_SIZE // 2
        self.circle.setStyleSheet(f"background-color: {color}; border-radius: {radius}px;")
