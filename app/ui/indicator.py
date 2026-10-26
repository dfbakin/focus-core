"""Large colored engagement indicator for the lecture screen."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from app.core.analysis import EngagementLevel
from app.ui.theme import set_role

CIRCLE_SIZE = 160
NEUTRAL_COLOR = "#3a3a40"
LEVEL_COLORS = {
    EngagementLevel.HIGH: "#16a34a",
    EngagementLevel.MEDIUM: "#eab308",
    EngagementLevel.LOW: "#dc2626",
}
LEVEL_TEXTS = {
    EngagementLevel.HIGH: "Высокая вовлечённость",
    EngagementLevel.MEDIUM: "Средняя вовлечённость",
    EngagementLevel.LOW: "Низкая вовлечённость",
}


class EngagementIndicator(QWidget):
    """A colored circle with a caption: green, yellow, red or neutral grey."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
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
        self._paint(LEVEL_COLORS[level])
        self.caption.setText(LEVEL_TEXTS[level])

    def show_neutral(self, caption: str) -> None:
        """Grey circle for states without an estimate: waiting, pause."""
        self._paint(NEUTRAL_COLOR)
        self.caption.setText(caption)

    def _paint(self, color: str) -> None:
        """Fill the circle with a color."""
        radius = CIRCLE_SIZE // 2
        self.circle.setStyleSheet(f"background-color: {color}; border-radius: {radius}px;")
