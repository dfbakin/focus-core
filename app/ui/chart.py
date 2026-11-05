"""Engagement-over-time chart shared by the monitoring and report pages."""

import pyqtgraph as pg
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from app.core.report import CriticalMoment
from app.ui import theme

SECONDS_IN_MINUTE = 60
CRITICAL_ALPHA = 40
SELECTED_ALPHA = 120

pg.setConfigOptions(antialias=True)


class EngagementChart(pg.PlotWidget):
    """Curve of engagement 0..100 over minutes with a dashed threshold line.

    Optionally marks gaps with crosses and shades critical moments. pyqtgraph
    does not follow the Qt style, so its colors are taken from the theme and
    taken again when the system switches between light and dark appearance.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._moments: list[CriticalMoment] = []
        self._selected: int | None = None
        self._regions: list[pg.LinearRegionItem] = []
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.setYRange(0, 100, padding=0.02)
        self.setXRange(0, 1, padding=0)
        self.showGrid(x=True, y=True, alpha=0.12)
        self.threshold_line = pg.InfiniteLine(angle=0, movable=False, label="порог {value:.0f} %")
        self.addItem(self.threshold_line)
        self.curve = self.plot()
        self.gap_marks = pg.ScatterPlotItem(symbol="x", size=12)
        self.addItem(self.gap_marks)
        self._apply_colors()

    def changeEvent(self, event: QEvent) -> None:
        """Recolor the chart when the system appearance changes."""
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "curve"):
            self._apply_colors()

    def _apply_colors(self) -> None:
        """Paint the background, axes and lines in the colors of the current appearance."""
        c = theme.colors()
        self.setBackground(c.surface)
        axis_pen = pg.mkPen(c.secondary)
        for side, text in (("left", "Вовлечённость, %"), ("bottom", "Время, мин")):
            axis = self.getAxis(side)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
            self.setLabel(side, text, color=c.secondary)
        self.threshold_line.setPen(pg.mkPen(c.orange, width=1.5, style=Qt.PenStyle.DashLine))
        self.threshold_line.label.setColor(c.orange)
        self.threshold_line.label.setPosition(0.92)
        self.curve.setPen(pg.mkPen(c.accent_light, width=2.5))
        self.gap_marks.setPen(pg.mkPen(c.red, width=2))
        self.highlight_moments(self._moments, self._selected)

    def set_threshold(self, threshold_pct: float) -> None:
        """Move the dashed threshold line."""
        self.threshold_line.setPos(threshold_pct)

    def set_series(
        self,
        points: list[tuple[float, float]],
        gaps: list[tuple[float, float]] | None = None,
    ) -> None:
        """Draw the curve from (offset_sec, value) points and mark gaps."""
        minutes = [offset / SECONDS_IN_MINUTE for offset, _ in points]
        values = [value for _, value in points]
        self.curve.setData(minutes, values)

        gaps = gaps or []
        self.gap_marks.setData(
            [offset / SECONDS_IN_MINUTE for offset, _ in gaps],
            [value for _, value in gaps],
        )
        last_minute = minutes[-1] if minutes else 0.0
        self.setXRange(0, max(1.0, last_minute), padding=0.02)

    def highlight_moments(
        self, moments: list[CriticalMoment], selected: int | None = None
    ) -> None:
        """Shade the critical moments, the selected one more strongly.

        Previous shading is removed.
        """
        self._moments = list(moments)
        self._selected = selected
        for region in self._regions:
            self.removeItem(region)
        self._regions = []

        for index, moment in enumerate(moments):
            color = QColor(theme.colors().red)
            color.setAlpha(SELECTED_ALPHA if index == selected else CRITICAL_ALPHA)
            end_sec = max(moment.end_sec, moment.start_sec + 1)
            region = pg.LinearRegionItem(
                values=(moment.start_sec / SECONDS_IN_MINUTE, end_sec / SECONDS_IN_MINUTE),
                movable=False,
                brush=color,
            )
            for line in region.lines:
                line.setPen(pg.mkPen(None))
            region.setZValue(-10)
            self.addItem(region)
            self._regions.append(region)
