"""Engagement-over-time chart shared by the monitoring and report pages."""

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QWidget

from app.core.report import CriticalMoment
from app.ui import theme

SECONDS_IN_MINUTE = 60
CRITICAL_ALPHA = 40

pg.setConfigOptions(antialias=True)


class EngagementChart(pg.PlotWidget):
    """Curve of engagement 0..100 over minutes with a dashed threshold line.

    Optionally marks gaps with red crosses and shades critical moments.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_axes()
        self.threshold_line = pg.InfiniteLine(
            angle=0,
            movable=False,
            pen=pg.mkPen(theme.WARNING, width=1.5, style=Qt.PenStyle.DashLine),
            label="порог {value:.0f} %",
            labelOpts={"color": theme.WARNING, "position": 0.05},
        )
        self.addItem(self.threshold_line)
        self.curve = self.plot(pen=pg.mkPen(theme.TEXT, width=2.5))
        self.gap_marks = pg.ScatterPlotItem(
            symbol="x", size=12, pen=pg.mkPen(theme.DANGER, width=2)
        )
        self.addItem(self.gap_marks)
        self._regions: list[pg.LinearRegionItem] = []

    def _setup_axes(self) -> None:
        """Dark background, fixed 0..100 range, muted axes, no mouse zoom."""
        self.setBackground(theme.SURFACE)
        self.setMouseEnabled(x=False, y=False)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.setYRange(0, 100, padding=0.02)
        self.setXRange(0, 1, padding=0)
        self.showGrid(x=True, y=True, alpha=0.12)
        axis_pen = pg.mkPen(theme.MUTED)
        for side, text in (("left", "Вовлечённость, %"), ("bottom", "Время, мин")):
            axis = self.getAxis(side)
            axis.setPen(axis_pen)
            axis.setTextPen(axis_pen)
            self.setLabel(side, text, color=theme.MUTED)

    def set_threshold(self, threshold_pct: float) -> None:
        """Move the dashed threshold line."""
        self.threshold_line.setPos(threshold_pct)

    def set_series(
        self,
        points: list[tuple[int, float]],
        gaps: list[tuple[int, float]] | None = None,
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

    def highlight_moments(self, moments: list[CriticalMoment]) -> None:
        """Shade the critical moments; previous shading is removed."""
        for region in self._regions:
            self.removeItem(region)
        self._regions = []

        color = QColor(theme.DANGER)
        color.setAlpha(CRITICAL_ALPHA)
        for moment in moments:
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
