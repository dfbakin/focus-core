"""Export of a lecture report to PDF (user scenarios 5 and 6).

The first page holds the summary and the chart; the list of critical
moments with transcripts and comments continues on further pages when it
does not fit. Uses the object-oriented matplotlib API without pyplot, so it
works without a display and does not interfere with the Qt event loop.
Fonts are embedded as TrueType, so the Cyrillic text can be selected.
"""

import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from app.core.formatting import (
    format_datetime,
    format_duration,
    format_engagement,
    format_time,
)
from app.core.report import SessionReport

A4_INCHES = (8.27, 11.69)
LEFT = 0.08
TEXT_COLOR = "#1f2430"
MUTED_COLOR = "#6b7280"
THRESHOLD_COLOR = "#b45309"
CRITICAL_COLOR = "#dc2626"
WRAP_CHARACTERS = 100
LINE_STEP = 0.019
PARAGRAPH_GAP = 0.012
PAGE_TOP = 0.94
PAGE_BOTTOM = 0.07
SECONDS_IN_MINUTE = 60
EMBEDDED_FONT_TYPE = 42


def export_report_pdf(report: SessionReport, path: Path) -> None:
    """Draw the report on an A4 page and save it as PDF.

    Raises:
        ValueError: The report has no data to show.
        OSError: The file cannot be written, e.g. no write permission.
    """
    if not report.has_data:
        raise ValueError("Report has no data")

    figure = Figure(figsize=A4_INCHES)
    _draw_header(figure, report)
    _draw_summary(figure, report)
    _draw_chart(figure, report)
    pages = _draw_moments(figure, report)

    footer = f"Сформировано FocusCore {format_datetime(datetime.now())}"
    with matplotlib.rc_context({"pdf.fonttype": EMBEDDED_FONT_TYPE}):
        with PdfPages(path) as pdf:
            for number, page in enumerate(pages, start=1):
                page.text(LEFT, 0.03, footer, fontsize=7, color=MUTED_COLOR)
                page.text(1 - LEFT, 0.03, str(number), fontsize=7, color=MUTED_COLOR,
                          ha="right")
                pdf.savefig(page)


def _draw_header(figure: Figure, report: SessionReport) -> None:
    """Title and the date of the lecture."""
    figure.text(LEFT, 0.945, "Отчёт о занятии", fontsize=20, weight="bold", color=TEXT_COLOR)
    figure.text(
        LEFT, 0.922, f"Начало: {format_datetime(report.started_at)}",
        fontsize=10, color=MUTED_COLOR,
    )


def _draw_summary(figure: Figure, report: SessionReport) -> None:
    """Four key numbers in a row."""
    items = [
        (format_engagement(report.average), "Средняя вовлечённость"),
        (format_engagement(report.minimum), "Минимум"),
        (str(len(report.critical_moments)), "Критические моменты"),
        (format_duration(report.duration_sec), "Длительность"),
    ]
    for column, (value, caption) in enumerate(items):
        x = LEFT + column * 0.22
        figure.text(x, 0.87, value, fontsize=17, weight="bold", color=TEXT_COLOR)
        figure.text(x, 0.852, caption, fontsize=8, color=MUTED_COLOR)


def _draw_chart(figure: Figure, report: SessionReport) -> None:
    """Engagement timeline with the threshold and critical moments shaded."""
    axes = figure.add_axes((LEFT, 0.53, 0.86, 0.28))
    minutes = [offset / SECONDS_IN_MINUTE for offset, _ in report.points]
    values = [value for _, value in report.points]

    axes.plot(minutes, values, color=TEXT_COLOR, linewidth=1.8, marker="o", markersize=2.5)
    axes.axhline(
        report.threshold_pct, color=THRESHOLD_COLOR, linestyle="--", linewidth=1,
        label=f"Порог {report.threshold_pct} %",
    )
    for moment in report.critical_moments:
        axes.axvspan(
            moment.start_sec / SECONDS_IN_MINUTE,
            max(moment.end_sec, moment.start_sec + 1) / SECONDS_IN_MINUTE,
            color=CRITICAL_COLOR, alpha=0.12, linewidth=0,
        )

    axes.set_ylim(0, 100)
    axes.set_xlim(0, max(minutes[-1], 1 / SECONDS_IN_MINUTE))
    axes.set_xlabel("Время от начала занятия, мин", fontsize=9, color=MUTED_COLOR)
    axes.set_ylabel("Вовлечённость, %", fontsize=9, color=MUTED_COLOR)
    axes.tick_params(labelsize=8, colors=MUTED_COLOR)
    axes.grid(alpha=0.2)
    axes.legend(loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


class _TextFlow:
    """Writes lines from top to bottom and starts a new page when needed."""

    def __init__(self, first_page: Figure, top: float) -> None:
        self.pages = [first_page]
        self._y = top

    def line(self, text: str, **style) -> None:
        """Write one line of text."""
        if self._y < PAGE_BOTTOM:
            self._new_page()
        self.pages[-1].text(LEFT, self._y, text, **style)
        self._y -= LINE_STEP

    def paragraph(self, text: str, **style) -> None:
        """Write text wrapped to the page width."""
        for line in textwrap.wrap(text, WRAP_CHARACTERS):
            self.line(line, **style)

    def gap(self) -> None:
        """Leave some space between blocks."""
        self._y -= PARAGRAPH_GAP

    def _new_page(self) -> None:
        """Add a page that continues the list of moments."""
        page = Figure(figsize=A4_INCHES)
        page.text(LEFT, PAGE_TOP, "Критические моменты (продолжение)", fontsize=13,
                  weight="bold", color=TEXT_COLOR)
        self.pages.append(page)
        self._y = PAGE_TOP - 0.04


def _draw_moments(figure: Figure, report: SessionReport) -> list[Figure]:
    """List critical moments with wall-clock times, transcripts and comments.

    Returns all pages of the report, starting with `figure`.
    """
    figure.text(LEFT, 0.46, "Критические моменты", fontsize=13, weight="bold", color=TEXT_COLOR)
    flow = _TextFlow(figure, top=0.43)
    moments = report.critical_moments
    if not moments:
        flow.line("Критических моментов не зафиксировано.", fontsize=10, color=MUTED_COLOR)
        return flow.pages

    for moment in moments:
        start = format_time(report.clock_time(moment.start_sec))
        end = format_time(report.clock_time(moment.end_sec))
        flow.line(
            f"{start} – {end}    минимум {format_engagement(moment.min_value)}",
            fontsize=10, weight="bold", color=TEXT_COLOR,
        )
        note = report.note_for(moment)
        for part in note.transcript_parts:
            flow.paragraph(f"«{part}»", fontsize=9, color=TEXT_COLOR)
        if note.comment:
            flow.paragraph(f"Комментарий: {note.comment}", fontsize=9, color=MUTED_COLOR,
                           style="italic")
        flow.gap()
    return flow.pages
