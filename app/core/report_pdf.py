"""Export of a lecture report to a one-page PDF (user scenario 5).

Uses the object-oriented matplotlib API without pyplot, so it works without
a display and does not interfere with the Qt event loop. Fonts are embedded
as TrueType, so the Cyrillic text in the PDF can be selected and copied.
"""

from datetime import datetime
from pathlib import Path

import matplotlib
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
MAX_LISTED_MOMENTS = 16
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
    _draw_moments(figure, report)
    figure.text(
        LEFT, 0.03, f"Сформировано FocusCore {format_datetime(datetime.now())}",
        fontsize=7, color=MUTED_COLOR,
    )
    with matplotlib.rc_context({"pdf.fonttype": EMBEDDED_FONT_TYPE}):
        figure.savefig(path, format="pdf")


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


def _draw_moments(figure: Figure, report: SessionReport) -> None:
    """List of critical moments with wall-clock times."""
    figure.text(LEFT, 0.46, "Критические моменты", fontsize=13, weight="bold", color=TEXT_COLOR)
    moments = report.critical_moments
    if not moments:
        figure.text(LEFT, 0.43, "Критических моментов не зафиксировано.", fontsize=10,
                    color=MUTED_COLOR)
        return

    y = 0.43
    for moment in moments[:MAX_LISTED_MOMENTS]:
        start = format_time(report.clock_time(moment.start_sec))
        end = format_time(report.clock_time(moment.end_sec))
        figure.text(
            LEFT, y, f"{start} – {end}    минимум {format_engagement(moment.min_value)}",
            fontsize=10, color=TEXT_COLOR,
        )
        y -= 0.022
    hidden = len(moments) - MAX_LISTED_MOMENTS
    if hidden > 0:
        figure.text(LEFT, y, f"… и ещё {hidden}", fontsize=10, color=MUTED_COLOR)
