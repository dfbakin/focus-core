"""Tests for UI formatting helpers."""

from datetime import datetime

import pytest

from app.core.formatting import (
    date_matches,
    format_clock,
    format_duration,
    format_engagement,
    format_days,
    format_memory,
    format_size,
    plural,
)

MOMENT = datetime(2026, 9, 26, 14, 5)


def test_clock_pads_all_parts() -> None:
    assert format_clock(3723) == "01:02:03"


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "—"), (30, "< 1 мин"), (45 * 60, "45 мин"), (65 * 60, "1 ч 05 мин")],
)
def test_duration(seconds: int | None, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_engagement_is_rounded() -> None:
    assert format_engagement(62.6) == "63 %"
    assert format_engagement(None) == "—"


def test_memory_switches_to_gigabytes() -> None:
    assert format_memory(812) == "812 МБ"
    assert format_memory(3277) == "3,2 ГБ"


@pytest.mark.parametrize("query", ["", "  ", "26.09.2026", "09.2026", "26.09"])
def test_matching_date_queries(query: str) -> None:
    assert date_matches(MOMENT, query)


@pytest.mark.parametrize("query", ["27.09", "2025", "10.2026"])
def test_non_matching_date_queries(query: str) -> None:
    assert not date_matches(MOMENT, query)


@pytest.mark.parametrize(
    ("count", "word"),
    [(1, "сессия"), (2, "сессии"), (5, "сессий"), (11, "сессий"), (21, "сессия"), (104, "сессии")],
)
def test_plural(count: int, word: str) -> None:
    assert plural(count, "сессия", "сессии", "сессий") == word


def test_size_and_days() -> None:
    assert format_size(512) == "512 Б"
    assert format_size(24 * 1024) == "24 КБ"
    assert format_size(3 * 1024 * 1024 + 100 * 1024) == "3,1 МБ"
    assert format_days(7) == "7 дней"
    assert format_days(90) == "3 месяца"
    assert format_days(365) == "1 год"
