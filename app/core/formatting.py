"""Human-readable formatting of dates, durations and values for the UI."""

from datetime import datetime

EMPTY_VALUE = "—"


def format_datetime(moment: datetime) -> str:
    """Format as '26.09.2026 14:05'."""
    return moment.strftime("%d.%m.%Y %H:%M")


def format_time(moment: datetime) -> str:
    """Format as '14:05:30'."""
    return moment.strftime("%H:%M:%S")


def format_clock(seconds: int) -> str:
    """Format elapsed seconds as a timer, '01:02:03'."""
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_seconds(seconds: int) -> str:
    """Format a short time span as '45 с', '2 мин 05 с' or '1 ч 03 мин'."""
    minutes, secs = divmod(max(0, seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} ч {minutes:02d} мин"
    if minutes:
        return f"{minutes} мин {secs:02d} с"
    return f"{secs} с"


def format_duration(seconds: int | None) -> str:
    """Format a duration as '1 ч 05 мин', '45 мин' or '< 1 мин'."""
    if seconds is None:
        return EMPTY_VALUE
    hours, rest = divmod(seconds, 3600)
    minutes = rest // 60
    if hours:
        return f"{hours} ч {minutes:02d} мин"
    if minutes:
        return f"{minutes} мин"
    return "< 1 мин"


def format_engagement(value: float | None) -> str:
    """Format an engagement score 0..100 as '63 %'."""
    if value is None:
        return EMPTY_VALUE
    return f"{value:.0f} %"


def format_memory(megabytes: int) -> str:
    """Format RAM as '812 МБ' or '3,2 ГБ'."""
    if megabytes < 1024:
        return f"{megabytes} МБ"
    return f"{megabytes / 1024:.1f} ГБ".replace(".", ",")


def date_matches(moment: datetime, query: str) -> bool:
    """True if the query is a part of the date written as '26.09.2026'.

    An empty query matches everything, so '09.2026' finds all of September.
    """
    query = query.strip()
    return not query or query in moment.strftime("%d.%m.%Y")


def plural(count: int, one: str, few: str, many: str) -> str:
    """Pick the Russian word form for a number: 1 сессия, 2 сессии, 5 сессий."""
    last_two = count % 100
    last = count % 10
    if 11 <= last_two <= 14:
        return many
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


def format_size(size_bytes: int) -> str:
    """Format a file size as '512 Б', '24 КБ' or '3,1 МБ'."""
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    kilobytes = size_bytes / 1024
    if kilobytes < 1024:
        return f"{kilobytes:.0f} КБ"
    return f"{kilobytes / 1024:.1f} МБ".replace(".", ",")


def format_days(days: int) -> str:
    """Format a retention period as '7 дней', '1 год' and so on."""
    if days == 365:
        return "1 год"
    if days % 30 == 0 and days >= 90:
        months = days // 30
        return f"{months} {plural(months, 'месяц', 'месяца', 'месяцев')}"
    return f"{days} {plural(days, 'день', 'дня', 'дней')}"
