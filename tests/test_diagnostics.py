"""Tests for the RAM check."""

from app.core.diagnostics import MemoryStatus, check_memory


def test_enough_when_available_reaches_required() -> None:
    assert MemoryStatus(available_mb=1024, required_mb=1024).is_enough
    assert not MemoryStatus(available_mb=1023, required_mb=1024).is_enough


def test_real_measurement_is_positive() -> None:
    status = check_memory(required_mb=1)
    assert status.available_mb > 0
    assert status.required_mb == 1
