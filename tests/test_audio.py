"""Tests for the in-memory audio buffer."""

import numpy as np
import pytest

from app.core.audio import AudioBuffer


def test_empty_buffer_has_no_duration() -> None:
    assert AudioBuffer(sample_rate=10).duration_sec == 0.0


def test_duration_grows_with_chunks() -> None:
    buffer = AudioBuffer(sample_rate=10)
    buffer.append(np.zeros(15))
    buffer.append(np.zeros(5))
    assert buffer.duration_sec == pytest.approx(2.0)


def test_clip_returns_samples_across_chunks() -> None:
    buffer = AudioBuffer(sample_rate=10)
    buffer.append(np.arange(0, 10))
    buffer.append(np.arange(10, 20))
    assert buffer.clip(0.5, 1.5).tolist() == list(range(5, 15))


def test_clip_is_clamped_to_the_recording() -> None:
    buffer = AudioBuffer(sample_rate=10)
    buffer.append(np.arange(10))
    assert buffer.clip(-1.0, 5.0).tolist() == list(range(10))
    assert len(buffer.clip(3.0, 4.0)) == 0


def test_clear_forgets_the_recording() -> None:
    buffer = AudioBuffer(sample_rate=10)
    buffer.append(np.ones(10))
    buffer.clear()
    assert buffer.duration_sec == 0.0
