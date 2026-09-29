"""Tests for the lecture session state machine."""

import pytest

from app.core.session import InvalidTransitionError, SessionClock, SessionState


class FakeClock:
    """Manually controlled time source."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def test_new_session_is_idle(clock: FakeClock) -> None:
    assert SessionClock(clock).state == SessionState.IDLE


def test_running_time_is_counted(clock: FakeClock) -> None:
    session = SessionClock(clock)
    session.start()
    clock.now = 90
    assert session.elapsed() == 90


def test_paused_time_is_not_counted(clock: FakeClock) -> None:
    session = SessionClock(clock)
    session.start()
    clock.now = 60
    session.pause()
    clock.now = 600
    assert session.elapsed() == 60
    session.resume()
    clock.now = 630
    assert session.finish() == 90
    assert session.state == SessionState.FINISHED


def test_finish_from_pause(clock: FakeClock) -> None:
    session = SessionClock(clock)
    session.start()
    clock.now = 10
    session.pause()
    clock.now = 100
    assert session.finish() == 10


@pytest.mark.parametrize("action", ["pause", "resume", "finish"])
def test_actions_before_start_are_rejected(clock: FakeClock, action: str) -> None:
    with pytest.raises(InvalidTransitionError):
        getattr(SessionClock(clock), action)()


def test_double_start_is_rejected(clock: FakeClock) -> None:
    session = SessionClock(clock)
    session.start()
    with pytest.raises(InvalidTransitionError):
        session.start()
