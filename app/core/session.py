"""Lecture session state machine: idle -> running <-> paused -> finished.

Tracks only the state and the active (non-paused) time. Knows nothing about
cameras, the model or the interface, so every transition is easy to test.
"""

import time
from collections.abc import Callable
from enum import Enum


class SessionState(Enum):
    """Possible states of a lecture session."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class InvalidTransitionError(Exception):
    """Raised when an action is not allowed in the current state."""


class SessionClock:
    """Lecture state together with the time spent in the RUNNING state.

    Args:
        clock: Function returning the current time in seconds. The default
            is time.monotonic, which never jumps when the system clock changes;
            tests pass a fake clock instead.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._state = SessionState.IDLE
        self._accumulated = 0.0
        self._resumed_at = 0.0

    @property
    def state(self) -> SessionState:
        """Current state of the session."""
        return self._state

    def start(self) -> None:
        """Begin the session: IDLE -> RUNNING."""
        self._require(SessionState.IDLE)
        self._state = SessionState.RUNNING
        self._resumed_at = self._clock()

    def pause(self) -> None:
        """Stop counting time: RUNNING -> PAUSED."""
        self._require(SessionState.RUNNING)
        self._accumulated += self._clock() - self._resumed_at
        self._state = SessionState.PAUSED

    def resume(self) -> None:
        """Continue counting time: PAUSED -> RUNNING."""
        self._require(SessionState.PAUSED)
        self._resumed_at = self._clock()
        self._state = SessionState.RUNNING

    def finish(self) -> int:
        """End the session from RUNNING or PAUSED and return active seconds."""
        self._require(SessionState.RUNNING, SessionState.PAUSED)
        if self._state == SessionState.RUNNING:
            self._accumulated += self._clock() - self._resumed_at
        self._state = SessionState.FINISHED
        return self.elapsed()

    def elapsed(self) -> int:
        """Return whole seconds spent in the RUNNING state so far."""
        return int(self.elapsed_exact())

    def elapsed_exact(self) -> float:
        """Return seconds spent in the RUNNING state so far, with fractions.

        Estimates can come several times per second, so their offsets
        need more precision than the timer on the screen.
        """
        total = self._accumulated
        if self._state == SessionState.RUNNING:
            total += self._clock() - self._resumed_at
        return total

    def _require(self, *allowed: SessionState) -> None:
        """Raise InvalidTransitionError unless the state is one of `allowed`."""
        if self._state not in allowed:
            raise InvalidTransitionError(f"Not allowed in state {self._state.value}")
