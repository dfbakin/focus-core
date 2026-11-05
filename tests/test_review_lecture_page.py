"""Review tests for the lecture page logic (app/ui/pages/lecture_page.py).

The page is driven without a camera, a model or a real worker: the session
clock is replaced by a fake one and the worker by a stub, so every test is
deterministic and runs offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from datetime import datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app.core.analysis import ExponentialMovingAverage  # noqa: E402
from app.core.report import SessionReport  # noqa: E402
from app.core.session import SessionClock, SessionState  # noqa: E402
from app.core.video_source import CAMERA, FILE  # noqa: E402
from app.db.local import init_db  # noqa: E402
from app.db.sessions import create_session, list_sessions, load_engagement_points  # noqa: E402
from app.ui.pages.lecture_page import LecturePage  # noqa: E402

START = datetime(2026, 9, 28, 14, 0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class StubWorker:
    """Stands in for the analysis thread; `stop` can take simulated time."""

    def __init__(self, clock: FakeClock, stop_takes_sec: float = 0.0) -> None:
        self._clock = clock
        self._stop_takes = stop_takes_sec
        self.paused: bool | None = None
        self.stopped = False

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def stop(self) -> None:
        self.stopped = True
        self._clock.now += self._stop_takes


@pytest.fixture(scope="module", autouse=True)
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


def _running_page(
    db_path: Path, clock: FakeClock, source_kind: str = CAMERA, stop_takes_sec: float = 0.0
) -> LecturePage:
    """A page in the RUNNING state with a stored session and a stub worker."""
    page = LecturePage(db_path=db_path)
    page._session_id = create_session(source_kind, 40, db_path, started_at=START)
    page._is_file = source_kind == FILE
    page._smoother = ExponentialMovingAverage(1)
    page._points = []
    page._clock = SessionClock(clock)
    page._clock.start()
    page._worker = StubWorker(clock, stop_takes_sec)
    return page


def test_closing_window_during_lecture_saves_the_session(db_path: Path) -> None:
    """Bug: MainWindow.closeEvent -> LecturePage.shutdown() only stops the thread.

    The session row stays unfinished (hidden from history and reports) and
    the in-memory engagement series is discarded: closing the window after
    a 90-minute lecture silently loses all of it.
    """
    clock = FakeClock()
    page = _running_page(db_path, clock)
    clock.now = 10.0
    page._record_estimate(page._clock.elapsed_exact(), 55.0)
    clock.now = 20.0

    page.shutdown()

    [session] = list_sessions(db_path)
    assert session.duration_sec == 20
    assert session.avg_engagement == 55.0
    assert load_engagement_points(session.id, db_path) == [(10.0, 55.0)]
    assert page._clock.state == SessionState.IDLE


def test_stored_duration_excludes_time_spent_stopping_the_worker(db_path: Path) -> None:
    """Bug: _finish_session stops the worker (blocking up to 3 s for the running
    inference) before reading the clock, so that wait is added to the lecture's
    active time."""
    clock = FakeClock()
    page = _running_page(db_path, clock, stop_takes_sec=2.5)
    clock.now = 10.0

    page._finish_session()

    [session] = list_sessions(db_path)
    assert session.duration_sec == 10


@pytest.mark.xfail(
    strict=True,
    reason="review: design decision - for a video file the estimate's offset is video "
    "time, so an estimate that finishes while the user has paused should be kept; "
    "currently _record_estimate drops it and the series has a hole at that offset",
)
def test_file_estimate_finishing_during_pause_is_kept(db_path: Path) -> None:
    clock = FakeClock()
    page = _running_page(db_path, clock, source_kind=FILE)
    page._record_estimate(5.0, 60.0)
    page._on_pause_clicked()
    assert page._clock.state == SessionState.PAUSED

    page._record_estimate(6.0, 40.0)  # the clip read before the pause finishes now

    assert [offset for offset, _ in page._points] == [5.0, 6.0]


@pytest.mark.xfail(
    strict=True,
    reason="review: design decision - points store active (pause-free) offsets, but "
    "SessionReport.clock_time and the PDF treat them as wall-clock offsets from "
    "started_at, so the listed times of critical moments are wrong after any pause "
    "(and meaningless for a video file)",
)
def test_report_clock_time_accounts_for_pauses(db_path: Path) -> None:
    clock = FakeClock()
    page = _running_page(db_path, clock)
    clock.now = 20.0
    page._on_pause_clicked()
    clock.now = 80.0  # one minute of pause
    page._on_pause_clicked()
    clock.now = 90.0
    page._record_estimate(page._clock.elapsed_exact(), 20.0)

    report = SessionReport("id", START, None, 40, page._points)
    offset = page._points[-1][0]

    assert report.clock_time(offset) == START + timedelta(seconds=90)
