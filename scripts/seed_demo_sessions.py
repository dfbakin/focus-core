"""Fill the local database with demo lecture sessions.

Every session gets a synthetic engagement series, so history, reports and
PDF export can be shown without recording real lectures. The oldest session
has no series at all, to demonstrate the "not enough data" message.

Run from the repository root:
    python -m scripts.seed_demo_sessions
"""

import random
from datetime import datetime, timedelta

from app.core.analysis import summarize
from app.core.video_source import CAMERA
from app.db.local import init_db
from app.db.sessions import (
    create_session,
    finish_session,
    save_engagement_points,
    save_engagement_summary,
)

SESSION_COUNT = 10
RANDOM_SEED = 7
THRESHOLD_PCT = 40
POINT_STEP_SEC = 30


def synthetic_series(rng: random.Random, duration_sec: int) -> list[tuple[int, float]]:
    """A random walk with one dip in the middle of the lecture."""
    value = rng.uniform(55, 80)
    dip_start = rng.randint(duration_sec // 4, duration_sec // 2)
    dip_end = dip_start + rng.randint(180, 480)
    points = []
    for offset in range(0, duration_sec, POINT_STEP_SEC):
        pull = -3.0 if dip_start <= offset < dip_end else (1.5 if value < 60 else 0.0)
        value = min(98.0, max(5.0, value + rng.uniform(-6, 6) + pull))
        points.append((offset, round(value, 1)))
    return points


def main() -> None:
    """Create finished sessions spread over the last three weeks."""
    init_db()
    rng = random.Random(RANDOM_SEED)
    today = datetime.now().replace(minute=0, second=0, microsecond=0)

    for i in range(SESSION_COUNT):
        started = (today - timedelta(days=2 * i + 1)).replace(hour=rng.choice([9, 11, 13, 15]))
        duration_sec = rng.randint(40, 90) * 60
        session_id = create_session(CAMERA, THRESHOLD_PCT, started_at=started)
        finish_session(session_id, finished_at=started + timedelta(seconds=duration_sec))

        is_last = i == SESSION_COUNT - 1
        if is_last:
            continue
        points = synthetic_series(rng, duration_sec)
        save_engagement_points(session_id, points)
        average, minimum = summarize([value for _, value in points])
        save_engagement_summary(session_id, average, minimum)

    print(f"Added {SESSION_COUNT} demo sessions (the oldest one without data).")


if __name__ == "__main__":
    main()
