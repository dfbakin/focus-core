"""Make a share of model estimates fail on purpose.

Used to test and demonstrate how the monitoring chart marks gaps
(user scenario 4, step 5). Applies to the next lecture that starts.

Run from the repository root:
    python -m scripts.set_error_rate 0.3   (fail about 30 % of estimates)
    python -m scripts.set_error_rate       (turn simulation off)
"""

import sys

from app.db.local import init_db
from app.db.settings import save_simulated_error_rate


def main() -> None:
    """Store the rate from the command line, or 0 to turn simulation off."""
    init_db()
    rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    if not 0.0 <= rate <= 1.0:
        raise SystemExit("Rate must be between 0 and 1")
    save_simulated_error_rate(rate)
    print(f"Simulated error rate: {rate:.0%}")


if __name__ == "__main__":
    main()
