"""Change the free RAM required to start a lecture.

Used to demonstrate and test the low-memory warning on a machine that has
plenty of free RAM. Works while the application is running: the value is
re-read on every check.

Run from the repository root:
    python -m scripts.set_min_ram 64000   (require 64000 MB)
    python -m scripts.set_min_ram         (restore the default)
"""

import sys

from app.core.diagnostics import DEFAULT_MIN_FREE_RAM_MB
from app.db.local import init_db
from app.db.settings import save_min_free_ram_mb


def main() -> None:
    """Store the value from the command line or the default one."""
    init_db()
    megabytes = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MIN_FREE_RAM_MB
    save_min_free_ram_mb(megabytes)
    print(f"Required free RAM: {megabytes} MB")


if __name__ == "__main__":
    main()
