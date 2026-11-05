"""Skip test modules whose heavy libraries are not installed.

The CI job installs only requirements-test.txt: the analysis core, the
database and the cloud client. Tests of the interface need PyQt6, tests of
the training code need torch and lightning; they run where those libraries
are installed, for example on the developer's machine, and are skipped in CI
instead of failing on import.
"""

import importlib.util

QT_TESTS = [
    "test_review_lecture_page.py",
    "test_review_profile_page.py",
    "test_review_reports_page.py",
    "test_review_workers.py",
]
TRAINING_TESTS = ["test_src_*.py"]


def _missing(*modules: str) -> bool:
    """True if any of the modules cannot be imported."""
    return any(importlib.util.find_spec(module) is None for module in modules)


collect_ignore_glob = []
if _missing("PyQt6"):
    collect_ignore_glob += QT_TESTS
if _missing("torch", "lightning"):
    collect_ignore_glob += TRAINING_TESTS
