"""Thin bridge so the dashboard's Sanity Checks tab can run
tests/sanity_checks.py's checks live and show the captured PASS/FAIL
output, without duplicating the check logic."""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1] / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


def run_all() -> tuple[str, bool]:
    import sanity_checks  # tests/sanity_checks.py

    buf = io.StringIO()
    with redirect_stdout(buf):
        failures, ok = sanity_checks.run_all()
    return buf.getvalue(), ok
