"""Test-only settings overrides — MUST run before any other module imports
`config.settings`, so it lives at the very top of conftest.py which pytest
always imports first. Isolates tests from the real dev database/logs and
shrinks workflow timers so timing-based state-machine tests run in
milliseconds instead of real seconds.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_tmp_dir = Path(tempfile.mkdtemp(prefix="omni_mind_test_"))
os.environ["DB_PATH"] = str(_tmp_dir / "test.db")
os.environ["LOG_DIR"] = str(_tmp_dir / "logs")
os.environ["EVIDENCE_DIR"] = str(_tmp_dir / "evidence")
os.environ["WEIGHTS_DIR"] = str(ROOT / "models" / "weights")
os.environ["JWT_SECRET"] = "test-secret-not-for-production-use"
os.environ["FIELD_ENCRYPTION_KEY"] = ""
os.environ["FACE_COVER_GRACE_SECONDS"] = "0.3"
os.environ["VOICE_WARNING_WAIT_SECONDS"] = "0.2"
os.environ["SIREN_DURATION_SECONDS"] = "0.3"
os.environ["CONFIRM_WINDOW_FRAMES"] = "5"
os.environ["CONFIRM_MIN_HITS"] = "4"

import pytest


@pytest.fixture(scope="session", autouse=True)
def _init_test_database():
    from database.db import init_db, start_writer_thread, stop_writer_thread

    init_db()
    start_writer_thread()
    yield
    stop_writer_thread()
