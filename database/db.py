"""DB engine/session management plus a dedicated background writer thread.

Runs on its own thread (per the "separate threads" architecture requirement)
so detection/tracking/WebSocket threads never block on disk I/O. Writes are
submitted via `enqueue_write(fn)`; the worker drains a bounded queue so a
slow disk can never cause unbounded memory growth.
"""
from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from database.models import Base, User
from utils.logger import get_logger
from utils.security import hash_password

log = get_logger("database")

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

_write_queue: "queue.Queue[Callable[[Session], None]]" = queue.Queue(maxsize=1000)
_stop_event = threading.Event()
_worker_thread: threading.Thread | None = None


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    Base.metadata.create_all(engine)
    with get_session() as session:
        if session.query(User).count() == 0:
            session.add_all(
                [
                    User(username="admin", password_hash=hash_password("admin123"), role="admin"),
                    User(username="operator", password_hash=hash_password("operator123"), role="operator"),
                ]
            )
            session.commit()
            log.info("Seeded default users (admin/admin123, operator/operator123) — change these before production use.")


def _writer_loop() -> None:
    log.info("Database writer thread started")
    while not _stop_event.is_set():
        try:
            fn = _write_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            with get_session() as session:
                fn(session)
                session.commit()
        except Exception:
            log.exception("Database write failed; event dropped rather than blocking the pipeline")
        finally:
            _write_queue.task_done()
    log.info("Database writer thread stopped")


def start_writer_thread() -> None:
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    _stop_event.clear()
    _worker_thread = threading.Thread(target=_writer_loop, name="db-writer", daemon=True)
    _worker_thread.start()


def stop_writer_thread() -> None:
    _stop_event.set()
    if _worker_thread:
        _worker_thread.join(timeout=2)


def enqueue_write(fn: Callable[[Session], None]) -> None:
    try:
        _write_queue.put_nowait(fn)
    except queue.Full:
        log.warning("DB write queue full — dropping oldest-priority write to stay non-blocking")
        try:
            _write_queue.get_nowait()
            _write_queue.put_nowait(fn)
        except queue.Empty:
            pass
