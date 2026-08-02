import time

from database.db import enqueue_write, get_session
from database.models import DetectionEvent, User


def test_seeded_users_exist():
    with get_session() as db:
        usernames = {u.username for u in db.query(User).all()}
    assert {"admin", "operator"} <= usernames


def test_enqueue_write_persists_via_background_thread():
    def _write(db):
        db.add(DetectionEvent(detection_type="test_event", confidence=0.5))

    enqueue_write(_write)

    deadline = time.time() + 3
    found = False
    while time.time() < deadline:
        with get_session() as db:
            if db.query(DetectionEvent).filter(DetectionEvent.detection_type == "test_event").first():
                found = True
                break
        time.sleep(0.05)
    assert found, "DB writer thread did not persist the enqueued event in time"
