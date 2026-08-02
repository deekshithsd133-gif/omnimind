"""Notification dispatch tests — no real network I/O: smtplib.SMTP is
monkeypatched with a fake context manager so these run offline and fast."""
from __future__ import annotations

import time

from alerts.notifier import AlertPayload, _send_email, _send_police_alert, dispatch_alert_async
from config.settings import settings


class _FakeSMTP:
    sent = []

    def __init__(self, host, port, timeout=10):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        _FakeSMTP.sent.append(msg)


def _payload():
    return AlertPayload(
        detection_type="face_covered_siren",
        confidence=1.0,
        timestamp="2026-07-18 12:00:00",
        atm_id="ATM-001",
        location="Canara Bank ATM, Madanapalle",
        snapshot_path=None,
    )


def test_police_alert_skipped_when_not_configured(monkeypatch):
    monkeypatch.setattr(settings, "police_email", "")
    status, detail = _send_police_alert(_payload())
    assert status == "skipped_not_configured"


def test_police_alert_sends_when_configured(monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_user", "alerts@example.com")
    monkeypatch.setattr(settings, "smtp_password", "x")
    monkeypatch.setattr(settings, "police_email", "police@example.gov")
    monkeypatch.setattr(settings, "police_station_name", "Madanapalle Town Police Station")
    monkeypatch.setattr("alerts.notifier.smtplib.SMTP", _FakeSMTP)
    _FakeSMTP.sent.clear()

    status, detail = _send_police_alert(_payload())

    assert status == "sent"
    assert len(_FakeSMTP.sent) == 1
    msg = _FakeSMTP.sent[0]
    assert msg["To"] == "police@example.gov"
    assert "EMERGENCY" in msg["Subject"]
    assert "face_covered_siren" in msg["Subject"]


def test_police_alert_does_not_fire_admin_alert_email_alone(monkeypatch):
    """police_email and alert_email_to are independent — configuring one
    must not silently also start sending the other."""
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_user", "alerts@example.com")
    monkeypatch.setattr(settings, "smtp_password", "x")
    monkeypatch.setattr(settings, "alert_email_to", "")
    monkeypatch.setattr(settings, "police_email", "police@example.gov")

    status, _ = _send_email(_payload())
    assert status == "skipped_not_configured"


def test_dispatch_alert_async_attempts_all_three_channels(monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "")
    monkeypatch.setattr(settings, "police_email", "")
    monkeypatch.setattr(settings, "telegram_bot_token", "")

    seen = []
    dispatch_alert_async(_payload(), lambda channel, status, detail: seen.append((channel, status)))

    deadline = time.monotonic() + 2.0
    while len(seen) < 3 and time.monotonic() < deadline:
        time.sleep(0.02)

    channels = {c for c, _ in seen}
    assert channels == {"email", "telegram", "police"}
    assert all(status == "skipped_not_configured" for _, status in seen)
