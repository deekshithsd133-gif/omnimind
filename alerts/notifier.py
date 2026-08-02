"""Multi-channel notification dispatch: email (SMTP) and Telegram are real
integrations that activate automatically once credentials are present in
`.env`; anything left unconfigured logs a clear "skipped_not_configured"
event instead of pretending to send. SMS/WhatsApp/push are documented as a
pluggable `NotificationChannel` interface (e.g. wire up Twilio) rather than
implemented against a paid API this session has no credentials for.

The "police" channel below is the same real SMTP mechanism as the admin
email alert, just addressed to `settings.police_email` instead — there is
no automated phone/SMS dispatch to emergency services (that would need a
paid telephony API and a real integration with local emergency dispatch,
well beyond a prototype's scope), and none is claimed. `police_email` is
deliberately blank by default: it must be filled in with the real,
verified contact for this ATM's actual nearest police station, not a
guessed value.

Runs on a background daemon thread pool so a slow/unreachable SMTP or
Telegram endpoint can never block the detection pipeline.
"""
from __future__ import annotations

import smtplib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.message import EmailMessage

import httpx

from config.settings import settings
from utils.logger import get_logger

log = get_logger("notifier")

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="notifier")


@dataclass
class AlertPayload:
    detection_type: str
    confidence: float
    timestamp: str
    atm_id: str
    location: str
    snapshot_path: str | None = None


def _send_email(payload: AlertPayload) -> tuple[str, str]:
    if not (settings.smtp_host and settings.smtp_user and settings.alert_email_to):
        return "skipped_not_configured", "SMTP not configured"
    try:
        msg = EmailMessage()
        msg["Subject"] = f"[Omni Mind] {payload.detection_type} at {payload.atm_id}"
        msg["From"] = settings.smtp_user
        msg["To"] = settings.alert_email_to
        msg.set_content(
            f"Detection: {payload.detection_type}\nConfidence: {payload.confidence:.2f}\n"
            f"ATM: {payload.atm_id}\nLocation: {payload.location}\nTime: {payload.timestamp}"
        )
        if payload.snapshot_path:
            with open(payload.snapshot_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="snapshot.jpg")
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
        return "sent", "ok"
    except Exception as exc:
        log.exception("Email alert failed")
        return "failed", str(exc)


def _send_police_alert(payload: AlertPayload) -> tuple[str, str]:
    """Emergency notification for the ATM's configured nearest police
    station — only ever dispatched for the two highest-severity detection
    types (weapon lockdown, face-covering siren), since those are the only
    two call sites that invoke dispatch_alert_async at all
    (workflow/pipeline.py). Uses the same SMTP server/credentials as the
    admin alert, just a different recipient and a subject/body that makes
    clear this is an automated emergency alert requiring verification."""
    if not (settings.smtp_host and settings.smtp_user and settings.police_email):
        return "skipped_not_configured", "Police email not configured"
    try:
        msg = EmailMessage()
        msg["Subject"] = f"EMERGENCY ALERT — {payload.detection_type} at {payload.location}"
        msg["From"] = settings.smtp_user
        msg["To"] = settings.police_email
        station = f" ({settings.police_station_name})" if settings.police_station_name else ""
        msg.set_content(
            f"Automated emergency alert from an unattended ATM security system.\n"
            f"This alert has not been reviewed by a human — please verify before dispatching a response.\n\n"
            f"Detection: {payload.detection_type}\nConfidence: {payload.confidence:.2f}\n"
            f"ATM ID: {payload.atm_id}\nLocation: {payload.location}\nTime: {payload.timestamp}\n\n"
            f"Sent to: {settings.police_email}{station}"
        )
        if payload.snapshot_path:
            with open(payload.snapshot_path, "rb") as f:
                msg.add_attachment(f.read(), maintype="image", subtype="jpeg", filename="snapshot.jpg")
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
        return "sent", "ok"
    except Exception as exc:
        log.exception("Police alert failed")
        return "failed", str(exc)


def _send_telegram(payload: AlertPayload) -> tuple[str, str]:
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        return "skipped_not_configured", "Telegram not configured"
    try:
        url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
        text = (
            f"🚨 Omni Mind Alert 🚨\nType: {payload.detection_type}\nConfidence: {payload.confidence:.2f}\n"
            f"ATM: {payload.atm_id}\nLocation: {payload.location}\nTime: {payload.timestamp}"
        )
        resp = httpx.post(url, json={"chat_id": settings.telegram_chat_id, "text": text}, timeout=10)
        resp.raise_for_status()
        return "sent", "ok"
    except Exception as exc:
        log.exception("Telegram alert failed")
        return "failed", str(exc)


def dispatch_alert_async(payload: AlertPayload, on_result) -> None:
    """Fire-and-forget across all channels; `on_result(channel, status, detail)`
    is invoked from the worker thread once each channel completes."""

    def _run(channel: str, fn) -> None:
        status, detail = fn(payload)
        log.info("Alert channel=%s status=%s", channel, status)
        try:
            on_result(channel, status, detail)
        except Exception:
            log.exception("on_result callback raised for channel=%s", channel)

    _executor.submit(_run, "email", _send_email)
    _executor.submit(_run, "telegram", _send_telegram)
    _executor.submit(_run, "police", _send_police_alert)
