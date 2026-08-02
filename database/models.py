"""SQLAlchemy ORM models. Sensitive columns (face embeddings, evidence paths)
store Fernet-encrypted bytes — see utils.security. Plain face images are
never written to any column."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="operator")  # "admin" | "operator"
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class EnrolledFace(Base):
    """A known/verified identity's face embedding. No raw images stored."""

    __tablename__ = "enrolled_faces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    label: Mapped[str] = mapped_column(String(128))
    embedding_encrypted: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ATMSession(Base):
    """One tracked-person visit at the ATM, from arrival to departure/timeout."""

    __tablename__ = "atm_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    atm_id: Mapped[str] = mapped_column(String(64))
    track_id: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="pending")
    # "pending" | "face_covered" | "verifying" | "verified" | "failed" | "locked_out"
    outcome: Mapped[str] = mapped_column(String(32), default="in_progress")
    # "in_progress" | "completed" | "abandoned" | "security_lockdown"

    events: Mapped[list["DetectionEvent"]] = relationship(back_populates="session")


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("atm_sessions.id"), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, index=True)
    detection_type: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)
    person_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verification_status: Mapped[str] = mapped_column(String(32), default="n/a")
    evidence_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    alert_status: Mapped[str] = mapped_column(String(32), default="none")
    # "none" | "queued" | "sent" | "failed" | "skipped_not_configured"
    heuristic: Mapped[bool] = mapped_column(default=False)  # True = best-effort fallback, not a certified model

    session: Mapped["ATMSession | None"] = relationship(back_populates="events")


class AlertLog(Base):
    __tablename__ = "alert_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[int | None] = mapped_column(ForeignKey("detection_events.id"), nullable=True)
    channel: Mapped[str] = mapped_column(String(32))  # "email" | "telegram" | "police" | "console"
    status: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
