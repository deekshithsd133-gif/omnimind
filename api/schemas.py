from __future__ import annotations

from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class HealthResponse(BaseModel):
    status: str
    device: str
    active_sessions: int
    active_camera_connections: int
    uptime_seconds: float
    atm_id: str
    atm_location: str


class EventOut(BaseModel):
    id: int
    timestamp: str
    detection_type: str
    confidence: float
    person_track_id: int | None
    verification_status: str
    alert_status: str
    heuristic: bool
    evidence_image_path: str | None
    video_path: str | None


class UnlockRequest(BaseModel):
    connection_id: str
    track_id: int | None = None
