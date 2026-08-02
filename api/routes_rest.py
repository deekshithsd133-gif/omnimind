"""REST endpoints: auth, health, event/alert history, face enrollment, and
the admin ATM-unlock action for simulated lockdown clearing."""
from __future__ import annotations

import time

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from api.auth import authenticate_user, get_current_user, issue_token, require_role
from api.registry import SERVER_START_TIME, active_pipelines, get_detection_engine
from api.schemas import EventOut, HealthResponse, LoginRequest, LoginResponse, UnlockRequest
from config.settings import settings
from database.db import get_session
from database.models import AlertLog, DetectionEvent, EnrolledFace
from liveness.liveness import FaceDetectorScalar, face_bbox_from_landmarks
import mediapipe as mp
from utils.logger import get_logger
from utils.security import encrypt_bytes
from verification.face_verify import FaceVerifier
from workflow.pipeline import SessionState

router = APIRouter()
log = get_logger("rest")


@router.post("/api/auth/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    user = authenticate_user(body.username, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password")
    token = issue_token(user)
    return LoginResponse(access_token=token, role=user.role, username=user.username)


@router.get("/api/health", response_model=HealthResponse)
def health(user: dict = Depends(get_current_user)) -> HealthResponse:
    active_sessions = sum(len(p.sessions) for p in active_pipelines.values())
    device = "cpu"
    try:
        device = "cuda" if torch.cuda.is_available() and settings.device_preference != "cpu" else "cpu"
    except Exception:
        pass
    return HealthResponse(
        status="ok",
        device=device,
        active_sessions=active_sessions,
        active_camera_connections=len(active_pipelines),
        uptime_seconds=time.monotonic() - SERVER_START_TIME,
        atm_id=settings.atm_id,
        atm_location=settings.atm_location,
    )


@router.get("/api/events", response_model=list[EventOut])
def list_events(limit: int = 50, user: dict = Depends(get_current_user)) -> list[EventOut]:
    limit = max(1, min(limit, 500))
    with get_session() as db:
        rows = db.query(DetectionEvent).order_by(DetectionEvent.id.desc()).limit(limit).all()
        return [
            EventOut(
                id=r.id,
                timestamp=r.timestamp.isoformat(),
                detection_type=r.detection_type,
                confidence=r.confidence,
                person_track_id=r.person_track_id,
                verification_status=r.verification_status,
                alert_status=r.alert_status,
                heuristic=r.heuristic,
                evidence_image_path=r.evidence_image_path,
                video_path=r.video_path,
            )
            for r in rows
        ]


@router.get("/api/alerts")
def list_alerts(limit: int = 50, user: dict = Depends(get_current_user)) -> list[dict]:
    limit = max(1, min(limit, 500))
    with get_session() as db:
        rows = db.query(AlertLog).order_by(AlertLog.id.desc()).limit(limit).all()
        return [
            {"id": r.id, "channel": r.channel, "status": r.status, "detail": r.detail, "sent_at": r.sent_at.isoformat()}
            for r in rows
        ]


@router.post("/api/atm/unlock")
def unlock(body: UnlockRequest, user: dict = Depends(require_role("admin", "operator"))) -> dict:
    pipeline = active_pipelines.get(body.connection_id)
    if pipeline is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No active camera session with that connection id")
    count = pipeline.admin_unlock(body.track_id)
    log.info("Admin %s unlocked %d session(s) on connection %s", user["username"], count, body.connection_id)
    return {"unlocked": count}


@router.get("/api/atm/status")
def atm_status(user: dict = Depends(get_current_user)) -> dict:
    out = []
    for conn_id, pipeline in active_pipelines.items():
        out.append({
            "connection_id": conn_id,
            "sessions": [
                {"track_id": s.track_id, "state": s.state.value}
                for s in pipeline.sessions.values()
            ],
        })
    return {"connections": out}


@router.post("/api/enroll")
def enroll_face(
    label: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(require_role("admin")),
) -> dict:
    data = file.file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not decode image")

    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
    try:
        result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "No face detected in the uploaded image")
        h, w = frame.shape[:2]
        bbox = face_bbox_from_landmarks(result.multi_face_landmarks[0].landmark, w, h)
        face_crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if face_crop.size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Face crop was empty")
    finally:
        mesh.close()

    engine = get_detection_engine()
    verifier = FaceVerifier(device=engine.device)
    embedding = verifier.embed(face_crop)
    encrypted = encrypt_bytes(embedding.astype(np.float32).tobytes())

    with get_session() as db:
        db.add(EnrolledFace(label=label, embedding_encrypted=encrypted))
        db.commit()

    return {"status": "enrolled", "label": label}
