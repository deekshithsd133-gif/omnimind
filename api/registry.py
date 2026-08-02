"""Process-wide shared state: the live camera-session pipelines and the
lazily-created singleton DetectionEngine (loading 3 YOLO models per engine
instance is expensive — one shared engine serves every connection)."""
from __future__ import annotations

import threading
import time

from models.yolo_engine import DetectionEngine
from workflow.pipeline import SecurityPipeline

SERVER_START_TIME = time.monotonic()

# Keyed by WebSocket connection id — used for the admin-unlock/status/health
# endpoints, which target a specific live connection. Multiple connection ids
# can point at the same SecurityPipeline instance (see get_or_create_pipeline).
active_pipelines: dict[str, SecurityPipeline] = {}

_engine: DetectionEngine | None = None
_engine_lock = threading.Lock()


def get_detection_engine() -> DetectionEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = DetectionEngine()
        return _engine


# One SecurityPipeline per logged-in user, not per WebSocket connection — a
# reconnect (phone WiFi hiccup, a backgrounded tab recovering, the client's
# stream-health watchdog forcing a fresh socket) used to hand the same person
# a brand-new pipeline with empty session/tracker state, silently discarding
# an in-flight covered-face countdown or verification and restarting it from
# MONITORING with zero visible indication to the user that anything reset.
# Constructing a fresh one was also measured at ~0.5-0.8s (loads several
# MediaPipe models plus the face-verification embedding model) — doing that
# synchronously inside the WebSocket handler stalled the asyncio event loop
# on every reconnect (see routes_ws.py, which now does this via
# asyncio.to_thread and guards process_frame with the paired lock below,
# since two connections for the same user could otherwise briefly overlap
# during a fast reconnect and call into the same non-thread-safe pipeline).
_pipelines_by_user: dict[str, SecurityPipeline] = {}
_pipeline_locks: dict[str, threading.Lock] = {}
_pipelines_lock = threading.Lock()


def get_or_create_pipeline(user: str) -> tuple[SecurityPipeline, threading.Lock]:
    with _pipelines_lock:
        pipeline = _pipelines_by_user.get(user)
        if pipeline is None:
            pipeline = SecurityPipeline(get_detection_engine())
            _pipelines_by_user[user] = pipeline
            _pipeline_locks[user] = threading.Lock()
        return pipeline, _pipeline_locks[user]
