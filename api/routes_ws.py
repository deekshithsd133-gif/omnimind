"""The live video WebSocket endpoint.

Architecture: the async receive loop only ever does two cheap things —
decode a JPEG and drop it into a single-slot LatestFrameBuffer. All actual
AI inference happens on a dedicated worker thread per connection, so a slow
GPU frame can never stall the WebSocket's ability to keep accepting new
frames or make the event loop unresponsive to other connections. Results
cross back from the worker thread to the async send loop via
`loop.call_soon_threadsafe`, the standard non-blocking handoff pattern.
"""
from __future__ import annotations

import asyncio
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, Query, WebSocket
from starlette.websockets import WebSocketState

from api.registry import active_pipelines, get_or_create_pipeline
from camera.frame_buffer import LatestFrameBuffer
from config.settings import settings
from utils.logger import get_logger
from utils.security import decode_access_token

router = APIRouter()
log = get_logger("ws")


def _safe_put(q: asyncio.Queue, item: dict) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        q.put_nowait(item)
    except asyncio.QueueFull:
        pass


@router.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket, token: str = Query(...)) -> None:
    payload = decode_access_token(token)
    if payload is None:
        await websocket.close(code=4401)
        return

    await websocket.accept()
    conn_id = str(id(websocket))
    log.info("Camera connection opened: %s (user=%s)", conn_id, payload.get("sub"))

    frame_buffer = LatestFrameBuffer()
    out_queue: asyncio.Queue = asyncio.Queue(maxsize=5)
    stop_event = threading.Event()
    loop = asyncio.get_running_loop()

    try:
        # get_or_create_pipeline reuses the same SecurityPipeline (and its
        # in-flight session/tracker state) across a reconnect for this same
        # user instead of constructing a fresh one — see registry.py. Run off
        # the event loop thread since first-time construction has been
        # measured at ~0.5-0.8s (several MediaPipe models + the face
        # embedding model) and doing that synchronously here stalled the
        # entire event loop, including other connections, on every reconnect.
        pipeline, pipeline_lock = await asyncio.to_thread(get_or_create_pipeline, payload.get("sub", "unknown"))
    except Exception:
        log.exception("Failed to initialize detection pipeline")
        await websocket.close(code=1011)
        return

    active_pipelines[conn_id] = pipeline

    def worker() -> None:
        interval = 1.0 / max(settings.target_fps, 1)
        last_frame_id = -1
        while not stop_event.is_set():
            start = time.monotonic()
            frame, frame_id, _ = frame_buffer.get()
            if frame is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                try:
                    # The pipeline can briefly be shared across an old
                    # connection's teardown and a new one's startup for the
                    # same user during a fast reconnect — process_frame isn't
                    # safe to call concurrently, hence the lock.
                    with pipeline_lock:
                        result = pipeline.process_frame(frame)
                    msg = {
                        "connection_id": conn_id,
                        "system_status": result.system_status,
                        "boxes": [dict(b, bbox=list(b["bbox"])) for b in result.boxes],
                        "sessions": result.sessions,
                        "audio_events": result.audio_events,
                        "log_events": result.log_events,
                        "frame_id": frame_id,
                    }
                    loop.call_soon_threadsafe(_safe_put, out_queue, msg)
                except Exception:
                    log.exception("Frame processing error on connection %s — skipping frame", conn_id)
            elapsed = time.monotonic() - start
            time.sleep(max(0.0, interval - elapsed))

    worker_thread = threading.Thread(target=worker, name=f"pipeline-{conn_id}", daemon=True)
    worker_thread.start()

    async def sender() -> None:
        while True:
            item = await out_queue.get()
            if websocket.application_state != WebSocketState.CONNECTED:
                return
            await websocket.send_json(item)

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    frame_buffer.put(frame)
    except Exception:
        log.info("WebSocket connection %s closed (%s)", conn_id, "client disconnect or transport error")
    finally:
        stop_event.set()
        sender_task.cancel()
        # Off the event loop thread: this connection's worker can take up to
        # 2s to notice stop_event, and blocking the loop on every disconnect
        # was itself a contributor to slow/stuck reconnects.
        await asyncio.to_thread(worker_thread.join, 2)
        # Deliberately NOT pipeline.close() — this pipeline is shared across
        # this user's reconnects (see get_or_create_pipeline) and stays alive
        # for the life of the server so a reconnect resumes in-flight session
        # state instead of losing it. Only this connection's own resources
        # (frame buffer, worker thread, queue) are torn down here.
        active_pipelines.pop(conn_id, None)
        log.info("Camera connection cleaned up: %s", conn_id)
