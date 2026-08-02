"""Evidence capture: snapshot JPEGs + short pre/post-event video clips.

A rolling buffer keeps the last few seconds of raw frames at all times so a
triggered recording includes lead-up footage, not just what happens after
the trigger fires. All disk writes happen on a dedicated background thread
so a slow disk can never stall the detection loop (mirrors database.db's
writer-thread pattern).
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from config.settings import settings
from utils.logger import get_logger

log = get_logger("evidence")


@dataclass
class _ActiveRecording:
    writer: cv2.VideoWriter
    path: Path
    expires_at: float


@dataclass
class _WriteJob:
    kind: str  # "frame" | "snapshot"
    frame: np.ndarray
    tag: str = ""
    result_holder: dict = field(default_factory=dict)


class EvidenceManager:
    def __init__(self, buffer_seconds: float = 5.0, post_event_seconds: float = 8.0, fps: int = 10) -> None:
        self.fps = fps
        self.post_event_seconds = post_event_seconds
        self._pre_buffer: deque[np.ndarray] = deque(maxlen=int(buffer_seconds * fps))
        self._active: list[_ActiveRecording] = []
        self._queue: "queue.Queue[_WriteJob]" = queue.Queue(maxsize=500)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="evidence-writer", daemon=True)
        self._thread.start()

    def add_frame(self, frame: np.ndarray) -> None:
        self._pre_buffer.append(frame.copy())
        try:
            self._queue.put_nowait(_WriteJob(kind="frame", frame=frame))
        except queue.Full:
            pass  # dropping a video frame is acceptable; blocking the pipeline is not

    def capture_image(self, frame: np.ndarray, tag: str) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{tag}_{timestamp}_{int(time.time() * 1000) % 1000}.jpg"
        path = settings.evidence_dir / filename
        try:
            cv2.imwrite(str(path), frame)
        except Exception:
            log.exception("Failed to write evidence snapshot")
            return ""
        return str(path)

    def trigger_video(self, tag: str) -> str:
        if not self._pre_buffer:
            return ""
        h, w = self._pre_buffer[-1].shape[:2]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = settings.evidence_dir / f"{tag}_{timestamp}.avi"
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
            for buffered_frame in list(self._pre_buffer):
                writer.write(buffered_frame)
            self._active.append(_ActiveRecording(writer=writer, path=path, expires_at=time.monotonic() + self.post_event_seconds))
            log.info("Evidence video started: %s", path)
            return str(path)
        except Exception:
            log.exception("Failed to start evidence video recording")
            return ""

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._prune_expired()
                continue
            if job.kind == "frame" and self._active:
                for rec in self._active:
                    try:
                        rec.writer.write(job.frame)
                    except Exception:
                        log.exception("Evidence writer failed for %s", rec.path)
            self._prune_expired()
            self._queue.task_done()

    def _prune_expired(self) -> None:
        now = time.monotonic()
        still_active = []
        for rec in self._active:
            if now >= rec.expires_at:
                try:
                    rec.writer.release()
                    log.info("Evidence video finalized: %s", rec.path)
                except Exception:
                    log.exception("Failed to finalize evidence video %s", rec.path)
            else:
                still_active.append(rec)
        self._active = still_active

    def shutdown(self) -> None:
        self._stop.set()
        for rec in self._active:
            try:
                rec.writer.release()
            except Exception:
                pass
