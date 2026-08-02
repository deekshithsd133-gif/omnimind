"""Thread-safe single-slot frame buffer.

Deliberately NOT a queue: the WebSocket receive thread always overwrites the
latest frame, and the inference thread always reads the newest one. This is
the standard fix for camera pipelines that would otherwise fall behind and
either block (bounded queue, full) or leak memory (unbounded queue) when
inference is slower than the incoming frame rate.
"""
from __future__ import annotations

import threading
import time

import numpy as np


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._updated_at = 0.0

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1
            self._updated_at = time.monotonic()

    def get(self) -> tuple[np.ndarray | None, int, float]:
        with self._lock:
            return self._frame, self._frame_id, self._updated_at

    def is_stale(self, max_age_seconds: float) -> bool:
        with self._lock:
            if self._frame is None:
                return True
            return (time.monotonic() - self._updated_at) > max_age_seconds
