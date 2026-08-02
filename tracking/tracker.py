"""Multi-person tracking via ByteTrack (supervision's implementation).

Only "person" detections are tracked — weapons/helmet/mask boxes are matched
to a person by containment, not tracked as independent identities. Stable
track IDs are what let the workflow state machine (workflow/pipeline.py) run
one state-machine instance per physical person and avoid re-firing an alarm
every frame for the same individual.
"""
from __future__ import annotations

import numpy as np
import supervision as sv

from config.settings import settings
from models.yolo_engine import Detection


class PersonTracker:
    def __init__(self) -> None:
        # ByteTrack's IoU matching defaults (minimum_matching_threshold=0.8)
        # are tuned for dense ~30fps video. We process at settings.target_fps
        # (throttled, plus real network jitter from a phone camera), so a
        # person's box legitimately shifts more between our sampled frames.
        # At 0.8 that mismatch was enough to drop and re-create a new track
        # ID for the same physical person every few seconds — which then
        # resets every rolling-window confirmation (face-visible, liveness,
        # verification) keyed on that track_id, so verification could never
        # accumulate enough confirmed frames to complete. Passing the real
        # frame_rate also fixes ByteTrack's internal lost-track buffer math
        # (max_time_lost = frame_rate/30 * lost_track_buffer), which
        # otherwise assumes 30fps and mis-times how long a briefly-occluded
        # track is kept alive.
        # lost_track_buffer bumped from 60 (~4s at our 15fps) to 225 (~15s):
        # 4s wasn't enough to survive a WebSocket reconnect (network hiccup,
        # watchdog-forced reconnect), during which frames stop arriving
        # entirely for several seconds. The physical person got a brand-new
        # track_id once frames resumed, which resets every rolling-window
        # confirmation and the covered-face/verification state machine (both
        # keyed on track_id) even though the SecurityPipeline itself now
        # persists across reconnects (see api/registry.py). 15s covers a
        # realistic reconnect without meaningfully increasing the risk of
        # merging two different people, since this is a single-camera,
        # one-person-at-a-time ATM booth.
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=225,
            minimum_matching_threshold=0.5,
            frame_rate=settings.target_fps,
        )

    def update(self, person_detections: list[Detection]) -> list[tuple[int, Detection]]:
        """Returns (track_id, detection) pairs, one per tracked person in this frame."""
        if not person_detections:
            # Still call update with an empty batch so internal track ages advance
            # and stale tracks get pruned instead of lingering forever.
            self._tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d.bbox for d in person_detections], dtype=np.float32)
        confidence = np.array([d.confidence for d in person_detections], dtype=np.float32)
        class_id = np.zeros(len(person_detections), dtype=int)

        sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        tracked = self._tracker.update_with_detections(sv_detections)

        results: list[tuple[int, Detection]] = []
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
            bbox = tuple(float(v) for v in tracked.xyxy[i])
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            det = Detection(category="person", confidence=conf, bbox=bbox, source_model="tracked")
            results.append((track_id, det))
        return results


def bbox_contains_center(outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]) -> bool:
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = inner
    cx, cy = (ix1 + ix2) / 2.0, (iy1 + iy2) / 2.0
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def head_region(person_bbox: tuple[float, float, float, float], fraction: float = 0.5) -> tuple[float, float, float, float]:
    """Top slice of a person's bbox — used to scope face-covering detections
    (helmet/mask) to the head, not the whole body. Without this, a mask that's
    been pulled down to the chin/neck, is being held in a hand, or is hanging
    off an ear still has its box center somewhere inside the full head-to-toe
    person bbox and keeps re-triggering "mask detected" long after the face
    itself is uncovered. 0.5 is a deliberately generous fraction (not
    anatomical head proportion) because the browser camera view here is a
    head-and-shoulders framing, not a full standing body."""
    x1, y1, x2, y2 = person_bbox
    return x1, y1, x2, y1 + (y2 - y1) * fraction
