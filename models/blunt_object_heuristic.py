"""Best-effort fallback for weapon classes with no available open pretrained
model: "iron rod" and generic rigid blunt objects.

No open-source, pip-installable, trained detector exists for this class (see
README). Rather than fabricate a confident "iron rod: 97%" detection with
nothing real behind it, this module does classic-CV edge/line analysis and
deliberately caps its own output confidence at `MAX_HEURISTIC_CONFIDENCE`,
which is always below `settings.conf_weapon`. It can therefore never by
itself satisfy the 96%-confidence / 15-frame weapon-lockdown gate — it only
ever contributes to the softer "suspicious behaviour" signal. Treat this as a
placeholder to swap for a real trained model (see models/yolo_engine.py
docstring for how detections plug in) once labeled training data exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MAX_HEURISTIC_CONFIDENCE = 0.65
MIN_LINE_LENGTH_RATIO = 0.22  # relative to person bbox height
MAX_LINE_ANGLE_TOLERANCE_DEG = 35  # how far from person's hand-holding orientation we tolerate


@dataclass
class HeuristicHit:
    confidence: float
    bbox: tuple[float, float, float, float]


def detect_rigid_elongated_object(frame_bgr: np.ndarray, person_bbox: tuple[float, float, float, float]) -> HeuristicHit | None:
    x1, y1, x2, y2 = (int(v) for v in person_bbox)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame_bgr.shape[1]), min(y2, frame_bgr.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    person_height = y2 - y1
    min_len = max(20, int(person_height * MIN_LINE_LENGTH_RATIO))

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=8)
    if lines is None:
        return None

    best_len = 0.0
    best_line = None
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        length = float(np.hypot(lx2 - lx1, ly2 - ly1))
        if length > best_len:
            best_len = length
            best_line = (lx1, ly1, lx2, ly2)

    if best_line is None or best_len < min_len:
        return None

    # Confidence scales with how much the strongest line exceeds the minimum
    # threshold, capped so this heuristic can never reach weapon-alarm territory.
    excess_ratio = min(best_len / (min_len * 2), 1.0)
    confidence = min(0.35 + 0.30 * excess_ratio, MAX_HEURISTIC_CONFIDENCE)

    lx1, ly1, lx2, ly2 = best_line
    bbox = (x1 + min(lx1, lx2), y1 + min(ly1, ly2), x1 + max(lx1, lx2), y1 + max(ly1, ly2))
    return HeuristicHit(confidence=confidence, bbox=bbox)
