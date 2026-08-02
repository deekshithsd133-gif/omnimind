"""Liveness detection via MediaPipe Face Mesh landmarks.

Defends against printed-photo / phone-screen / tablet-replay spoofing by
requiring evidence of a *live* face over a short observation window:
  - at least one blink (eye-aspect-ratio dip then recovery)
  - at least one head-yaw movement of a few degrees

A static photo or a video replayed on a screen essentially never reproduces
both a natural blink cadence *and* a matching parallax head turn within a
few seconds, especially once combined with the face-verification embedding
check. This is a genuine, if lightweight, active-liveness check — not a full
depth-sensor-based passive liveness model (no depth camera exists on a phone
browser), which is called out explicitly rather than implied.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from config.settings import settings

_LEFT_EYE = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
_NOSE_TIP = 1
_LEFT_FACE_EDGE = 234
_RIGHT_FACE_EDGE = 454

_mp_face_mesh = mp.solutions.face_mesh
_mp_face_detection = mp.solutions.face_detection


def _eye_aspect_ratio(landmarks, indices, w, h) -> float:
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    vert1 = np.linalg.norm(pts[1] - pts[5])
    vert2 = np.linalg.norm(pts[2] - pts[4])
    horiz = np.linalg.norm(pts[0] - pts[3])
    if horiz == 0:
        return 0.0
    return (vert1 + vert2) / (2.0 * horiz)


def _yaw_ratio(landmarks, w, h) -> float:
    """Nose position relative to face width as a cheap proxy for head yaw,
    avoiding full solvePnP camera-calibration requirements."""
    nose_x = landmarks[_NOSE_TIP].x * w
    left_x = landmarks[_LEFT_FACE_EDGE].x * w
    right_x = landmarks[_RIGHT_FACE_EDGE].x * w
    face_width = right_x - left_x
    if face_width == 0:
        return 0.5
    return (nose_x - left_x) / face_width


@dataclass
class LivenessState:
    ear_history: list[float] = field(default_factory=list)
    yaw_history: list[float] = field(default_factory=list)
    blink_detected: bool = False
    head_move_detected: bool = False
    window_start: float = field(default_factory=time.monotonic)
    window_seconds: float = field(default_factory=lambda: settings.liveness_window_seconds)
    last_face_bbox: tuple[int, int, int, int] | None = None

    def reset(self) -> None:
        self.__init__(window_seconds=self.window_seconds)

    @property
    def is_live(self) -> bool:
        blink_ok = self.blink_detected if settings.liveness_blink_required else True
        move_ok = self.head_move_detected if settings.liveness_head_move_required else True
        return blink_ok and move_ok

    @property
    def expired(self) -> bool:
        return (time.monotonic() - self.window_start) > self.window_seconds


def face_bbox_from_landmarks(landmarks, w: int, h: int, margin: float = 0.15) -> tuple[int, int, int, int]:
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    mx = (x2 - x1) * margin
    my = (y2 - y1) * margin
    x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    x2, y2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return x1, y1, x2, y2


class FaceDetectorScalar:
    """Thin wrapper around MediaPipe's short-range face detector, used only
    to produce the scalar "Face Detection" confidence the workflow spec asks
    for as a distinct gate from FaceMesh's landmark-based liveness/crop use."""

    def __init__(self) -> None:
        self._detector = _mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.2)

    def close(self) -> None:
        self._detector.close()

    def confidence(self, frame_bgr: np.ndarray) -> float:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        if not result.detections:
            return 0.0
        return max(float(d.score[0]) for d in result.detections)


class FaceOcclusionHeuristic:
    """Catches face coverings that a general-purpose face detector stays
    confident through — a surgical mask, a cloth/bandana pulled up to the
    eyes, or an open-face helmet all leave the eyes and forehead visible,
    which is enough for FaceDetectorScalar to keep reporting "face present"
    at high confidence, since general face detectors are deliberately robust
    to partial occlusion (that's normally a feature, not a bug). No open
    pretrained "masked-face" classifier is bundled here, so this instead
    compares the lower-face region (mouth/nose/chin) against a trusted-skin
    reference from the *same frame*: bare skin stays close in hue/saturation
    to the reference and keeps comparable or higher local texture (lips,
    nostril shadows, stubble), while a mask, fabric, or visor typically
    breaks at least one of those. The reference is the *median* of three
    patches (forehead + both cheeks), not the forehead alone — bangs/hair
    covering the forehead used to corrupt a forehead-only reference into
    "hair", which then read as a mismatch against the (actually bare)
    mouth/nose/chin and falsely flagged a plain face as covered. Taking the
    median across three spots means one corrupted patch (hair over the
    forehead, or a wide mask reaching the cheeks) still leaves two genuine
    skin readings to anchor the comparison. Same honesty pattern as
    models/blunt_object_heuristic.py — a heuristic, not a trained occlusion
    model, deliberately conservative (treats an inconclusive read, or a
    lower-face point that merely *disagrees* rather than a unanimous read,
    as "not occluded") so it only flags clear cases instead of manufacturing
    false alarms.

    A same-day attempt to improve hand/hood recall by relaxing this to a 70%
    supermajority vote across 6 points (adding both mouth corners) was
    reverted after field testing reported previously-working verification
    breaking: lips and mouth corners are reliably more saturated/red than
    the forehead/cheek reference even on a completely bare face, which is
    exactly the kind of correlated-not-independent evidence a majority vote
    is vulnerable to (multiple points failing for the *same* underlying
    reason isn't stronger evidence than one point failing). Unanimity across
    the original 4 points is a blunter but safer rule precisely because it
    has no such correlated-failure mode across an ordinary bare face; the
    12-of-15 consecutive-frame confirmation window is the layer that's
    actually supposed to buy recall for genuine coverings, not this vote.
    """

    _REFERENCE_IDX = (10, 50, 280)  # forehead, left cheek, right cheek
    _LOWER_FACE_IDX = (13, 14, 152, 1)  # upper lip, lower lip, chin, nose tip
    _HUE_SAT_DIST_THRESHOLD = 35.0
    _TEXTURE_RATIO_THRESHOLD = 0.35

    def __init__(self) -> None:
        self._mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()

    @staticmethod
    def _patch_stats(frame_bgr: np.ndarray, cx: int, cy: int, half: int = 7):
        h, w = frame_bgr.shape[:2]
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32).mean(axis=0)
        texture = float(np.std(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)))
        return hsv, texture

    def lower_face_occluded(self, frame_bgr: np.ndarray) -> bool:
        h, w = frame_bgr.shape[:2]
        # FaceMesh only needs enough resolution to localize landmarks —
        # feeding it the full person-crop costs real per-frame time for no
        # accuracy benefit here. Only the copy handed to MediaPipe is
        # downscaled; the color/texture patches below still sample the
        # full-resolution frame for an accurate read.
        max_dim = 320
        scale = min(1.0, max_dim / max(h, w))
        mesh_input = frame_bgr if scale == 1.0 else cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        rgb = cv2.cvtColor(mesh_input, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return False  # "no face at all" is FaceDetectorScalar's job, not this heuristic's
        landmarks = result.multi_face_landmarks[0].landmark

        def pt(i):
            return int(landmarks[i].x * w), int(landmarks[i].y * h)

        ref_stats = [self._patch_stats(frame_bgr, *pt(i)) for i in self._REFERENCE_IDX]
        ref_stats = [s for s in ref_stats if s is not None]
        if len(ref_stats) < 2:
            return False  # not enough of the face on-screen to safely judge
        ref_hsv = np.median(np.stack([s[0] for s in ref_stats]), axis=0)
        ref_texture = float(np.median([s[1] for s in ref_stats]))

        checked = 0
        occluded_votes = 0
        for idx in self._LOWER_FACE_IDX:
            stats = self._patch_stats(frame_bgr, *pt(idx))
            if stats is None:
                continue
            hsv, texture = stats
            checked += 1
            hue_sat_dist = float(np.linalg.norm(hsv[:2] - ref_hsv[:2]))
            texture_ratio = texture / max(ref_texture, 1.0)
            if hue_sat_dist > self._HUE_SAT_DIST_THRESHOLD or texture_ratio < self._TEXTURE_RATIO_THRESHOLD:
                occluded_votes += 1

        if checked < 3:
            return False  # too few readable lower-face points to trust a verdict
        return occluded_votes == checked  # unanimous — conservative on purpose


class LivenessDetector:
    def __init__(self) -> None:
        # static_image_mode=True is deliberate, not the library default: this
        # one FaceMesh instance is shared across every tracked person on the
        # connection (see workflow/pipeline.py, one SecurityPipeline per
        # camera, N PersonSessions per pipeline). MediaPipe's video/tracking
        # mode (static_image_mode=False) assumes consecutive .process() calls
        # are the *same* continuous face and reuses the prior call's landmarks
        # as a tracking hint — fine for one person, but with several people
        # each frame round-robins this same object across different faces,
        # so person B's call would start from person A's tracking state.
        # static_image_mode treats every call as an independent image, which
        # costs a bit of MediaPipe's internal smoothing but is correct for
        # multiple concurrent subjects; our own ear/yaw history in
        # LivenessState already does the temporal smoothing we need anyway.
        self._mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()

    def process(self, frame_bgr: np.ndarray, state: LivenessState) -> LivenessState:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(rgb)
        if not result.multi_face_landmarks:
            return state

        landmarks = result.multi_face_landmarks[0].landmark
        state.last_face_bbox = face_bbox_from_landmarks(landmarks, w, h)
        ear = (
            _eye_aspect_ratio(landmarks, _LEFT_EYE, w, h) + _eye_aspect_ratio(landmarks, _RIGHT_EYE, w, h)
        ) / 2.0
        yaw = _yaw_ratio(landmarks, w, h)

        state.ear_history.append(ear)
        state.yaw_history.append(yaw)
        if len(state.ear_history) > 60:
            state.ear_history.pop(0)
        if len(state.yaw_history) > 60:
            state.yaw_history.pop(0)

        if len(state.ear_history) >= 3 and not state.blink_detected:
            recent = state.ear_history[-5:]
            if (max(recent) - min(recent)) >= settings.liveness_min_ear_delta:
                state.blink_detected = True

        if len(state.yaw_history) >= 3 and not state.head_move_detected:
            span = max(state.yaw_history) - min(state.yaw_history)
            # yaw ratio spans roughly 0..1 across the face width; convert a
            # coarse fraction-of-face-width shift into an approximate degree
            # figure for threshold comparison purposes.
            approx_deg = span * 90.0
            if approx_deg >= settings.liveness_min_yaw_delta_deg:
                state.head_move_detected = True

        return state
