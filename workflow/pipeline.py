"""The ATM security workflow state machine (spec Steps 1-6), one instance
per active WebSocket/camera session. Owns detection, tracking, liveness,
verification, evidence capture, DB writes, and alert dispatch for that
camera feed. Designed to run its own dedicated inference thread — see
api/routes_ws.py, which owns the thread lifecycle.

Face-verification scope note: this prototype has no card/account linkage, so
"Face Verification" means one of two things depending on deployment state:
  1. If admin-enrolled reference faces exist (`EnrolledFace` table, e.g. a
     staff/watchlist allowlist), the live embedding is matched against them.
  2. Otherwise, it falls back to *self-consistency* verification: the first
     embedding captured once liveness passes becomes the session's reference,
     and subsequent frames must keep matching it — i.e. "the same live human
     who passed liveness is still the one present," not a KYC identity check
     against a national database. This is stated explicitly rather than
     silently redefining what "verified" means.
Also: the spec's ">99% face verification confidence" doesn't translate
literally onto raw cosine similarity (genuine FaceNet/VGGFace2 matches
typically score ~0.6-0.9, not 0.99) — applying 0.99 as a raw cosine cutoff
would make verification fail for real, live users. `settings.face_match_cosine_threshold`
is the actual calibrated operating point; the 15-frame confirmation window
plus the liveness+detection gates are what deliver the spec's intent of
"very high confidence before allowing a transaction."
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import numpy as np

from alerts.audio import DANGER_TEXT, HELMET_REMOVE_TEXT, UNCOVERED_TEXT, VERIFIED_TEXT, VOICE_WARNING_TEXT, AudioEventFlags
from alerts.evidence import EvidenceManager
from alerts.notifier import AlertPayload, dispatch_alert_async
from config.settings import settings
from database.db import enqueue_write
from database.models import AlertLog, ATMSession, DetectionEvent, EnrolledFace
from liveness.liveness import FaceDetectorScalar, FaceOcclusionHeuristic, LivenessDetector, LivenessState
from models.blunt_object_heuristic import detect_rigid_elongated_object
from models.yolo_engine import Detection, DetectionEngine
from tracking.tracker import PersonTracker, bbox_contains_center, head_region
from utils.logger import get_logger
from utils.security import decrypt_bytes
from verification.face_verify import FaceVerifier
from workflow.confirmer import ConsecutiveConfirmer

log = get_logger("pipeline")

PERSON_ABSENCE_TIMEOUT_SECONDS = 2.5
AGGRESSIVE_SPEED_THRESHOLD_PX_PER_SEC = 900.0


class SessionState(str, Enum):
    MONITORING = "monitoring"
    FACE_COVERED_GRACE = "face_covered_grace"
    FACE_COVERED_WARNING = "face_covered_warning"
    SIREN = "siren"
    VERIFYING = "verifying"
    TRANSACTION_ACTIVE = "transaction_active"
    LOCKDOWN = "lockdown"
    DONE = "done"


@dataclass
class PersonSession:
    track_id: int
    state: SessionState = SessionState.MONITORING
    state_entered_at: float = field(default_factory=time.monotonic)
    last_seen: float = field(default_factory=time.monotonic)
    db_session_id: int | None = None
    liveness_state: LivenessState = field(default_factory=LivenessState)
    reference_embedding: np.ndarray | None = None
    audio: AudioEventFlags = field(default_factory=AudioEventFlags)
    lockdown_frames_to_capture: int = 0
    last_weapon_detection: Detection | None = None
    last_face_score: float = 0.0
    face_covered_alert_sent: bool = False
    weapon_alert_sent: bool = False
    multi_person_alert_sent: bool = False
    center_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # Which signal most recently drove face_occluded_by_gear=True for this
    # session ("helmet" | "mask" | "covering" | None) — lets the warning step
    # give a specific "remove your helmet" instruction instead of a generic
    # one when that's actually what's covering the face.
    last_cover_reason: str | None = None


@dataclass
class FrameResult:
    system_status: str
    boxes: list[dict]
    sessions: list[dict]
    audio_events: list[dict]
    log_events: list[str]


class SecurityPipeline:
    def __init__(self, detection_engine: DetectionEngine, atm_id: str = settings.atm_id) -> None:
        self.atm_id = atm_id
        self.detector = detection_engine
        self.tracker = PersonTracker()
        self.liveness = LivenessDetector()
        self.face_detector = FaceDetectorScalar()
        self.occlusion_heuristic = FaceOcclusionHeuristic()
        self.verifier = FaceVerifier(device=detection_engine.device)
        self.evidence = EvidenceManager(fps=settings.target_fps)

        self._build_confirmers()

        self.sessions: dict[int, PersonSession] = {}
        self._enrolled_cache: list[tuple[str, np.ndarray]] | None = None
        self._enrolled_cache_at = 0.0
        self._last_frame_wall_time: float | None = None

        log.info("SecurityPipeline started for atm_id=%s", atm_id)

    def _build_confirmers(self) -> None:
        """(Re)creates every ConsecutiveConfirmer this pipeline owns. Shared
        between __init__ and _reset_for_new_arrival so both stay in sync —
        helmet/mask/covering_heuristic (clean, class-level YOLO/heuristic
        signals) deliberately use a faster window than weapon/verification
        (see settings.cover_confirm_*)."""
        cover_kwargs = dict(
            window=settings.cover_confirm_window_frames,
            min_hits=settings.cover_confirm_min_hits,
            release_after_misses=settings.cover_confirm_release_after_misses,
        )
        self.helmet_confirmer = ConsecutiveConfirmer(**cover_kwargs)
        self.mask_confirmer = ConsecutiveConfirmer(**cover_kwargs)
        self.covering_heuristic_confirmer = ConsecutiveConfirmer(**cover_kwargs)

        # face_covered/face_visible consume the *raw* MediaPipe face-presence
        # score directly (not just confirmed gear detections) — that score
        # naturally dips for a frame or two during an ordinary blink, or
        # during the head-turn the liveness check itself requires the user
        # to perform. Putting these on the fast window (same-day attempt,
        # now reverted) meant a blink or head-turn could get misread as "a
        # covering just appeared" within ~4 frames, which resets
        # session.liveness_state on every such false alarm — permanently
        # preventing blink+head-turn from ever being observed together,
        # i.e. verification could never complete. The standard/slower
        # window is what this ran on before that change and is restored
        # here; covering-detection speed still comes from helmet/mask/
        # covering_heuristic (above) being fast, correctly-confirmed
        # signals in the first place.
        self.face_covered_confirmer = ConsecutiveConfirmer()
        self.face_visible_confirmer = ConsecutiveConfirmer()

        self.face_detect_confirmer = ConsecutiveConfirmer()
        self.face_verify_confirmer = ConsecutiveConfirmer()
        self.weapon_confirmer = ConsecutiveConfirmer()
        self.aggressive_confirmer = ConsecutiveConfirmer()

    def close(self) -> None:
        self.liveness.close()
        self.face_detector.close()
        self.occlusion_heuristic.close()
        self.evidence.shutdown()

    # ------------------------------------------------------------------ #
    def _load_enrolled(self) -> list[tuple[str, np.ndarray]]:
        now = time.monotonic()
        if self._enrolled_cache is not None and (now - self._enrolled_cache_at) < 10.0:
            return self._enrolled_cache
        from database.db import get_session

        result: list[tuple[str, np.ndarray]] = []
        try:
            with get_session() as db:
                for row in db.query(EnrolledFace).all():
                    raw = decrypt_bytes(row.embedding_encrypted)
                    vec = np.frombuffer(raw, dtype=np.float32)
                    result.append((row.label, vec))
        except Exception:
            log.exception("Failed to load enrolled faces")
        self._enrolled_cache = result
        self._enrolled_cache_at = now
        return result

    def _best_enrolled_match(self, probe: np.ndarray) -> tuple[str | None, float]:
        best_label, best_sim = None, -1.0
        for label, vec in self._load_enrolled():
            sim = self.verifier.cosine_similarity(probe, vec)
            if sim > best_sim:
                best_label, best_sim = label, sim
        return best_label, best_sim

    def _crop(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(max(0, v)) for v in bbox)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return frame
        return frame[y1:y2, x1:x2]

    def _end_session(self, session: PersonSession, outcome: str) -> None:
        session.state = SessionState.DONE
        db_id = session.db_session_id
        if db_id is not None:
            def _write(db, db_id=db_id, outcome=outcome):
                row = db.get(ATMSession, db_id)
                if row:
                    row.ended_at = datetime.now(timezone.utc)
                    row.outcome = outcome
            enqueue_write(_write)
        for confirmer in (
            self.face_covered_confirmer, self.face_visible_confirmer, self.face_detect_confirmer,
            self.face_verify_confirmer, self.weapon_confirmer, self.helmet_confirmer,
            self.mask_confirmer, self.covering_heuristic_confirmer, self.aggressive_confirmer,
        ):
            confirmer.forget_prefix(session.track_id)

    def _ensure_db_session(self, session: PersonSession) -> None:
        if session.db_session_id is not None:
            return
        track_id = session.track_id

        # Executed synchronously (not via the async write queue) so we can
        # read back the generated id immediately; still off the hot path
        # because it only runs once per session, not once per frame.
        from database.db import get_session
        try:
            with get_session() as db:
                row = ATMSession(atm_id=self.atm_id, track_id=track_id, verification_status="pending")
                db.add(row)
                db.commit()
                db.refresh(row)
                session.db_session_id = row.id
        except Exception:
            log.exception("Failed to create ATM session row")

    def _log_event(self, session: PersonSession | None, detection_type: str, confidence: float, heuristic: bool = False,
                    evidence_image_path: str | None = None, video_path: str | None = None) -> None:
        session_db_id = session.db_session_id if session else None
        track_id = session.track_id if session else None

        def _write(db):
            db.add(DetectionEvent(
                session_id=session_db_id,
                detection_type=detection_type,
                confidence=confidence,
                person_track_id=track_id,
                heuristic=heuristic,
                evidence_image_path=evidence_image_path,
                video_path=video_path,
            ))

        enqueue_write(_write)

    def _dispatch_alert(self, detection_type: str, confidence: float, snapshot_path: str | None) -> None:
        payload = AlertPayload(
            detection_type=detection_type,
            confidence=confidence,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            atm_id=self.atm_id,
            location=settings.atm_location,
            snapshot_path=snapshot_path,
        )

        def _on_result(channel: str, status: str, detail: str) -> None:
            def _write(db, channel=channel, status=status, detail=detail):
                db.add(AlertLog(event_id=None, channel=channel, status=status, detail=detail))
            enqueue_write(_write)

        dispatch_alert_async(payload, _on_result)

    def admin_unlock(self, track_id: int | None = None) -> int:
        """Manually clears a LOCKDOWN state (simulated 'security cleared the
        alert' action). Returns how many sessions were unlocked."""
        unlocked = 0
        for session in self.sessions.values():
            if session.state != SessionState.LOCKDOWN:
                continue
            if track_id is not None and session.track_id != track_id:
                continue
            session.state = SessionState.MONITORING
            session.state_entered_at = time.monotonic()
            session.weapon_alert_sent = False
            self.weapon_confirmer.reset((session.track_id, "weapon"))
            unlocked += 1
        return unlocked

    def _reset_for_new_arrival(self) -> None:
        """Wall-clock gap since the last processed frame exceeded
        settings.session_resume_max_gap_seconds — treat this as a genuinely
        new arrival (app closed/backgrounded, not a brief network hiccup)
        rather than silently resuming whatever SessionState (including
        SIREN/LOCKDOWN) this user's tracker/sessions were left in.
        ByteTrack's own lost-track aging (tracking/tracker.py,
        lost_track_buffer) only advances on calls to update(), which only
        happen while frames are actually arriving — so on its own it cannot
        detect or age out a gap like this; without this wall-clock check a
        stale mid-alarm session could be immediately re-matched to the next
        person who steps in front of the camera and its audio replayed with
        no real detection having happened yet this visit."""
        for session in list(self.sessions.values()):
            outcome = "completed" if session.state == SessionState.TRANSACTION_ACTIVE else "abandoned"
            self._end_session(session, outcome)
        self.sessions.clear()
        self.tracker = PersonTracker()
        self._build_confirmers()

    # ------------------------------------------------------------------ #
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        self.evidence.add_frame(frame)
        now = time.monotonic()
        log_events: list[str] = []
        audio_events: list[dict] = []
        boxes: list[dict] = []

        if self._last_frame_wall_time is not None and (now - self._last_frame_wall_time) > settings.session_resume_max_gap_seconds:
            self._reset_for_new_arrival()
            log_events.append("Camera stream resumed after a long gap — session state reset.")
        self._last_frame_wall_time = now

        all_detections = self.detector.infer(frame)
        person_dets = [d for d in all_detections if d.category == "person" and d.passes_threshold]
        other_dets = [d for d in all_detections if d.category != "person"]

        tracked = self.tracker.update(person_dets)
        seen_track_ids: set[int] = set()

        for track_id, person_det in tracked:
            seen_track_ids.add(track_id)
            session = self.sessions.setdefault(track_id, PersonSession(track_id=track_id))
            session.last_seen = now
            boxes.append({"category": "person", "confidence": person_det.confidence, "bbox": person_det.bbox, "track_id": track_id})

            cx = (person_det.bbox[0] + person_det.bbox[2]) / 2.0
            cy = (person_det.bbox[1] + person_det.bbox[3]) / 2.0
            session.center_history.append((now, cx, cy))

            self._process_session(frame, session, person_det, other_dets, now, boxes, audio_events, log_events)

        # sessions whose track disappeared this frame
        for track_id, session in list(self.sessions.items()):
            if track_id in seen_track_ids:
                continue
            if now - session.last_seen > PERSON_ABSENCE_TIMEOUT_SECONDS:
                outcome = "completed" if session.state == SessionState.TRANSACTION_ACTIVE else "abandoned"
                self._end_session(session, outcome)
                del self.sessions[track_id]
                log_events.append(f"Person (track {track_id}) left — session {outcome}.")

        # multiple-person suspicious-behaviour check (non-blocking observability)
        active_states = {SessionState.VERIFYING, SessionState.TRANSACTION_ACTIVE}
        active_sessions = [s for s in self.sessions.values() if s.state in active_states]
        if len(self.sessions) > 1 and active_sessions:
            for session in active_sessions:
                if not session.multi_person_alert_sent:
                    session.multi_person_alert_sent = True
                    self._log_event(session, "multiple_persons", confidence=1.0)
                    log_events.append(f"Multiple persons present during track {session.track_id}'s transaction.")
        else:
            for session in self.sessions.values():
                session.multi_person_alert_sent = False

        if not self.sessions:
            system_status = "IDLE"
        elif any(s.state == SessionState.LOCKDOWN for s in self.sessions.values()):
            system_status = "LOCKDOWN"
        elif any(s.state == SessionState.SIREN for s in self.sessions.values()):
            system_status = "SIREN"
        else:
            system_status = "MONITORING"

        return FrameResult(
            system_status=system_status,
            boxes=boxes,
            sessions=[self._session_view(s) for s in self.sessions.values()],
            audio_events=audio_events,
            log_events=log_events,
        )

    def _session_view(self, session: PersonSession) -> dict:
        remaining = None
        now = time.monotonic()
        if session.state == SessionState.FACE_COVERED_GRACE:
            remaining = max(0.0, settings.face_cover_grace_seconds - (now - session.state_entered_at))
        elif session.state == SessionState.FACE_COVERED_WARNING:
            remaining = max(0.0, settings.voice_warning_wait_seconds - (now - session.state_entered_at))
        elif session.state == SessionState.SIREN:
            remaining = max(0.0, settings.siren_duration_seconds - (now - session.state_entered_at))
        return {
            "track_id": session.track_id,
            "state": session.state.value,
            "countdown_seconds": remaining,
            "cover_reason": session.last_cover_reason,
            "verified": session.state == SessionState.TRANSACTION_ACTIVE,
            "checklist": {
                # A session lingers in self.sessions for up to
                # PERSON_ABSENCE_TIMEOUT_SECONDS after the person actually
                # left frame (so a brief tracking drop-out doesn't reset
                # everything) — this used to always read "detected" during
                # that entire window even though nobody was there. Using
                # half of PERSON_ABSENCE_TIMEOUT_SECONDS (not a tight
                # per-frame threshold) only flips this false once a person
                # has been genuinely gone for a while, so it can't flicker
                # false from ordinary frame-to-frame timing jitter while
                # someone is still actually standing there.
                "person_detected": (now - session.last_seen) < (PERSON_ABSENCE_TIMEOUT_SECONDS / 2),
                "face_visible": session.last_face_score >= settings.face_presence_threshold,
                "face_score": round(session.last_face_score, 2),
                "liveness_blink": session.liveness_state.blink_detected,
                "liveness_head_move": session.liveness_state.head_move_detected,
                "verified": session.state == SessionState.TRANSACTION_ACTIVE,
            },
        }

    # ------------------------------------------------------------------ #
    def _process_session(self, frame, session: PersonSession, person_det: Detection, other_dets: list[Detection],
                          now: float, boxes: list[dict], audio_events: list[dict], log_events: list[str]) -> None:
        pbbox = person_det.bbox

        # --- face visibility, computed early so mask_hit below can
        # cross-check it. Skipped once already locked down — nothing later
        # in this state reads it, matching the original control flow.
        crop = frame
        face_score = 0.0
        face_visible_this_frame = False
        if session.state != SessionState.LOCKDOWN:
            crop = self._crop(frame, pbbox)
            face_score = self.face_detector.confidence(crop)
            session.last_face_score = face_score
            face_visible_this_frame = face_score >= settings.face_presence_threshold

        # --- associate helmet/mask/weapon detections with this person ---
        my_others = [d for d in other_dets if bbox_contains_center(pbbox, d.bbox)]

        # Helmet/mask specifically must land in the head region, not just
        # anywhere on the body — otherwise a mask pulled down to the chin,
        # held in a hand, or dangling at chest level keeps counting as "worn"
        # long after the face itself is uncovered. Weapons stay full-body
        # scoped since those are legitimately carried anywhere.
        head_bbox = head_region(pbbox)
        helmet_hit = any(
            d.category == "helmet" and d.passes_threshold and bbox_contains_center(head_bbox, d.bbox) for d in my_others
        )
        # This used to also require "and not face_visible_this_frame" (on the
        # theory that a genuine mask depresses MediaPipe's own presence
        # score, so requiring both agree would filter out the PPE model
        # misreading bare skin/chin/shadow as "mask"). In practice a worn
        # mask that leaves eyes and forehead visible does NOT reliably
        # depress MediaPipe's score — general face detectors are robust to
        # partial occlusion by design — so that guard made mask_hit only
        # true on the sporadic frames where MediaPipe's raw reading happened
        # to already dip, which meant it never stayed true for enough
        # consecutive frames to drive the covered-face escalation below:
        # mask_confirmed could flicker on but the voice warning/siren never
        # reliably followed. Symmetric with helmet_hit now (no such guard);
        # false positives are still bounded by conf_mask plus the 12-of-15
        # consecutive-frame confirmation window.
        mask_hit = any(
            d.category == "mask" and d.passes_threshold and bbox_contains_center(head_bbox, d.bbox)
            for d in my_others
        )
        weapon_hits = [d for d in my_others if d.is_weapon and d.passes_threshold]
        if weapon_hits:
            session.last_weapon_detection = max(weapon_hits, key=lambda d: d.confidence)

        # Confirm helmet/mask through their own fast, low-stakes window (see
        # settings.cover_confirm_*) *before* using them to override face
        # visibility below — using the raw single-frame hit here used to let
        # one stray false-positive PPE read (more likely now that
        # conf_helmet/conf_mask are tuned for recall) keep re-triggering
        # "covered" indefinitely, including well after the person actually
        # removed the helmet/mask, since the override ran on unsmoothed
        # per-frame noise with no release behavior of its own. The
        # confirmer's asymmetric hysteresis (cover_confirm_min_hits to rise,
        # cover_confirm_release_after_misses to fall) reacts in a few frames
        # either direction but is immune to single-frame flicker.
        helmet_confirmed = self.helmet_confirmer.update((session.track_id, "helmet"), helmet_hit)
        mask_confirmed = self.mask_confirmer.update((session.track_id, "mask"), mask_hit)
        weapon_confirmed = self.weapon_confirmer.update((session.track_id, "weapon"), bool(weapon_hits))

        # A helmet, mask, or a heuristically-detected covering (cloth,
        # bandana, anything the YOLO PPE models have no class for) overrides
        # MediaPipe's own presence score rather than merely coexisting with
        # it. General face detectors are deliberately robust to partial
        # occlusion, so a mask/cloth/open-face-helmet that still leaves the
        # eyes and forehead visible keeps MediaPipe confidently reporting
        # "face present" even though the face isn't actually usable for
        # identification (or acceptable at all, security-wise) — without this
        # override that combination sailed straight into VERIFYING instead of
        # being treated as a covered face. The occlusion heuristic only runs
        # when MediaPipe would otherwise call the face visible, since that's
        # the only case it needs to correct.
        # Skipped whenever helmet/mask already caught it (no need to also pay
        # for a MediaPipe FaceMesh pass on top of the two FaceDetection-scale
        # calls already made this frame) — this heuristic only exists for
        # the cases YOLO has no class for (cloth, bandanas, etc.), so it's
        # only worth invoking when nothing else already found gear.
        lower_face_occluded_raw = (
            session.state != SessionState.LOCKDOWN
            and face_visible_this_frame
            and not (helmet_confirmed or mask_confirmed)
            and self.occlusion_heuristic.lower_face_occluded(crop)
        )
        covering_confirmed = self.covering_heuristic_confirmer.update(
            (session.track_id, "covering"), lower_face_occluded_raw
        )
        face_occluded_by_gear = helmet_confirmed or mask_confirmed or covering_confirmed
        if session.state != SessionState.LOCKDOWN and face_occluded_by_gear:
            face_visible_this_frame = False
            face_score = 0.0
            session.last_face_score = 0.0
            # Helmet takes priority in the reason label even if mask/heuristic
            # also fired this frame — it's the more specific, actionable
            # instruction ("remove your helmet" beats a generic "uncover
            # your face" when we actually know it's a helmet).
            if helmet_confirmed:
                session.last_cover_reason = "helmet"
            elif mask_confirmed:
                session.last_cover_reason = "mask"
            else:
                session.last_cover_reason = "covering"

        # Only draw detections worth a human's attention: below half the real
        # alarm threshold it's pre-filter noise (see Detection.worth_showing),
        # and mislabeling noise as e.g. "weapon 27%" is actively misleading.
        # A box is "confirmed" (solid) once its category's rolling-window
        # check has actually tripped; otherwise it's shown dashed/tentative.
        confirmed_by_category = {"helmet": helmet_confirmed, "mask": mask_confirmed}
        for d in my_others:
            if not d.worth_showing:
                continue
            confirmed = weapon_confirmed if d.is_weapon else confirmed_by_category.get(d.category, False)
            boxes.append({
                "category": d.category, "confidence": d.confidence, "bbox": d.bbox,
                "track_id": session.track_id, "confirmed": confirmed,
            })

        heuristic_hit = detect_rigid_elongated_object(frame, pbbox)
        if heuristic_hit:
            boxes.append({"category": "possible_blunt_object", "confidence": heuristic_hit.confidence, "bbox": heuristic_hit.bbox,
                          "track_id": session.track_id, "heuristic": True, "confirmed": False})

        # --- aggressive-movement heuristic (coarse motion speed, not a trained action model) ---
        if len(session.center_history) >= 2:
            (t0, x0, y0), (t1, x1, y1) = session.center_history[-2], session.center_history[-1]
            dt = max(t1 - t0, 1e-3)
            speed = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt
            aggressive_confirmed = self.aggressive_confirmer.update((session.track_id, "aggressive"), speed > AGGRESSIVE_SPEED_THRESHOLD_PX_PER_SEC)
            if aggressive_confirmed:
                log_events.append(f"Track {session.track_id}: sustained rapid movement flagged as suspicious behaviour.")

        # --- Step 6: weapon detection takes priority over everything ---
        if weapon_confirmed and session.state != SessionState.LOCKDOWN:
            self._ensure_db_session(session)
            session.state = SessionState.LOCKDOWN
            session.state_entered_at = now
            session.lockdown_frames_to_capture = 3
            audio_events.append({"type": "danger_voice", "text": DANGER_TEXT, "track_id": session.track_id})
            audio_events.append({"type": "siren_start", "track_id": session.track_id})
            video_path = self.evidence.trigger_video(f"weapon_lockdown_track{session.track_id}")
            log_events.append(f"WEAPON DETECTED — track {session.track_id} locked down. Security notified.")
            worst = session.last_weapon_detection or Detection(
                category="weapon_unknown", confidence=settings.conf_weapon, bbox=pbbox, source_model="confirmed-fallback"
            )
            snapshot = self.evidence.capture_image(frame, f"weapon_{worst.category}")
            self._log_event(session, worst.category, worst.confidence, video_path=video_path, evidence_image_path=snapshot)
            if not session.weapon_alert_sent:
                session.weapon_alert_sent = True
                self._dispatch_alert(worst.category, worst.confidence, snapshot)

        if session.state == SessionState.LOCKDOWN:
            if session.lockdown_frames_to_capture > 0:
                self.evidence.capture_image(frame, f"weapon_followup_track{session.track_id}")
                session.lockdown_frames_to_capture -= 1
            return  # nothing else happens for this session while locked down

        # --- face visibility check (shared by Steps 2-4; computed at the top of this method) ---
        face_covered_confirmed = self.face_covered_confirmer.update((session.track_id, "covered"), not face_visible_this_frame)
        face_visible_confirmed = self.face_visible_confirmer.update((session.track_id, "visible"), face_visible_this_frame)

        # No trained class exists for scarves/cloth/hands/etc. — but the
        # workflow's face-covered escalation (grace -> warning -> siren below)
        # already reacts to *any* covering because it's driven by this
        # mediapipe presence score, not by helmet/mask category matches. This
        # box just makes that generic signal visible on the dashboard too,
        # honestly labeled as a coverage finding rather than a fake specific
        # class, same pattern as the blunt-object heuristic.
        if face_covered_confirmed and not (helmet_confirmed or mask_confirmed):
            boxes.append({
                "category": "face_covered", "confidence": 1.0, "bbox": head_bbox,
                "track_id": session.track_id, "heuristic": True, "confirmed": True,
            })

        if session.state in (SessionState.MONITORING, SessionState.VERIFYING, SessionState.TRANSACTION_ACTIVE):
            if face_covered_confirmed:
                session.state = SessionState.FACE_COVERED_GRACE
                session.state_entered_at = now
                session.audio.reset()
                session.liveness_state.reset()
                # Warn immediately on detection, not only after the full
                # grace period elapses — previously the voice/on-screen
                # warning only fired once FACE_COVERED_WARNING was reached
                # (after face_cover_grace_seconds of silence), which read as
                # "the siren just starts, no warning happens first" even
                # though a warning eventually did, just much later than the
                # user expected "as soon as a covering is detected" to mean.
                session.audio.voice_warning_fired = True
                if session.last_cover_reason == "helmet":
                    audio_events.append({"type": "helmet_warning", "text": HELMET_REMOVE_TEXT, "track_id": session.track_id})
                else:
                    audio_events.append({"type": "voice_warning", "text": VOICE_WARNING_TEXT, "track_id": session.track_id})
                log_events.append(f"Track {session.track_id}: face covering detected, starting grace period.")
            elif session.state == SessionState.MONITORING and face_visible_confirmed:
                session.state = SessionState.VERIFYING
                session.state_entered_at = now

        elif session.state == SessionState.FACE_COVERED_GRACE:
            if face_visible_confirmed:
                session.state = SessionState.VERIFYING
                session.state_entered_at = now
            elif now - session.state_entered_at >= settings.face_cover_grace_seconds:
                session.state = SessionState.FACE_COVERED_WARNING
                session.state_entered_at = now

        elif session.state == SessionState.FACE_COVERED_WARNING:
            # Normally already fired at FACE_COVERED_GRACE entry above; this
            # is only a fallback in case a session is somehow created
            # directly in this state (e.g. a future admin/API path).
            if not session.audio.voice_warning_fired:
                session.audio.voice_warning_fired = True
                if session.last_cover_reason == "helmet":
                    audio_events.append({"type": "helmet_warning", "text": HELMET_REMOVE_TEXT, "track_id": session.track_id})
                else:
                    audio_events.append({"type": "voice_warning", "text": VOICE_WARNING_TEXT, "track_id": session.track_id})
            if face_visible_confirmed:
                session.state = SessionState.VERIFYING
                session.state_entered_at = now
                audio_events.append({"type": "uncovered_voice", "text": UNCOVERED_TEXT, "track_id": session.track_id})
                session.audio.reset()
                session.last_cover_reason = None
            elif now - session.state_entered_at >= settings.voice_warning_wait_seconds:
                session.state = SessionState.SIREN
                session.state_entered_at = now

        elif session.state == SessionState.SIREN:
            if not session.audio.siren_fired:
                session.audio.siren_fired = True
                session.audio.siren_started_at = now
                audio_events.append({"type": "siren_start", "track_id": session.track_id})
                self._ensure_db_session(session)
                video_path = self.evidence.trigger_video(f"face_covered_track{session.track_id}")
                snapshot = self.evidence.capture_image(frame, "face_covered")
                self._log_event(session, "face_covered_siren", confidence=1.0, video_path=video_path, evidence_image_path=snapshot)
                log_events.append(f"SIREN — track {session.track_id} refused to uncover face.")
                if not session.face_covered_alert_sent:
                    session.face_covered_alert_sent = True
                    self._dispatch_alert("face_covered_siren", 1.0, snapshot)
            # Unlike GRACE/WARNING above, this used to ONLY check the elapsed
            # time — meaning it stayed deaf to the person's actual face for a
            # fixed siren_duration_seconds (20s) no matter what they did in
            # front of the camera, including uncovering or swapping to a
            # different covering. That's what made a same-session retest look
            # "broken" until either the full 20s passed or the page was
            # refreshed into a brand-new pipeline. Now it exits the instant
            # compliance is confirmed; the duration is only a safety cap for
            # the case where the face is never shown again.
            siren_duration_elapsed = now - session.state_entered_at >= settings.siren_duration_seconds
            if face_visible_confirmed or siren_duration_elapsed:
                audio_events.append({"type": "siren_stop", "track_id": session.track_id})
                if face_visible_confirmed:
                    audio_events.append({"type": "uncovered_voice", "text": UNCOVERED_TEXT, "track_id": session.track_id})
                session.audio.reset()
                session.face_covered_alert_sent = False
                if face_visible_confirmed:
                    session.state = SessionState.VERIFYING
                    session.last_cover_reason = None
                else:
                    session.state = SessionState.FACE_COVERED_WARNING
                session.state_entered_at = now

        # --- Step 4: face detection -> liveness -> verification ---
        if session.state == SessionState.VERIFYING:
            if session.liveness_state.expired and not session.liveness_state.is_live:
                session.liveness_state.reset()  # give the user a fresh 6s attempt rather than waiting forever on one stale partial signal
            self.liveness.process(crop, session.liveness_state)
            face_detect_confirmed = self.face_detect_confirmer.update(
                (session.track_id, "detect"), face_score >= settings.conf_face_detect
            )
            live_ok = session.liveness_state.is_live

            verify_ok = False
            verify_conf = 0.0
            if face_detect_confirmed and live_ok and session.liveness_state.last_face_bbox:
                fb = session.liveness_state.last_face_bbox
                face_crop = crop[fb[1]:fb[3], fb[0]:fb[2]]
                if face_crop.size > 0:
                    probe = self.verifier.embed(face_crop)
                    enrolled = self._load_enrolled()
                    if enrolled:
                        label, sim = self._best_enrolled_match(probe)
                        verify_ok = sim >= settings.face_match_cosine_threshold
                        verify_conf = sim
                    elif session.reference_embedding is None:
                        session.reference_embedding = probe
                        verify_ok, verify_conf = True, 1.0
                    else:
                        ok, sim = self.verifier.matches(probe, session.reference_embedding)
                        verify_ok, verify_conf = ok, sim

            verify_confirmed = self.face_verify_confirmer.update((session.track_id, "verify"), verify_ok)

            if face_detect_confirmed and live_ok and verify_confirmed:
                self._ensure_db_session(session)
                session.state = SessionState.TRANSACTION_ACTIVE
                session.state_entered_at = now
                audio_events.append({"type": "verified_voice", "text": VERIFIED_TEXT, "track_id": session.track_id})
                log_events.append(f"Track {session.track_id}: verified — transaction started.")

                def _write(db, sid=session.db_session_id):
                    row = db.get(ATMSession, sid) if sid else None
                    if row:
                        row.verification_status = "verified"
                enqueue_write(_write)

        # --- Step 5: continuous monitoring during an active transaction ---
        if session.state == SessionState.TRANSACTION_ACTIVE:
            if helmet_confirmed:
                self._log_event(session, "helmet_during_transaction", confidence=1.0)
                log_events.append(f"Track {session.track_id}: helmet detected during transaction.")
                self.helmet_confirmer.reset((session.track_id, "helmet"))
            if mask_confirmed:
                self._log_event(session, "mask_during_transaction", confidence=1.0)
                log_events.append(f"Track {session.track_id}: mask detected during transaction.")
                self.mask_confirmer.reset((session.track_id, "mask"))
            # This state used to have no exit at all besides the person
            # leaving frame — if they stayed in view after being verified
            # once, nothing further could ever happen for that track without
            # a page refresh (see settings.transaction_active_timeout_seconds
            # for the full story). Returning to MONITORING here — and
            # requiring a fresh liveness proof, not reusing a stale
            # blink/head-turn from the previous cycle — lets a new
            # verification cycle start on its own the moment the person is
            # ready to go through it again, no refresh needed.
            if now - session.state_entered_at >= settings.transaction_active_timeout_seconds:
                session.state = SessionState.MONITORING
                session.state_entered_at = now
                session.liveness_state.reset()
                log_events.append(f"Track {session.track_id}: transaction session ended — ready for re-verification.")
