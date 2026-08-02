"""State-machine behavior tests. Uses a stub DetectionEngine (no real YOLO
weights loaded) so these stay fast and independent of model downloads —
liveness/verification still run for real via MediaPipe/FaceNet since those
are cheap to construct and a blank frame deterministically has "no face",
which is exactly the scenario the face-covering workflow needs to exercise.
Timing thresholds are shrunk to fractions of a second via tests/conftest.py
env overrides so this suite runs in well under a second of wall-clock time.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from config.settings import settings
from models.yolo_engine import Detection
from workflow.pipeline import SecurityPipeline, SessionState


class StubEngine:
    device = "cpu"

    def __init__(self, detections_fn):
        self._fn = detections_fn

    def infer(self, frame):
        return self._fn(frame)


@pytest.fixture
def blank_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_face_covered_escalates_to_grace_then_warning_then_siren(blank_frame):
    person_bbox = (50, 50, 300, 450)
    engine = StubEngine(lambda f: [Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub")])
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        seen_states = set()
        deadline = time.monotonic() + 5.0
        siren_fired = False
        while time.monotonic() < deadline:
            result = pipeline.process_frame(blank_frame)
            for s in result.sessions:
                seen_states.add(s["state"])
            if any(e["type"] == "siren_start" for e in result.audio_events):
                siren_fired = True
                break
            time.sleep(0.05)

        assert SessionState.FACE_COVERED_GRACE.value in seen_states
        assert siren_fired, f"siren never fired; states observed: {seen_states}"
    finally:
        pipeline.close()


def test_weapon_detection_triggers_lockdown_within_confirmation_window(blank_frame):
    person_bbox = (50, 50, 300, 450)
    weapon_bbox = (100, 100, 150, 300)

    def detections(_frame):
        return [
            Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub"),
            Detection(category="weapon_gun", confidence=0.99, bbox=weapon_bbox, source_model="stub"),
        ]

    engine = StubEngine(detections)
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        locked_down = False
        for _ in range(20):
            result = pipeline.process_frame(blank_frame)
            if any(s["state"] == SessionState.LOCKDOWN.value for s in result.sessions):
                locked_down = True
                break
        assert locked_down, "weapon was not confirmed into a lockdown state within 20 frames"
    finally:
        pipeline.close()


def test_helmet_triggers_helmet_specific_warning_not_generic(blank_frame):
    """A worn helmet must produce the helmet-specific voice event/reason, not
    the generic face-covering one — otherwise the dashboard/voice can't tell
    the user *what* to remove."""
    person_bbox = (50, 50, 300, 450)
    helmet_bbox = (100, 60, 200, 150)  # center lands in the top-50% head region

    def detections(_frame):
        return [
            Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub"),
            Detection(category="helmet", confidence=0.8, bbox=helmet_bbox, source_model="stub"),
        ]

    engine = StubEngine(detections)
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        helmet_warning_fired = False
        cover_reason_seen = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = pipeline.process_frame(blank_frame)
            for s in result.sessions:
                if s.get("cover_reason"):
                    cover_reason_seen = s["cover_reason"]
            if any(e["type"] == "helmet_warning" for e in result.audio_events):
                helmet_warning_fired = True
                break
            assert not any(e["type"] == "voice_warning" for e in result.audio_events), \
                "generic voice_warning fired for a helmet-only covering"
            time.sleep(0.02)

        assert helmet_warning_fired, "helmet-specific voice event never fired"
        assert cover_reason_seen == "helmet"
    finally:
        pipeline.close()


def test_stale_session_does_not_resume_after_long_disconnect_gap(blank_frame):
    """A long wall-clock gap between frames (app closed/backgrounded, not a
    brief reconnect) must reset to a clean MONITORING state instead of
    silently resuming whatever alarm state the session was left in — this is
    what previously made the siren appear to fire "instantly" for a brand
    new visit that inherited a stale SIREN session from before the gap."""
    person_bbox = (50, 50, 300, 450)
    engine = StubEngine(lambda f: [Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub")])
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        siren_fired = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            result = pipeline.process_frame(blank_frame)
            if any(e["type"] == "siren_start" for e in result.audio_events):
                siren_fired = True
                break
            time.sleep(0.02)
        assert siren_fired, "setup failed: siren never fired before simulating the gap"
        assert any(s["state"] == SessionState.SIREN.value for s in result.sessions)

        # Simulate a long disconnect by rewinding the pipeline's last-frame
        # clock rather than actually sleeping settings.session_resume_max_gap_seconds.
        pipeline._last_frame_wall_time -= (settings.session_resume_max_gap_seconds + 1.0)

        result = pipeline.process_frame(blank_frame)
        assert not any(s["state"] == SessionState.SIREN.value for s in result.sessions), \
            "session resumed SIREN state immediately after a long gap instead of resetting"
        assert not any(e["type"] == "siren_start" for e in result.audio_events), \
            "siren_start fired on the very first frame after a long gap with no real detection yet"
    finally:
        pipeline.close()


def test_voice_warning_fires_on_grace_entry_not_after_full_grace_period(blank_frame):
    """The voice warning must fire the moment a covering is confirmed (grace
    entry) — not settings.face_cover_grace_seconds later — otherwise the
    siren looks like the very first reaction with no warning beforehand."""
    person_bbox = (50, 50, 300, 450)
    mask_bbox = (100, 60, 200, 150)

    def detections(_frame):
        return [
            Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub"),
            Detection(category="mask", confidence=0.8, bbox=mask_bbox, source_model="stub"),
        ]

    engine = StubEngine(detections)
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        grace_entered_frame = None
        voice_warning_frame = None
        for i in range(30):
            result = pipeline.process_frame(blank_frame)
            if grace_entered_frame is None and any(
                s["state"] == SessionState.FACE_COVERED_GRACE.value for s in result.sessions
            ):
                grace_entered_frame = i
            if voice_warning_frame is None and any(e["type"] == "voice_warning" for e in result.audio_events):
                voice_warning_frame = i
            if voice_warning_frame is not None:
                break

        assert grace_entered_frame is not None, "covering was never confirmed into grace"
        assert voice_warning_frame is not None, "voice_warning never fired"
        assert voice_warning_frame == grace_entered_frame, \
            "voice_warning fired later than the frame grace was entered on"
    finally:
        pipeline.close()


def test_mask_confirmed_releases_and_stays_released_after_removal(blank_frame):
    """After a genuine mask is removed, mask_confirmed must clear quickly and
    stay clear even if an occasional isolated stray false-positive mask
    detection still fires — this is the exact scenario that used to leave
    the system 'stuck' behaving as if the mask were still present, because
    the override driving that behavior used to read the noisy raw per-frame
    detection instead of this smoothed/confirmed signal."""
    person_bbox = (50, 50, 300, 450)
    mask_bbox = (100, 60, 200, 150)
    state = {"n": 0, "mask_on": True}

    def detections(_frame):
        state["n"] += 1
        dets = [Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub")]
        if state["mask_on"]:
            dets.append(Detection(category="mask", confidence=0.8, bbox=mask_bbox, source_model="stub"))
        elif state["n"] % 10 == 0:  # sparse stray false positive after real removal
            dets.append(Detection(category="mask", confidence=0.55, bbox=mask_bbox, source_model="stub"))
        return dets

    engine = StubEngine(detections)
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        for _ in range(20):
            pipeline.process_frame(blank_frame)

        state["mask_on"] = False
        for _ in range(15):
            pipeline.process_frame(blank_frame)  # let the confirmer release

        still_confirmed_count = 0
        for _ in range(40):
            result = pipeline.process_frame(blank_frame)
            mask_boxes = [b for b in result.boxes if b["category"] == "mask" and b.get("confirmed")]
            if mask_boxes:
                still_confirmed_count += 1
        assert still_confirmed_count == 0, "mask stayed 'confirmed' after removal despite only sparse stray detections"
    finally:
        pipeline.close()


def test_brief_face_score_dip_does_not_reset_liveness_progress(blank_frame):
    """A momentary dip in the raw face-presence score (e.g. an ordinary
    blink, or the head-turn the liveness check itself requires the user to
    perform) must not be misread as a face covering appearing — that would
    call session.liveness_state.reset() and make blink+head-turn
    impossible to ever complete together, i.e. verification could never
    finish. Regression test for the 2026-07-17 same-day fix/revert where
    face_covered_confirmer/face_visible_confirmer were briefly moved onto
    the fast covering-detection window."""
    person_bbox = (50, 50, 300, 450)
    engine = StubEngine(lambda f: [Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub")])
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        # Blank frames deterministically have "no face" to MediaPipe, so
        # drive the face-presence score directly instead, to simulate a
        # real visible face without needing a real face image fixture.
        scores = {"value": 0.9}
        pipeline.face_detector.confidence = lambda crop: scores["value"]

        result = None
        for _ in range(20):
            result = pipeline.process_frame(blank_frame)
            if any(s["state"] == SessionState.VERIFYING.value for s in result.sessions):
                break
        assert any(s["state"] == SessionState.VERIFYING.value for s in result.sessions), \
            "setup failed: never reached VERIFYING with a controllable high face score"

        # Simulate a several-frame dip below face_presence_threshold (a
        # blink, or reduced detector confidence mid head-turn) sandwiched
        # between normal high-confidence frames. 4 frames reproduces the
        # exact bug: confirmed by temporarily forcing face_covered/
        # face_visible onto the fast covering window, which flipped into
        # FACE_COVERED_GRACE mid-dip on precisely this length.
        scores["value"] = 0.1
        for _ in range(4):
            pipeline.process_frame(blank_frame)
        scores["value"] = 0.9
        for _ in range(5):
            result = pipeline.process_frame(blank_frame)

        assert not any(s["state"] == SessionState.FACE_COVERED_GRACE.value for s in result.sessions), \
            "a brief blink-like face-score dip was misread as a face covering appearing"
    finally:
        pipeline.close()


def test_transaction_active_returns_to_monitoring_after_timeout(blank_frame):
    """A verified session must return to MONITORING on its own after
    settings.transaction_active_timeout_seconds, even if the person never
    leaves frame — previously TRANSACTION_ACTIVE had no exit path at all
    without a page refresh forcing a long enough frame gap to trip the
    unrelated person-absence cleanup, which is what made the system look
    like it "only works once" until refreshed."""
    person_bbox = (50, 50, 300, 450)
    engine = StubEngine(lambda f: [Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub")])
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        pipeline.face_detector.confidence = lambda crop: 0.95

        def fake_liveness_process(crop, state):
            state.blink_detected = True
            state.head_move_detected = True
            state.last_face_bbox = (0, 0, 50, 50)
            return state
        pipeline.liveness.process = fake_liveness_process

        result = None
        for _ in range(30):
            result = pipeline.process_frame(blank_frame)
            if any(s["state"] == SessionState.TRANSACTION_ACTIVE.value for s in result.sessions):
                break
        assert any(s["state"] == SessionState.TRANSACTION_ACTIVE.value for s in result.sessions), \
            "setup failed: never reached TRANSACTION_ACTIVE"

        track_id = next(s["track_id"] for s in result.sessions if s["state"] == SessionState.TRANSACTION_ACTIVE.value)
        pipeline.sessions[track_id].state_entered_at -= (settings.transaction_active_timeout_seconds + 1.0)

        result = pipeline.process_frame(blank_frame)
        assert not any(s["state"] == SessionState.TRANSACTION_ACTIVE.value for s in result.sessions), \
            "session stayed in TRANSACTION_ACTIVE past the timeout without the person ever leaving frame"
        assert any(s["state"] == SessionState.MONITORING.value for s in result.sessions)
    finally:
        pipeline.close()


def test_single_frame_weapon_spike_does_not_trigger_lockdown(blank_frame):
    """Anti-false-positive requirement: one noisy frame must never alarm."""
    person_bbox = (50, 50, 300, 450)
    weapon_bbox = (100, 100, 150, 300)
    call_count = {"n": 0}

    def detections(_frame):
        call_count["n"] += 1
        dets = [Detection(category="person", confidence=0.9, bbox=person_bbox, source_model="stub")]
        if call_count["n"] == 1:  # weapon appears on exactly one frame only
            dets.append(Detection(category="weapon_gun", confidence=0.99, bbox=weapon_bbox, source_model="stub"))
        return dets

    engine = StubEngine(detections)
    pipeline = SecurityPipeline(engine, atm_id="TEST-ATM")
    try:
        for _ in range(10):
            result = pipeline.process_frame(blank_frame)
            assert not any(s["state"] == SessionState.LOCKDOWN.value for s in result.sessions)
    finally:
        pipeline.close()
