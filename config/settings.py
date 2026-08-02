"""Central configuration. All tunables live here — nowhere else should hardcode
a threshold, duration, or credential. Values can be overridden via a `.env` file
or real environment variables without touching code.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), env_file_encoding="utf-8", extra="ignore")

    # --- Identity ---
    atm_id: str = "ATM-001"
    atm_location: str = "Canara Bank ATM, Madanapalle"

    # --- Security ---
    jwt_secret: str = "CHANGE_ME_INSECURE_DEFAULT_DEV_SECRET_ROTATE_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480  # 8h — this console is meant to stay open on a monitor, not re-login every hour
    field_encryption_key: str = ""  # Fernet key (base64, 32 bytes); auto-generated on first run if blank

    # --- Paths ---
    db_path: Path = BASE_DIR / "database" / "omni_mind.db"
    log_dir: Path = BASE_DIR / "logs"
    evidence_dir: Path = BASE_DIR / "evidence"
    weights_dir: Path = BASE_DIR / "models" / "weights"

    # --- Inference ---
    device_preference: str = "auto"  # "auto" | "cuda" | "cpu"
    inference_img_size: int = 640
    target_fps: int = 15  # frames actually run through YOLO per second (browser can send more; we throttle)

    # --- Anti-false-positive ---
    confirm_window_frames: int = 15         # consecutive-frame confirmation window (spec requirement)
    confirm_min_hits: int = 12              # hits required within the window before a detection is "confirmed"
    confirm_release_after_misses: int = 2   # consecutive real negative reads that clear an already-confirmed detection

    # A worn helmet/mask/covering is a much lower-stakes, cleaner, more
    # binary visual signal than a weapon or a face-verification match — the
    # worst case of reacting fast is asking someone to show a face they
    # weren't actually covering, not a false security lockdown. Field
    # testing reported the shared 15-of-12 window above made the covering
    # workflow feel sluggish both to react to a covering *appearing* and,
    # worse, to notice one had been *removed* (a stray single-frame false
    # helmet/mask read could keep re-triggering the covered state for a
    # long time under the slower window). This dedicated, faster window is
    # used only for the covering-detection channel; weapon/verification
    # keep the conservative window above untouched.
    cover_confirm_window_frames: int = 5
    cover_confirm_min_hits: int = 4
    cover_confirm_release_after_misses: int = 2

    # --- Confidence thresholds (spec-mandated) ---
    # conf_helmet/conf_mask started at the spec's literal 0.95/0.92 but, like
    # conf_face_detect below, that was never reachable in practice — these are
    # small nano PPE models running on a compressed live webcam frame, and
    # real worn-helmet/worn-mask detections were empirically landing well
    # below that, so the confirmation window (12 of 15 frames) could never
    # fill regardless of what the user did. conf_mask was confirmed reliable
    # at 0.5 on a laptop webcam in earlier testing, but real-device (iPhone)
    # field testing showed a worn, open-face helmet with the face partly
    # visible still slipped past the covering check into verification —
    # helmet confidence on a live phone stream (more compression, more
    # motion, worse average lighting than the laptop-webcam baseline) reads
    # lower still. Both are lowered again here; false-positive risk stays
    # bounded by the unchanged 12-of-15 confirmation window plus
    # confirm_release_after_misses, so this is trading a bit more margin for
    # the recall a real handheld phone camera needs.
    conf_helmet: float = 0.3
    conf_mask: float = 0.42
    conf_weapon: float = 0.96
    # NOTE: the spec's literal 0.98 was never reachable in practice — MediaPipe's
    # raw short-range face-detector score on a compressed webcam frame rarely
    # sustains that high for 12/15 consecutive frames, so verification could
    # never advance past this gate regardless of anything the user did (see
    # face_match_cosine_threshold below for the same recalibration done for
    # the verify step). 0.85 is a real, empirically-reachable operating point
    # still well above face_presence_threshold's "some face is there" bar.
    conf_face_detect: float = 0.85
    conf_face_verify: float = 0.99
    conf_person: float = 0.5

    # --- Workflow timings (seconds) ---
    face_cover_grace_seconds: float = 10      # Step 2: countdown before voice warning
    voice_warning_wait_seconds: float = 5     # Step 3: wait after voice warning before siren
    siren_duration_seconds: float = 20
    # TRANSACTION_ACTIVE previously had no exit path at all besides the
    # person physically leaving the camera's view for
    # PERSON_ABSENCE_TIMEOUT_SECONDS (workflow/pipeline.py) — if they simply
    # stayed in frame after being verified once, the session sat in
    # TRANSACTION_ACTIVE forever with nothing left to do, and the *only*
    # thing that could unstick it was a page refresh forcing enough of a
    # real frame gap to trip that same absence cleanup. This timeout gives
    # it a real exit back to MONITORING on its own, so a second
    # verification attempt works without needing a refresh in between.
    transaction_active_timeout_seconds: float = 15
    # A WebSocket gap longer than this means the app was actually closed/
    # backgrounded, not a brief network hiccup — the pipeline resets to a
    # clean MONITORING state instead of resuming stale session/track state
    # (see SecurityPipeline._reset_for_new_arrival). Live server logs showed
    # real phone reconnects (mobile Safari tab throttling / screen lock)
    # recurring every ~10-20s even during otherwise-normal use, well above
    # this setting's first value of 20s — that was tight enough to
    # occasionally reset genuine in-progress sessions mid-reconnect, which
    # is indistinguishable from "detection got slow/stuck" to the user. 45s
    # keeps real multi-minute absences (app actually closed) resetting
    # cleanly while giving normal reconnect churn a safe margin.
    session_resume_max_gap_seconds: float = 45.0

    # --- Face presence (Steps 2/3: is a face visible at all, covering check) ---
    face_presence_threshold: float = 0.6  # MediaPipe face-detector score above which we call a face "visible"

    # --- Liveness ---
    liveness_blink_required: bool = True
    liveness_head_move_required: bool = True
    liveness_min_ear_delta: float = 0.08    # eye-aspect-ratio drop that counts as a blink
    # 8.0deg within the old hardcoded 6s window required a deliberate, prompted
    # head turn — a person just standing normally rarely produces that much yaw
    # that fast, so liveness (and therefore verification) silently never
    # completed. 4.0deg catches normal postural sway/settling; window_seconds
    # below gives it enough time to occur naturally instead of forcing it.
    liveness_min_yaw_delta_deg: float = 4.0
    liveness_window_seconds: float = 12.0

    # --- Face verification ---
    face_match_cosine_threshold: float = 0.55  # facenet-pytorch (vggface2) empirical operating point

    # --- Notifications (all optional; disabled gracefully if unset) ---
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    alert_email_to: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Emergency / police notification ---
    # Fires alongside the admin email/Telegram alerts, only for the two
    # highest-severity events (weapon lockdown, siren) — see
    # workflow/pipeline.py's two _dispatch_alert() call sites, both already
    # scoped to exactly those events. Deliberately left blank by default:
    # this must be the real, verified contact for this ATM's actual nearest
    # police station, not a guessed value — same "real integration, inert
    # until configured" pattern as the SMTP/Telegram settings above, not a
    # fake capability. police_phone is informational only (shown so a human
    # operator can dial it manually) — no automated voice-call/SMS dispatch
    # to emergency services exists, and none is claimed to.
    police_station_name: str = ""
    police_email: str = ""
    police_phone: str = ""

    # --- Network ---
    host: str = "0.0.0.0"
    port: int = 8420


settings = Settings()
for _dir in (settings.log_dir, settings.evidence_dir, settings.weights_dir, settings.db_path.parent):
    _dir.mkdir(parents=True, exist_ok=True)
