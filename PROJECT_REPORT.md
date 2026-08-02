# Omni Mind — AI-Powered ATM Security System
### Final Year Project Report

---

## Abstract

Omni Mind is a real-time, AI-powered security system designed for ATM booths and similar unattended retail kiosks. It uses a standard browser camera (no specialized hardware) to continuously watch a booth, detect threats such as weapons or attempts to disguise identity, and verify that the person present is a live human being before allowing a transaction to proceed. The system combines three areas of computer vision — object detection, face liveness analysis, and face verification — into a single coordinated state machine, and exposes a live operator dashboard over a secure web connection. It is built as a full-stack application: a Python/FastAPI backend performs all AI inference on the server (so it can use a GPU and stay hardware-agnostic on the client side), while a lightweight browser frontend handles camera capture and live visualization. The deployed instance is configured for a fixed real-world location — **Canara Bank ATM, Madanapalle** — and automatically escalates the two highest-severity events (a detected weapon, or a refused face-covering siren) to the ATM's nearest police station by email, in addition to the existing administrator notification channels. The project emphasizes engineering honesty — every detection capability is documented as either a real, trained model or an explicitly-labelled best-effort heuristic — and anti-false-positive design, using rolling-window confirmation rather than single-frame triggers before raising any alert.

---

## 1. Introduction

### 1.1 Problem Statement

ATMs and unattended kiosks are common targets for two distinct classes of risk: **armed threats** (robbery, assault near the machine) and **identity concealment** (a person hiding their face with a helmet, mask, or cloth before or during a transaction, which is a strong indicator of intended fraud or crime). Most existing ATM security relies on passive CCTV recording that is reviewed *after* an incident, not systems that actively watch, reason about, and respond to a threat *as it happens*.

### 1.2 Objectives

The project set out to build a prototype that:

1. Detects the presence of a person and continuously monitors them via a live camera feed.
2. Detects weapons (guns, knives, blunt objects) and immediately escalates to a lockdown/alert state.
3. Detects attempts to cover the face (helmet, mask, or other covering) and asks the user, via voice and on-screen prompt, to uncover their face — escalating to an audible alarm only if they refuse.
4. Confirms the person in front of the camera is a live human (not a photo or screen replay) using liveness checks (blink + head movement).
5. Verifies the person's identity via face embeddings before allowing a transaction to be marked as "verified."
6. Logs every event, captures evidence (snapshots/video clips), and can notify administrators through email/Telegram.
7. Automatically escalates the two highest-severity events — a confirmed weapon, or a refused face-covering siren — to the fixed ATM's nearest police station, in addition to the administrator alert.
8. Does all of this in real time, on ordinary consumer hardware (a phone or laptop browser as the camera, one GPU-equipped PC as the server), with **no specialized security hardware**.
9. Is honest about which detections are backed by real trained models versus best-effort heuristics — an explicit design principle carried through the whole project, not an afterthought.

### 1.3 Scope

This is a prototype/demonstration system, not a certified banking product. It has no real card-reader or banking backend integration — "verification" means proving liveness and matching a face against either an admin-enrolled allowlist or the person's own reference embedding captured earlier in the same session (self-consistency). This scope decision is deliberate and stated explicitly rather than implied, and is one of several places the project favors technical honesty over an inflated feature list — a point worth raising directly in an interview.

---

## 2. Technologies, Frameworks & Tools

The project is written entirely in **Python** (backend) and **vanilla JavaScript/HTML/CSS** (frontend) — no frontend build tooling (React/Webpack/npm) was used, which keeps the client extremely lightweight and dependency-free.

### 2.1 Backend

| Technology | Purpose |
|---|---|
| **Python 3.12** | Primary implementation language for the entire backend and AI pipeline. |
| **FastAPI** | Modern async web framework. Serves the REST API, the WebSocket video stream, and the static dashboard files. Chosen for native `async`/`WebSocket` support and automatic request validation via Pydantic. |
| **Uvicorn** | ASGI server that actually runs the FastAPI application (with HTTPS, using a self-signed certificate). |
| **WebSockets** | The live video stream (browser → server) and live results (server → browser) both run over a persistent WebSocket connection rather than repeated HTTP requests, which is what makes near-real-time feedback possible. |
| **SQLAlchemy (ORM) + SQLite** | Relational database layer. SQLite was chosen for a self-contained, zero-install prototype; SQLAlchemy gives a clean upgrade path to PostgreSQL later without rewriting the data layer. |
| **Pydantic / pydantic-settings** | Type-safe request/response schemas, and a single typed `Settings` object that loads every tunable value (thresholds, timings, secrets) from a `.env` file instead of hardcoding them. |
| **python-jose** | Encodes and verifies JWT (JSON Web Token) access tokens for authentication. |
| **passlib + bcrypt** | Secure one-way password hashing for stored user credentials. |
| **cryptography (Fernet)** | Symmetric field-level encryption applied to sensitive database columns (face embeddings), so even direct database access doesn't expose raw biometric data. |

### 2.2 AI / Computer Vision Stack

| Technology | Purpose |
|---|---|
| **Ultralytics YOLO (YOLO11n / YOLOv8n)** | Real-time object detection. Three separate nano-sized YOLO models run per frame: a COCO-pretrained model (person, knife, baseball bat), a fine-tuned weapon/threat model (gun, knife, explosive, grenade), and a PPE model (helmet, mask). "Nano" models are used specifically because they are small and fast enough to run at real-time frame rates on a single consumer GPU. |
| **MediaPipe (Face Detection + Face Mesh)** | Google's on-device ML framework, used for two jobs: a lightweight scalar "is a face present" detector, and full facial landmark tracking (468 points) used for liveness (blink/head-turn) and precise face cropping. |
| **FaceNet (facenet-pytorch, InceptionResnetV1, VGGFace2-pretrained)** | Converts a cropped face into a 512-dimension numeric "embedding" — a fingerprint of that face. Two embeddings are compared with cosine similarity to decide if they're the same person. |
| **supervision (ByteTrack)** | Multi-object tracking. Turns a series of independent per-frame detections into a *stable identity* (a track ID) for each physical person across frames, which is what allows the system to reason about "this specific person's" state over time instead of re-analyzing from scratch every frame. |
| **PyTorch (CUDA build)** | The underlying deep-learning runtime all of the above models run on; configured to use GPU acceleration (NVIDIA CUDA) when available, with automatic CPU fallback. |
| **OpenCV** | Image decoding (incoming JPEG frames), cropping, color-space conversion, classic computer-vision heuristics (e.g. edge/contour detection for the blunt-object heuristic), and evidence video writing. |
| **NumPy** | Array/matrix operations underpinning virtually every image and embedding computation in the pipeline. |

### 2.3 Frontend

| Technology | Purpose |
|---|---|
| **HTML5 / CSS3** | Dashboard structure and a custom-designed dark theme (no CSS framework). |
| **Vanilla JavaScript** | Camera access (`getUserMedia`), frame capture and JPEG encoding, the WebSocket client, live canvas overlay drawing (bounding boxes, countdown banners), and audio playback logic — all hand-written with no build step, so the app runs by simply serving static files. |
| **Canvas API** | Renders the live video frame and draws detection boxes / labels on top of it in real time. |
| **HTML5 `<audio>` elements** | Plays pre-recorded voice-warning and siren `.wav` files. Chosen deliberately over the Web Speech API / Web Audio oscillators because iOS Safari's hardware mute switch silences those, but not `<audio>` element playback — a real cross-device compatibility finding from testing on an actual iPhone. |
| **Screen Wake Lock API** | Keeps the phone screen from dimming/locking during an active monitoring session, preventing the mobile browser from throttling the camera/WebSocket in the background. |

### 2.4 Tooling & Testing

| Technology | Purpose |
|---|---|
| **pytest / pytest-asyncio** | Automated test suite (29 tests) covering the state machine, authentication, encryption, notification dispatch, and the database writer. |
| **Git** | Version control. |
| **.env / pydantic-settings** | Environment-based configuration so secrets and tunables never live in source code. |
| **Self-signed TLS certificate generation (`utils/certs.py`)** | Enables HTTPS on the local network without a paid certificate — required because mobile browsers refuse camera access (`getUserMedia`) on non-secure origins. |

---

## 3. System Architecture

### 3.1 High-Level Data Flow

```
 Browser (phone/laptop)                         Server (Python / FastAPI, GPU)
┌─────────────────────────┐                    ┌──────────────────────────────────────┐
│  Camera (getUserMedia)  │  binary JPEG frame  │  WebSocket endpoint (/ws/stream)      │
│  Capture loop (~18fps)  │ ───────────────────▶│  Latest-frame buffer (thread-safe)    │
│  Canvas overlay drawing │                      │            │                          │
│  Voice / siren playback │◀────────────────────│  Dedicated inference worker thread    │
│  Live dashboard UI      │   JSON results       │            ▼                          │
└─────────────────────────┘                      │  SecurityPipeline (state machine)     │
                                                   │   ├─ YOLO detection engine            │
                                                   │   ├─ ByteTrack multi-person tracking  │
                                                   │   ├─ MediaPipe liveness / face score  │
                                                   │   ├─ FaceNet verification              │
                                                   │   └─ Rolling-window confirmation       │
                                                   │            │                          │
                                                   │            ▼                          │
                                                   │  Alerts (email/Telegram) · Evidence    │
                                                   │  capture · SQLite database (async     │
                                                   │  writer thread)                        │
                                                   └──────────────────────────────────────┘
```

### 3.2 Why This Architecture

- **All AI inference runs server-side, not in the browser.** This keeps the client extremely lightweight (any phone browser works) and lets the heavy models run on a real GPU rather than being constrained to whatever a phone's browser-based ML runtime could manage. The trade-off is that the browser must have a live network connection to the server at all times — an accepted constraint for a single-booth, single-camera deployment.
- **One dedicated worker thread per camera connection** decouples slow, synchronous AI inference from FastAPI's async event loop, so a slow inference frame can never stall the server's ability to keep accepting new video frames or serve other requests.
- **One `SecurityPipeline` (and its in-progress session state) is reused per logged-in user across reconnects**, not recreated per WebSocket connection. This was a deliberate fix during development: a brief WiFi hiccup or a phone screen-lock forcing a reconnect used to silently discard an in-progress face-covering countdown or verification attempt and restart it from zero, with no indication to the user that anything had been lost.
- **Database writes happen on a dedicated background thread**, via a queue, so a slow disk write can never block the real-time detection loop.

### 3.3 Component Responsibilities

| Layer | Responsibility |
|---|---|
| `api/` | HTTP/WebSocket routing, authentication, request validation. |
| `workflow/pipeline.py` | The core state machine — owns per-person session state and decides what happens next. |
| `models/` | Runs the three YOLO models and merges their output into a single detection list. |
| `tracking/` | Assigns and maintains stable per-person track IDs frame-to-frame. |
| `liveness/` | Face-presence scoring, blink/head-movement liveness, and a color/texture heuristic for generic face coverings YOLO has no trained class for. |
| `verification/` | Turns a face crop into an embedding and compares it against known/reference embeddings. |
| `alerts/` | Dispatches email/Telegram notifications, captures evidence snapshots and short video clips, tracks one-shot audio-alert flags. |
| `database/` | ORM models and a background-thread-safe write queue. |
| `frontend/` | The operator-facing dashboard: camera capture, live overlay, audio playback. |

---

## 4. Key Features

1. **Real-time person, weapon, and PPE detection** using three purpose-specific YOLO models running in parallel per frame.
2. **Multi-person tracking** with stable identities via ByteTrack, so the system reasons per-person rather than per-frame.
3. **Face-covering detection and escalation workflow**: grace period → voice warning + on-screen prompt → audible siren — not an instant alarm, giving a legitimate user a fair chance to comply first.
4. **Weapon detection with immediate lockdown**, which always takes priority over every other state, since it represents the highest-severity threat.
5. **Active liveness detection** (blink + head-turn within a time window) to reject a printed photo or a static image held up to the camera.
6. **Face verification** via cosine-similarity matching of FaceNet embeddings, either against an admin-enrolled allowlist or a self-consistency reference captured within the same session.
7. **Anti-false-positive rolling-window confirmation** on every detection category — no alert is ever raised off a single noisy frame.
8. **Evidence capture**: automatic snapshot images and short pre/post-event video clips saved whenever an alert fires, giving a human reviewer context.
9. **Multi-channel alerting** (email, Telegram) with graceful "not configured" fallback rather than silently failing.
10. **Fixed ATM identity and automatic emergency police notification**: the console is configured for a specific real-world location (Canara Bank ATM, Madanapalle), and the two highest-severity events — weapon lockdown and a refused face-covering siren — automatically send an emergency email to the ATM's configured nearest police station, distinct from the general administrator alert.
11. **Role-based authentication** (`admin` / `operator`) via JWT, with an admin-only face-enrollment endpoint and an admin/operator-gated lockdown-clear action.
12. **Encrypted-at-rest biometric data**: face embeddings are Fernet-encrypted in the database; raw face images are never stored at all.
13. **Live operator dashboard**: real-time bounding boxes, a per-person requirements checklist (person detected / face visible / liveness / verified), system health (including the fixed ATM ID and location), live event log, and recent alerts — all updating over the WebSocket without a page refresh.
14. **Mobile-first, no app install required**: runs in any modern phone browser over HTTPS on the local network.

---

## 5. Detection & Security Workflow

This is the heart of the project: a finite state machine, one instance per tracked person, that decides what the system should do moment-to-moment.

### 5.1 State Machine

| State | Meaning |
|---|---|
| `MONITORING` | A person is present; system is watching, no issue detected yet. |
| `FACE_COVERED_GRACE` | A face covering was just confirmed; a short countdown begins, with an immediate voice + on-screen warning. |
| `FACE_COVERED_WARNING` | Grace period elapsed and the face is still covered; siren is imminent. |
| `SIREN` | Audible alarm active; exits the instant the face becomes visible again, not only after a fixed duration. |
| `VERIFYING` | Face is visible; running face-detection confidence, liveness, and identity-verification checks. |
| `TRANSACTION_ACTIVE` | All checks passed — "You are verified" plays. Times out back to `MONITORING` on its own after a fixed window, so the system is immediately ready for a new person without requiring a manual reset. |
| `LOCKDOWN` | A weapon was confirmed. Highest-priority state; overrides everything else and requires an explicit admin action to clear. |

State transitions, evidence capture, database writes, and voice/siren triggers are all driven from this single place (`workflow/pipeline.py`), which keeps the security logic auditable in one location rather than scattered across the codebase.

### 5.2 Anti-False-Positive Design

A key engineering principle throughout the project: **no alert fires off a single frame.** Every raw detection (a helmet, a mask, a weapon, a "face visible" reading) is passed through a `ConsecutiveConfirmer` — a rolling window that requires a *majority* of recent frames to agree (e.g. 12 of the last 15) before treating something as confirmed, with a faster "release" rule (as few as 2 consecutive clear frames) for clearing an already-confirmed state. This is deliberately asymmetric: raising an alert should be slow and certain; clearing one should be fast, since there's no false-positive risk in believing a threat is gone sooner rather than later.

Two different confirmation windows are used for two different risk profiles:
- **Weapon detection and identity verification** use a conservative, slower window — the cost of a false positive here (a false lockdown, or a false "verified") is high.
- **Face-covering detection (helmet/mask)** uses a faster, shorter window — worst case, a legitimate user is asked to show their face once too eagerly, which is a low-stakes nuisance, not a security failure. This distinction was arrived at iteratively, after real-device testing showed the shared conservative window made the covering-detection workflow feel sluggish.

### 5.3 Face-Covering Detection

Three independent signals are combined:
1. The PPE YOLO model's `helmet` / `mask` class output, scoped to the top half of the person's bounding box (so a mask pulled down to the chin, or held in a hand, doesn't keep reading as "worn").
2. A generic color/texture heuristic (`liveness/liveness.py`) that catches coverings with no trained class at all — hoods, bandanas, a hand over the face — by comparing the lower-face region against a trusted bare-skin reference sampled from the same frame (forehead and both cheeks). This is explicitly labelled a heuristic, not a certified detector, and is deliberately conservative to avoid manufacturing false alarms.
3. MediaPipe's own face-presence score, which naturally reads low if no face is visible at all for any reason.

An important design correction made during development: the system must use the *confirmed* (smoothed) helmet/mask signal to decide whether the face is "covered," not the *raw* single-frame detection. Using the raw signal directly meant one stray false-positive PPE reading (more likely after tuning detection thresholds for better recall) could keep the system stuck in a "covered" state long after a mask was actually removed.

### 5.4 Weapon Detection

Weapon detection always takes priority over every other state — if confirmed, the session moves immediately to `LOCKDOWN`, triggers a "Danger detected" voice alert, starts evidence video/snapshot capture, and dispatches an alert notification. Recovery requires an explicit admin "clear lockdown" action; the system will never auto-clear a weapon lockdown on its own.

### 5.5 Emergency Police Notification

The system is configured for a fixed, real-world ATM location (**Canara Bank ATM, Madanapalle** — set via `ATM_LOCATION` in `.env`, and shown live on the operator dashboard header and System Health card, not just embedded invisibly in outbound alerts). Whenever a weapon is confirmed (`LOCKDOWN`) or the face-covering siren fires (`SIREN`), the system automatically dispatches an emergency email to the ATM's configured nearest police station, in addition to the existing administrator email/Telegram alert — reusing the same real SMTP mechanism, just a distinct recipient and a subject/body clearly marked as an unreviewed automated emergency alert requiring verification.

This required no new trigger logic: both events already funnel through the same single alert-dispatch call in `workflow/pipeline.py`, and those two calls are the *only* two places in the codebase that invoke it — so registering the police channel alongside the existing email/Telegram channels automatically and exactly covers both severity-appropriate triggers, with no risk of it firing for a lower-stakes event by mistake.

Consistent with the project's honesty principle (Section 10): the police station's name, email, and phone number are **not** hardcoded or guessed — they are blank by default and must be filled in with the real, verified contact for the ATM's actual nearest station. There is no automated phone/SMS dispatch to emergency services; the configured phone number is informational only, for a human operator to dial manually, matching the project's existing stance that SMS/WhatsApp/push are unimplemented pluggable extension points, not simulated capabilities.

### 5.6 Liveness & Verification

Liveness requires observing **both** an eye blink (a measurable dip in eye-aspect-ratio) **and** a head-yaw movement within a rolling time window, using MediaPipe's 468-point Face Mesh. This defeats a static printed photo (which can't blink or produce natural parallax) without needing a depth camera, which no phone browser has access to.

Once liveness passes, a FaceNet embedding is computed from the live face crop and compared (cosine similarity) against either an admin-enrolled reference face or — if no enrollment system is in use — the *first* embedding captured for that session, which becomes a self-consistency reference for the rest of that visit. This scope decision is explained explicitly in the codebase rather than silently redefining "verified" to mean something the system doesn't actually do (there is no real card/account-linked identity check, since this is a security-layer prototype, not a banking integration).

---

## 6. Database Design

A relational schema (SQLite via SQLAlchemy) with five tables:

| Table | Purpose |
|---|---|
| **users** | Operator/admin login credentials (bcrypt-hashed passwords) and role. |
| **enrolled_faces** | The admin-managed face allowlist — a label plus an *encrypted* FaceNet embedding. No raw images are ever stored. |
| **atm_sessions** | One row per tracked-person visit: start/end time, verification status, and outcome (completed / abandoned / security lockdown). |
| **detection_events** | Every individual detection worth recording (weapon, face-covering, siren, transaction events), linked to a session, with confidence, evidence file paths, and whether it came from a real model or a best-effort heuristic. |
| **alert_log** | A record of every outbound notification attempt across all three channels — `email`, `telegram`, and `police` — (channel, status, detail), including ones that were skipped because credentials/contact details weren't configured, so the audit trail stays honest about what actually happened. |

Sensitive columns (face embeddings) are stored as Fernet-encrypted bytes, so possession of the database file alone is not enough to recover usable biometric data without the separate encryption key.

---

## 7. API Design

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/login` | Authenticate and issue a JWT access token. |
| GET | `/api/health` | Server status, inference device, active sessions/connections, uptime. |
| GET | `/api/events` | Recent detection events. |
| GET | `/api/alerts` | Recent alert-dispatch log. |
| POST | `/api/atm/unlock` | Admin/operator action to clear a lockdown. |
| GET | `/api/atm/status` | Live status of every active camera connection. |
| POST | `/api/enroll` | Admin-only: enroll a new known face into the allowlist. |
| WebSocket `/ws/stream` | The live video pipeline — binary JPEG frames in, JSON detection/state results out. |

Every REST endpoint (other than login) requires a valid JWT; the enroll and unlock endpoints are further gated by role.

---

## 8. Security Implementation

- **Authentication**: JWT-based, with role claims (`admin` / `operator`) checked per endpoint.
- **Password storage**: bcrypt hashing via passlib — passwords are never stored or logged in plain text.
- **Biometric data**: face embeddings are Fernet (AES-based symmetric encryption) encrypted at rest; raw face images are never written to disk or database at all, only derived numeric embeddings.
- **Transport security**: the entire application runs over HTTPS (self-signed certificate for LAN use), which is also a hard requirement for browsers to allow camera access at all.
- **Configuration hygiene**: all secrets (JWT signing key, SMTP/Telegram credentials, encryption key) are loaded from environment variables via a `.env` file, never hardcoded, with a documented insecure default that must be rotated before any real deployment.

---

## 9. Testing & Validation

The project includes an automated pytest suite (29 tests) covering:

- The rolling-window anti-false-positive confirmation logic in isolation.
- JWT issuance/validation, password hashing, and field-level encryption.
- The background database writer thread.
- JWT-gated REST endpoint access control.
- The full state-machine workflow — face-covering escalation through grace/warning/siren, weapon detection through lockdown, and (critically) the complete verification path from face-visible through liveness through FaceNet embedding comparison to a verified transaction — run against a stub detection engine but real MediaPipe/FaceNet models, so the actual AI logic is exercised, not just mocked out.
- Notification dispatch — the admin email, Telegram, and police-alert channels are each verified independently: correctly skipped when unconfigured, correctly sent (with `smtplib.SMTP` mocked out so the suite runs offline) when configured, and confirmed independent of each other so configuring the police contact doesn't accidentally also start sending the general admin alert.

Because the development machine has no physical webcam, real end-to-end validation (actual camera, actual lighting, actual human face) was performed on a real phone browser over the LAN, and several threshold/timing values in the system were iteratively recalibrated based on that real-device testing rather than left at their initial theoretical values — a good example of the gap between "runs in a unit test" and "works for a real user," worth mentioning directly in an interview.

---

## 10. Honest Scope: What's Real vs. Best-Effort

A deliberate project principle, carried from the README into this report: every capability is labelled honestly rather than oversold.

| Capability | Status |
|---|---|
| Person / knife / baseball bat detection | **Real** — YOLO11n, COCO-pretrained. |
| Gun / knife / explosive / grenade detection | **Real** — a fine-tuned YOLOv8n threat-detection model. |
| Helmet / mask detection | **Real** — a fine-tuned YOLOv8n PPE model. |
| Generic covering (iron rod / blunt object / cloth) | **Best-effort heuristic** — no trained model exists for these; confidence is deliberately capped so a heuristic alone can never trigger a lockdown by itself. |
| Face detection | **Real** — MediaPipe. |
| Liveness (blink + head movement) | **Real, active liveness** — but not robust against a *prepared video replay* attack, since no phone browser has depth-camera access. |
| Face verification | **Real embeddings and matching**, scoped honestly as liveness + self-consistency (or an admin allowlist), not a KYC-grade identity check against a national database. |
| Encrypted storage | **Field-level** (embeddings only), not full-disk database encryption. |
| Email / Telegram alerts | **Real integrations**, inert (and clearly logged as such) until credentials are configured. |
| Police/emergency notification | **Real integration** — same SMTP mechanism as the admin email alert, fired only for weapon lockdown and face-covering siren. Deliberately left blank until filled in with the ATM's *actual, verified* nearest police station contact, not a guessed value. |
| SMS / WhatsApp / push notifications | **Not implemented** — no paid API credentials were available; the code has a clear extension point. The police-alert phone number is informational only (for a human to dial) — there is no automated voice-call/SMS dispatch to emergency services. |

This "real vs. heuristic" transparency is itself a demonstrable engineering skill: knowing the difference between a trained model and a fallback heuristic, and refusing to blur that line even under pressure to claim a fuller feature set, is exactly the kind of judgment interviewers look for.

---

## 11. Limitations & Future Scope

**Current limitations:**
- No real banking/card-system integration — verification is a security layer, not a transaction authorization system.
- Liveness defends against printed photos but not against a sophisticated prepared video replay.
- No automated browser-level end-to-end tests (camera/WebSocket reconnect logic is validated manually on a real device, not in CI).
- SQLite is appropriate for a single-booth prototype but would need to move to PostgreSQL for a multi-branch deployment.
- No trained model exists yet for some threat classes (e.g. an iron rod), which currently fall back to a capped-confidence heuristic.

**Natural next steps:**
- Train or fine-tune a model for the remaining untrained threat classes.
- Add a lightweight anti-spoofing (video-replay-resistant) liveness signal.
- Move to PostgreSQL and add multi-ATM/multi-branch aggregation to the dashboard.
- Add SMS/WhatsApp notification channels.
- Add automated browser-level (Playwright) end-to-end tests.

---

## 12. Conclusion

Omni Mind demonstrates a complete, real-time, full-stack AI security pipeline — from raw camera bytes over a WebSocket, through multi-model object detection, person tracking, liveness analysis, and face verification, to a coordinated state machine driving voice/visual alerts, evidence capture, and multi-channel notifications — running entirely on consumer hardware. Beyond the AI components themselves, the project required solving real systems-engineering problems: keeping a real-time inference pipeline responsive under a persistent WebSocket connection, tuning anti-false-positive behavior against real-world noise, correctly reasoning about state across unreliable mobile network conditions, and being explicit and honest about the boundary between a trained model and a fallback heuristic. Those are the same skills — not just "I used YOLO" — that this report, and the project itself, are built to make easy to explain in an interview.

---

## Appendix A — Quick Interview Reference

**"Walk me through what happens when someone approaches the ATM."**
The browser streams JPEG frames over a WebSocket. The server decodes each frame, runs three YOLO models plus a person tracker, and a state machine (one per tracked person) decides what state that person is in — monitoring, face-covering escalation, verifying, or lockdown. Results (bounding boxes, state, checklist) stream back over the same socket and render live on the dashboard.

**"How do you avoid false alarms?"**
Every detection goes through a rolling-window confirmer before it's trusted — a majority of the last N frames must agree, not just one. Weapon/verification use a slower, more conservative window; face-covering uses a faster one, since the cost of being wrong is much lower there.

**"Why not just trust the first frame that sees a weapon?"**
A single frame can be wrong — motion blur, an object briefly resembling a weapon, a misclassification. A false lockdown has a real cost (a legitimate user is locked out and security is falsely alerted), so the system requires sustained evidence before acting, while still keeping that window short enough to be genuinely real-time.

**"How does face verification actually work if there's no ID card?"**
It proves two things: that a live human (not a photo) is present, via blink+head-turn liveness, and that the same physical person is still present throughout the session, via face-embedding similarity — either against an admin-enrolled allowlist, or the person's own first embedding as a self-consistency reference if no allowlist is in use. It's explicitly not a national-ID-grade identity check, and the project says so rather than implying otherwise.

**"How does the police-alert feature work, and why is the contact info blank by default?"**
It reuses the same real SMTP mechanism as the admin email alert — just a second, distinct recipient — and only fires for the two highest-severity events (weapon lockdown, face-covering siren), since those are the only two places in the codebase that already trigger an alert dispatch at all. The police station's actual name/email/phone are left blank by default rather than filled in with a guessed value, because inventing real-world emergency-contact details for a security system — even a prototype — would be actively harmful if trusted. It has to be the deployer's own verified information, exactly like the admin SMTP credentials already work.

**"What was the hardest bug to find?"**
Several state-transition bugs only showed up on real hardware, not in unit tests — for example, a session could get permanently stuck in one state if there was no way for it to naturally return to monitoring, which looked like "the system only works once" until traced back to a missing timeout. Root-causing it required reasoning about the interaction between a persistent per-user pipeline, network reconnects, and the confirmation windows — not something a single stack trace pointed to directly.

**"What would you do differently with more time?"**
Add a trained model for the remaining heuristic-only threat class, add anti-spoofing liveness, and build real automated browser-level end-to-end tests instead of relying on manual real-device testing.

---

*Report generated for academic submission and interview preparation. Project: Omni Mind — AI ATM Security System.*
