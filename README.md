# Omni Mind — AI ATM Security System

Real-time ATM security prototype: browser camera → FastAPI/WebSocket →
YOLO11n + MediaPipe + FaceNet pipeline on the GPU → live dashboard.

## Run it

```powershell
cd C:\Users\Deekshit\omni-mind-project
.\.venv\Scripts\python.exe run_server.py
```

This prints two URLs — a `127.0.0.1` one for this PC and a LAN one for your
phone (same WiFi required). Both are **HTTPS with a self-signed
certificate**: your browser will warn "connection is not private" the first
time — tap **Advanced → Proceed**. This isn't optional cosmetics: mobile
browsers refuse camera access (`getUserMedia`) on any non-secure origin, and
a plain LAN IP over HTTP does not count as secure. See `utils/certs.py`.

Default logins: `admin` / `admin123`, `operator` / `operator123` — change
these before any real deployment (`database/models.py:User`).

## What's real vs. best-effort

This spec asks for capabilities that don't all have an equally strong
foundation to build on in one session. Rather than paper over the gaps, here
is exactly what's real:

| Capability | Status |
|---|---|
| Person detection | Real — YOLO11n, COCO-pretrained |
| Knife / baseball bat | Real — YOLO11n COCO classes |
| Gun / knife / explosive / grenade | Real — `Subh775/Threat-Detection-YOLOv8n` (MIT, HuggingFace), fine-tuned YOLOv8n |
| Helmet / mask | Real — `Tanishjain9/yolov8n-ppe-detection-6classes` (MIT, HuggingFace) |
| Iron rod / generic blunt object | **Heuristic only** — no open pretrained model exists for this class. `models/blunt_object_heuristic.py` does classic-CV line/edge detection, confidence deliberately capped below the weapon-alarm threshold so it can never itself trigger a lockdown. Swap in a real trained model here if you get labeled data. |
| Face detection | Real — MediaPipe short-range face detector |
| Liveness (blink + head movement) | Real, but **active liveness only** — no depth camera exists on a phone browser, so this is not passive/depth-based anti-spoofing. It genuinely rejects a static photo (no blink) and is a meaningfully higher bar than nothing, but a sufficiently prepared video replay attack is a research problem beyond this scope. |
| Face verification | Real embeddings (FaceNet/VGGFace2, `facenet-pytorch`), cosine similarity. **Scope note:** there's no card/account system to verify identity against. If you enroll faces via `/api/enroll` (admin only), it matches against that allowlist; otherwise it falls back to *self-consistency* — confirming the same live person who passed liveness is still the one present, not KYC-grade identity verification. |
| Multi-person tracking | Real — ByteTrack (`supervision`) |
| Encrypted storage | Field-level (Fernet/AES) on embeddings and evidence paths, not full-disk SQLCipher — see `utils/security.py` docstring for why (no reliable Windows wheel without a C toolchain). Plain face images are never stored, only embeddings. |
| Email / Telegram alerts | Real integrations, inert until you fill in `.env` — logs `skipped_not_configured` rather than pretending to send |
| Police/emergency notification | Real — same SMTP mechanism as the admin email alert, addressed to `POLICE_EMAIL` instead. Fires only for the two highest-severity events (weapon lockdown, face-covering siren). Inert until you fill in the **real, verified** contact for this ATM's actual nearest police station — see `.env.example`. No automated phone/SMS dispatch to emergency services exists; `POLICE_PHONE` is informational only, for a human operator to dial manually. |
| SMS / WhatsApp / push | Not implemented — no credentials for a paid API were available. `alerts/notifier.py` has the pattern to extend (e.g. wire in Twilio). |
| JWT auth / role-based access | Real |
| 15-frame anti-false-positive confirmation | Real, rolling-window interpretation (12-of-15 hits, not a zero-tolerance streak) — see `workflow/confirmer.py` docstring |
| >99% face-verification confidence threshold | Recalibrated — a literal 0.99 cosine-similarity cutoff would reject real users (genuine FaceNet matches score ~0.6-0.9). `settings.face_match_cosine_threshold` (0.55) is the real operating point; overall confidence comes from stacking detection+liveness+verification+15-frame confirmation, not one inflated number. |

## Architecture

```
frontend/ (browser)  --WebSocket(binary JPEG)-->  api/routes_ws.py
                                                        |
                                          camera/frame_buffer.py (latest-frame slot)
                                                        |
                                     dedicated worker thread per connection
                                                        |
                                          workflow/pipeline.py (state machine)
                                       /        |         |          \
                          models/yolo_engine  tracking  liveness  verification
                                       \        |         |          /
                                     alerts/{audio,notifier,evidence}
                                                        |
                                        database/db.py (background writer thread)
```

Folders match the spec: `camera/ models/ tracking/ verification/ liveness/
alerts/ database/ api/ frontend/ utils/ config/ logs/ tests/`, plus
`workflow/` for the state machine (not enumerated in the original folder
list, but there's nowhere else it cleanly belongs).

## Setup gotchas (if recreating the venv)

- `pip install ultralytics`/`facenet-pytorch` will silently swap your CUDA
  `torch` for a CPU-only wheel (PyPI's default Windows `torch` has no CUDA).
  Always `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
  **last**, after everything else.
- `mediapipe>=0.10.18` dropped the legacy `mediapipe.solutions` API this
  project uses for FaceMesh/FaceDetection — pin `mediapipe==0.10.14`. Install
  it with `--no-deps` (letting pip free-resolve its full tree, including
  jax/jaxlib, can hang for many minutes) and then separately
  `pip install "protobuf<5,>=3.20" absl-py flatbuffers sounddevice attrs`.
- `passlib`'s bundled bcrypt self-test breaks under `bcrypt>=4.1` (that
  version raises instead of silently truncating on passlib's >72-byte test
  string) — pin `bcrypt==4.0.1`.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

Unit tests cover: anti-false-positive confirmation logic, JWT/password/field
encryption, background DB writer thread, JWT-gated REST endpoints, and the
face-covering→siren and weapon→lockdown state-machine transitions (against a
stub detection engine, real MediaPipe/FaceNet). Browser-side behavior
(camera retry, WebSocket reconnect) is implemented in `frontend/js/app.js`
but not covered by automated E2E tests — there's no Node/Playwright in this
environment, matching the constraint noted in the Smart Era project. Test on
a real device and report back if thresholds need tuning for your camera/lighting.
