"use strict";

// ---------------------------------------------------------------- state ---
const TARGET_CAPTURE_WIDTH = 640;
// Must send faster than the server's target_fps (config/settings.py,
// currently 15) so the server is always the one throttling/deduping, not
// the client's own capture cadence. This was 1000/12 (~12fps) — *below*
// target_fps despite the comment's stated intent — measured backend
// per-frame cost is only ~25-30ms (well under the ~66ms/frame budget at
// 15fps), so the client, not the server, was the bottleneck on how soon a
// newly-visible person's frame actually got sent for detection.
const CAPTURE_INTERVAL_MS = 1000 / 18;
const WS_RECONNECT_DELAY_MS = 2000;
const HEALTH_POLL_MS = 5000;
const WATCHDOG_CHECK_MS = 3000;
const WATCHDOG_STALE_MS = 8000; // no server frame reply in this long -> stream is stuck (backgrounded tab, OS-suspended camera, dead socket that never fired onclose)

let authToken = sessionStorage.getItem("omni_token") || null;
let authRole = sessionStorage.getItem("omni_role") || null;
let ws = null;
let wsReconnectTimer = null;
let mediaStream = null;
let captureTimer = null;
let latestBoxes = [];
let currentConnectionId = null;
let audioUnlocked = false;
let lastServerMsgAt = 0;
let watchdogTimer = null;
let recovering = false;

const el = (id) => document.getElementById(id);

// ------------------------------------------------------------- audio unlock --
// Voice warnings and the siren play through real <audio> elements (see
// index.html) rather than the Web Speech API or a raw Web Audio oscillator.
// That's not a style choice: iOS Safari mutes speechSynthesis and Web Audio
// output whenever the phone's hardware mute/silent switch is on, but
// specifically exempts <audio>/<video> element playback from that switch —
// so the old implementation could be fully "unlocked" per this code's own
// bookkeeping and still produce total silence on a real phone. It also
// doesn't depend on a TTS voice being installed on the device, which is
// inconsistent across Android browsers/WebViews.
//
// Browsers only allow an <audio> element's play() to succeed unprompted
// (i.e. later, from a WebSocket event handler with no user gesture on the
// stack) once that *same element* has already played during a real
// gesture. So this does one muted play+pause per element right inside the
// click/tap handler to "activate" it, and starts the siren element looping
// at volume 0 forever after — flipping its volume later needs no further
// gesture since it never stops playing.
// Unlocking the siren element and the voice element used to be one atomic
// Promise.all — if either element's activation rejected (for any transient
// reason: the wav not fully buffered yet, a momentary browser quirk), the
// *whole* unlock failed and neither was marked unlocked, even if the other
// had genuinely succeeded. That's exactly what an intermittent "works
// sometimes, not other times" symptom looks like. Each element now tracks
// its own unlocked state independently, so a tap that only manages to
// unlock one of them still keeps that progress instead of discarding it,
// and the next tap only retries whichever one is still missing.
let sirenUnlocked = false;
let voiceUnlocked = false;

function updateAudioHint() {
  audioUnlocked = sirenUnlocked && voiceUnlocked;
  el("audio-hint").style.display = audioUnlocked ? "none" : "inline";
  // Only demand the tap once the dashboard is actually up — showing it over
  // the login screen would be pointless friction since submitting login is
  // itself a real gesture that unlocks audio a moment later.
  const dashboardActive = el("dashboard").classList.contains("active");
  el("audio-unlock-overlay").classList.toggle("show", !audioUnlocked && dashboardActive);
  return audioUnlocked;
}

function unlockAudio() {
  const siren = el("siren-audio");
  const voice = el("voice-audio");

  if (!sirenUnlocked) {
    const wasMuted = siren.muted;
    siren.muted = true;
    siren.play()
      .then(() => {
        siren.pause();
        siren.currentTime = 0;
        siren.muted = false;
        siren.volume = 0; // keep looping silently so a later siren_start needs no further gesture
        return siren.play();
      })
      .then(() => { sirenUnlocked = true; updateAudioHint(); })
      .catch((err) => {
        siren.muted = wasMuted;
        console.warn("Siren unlock failed — will retry on next tap", err);
      });
  }

  if (!voiceUnlocked) {
    const wasMuted = voice.muted;
    voice.muted = true;
    voice.play()
      .then(() => {
        voice.pause();
        voice.currentTime = 0;
        voice.muted = wasMuted;
        voiceUnlocked = true;
        updateAudioHint();
      })
      .catch((err) => {
        voice.muted = wasMuted;
        console.warn("Voice unlock failed — will retry on next tap", err);
      });
  }
}

// A tap anywhere — login screen or dashboard, at any point in the page's
// life — retries the unlock. Cheap/idempotent once already running, and
// this is also what recovers audio after a background/screen-lock cycle.
document.addEventListener("click", unlockAudio);
document.addEventListener("touchstart", unlockAudio);
el("audio-unlock-btn").addEventListener("click", unlockAudio);

// ------------------------------------------------------------------ auth --
el("login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const btn = el("login-btn");
  btn.disabled = true;
  el("login-error").textContent = "";

  unlockAudio(); // this click is a real user gesture

  try {
    const resp = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username: el("login-username").value.trim(),
        password: el("login-password").value,
      }),
    });
    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || "Login failed");
    }
    const data = await resp.json();
    authToken = data.access_token;
    authRole = data.role;
    sessionStorage.setItem("omni_token", authToken);
    sessionStorage.setItem("omni_role", authRole);
    startDashboard();
  } catch (err) {
    el("login-error").textContent = err.message || "Login failed";
  } finally {
    btn.disabled = false;
  }
});

el("logout-btn").addEventListener("click", () => signOut(""));

function signOut(message) {
  stopEverything();
  sessionStorage.removeItem("omni_token");
  sessionStorage.removeItem("omni_role");
  authToken = null;
  el("dashboard").classList.remove("active");
  el("login-screen").style.display = "flex";
  el("login-error").textContent = message || "";
}

function authHeaders() {
  return { Authorization: `Bearer ${authToken}` };
}

// Every authenticated request goes through this so an expired/invalid token
// reliably kicks the user back to login instead of failing silently forever
// (this was the actual root cause behind "nothing updates" reports — the
// dashboard stayed on screen with stale data while every request 401'd).
async function authFetch(url, options = {}) {
  const resp = await fetch(url, { ...options, headers: { ...(options.headers || {}), ...authHeaders() } });
  if (resp.status === 401 || resp.status === 403) {
    signOut("Session expired — please sign in again.");
    throw new Error("session expired");
  }
  return resp;
}

// --------------------------------------------------------------- startup --
function startDashboard() {
  el("login-screen").style.display = "none";
  el("dashboard").classList.add("active");
  el("atm-meta").textContent = `role: ${authRole}`;
  updateAudioHint();
  startCamera();
  connectWebSocket();
  pollHealthAndAlerts();
  setInterval(pollHealthAndAlerts, HEALTH_POLL_MS);
  clearInterval(watchdogTimer);
  watchdogTimer = setInterval(checkStreamHealth, WATCHDOG_CHECK_MS);
  requestWakeLock();
}

// A phone's screen dimming/locking mid-session is the single most common
// real-world cause of the capture loop and WebSocket going quiet for
// several seconds at a stretch (mobile Safari throttles timers and camera
// delivery in a backgrounded/dimmed tab) — which then trips the stream
// watchdog above and forces a reconnect, over and over, for as long as the
// screen keeps dimming. Holding a screen wake lock while the dashboard is
// open prevents that entire cycle instead of just recovering faster from
// it. Support is best-effort (iOS Safari 16.4+, most modern Android
// browsers); older browsers silently get no wake lock and fall back to
// the watchdog-recovery behavior that already existed.
let wakeLock = null;

async function requestWakeLock() {
  if (!("wakeLock" in navigator) || document.hidden) return;
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    wakeLock.addEventListener("release", () => { wakeLock = null; });
  } catch (err) {
    console.warn("Screen wake lock unavailable", err);
  }
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && authToken && !wakeLock) requestWakeLock();
});

// Mobile browsers routinely suspend getUserMedia streams and throttle
// timers when the tab is backgrounded or the screen locks, and don't
// always resume cleanly on their own — that was the actual cause behind
// "works once, then needs a page refresh" reports. Rather than chase every
// individual OS/browser suspension quirk, this watches for the one
// symptom they all share: the server stops replying to frames. If no
// server message arrives for WATCHDOG_STALE_MS while logged in, silently
// tear down and reacquire the camera + socket — the same recovery a
// manual refresh was doing, without making the user do it.
function checkStreamHealth() {
  if (!authToken || !lastServerMsgAt) return;
  if (Date.now() - lastServerMsgAt > WATCHDOG_STALE_MS) {
    console.warn("Stream watchdog: no live frames for a while — recovering camera/connection.");
    recoverStream();
  }
}

function recoverStream() {
  if (recovering || !authToken) return;
  recovering = true;
  lastServerMsgAt = Date.now(); // give the fresh stream a full window before re-checking
  clearInterval(captureTimer);
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  clearTimeout(wsReconnectTimer);
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
  startCamera();
  connectWebSocket();
  setTimeout(() => { recovering = false; }, WATCHDOG_STALE_MS);
}

// A backgrounded tab coming back to the foreground is the single most
// common trigger for the stall above — check immediately on return
// instead of waiting for the next watchdog tick.
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    checkStreamHealth();
    if (!audioUnlocked || el("siren-audio").paused) unlockAudio();
  }
});

if (authToken) startDashboard();

// ---------------------------------------------------------------- camera --
async function startCamera() {
  setCaption("Requesting camera access…");
  // Browsers treat camera/mic permission and audio-autoplay permission as
  // two entirely separate systems — there's no single API to request both
  // at once, and granting camera access via the browser's native dialog
  // doesn't count as the "user gesture" the audio-autoplay policy checks
  // for (that dialog isn't part of the page, so it dispatches no DOM event
  // here). This attempts the unlock at the same moment camera access is
  // requested so the two happen together whenever possible instead of
  // requiring a separate later tap.
  unlockAudio();
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 } },
      audio: false,
    });
    const video = el("camera-video");
    video.srcObject = mediaStream;
    await video.play();
    setCaption("Camera live — analyzing…");
    beginCaptureLoop();
  } catch (err) {
    console.error("Camera error", err);
    setCaption(
      "Camera unavailable (" + err.name + "). Retrying in 4s… " +
      "If this persists, ensure you accepted the HTTPS certificate warning and granted camera permission."
    );
    setTimeout(startCamera, 4000);
  }
}

function beginCaptureLoop() {
  clearInterval(captureTimer);
  const video = el("camera-video");
  const capture = el("capture-canvas");
  const overlay = el("overlay");
  const ctx = capture.getContext("2d", { willReadFrequently: false });
  const octx = overlay.getContext("2d");

  captureTimer = setInterval(() => {
    if (!video.videoWidth) return;
    const scale = TARGET_CAPTURE_WIDTH / video.videoWidth;
    const w = TARGET_CAPTURE_WIDTH;
    const h = Math.round(video.videoHeight * scale);
    if (capture.width !== w || capture.height !== h) {
      capture.width = w; capture.height = h;
      overlay.width = w; overlay.height = h;
    }
    ctx.drawImage(video, 0, 0, w, h);
    octx.drawImage(capture, 0, 0);
    drawBoxes(octx, w, h);

    if (ws && ws.readyState === WebSocket.OPEN) {
      capture.toBlob((blob) => {
        if (blob && ws && ws.readyState === WebSocket.OPEN) ws.send(blob);
      }, "image/jpeg", 0.7);
    }
  }, CAPTURE_INTERVAL_MS);
}

function drawBoxes(ctx, w, h) {
  const colors = {
    person: "#34d399", helmet: "#f5a623", mask: "#f5a623", face_covered: "#f5a623",
    weapon_gun: "#ef4444", weapon_knife: "#ef4444", weapon_baseball_bat: "#ef4444",
    weapon_explosive: "#ef4444", weapon_grenade: "#ef4444", possible_blunt_object: "#8b5cf6",
  };
  for (const box of latestBoxes) {
    const [x1, y1, x2, y2] = box.bbox;
    const color = colors[box.category] || "#6b7680";
    // Person boxes and confirmed alarms are solid; everything else (a
    // detection that cleared the display floor but hasn't been confirmed by
    // the rolling-window check yet) is drawn dashed and dimmer so it reads
    // as "the system is watching this" rather than "alarm."
    const isConfirmed = box.category === "person" || box.confirmed;
    ctx.globalAlpha = isConfirmed ? 1.0 : 0.55;
    ctx.strokeStyle = color;
    ctx.lineWidth = isConfirmed ? 2 : 1.25;
    ctx.setLineDash(isConfirmed ? [] : [5, 4]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const pct = (box.confidence * 100).toFixed(0);
    const idPart = box.track_id !== undefined && box.track_id !== null ? " #" + box.track_id : "";
    const label = `${isConfirmed ? "" : "~ "}${box.category}${idPart} ${pct}%`;
    ctx.font = "11px 'Cascadia Code', 'SF Mono', monospace";
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - 16), tw + 8, 16);
    ctx.fillStyle = "#07080a";
    ctx.fillText(label, x1 + 4, Math.max(11, y1 - 4));
    ctx.globalAlpha = 1.0;
  }
  ctx.setLineDash([]);
}

function setCaption(text) {
  el("stage-caption").textContent = text;
}

// ------------------------------------------------------------- websocket --
function connectWebSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/stream?token=${encodeURIComponent(authToken)}`);

  ws.onopen = () => {
    el("conn-dot").classList.add("up");
    el("conn-text").textContent = "connected";
    lastServerMsgAt = Date.now(); // baseline so the watchdog gives this fresh connection a fair window
  };
  ws.onclose = () => {
    el("conn-dot").classList.remove("up");
    el("conn-text").textContent = "reconnecting…";
    clearTimeout(wsReconnectTimer);
    if (authToken) wsReconnectTimer = setTimeout(connectWebSocket, WS_RECONNECT_DELAY_MS);
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    handleServerMessage(msg);
  };
}

function handleServerMessage(msg) {
  lastServerMsgAt = Date.now();
  currentConnectionId = msg.connection_id;
  latestBoxes = msg.boxes || [];
  updateStatusPill(msg.system_status);
  updateSessions(msg.sessions || []);
  updateCountdownBanner(msg.sessions || []);
  for (const line of msg.log_events || []) appendLog(el("log-list"), line);
  reactToSessionTransitions(msg.sessions || []);
}

// Voice/siren playback is driven by watching sessions[].state change, not by
// the server's one-shot audio_events list. audio_events rides the same
// bounded, drop-oldest queue as everything else (api/routes_ws.py) — if that
// particular frame's message got evicted before it reached the client (a
// brief slow patch, a reconnect), the one-shot event was gone forever with
// no trace, since the server never resends it. sessions[].state is resent on
// *every* message, so even after dropping several frames in a row, the next
// message that does arrive still shows the current state — comparing it
// against the last state seen per track_id catches the transition just as
// correctly, without depending on any single message getting through.
const sessionAudioState = new Map(); // track_id -> { state, wasWarned }

function reactToSessionTransitions(sessions) {
  if (!audioUnlocked) unlockAudio(); // best-effort; only really works inside a user gesture
  const seenTrackIds = new Set();
  for (const s of sessions) {
    seenTrackIds.add(s.track_id);
    let memo = sessionAudioState.get(s.track_id);
    if (!memo) {
      memo = { state: null, wasWarned: false };
      sessionAudioState.set(s.track_id, memo);
    }
    if (s.state !== memo.state) {
      handleStateTransition(memo.state, s.state, memo, s);
      memo.state = s.state;
    }
  }
  for (const trackId of Array.from(sessionAudioState.keys())) {
    if (!seenTrackIds.has(trackId)) sessionAudioState.delete(trackId);
  }
}

function handleStateTransition(prevState, newState, memo, session) {
  const wasAlarm = prevState === "siren" || prevState === "lockdown";
  const isAlarm = newState === "siren" || newState === "lockdown";

  // The voice warning now fires as soon as a covering is detected (grace
  // entry), matching the backend — not after the silent grace countdown
  // finishes (face_covered_warning), which used to make it look like the
  // siren was the very first reaction with no warning beforehand.
  if (newState === "face_covered_grace") {
    memo.wasWarned = true;
    playVoice(session && session.cover_reason === "helmet" ? "helmet_warning" : "voice_warning");
  } else if (newState === "lockdown") {
    playVoice("danger_voice");
  } else if (newState === "transaction_active") {
    playVoice("verified_voice");
  }

  if (isAlarm && !wasAlarm) startSiren();
  if (wasAlarm && !isAlarm) stopSiren();

  const compliedAfterWarning = memo.wasWarned && newState === "verifying"
    && (prevState === "face_covered_grace" || prevState === "face_covered_warning" || prevState === "siren");
  if (compliedAfterWarning) {
    playVoice("uncovered_voice");
    memo.wasWarned = false;
  }
}

function updateStatusPill(status) {
  const pill = el("status-pill");
  pill.textContent = status;
  pill.className = "pill " + status.toLowerCase();
}

function updateSessions(sessions) {
  const list = el("session-list");
  if (!sessions.length) {
    const emptyHtml = '<span class="kv"><span>No one present.</span></span>';
    if (list.dataset.lastHtml !== emptyHtml) {
      list.innerHTML = emptyHtml;
      list.dataset.lastHtml = emptyHtml;
    }
    el("rec-badge").style.display = "none";
    return;
  }

  const recording = sessions.some((s) => ["siren", "lockdown", "face_covered_warning"].includes(s.state));
  el("rec-badge").style.display = recording ? "flex" : "none";

  // Server messages arrive 15-18x/second, but most fields here (state,
  // checklist) only actually change a few times per session lifetime — a
  // full innerHTML teardown/rebuild on every single message was destroying
  // and recreating every row (including the "Clear lockdown" button) that
  // often, which is real, visible DOM churn on a phone, not just wasted
  // work. Building the target HTML first and only touching the DOM when it
  // actually differs from what's already rendered keeps steady states
  // (nothing changing) untouched while still updating every tick during an
  // active countdown (grace/warning/siren), since the seconds value makes
  // the string differ each time.
  const html = sessions.map((s) => {
    const countdown = s.countdown_seconds !== null && s.countdown_seconds !== undefined
      ? `<span class="countdown-val">${Math.ceil(s.countdown_seconds)}s</span>` : "";

    const c = s.checklist || {};
    const rows = [
      ["Person detected", c.person_detected, ""],
      ["Face visible", c.face_visible, `${Math.round((c.face_score || 0) * 100)}%`],
      ["Liveness · blink", c.liveness_blink, ""],
      ["Liveness · head move", c.liveness_head_move, ""],
      ["Verified", c.verified, ""],
    ];
    const checklistHtml = rows.map(([label, ok, val]) => `
      <div class="check-row ${ok ? "ok" : ""}">
        <span class="box"></span><span>${label}</span>
        ${val ? `<span class="val">${val}</span>` : ""}
      </div>`).join("");

    const unlockBtn = s.state === "lockdown"
      ? `<button class="small" onclick="adminUnlock(${s.track_id})">Clear lockdown (admin)</button>` : "";

    return `<div class="session-row">
      <div class="headline">
        <span class="id">TRACK #${s.track_id}</span>
        <span class="state-tag state-${s.state}">${s.state.replace(/_/g, " ")}</span>
        ${countdown}
      </div>
      <div class="checklist">${checklistHtml}</div>
      ${unlockBtn}
    </div>`;
  }).join("");

  if (html !== list.dataset.lastHtml) {
    list.innerHTML = html;
    list.dataset.lastHtml = html;
  }
}

function updateCountdownBanner(sessions) {
  const banner = el("countdown-banner");
  const priority = ["lockdown", "siren", "face_covered_warning", "face_covered_grace"];
  let target = null;
  for (const state of priority) {
    target = sessions.find((s) => s.state === state);
    if (target) break;
  }
  if (!target) { banner.style.display = "none"; banner.className = ""; return; }

  const secs = target.countdown_seconds !== null && target.countdown_seconds !== undefined
    ? Math.ceil(target.countdown_seconds) : null;
  const isHelmet = target.cover_reason === "helmet";
  const texts = {
    face_covered_grace: isHelmet
      ? `Please remove your helmet for identity verification${secs !== null ? " — " + secs + "s" : ""}`
      : `Please show your face for identity verification${secs !== null ? " — " + secs + "s" : ""}`,
    face_covered_warning: isHelmet
      ? `Helmet not removed — siren in ${secs}s`
      : `Face covering not removed — siren in ${secs}s`,
    siren: `ALARM ACTIVE — face covering not removed`,
    lockdown: `DANGER DETECTED — SECURITY HAS BEEN INFORMED`,
  };
  banner.textContent = texts[target.state] || target.state;
  banner.className = target.state === "lockdown" || target.state === "siren" ? "danger" : "warn";
}

async function adminUnlock(trackId) {
  if (!currentConnectionId) return;
  try {
    await authFetch("/api/atm/unlock", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ connection_id: currentConnectionId, track_id: trackId }),
    });
  } catch (err) {
    console.error("Unlock failed", err);
  }
}

// ------------------------------------------------------------------ audio --
const VOICE_SRC = {
  voice_warning: "/audio/voice_warning.wav",
  helmet_warning: "/audio/helmet_warning.wav",
  danger_voice: "/audio/danger_voice.wav",
  verified_voice: "/audio/verified.wav",
  uncovered_voice: "/audio/uncovered.wav",
};

function playVoice(type) {
  const voice = el("voice-audio");
  const src = VOICE_SRC[type];
  if (!src) return;
  if (!voice.src.endsWith(src)) voice.src = src;
  voice.currentTime = 0;
  voice.play().catch((err) => console.warn("Voice playback blocked", err));
}

function startSiren() {
  const siren = el("siren-audio");
  siren.volume = 1.0;
  if (siren.paused) siren.play().catch((err) => console.warn("Siren playback blocked", err));
}

function stopSiren() {
  el("siren-audio").volume = 0.0; // keep looping silently, ready for the next activation with no gesture needed
}

// -------------------------------------------------------------- log utils --
function appendLog(listEl, text) {
  const li = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  li.textContent = `[${time}] ${text}`;
  const lower = text.toLowerCase();
  if (lower.includes("weapon") || lower.includes("danger") || lower.includes("siren")) li.className = "crit";
  else if (lower.includes("covering") || lower.includes("suspicious")) li.className = "warn";
  listEl.prepend(li);
  while (listEl.children.length > 100) listEl.removeChild(listEl.lastChild);
}

// ----------------------------------------------------------- health/alerts --
async function pollHealthAndAlerts() {
  if (!authToken) return;
  try {
    const resp = await authFetch("/api/health");
    if (resp.ok) {
      const h = await resp.json();
      el("h-atmid").textContent = h.atm_id;
      el("h-location").textContent = h.atm_location;
      el("h-device").textContent = h.device;
      el("h-sessions").textContent = h.active_sessions;
      el("h-conns").textContent = h.active_camera_connections;
      el("h-uptime").textContent = formatUptime(h.uptime_seconds);
      el("atm-meta").textContent = `${h.atm_location} · role: ${authRole}`;
    }
  } catch (err) { console.warn("health poll failed", err); return; }

  try {
    const resp = await authFetch("/api/alerts?limit=15");
    if (resp.ok) {
      const alerts = await resp.json();
      const list = el("alert-list");
      list.innerHTML = "";
      for (const a of alerts) {
        const li = document.createElement("li");
        li.textContent = `${a.channel}: ${a.status}${a.detail ? " — " + a.detail : ""}`;
        list.appendChild(li);
      }
    }
  } catch (err) { console.warn("alerts poll failed", err); }
}

function formatUptime(seconds) {
  const m = Math.floor(seconds / 60);
  const h = Math.floor(m / 60);
  return h > 0 ? `${h}h ${m % 60}m` : `${m}m ${Math.floor(seconds % 60)}s`;
}

// -------------------------------------------------------------- teardown --
function stopEverything() {
  clearInterval(captureTimer);
  clearInterval(watchdogTimer);
  clearTimeout(wsReconnectTimer);
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  if (wakeLock) { wakeLock.release().catch(() => {}); wakeLock = null; }
  stopSiren();
}

window.addEventListener("beforeunload", stopEverything);
