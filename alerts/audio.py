"""Audio-alert event flags.

The server has no guaranteed speaker output (and definitely no access to a
phone's speaker), so voice warnings and the siren are triggered as one-shot
*event flags* sent to the browser over the WebSocket connection; the
dashboard JS (frontend/js/app.js) plays them through real pre-rendered
<audio> elements (static .wav files under frontend/audio/), not the Web
Speech API or a Web Audio oscillator — iOS Safari's hardware mute switch
silences speechSynthesis/Web Audio but specifically exempts <audio>/<video>
element playback, so a phone left on silent would otherwise get a fully
"unlocked" dashboard that plays back total silence. This module owns the
"only fire once per event" bookkeeping so a flaky network or a repeated
server-side check can never cause the browser to replay the siren mid-blast.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AudioEventFlags:
    voice_warning_fired: bool = False
    siren_fired: bool = False
    siren_started_at: float | None = None

    def reset(self) -> None:
        self.voice_warning_fired = False
        self.siren_fired = False
        self.siren_started_at = None


VOICE_WARNING_TEXT = "Please remove your face-covering objects."
HELMET_REMOVE_TEXT = "Please remove your helmet for identity verification."
DANGER_TEXT = "Danger detected. Security has been informed."
VERIFIED_TEXT = "You are verified."
UNCOVERED_TEXT = "Thank you for removing the face-covering object."
