"""Rolling-window consecutive-frame confirmation — the anti-false-positive
gate required by spec: "never trigger an alarm using a single frame."

Interpretation note: a literal zero-tolerance "15 consecutive frames" streak
counter is fragile in practice because YOLO confidence flickers frame to
frame even for genuine objects (motion blur, brief occlusion). This uses a
15-frame rolling window and requires `min_hits` (default 12/15) within it —
still rejects single-frame spikes, but tolerates the occasional dropped
frame a real detection naturally has. Tune via config.settings if the
literal all-15 behavior is preferred for a given deployment.

Confirm vs release are deliberately asymmetric. Rising edge (going from
"clear" to "confirmed") keeps the full slow rolling-window gate above — that's
the actual anti-false-positive requirement, and it should stay conservative.
Falling edge (going from "confirmed" back to "clear") does not need the same
caution: once something is confirmed, a short run of real negative reads is
strong enough evidence that it's gone, and there's no false-positive risk in
clearing an alarm state faster. Without this, a confirmed detection only
decays as its old `True` frames age out of the rolling window — for a status
that was fully saturated (all 15 hits), that's still 3-4 frames of lag before
the sum drops below min_hits, but it reads to a user as "stuck" state,
especially compounded across multiple categories. `release_after_misses`
frames of consecutive misses now clear it near-instantly instead.
"""
from __future__ import annotations

from collections import defaultdict, deque

from config.settings import settings


class ConsecutiveConfirmer:
    def __init__(
        self,
        window: int = settings.confirm_window_frames,
        min_hits: int = settings.confirm_min_hits,
        release_after_misses: int = settings.confirm_release_after_misses,
    ) -> None:
        self.window = window
        self.min_hits = min_hits
        self.release_after_misses = release_after_misses
        self._history: dict[tuple, deque[bool]] = defaultdict(lambda: deque(maxlen=window))
        self._confirmed: dict[tuple, bool] = defaultdict(bool)
        self._miss_streak: dict[tuple, int] = defaultdict(int)

    def update(self, key: tuple, hit: bool) -> bool:
        dq = self._history[key]
        dq.append(hit)
        self._miss_streak[key] = 0 if hit else self._miss_streak[key] + 1

        if self._confirmed[key]:
            if self._miss_streak[key] >= self.release_after_misses:
                self._confirmed[key] = False
                dq.clear()  # require a fresh full window before re-confirming
        elif len(dq) == self.window and sum(dq) >= self.min_hits:
            self._confirmed[key] = True

        return self._confirmed[key]

    def forget_prefix(self, prefix) -> None:
        for key in [k for k in self._history if k[0] == prefix]:
            del self._history[key]
            self._confirmed.pop(key, None)
            self._miss_streak.pop(key, None)

    def reset(self, key: tuple) -> None:
        self._history.pop(key, None)
        self._confirmed.pop(key, None)
        self._miss_streak.pop(key, None)
