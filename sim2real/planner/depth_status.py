"""Terminal messages about the Planner's depth input: stale frames, camera disconnects.

The Planner itself is unaffected (as in Scalecar it keeps planning from the last frame
with that frame's real age); these messages only tell the operator. Each condition is
reported when it starts, repeated at most once per second while it lasts, and reported
once when it ends:

- stale: the newest frame is more than 0.2 s old (the Go gate's limit);
- disconnected: the depth receiver reports an error (stream closed, capture program
  failed, clock sync refused); it covers a stale frame, and ends with the first frame
  after the receiver reconnects. A new camera session without a reported error is a
  reconnect as well.
"""

from __future__ import annotations

STALE_AFTER_S = 0.2
REPEAT_S = 1.0


class DepthStatus:
    """Edge-triggered depth messages; ``update`` returns ``(level, message)`` pairs to log."""

    def __init__(self) -> None:
        self.state = "waiting"  # waiting (no frame yet), fresh, stale, disconnected
        self.since: float | None = None
        self.next_repeat: float | None = None
        self.session = None

    def update(self, now: float, *, age_s: float | None, error: str | None, session) -> list[tuple[str, str]]:
        """``age_s``: age of the newest frame (None before the first); ``error``: the receiver's
        current error (None once a new frame arrived); ``session``: the camera session."""
        out: list[tuple[str, str]] = []
        new_session = self.session is not None and session != self.session
        if session is not None:
            self.session = session
        if error is not None:
            if self.state != "disconnected":
                last = "no frame yet" if age_s is None else f"last frame {age_s:.3f} s old"
                out.append(("WARNING", f"Depth camera disconnected ({last}); the receiver keeps reconnecting: {error}"))
                self._enter("disconnected", now)
            elif now + 1e-9 >= self.next_repeat:
                out.append(("WARNING", f"Depth camera still disconnected ({now - self.since:.1f} s): {error}"))
                self.next_repeat += REPEAT_S
            return out
        if age_s is None:
            return out
        stale = age_s > STALE_AFTER_S
        if self.state == "disconnected" or new_session:
            after = f" after {now - self.since:.1f} s" if self.state == "disconnected" else ""
            out.append(("INFO", f"Depth camera reconnected{after} (new session, depth history cleared); "
                                f"frame age {age_s:.3f} s"))
            self._enter("fresh", now)
        elif self.state == "waiting":
            out.append(("INFO", f"Depth: first frame received ({age_s:.3f} s old)"))
            self._enter("fresh", now)
        if stale and self.state != "stale":
            out.append(("WARNING", f"Depth stale: last frame is {age_s:.3f} s old (limit {STALE_AFTER_S} s); "
                                   "the Planner keeps using it with its real age"))
            self._enter("stale", now)
        elif stale and now + 1e-9 >= self.next_repeat:
            out.append(("WARNING", f"Depth still stale: last frame is {age_s:.3f} s old"))
            self.next_repeat += REPEAT_S
        elif not stale and self.state == "stale":
            out.append(("INFO", f"Depth recovered after {now - self.since:.1f} s stale: frame age {age_s:.3f} s"))
            self._enter("fresh", now)
        return out

    def _enter(self, state: str, now: float) -> None:
        self.state, self.since, self.next_repeat = state, now, now + REPEAT_S
