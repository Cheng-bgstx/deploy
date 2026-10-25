"""The ``planner`` motion backend: Planner Chunks as the Tracker's Reference.

``PlannerReference.tick(state, now)`` runs one control cycle of the Planner's 50 Hz
side in Scalecar's order (poll, warmup, Go if requested, sample) and returns the
Reference as sim2real ``MotionData``: joints and pelvis only, no FK.

- Hold (before Go, and after Go until the first Chunk is applied): the Motion's frame 0,
  frozen, zero velocity, exactly what File Playback shows while paused.
- Running: the 33 rows resampled at ``now`` (``ChunkBuffer.rows_at``), decoded like
  Scalecar's ``tracker_tokens`` (abs38 -> ECHO 38D -> joints, pelvis position and
  orientation; joint velocity by ``np.gradient`` at 50 Hz), taken at the Tracker's
  future steps. Past the end of the Chunk the last row is held (coverage exhausted).
- Invalid tokens: the last valid Reference is kept.

Stale depth and camera disconnects are reported on the terminal (``depth_status``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

from sim2real.rl_policy.utils.motion import MotionData

from .depth_status import DepthStatus
from .provider import JOINT_NAMES, PlannerProvider, PlannerState
from .scalecar_dp import motion38d as m38
from .scalecar_dp.dp import contract as C

BODY_NAMES = ("pelvis",)
STATS_PERIOD_S = 5.0


@dataclass(frozen=True)
class PlannerIO:
    """Where the Planner's inputs and time come from.

    On the robot: the Host depth receiver, ``worker.GpuPlannerWorker`` and
    ``time.monotonic``. In the vehicle sim2sim: ``sim.SimulatedDepthSource``,
    ``sim.SimulatedTimeWorker`` around ``GpuPlannerWorker``, and simulation time.
    ``source.poll()`` returns the newest depth packet or None; ``worker`` has
    ``submit``/``poll``/``close``; ``clock()`` is the 50 Hz tick time in seconds.
    """

    source: Any
    worker: Any
    clock: Callable[[], float]


def _motion_data(joint_pos, joint_vel, pelvis_pos, pelvis_quat_wxyz) -> MotionData:
    steps = len(joint_pos)
    return MotionData(
        joint_pos=np.asarray(joint_pos, dtype=np.float64)[None],
        joint_vel=np.asarray(joint_vel, dtype=np.float64)[None],
        body_pos_w=np.asarray(pelvis_pos, dtype=np.float64)[None, :, None],
        body_quat_w=np.asarray(pelvis_quat_wxyz, dtype=np.float64)[None, :, None],
        body_lin_vel_w=np.zeros((1, steps, 1, 3)),
        body_ang_vel_w=np.zeros((1, steps, 1, 3)),
    )


def hold_motion_data(*, joint_pos, pelvis_pos, pelvis_quat_wxyz, future_steps) -> MotionData:
    """The frozen Hold Reference: one frame (joints in ``JOINT_NAMES`` order) at every future step."""
    steps = len(future_steps)
    return _motion_data(
        np.repeat(np.asarray(joint_pos, dtype=np.float64)[None], steps, 0),
        np.zeros((steps, len(JOINT_NAMES))),
        np.repeat(np.asarray(pelvis_pos, dtype=np.float64)[None], steps, 0),
        np.repeat(np.asarray(pelvis_quat_wxyz, dtype=np.float64)[None], steps, 0),
    )


def motion_data_from_rows(rows: np.ndarray) -> MotionData:
    """Resampled abs38 rows (row 0 = now) -> the Reference at the Tracker offsets.

    The same decoding as ``contract.tracker_tokens`` -> ``motion38d.tracker_tokens``,
    stopping before ``reference_only_features`` (the Tracker computes its tokens)."""
    q, pos, quat = m38.decode(C.to_echo38(rows))
    dq = m38.joint_velocity(q)
    idx = list(m38.TRACKER_OFFSETS)
    return _motion_data(q[idx], dq[idx], pos[idx], quat[idx])


def check_future_steps(future_steps) -> None:
    """The Tracker's ``motion.future_steps`` must be the offsets the Planner's Reference is decoded at."""
    if [int(s) for s in future_steps] != list(m38.TRACKER_OFFSETS):
        raise ValueError(
            f"motion.future_steps {list(future_steps)} must be the Planner's Tracker offsets "
            f"{list(m38.TRACKER_OFFSETS)}"
        )


class PlannerReference:
    """The Planner as a Reference Source for ``tracking.py`` (``motion_backend: planner``)."""

    joint_names = list(JOINT_NAMES)
    body_names = list(BODY_NAMES)

    def __init__(self, provider: PlannerProvider, *, hold: MotionData, future_steps) -> None:
        check_future_steps(future_steps)
        self.provider = provider
        self.hold = hold
        self.motion_data: MotionData = hold
        self.phase = "standing"
        self.go_requested = False
        self.go_time: float | None = None
        self.first_reference_time: float | None = None
        self.invalid_token_ticks = 0
        self._last_valid: MotionData | None = None
        self._next_stats: float | None = None
        self._counted_from = self._provider_counts()
        self.depth_status = DepthStatus()

    # -- operator hooks -------------------------------------------------------

    def request_go(self) -> None:
        """Go (B): attempted at the next ``tick`` with that tick's state and time."""
        self.go_requested = True

    def abort(self, reason: str = "E-stop") -> None:
        """E-stop: the current Go is dropped and the Reference returns to Hold."""
        self.go_requested = False
        self.provider.abort()
        self.go_time = self.first_reference_time = self._next_stats = None
        self._last_valid = None
        self._counted_from, self.invalid_token_ticks = self._provider_counts(), 0
        self.motion_data, self.phase = self.hold, "standing"
        logger.warning(
            "Planner: Go dropped ({}); the Reference is back in Hold at the Motion's frame 0. "
            "Restart with A -> R1 -> B", reason,
        )

    @property
    def go_active(self) -> bool:
        """A Go is requested or accepted (planning, or Hold until its first Chunk)."""
        return self.go_requested or self.go_time is not None

    @property
    def holding(self) -> bool:
        """The Reference is frozen at the Motion's frame 0 (Hold)."""
        return self.motion_data is self.hold

    # -- control cycle --------------------------------------------------------

    def tick(self, state: PlannerState, now: float) -> MotionData:
        provider = self.provider
        provider.poll(state, now)
        self._report_depth(now)
        provider.warmup(state, now)
        if self.go_requested:
            self.go_requested = False
            self._go(state, now)
        decision = provider.sample(now)
        self.phase = decision.phase
        if decision.tokens is None:
            self.motion_data = self.hold
        elif decision.phase == "holding_valid_tokens":
            self.invalid_token_ticks += 1
            self.motion_data = self._last_valid
        else:
            # The provider resamples fresh rows every tick, so they are decoded every tick
            # (about 0.13 ms; the provider decodes the same rows once more for its tokens).
            self._last_valid = motion_data_from_rows(decision.rows)
            self.motion_data = self._last_valid
            if self.first_reference_time is None:
                self.first_reference_time = now
                logger.info("Planner: first Chunk applied {:.3f} s after Go", now - self.go_time)
        self._log_stats(now)
        return self.motion_data

    def _go(self, state: PlannerState, now: float) -> None:
        counts = self._provider_counts()  # before start(), which requests the first plan
        if self.provider.start(state, now):
            self.go_time = now
            self._counted_from, self.invalid_token_ticks = counts, 0
            init = self.provider.initialization
            logger.info(
                "Planner: Go accepted. Pelvis height {:.3f} m (feet {:.3f} / {:.3f} m); Hold until the first Chunk",
                init["height_m"], init["left_m"], init["right_m"],
            )
        else:
            logger.warning("Planner: Go refused, staying in Hold: {}", self.provider.start_rejection)

    def _report_depth(self, now: float) -> None:
        provider = self.provider
        age = None if not provider.depth.t else now - provider.depth.t[-1]
        for level, message in self.depth_status.update(
            now, age_s=age, error=provider.depth_error, session=provider.camera_session
        ):
            logger.log(level, message)

    # -- statistics -----------------------------------------------------------

    def stats(self) -> dict:
        """Planning latency, late Chunks, coverage exhaustion and the other counts of the current Go.

        Every count starts over when a Go is accepted and when ``abort`` drops it (so it is
        zero until the next Go). ``warmup_latency_ms`` is the latest warmup plan's.
        """
        provider = self.provider
        plans = [p for p in provider.plans if not p["warmup"] and p["identity"][0] == provider.generation]
        latency_ms = [1000.0 * p["latency_s"] for p in plans]
        inference_ms = [p["inference_ms"] for p in plans if p["inference_ms"] is not None]
        warmups = [p for p in provider.plans if p["warmup"]]
        counts = {key: value - self._counted_from[key] for key, value in self._provider_counts().items()}
        return {
            "plans_requested": counts["plans_requested"],
            "plans_received": len(plans),
            "chunks_applied": sum(p["accepted_at"] is not None for p in plans),
            "latency_ms": _percentiles(latency_ms),
            "inference_ms": _percentiles(inference_ms),
            "prefix_frames": _percentiles([p["prefix"] for p in plans]),
            "late_chunks": provider.buffer.late_switches,
            "coverage_exhausted_events": counts["coverage_exhausted_events"],
            "coverage_exhausted_ticks": provider.buffer.stale,
            "invalid_token_ticks": self.invalid_token_ticks,
            "warmup_latency_ms": 1000.0 * warmups[-1]["latency_s"] if warmups else None,
            "go_time_s": self.go_time,
            "first_chunk_delay_s": None
            if self.first_reference_time is None
            else self.first_reference_time - self.go_time,
            "go_initialization": provider.initialization,
            "depth_delay_events": counts["depth_delay_events"],
            "camera_recovery_events": counts["camera_recovery_events"],
        }

    def _provider_counts(self) -> dict:
        """The provider's running totals (Scalecar's, never reset); ``stats`` reports them per Go."""
        provider = self.provider
        return {
            "plans_requested": provider.plans_requested,
            "coverage_exhausted_events": provider.reference_hold_events,
            "depth_delay_events": provider.depth_delay_events,
            "camera_recovery_events": provider.recovery_events,
        }

    def _log_stats(self, now: float) -> None:
        if self.go_time is None:
            return
        if self._next_stats is None:
            self._next_stats = self.go_time + STATS_PERIOD_S
        if now + 1e-9 < self._next_stats:
            return
        self._next_stats += STATS_PERIOD_S * (math.floor((now - self._next_stats) / STATS_PERIOD_S) + 1)
        stats, depth_age = self.stats(), self.provider.readiness(now)["depth_age_s"]
        latency = stats["latency_ms"]
        logger.info(
            "Planner: {} plans, latency p50/p95 {}, {} late Chunks, coverage exhausted {}x ({} ticks), "
            "depth age {}, phase {}",
            stats["plans_received"],
            "-" if latency is None else f"{latency['p50']:.1f}/{latency['p95']:.1f} ms",
            stats["late_chunks"],
            stats["coverage_exhausted_events"],
            stats["coverage_exhausted_ticks"],
            "-" if depth_age is None else f"{depth_age:.3f} s",
            self.phase,
        )

    def close(self) -> None:
        self.provider.close()


def _percentiles(values) -> dict | None:
    if not len(values):
        return None
    p50, p95, p99 = np.percentile(values, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99), "max": float(np.max(values)), "n": len(values)}
