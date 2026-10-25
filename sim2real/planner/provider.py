"""The Planner's 50 Hz side, ported from Scalecar_v2 ``hardware/dp.py`` (SOURCES.json)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .height import standing_height
from .scalecar_dp import motion38d as m38
from .scalecar_dp.dp import contract as C
from .scalecar_dp.dp import runtime

# The Planner's joint order: Scalecar's ``scalecar_dp.spec.JOINT_NAMES``, which is also
# the Tracker's (deployment.json ``joint_names``, "Isaac order").
JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint", "left_wrist_roll_joint",
    "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)


@dataclass(frozen=True)
class PlannerState:
    """The measured robot state the Planner side needs each tick."""

    joint_pos: np.ndarray  # (29,) in JOINT_NAMES order
    root_quat_wxyz: np.ndarray  # (4,) IMU orientation


@dataclass(frozen=True)
class ReferenceDecision:
    tokens: np.ndarray | None
    phase: str
    fault: str | None
    chunk_id: int | None
    terminal_hold: bool = False
    rows: np.ndarray | None = None  # the (33, 38) abs38 rows the tokens were made from


class PlannerProvider:
    """All control-side Planner state, owned by the 50 Hz loop; inference and capture are children.

    Ported from Scalecar_v2 ``hardware/dp.py`` (``HardwareDpProvider``). The per-tick
    protocol is Scalecar's hardware runtime (``hardware/runtime.py:316-381``):
    ``poll`` and ``warmup`` every tick, ``start`` on Go, then ``sample``. Differences:

    - the depth source and the Planner worker are passed in (real receiver or simulation);
    - the joint order is checked against ``JOINT_NAMES`` instead of ``scalecar_dp.spec``;
    - ``sample`` also returns the resampled rows (``ReferenceDecision.rows``);
    - ``start`` records why it refused a Go (``start_rejection``);
    - ``abort`` (E-stop) drops the current Go;
    - every Planner result is kept in ``plans`` for latency statistics (no TraceWriter).
    """

    def __init__(self, joint_names, *, source, worker):
        if list(joint_names) != list(JOINT_NAMES):
            raise ValueError("Planner/Tracker joint order mismatch")
        self.joint_names = list(joint_names)
        self.depth, self.states = runtime.DepthBuffer(), runtime.StateBuffer()
        self.buffer = runtime.ChunkBuffer()
        self.source = source
        self.worker = worker
        self.generation = 0
        self.request_id = 0
        self.result_id = None
        self.go = None
        self.next_plan = None
        self.active = False
        self.warmed = False
        self.warm_submitted = False
        self.fault = None
        self.failed_start = None
        self.camera_session = None
        self.chunk_id = None
        self.last_tokens = None
        self.last_rows = None
        self.last_robot = None
        self.applied_chunk = None
        self.initialization = None
        self.plans_requested = 0
        self.depth_error = None
        self.worker_error = None
        self.last_depth_frame = None
        self.initial_yaw = None
        self.depth_delayed = False
        self.depth_delay_events = 0
        self.reference_hold_events = 0
        self.reference_held = False
        self.recovery_events = 0
        self.start_rejection = None
        self.plans: deque[dict] = deque(maxlen=18000)  # one hour at 5 Hz

    def poll(self, state, now: float) -> None:
        """Ingest latest depth/result and keep the last actually applied reference history."""
        packet = self.source.poll()
        if packet is not None:
            if "error" in packet:
                self.depth_error = packet["error"]
            elif (packet["session"], packet["frame"]) != self.last_depth_frame:
                if self.camera_session is not None and packet["session"] != self.camera_session:
                    self.recovery_events += 1
                    self.depth = runtime.DepthBuffer()
                self.camera_session = packet["session"]
                if not self.depth.t or packet["acquired"] > self.depth.t[-1]:
                    self.depth.push(packet["depth_mm"], packet["acquired"], True)
                    self.last_depth_frame = (packet["session"], packet["frame"])
                    self.depth_error = None
        item = self.worker.poll()
        if item is not None and item.get("status") == "error":
            self.worker_error = item["error"]
        if item is not None and item.get("status") == "ready":
            identity = (item["generation"], item["request_id"])
            self._record_plan(identity, item)
            if (
                identity != self.result_id
                and item["generation"] == self.generation
                and (item["warmup"] or item["chunk"].t_ready <= now)
            ):
                self.result_id = identity
                self.worker_error = None
                if item["warmup"]:
                    self.warmed = True
                elif self.go is not None:
                    chunk = item["chunk"]
                    if (
                        chunk.t_ready <= now
                        and chunk.rows.shape == (128, 38)
                        and np.isfinite(chunk.rows).all()
                        and (self.buffer.current is None or chunk.t0 > self.buffer.current.t0)
                    ):
                        self.buffer.set(chunk)
                        self.chunk_id = item["request_id"]
                        self.fault = None
                        self.plans[-1]["accepted_at"] = now
        self.last_robot = state
        if self.go is not None and self.failed_start is None:
            robot = self._robot_row(state, self.initialization["height_m"])
            applied = runtime.ChunkBuffer(current=self.applied_chunk)
            values = runtime.history_state("self", applied, now, robot)
            if not self.states.t or now > self.states.t[-1]:
                self.states.push(now, *values)
            delayed = not self.depth.t or not 0 <= now - self.depth.t[-1] <= 0.2
            self.depth_delay_events += delayed and not self.depth_delayed
            self.depth_delayed = delayed
            # The frozen observer includes the actual age of every depth input.
            # Keep planning from the last valid frame; never invent a fresh stamp.
            if self.depth.t and now + 1e-8 >= self.next_plan:
                self._submit(now, warmup=False)
                self.next_plan += (int(max(0, now - self.next_plan) / 0.2) + 1) * 0.2

    def _record_plan(self, identity, item) -> None:
        # The worker keeps returning its latest result, and results arrive in order.
        if self.plans and self.plans[-1]["identity"] == identity:
            return
        chunk = item["chunk"]
        self.plans.append(
            dict(
                identity=identity,
                warmup=bool(item["warmup"]),
                t0=float(chunk.t0),
                t_ready=float(chunk.t_ready),
                latency_s=float(chunk.t_ready - chunk.t0),
                inference_ms=item.get("inference_ms"),
                prefix=int(chunk.prefix),
                accepted_at=None,
            )
        )

    def _robot_row(self, state, height: float):
        yaw = (
            float(m38.heading(state.root_quat_wxyz))
            if self.initial_yaw is None
            else self.initial_yaw
        )
        quat = m38.quat_mul(m38.yaw_quat(-yaw), state.root_quat_wxyz)
        gravity = m38.quat_to_matrix(quat).T @ np.array([0.0, 0.0, -1.0])
        return state.joint_pos, np.array([0.0, 0.0, height]), quat, gravity

    def _submit(self, now: float, warmup: bool) -> None:
        observer = runtime.Observer(self.depth, self.states)
        observer.start(self.go if self.go is not None else now)
        obs = observer.observe(now)
        times = [self.depth.t[i] for i in obs.depth_idx]
        self.request_id += 1
        if self.worker.submit(
            dict(
                generation=self.generation,
                request_id=self.request_id,
                observation=obs,
                depth_times=times,
                warmup=warmup,
            )
        ):
            self.plans_requested += not warmup

    def warmup(self, state, now: float) -> None:
        if self.warmed or (self.warm_submitted and not self.worker_error) or not self.depth.t:
            return
        # Warmup is never used for movement or accepted as a measured initialization.
        self.states = runtime.StateBuffer()
        self.states.push(now, *self._robot_row(state, 0.7842))
        self._submit(now, warmup=True)
        self.warm_submitted = True

    def readiness(self, now: float) -> dict:
        age = None if not self.depth.t else now - self.depth.t[-1]
        return dict(
            ready=bool(
                self.warmed
                and age is not None
                and 0 <= age <= 0.2
                and self.depth_error is None
                and self.worker_error is None
            ),
            warmed=self.warmed,
            depth_age_s=age,
            depth_frames=len(self.depth.t),
            depth_error=self.depth_error,
            worker_error=self.worker_error,
            failed_start=self.failed_start,
            fault=self.fault,
            depth_delayed=self.depth_delayed,
            depth_delay_events=self.depth_delay_events,
            reference_held=self.reference_held,
            reference_hold_events=self.reference_hold_events,
            recovery_events=self.recovery_events,
        )

    def start(self, state, now: float) -> bool:
        """A fresh Go discards all warmup generations; first valid chunk activates motion."""
        readiness = self.readiness(now)
        if self.active or self.fault or (self.go is not None and self.failed_start is None) or not readiness["ready"]:
            self.start_rejection = _rejection(self, readiness)
            return False
        try:
            initialization = standing_height(state.joint_pos, self.joint_names, state.root_quat_wxyz)
        except ValueError as error:
            self.failed_start = str(error)
            self.start_rejection = f"pelvis height check failed: {error}"
            return False
        self.generation += 1
        self.initial_yaw = float(m38.heading(state.root_quat_wxyz))
        self.go = now
        self.next_plan = now + 0.2
        self.failed_start = None
        self.start_rejection = None
        self.result_id = None
        self.initialization = initialization
        self.buffer = runtime.ChunkBuffer()
        self.states = runtime.StateBuffer()
        self.states.push(now, *self._robot_row(state, initialization["height_m"]))
        self._submit(now, warmup=False)
        return True

    def abort(self) -> None:
        """E-stop: drop the current Go. Results still in flight belong to a dead generation,
        and only a new Go (after the Tracker takes over again) restarts planning."""
        self.generation += 1
        self.go = None
        self.next_plan = None
        self.active = False
        self.fault = None
        self.failed_start = None
        self.start_rejection = None
        self.result_id = None
        self.chunk_id = None
        self.last_tokens = None
        self.last_rows = None
        self.applied_chunk = None
        self.initialization = None
        self.initial_yaw = None
        self.reference_held = False
        self.buffer = runtime.ChunkBuffer()
        self.states = runtime.StateBuffer()

    def sample(self, now: float) -> ReferenceDecision:
        """Use frozen endpoint sampling beyond coverage and resume on new chunks."""
        if self.go is None:
            return ReferenceDecision(None, "standing", self.failed_start, None)
        chunk = self.buffer.current
        if chunk is None:
            return ReferenceDecision(
                None, "waiting_first_chunk", self.depth_error or self.worker_error, None
            )
        held = (now - chunk.t0) / 0.02 + 32 > 127
        self.reference_hold_events += held and not self.reference_held
        self.reference_held = held
        try:
            rows = self.buffer.rows_at(now)  # = ChunkBuffer.tokens(now), keeping the rows
            tokens = None if rows is None else C.tracker_tokens(rows)
            if tokens is None or tokens.shape != (6, 69) or not np.isfinite(tokens).all():
                raise ValueError("Invalid DP tracker tokens")
        except (ValueError, RuntimeError, FloatingPointError) as error:
            self.fault = str(error)
            return ReferenceDecision(
                self.last_tokens, "holding_valid_tokens", self.fault, self.chunk_id, rows=self.last_rows
            )
        self.active = True
        self.fault = None
        self.last_tokens = tokens.copy()
        self.last_rows = rows.copy()
        self.applied_chunk = chunk
        return ReferenceDecision(
            tokens,
            "holding_reference" if held else "running",
            self.depth_error or self.worker_error,
            self.chunk_id,
            rows=rows,
        )

    def close(self) -> None:
        self.source.close()
        self.worker.close()


def _rejection(provider: PlannerProvider, readiness: dict) -> str:
    """Why ``start`` refused a Go, for the operator."""
    if provider.go is not None and provider.failed_start is None:
        return "Go was already accepted; E-stop and take over again to restart"
    if provider.active:
        return "the Planner is already running"
    if provider.fault:
        return f"Planner fault: {provider.fault}"
    if not readiness["warmed"]:
        return "the Planner has not finished its warmup plan"
    if readiness["depth_age_s"] is None:
        return "no depth frame yet"
    if not 0 <= readiness["depth_age_s"] <= 0.2:
        return f"depth is {readiness['depth_age_s']:.3f} s old (limit 0.2 s)"
    if readiness["depth_error"]:
        return f"depth error: {readiness['depth_error']}"
    if readiness["worker_error"]:
        return f"Planner worker error: {readiness['worker_error']}"
    return "not ready"
