"""The ``planner`` motion backend: the Planner's 50 Hz side (dixiatielu/sim2real_scalecar#7).

Seams under test (the depth sensor and the CUDA Planner process are the system
boundaries, replaced by scripted fakes):
- ``PlannerReference.tick`` against Scalecar_v2's own ``HardwareDpProvider`` driven the
  way its hardware runtime drives it (poll, warmup, Go, sample): Chunk acceptance,
  resampling, late blend, coverage exhaustion, invalid tokens, the history fed back
  to the Planner and the requests it receives;
- Go and E-stop at that seam (height check, rejection reasons, a new generation);
- the simulation clock with injected latency (``sim.SimulatedTimeWorker``);
- the Reference the Tracker sees through ``tracking.py`` (``motion_backend: planner``).

    env -u PYTHONPATH HF_HUB_OFFLINE=1 UV_FROZEN=1 uv run --with pytest python -m pytest tests/test_planner_reference.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sim2real.planner.provider import JOINT_NAMES, PlannerProvider, PlannerState
from sim2real.planner.reference import PlannerReference, hold_motion_data
from sim2real.planner.scalecar_dp import motion38d as m38
from sim2real.planner.scalecar_dp.dp import runtime

SCALECAR = Path("/home/galbot/projects/Scalecar_v2")
OFFSETS = list(m38.TRACKER_OFFSETS)
T_START = 1000.0
DT = 0.02


def scalecar_module(name: str):
    """Import a module from the Scalecar_v2 checkout without writing bytecode into it."""
    import importlib

    sys.dont_write_bytecode = True
    for path in (SCALECAR / "source/scalecar_dp", SCALECAR / "source/scalecar_deploy"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return importlib.import_module(name)


needs_scalecar = pytest.mark.skipif(not SCALECAR.is_dir(), reason="Scalecar_v2 checkout not present")


# ---------------------------------------------------------------------------
# Scripted boundaries: clock, depth sensor, CUDA Planner process
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.tick = 0

    @property
    def now(self) -> float:
        return T_START + self.tick * DT


def depth_image(frame: int, session: int) -> np.ndarray:
    rng = np.random.default_rng(1000 * session + frame)
    return rng.integers(0, 4000, size=(120, 160), dtype=np.uint16)


class ScriptedDepth:
    """30 Hz downsampled depth packets, newest only, like the Host receiver's mailbox.

    ``gaps`` are clock intervals without packets; ``session`` changes after each gap
    (a camera reconnect)."""

    def __init__(self, clock: Clock, gaps=()) -> None:
        self.clock, self.gaps = clock, list(gaps)
        self.frame = None

    def poll(self):
        now = self.clock.now
        if any(lo <= now < hi for lo, hi in self.gaps):
            return None
        frame = int(math.floor((now - T_START) * 30 + 1e-9))
        if frame == self.frame:
            return None
        self.frame = frame
        session = 1 + sum(now >= hi for _, hi in self.gaps)
        return dict(session=session, frame=frame, acquired=T_START + frame / 30, depth_mm=depth_image(frame, session))

    def close(self) -> None:
        pass


def chunk_rows(request_index: int, t0: float) -> np.ndarray:
    """A plausible abs38 chunk in its own t0 frame: walking forward while slowly turning."""
    t = np.arange(128) * DT
    rows = np.zeros((128, 38))
    rows[:, :29] = 0.1 * np.sin(1.3 * (t0 + t)[:, None] + np.arange(29)[None, :]) + 0.002 * request_index
    yaw = 0.15 * t
    rows[:, 29] = 0.3 * t
    rows[:, 30] = 0.05 * t**2
    rows[:, 31] = 0.78 - 0.01 * np.sin(2.0 * t)
    rows[:, 32], rows[:, 33], rows[:, 34] = np.cos(yaw), np.sin(yaw), 0.0
    rows[:, 35], rows[:, 36], rows[:, 37] = -np.sin(yaw), np.cos(yaw), 0.0
    return rows


class ScriptedWorker:
    """The CUDA Planner process: answers each request per ``script`` on the scripted clock.

    ``script(index, item) -> None | dict(publish_after, latency, prefix, rows?, generation?, t0?)``.
    ``poll()`` returns the latest published result (and keeps returning it), like
    ``GpuPlannerWorker.poll``."""

    def __init__(self, clock: Clock, script, chunk_type) -> None:
        self.clock, self.script, self.chunk_type = clock, script, chunk_type
        self.submitted: list[dict] = []
        self._scheduled: list[tuple[float, dict]] = []
        self.last = None

    def submit(self, item: dict) -> bool:
        self.submitted.append(item)
        plan = self.script(len(self.submitted) - 1, item)
        if plan is not None:
            obs = item["observation"]
            t0 = plan.get("t0", obs.t0)
            rows = plan.get("rows", chunk_rows(len(self.submitted), obs.t0))
            chunk = self.chunk_type(t0=t0, origin=obs.origin, rows=rows, t_ready=obs.t0 + plan["latency"],
                                    prefix=plan["prefix"])
            result = dict(status="ready", generation=plan.get("generation", item["generation"]),
                          request_id=item["request_id"], warmup=item["warmup"], chunk=chunk, t0=chunk.t0,
                          t_ready=chunk.t_ready, prefix=chunk.prefix, inference_ms=1000 * plan["latency"])
            self._scheduled.append((self.clock.now + plan["publish_after"], result))
        return True

    def poll(self):
        due = [entry for entry in self._scheduled if entry[0] <= self.clock.now + 1e-9]
        self._scheduled = [entry for entry in self._scheduled if entry[0] > self.clock.now + 1e-9]
        if due:
            self.last = due[-1][1]
        return self.last

    def close(self) -> None:
        pass


def episode_script(index: int, item: dict):
    """Per request: on time, late (blend), non-finite, wrong generation, stale t0, early
    publication, a 2.4 s outage (coverage exhaustion), then on time again."""
    if item["warmup"]:
        return dict(publish_after=0.02, latency=0.3, prefix=0)
    plan = index  # 1 = first Chunk after Go
    # Latencies avoid exact multiples of 20 ms: k = ceil(latency / 0.02) is the switch frame.
    if plan in (1, 2):
        return dict(publish_after=0.06, latency=0.047, prefix=3)
    if plan == 3:
        return dict(publish_after=0.10, latency=0.087, prefix=3)  # k = 5 > prefix: late, blended
    if plan == 4:
        rows = chunk_rows(plan, item["observation"].t0)
        rows[40, 3] = np.nan
        return dict(publish_after=0.06, latency=0.047, prefix=3, rows=rows)
    if plan == 5:
        return dict(publish_after=0.06, latency=0.047, prefix=3, generation=item["generation"] - 1)
    if plan == 6:
        return dict(publish_after=0.06, latency=0.047, prefix=3, t0=item["observation"].t0 - 0.8)  # older than the current Chunk
    if plan == 7:
        return dict(publish_after=0.0, latency=0.067, prefix=4)  # published before it is ready
    if 8 <= plan <= 19:
        return None  # outage
    if plan == 20:
        return dict(publish_after=0.08, latency=0.077, prefix=4)
    if plan == 21:  # finite rows whose joint velocity overflows: invalid tokens
        rows = chunk_rows(plan, item["observation"].t0)
        rows[:, 0] = 1e308 * (-1.0) ** np.arange(128)
        return dict(publish_after=0.04, latency=0.037, prefix=3, rows=rows)
    return dict(publish_after=0.04, latency=0.037, prefix=3)


def robot_state(tick: int) -> PlannerState:
    """Standing, swaying slightly, heading 0.3 rad with a small roll (measured, Tracker order)."""
    q = np.zeros(29)
    names = list(JOINT_NAMES)
    for side in ("left", "right"):
        q[names.index(f"{side}_hip_pitch_joint")] = -0.2
        q[names.index(f"{side}_knee_joint")] = 0.42
        q[names.index(f"{side}_ankle_pitch_joint")] = -0.23
    q += 0.01 * np.sin(0.05 * tick + np.arange(29))
    yaw, roll = 0.3, 0.01 * math.sin(0.1 * tick)
    quat = m38.quat_mul(m38.yaw_quat(yaw), np.array([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0]))
    return PlannerState(joint_pos=q, root_quat_wxyz=quat)


def hold_frame() -> object:
    return hold_motion_data(
        joint_pos=np.linspace(-0.5, 0.5, 29), pelvis_pos=np.array([0.2, -1.6, 0.79]),
        pelvis_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), future_steps=OFFSETS,
    )


def tokens_of(motion) -> np.ndarray:
    """The Reference Token of a Planner MotionData (Scalecar's definition)."""
    return m38.reference_only_features(
        motion.joint_pos[0], motion.joint_vel[0], motion.body_pos_w[0, :, 0], motion.body_quat_w[0, :, 0], OFFSETS
    )


# ---------------------------------------------------------------------------
# Differential: PlannerReference vs Scalecar's HardwareDpProvider
# ---------------------------------------------------------------------------


@needs_scalecar
def test_planner_joint_order_is_scalecars():
    assert tuple(JOINT_NAMES) == tuple(scalecar_module("scalecar_dp.spec").JOINT_NAMES)


@needs_scalecar
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_reference_and_planner_requests_match_scalecar_over_an_episode():
    scalecar_dp = scalecar_module("scalecar_deploy.hardware.dp")
    scalecar_runtime = scalecar_module("scalecar_dp.dp.runtime")

    gaps = [(T_START + 5.0, T_START + 5.5)]
    clock = Clock()
    theirs_worker = ScriptedWorker(clock, episode_script, scalecar_runtime.Chunk)
    theirs = scalecar_dp.HardwareDpProvider(
        {}, {"joint_names": list(JOINT_NAMES)}, Path("."), source=ScriptedDepth(clock, gaps), worker=theirs_worker
    )
    ours_worker = ScriptedWorker(clock, episode_script, runtime.Chunk)
    hold = hold_frame()
    ours = PlannerReference(
        PlannerProvider(JOINT_NAMES, source=ScriptedDepth(clock, gaps), worker=ours_worker),
        hold=hold, future_steps=OFFSETS,
    )

    go_tick, phases = 30, set()
    for tick in range(600):
        clock.tick = tick
        state, now = robot_state(tick), clock.now
        # Scalecar's hardware runtime, one control cycle (hardware/runtime.py:316-381).
        theirs.poll(state, now)
        theirs.warmup(state, now)
        if tick == go_tick:
            assert theirs.start(state, now)
            ours.request_go()
        decision = theirs.sample(now)

        motion = ours.tick(state, now)

        phases.add(decision.phase)
        assert ours.phase == decision.phase, tick
        if decision.tokens is None:
            assert motion is hold, tick
        else:
            np.testing.assert_array_equal(tokens_of(motion), decision.tokens, err_msg=f"tick {tick}")

    assert phases == {"standing", "waiting_first_chunk", "running", "holding_reference", "holding_valid_tokens"}
    assert ours.invalid_token_ticks == 10
    assert len(ours_worker.submitted) == len(theirs_worker.submitted) > 30
    for mine, expected in zip(ours_worker.submitted, theirs_worker.submitted):
        assert {k: mine[k] for k in ("generation", "request_id", "warmup", "depth_times")} == {
            k: expected[k] for k in ("generation", "request_id", "warmup", "depth_times")
        }
        a, b = mine["observation"], expected["observation"]
        assert a.t0 == b.t0 and a.go_clock == b.go_clock and a.depth_idx == b.depth_idx
        np.testing.assert_array_equal(a.origin[0], b.origin[0])
        assert a.origin[1] == b.origin[1]
        for field in ("depth_mm", "depth_age_s", "history", "history_mask"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field), err_msg=field)
    provider = ours.provider
    assert provider.buffer.late_switches == theirs.buffer.late_switches == 1
    assert provider.buffer.stale == theirs.buffer.stale > 0
    assert provider.reference_hold_events == theirs.reference_hold_events == 1
    assert provider.depth_delay_events == theirs.depth_delay_events == 1
    assert provider.recovery_events == theirs.recovery_events == 1


# ---------------------------------------------------------------------------
# Literal behaviour at the same seam (no Scalecar checkout needed)
# ---------------------------------------------------------------------------

IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def constant_rows(q_value: float, z: float = 0.78) -> np.ndarray:
    rows = np.zeros((128, 38))
    rows[:, :29] = q_value
    rows[:, 31] = z
    rows[:, 32:38] = IDENTITY_ROT6D
    return rows


class ManualWorker:
    """The CUDA Planner process, answered by hand: ``respond`` publishes a Chunk for the latest request."""

    def __init__(self) -> None:
        self.submitted: list[dict] = []
        self.last = None

    def submit(self, item: dict) -> bool:
        self.submitted.append(item)
        return True

    def respond(self, *, rows=None, latency: float = 0.037, prefix: int = 3, request: dict | None = None) -> None:
        item = request or self.submitted[-1]
        obs = item["observation"]
        rows = constant_rows(0.0) if rows is None else rows
        chunk = runtime.Chunk(t0=obs.t0, origin=obs.origin, rows=rows, t_ready=obs.t0 + latency, prefix=prefix)
        self.last = dict(status="ready", generation=item["generation"], request_id=item["request_id"],
                         warmup=item["warmup"], chunk=chunk, inference_ms=1000 * latency)

    def poll(self):
        return self.last

    def close(self) -> None:
        pass


class Rig:
    """One PlannerReference on a scripted clock, with a standing robot and fresh depth."""

    def __init__(self, *, warm: bool = True) -> None:
        self.clock, self.worker = Clock(), ManualWorker()
        self.hold = hold_frame()
        self.reference = PlannerReference(
            PlannerProvider(JOINT_NAMES, source=ScriptedDepth(self.clock), worker=self.worker),
            hold=self.hold, future_steps=OFFSETS,
        )
        self.state = robot_state(0)
        self.step()
        if warm:
            self.worker.respond()  # warmup result
            self.step()

    def step(self, ticks: int = 1):
        for _ in range(ticks):
            motion = self.reference.tick(self.state, self.clock.now)
            self.clock.tick += 1
        return motion

    def go(self):
        self.reference.request_go()
        return self.step()


def test_hold_lasts_after_go_until_the_first_chunk_is_ready():
    rig = Rig()
    assert rig.go() is rig.hold
    assert rig.reference.phase == "waiting_first_chunk"
    rig.worker.respond(rows=constant_rows(0.25), latency=0.047)  # ready at Go + 0.047 s

    assert rig.step() is rig.hold  # Go + 0.02 s
    assert rig.step() is rig.hold  # Go + 0.04 s
    motion = rig.step()  # Go + 0.06 s

    assert rig.reference.phase == "running"
    np.testing.assert_allclose(motion.joint_pos[0], np.full((6, 29), 0.25))
    np.testing.assert_allclose(motion.joint_vel[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(motion.body_pos_w[0, :, 0], [[0.0, 0.0, 0.78]] * 6, atol=1e-12)
    np.testing.assert_allclose(motion.body_quat_w[0, :, 0], [[1.0, 0.0, 0.0, 0.0]] * 6, atol=1e-12)
    assert rig.reference.stats()["first_chunk_delay_s"] == pytest.approx(0.06)


def test_a_late_chunk_is_blended_into_the_old_one_over_three_frames():
    rig = Rig()
    rig.go()
    rig.worker.respond(rows=constant_rows(1.0), latency=0.037)
    rig.step(9)  # next plan is due at Go + 0.2 s, i.e. the next tick
    rig.step()
    request = rig.worker.submitted[-1]
    rig.worker.respond(rows=constant_rows(2.0), latency=0.087, prefix=3)  # switch frame k = 5 > prefix 3
    rig.step(4)
    motion = rig.step()  # t0 + 0.10 s: frame 5 of the new Chunk

    # frames 5, 6, 7 are 3/4, 2/4, 1/4 old; Tracker offsets 0, 2, 4 -> frames 5, 7, 9.
    np.testing.assert_allclose(motion.joint_pos[0, :3, 0], [1.25, 1.75, 2.0], atol=1e-12)
    assert rig.reference.stats()["late_chunks"] == 1
    assert request["observation"].t0 == pytest.approx(T_START + 0.24)  # Go at T_START + 0.04


def test_beyond_its_last_frame_the_chunk_is_held_and_counted():
    rig = Rig()
    rig.go()
    rows = constant_rows(0.0)
    rows[:, 0] = 0.01 * np.arange(128)  # left hip pitch ramps one centiradian per frame
    rig.worker.respond(rows=rows, latency=0.037)
    rig.worker.submit = lambda item: True  # the Planner stops answering: no newer Chunk
    rig.step(2)
    motion = rig.step(98)  # t0 + 2.00 s: frame 100; the 33-frame window ends past frame 127

    np.testing.assert_allclose(motion.joint_pos[0, :, 0], [1.00, 1.02, 1.04, 1.08, 1.16, 1.27], atol=1e-12)
    np.testing.assert_allclose(motion.joint_vel[0, :, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.0], atol=1e-9)
    assert rig.reference.phase == "holding_reference"
    stats = rig.reference.stats()
    # Frames 96-100 need rows past 127; frame 95 too when (now - t0) / 0.02 rounds above 95.
    assert stats["coverage_exhausted_events"] == 1 and stats["coverage_exhausted_ticks"] in (5, 6)


def test_go_is_refused_until_the_warmup_plan_is_back():
    rig = Rig(warm=False)
    rig.go()
    assert rig.reference.phase == "standing" and rig.reference.go_time is None
    assert "warmup" in rig.reference.provider.start_rejection

    rig.worker.respond()
    rig.step()
    rig.go()
    assert rig.reference.phase == "waiting_first_chunk"


def test_go_is_refused_with_a_foot_off_the_ground_and_can_be_retried():
    rig = Rig()
    names = list(JOINT_NAMES)
    lifted = rig.state.joint_pos.copy()
    lifted[[names.index("left_hip_pitch_joint"), names.index("left_knee_joint")]] = (-0.8, 1.6)
    standing = rig.state
    rig.state = PlannerState(joint_pos=lifted, root_quat_wxyz=standing.root_quat_wxyz)

    assert rig.go() is rig.hold
    assert rig.reference.phase == "standing"
    assert rig.reference.provider.start_rejection.startswith("pelvis height check failed")

    rig.state = standing
    rig.go()
    assert rig.reference.phase == "waiting_first_chunk"
    assert rig.reference.provider.initialization["height_m"] == pytest.approx(0.7785, abs=2e-3)


def test_e_stop_drops_the_go_and_its_chunks_until_a_new_go():
    rig = Rig()
    rig.go()
    rig.worker.respond(rows=constant_rows(0.5), latency=0.037)
    rig.step(2)
    assert rig.reference.phase == "running"
    rig.step(8)
    in_flight = rig.worker.submitted[-1]

    rig.reference.abort()
    rig.worker.respond(rows=constant_rows(0.9), request=in_flight)
    assert rig.step(5) is rig.hold
    assert rig.reference.phase == "standing"

    rig.go()  # the worker is warmed up again first
    assert rig.reference.go_time is None and "warmup" in rig.reference.provider.start_rejection
    rig.worker.respond()
    rig.step()
    rig.go()
    rig.worker.respond(rows=constant_rows(0.7), latency=0.037)
    motion = rig.step(2)
    np.testing.assert_allclose(motion.joint_pos[0], 0.7)


def test_after_a_worker_crash_and_an_e_stop_the_next_go_waits_for_the_restarted_worker():
    rig = Rig()
    rig.go()
    rig.worker.respond(rows=constant_rows(0.5), latency=0.037)
    rig.step(2)
    assert rig.reference.phase == "running"
    rig.worker.last = dict(status="error", error="GPU planner exited; restarting on CUDA")  # the worker crashed
    rig.step(3)

    rig.reference.abort()  # E-stop
    rig.step(3)
    rig.go()  # the worker is still down
    assert rig.reference.go_time is None and "worker" in rig.reference.provider.start_rejection

    rig.worker.last = dict(status="loaded")  # restarted
    rig.step()
    assert rig.worker.submitted[-1]["warmup"]  # the newest request is a fresh warmup plan
    rig.worker.respond()
    rig.step()
    rig.go()
    assert rig.reference.go_time is not None
    rig.worker.respond(rows=constant_rows(0.7), latency=0.037)
    np.testing.assert_allclose(rig.step(2).joint_pos[0], 0.7)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_planner_stats_restart_with_each_go():
    rig = Rig()
    depth = rig.reference.provider.source
    rig.go()
    rig.worker.respond(rows=constant_rows(0.5), latency=0.037)
    rig.step(10)
    invalid = constant_rows(0.5)
    invalid[:, 0] = 1e308 * (-1.0) ** np.arange(128)  # finite rows, overflowing joint velocity
    rig.worker.respond(rows=invalid, latency=0.037)
    depth.gaps = [(rig.clock.now + 0.1, rig.clock.now + 0.6)]  # the camera drops out and reconnects
    rig.step(130)  # no more answers: the last Chunk runs out
    first = rig.reference.stats()
    assert first["plans_received"] == 2 and first["plans_requested"] > 2
    assert first["invalid_token_ticks"] > 0 and first["coverage_exhausted_events"] == 1
    assert first["depth_delay_events"] == 1 and first["camera_recovery_events"] == 1

    rig.reference.abort()
    rig.step()
    rig.worker.respond()  # the re-warm plan
    rig.step()
    rig.go()
    assert rig.reference.go_time is not None
    rig.worker.respond(rows=constant_rows(0.7), latency=0.037)
    rig.step(12)
    second = rig.reference.stats()

    assert (second["plans_requested"], second["plans_received"], second["chunks_applied"]) == (2, 1, 1)
    for key in ("late_chunks", "coverage_exhausted_events", "coverage_exhausted_ticks", "invalid_token_ticks",
                "depth_delay_events", "camera_recovery_events"):
        assert second[key] == 0, key


# ---------------------------------------------------------------------------
# Simulation clock with injected latency
# ---------------------------------------------------------------------------


class ThreadedWorker:
    """The CUDA Planner process with a fixed wall-clock latency.

    Like the real worker, it stamps ``t_ready`` with its own clock, ``time.monotonic()``
    plus the request's ``clock_offset_s``."""

    def __init__(self, latency_s: float) -> None:
        import threading

        self.latency_s = latency_s
        self.last = None
        self._lock = threading.Lock()
        self._threads = []

    def submit(self, item: dict) -> bool:
        import threading
        import time

        def plan():
            time.sleep(self.latency_s)
            obs = item["observation"]
            t_ready = time.monotonic() + item.get("clock_offset_s", 0.0)
            chunk = runtime.Chunk(t0=obs.t0, origin=obs.origin, rows=constant_rows(0.1), t_ready=t_ready, prefix=3)
            with self._lock:
                self.last = dict(status="ready", generation=item["generation"], request_id=item["request_id"],
                                 warmup=item["warmup"], chunk=chunk, inference_ms=1000 * self.latency_s)

        thread = threading.Thread(target=plan, daemon=True)
        thread.start()
        self._threads.append(thread)
        return True

    def poll(self):
        with self._lock:
            return self.last

    def close(self) -> None:
        for thread in self._threads:
            thread.join()


def simulated_rig(latency_s: float):
    from sim2real.planner.sim import SimulatedTimeWorker

    clock = Clock()
    worker = SimulatedTimeWorker(ThreadedWorker(latency_s), clock=lambda: clock.now)
    reference = PlannerReference(
        PlannerProvider(JOINT_NAMES, source=ScriptedDepth(clock), worker=worker), hold=hold_frame(), future_steps=OFFSETS
    )
    return clock, reference


def run_until_first_reference(clock: Clock, reference: PlannerReference, *, go_tick: int, tick_wall_s: float = 0.0):
    import time

    for tick in range(go_tick + 50):
        clock.tick = tick
        if tick == go_tick:
            reference.request_go()
        reference.tick(robot_state(tick), clock.now)
        if not reference.holding:
            return
        time.sleep(tick_wall_s)
    raise AssertionError("no Chunk was applied")


def test_a_chunk_is_ready_at_its_request_time_plus_the_measured_latency_when_simulation_outruns_the_wall():
    clock, reference = simulated_rig(latency_s=0.065)
    run_until_first_reference(clock, reference, go_tick=20)  # ticks take no wall time at all

    stats = reference.stats()
    assert 65.0 <= stats["latency_ms"]["p50"] < 80.0
    assert stats["first_chunk_delay_s"] == pytest.approx(0.08)  # first tick at or after Go + latency
    assert reference.go_time == pytest.approx(T_START + 0.4)  # the warmup did not hold the clock back past Go


def test_a_slow_simulation_does_not_add_its_own_slowness_to_the_latency():
    clock, reference = simulated_rig(latency_s=0.025)
    run_until_first_reference(clock, reference, go_tick=10, tick_wall_s=0.1)  # 0.1 s of wall per 0.02 s tick

    stats = reference.stats()
    assert 25.0 <= stats["latency_ms"]["p50"] < 40.0
    assert stats["first_chunk_delay_s"] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# tracking.py with motion_backend: planner -- what the Tracker's network is given
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

import yaml  # noqa: E402

from sim2real.config.robots import get_robot_cfg  # noqa: E402
from sim2real.planner.reference import PlannerIO  # noqa: E402
from sim2real.rl_policy import base_policy as base_policy_module  # noqa: E402
from sim2real.rl_policy.controllers.base import ControllerBase  # noqa: E402
from sim2real.rl_policy.robot_io import RobotIO, RobotState  # noqa: E402
from sim2real.rl_policy.tracking import Tracking, TrackingArgs  # noqa: E402
from sim2real.rl_policy.utils.motion import MotionDataset  # noqa: E402

G1_JOINTS = list(get_robot_cfg("g1").joint_names)
TRACKER_JOINTS = list(JOINT_NAMES)
FRAME0_Q = np.linspace(-0.3, 0.3, len(G1_JOINTS))  # robot order


class StandingG1(RobotIO):
    def __init__(self) -> None:
        state = robot_state(0)
        self.joint_pos = np.zeros(len(G1_JOINTS))
        for i, name in enumerate(TRACKER_JOINTS):
            self.joint_pos[G1_JOINTS.index(name)] = state.joint_pos[i]
        self.quat = state.root_quat_wxyz
        self.commands = []

    def read_state(self) -> RobotState:
        qpos = np.concatenate([[0.0, 0.0, 0.78], self.quat, self.joint_pos])
        return RobotState(qpos=qpos, qvel=np.zeros(6 + len(G1_JOINTS)), joint_torque=np.zeros(len(G1_JOINTS)), tick=0)

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        self.commands.append(np.array(q_target))


class Operator(ControllerBase):
    name = "test"

    def __init__(self) -> None:
        self.key_pressed, self.requested_mode = set(), None

    def get_control_mode(self):
        mode, self.requested_mode = self.requested_mode, None
        return mode


def frame0_motion() -> MotionDataset:
    """A two-frame Motion: frame 0 is FRAME0_Q with the pelvis at (0.2, -1.6, 0.79), turned 0.5 rad."""
    joints = np.stack([FRAME0_Q, FRAME0_Q + 0.1])
    body = np.tile([[[0.2, -1.6, 0.79]]], (2, 1, 1))
    quat = np.tile([[[math.cos(0.25), 0.0, 0.0, math.sin(0.25)]]], (2, 1, 1))
    source = SimpleNamespace(
        num_motions=1, starts=np.array([0]), ends=np.array([2]),
        data=SimpleNamespace(joint_pos=joints, joint_vel=np.ones_like(joints), body_pos_w=body,
                             body_lin_vel_w=np.zeros_like(body), body_quat_w=quat, body_ang_vel_w=np.zeros_like(body)),
    )
    return MotionDataset(dataset=source, body_names=["pelvis"], joint_names=G1_JOINTS,
                         src_joint_indices=list(range(29)), dest_joint_indices=list(range(29)))


def tracker_yaml() -> dict:
    """The Reference part of the Scalecar Tracker's policy.yaml (scripts/export_scalecar_tracker_onnx.py)."""
    steps = OFFSETS
    return {
        "model_path": "unused.onnx",
        "observation": {
            "ref_command": {"sonic_command_multi_future_nonflat": {
                "_target_": "sonic.sonic_command_multi_future_nonflat", "future_steps": steps,
                "joint_names": TRACKER_JOINTS}},
            "ref_pelvis_pos": {"ref_body_pos_future_local": {
                "_target_": "mimic_lite.ref_body_pos_future_local", "future_steps": steps,
                "body_names": ["pelvis"], "anchor_body_name": "pelvis"}},
            "ref_pelvis_ori": {"ref_body_ori_future_local": {
                "_target_": "mimic_lite.ref_body_ori_future_local", "future_steps": steps,
                "body_names": ["pelvis"], "anchor_body_name": "pelvis"}},
        },
        "joint_names_simulation": G1_JOINTS,
        "body_names_simulation": ["pelvis"],
        "policy_joint_names": TRACKER_JOINTS,
        "default_joint_pos": {".*": 0.0},
        "joint_kp": {".*": 40.0},
        "joint_kd": {".*": 2.0},
        "action_scale": 0.5,
        "motion": {"motion_backend": "npz", "future_steps": steps, "joint_names": TRACKER_JOINTS,
                   "body_names": ["pelvis"], "root_body_name": "pelvis"},
    }


@pytest.fixture
def planner_tracking(tmp_path, monkeypatch):
    clock, worker, inputs = Clock(), ManualWorker(), []

    def network(input_dict):
        inputs.append({k: np.array(input_dict[k]) for k in ("ref_command", "ref_pelvis_pos", "ref_pelvis_ori")})
        return {"action": np.zeros(29, dtype=np.float32)}

    class PlannerTracking(Tracking):
        def _planner_io(self, options):
            assert options["checkpoint_sha256"]  # the pinned defaults reach the hook
            return PlannerIO(source=ScriptedDepth(clock), worker=worker, clock=lambda: clock.now)

    robot, operator = StandingG1(), Operator()
    monkeypatch.setattr(base_policy_module, "create_robot_io", lambda **_: robot)
    monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: network)
    monkeypatch.setattr(base_policy_module, "KeyboardController", lambda: operator)
    monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: frame0_motion()))
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(tracker_yaml()))
    tracker = PlannerTracking(args=TrackingArgs(policy_config=str(path), motion_backend="planner", motion_path="take.npz"))

    def tick(mode=None):
        operator.requested_mode = mode
        tracker.step()
        clock.tick += 1
        return inputs[-1]

    return SimpleNamespace(tracker=tracker, worker=worker, clock=clock, tick=tick)


def split_inputs(inputs: dict):
    command = inputs["ref_command"].reshape(2, 6, 29)
    return command[0], command[1], inputs["ref_pelvis_pos"].reshape(6, 3), inputs["ref_pelvis_ori"].reshape(6, 2, 3)


def test_the_tracker_holds_motion_frame_0_until_the_first_chunk_then_follows_it(planner_tracking):
    rig = planner_tracking
    rig.tick()  # streaming state; the Planner warms up
    rig.worker.respond()
    hold = rig.tick(mode="policy")  # R1: the Tracker takes over

    q, dq, pelvis_pos, pelvis_ori = split_inputs(hold)
    frame0 = np.array([FRAME0_Q[G1_JOINTS.index(name)] for name in TRACKER_JOINTS])
    np.testing.assert_allclose(q, np.tile(frame0, (6, 1)), atol=1e-3)  # float16 Motion storage
    assert not dq.any()
    np.testing.assert_allclose(pelvis_pos, [[0.0, 0.0, 0.79]] * 6, atol=1e-3)

    rig.tracker.motion_buffer.request_go()
    rig.tick()
    go = rig.clock.now - DT
    request = rig.worker.submitted[-1]
    rows = chunk_rows(1, request["observation"].t0)
    rig.worker.respond(rows=rows, latency=0.037)
    assert np.array_equal(rig.tick()["ref_command"], hold["ref_command"])  # Go + 0.02: not ready yet
    following = rig.tick()  # Go + 0.04
    now = go + 2 * DT

    obs = request["observation"]
    chunk = runtime.Chunk(t0=obs.t0, origin=obs.origin, rows=rows, t_ready=obs.t0 + 0.037, prefix=3)
    expected = runtime.ChunkBuffer(current=chunk).tokens(now)  # Scalecar's Reference Token at `now`
    q, dq, pelvis_pos, pelvis_ori = split_inputs(following)
    np.testing.assert_allclose(q, expected[:, 0:29], atol=1e-6)
    np.testing.assert_allclose(dq, expected[:, 29:58], atol=1e-5)
    np.testing.assert_allclose(pelvis_pos[:, :2], expected[:, 58:60], atol=1e-6)  # displacement, frame-0 heading
    np.testing.assert_allclose(pelvis_pos[:, 2], expected[:, 61], atol=1e-6)  # height above the ground
    rotation = np.concatenate([pelvis_ori, np.cross(pelvis_ori[:, 0], pelvis_ori[:, 1])[:, None]], axis=1)
    np.testing.assert_allclose(rotation[:, :, 0], expected[:, 62:65], atol=1e-6)  # pelvis x axis
    np.testing.assert_allclose(rotation[:, :, 2], expected[:, 65:68], atol=1e-6)  # pelvis z axis


def test_planner_backend_rejects_other_future_steps(tmp_path, monkeypatch):
    config = tracker_yaml()
    config["motion"]["future_steps"] = [0, 2, 4, 8, 16]
    for group in config["observation"].values():
        for obs in group.values():
            obs["future_steps"] = [0, 2, 4, 8, 16]
    monkeypatch.setattr(base_policy_module, "create_robot_io", lambda **_: StandingG1())
    monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: (lambda d: d))
    monkeypatch.setattr(base_policy_module, "KeyboardController", Operator)
    monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: frame0_motion()))
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))

    class PlannerTracking(Tracking):
        def _planner_io(self, options):
            return PlannerIO(source=ScriptedDepth(Clock()), worker=ManualWorker(), clock=lambda: 0.0)

    with pytest.raises(ValueError, match="future_steps"):
        PlannerTracking(args=TrackingArgs(policy_config=str(path), motion_backend="planner", motion_path="take.npz"))
