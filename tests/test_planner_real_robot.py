"""The Planner on the real robot (dixiatielu/sim2real_scalecar#8): operator keys and depth prints.

Seams under test (the depth receiver, the CUDA Planner process, the robot transport
and the Controller's input device are the system boundaries, replaced by fakes):
- ``PlannerReference.tick``: the terminal messages for stale depth, recovery, and a
  camera disconnect and reconnect (edge-triggered, at most once per second while the
  condition lasts, one message when it ends);
- ``Tracking.step()`` with ``motion_backend: planner``: A / R1 / B / R2 from the keyboard
  and the Unitree joystick (Go and its height check, B having no effect after Go, R2
  dropping the plan), and File Playback's B left unchanged;
- ``Tracking._planner_io`` on the robot: the depth transport options from YAML and CLI.

    env -u PYTHONPATH HF_HUB_OFFLINE=1 UV_FROZEN=1 uv run --with pytest python -m pytest tests/test_planner_real_robot.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import math
import re

import numpy as np
import pytest
from loguru import logger

from sim2real.planner.provider import JOINT_NAMES, PlannerProvider, PlannerState
from sim2real.planner.reference import PlannerReference, hold_motion_data
from sim2real.planner.scalecar_dp import motion38d as m38
from sim2real.planner.scalecar_dp.dp import runtime

OFFSETS = list(m38.TRACKER_OFFSETS)
T_START = 500.0
DT = 0.02


# ---------------------------------------------------------------------------
# Boundaries: clock, camera (Host depth receiver), CUDA Planner process
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.tick = 0

    @property
    def now(self) -> float:
        return T_START + self.tick * DT


class Camera:
    """The Host depth receiver's ``poll``: the newest 30 Hz frame, an error after a
    disconnect, a new session after a reconnect. Frames are about 30 ms old on arrival."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.session, self.frozen, self.error = 1, False, None
        self.latest = None

    def poll(self):
        if self.error is not None:
            self.latest = dict(error=self.error)
        elif not self.frozen:
            frame = int(math.floor((self.clock.now - T_START) * 30 + 1e-9))
            if self.latest is None or "error" in self.latest or self.latest["frame"] != frame:
                acquired = T_START + frame / 30 - 0.03
                self.latest = dict(session=self.session, frame=frame, acquired=acquired,
                                   depth_mm=np.full((120, 160), 1500, dtype=np.uint16))
        return self.latest

    def disconnect(self, error: str = "EOFError: Depth stream closed during packet") -> None:
        self.error = error

    def reconnect(self) -> None:
        self.error, self.frozen = None, False
        self.session += 1

    def close(self) -> None:
        pass


IDENTITY_ROT6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def constant_rows(q_value: float, z: float = 0.78) -> np.ndarray:
    rows = np.zeros((128, 38))
    rows[:, :29] = q_value
    rows[:, 31] = z
    rows[:, 32:38] = IDENTITY_ROT6D
    return rows


class Planner:
    """The CUDA Planner process, answering every request at once with a constant Chunk
    ready 37 ms after its t0 (warmups included)."""

    def __init__(self, q_value: float = 0.25) -> None:
        self.q_value = q_value
        self.submitted: list[dict] = []
        self.last = None

    def submit(self, item: dict) -> bool:
        self.submitted.append(item)
        obs = item["observation"]
        chunk = runtime.Chunk(t0=obs.t0, origin=obs.origin, rows=constant_rows(self.q_value),
                              t_ready=obs.t0 + 0.037, prefix=3)
        self.last = dict(status="ready", generation=item["generation"], request_id=item["request_id"],
                         warmup=item["warmup"], chunk=chunk, inference_ms=37.0)
        return True

    def poll(self):
        return self.last

    def close(self) -> None:
        pass


def standing_state(*, left_foot_lifted: bool = False) -> PlannerState:
    """Standing, slightly crouched (Go height about 0.78 m); optionally with the left foot raised."""
    q = np.zeros(29)
    names = list(JOINT_NAMES)
    for side in ("left", "right"):
        q[names.index(f"{side}_hip_pitch_joint")] = -0.2
        q[names.index(f"{side}_knee_joint")] = 0.42
        q[names.index(f"{side}_ankle_pitch_joint")] = -0.23
    if left_foot_lifted:
        q[[names.index("left_hip_pitch_joint"), names.index("left_knee_joint")]] = (-0.8, 1.6)
    return PlannerState(joint_pos=q, root_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))


@pytest.fixture
def messages():
    """The runtime's log lines as ``(level, message)``."""
    lines: list[tuple[str, str, float]] = []
    sink = logger.add(lambda m: lines.append((m.record["level"].name, m.record["message"])), level="DEBUG")
    yield lines
    logger.remove(sink)


def depth_lines(lines):
    return [(level, text) for level, text in lines if text.startswith("Depth")]


# ---------------------------------------------------------------------------
# Depth prints (PlannerReference.tick)
# ---------------------------------------------------------------------------


class Rig:
    """A PlannerReference on a scripted clock with the camera above; one ``step`` is one 50 Hz tick."""

    def __init__(self) -> None:
        self.clock, self.camera, self.planner = Clock(), Camera(Clock()), Planner()
        self.camera.clock = self.clock
        self.hold = hold_motion_data(joint_pos=np.zeros(29), pelvis_pos=np.array([0.0, 0.0, 0.79]),
                                     pelvis_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]), future_steps=OFFSETS)
        self.reference = PlannerReference(
            PlannerProvider(JOINT_NAMES, source=self.camera, worker=self.planner), hold=self.hold,
            future_steps=OFFSETS,
        )
        self.state = standing_state()
        self.stamps: list[tuple[float, int]] = []

    def step(self, seconds: float, lines=None) -> None:
        for _ in range(round(seconds / DT)):
            before = 0 if lines is None else len(lines)
            self.reference.tick(self.state, self.clock.now)
            if lines is not None:
                self.stamps += [(self.clock.now, i) for i in range(before, len(lines))]
            self.clock.tick += 1

    def when(self, lines, index: int) -> float:
        return next(t for t, i in self.stamps if i == index)


def age_in(text: str) -> float:
    return float(re.search(r"(\d+\.\d+) s old", text).group(1))


def test_stale_depth_warns_at_once_repeats_at_most_once_a_second_and_reports_the_recovery(messages):
    rig = Rig()
    rig.step(1.0, messages)
    rig.camera.frozen = True  # frames stop arriving
    frozen_at = rig.clock.now
    rig.step(2.5, messages)
    rig.camera.frozen = False
    rig.step(1.0, messages)

    lines = depth_lines(messages)
    assert [level for level, _ in lines] == ["INFO", "WARNING", "WARNING", "WARNING", "INFO"]
    assert lines[0][1].startswith("Depth: first frame received")
    first, repeats, recovery = lines[1][1], [text for _, text in lines[2:4]], lines[4][1]
    assert first.startswith("Depth stale") and "limit 0.2 s" in first and "real age" in first
    assert 0.2 < age_in(first) <= 0.2 + DT
    assert all(text.startswith("Depth still stale") for text in repeats)
    assert recovery.startswith("Depth recovered")

    index = [i for i, (_, text) in enumerate(messages) if text.startswith("Depth")]
    times = [rig.when(messages, i) for i in index]
    assert times[2] - times[1] == pytest.approx(1.0, abs=DT / 2)
    assert times[3] - times[2] == pytest.approx(1.0, abs=DT / 2)
    assert times[4] == pytest.approx(frozen_at + 2.5, abs=DT / 2)  # the first fresh frame
    assert age_in(repeats[0]) == pytest.approx(age_in(first) + 1.0, abs=1e-6)


def test_a_camera_disconnect_and_reconnect_are_reported_the_same_way(messages):
    rig = Rig()
    rig.step(1.0, messages)
    rig.camera.disconnect()
    rig.step(2.5, messages)
    rig.camera.disconnect("RuntimeError: G1 depth source did not become ready; see depth_source.log")
    rig.step(1.0, messages)
    rig.camera.reconnect()
    rig.step(1.0, messages)

    lines = depth_lines(messages)[1:]  # after "first frame received"
    levels = [level for level, _ in lines]
    assert levels == ["WARNING", "WARNING", "WARNING", "WARNING", "INFO"]
    assert lines[0][1].startswith("Depth camera disconnected (last frame") and "Depth stream closed" in lines[0][1]
    assert all(text.startswith("Depth camera still disconnected") for _, text in lines[1:4])
    assert "did not become ready" in lines[3][1]  # the newest error
    assert lines[4][1].startswith("Depth camera reconnected")
    assert not any("stale" in text for _, text in lines)  # the disconnect covers the stale frame


def test_a_new_camera_session_without_a_reported_error_is_a_reconnect(messages):
    rig = Rig()
    rig.step(0.5, messages)
    rig.camera.reconnect()
    rig.step(0.5, messages)

    lines = depth_lines(messages)
    assert [level for level, _ in lines] == ["INFO", "INFO"]
    assert lines[1][1].startswith("Depth camera reconnected")


def test_the_planner_keeps_planning_from_the_last_frame_while_depth_is_stale(messages):
    rig = Rig()
    rig.step(0.2)
    rig.reference.request_go()
    rig.step(0.5)
    assert rig.reference.phase == "running"
    rig.camera.frozen = True
    requested = len(rig.planner.submitted)
    rig.step(1.0)

    assert rig.reference.phase == "running" and len(rig.planner.submitted) == requested + 5
    assert rig.planner.submitted[-1]["observation"].depth_age_s.max() > 0.9


def test_the_planner_stats_line_starts_over_with_each_go(messages):
    rig = Rig()
    rig.step(0.2)
    rig.reference.request_go()
    rig.step(6.0, messages)
    rig.reference.abort()
    rig.step(2.0)
    rig.reference.request_go()
    second_go = rig.clock.now
    rig.step(6.0, messages)

    stats = [i for i, (_, text) in enumerate(messages) if text.startswith("Planner: ") and " plans, latency" in text]
    times = [rig.when(messages, i) for i in stats]
    assert len(times) == 2 and times[1] == pytest.approx(second_go + 5.0, abs=DT / 2)


# ---------------------------------------------------------------------------
# Operator keys through Tracking.step() (motion_backend: planner)
# ---------------------------------------------------------------------------

import threading  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import yaml  # noqa: E402

from sim2real.config.robots import get_robot_cfg  # noqa: E402
from sim2real.planner.reference import PlannerIO  # noqa: E402
from sim2real.rl_policy import base_policy as base_policy_module  # noqa: E402
from sim2real.rl_policy.control_mode import resolve_keyboard_control_mode  # noqa: E402
from sim2real.rl_policy.controllers.keyboard import KeyboardController  # noqa: E402
from sim2real.rl_policy.controllers.unitree_joystick import (  # noqa: E402
    UnitreeJoystickController,
    UnitreeJoystickState,
)
from sim2real.rl_policy.robot_io import RobotIO, RobotState  # noqa: E402
from sim2real.rl_policy.tracking import Tracking, TrackingArgs  # noqa: E402
from sim2real.rl_policy.utils.motion import MotionDataset  # noqa: E402

G1_JOINTS = list(get_robot_cfg("g1").joint_names)
TRACKER_JOINTS = list(JOINT_NAMES)
MOTION_FRAMES = 20


class StandingG1(RobotIO):
    """The robot transport: a G1 standing in ``standing_state``'s pose (or with its left foot raised)."""

    def __init__(self) -> None:
        self.left_foot_lifted = False

    def read_state(self) -> RobotState:
        state = standing_state(left_foot_lifted=self.left_foot_lifted)
        joint_pos = np.zeros(len(G1_JOINTS))
        for i, name in enumerate(TRACKER_JOINTS):
            joint_pos[G1_JOINTS.index(name)] = state.joint_pos[i]
        qpos = np.concatenate([[0.0, 0.0, 0.78], state.root_quat_wxyz, joint_pos])
        return RobotState(qpos=qpos, qvel=np.zeros(6 + len(G1_JOINTS)), joint_torque=np.zeros(len(G1_JOINTS)), tick=0)

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        pass


class Keyboard(KeyboardController):
    """The keyboard Controller without its terminal listener: ``press`` is the listener's key press."""

    def __init__(self) -> None:
        self.key_pressed: set[str] = set()
        self._pending_control_mode = None
        self._pending_extra_keys: list[str] = []
        self._lock = threading.Lock()

    def press(self, key: str) -> None:
        mode = resolve_keyboard_control_mode(key)
        with self._lock:
            if mode is not None:
                self._pending_control_mode = mode
            else:
                self._pending_extra_keys.append(key)

    def close(self) -> None:
        pass


class Joystick(UnitreeJoystickController):
    """The Unitree joystick Controller without DDS: ``press`` holds a button for one read."""

    def __init__(self) -> None:
        self._wc_lock = threading.Lock()
        self._pending_extra_keys: list[str] = []
        self._wc_msg = UnitreeJoystickState()
        self._last_wc_msg = UnitreeJoystickState()

    def press(self, button: str) -> None:
        self._wc_msg = UnitreeJoystickState(**{button: True})

    def get_control_mode(self):
        mode = super().get_control_mode()
        self._wc_msg = UnitreeJoystickState()  # released after one read
        return mode

    def close(self) -> None:
        pass


def motion_dataset() -> MotionDataset:
    """A short Motion: joints drift away from frame 0, pelvis at (0.2, -1.6, 0.79)."""
    joints = np.linspace(0.0, 0.2, MOTION_FRAMES)[:, None] + np.linspace(-0.3, 0.3, len(G1_JOINTS))[None]
    body = np.tile([[[0.2, -1.6, 0.79]]], (MOTION_FRAMES, 1, 1))
    quat = np.tile([[[1.0, 0.0, 0.0, 0.0]]], (MOTION_FRAMES, 1, 1))
    source = SimpleNamespace(
        num_motions=1, starts=np.array([0]), ends=np.array([MOTION_FRAMES]),
        data=SimpleNamespace(joint_pos=joints, joint_vel=np.zeros_like(joints), body_pos_w=body,
                             body_lin_vel_w=np.zeros_like(body), body_quat_w=quat, body_ang_vel_w=np.zeros_like(body)),
    )
    return MotionDataset(dataset=source, body_names=["pelvis"], joint_names=G1_JOINTS,
                         src_joint_indices=list(range(29)), dest_joint_indices=list(range(29)))


def tracker_yaml(backend: str) -> dict:
    """The Reference part of the Scalecar Tracker's policy.yaml (scripts/export_scalecar_tracker_onnx.py)."""
    return {
        "model_path": "unused.onnx",
        "observation": {
            "ref_command": {"sonic_command_multi_future_nonflat": {
                "_target_": "sonic.sonic_command_multi_future_nonflat", "future_steps": OFFSETS,
                "joint_names": TRACKER_JOINTS}},
            "ref_pelvis_pos": {"ref_body_pos_future_local": {
                "_target_": "mimic_lite.ref_body_pos_future_local", "future_steps": OFFSETS,
                "body_names": ["pelvis"], "anchor_body_name": "pelvis"}},
        },
        "joint_names_simulation": G1_JOINTS,
        "body_names_simulation": ["pelvis"],
        "policy_joint_names": TRACKER_JOINTS,
        "default_joint_pos": {".*": 0.0},
        "joint_kp": {".*": 40.0},
        "joint_kd": {".*": 2.0},
        "action_scale": 0.5,
        "motion": {"motion_backend": backend, "motion_path": "take.npz", "future_steps": OFFSETS,
                   "joint_names": TRACKER_JOINTS, "body_names": ["pelvis"], "root_body_name": "pelvis"},
    }


@pytest.fixture
def operator_rig(tmp_path, monkeypatch):
    """``make(backend, controller)`` -> a Tracking runtime with faked boundaries, ticked by ``rig.tick``."""

    def make(backend: str = "planner", controller: str = "keyboard", **args):
        clock, robot = Clock(), StandingG1()
        camera, planner = Camera(clock), Planner()
        device = Keyboard() if controller == "keyboard" else Joystick()
        monkeypatch.setattr(base_policy_module, "create_robot_io", lambda **_: robot)
        monkeypatch.setattr(base_policy_module, "build_inference_module",
                            lambda *_: (lambda inputs: {"action": np.zeros(29, dtype=np.float32)}))
        monkeypatch.setattr(base_policy_module, "KeyboardController", lambda: device)
        monkeypatch.setattr(base_policy_module, "UnitreeJoystickController", lambda *_: device)
        monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: motion_dataset()))

        class RobotTracking(Tracking):
            def _planner_io(self, options):
                return PlannerIO(source=camera, worker=planner, clock=lambda: clock.now)

        path = tmp_path / f"{backend}.yaml"
        path.write_text(yaml.safe_dump(tracker_yaml(backend)))
        tracker = RobotTracking(args=TrackingArgs(policy_config=str(path), controller=controller, **args))
        tracker.state_dict = {"action": np.zeros(29, dtype=np.float32), "paused": True, "control_mode": "zero"}

        def tick(ticks: int = 1) -> None:
            for _ in range(ticks):
                tracker.step()
                clock.tick += 1

        def key(name: str, ticks: int = 1) -> None:
            device.press(name)
            tick(ticks)

        return SimpleNamespace(tracker=tracker, clock=clock, robot=robot, planner=planner, tick=tick, key=key,
                               reference=getattr(tracker, "motion_buffer", None))

    return make


def mode(rig) -> str:
    return rig.tracker.state_dict["control_mode"]


def said(lines, text: str) -> int:
    return sum(text in message for _, message in lines)


A, R1, B, R2 = "i", "]", " ", "o"  # the keyboard's A / R1 / B / R2


def test_keys_follow_the_planner_table_from_shaping_to_e_stop_and_back(operator_rig, messages):
    rig = operator_rig()
    reference, provider = rig.reference, rig.reference.provider
    rig.tick(2)  # streaming state; the warmup plan comes back

    rig.key(B)  # B before the Tracker takes over: no Go
    assert reference.go_time is None and not reference.go_requested
    assert said(messages, "Go ignored")

    rig.key(A, ticks=5)  # A: Shaping
    assert mode(rig) == "init"
    rig.key(R1)  # R1: the Tracker takes over, Reference frozen at frame 0
    assert mode(rig) == "policy" and reference.holding and reference.phase == "standing"

    go_tick = rig.clock.now
    rig.key(B)  # B: Go
    assert reference.go_time == go_tick and reference.phase == "waiting_first_chunk"
    rig.tick(3)
    assert reference.phase == "running" and not reference.holding
    generation, requests = provider.generation, len(rig.planner.submitted)

    rig.key(B)  # B again: no effect
    rig.tick(2)
    assert reference.go_time == go_tick and provider.generation == generation
    assert reference.phase == "running" and len(rig.planner.submitted) >= requests
    assert said(messages, "B ignored") == 1 and not said(messages, "Go refused")

    rig.key(R2)  # R2: E-stop drops the plan
    assert mode(rig) == "zero"
    assert reference.holding and reference.phase == "standing" and reference.go_time is None
    assert provider.generation == generation + 1 and provider.go is None
    rig.tick(10)
    assert reference.holding  # results of the dropped generation are ignored

    rig.key(B)  # B under E-stop: no Go
    assert reference.go_time is None
    rig.key(A, ticks=5)
    rig.key(R1)
    restart = rig.clock.now
    rig.key(B)  # A -> R1 -> B: a new Go
    assert reference.go_time == restart and provider.generation == generation + 2
    rig.tick(3)
    assert reference.phase == "running"


def test_go_is_refused_while_the_height_check_fails_and_can_be_retried(operator_rig, messages):
    rig = operator_rig()
    rig.tick(2)
    rig.key(A, ticks=5)
    rig.key(R1)
    rig.robot.left_foot_lifted = True

    rig.key(B)
    rig.tick(3)
    assert rig.reference.go_time is None and rig.reference.holding and rig.reference.phase == "standing"
    refusal = [m for level, m in messages if level == "WARNING" and "Go refused" in m]
    assert len(refusal) == 1 and "pelvis height check failed" in refusal[0]

    rig.robot.left_foot_lifted = False
    rig.key(B)
    assert rig.reference.go_time is not None and rig.reference.provider.initialization["height_m"] == pytest.approx(
        0.7785, abs=2e-3)


def test_taking_over_again_or_shaping_while_planning_also_returns_to_hold(operator_rig):
    rig = operator_rig()
    rig.tick(2)
    rig.key(A, ticks=5)
    rig.key(R1)
    rig.key(B, ticks=4)
    assert rig.reference.phase == "running"

    rig.key(R1)  # R1 always means Hold at frame 0
    assert rig.reference.holding and rig.reference.go_time is None and mode(rig) == "policy"
    rig.key(B, ticks=4)
    assert rig.reference.phase == "running"

    rig.key(A)  # leaving the Tracker drops the plan too
    assert rig.reference.holding and rig.reference.go_time is None and mode(rig) == "init"


def test_the_joystick_b_is_go_and_r2_drops_it(operator_rig):
    rig = operator_rig(controller="joystick")
    rig.tick(2)
    rig.key("A", ticks=5)
    rig.key("R1")
    assert mode(rig) == "policy" and rig.reference.holding

    rig.key("B", ticks=4)
    assert rig.reference.go_time is not None and rig.reference.phase == "running"
    rig.key("R2")
    assert mode(rig) == "zero" and rig.reference.holding and rig.reference.go_time is None


def test_file_playback_b_still_toggles_play_and_pause(operator_rig):
    rig = operator_rig(backend="npz")
    tracker = rig.tracker
    rig.key(B)  # unchanged: B toggles even before the Tracker takes over
    assert tracker.state_dict["paused"] is False
    rig.key(A, ticks=3)
    rig.key(R1)  # taking over pauses at frame 0
    assert tracker.state_dict["paused"] is True and int(tracker.motion_t[0]) == 0

    rig.key(B, ticks=5)  # play
    assert tracker.state_dict["paused"] is False and int(tracker.motion_t[0]) == 5
    rig.key(B, ticks=5)  # pause
    assert tracker.state_dict["paused"] is True and int(tracker.motion_t[0]) == 5
    rig.key(R2)
    rig.key(B, ticks=2)  # unchanged: B still toggles under E-stop
    assert tracker.state_dict["paused"] is False


# ---------------------------------------------------------------------------
# The robot's Planner IO: depth receiver, CUDA worker, Host monotonic clock
# ---------------------------------------------------------------------------


def test_on_the_robot_the_planner_reads_depth_through_the_host_receiver(tmp_path, monkeypatch):
    import sim2real.planner.depth as depth_module
    import sim2real.planner.worker as worker_module

    created = {}

    class Receiver:
        def __init__(self, options, output):
            created["depth"] = (options, output)

        def poll(self):
            return None

        def close(self):
            pass

    class Worker:
        def __init__(self, options):
            created["worker"] = options

        def poll(self):
            return None

        def close(self):
            pass

    monkeypatch.setattr(depth_module, "DepthSource", Receiver)
    monkeypatch.setattr(worker_module, "GpuPlannerWorker", Worker)
    monkeypatch.setattr(base_policy_module, "create_robot_io", lambda **_: StandingG1())
    monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: (lambda inputs: inputs))
    monkeypatch.setattr(base_policy_module, "KeyboardController", Keyboard)
    monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: motion_dataset()))
    config = tracker_yaml("planner")
    config["motion"]["depth"] = {"port": 18800, "local_port": 18801, "log_dir": str(tmp_path / "yaml_logs")}
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))

    tracker = Tracking(args=TrackingArgs(
        policy_config=str(path), depth_ssh_host="g1-onboard", depth_remote_executable="/opt/depth/scalecar-depth",
        depth_log_dir=str(tmp_path / "cli_logs"),
    ))

    options, output = created["depth"]
    assert options == {"ssh_host": "g1-onboard", "remote_executable": "/opt/depth/scalecar-depth",
                       "port": 18800, "local_port": 18801, "cpu": None}
    assert output == tmp_path / "cli_logs" and output.is_dir()
    assert created["worker"]["checkpoint_sha256"] and created["worker"]["device"] == "cuda:0"
    assert tracker._planner_clock is time.monotonic


# ---------------------------------------------------------------------------
# Startup errors leave no child process behind (review finding 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def recorded_planner_io(tmp_path, monkeypatch):
    """The robot's depth receiver and CUDA worker replaced by recorders of start and close."""
    import sim2real.planner.depth as depth_module
    import sim2real.planner.worker as worker_module

    events: list[str] = []

    class Receiver:
        def __init__(self, options, output):
            events.append("depth started")

        def poll(self):
            return None

        def close(self):
            events.append("depth closed")

    class Worker:
        fail = False

        def __init__(self, options):
            if Worker.fail:
                raise RuntimeError("could not spawn the Planner process")
            events.append("worker started")

        def poll(self):
            return None

        def close(self):
            events.append("worker closed")

    monkeypatch.setattr(depth_module, "DepthSource", Receiver)
    monkeypatch.setattr(worker_module, "GpuPlannerWorker", Worker)
    monkeypatch.setattr(base_policy_module, "create_robot_io", lambda **_: StandingG1())
    monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: (lambda inputs: inputs))
    monkeypatch.setattr(base_policy_module, "KeyboardController", Keyboard)
    monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: motion_dataset()))

    def build(config: dict) -> Tracking:
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(config))
        return Tracking(args=TrackingArgs(policy_config=str(path), depth_log_dir=str(tmp_path / "logs")))

    return SimpleNamespace(events=events, worker=Worker, build=build)


def _other_future_steps(config: dict) -> None:
    config["motion"]["future_steps"] = [0, 2, 4, 8, 16]
    for group in config["observation"].values():
        for obs in group.values():
            obs["future_steps"] = [0, 2, 4, 8, 16]


@pytest.mark.parametrize(
    "misconfigure",
    [
        _other_future_steps,
        lambda config: config.update(shaping_target="frame0"),
        lambda config: config["observation"]["ref_command"]["sonic_command_multi_future_nonflat"].update(
            _target_="sonic.no_such_observation"),
        lambda config: config["motion"].update(planner={"precision": "fp16"}),
    ],
    ids=["future_steps", "shaping_target", "observation", "planner_option"],
)
def test_a_startup_error_starts_no_planner_process(recorded_planner_io, misconfigure):
    config = tracker_yaml("planner")
    misconfigure(config)

    with pytest.raises(Exception):
        recorded_planner_io.build(config)

    assert recorded_planner_io.events == []


def test_a_worker_that_fails_to_start_closes_the_depth_receiver(recorded_planner_io):
    recorded_planner_io.worker.fail = True

    with pytest.raises(RuntimeError, match="could not spawn"):
        recorded_planner_io.build(tracker_yaml("planner"))

    assert recorded_planner_io.events == ["depth started", "depth closed"]


def test_a_started_planner_is_closed_with_the_tracker(recorded_planner_io):
    tracker = recorded_planner_io.build(tracker_yaml("planner"))
    assert recorded_planner_io.events == ["depth started", "worker started"]

    tracker.close_reference_source()

    assert recorded_planner_io.events[2:] == ["depth closed", "worker closed"]
