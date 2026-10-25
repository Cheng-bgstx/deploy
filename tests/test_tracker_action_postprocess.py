"""Tracker action post-processing (soft clip + target rate limit) and the Shaping target.

The seam is the policy runtime's control tick: a scripted network action and a
scripted measured robot state go in, and the joint command written to the robot
comes out. The ONNX module, the robot transport and the operator Controller are
replaced by small fakes; everything between them is the real runtime.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy import base_policy as base_policy_module
from sim2real.rl_policy.base_policy import BasePolicy, BasePolicyArgs
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.rl_policy.utils.motion import MotionDataset
from sim2real.sim_env import integrated_sim2sim

G1_JOINTS = list(get_robot_cfg("g1").joint_names)
# Tracker (Isaac) joint order, as in Scalecar's deployment.json `joint_names`.
TRACKER_TO_SDK = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
TRACKER_JOINTS = [G1_JOINTS[i] for i in TRACKER_TO_SDK]

KNEE = G1_JOINTS.index("left_knee_joint")
HIP_PITCH = G1_JOINTS.index("left_hip_pitch_joint")
ELBOW = G1_JOINTS.index("right_elbow_joint")


class FakeRobot(RobotIO):
    def __init__(self) -> None:
        self.joint_pos = np.zeros(len(G1_JOINTS))
        self.commands: list[np.ndarray] = []

    def read_state(self) -> RobotState:
        qpos = np.zeros(7 + len(G1_JOINTS))
        qpos[3] = 1.0
        qpos[7:] = self.joint_pos
        return RobotState(
            qpos=qpos,
            qvel=np.zeros(6 + len(G1_JOINTS)),
            joint_torque=np.zeros(len(G1_JOINTS)),
            tick=len(self.commands),
        )

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        self.commands.append(np.array(q_target, dtype=np.float64, copy=True))


class ScriptedController(ControllerBase):
    name = "test"

    def __init__(self) -> None:
        self.key_pressed: set[str] = set()
        self.requested_mode = None

    def get_control_mode(self):
        mode, self.requested_mode = self.requested_mode, None
        return mode


class ScriptedNetwork:
    """Stands in for the ONNX module: returns whatever action the test sets (Tracker order)."""

    def __init__(self) -> None:
        self.action = np.zeros(len(TRACKER_JOINTS), dtype=np.float32)

    def __call__(self, input_dict):
        return {"action": self.action.copy()}


def tracker_action(**by_joint: float) -> np.ndarray:
    action = np.zeros(len(TRACKER_JOINTS), dtype=np.float32)
    for name, value in by_joint.items():
        action[TRACKER_JOINTS.index(f"{name}_joint")] = value
    return action


def g1_vector(**by_joint: float) -> np.ndarray:
    values = np.zeros(len(G1_JOINTS))
    for name, value in by_joint.items():
        values[G1_JOINTS.index(f"{name}_joint")] = value
    return values


def policy_config(**extra) -> dict:
    """A Tracker-shaped policy YAML: G1 robot order, Tracker policy order, action scale 0.5."""
    config = {
        "model_path": "unused.onnx",
        "observation": {},
        "joint_names_simulation": G1_JOINTS,
        "body_names_simulation": ["pelvis"],
        "policy_joint_names": TRACKER_JOINTS,
        "default_joint_pos": {name: 0.0 for name in TRACKER_JOINTS},
        "joint_kp": {".*": 40.0},
        "joint_kd": {".*": 2.0},
        "action_scale": {name: 0.5 for name in TRACKER_JOINTS},
    }
    config.update(extra)
    return config


def soft_limits(**overrides: tuple[float, float]) -> dict:
    """Name-keyed soft limits in Tracker order; every joint [-1, 1] unless overridden."""
    limits = {name: [-1.0, 1.0] for name in TRACKER_JOINTS}
    for name, bounds in overrides.items():
        limits[f"{name}_joint"] = list(bounds)
    return limits


class Rig:
    def __init__(self, policy, robot: FakeRobot, controller: ScriptedController, network: ScriptedNetwork):
        self.policy = policy
        self.robot = robot
        self.controller = controller
        self.network = network

    def tick(self, *, mode=None, action=None, measured=None) -> np.ndarray:
        if mode is not None:
            self.controller.requested_mode = mode
        if action is not None:
            self.network.action = np.asarray(action, dtype=np.float32)
        if measured is not None:
            self.robot.joint_pos = np.asarray(measured, dtype=np.float64)
        count = len(self.robot.commands)
        self.policy.step()
        assert len(self.robot.commands) == count + 1, "the tick did not send a command"
        return self.robot.commands[-1]


@pytest.fixture
def make_rig(tmp_path, monkeypatch):
    def make(config: dict, *, policy_cls=BasePolicy, args_cls=BasePolicyArgs, **args) -> Rig:
        robot, controller, network = FakeRobot(), ScriptedController(), ScriptedNetwork()
        monkeypatch.setattr(base_policy_module, "create_robot_io", lambda **_: robot)
        monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: network)
        monkeypatch.setattr(base_policy_module, "KeyboardController", lambda: controller)
        config_path = tmp_path / "policy.yaml"
        config_path.write_text(yaml.safe_dump(config))
        policy = policy_cls(args=args_cls(policy_config=str(config_path), **args))
        return Rig(policy, robot, controller, network)

    return make


def test_soft_clip_limits_policy_targets(make_rig):
    rig = make_rig(policy_config(soft_joint_pos_limits=soft_limits(left_knee=(-0.0131, 2.8056))))

    command = rig.tick(mode="policy", action=tracker_action(left_knee=-1.0, left_hip_pitch=4.0, right_elbow=1.0))

    np.testing.assert_array_equal(command, g1_vector(left_knee=-0.0131, left_hip_pitch=1.0, right_elbow=0.5))


def test_rate_limit_moves_targets_at_most_0p3_rad_per_step(make_rig):
    rig = make_rig(policy_config(max_target_velocity_rad_s=15.0), rl_rate=50.0)

    action = tracker_action(left_hip_pitch=4.0, left_knee=-2.0, right_elbow=0.2)
    commands = [rig.tick(mode="policy" if step == 0 else None, action=action) for step in range(8)]

    hip = [command[HIP_PITCH] for command in commands]
    knee = [command[KNEE] for command in commands]
    elbow = [command[ELBOW] for command in commands]
    np.testing.assert_allclose(hip, [0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0, 2.0], atol=1e-12)
    np.testing.assert_allclose(knee, [-0.3, -0.6, -0.9, -1.0, -1.0, -1.0, -1.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(elbow, [0.1] * 8, atol=1e-7)
    steps = np.diff(np.stack([np.zeros(len(G1_JOINTS))] + commands), axis=0)
    assert np.abs(steps).max() <= 0.3 + 1e-12


def test_entering_policy_mode_starts_rate_limit_from_measured_q(make_rig):
    rig = make_rig(policy_config(max_target_velocity_rad_s=15.0), rl_rate=50.0)
    hold_default = tracker_action()  # raw target = default pose (all zeros)

    rig.tick(mode="zero", measured=g1_vector(left_knee=1.2, right_elbow=0.5))
    first = rig.tick(mode="policy", action=hold_default)
    second = rig.tick()
    # Measured q passes through float32 in StateProcessor, hence atol=1e-6.
    np.testing.assert_allclose(first, g1_vector(left_knee=0.9, right_elbow=0.2), atol=1e-6)
    np.testing.assert_allclose(second, g1_vector(left_knee=0.6), atol=1e-6)

    # E-stop: targets follow measured q, untouched by post-processing.
    estop = rig.tick(mode="zero", measured=g1_vector(left_knee=-0.5))
    np.testing.assert_allclose(estop, g1_vector(left_knee=-0.5), atol=1e-6)
    rig.tick(measured=g1_vector(left_knee=-0.7, left_hip_pitch=0.4))

    # Re-entering the Tracker restarts from the newly measured q, not from the last target (0.6).
    reentry = rig.tick(mode="policy")
    np.testing.assert_allclose(reentry, g1_vector(left_knee=-0.4, left_hip_pitch=0.1), atol=1e-6)


def test_init_and_zero_modes_are_not_post_processed(make_rig):
    rig = make_rig(
        policy_config(
            soft_joint_pos_limits=soft_limits(left_knee=(-0.0131, 2.8056)),
            max_target_velocity_rad_s=15.0,
        ),
        rl_rate=50.0,
    )
    rig.network.action = tracker_action(left_knee=-2.0, left_hip_pitch=4.0)
    measured = g1_vector(left_knee=-0.0873, left_hip_pitch=1.8)

    # E-stop follows measured q even below the soft lower limit.
    estop = rig.tick(mode="zero", measured=measured)
    np.testing.assert_allclose(estop, measured, atol=1e-6)

    # Shaping (default target: default_joint_pos = zeros) interpolates without clip or rate limit.
    shaping = [rig.tick(mode="init" if step == 0 else None) for step in range(3)]
    for count, command in enumerate(shaping):
        np.testing.assert_allclose(command, measured * (1 - count / 500), atol=1e-6)


def motion_dataset(first_frame: np.ndarray, num_frames: int = 3) -> MotionDataset:
    """A one-Motion dataset whose frame 0 is ``first_frame`` (G1 order); later frames differ.

    The source joint order is the Tracker order, reordered by name the way
    ``MotionDataset.create_from_path`` does.
    """
    frames = np.repeat(first_frame[None, TRACKER_TO_SDK], num_frames, axis=0)
    frames[1:] += 0.25
    body = np.zeros((num_frames, 1, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1, 1))
    source = SimpleNamespace(
        num_motions=1,
        starts=np.array([0]),
        ends=np.array([num_frames]),
        data=SimpleNamespace(
            joint_pos=frames,
            joint_vel=np.zeros_like(frames),
            body_pos_w=body,
            body_lin_vel_w=body.copy(),
            body_quat_w=quat,
            body_ang_vel_w=body.copy(),
        ),
    )
    return MotionDataset(
        dataset=source,
        body_names=["pelvis"],
        joint_names=G1_JOINTS,
        src_joint_indices=list(range(len(TRACKER_JOINTS))),
        dest_joint_indices=[G1_JOINTS.index(name) for name in TRACKER_JOINTS],
    )


@pytest.fixture
def motion_files(monkeypatch):
    """Maps Motion paths to in-memory datasets for ``MotionDataset.create_from_path``."""
    files: dict[str, MotionDataset] = {}
    monkeypatch.setattr(
        MotionDataset,
        "create_from_path",
        classmethod(lambda cls, root_path, **_: files[str(root_path)]),
    )
    return files


# Motion frame 0 in the style of the 1146 data: knee at the hard lower limit, elbow ~1.5.
MOTION_FRAME0 = g1_vector(
    left_hip_pitch=0.17,
    left_knee=-0.0873,
    right_knee=0.3,
    right_elbow=1.49,
    left_wrist_pitch=1.6,
)
SHAPING_SOFT_LIMITS = soft_limits(
    left_knee=(-0.0131, 2.8056),
    right_knee=(-0.0131, 2.8056),
    right_elbow=(-0.9687, 2.0159),
    left_wrist_pitch=(-1.5337, 1.5337),
)
# Frame 0 clipped to the soft limits: knee -0.0873 -> -0.0131, wrist pitch 1.6 -> 1.5337.
SHAPING_TARGET = g1_vector(
    left_hip_pitch=0.17,
    left_knee=-0.0131,
    right_knee=0.3,
    right_elbow=1.49,
    left_wrist_pitch=1.5337,
)


def tracking_config(**extra) -> dict:
    defaults = {name: 0.0 for name in TRACKER_JOINTS}
    defaults["right_knee_joint"] = 0.669  # Tracker default differs from frame 0 (0.3)
    return policy_config(
        default_joint_pos=defaults,
        motion={"motion_backend": "npz", "future_steps": [0], "root_body_name": "pelvis"},
        **extra,
    )


def run_shaping(rig: Rig, measured: np.ndarray, ticks: int) -> list[np.ndarray]:
    return [
        rig.tick(mode="init" if step == 0 else None, measured=measured if step == 0 else None)
        for step in range(ticks)
    ]


def test_shaping_targets_motion_first_frame_within_soft_limits(make_rig, motion_files):
    motion_files["take1512"] = motion_dataset(MOTION_FRAME0)
    rig = make_rig(
        tracking_config(
            soft_joint_pos_limits=SHAPING_SOFT_LIMITS,
            max_target_velocity_rad_s=15.0,
            shaping_target="motion_first_frame",
        ),
        policy_cls=Tracking,
        args_cls=TrackingArgs,
        motion_path="take1512",
    )
    measured = g1_vector(left_knee=1.0, right_elbow=0.2, waist_yaw=-0.1)

    commands = run_shaping(rig, measured, ticks=503)

    # Same interpolation as before (count/500 over 500 ticks), only the end point changed.
    np.testing.assert_allclose(commands[0], measured, atol=1e-6)
    np.testing.assert_allclose(commands[250], measured + (SHAPING_TARGET - measured) * 0.5, atol=1e-6)
    for command in commands[500:]:
        np.testing.assert_allclose(command, SHAPING_TARGET, atol=1e-6)


def test_planner_mode_takes_shaping_target_from_the_given_motion(make_rig, motion_files):
    """With a streamed Reference (zmq backend) the Motion file only supplies frame 0."""
    motion_files["take0595"] = motion_dataset(MOTION_FRAME0)
    rig = make_rig(
        tracking_config(soft_joint_pos_limits=SHAPING_SOFT_LIMITS, shaping_target="motion_first_frame"),
        policy_cls=Tracking,
        args_cls=TrackingArgs,
        motion_backend="zmq",
        motion_zmq_connect="tcp://127.0.0.1:1",
        motion_path="take0595",
    )

    commands = run_shaping(rig, g1_vector(left_knee=1.0), ticks=502)

    np.testing.assert_allclose(commands[-1], SHAPING_TARGET, atol=1e-6)



def make_integrated_runtime(tmp_path, monkeypatch, config: dict):
    network = ScriptedNetwork()
    monkeypatch.setattr(integrated_sim2sim, "build_inference_module", lambda *_: network)
    config_path = tmp_path / "policy.yaml"
    config_path.write_text(yaml.safe_dump(config))
    args = integrated_sim2sim.IntegratedSim2SimArgs(policy_config=str(config_path), motion_path="take1512")
    runtime = integrated_sim2sim.IntegratedPolicyRuntime(args=args, robot_cfg=get_robot_cfg("g1"))
    return runtime, network


def start_playback(runtime, measured: np.ndarray) -> None:
    """What IntegratedSim2Sim._reset_playback does: robot at frame 0, policy mode, reset."""
    runtime.state_dict = {"action": np.zeros(runtime.num_actions, dtype=np.float32), "paused": True, "control_mode": "policy"}
    runtime.state_processor.restart_motion()
    runtime.state_processor.joint_pos[:] = measured
    runtime.reset()
    runtime.state_dict["control_mode"] = "policy"


def test_integrated_sim2sim_post_processes_policy_targets(tmp_path, monkeypatch, motion_files):
    motion_files["take1512"] = motion_dataset(MOTION_FRAME0)
    runtime, network = make_integrated_runtime(
        tmp_path,
        monkeypatch,
        tracking_config(soft_joint_pos_limits=SHAPING_SOFT_LIMITS, max_target_velocity_rad_s=15.0),
    )
    network.action = tracker_action(left_knee=-1.0, left_hip_pitch=4.0)
    start_playback(runtime, MOTION_FRAME0)

    commands = [runtime.step()[0] for _ in range(4)]

    # Knee: soft lower limit reached from the measured -0.0873 in one step (0.0742 < 0.3).
    np.testing.assert_allclose([c[KNEE] for c in commands], [-0.0131] * 4, atol=1e-6)
    # Hip pitch: from measured 0.17 towards the soft upper limit 1.0 at 0.3 rad per step.
    np.testing.assert_allclose([c[HIP_PITCH] for c in commands], [0.47, 0.77, 1.0, 1.0], atol=1e-6)

    # A new playback restarts the rate limit from the newly measured q.
    start_playback(runtime, g1_vector(left_hip_pitch=-0.5))
    np.testing.assert_allclose(runtime.step()[0][HIP_PITCH], -0.2, atol=1e-6)


def test_without_the_new_keys_targets_are_unchanged(make_rig, motion_files):
    motion_files["take1512"] = motion_dataset(MOTION_FRAME0)
    rig = make_rig(tracking_config(), policy_cls=Tracking, args_cls=TrackingArgs, motion_path="take1512")
    measured = g1_vector(left_knee=-0.0873, left_hip_pitch=1.8)

    np.testing.assert_allclose(rig.tick(mode="zero", measured=measured), measured, atol=1e-6)

    # Policy: default + 0.5 * action, no clip and no rate limit, from the first tick on.
    action = tracker_action(left_knee=-3.0, left_hip_pitch=8.0, right_elbow=1.0)
    for step in range(2):
        command = rig.tick(mode="policy" if step == 0 else None, action=action)
        np.testing.assert_array_equal(command, g1_vector(left_knee=-1.5, left_hip_pitch=4.0, right_knee=0.669, right_elbow=0.5))

    # Shaping still goes to default_joint_pos.
    shaping = run_shaping(rig, measured, ticks=501)
    np.testing.assert_allclose(shaping[-1], g1_vector(right_knee=0.669), atol=1e-12)


def _without(limits: dict, joint: str) -> dict:
    return {name: bounds for name, bounds in limits.items() if name != joint}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"shaping_target": "motion_first_frame"}, "soft_joint_pos_limits"),
        ({"shaping_target": "frame0"}, "shaping_target"),
        ({"soft_joint_pos_limits": _without(soft_limits(), "left_knee_joint")}, "left_knee_joint"),
        ({"soft_joint_pos_limits": soft_limits(left_knee=(0.5, -0.5))}, "left_knee_joint"),
        ({"max_target_velocity_rad_s": 0.0}, "max_target_velocity_rad_s"),
    ],
)
def test_misconfigured_post_processing_is_rejected_at_startup(make_rig, motion_files, extra, message):
    motion_files["take1512"] = motion_dataset(MOTION_FRAME0)
    with pytest.raises(ValueError, match=message):
        make_rig(tracking_config(**extra), policy_cls=Tracking, args_cls=TrackingArgs, motion_path="take1512")


# ---------------------------------------------------------------------------
# Non-finite inputs never latch the rate limit (review finding 1)
# ---------------------------------------------------------------------------

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from loguru import logger  # noqa: E402

from sim2real.rl_policy.utils.joint_targets import JointTargetPostprocessor  # noqa: E402

SCALECAR = Path("/home/galbot/projects/Scalecar_v2")


@pytest.fixture
def warnings_logged():
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="WARNING")
    yield lines
    logger.remove(sink)


def test_a_non_finite_action_keeps_the_previous_target_and_does_not_latch(make_rig, warnings_logged):
    rig = make_rig(policy_config(max_target_velocity_rad_s=15.0), rl_rate=50.0)
    moving = tracker_action(left_hip_pitch=4.0, left_knee=1.0)

    first = rig.tick(mode="policy", action=moving)
    held = [rig.tick(action=tracker_action(left_hip_pitch=bad, left_knee=1.0)) for bad in (np.nan, np.inf, -np.inf)]
    resumed = [rig.tick(action=moving) for _ in range(2)]

    # The bad joint keeps its last finite target; the others carry on.
    assert [c[HIP_PITCH] for c in [first, *held, *resumed]] == pytest.approx([0.3, 0.3, 0.3, 0.3, 0.6, 0.9])
    assert [c[KNEE] for c in [first, *held, *resumed]] == pytest.approx([0.3, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert all(np.isfinite(c).all() for c in [first, *held, *resumed])
    # One warning naming the joint, not one per tick.
    assert len(warnings_logged) == 1 and "left_hip_pitch_joint" in warnings_logged[0]


def test_a_non_finite_measured_position_on_entry_does_not_latch(make_rig, warnings_logged):
    rig = make_rig(policy_config(max_target_velocity_rad_s=15.0), rl_rate=50.0)
    broken = g1_vector(left_hip_pitch=0.4)
    broken[KNEE] = np.nan
    rig.tick(mode="zero", measured=broken)

    first = rig.tick(mode="policy", action=tracker_action(left_knee=1.0))
    second = rig.tick(action=tracker_action(left_knee=2.0), measured=g1_vector(left_knee=0.1, left_hip_pitch=0.4))

    # No finite position to start from: the knee's target passes unlimited once, then the
    # rate limit runs from it. The other joints start from their measured positions.
    assert first[KNEE] == pytest.approx(0.5) and second[KNEE] == pytest.approx(0.8)
    assert first[HIP_PITCH] == pytest.approx(0.1, abs=1e-6) and second[HIP_PITCH] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(first).all() and np.isfinite(second).all()
    assert len(warnings_logged) == 1 and "left_knee_joint" in warnings_logged[0]


@pytest.mark.skipif(not SCALECAR.is_dir(), reason="Scalecar_v2 checkout not present")
def test_finite_targets_are_bitwise_scalecars_position_controller():
    sys.dont_write_bytecode = True
    path = str(SCALECAR / "source/scalecar_deploy")
    if path not in sys.path:
        sys.path.append(path)
    from scalecar_deploy.control import PositionController

    rng = np.random.default_rng(7)
    n = len(TRACKER_JOINTS)
    lower = rng.uniform(-2.5, -0.01, n)
    contract = {
        "default_joint_pos": rng.uniform(-0.5, 0.5, n).tolist(),
        "action_scale": rng.uniform(0.1, 0.6, n).tolist(),
        "soft_joint_pos_limits": np.stack([lower, lower + rng.uniform(0.02, 3.0, n)], 1).tolist(),
        "stiffness": [1.0] * n, "damping": [1.0] * n, "effort_limit_sim": [1.0] * n,
        "max_target_velocity_rad_s": 15.0, "control_dt": 0.02,
    }
    theirs = PositionController(contract)
    ours = JointTargetPostprocessor.from_policy_config(
        {"soft_joint_pos_limits": {name: bound for name, bound in zip(TRACKER_JOINTS, contract["soft_joint_pos_limits"])},
         "max_target_velocity_rad_s": 15.0},
        joint_names=TRACKER_JOINTS, policy_joint_names=TRACKER_JOINTS, control_dt=0.02,
    )
    default, scale = np.asarray(contract["default_joint_pos"]), np.asarray(contract["action_scale"])
    for entry in range(3):
        measured = rng.uniform(-1.5, 1.5, n).astype(np.float32).astype(np.float64)
        theirs.reset(measured)
        ours.reset()
        for tick in range(200):
            action = (rng.standard_normal(n) * (4.0 if tick % 7 else 0.3)).astype(np.float32)
            raw = default.copy()
            raw += action * scale  # BasePolicy: default + action * action_scale
            expected = theirs.target(action)
            np.testing.assert_array_equal(ours(raw, measured), expected, err_msg=f"entry {entry} tick {tick}")
