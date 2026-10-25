"""Vehicle sim2sim runner (dixiatielu/sim2real_scalecar#6).

Seams under test: the Entry/Fall judgement rules, the Task Envelope guard, the
physics configuration and the File Playback clock/flow. The acceptance episodes
themselves are runs of the CLI, not unit tests.

    env -u PYTHONPATH HF_HUB_OFFLINE=1 UV_FROZEN=1 uv run --with pytest \
        python -m pytest tests/test_vehicle_sim2sim.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("MUJOCO_GL", "egl")  # the Planner's depth camera renders offscreen

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy import base_policy as base_policy_module
from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.tracking import TrackingArgs
from sim2real.rl_policy.utils.motion import MotionDataset
from sim2real.sim_env.vehicle_sim2sim import (
    EntryJudge,
    EnvelopeGuard,
    FilePlaybackFlow,
    PlannerFlow,
    SceneRobotIO,
    VehiclePhysics,
    VehicleSim2SimArgs,
    VehicleTracking,
    main,
    motion_start_state,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# vehicle.json `cabin_feet_region` (world frame, metres).
CABIN = np.array([[-1.0842757067, -0.7060332567, 0.18011796100000002], [0.1657242933, 0.0539667433, 0.700117961]])
DT = 0.02
UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


def seated(**overrides):
    """A pose with both feet and the pelvis inside the cabin region, upright."""
    pose = dict(
        pelvis_pos=np.array([-0.09, -0.32, 0.66]),
        pelvis_quat_wxyz=UPRIGHT,
        torso_pos=np.array([-0.09, -0.32, 0.95]),
        left_foot_pos=np.array([-0.55, -0.20, 0.25]),
        right_foot_pos=np.array([-0.55, -0.45, 0.25]),
    )
    pose.update(overrides)
    return pose


def standing_outside(**overrides):
    """Standing next to the car: feet and pelvis outside the cabin region."""
    pose = dict(
        pelvis_pos=np.array([-0.22, -0.95, 0.79]),
        pelvis_quat_wxyz=UPRIGHT,
        torso_pos=np.array([-0.22, -0.95, 1.05]),
        left_foot_pos=np.array([-0.22, -0.85, 0.03]),
        right_foot_pos=np.array([-0.22, -1.05, 0.03]),
    )
    pose.update(overrides)
    return pose


def tilted(degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2
    return np.array([math.cos(half), math.sin(half), 0.0, 0.0])


def run(judge: EntryJudge, poses) -> EntryJudge:
    for pose in poses:
        judge.update(**pose)
    return judge


# ---------------------------------------------------------------------------
# Judgement: Entry, Fall, Task Success (Scalecar simulation/entry.py rules)
# ---------------------------------------------------------------------------


def test_entry_needs_0p75_s_continuously_inside_the_cabin():
    judge = run(EntryJudge(CABIN, DT), [standing_outside()] * 10 + [seated()] * 37)
    assert not judge.entered

    judge.update(**seated())

    assert judge.entered
    assert judge.entry_time_s == pytest.approx(0.96)  # 48 ticks after the judge started


def test_leaving_the_cabin_restarts_the_entry_count():
    judge = run(EntryJudge(CABIN, DT), [seated()] * 30 + [standing_outside()] + [seated()] * 37)
    assert not judge.entered

    judge.update(**seated())

    assert judge.entry_time_s == pytest.approx(1.38)  # 30 + 1 + 38 ticks


@pytest.mark.parametrize(
    "pose",
    [
        seated(pelvis_pos=np.array([-0.09, -0.32, 0.34])),  # pelvis too low to count as seated
        seated(pelvis_pos=np.array([-0.09, -0.32, 1.61])),  # pelvis too high
        seated(pelvis_pos=np.array([-0.09, 0.10, 0.66])),  # pelvis y past the region
        seated(left_foot_pos=np.array([-0.55, -0.20, 0.15])),  # one foot below the region
    ],
)
def test_pelvis_and_both_feet_must_be_inside(pose):
    judge = run(EntryJudge(CABIN, DT), [pose] * 100)

    assert not judge.entered


def test_pelvis_z_outside_the_region_box_still_counts_when_within_height_band():
    # Only the pelvis xy is checked against the box; its height has its own band.
    judge = run(EntryJudge(CABIN, DT), [seated(pelvis_pos=np.array([-0.09, -0.32, 0.90]))] * 38)

    assert judge.entered


@pytest.mark.parametrize(
    "falling",
    [
        standing_outside(pelvis_pos=np.array([-0.22, -0.95, 0.24])),  # pelvis below 0.25 m
        standing_outside(torso_pos=np.array([-0.22, -0.95, 0.34])),  # torso below 0.35 m
        standing_outside(pelvis_quat_wxyz=tilted(76.0)),  # pelvis tilted past 75 degrees
    ],
)
def test_fall_needs_0p2_s_in_a_row(falling):
    judge = run(EntryJudge(CABIN, DT), [standing_outside()] * 5 + [falling] * 9)
    assert not judge.fallen

    judge.update(**falling)

    assert judge.fallen
    assert judge.fall_time_s == pytest.approx(0.30)  # 5 + 10 ticks


def test_tilt_up_to_75_degrees_is_not_a_fall():
    judge = run(EntryJudge(CABIN, DT), [standing_outside(pelvis_quat_wxyz=tilted(74.0))] * 50)

    assert not judge.fallen


def test_a_short_stumble_is_not_a_fall():
    stumble = standing_outside(pelvis_pos=np.array([-0.22, -0.95, 0.20]))
    judge = run(EntryJudge(CABIN, DT), ([stumble] * 9 + [standing_outside()]) * 5)

    assert not judge.fallen


def test_no_entry_after_a_fall():
    slumped = seated(torso_pos=np.array([-0.09, -0.32, 0.30]))  # inside, but torso collapsed
    judge = run(EntryJudge(CABIN, DT), [slumped] * 10 + [seated()] * 100)

    assert judge.fallen
    assert not judge.entered


# ---------------------------------------------------------------------------
# Task Envelope guard
# ---------------------------------------------------------------------------

# A free-floating "robot" (pelvis sphere + one hinged arm capsule) next to a static "car" hull.
ENVELOPE_SCENE = """
<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="vehicle_carbody">
      <geom name="hull" type="box" size="0.1 0.1 0.1" pos="1.95 0 0.5"/>
    </body>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="floating_base_joint"/>
      <geom name="pelvis_collision" type="sphere" size="0.07"/>
      <geom name="pelvis_visual" type="sphere" size="0.5" contype="0" conaffinity="0"/>
      <body name="arm_link">
        <joint name="arm_joint" type="hinge" axis="0 1 0"/>
        <geom name="arm_collision" type="capsule" fromto="0 0 0 0.5 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
ENVELOPE_MIN = np.array([-2.0, -2.0, -0.3])
ENVELOPE_MAX = np.array([2.0, 2.0, 2.0])


def envelope_rig(root_x=0.0, arm_angle=0.0):
    model = mujoco.MjModel.from_xml_string(ENVELOPE_SCENE)
    data = mujoco.MjData(model)
    data.qpos[0] = root_x
    data.qpos[7] = arm_angle
    mujoco.mj_forward(model, data)
    return model, data


def test_envelope_margin_is_the_closest_robot_collision_box_to_the_boundary():
    model, data = envelope_rig()
    guard = EnvelopeGuard(model, ENVELOPE_MIN, ENVELOPE_MAX)

    guard.check(data, time_s=0.0)

    # Pelvis sphere top at z = 1.07 is 0.93 m below the envelope top; the 0.5 m
    # visual sphere and the static car hull 5 cm from x max do not count.
    assert guard.valid
    assert guard.min_margin_m == pytest.approx(0.93, abs=1e-9)
    assert guard.closest["geom"] == "pelvis_collision"


def test_envelope_uses_world_boxes_of_rotated_geoms():
    model, data = envelope_rig(arm_angle=-math.pi / 2)  # arm capsule now points straight up
    guard = EnvelopeGuard(model, ENVELOPE_MIN, ENVELOPE_MAX)

    guard.check(data, time_s=0.0)

    assert guard.min_margin_m == pytest.approx(2.0 - 1.53, abs=1e-9)
    assert guard.closest["body"] == "arm_link"


def test_robot_within_10cm_of_the_boundary_makes_the_episode_invalid():
    model, data = envelope_rig()
    guard = EnvelopeGuard(model, ENVELOPE_MIN, ENVELOPE_MAX)
    guard.check(data, time_s=0.0)

    data.qpos[0] = 2.0 - 0.53 - 0.05  # arm tip 5 cm from x max
    mujoco.mj_forward(model, data)
    guard.check(data, time_s=1.5)
    data.qpos[0] = 0.0  # coming back does not make the episode valid again
    mujoco.mj_forward(model, data)
    guard.check(data, time_s=1.52)

    assert not guard.valid
    assert guard.first_violation["time_s"] == 1.5
    assert guard.first_violation["geom"] == "arm_collision"
    assert guard.first_violation["margin_m"] == pytest.approx(0.05, abs=1e-9)


# ---------------------------------------------------------------------------
# Physics: Scalecar's d1146 MuJoCo setup (BE8, position actuators, contract values)
# ---------------------------------------------------------------------------

# A three-joint stand-in for the Scalecar G1: motor actuators named like their joints,
# declared in an order that differs from both the robot order and the contract order.
PHYSICS_SCENE = """
<mujoco>
  <option timestep="0.005" integrator="implicitfast" solver="Newton" iterations="100" tolerance="1e-8" cone="elliptic"/>
  <worldbody>
    <geom name="ground" type="plane" size="5 5 0.1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.05" mass="5"/>
      <body name="b_link" pos="0 0.1 0">
        <joint name="b_joint" axis="0 1 0" armature="0.01" frictionloss="0.1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="2"/>
        <body name="c_link" pos="0 0 -0.4">
          <joint name="c_joint" axis="0 1 0" frictionloss="0.1"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="1.5"/>
        </body>
      </body>
      <body name="a_link" pos="0 -0.1 0">
        <joint name="a_joint" axis="1 0 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="2"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="c_joint" joint="c_joint"/>
    <motor name="a_joint" joint="a_joint"/>
    <motor name="b_joint" joint="b_joint"/>
  </actuator>
</mujoco>
"""
ROBOT_JOINTS = ["a_joint", "b_joint", "c_joint"]  # robot (SDK-like) order
CONTRACT = {
    "joint_names": ["b_joint", "c_joint", "a_joint"],  # Tracker order
    "physics_dt": 0.005,
    "control_dt": 0.02,
    "stiffness": [40.0, 30.0, 20.0],
    "damping": [2.0, 1.5, 1.0],
    "effort_limit_sim": [50.0, 25.0, 4.0],
    "armature": [0.02, 0.01, 0.03],
}
START = dict(root_pos=[0.1, -0.2, 0.8], root_quat_wxyz=[0.99, 0.0, 0.1, 0.0], joint_pos=[0.3, -0.2, 0.5])
TARGETS = np.array([1.2, 0.6, -0.4])  # robot order; a_joint saturates at its 4 N·m limit


def scalecar_reference_steps(ticks: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Scalecar's run_episode setup and tick, with its own configure_actuators."""
    path = REPO_ROOT / "scripts/scalecar/vendor/scalecar_deploy/simulation/physics.py"
    spec = importlib.util.spec_from_file_location("scalecar_physics", path)
    scalecar_physics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scalecar_physics)

    model = mujoco.MjModel.from_xml_string(PHYSICS_SCENE)
    model.opt.timestep = CONTRACT["physics_dt"] / 8
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in CONTRACT["joint_names"]]
    mapper = SimpleNamespace(
        qpos=model.jnt_qposadr[joints],
        qvel=model.jnt_dofadr[joints],
        actuators=np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in CONTRACT["joint_names"]]),
    )
    model.dof_armature[mapper.qvel] = CONTRACT["armature"]
    model.dof_frictionloss[mapper.qvel] = 0.0
    kp, kd = np.asarray(CONTRACT["stiffness"]), np.asarray(CONTRACT["damping"])
    scalecar_physics.configure_actuators(model, mapper, kp, kd, CONTRACT, drive_integration="backward_euler")
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    data.qpos[:3] = START["root_pos"]
    data.qpos[3:7] = START["root_quat_wxyz"]
    data.qpos[mapper.qpos] = [START["joint_pos"][ROBOT_JOINTS.index(n)] for n in CONTRACT["joint_names"]]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    target = np.array([TARGETS[ROBOT_JOINTS.index(n)] for n in CONTRACT["joint_names"]])
    states = []
    for _ in range(ticks):
        for _ in range(round(CONTRACT["control_dt"] / model.opt.timestep)):
            data.ctrl[mapper.actuators] = target
            scalecar_physics.step_physics(model, data, mapper, kp, kd, drive_integration="backward_euler")
        mujoco.mj_forward(model, data)
        states.append((data.qpos.copy(), data.qvel.copy()))
    return states


def test_vehicle_physics_steps_exactly_like_scalecar():
    physics = VehiclePhysics(mujoco.MjModel.from_xml_string(PHYSICS_SCENE), CONTRACT, joint_names=ROBOT_JOINTS)
    physics.reset(**START)
    states = []
    for _ in range(10):
        physics.step(TARGETS)
        states.append((physics.data.qpos.copy(), physics.data.qvel.copy()))

    reference = scalecar_reference_steps(10)
    for (qpos, qvel), (ref_qpos, ref_qvel) in zip(states, reference, strict=True):
        assert qpos.tobytes() == ref_qpos.tobytes()
        assert qvel.tobytes() == ref_qvel.tobytes()
    assert physics.model.opt.timestep == 0.000625
    assert physics.time_s == pytest.approx(0.2)



def test_robot_transport_reads_the_scene_in_robot_joint_order():
    physics = VehiclePhysics(mujoco.MjModel.from_xml_string(PHYSICS_SCENE), CONTRACT, joint_names=ROBOT_JOINTS)
    physics.reset(**START)
    robot = SceneRobotIO(physics)

    state = robot.read_state()

    np.testing.assert_array_equal(state.qpos[:3], START["root_pos"])
    np.testing.assert_array_equal(state.qpos[3:7], START["root_quat_wxyz"])
    np.testing.assert_array_equal(state.qpos[7:], START["joint_pos"])
    assert state.qvel.shape == (6 + 3,)


def test_robot_transport_hands_position_targets_to_the_scene():
    physics = VehiclePhysics(mujoco.MjModel.from_xml_string(PHYSICS_SCENE), CONTRACT, joint_names=ROBOT_JOINTS)
    robot = SceneRobotIO(physics)
    kp, kd = np.array([20.0, 40.0, 30.0]), np.array([1.0, 2.0, 1.5])  # contract gains in robot order

    robot.write_command(TARGETS, np.zeros(3), np.zeros(3), kp, kd)

    np.testing.assert_array_equal(robot.joint_targets, TARGETS)
    with pytest.raises(ValueError, match="gains"):
        robot.write_command(TARGETS, np.zeros(3), np.zeros(3), kp * 2, kd)
    with pytest.raises(ValueError, match="feed-forward"):
        robot.write_command(TARGETS, np.zeros(3), np.ones(3), kp, kd)



def test_start_state_is_motion_frame_0_unclipped_with_joints_matched_by_name(tmp_path):
    # An any4hdmi dataset whose qpos joint order differs from the robot order.
    (tmp_path / "motions").mkdir()
    names = ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz", "c_joint", "a_joint", "b_joint"]
    (tmp_path / "manifest.json").write_text(
        '{"format_version": 2, "motions_subdir": "motions", "qpos_names": %s}' % str(names).replace("'", '"')
    )
    qpos = np.array(
        [[-0.215925, -0.953545, 0.792668, 0.7, 0.01, -0.02, 0.71, 0.3, -0.0873, 1.6], [0.0] * 10], dtype=np.float32
    )
    np.savez(tmp_path / "motions" / "take.npz", qpos=qpos)

    start = motion_start_state(tmp_path / "motions" / "take.npz", joint_names=ROBOT_JOINTS)

    np.testing.assert_array_equal(start["root_pos"], qpos[0, :3])
    np.testing.assert_array_equal(start["root_quat_wxyz"], qpos[0, 3:7])
    np.testing.assert_array_equal(start["joint_pos"], np.array([-0.0873, 1.6, 0.3], dtype=np.float32))


# ---------------------------------------------------------------------------
# Clock and flow: Hold 5 s with the Tracker running, Go, File Playback, 2 s after the end
# ---------------------------------------------------------------------------


def drive(flow: FilePlaybackFlow, *, reference_frames: int, max_ticks: int = 10_000):
    """Runs the flow against a File Playback stand-in that ends `reference_frames` ticks after Go."""
    inputs, go_tick = {}, None
    while not flow.done and flow.tick < max_ticks:
        tick = flow.tick
        command = flow.operator_input()
        if command is not None:
            inputs[tick] = command
        if command == "go":
            go_tick = tick
        ended = go_tick is not None and tick - go_tick >= reference_frames - 1
        flow.tick_done(reference_ended=ended)
    return inputs


def test_tracker_takes_over_at_once_and_go_follows_5_s_of_hold():
    flow = FilePlaybackFlow(control_dt=0.02, hold_s=5.0, final_hold_s=2.0)

    inputs = drive(flow, reference_frames=779)

    assert inputs == {0: "take_over", 250: "go"}
    assert flow.go_time_s == pytest.approx(5.0)


def test_episode_ends_2_s_after_the_reference_ends():
    flow = FilePlaybackFlow(control_dt=0.02, hold_s=5.0, final_hold_s=2.0)

    drive(flow, reference_frames=779)

    # Scalecar's entry-only episode for take1512: 250 Hold ticks + 779 frames + 100 ticks.
    assert flow.tick == 1129
    assert flow.reference_end_time_s == pytest.approx(5.0 + 779 * 0.02)
    assert flow.stage == "done"


def drive_planner(flow: PlannerFlow, *, first_chunk_after: int | None, refuse_go: bool = False):
    """Runs the Planner flow against a Planner whose first Chunk applies `first_chunk_after` ticks after Go."""
    inputs, go_tick = {}, None
    while not flow.done and flow.tick < 10_000:
        tick = flow.tick
        command = flow.operator_input()
        if command is not None:
            inputs[tick] = command
        if command == "go":
            go_tick = tick
        started = (
            go_tick is not None and not refuse_go and first_chunk_after is not None and tick - go_tick >= first_chunk_after
        )
        flow.tick_done(go_refused=refuse_go and command == "go", reference_started=started)
    return inputs


def test_planner_episode_goes_after_5_s_of_hold_and_runs_40_s_from_go():
    flow = PlannerFlow(control_dt=0.02, hold_s=5.0, planner_s=40.0)

    inputs = drive_planner(flow, first_chunk_after=3)

    assert inputs == {0: "take_over", 250: "go"}
    assert flow.go_time_s == pytest.approx(5.0)
    assert flow.first_reference_time_s == pytest.approx(5.06)
    # Scalecar's DP episode: 250 Hold ticks, then dp.seconds = 40 s of control after Go.
    assert flow.tick == 2250 and flow.stage == "done" and not flow.go_refused


def test_planner_episode_holds_until_the_first_chunk():
    flow = PlannerFlow(control_dt=0.02, hold_s=5.0, planner_s=40.0)
    drive_planner(flow, first_chunk_after=None)

    assert flow.first_reference_time_s is None and flow.tick == 2250


def test_a_refused_go_ends_the_planner_episode():
    flow = PlannerFlow(control_dt=0.02, hold_s=5.0, planner_s=40.0)

    drive_planner(flow, first_chunk_after=3, refuse_go=True)

    assert flow.go_refused and flow.done and flow.tick == 251 and flow.go_time_s is None


@pytest.mark.parametrize(
    ("flow", "timing"),
    [
        (FilePlaybackFlow, dict(hold_s=0.0)),
        (FilePlaybackFlow, dict(hold_s=0.019)),
        (FilePlaybackFlow, dict(hold_s=-5.0)),
        (FilePlaybackFlow, dict(final_hold_s=-0.02)),
        (PlannerFlow, dict(hold_s=0.0)),
        (PlannerFlow, dict(hold_s=0.01)),
        (PlannerFlow, dict(planner_s=-40.0)),
    ],
)
def test_a_hold_shorter_than_one_tick_or_a_negative_duration_is_rejected(flow, timing):
    with pytest.raises(ValueError, match="_s must be"):
        flow(control_dt=0.02, **timing)


def test_a_one_tick_hold_goes_at_the_second_tick():
    playback = FilePlaybackFlow(control_dt=0.02, hold_s=0.02, final_hold_s=0.0)
    assert drive(playback, reference_frames=3) == {0: "take_over", 1: "go"}
    assert playback.go_time_s == pytest.approx(0.02) and playback.tick == 4

    planner = PlannerFlow(control_dt=0.02, hold_s=0.02, planner_s=0.0)
    assert drive_planner(planner, first_chunk_after=0) == {0: "take_over", 1: "go"}
    assert planner.go_time_s == pytest.approx(0.02) and planner.done


@pytest.fixture
def cli_inputs(tmp_path):
    """The files the CLI checks before it starts any episode (contents never read past control_dt)."""
    tracker = tmp_path / "tracker" / "ft500ema"
    tracker.mkdir(parents=True)
    (tracker / "policy.yaml").write_text("{}\n")
    (tracker / "deployment.json").write_text('{"control_dt": 0.02}\n')
    motions = tmp_path / "motions" / "motions"
    motions.mkdir(parents=True)
    (motions / "take1512.npz").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize(
    "timing",
    [dict(hold_s=0.01), dict(final_hold_s=-1.0), dict(reference_source="planner", planner_s=-1.0)],
)
def test_the_cli_rejects_bad_episode_timing_before_running_anything(cli_inputs, timing):
    args = VehicleSim2SimArgs(motions=["take1512"], artifact_root=str(cli_inputs), **timing)

    with pytest.raises(ValueError, match="_s must be"):
        main(args)

    assert not (cli_inputs / "sim2sim").exists()


@pytest.mark.parametrize("workers", [0, -1])
def test_the_cli_rejects_fewer_than_one_worker(cli_inputs, workers):
    with pytest.raises(ValueError, match="--workers"):
        VehicleSim2SimArgs(motions=["take1512"], artifact_root=str(cli_inputs), workers=workers)


# ---------------------------------------------------------------------------
# Tracker adapter: sim2real's Tracking runtime, stepped by the runner
# ---------------------------------------------------------------------------

G1_JOINTS = list(get_robot_cfg("g1").joint_names)
NUM_FRAMES = 5


class StandingRobot(RobotIO):
    """Robot transport stand-in: a fixed measured state; records the joint commands."""

    def __init__(self) -> None:
        self.commands: list[np.ndarray] = []

    def read_state(self) -> RobotState:
        qpos = np.zeros(7 + len(G1_JOINTS))
        qpos[3] = 1.0
        return RobotState(qpos=qpos, qvel=np.zeros(6 + len(G1_JOINTS)), joint_torque=np.zeros(len(G1_JOINTS)), tick=0)

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        self.commands.append(np.array(q_target, dtype=np.float64))


def counting_motion() -> MotionDataset:
    """Motion frame k has every joint at 0.1·k rad."""
    frames = np.repeat(0.1 * np.arange(NUM_FRAMES)[:, None], len(G1_JOINTS), axis=1)
    body = np.zeros((NUM_FRAMES, 1, 3))
    source = SimpleNamespace(
        num_motions=1,
        starts=np.array([0]),
        ends=np.array([NUM_FRAMES]),
        data=SimpleNamespace(
            joint_pos=frames,
            joint_vel=np.full_like(frames, 5.0),
            body_pos_w=body,
            body_lin_vel_w=body.copy(),
            body_quat_w=np.tile([1.0, 0.0, 0.0, 0.0], (NUM_FRAMES, 1, 1)),
            body_ang_vel_w=body.copy(),
        ),
    )
    return MotionDataset(
        dataset=source,
        body_names=["pelvis"],
        joint_names=G1_JOINTS,
        src_joint_indices=list(range(len(G1_JOINTS))),
        dest_joint_indices=list(range(len(G1_JOINTS))),
    )


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    """VehicleTracking over a stand-in robot, a zero-action network and a 5-frame Motion.

    ``network_s`` is how long the network takes per tick (0 by default)."""
    import time

    timing = SimpleNamespace(network_s=0.0)

    def network(input_dict):
        time.sleep(timing.network_s)
        return {"action": np.zeros(len(G1_JOINTS), dtype=np.float32)}

    monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: network)
    monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: counting_motion()))
    config = {
        "model_path": "unused.onnx",
        "observation": {},
        "joint_names_simulation": G1_JOINTS,
        "body_names_simulation": ["pelvis"],
        "policy_joint_names": G1_JOINTS,
        "default_joint_pos": {".*": 0.0},
        "joint_kp": {".*": 40.0},
        "joint_kd": {".*": 2.0},
        "action_scale": 0.5,
        "motion": {"motion_backend": "npz", "future_steps": [0, 2], "root_body_name": "pelvis"},
    }
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(config))
    robot = StandingRobot()
    tracker = VehicleTracking(TrackingArgs(policy_config=str(path), motion_path="take.npz"), robot_io=robot)
    return SimpleNamespace(tracker=tracker, robot=robot, timing=timing)


def reference_frames(tracker) -> np.ndarray:
    """The Motion frame index of every future step the Tracker is currently given."""
    return np.round(tracker.motion_data.joint_pos[0, :, 0] / 0.1).astype(int)


def test_hold_freezes_the_reference_at_frame_0_with_the_tracker_running(tracking):
    tracker = tracking.tracker

    tracker.take_over()
    for _ in range(3):
        tracker.step()

    assert tracker.state_dict["control_mode"] == "policy"
    assert len(tracking.robot.commands) == 3
    assert reference_frames(tracker).tolist() == [0, 0]
    assert not tracker.motion_data.joint_vel.any()
    assert not tracker.reference_ended


def test_go_plays_the_motion_and_reports_its_end(tracking):
    tracker = tracking.tracker
    tracker.take_over()
    tracker.step()

    tracker.go()
    seen = []
    for _ in range(NUM_FRAMES):
        assert not tracker.reference_ended
        tracker.step()
        seen.append(reference_frames(tracker).tolist())

    assert seen[:2] == [[1, 3], [2, 4]]
    assert tracker.reference_ended  # Final Hold: frozen at the last frame
    assert seen[-1] == [NUM_FRAMES - 1, NUM_FRAMES - 1]


def test_the_runner_drops_exactly_the_runtimes_wall_budget_warning(tracking):
    """The filter matches BasePolicy's "RL step took" warning by logger and text; this
    fails if that warning changes, so the filter cannot silently stop working."""
    from loguru import logger

    from sim2real.sim_env.vehicle_sim2sim import _WallBudgetFilter

    budget, warned, kept = _WallBudgetFilter(), [], []
    sinks = [
        logger.add(lambda m: warned.append(m.record["message"]), level="WARNING"),
        logger.add(lambda m: kept.append(m.record["message"]), level="WARNING", filter=budget),
    ]
    tracking.timing.network_s = 0.03  # over the 20 ms budget of a 50 Hz tick
    try:
        tracking.tracker.take_over()
        tracking.tracker.step()
    finally:
        for sink in sinks:
            logger.remove(sink)

    assert len(warned) == 1  # the runtime reported the slow tick
    assert budget.suppressed == 1 and kept == []


# ---------------------------------------------------------------------------
# Planner depth: the head D435i rendered from the physics model (dixiatielu/sim2real_scalecar#7)
# ---------------------------------------------------------------------------

SCALECAR = Path("/home/galbot/projects/Scalecar_v2")
CAMERA_SCENE = """
<mujoco>
  <worldbody>
    <geom name="ground" type="plane" size="6 6 0.1"/>
    <geom name="wall" type="box" pos="3.5 0 1" size="0.1 3 1"/>
    <geom name="post" type="cylinder" pos="1.2 0.4 0.5" size="0.08 0.5"/>
    <body name="torso_link" pos="0 0 0.6">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" group="3" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def camera_scene():
    model = mujoco.MjModel.from_xml_string(CAMERA_SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_depth_camera_sees_the_ground_along_its_optical_axis():
    from sim2real.planner.sim import DepthCamera

    model, data = camera_scene()
    camera = DepthCamera(model)
    try:
        depth = camera.render(data)
    finally:
        camera.close()

    assert depth.shape == (480, 640) and depth.dtype == np.uint16
    # Camera 0.6 + 0.42987 m above the ground, optical axis pitched 0.8307767 rad down:
    # the axis meets the ground at z = 1.02987 / sin(0.8307767) = 1.3946 m.
    assert depth[238, 325] == pytest.approx(1394.6, abs=3)


@pytest.mark.skipif(not SCALECAR.is_dir(), reason="Scalecar_v2 checkout not present")
def test_depth_camera_renders_like_scalecars():
    import importlib
    import sys

    from sim2real.planner.sim import DepthCamera

    sys.dont_write_bytecode = True
    for path in (SCALECAR / "source/scalecar_dp", SCALECAR / "source/scalecar_deploy"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    ScalecarCamera = importlib.import_module("scalecar_deploy.simulation.depth").DepthCamera

    model, data = camera_scene()
    data.qpos[:3] = [0.1, -0.2, 0.62]
    data.qpos[3:7] = [0.9887711, 0.0099833, 0.0, 0.1490712]  # turned and slightly rolled
    mujoco.mj_forward(model, data)
    ours, theirs = DepthCamera(model), ScalecarCamera(model)
    try:
        np.testing.assert_array_equal(ours.render(data), theirs.render(data))
    finally:
        ours.close()
        theirs.close()


def test_simulated_depth_arrives_at_30_hz_of_simulation_time_downsampled():
    from sim2real.planner.scalecar_dp.dp.depth_preprocess import downsample_z16
    from sim2real.planner.sim import DepthCamera, SimulatedDepthSource

    model, data = camera_scene()
    clock = SimpleNamespace(now=0.0)
    source = SimulatedDepthSource(model, data, clock=lambda: clock.now)
    packets = []
    try:
        for tick in range(50):
            clock.now = tick * DT
            packet = source.poll()
            if packet is not None:
                packets.append(packet)
        reference = DepthCamera(model)
        expected = downsample_z16(reference.render(data))
        reference.close()
    finally:
        source.close()

    assert len(packets) == 30
    assert [p["frame"] for p in packets] == list(range(30))
    assert {p["session"] for p in packets} == {0}
    # Frame k is rendered at the first tick at or after k/30 s.
    np.testing.assert_allclose([p["acquired"] for p in packets][:5], [0.0, 0.04, 0.08, 0.10, 0.14])
    np.testing.assert_array_equal(packets[-1]["depth_mm"], expected)


# ---------------------------------------------------------------------------
# Planner startup in the vehicle sim2sim leaves no child process behind (review finding 2)
# ---------------------------------------------------------------------------

PLANNER_OFFSETS = [0, 2, 4, 8, 16, 32]


def planner_tracker_yaml(**extra) -> dict:
    from sim2real.planner.provider import JOINT_NAMES

    config = {
        "model_path": "unused.onnx",
        "observation": {
            "ref_command": {"sonic_command_multi_future_nonflat": {
                "_target_": "sonic.sonic_command_multi_future_nonflat", "future_steps": PLANNER_OFFSETS,
                "joint_names": list(JOINT_NAMES)}},
        },
        "joint_names_simulation": G1_JOINTS,
        "body_names_simulation": ["pelvis"],
        "policy_joint_names": list(JOINT_NAMES),
        "default_joint_pos": {".*": 0.0},
        "joint_kp": {".*": 40.0},
        "joint_kd": {".*": 2.0},
        "action_scale": 0.5,
        "motion": {"motion_backend": "planner", "future_steps": PLANNER_OFFSETS, "root_body_name": "pelvis"},
    }
    config.update(extra)
    return config


@pytest.fixture
def planner_startup(tmp_path, monkeypatch):
    """VehicleTracking with the Planner, its rendered depth source and CUDA worker replaced by recorders."""
    import sim2real.planner.sim as sim_module
    import sim2real.planner.worker as worker_module

    events: list[str] = []

    class DepthSource:
        def __init__(self, model, data, *, clock):
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

        def submit(self, item):
            return True

        def poll(self):
            return None

        def close(self):
            events.append("worker closed")

    monkeypatch.setattr(sim_module, "SimulatedDepthSource", DepthSource)
    monkeypatch.setattr(worker_module, "GpuPlannerWorker", Worker)
    monkeypatch.setattr(base_policy_module, "build_inference_module", lambda *_: (lambda inputs: inputs))
    monkeypatch.setattr(MotionDataset, "create_from_path", classmethod(lambda cls, *_, **__: counting_motion()))
    physics = SimpleNamespace(model=None, data=None, time_s=0.0)

    def build(config: dict) -> VehicleTracking:
        path = tmp_path / "policy.yaml"
        path.write_text(yaml.safe_dump(config))
        args = TrackingArgs(policy_config=str(path), motion_backend="planner", motion_path="take.npz")
        return VehicleTracking(args, robot_io=StandingRobot(), physics=physics)

    return SimpleNamespace(events=events, worker=Worker, build=build)


def test_a_planner_episode_with_a_startup_error_starts_no_planner_process(planner_startup):
    with pytest.raises(ValueError, match="shaping_target"):
        planner_startup.build(planner_tracker_yaml(shaping_target="frame0"))

    assert planner_startup.events == []


def test_a_planner_worker_that_fails_to_start_closes_the_depth_source(planner_startup):
    planner_startup.worker.fail = True

    with pytest.raises(RuntimeError, match="could not spawn"):
        planner_startup.build(planner_tracker_yaml())

    assert planner_startup.events == ["depth started", "depth closed"]


def test_closing_a_planner_episode_closes_its_depth_source_and_worker(planner_startup):
    tracker = planner_startup.build(planner_tracker_yaml())

    tracker.close_reference_source()

    assert planner_startup.events == ["depth started", "worker started", "depth closed", "worker closed"]
