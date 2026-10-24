"""Vehicle sim2sim: sim2real's Tracker runtime in Scalecar's Vehicle Scene (ADR-0004).

One episode per (Tracker, Motion) case, each in its own process::

    HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 UV_FROZEN=1 uv run python -m sim2real.sim_env.vehicle_sim2sim \
        --trackers ft500ema ft1000ema --motions take1512 take0595

Inputs and results default to ``checkpoints/scalecar_d1146/`` of this checkout
(``--artifact-root`` points elsewhere, e.g. from a git worktree). Physics, scene
and judgement follow Scalecar's d1146 entry-only File Playback episode; the
Tracker side is sim2real's own runtime (``tracking.py``). Each case writes
``<case>.json`` (result), ``<case>.rollout.npz`` and ``<case>.log`` to the
output directory, plus one ``summary.json`` per invocation.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.robot_io import RobotIO, RobotState
from sim2real.rl_policy.tracking import Tracking, TrackingArgs

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "checkpoints" / "scalecar_d1146"
JUDGE_BODIES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
ROOT_QPOS_NAMES = ["root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz"]

ENTRY_HOLD_S = 0.75
FALL_HOLD_S = 0.20
ENVELOPE_THRESHOLD_M = 0.10
PHYSICS_SUBSTEPS = 8


class EntryJudge:
    """Entry and Fall, with Scalecar's thresholds (``simulation/entry.py``).

    Inside: both feet inside the cabin region, the pelvis xy inside it and the
    pelvis height in [0.35, 1.60] m. Entry: inside for 0.75 s in a row. Fall:
    pelvis below 0.25 m, torso below 0.35 m or pelvis tilt above 75 degrees,
    for 0.2 s in a row. Times count from the judge's first update.
    """

    def __init__(self, cabin_region: np.ndarray, control_dt: float):
        self.region = np.asarray(cabin_region, dtype=np.float64)
        if self.region.shape != (2, 3) or not np.isfinite(self.region).all():
            raise ValueError("cabin_region must be a finite [[x, y, z], [x, y, z]] box")
        self.control_dt = float(control_dt)
        self._entry_ticks = math.ceil(ENTRY_HOLD_S / self.control_dt)
        self._fall_ticks = math.ceil(FALL_HOLD_S / self.control_dt)
        self._inside_count = 0
        self._fall_count = 0
        self.ticks = 0
        self.entered = False
        self.entry_time_s: float | None = None
        self.fallen = False
        self.fall_time_s: float | None = None

    def update(self, *, pelvis_pos, pelvis_quat_wxyz, torso_pos, left_foot_pos, right_foot_pos) -> None:
        lo, hi = self.region
        feet = np.stack((left_foot_pos, right_foot_pos))
        pelvis = np.asarray(pelvis_pos, dtype=np.float64)
        inside = bool(
            np.all((feet >= lo) & (feet <= hi))
            and np.all((pelvis[:2] >= lo[:2]) & (pelvis[:2] <= hi[:2]))
            and 0.35 <= pelvis[2] <= 1.60
        )
        _, x, y, _ = np.asarray(pelvis_quat_wxyz, dtype=np.float64)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        falling = bool(pelvis[2] < 0.25 or torso_pos[2] < 0.35 or up_z < math.cos(math.radians(75.0)))
        self.ticks += 1
        self._inside_count = self._inside_count + 1 if inside and not self.fallen else 0
        self._fall_count = self._fall_count + 1 if falling else 0
        if self._fall_count >= self._fall_ticks and not self.fallen:
            self.fallen = True
            self.fall_time_s = self.ticks * self.control_dt
        if self._inside_count >= self._entry_ticks and not self.fallen and not self.entered:
            self.entered = True
            self.entry_time_s = self.ticks * self.control_dt


def robot_body_ids(model: mujoco.MjModel) -> np.ndarray:
    """Bodies of the robot: the subtree under the single free joint."""
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError(f"Expected exactly one floating robot base, found {len(free)} free joints")
    root = int(model.jnt_bodyid[free[0]])
    bodies = [b for b in range(model.nbody) if _has_ancestor(model, b, root)]
    return np.asarray(bodies, dtype=int)


def _has_ancestor(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body:
        if body == ancestor:
            return True
        body = int(model.body_parentid[body])
    return False


class EnvelopeGuard:
    """Marks an episode invalid when the robot comes near the Task Envelope boundary.

    Every check takes the world AABB of each robot collision geom (its local
    ``geom_aabb`` box rotated into the world) and its distance to the envelope
    faces. A margin under ``threshold_m`` logs a warning once and makes the
    episode invalid for good: the cropped scene has no vehicle hulls out there.
    """

    def __init__(self, model: mujoco.MjModel, envelope_min, envelope_max, *, threshold_m: float = ENVELOPE_THRESHOLD_M):
        self.model = model
        self.envelope_min = np.asarray(envelope_min, dtype=np.float64)
        self.envelope_max = np.asarray(envelope_max, dtype=np.float64)
        self.threshold_m = float(threshold_m)
        robot = set(robot_body_ids(model).tolist())
        collides = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        self.geoms = np.asarray(
            [g for g in range(model.ngeom) if collides[g] and int(model.geom_bodyid[g]) in robot], dtype=int
        )
        if not len(self.geoms):
            raise ValueError("The robot has no collision geoms")
        self._center_local = model.geom_aabb[self.geoms, :3]
        self._half_local = model.geom_aabb[self.geoms, 3:]
        self.valid = True
        self.min_margin_m = math.inf
        self.closest: dict | None = None
        self.first_violation: dict | None = None

    def check(self, data: mujoco.MjData, *, time_s: float) -> float:
        rot = data.geom_xmat[self.geoms].reshape(-1, 3, 3)
        center = data.geom_xpos[self.geoms] + np.einsum("gij,gj->gi", rot, self._center_local)
        half = np.einsum("gij,gj->gi", np.abs(rot), self._half_local)
        margins = np.minimum(center - half - self.envelope_min, self.envelope_max - center - half).min(axis=1)
        index = int(np.argmin(margins))
        margin = float(margins[index])
        if margin < self.min_margin_m:
            self.min_margin_m = margin
            self.closest = self._describe(index, margin, time_s)
        if margin < self.threshold_m and self.valid:
            self.valid = False
            self.first_violation = self._describe(index, margin, time_s)
            logger.warning(
                "Invalid episode: left the Task Envelope. {geom} ({body}) is {margin_m:.3f} m from the "
                "boundary at t={time_s:.2f} s (threshold {threshold:.2f} m)",
                threshold=self.threshold_m,
                **self.first_violation,
            )
        return margin

    def _describe(self, index: int, margin: float, time_s: float) -> dict:
        geom = int(self.geoms[index])
        body = int(self.model.geom_bodyid[geom])
        return {
            "time_s": float(time_s),
            "margin_m": margin,
            "geom": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}",
            "body": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body),
        }


class VehiclePhysics:
    """The Vehicle Scene with Scalecar's d1146 MuJoCo physics, one control tick per ``step``.

    Mirrors ``run_episode`` with the d1146 run config (Scalecar
    ``mujoco_runtime.py:81-115``, ``simulation/physics.py:19-66``):
    ``timestep = physics_dt / 8``; contract armature, frictionloss 0; the
    model's motor actuators become position actuators with contract kp/kd,
    backward-Euler damping (``kd + kp·timestep``) and contract effort limits;
    ``mj_setConst``. A tick holds the target for ``control_dt`` worth of
    substeps, then runs ``mj_forward``. Joints and actuators are matched by
    name; ``joint_names`` is the robot order of ``reset``, ``step`` and ``robot_state``.
    """

    def __init__(self, model: mujoco.MjModel, contract: dict, *, joint_names, substeps: int = PHYSICS_SUBSTEPS):
        self.model = model
        self.joint_names = list(joint_names)
        contract_names = list(contract["joint_names"])
        if sorted(contract_names) != sorted(self.joint_names):
            raise ValueError("The deployment contract and the robot must name the same joints")
        model.opt.timestep = float(contract["physics_dt"]) / substeps
        self.control_dt = float(contract["control_dt"])
        self.decimation = round(self.control_dt / model.opt.timestep)
        if abs(self.decimation * model.opt.timestep - self.control_dt) > 1e-10:
            raise ValueError("The physics timestep must divide control_dt")

        joints = np.array([_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.joint_names])
        self.qpos_adr = model.jnt_qposadr[joints]
        self.qvel_adr = model.jnt_dofadr[joints]
        self.actuators = np.array([_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.joint_names])
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise ValueError("Expected exactly one floating robot base")
        self.root_qpos = int(model.jnt_qposadr[free[0]])
        self.root_qvel = int(model.jnt_dofadr[free[0]])

        # Contract arrays are in contract (Tracker) order; reorder to robot order by name.
        order = [contract_names.index(n) for n in self.joint_names]

        def per_joint(key: str) -> np.ndarray:
            return np.asarray(contract[key], dtype=np.float64)[order]

        self.kp, self.kd = per_joint("stiffness"), per_joint("damping")
        effort = per_joint("effort_limit_sim")
        model.dof_armature[self.qvel_adr] = per_joint("armature")
        model.dof_frictionloss[self.qvel_adr] = 0.0
        ids = self.actuators
        model.actuator_gaintype[ids] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[ids] = 0
        model.actuator_biasprm[ids] = 0
        model.actuator_gainprm[ids, 0] = self.kp
        model.actuator_biasprm[ids, 1] = -self.kp
        # Backward Euler: the kp·dt term makes the position spring implicit too.
        model.actuator_biasprm[ids, 2] = -self.kd - self.kp * model.opt.timestep
        model.actuator_ctrllimited[ids] = False
        model.actuator_forcelimited[ids] = True
        model.actuator_forcerange[ids] = np.stack((-effort, effort), axis=1)

        self.data = mujoco.MjData(model)
        # Armature changed after compile: rebuild the derived constants.
        mujoco.mj_setConst(model, self.data)

    @property
    def time_s(self) -> float:
        return float(self.data.time)

    def reset(self, *, root_pos, root_quat_wxyz, joint_pos) -> None:
        """Place the robot (joints in robot order) at rest and run ``mj_forward``."""
        r = self.root_qpos
        self.data.qpos[r : r + 3] = root_pos
        self.data.qpos[r + 3 : r + 7] = root_quat_wxyz
        self.data.qpos[self.qpos_adr] = joint_pos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def robot_state(self) -> RobotState:
        """Measured state in the runtime layout: root pose, world linear and body angular velocity, joints."""
        r, v = self.root_qpos, self.root_qvel
        data = self.data
        return RobotState(
            qpos=np.concatenate((data.qpos[r : r + 7], data.qpos[self.qpos_adr])),
            qvel=np.concatenate((data.qvel[v : v + 6], data.qvel[self.qvel_adr])),
            joint_torque=data.actuator_force[self.actuators].copy(),
            tick=int(data.time * 1e3),
        )

    def step(self, joint_targets) -> None:
        """Hold the position targets (robot order) for one control tick."""
        for _ in range(self.decimation):
            self.data.ctrl[self.actuators] = joint_targets
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)


def _name2id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise ValueError(f"The scene has no {kind.name.removeprefix('mjOBJ_').lower()} named {name!r}")
    return int(index)


class SceneRobotIO(RobotIO):
    """The Tracker runtime's robot transport, wired straight to the Vehicle Scene.

    The scene's position actuators carry the contract kp/kd, so a command is
    only accepted with exactly those gains and without velocity or torque
    feed-forward; anything else would be silently dropped otherwise.
    """

    def __init__(self, physics: VehiclePhysics):
        self.physics = physics
        self.joint_targets: np.ndarray | None = None
        self.commands_sent = 0

    def read_state(self) -> RobotState:
        return self.physics.robot_state()

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        if not (np.array_equal(kp, self.physics.kp) and np.array_equal(kd, self.physics.kd)):
            raise ValueError("Joint gains differ from the deployment contract's stiffness/damping")
        if np.any(dq_target) or np.any(tau_ff):
            raise ValueError("The scene's position actuators take no velocity or torque feed-forward")
        self.joint_targets = np.array(q_target, dtype=np.float64)
        self.commands_sent += 1


class FilePlaybackFlow:
    """The operator script and clock of one File Playback episode, in control ticks.

    Tick 0: the Tracker takes over (R1) with the Reference frozen at Motion
    frame 0 (Hold). After ``hold_s`` of Hold: Go (B). Once the Tracker reports
    that the Reference has ended (Final Hold), ``final_hold_s`` more, then done.
    With take1512 (779 frames) that is Scalecar's entry-only episode: 250 + 779
    + 100 ticks. Times are simulation time at the end of the tick.
    """

    def __init__(self, *, control_dt: float, hold_s: float = 5.0, final_hold_s: float = 2.0):
        self.control_dt = float(control_dt)
        self.hold_ticks = round(hold_s / self.control_dt)
        self.final_hold_ticks = round(final_hold_s / self.control_dt)
        self.tick = 0
        self.stage = "hold"
        self.go_time_s: float | None = None
        self.reference_end_time_s: float | None = None
        self._end_tick: int | None = None

    @property
    def done(self) -> bool:
        return self.stage == "done"

    def operator_input(self) -> str | None:
        """What the operator does at the start of the current tick: ``take_over``, ``go`` or nothing."""
        if self.tick == 0:
            return "take_over"
        if self.tick == self.hold_ticks:
            return "go"
        return None

    def tick_done(self, *, reference_ended: bool) -> None:
        if self.done:
            raise RuntimeError("The episode is over")
        if self.tick == self.hold_ticks:
            self.stage = "playback"
            self.go_time_s = self.tick * self.control_dt
        self.tick += 1
        if self.stage == "playback" and reference_ended:
            self.stage = "final_hold"
            self._end_tick = self.tick
            self.reference_end_time_s = self.tick * self.control_dt
        if self.stage == "final_hold" and self.tick - self._end_tick >= self.final_hold_ticks:
            self.stage = "done"


class ScriptedOperator(ControllerBase):
    """Stands in for the operator's Controller: a stage switch requested by the runner."""

    name = "vehicle_sim2sim"

    def __init__(self) -> None:
        self._requested: str | None = None

    def request(self, mode: str) -> None:
        self._requested = mode

    def get_control_mode(self):
        mode, self._requested = self._requested, None
        return mode


class VehicleTracking(Tracking):
    """sim2real's Tracking runtime, stepped by the runner with the robot in-process.

    ``step()`` is the unchanged runtime tick: Controller input, robot state,
    Reference, observations, Tracker, action post-processing, joint command.
    ``take_over`` and ``go`` are the operator's R1 and B.
    """

    def __init__(self, args: TrackingArgs, *, robot_io: RobotIO):
        self._vehicle_robot_io = robot_io
        self._operator = ScriptedOperator()
        self._went = False
        super().__init__(args=args)

    def _create_robot_io(self) -> RobotIO:
        return self._vehicle_robot_io

    def _build_controller(self) -> ControllerBase:
        self.keyboard_controller = self.joystick_controller = self.pico_controller = None
        return self._operator

    def take_over(self) -> None:
        """R1 at the next tick: the Tracker takes over with the Reference frozen at frame 0 (Hold)."""
        # The robot has been streaming state before R1, so the Tracker's history
        # restarts from the robot's current state, not from empty buffers.
        self.state_processor._prepare_low_state()
        self._operator.request("policy")

    def go(self) -> None:
        """B (File Playback): the Reference starts moving at the next tick."""
        if self.state_dict.get("control_mode") != "policy" or not self.state_dict.get("paused", False):
            raise RuntimeError("Go needs the Tracker in Hold")
        self.toggle_paused(source=self._operator.name)
        self._went = True

    @property
    def reference_ended(self) -> bool:
        """File Playback has reached the Motion's last frame (Final Hold)."""
        return self._went and bool(self.state_dict.get("paused", False))


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------


def motion_start_state(motion_path: Path, *, joint_names) -> dict:
    """Motion frame 0 as the start state: pelvis world pose and joints (robot order), unclipped.

    Read from the any4hdmi qpos (float32, exact) rather than the runtime's
    Motion loader, whose storage is float16. Joints are matched by name through
    the dataset's ``manifest.json``.
    """
    from any4hdmi.core.format import load_motion

    motion_path = Path(motion_path)
    manifest = json.loads((motion_path.parent.parent / "manifest.json").read_text(encoding="utf-8"))
    names = list(manifest["qpos_names"])
    if names[:7] != ROOT_QPOS_NAMES:
        raise ValueError(f"{motion_path}: qpos must start with the root pose {ROOT_QPOS_NAMES}")
    missing = [n for n in joint_names if n not in names]
    if missing:
        raise ValueError(f"{motion_path}: Motion lacks joints {missing}")
    frame0 = load_motion(motion_path)[0]
    return {
        "root_pos": frame0[:3],
        "root_quat_wxyz": frame0[3:7],
        "joint_pos": frame0[[names.index(n) for n in joint_names]],
    }


@dataclass(frozen=True)
class EpisodeCase:
    """One vehicle sim2sim episode: a Tracker (policy YAML next to its deployment.json) and a Motion."""

    name: str
    policy_config: Path
    motion: Path
    scene_dir: Path
    output_dir: Path
    hold_s: float = 5.0
    final_hold_s: float = 2.0

    @property
    def result_path(self) -> Path:
        return self.output_dir / f"{self.name}.json"


def load_vehicle_scene(scene_dir: Path) -> tuple[mujoco.MjModel, dict]:
    """The cropped Vehicle Scene MJB and its ``build.json`` (Task Envelope, region data)."""
    build = json.loads((Path(scene_dir) / "build.json").read_text(encoding="utf-8"))
    if mujoco.__version__ != build["mujoco_version"]:
        raise RuntimeError(f"The scene needs MuJoCo {build['mujoco_version']}, found {mujoco.__version__}")
    model = mujoco.MjModel.from_binary_path(str(Path(scene_dir) / build["scene"]["file"]))
    for key, value in build["scene"]["load"].items():
        if key != "vis.global_.bvactive":
            raise ValueError(f"Unknown scene load setting {key}")
        model.vis.global_.bvactive = value
    return model, build


def run_episode(case: EpisodeCase) -> dict:
    """Run one File Playback episode and return its result (also written to ``case.result_path``)."""
    wall_start = time.perf_counter()
    contract_path = case.policy_config.parent / "deployment.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    control_dt = float(contract["control_dt"])
    robot_cfg = get_robot_cfg("g1")

    model, build = load_vehicle_scene(case.scene_dir)
    physics = VehiclePhysics(model, contract, joint_names=robot_cfg.joint_names)
    physics.reset(**motion_start_state(case.motion, joint_names=robot_cfg.joint_names))
    robot_io = SceneRobotIO(physics)
    tracker = VehicleTracking(
        TrackingArgs(
            policy_config=str(case.policy_config),
            motion_backend="npz",
            motion_path=str(case.motion),
            rl_rate=1.0 / control_dt,
            inference_backend="onnx-cpu",
        ),
        robot_io=robot_io,
    )
    if tracker.policy_joint_names != contract["joint_names"]:
        raise ValueError("policy.yaml and deployment.json disagree on the Tracker joint order")

    region = np.asarray(build["task_envelope"]["sources"]["cabin_feet_region"])
    flow = FilePlaybackFlow(control_dt=control_dt, hold_s=case.hold_s, final_hold_s=case.final_hold_s)
    judge = EntryJudge(region, control_dt)  # Hold: only a Fall can happen; replaced at Go
    guard = EnvelopeGuard(model, build["task_envelope"]["min"], build["task_envelope"]["max"])
    bodies = {name: _name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in JUDGE_BODIES}
    rollout = {key: [] for key in ("time", "qpos", "qvel", "joint_target", "reference_frame", "margin")}

    termination = "completed"
    setup_s = time.perf_counter() - wall_start
    loop_start = time.perf_counter()
    while not flow.done:
        command = flow.operator_input()
        if command == "take_over":
            tracker.take_over()
        elif command == "go":
            tracker.go()
            judge = EntryJudge(region, control_dt)  # Entry and Fall times count from Go
        sent = robot_io.commands_sent
        tracker.step()
        if robot_io.commands_sent != sent + 1:
            raise RuntimeError(f"The Tracker sent no joint command at t={physics.time_s:.2f} s")
        physics.step(robot_io.joint_targets)
        flow.tick_done(reference_ended=tracker.reference_ended)

        data = physics.data
        judge.update(
            pelvis_pos=data.xpos[bodies["pelvis"]],
            pelvis_quat_wxyz=data.xquat[bodies["pelvis"]],
            torso_pos=data.xpos[bodies["torso_link"]],
            left_foot_pos=data.xpos[bodies["left_ankle_roll_link"]],
            right_foot_pos=data.xpos[bodies["right_ankle_roll_link"]],
        )
        margin = guard.check(data, time_s=physics.time_s)
        rollout["time"].append(physics.time_s)
        rollout["qpos"].append(data.qpos.copy())
        rollout["qvel"].append(data.qvel.copy())
        rollout["joint_target"].append(robot_io.joint_targets.copy())
        rollout["reference_frame"].append(int(tracker.motion_t[0]))
        rollout["margin"].append(margin)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            termination = "nonfinite_state"
            break
        if judge.fallen:
            termination = "fall_detected" if flow.go_time_s is not None else "fall_during_hold"
            logger.warning("Fall at t={:.2f} s ({})", physics.time_s, termination)
            break
    loop_s = time.perf_counter() - loop_start

    after_go = flow.go_time_s is not None
    completed = termination == "completed"
    result = {
        "schema_version": 1,
        "case": case.name,
        "reference_source": "file_playback",
        "entered": bool(after_go and judge.entered),
        "entry_time_s": _seconds(judge.entry_time_s) if after_go else None,
        "fallen": judge.fallen,
        "fall_time_s": _seconds(judge.fall_time_s) if after_go else None,
        "completed": completed,
        "termination": termination,
        "task_success": bool(after_go and judge.entered and not judge.fallen and completed),
        "valid": guard.valid,
        "invalid_reason": None if guard.valid else "left_task_envelope",
        "envelope": {
            "threshold_m": guard.threshold_m,
            "min_margin_m": guard.min_margin_m,
            "closest": guard.closest,
            "first_violation": guard.first_violation,
        },
        "timeline": {
            "control_dt_s": control_dt,
            "hold_s": case.hold_s,
            "go_time_s": _seconds(flow.go_time_s),
            "reference_end_time_s": _seconds(flow.reference_end_time_s),
            "final_hold_s": case.final_hold_s,
            "end_time_s": _seconds(physics.time_s),
            "ticks": flow.tick,
            "motion_frames": int(tracker.motion_length),
        },
        "wall": {
            "total_s": time.perf_counter() - wall_start,
            "setup_s": setup_s,
            "loop_s": loop_s,
            "per_tick_ms": 1000.0 * loop_s / max(flow.tick, 1),
            "realtime_factor": physics.time_s / loop_s if loop_s else None,
        },
        "inputs": {
            "policy_config": str(case.policy_config),
            "deployment_contract": str(contract_path),
            "motion": str(case.motion),
            "scene": str(Path(case.scene_dir) / build["scene"]["file"]),
            "scene_sha256": build["scene"]["sha256"],
        },
        "physics": {
            "timestep_s": float(model.opt.timestep),
            "substeps_per_tick": physics.decimation,
            "integrator": mujoco.mjtIntegrator(model.opt.integrator).name,
            "solver": mujoco.mjtSolver(model.opt.solver).name,
            "iterations": int(model.opt.iterations),
            "tolerance": float(model.opt.tolerance),
            "cone": mujoco.mjtCone(model.opt.cone).name,
            "drive": "position actuators, backward Euler (kd + kp*timestep)",
            "bvactive": int(model.vis.global_.bvactive),
        },
        "versions": {"mujoco": mujoco.__version__, "onnxruntime": _onnxruntime_version()},
    }
    case.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case.output_dir / f"{case.name}.rollout.npz",
        **{key: np.asarray(values) for key, values in rollout.items()},
    )
    case.result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _seconds(value: float | None) -> float | None:
    """Tick-multiple times without float noise (8.54, not 8.540000000000001)."""
    return None if value is None else round(value, 6)


def _onnxruntime_version() -> str:
    import onnxruntime

    return onnxruntime.__version__


# ---------------------------------------------------------------------------
# CLI: one process per case
# ---------------------------------------------------------------------------


def _parse_cpu_list(text: str) -> list[int]:
    cpus: list[int] = []
    for part in text.strip().split(","):
        if part:
            lo, _, hi = part.partition("-")
            cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def performance_cpus() -> list[int]:
    """Logical CPUs on performance cores (Intel hybrid ``cpu_core``), else all usable CPUs."""
    hybrid = Path("/sys/devices/cpu_core/cpus")
    if hybrid.is_file():
        return _parse_cpu_list(hybrid.read_text())
    return sorted(os.sched_getaffinity(0))


def performance_core_count() -> int:
    """Physical performance cores (8 on the i9-14900KF development Host)."""
    cores = set()
    for cpu in performance_cpus():
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add(((topology / "physical_package_id").read_text(), (topology / "core_id").read_text()))
        except OSError:
            cores.add(("", str(cpu)))
    return max(1, len(cores))


def _init_worker(cpus: list[int]) -> None:
    # Keep episodes off the efficiency cores (about 1.85x slower per step).
    with contextlib.suppress(AttributeError, OSError):
        os.sched_setaffinity(0, cpus)


def _run_case_logged(case: EpisodeCase) -> dict:
    case.output_dir.mkdir(parents=True, exist_ok=True)
    with open(case.output_dir / f"{case.name}.log", "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        logger.remove()
        logger.add(log, level="INFO")
        logger.add(sys.stderr, level="WARNING", format=f"[{case.name}] {{level}}: {{message}}")
        try:
            return run_episode(case)
        except Exception:
            logger.exception("Episode {} failed", case.name)
            raise


@dataclass
class VehicleSim2SimArgs:
    """Vehicle sim2sim with File Playback: every Tracker × Motion is one case, one process each."""

    motions: list[str]
    """Motions: names in --motion-root (take1512) or paths to any4hdmi .npz files."""
    trackers: list[str] = field(default_factory=lambda: ["ft500ema"])
    """Trackers: names in --tracker-root (ft500ema) or paths to policy.yaml (deployment.json beside it)."""
    artifact_root: str = str(ARTIFACT_ROOT)
    """checkpoints/scalecar_d1146 with tracker/, motions/ and scene/ (default: this checkout's)."""
    output_dir: str | None = None
    """Results: <case>.json, <case>.rollout.npz and <case>.log per case, plus summary.json.
    Default: <artifact-root>/sim2sim."""
    scene_dir: str | None = None
    """Default: <artifact-root>/scene."""
    tracker_root: str | None = None
    """Default: <artifact-root>/tracker."""
    motion_root: str | None = None
    """any4hdmi dataset root. Default: <artifact-root>/motions."""
    workers: int | None = None
    """Parallel episodes; default min(cases, physical performance cores)."""
    hold_s: float = 5.0
    final_hold_s: float = 2.0

    def __post_init__(self) -> None:
        root = Path(self.artifact_root).expanduser()
        self.output_dir = self.output_dir or str(root / "sim2sim")
        self.scene_dir = self.scene_dir or str(root / "scene")
        self.tracker_root = self.tracker_root or str(root / "tracker")
        self.motion_root = self.motion_root or str(root / "motions")


def resolve_cases(args: VehicleSim2SimArgs) -> list[EpisodeCase]:
    def tracker_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.suffix in (".yaml", ".yml") else Path(args.tracker_root) / value / "policy.yaml"

    def motion_path(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.suffix == ".npz" else Path(args.motion_root) / "motions" / f"{value}.npz"

    cases = []
    for tracker in map(tracker_path, args.trackers):
        for motion in map(motion_path, args.motions):
            for required in (tracker, tracker.parent / "deployment.json", motion):
                if not required.is_file():
                    raise FileNotFoundError(required)
            cases.append(
                EpisodeCase(
                    name=f"{tracker.parent.name}_{motion.stem}",
                    policy_config=tracker.resolve(),
                    motion=motion.resolve(),
                    scene_dir=Path(args.scene_dir).resolve(),
                    output_dir=Path(args.output_dir).resolve(),
                    hold_s=args.hold_s,
                    final_hold_s=args.final_hold_s,
                )
            )
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError(f"Case names collide: {names}")
    return cases


def _summary_line(result: dict) -> str:
    entry = f"{result['entry_time_s']:.2f} s" if result["entered"] else "-"
    validity = "valid" if result["valid"] else "INVALID (left the Task Envelope)"
    return (
        f"{result['case']}: task_success={result['task_success']} entry={entry} fallen={result['fallen']} "
        f"termination={result['termination']} {validity} min_margin={result['envelope']['min_margin_m']:.3f} m "
        f"wall={result['wall']['total_s']:.0f} s"
    )


def main(args: VehicleSim2SimArgs) -> int:
    import multiprocessing

    cases = resolve_cases(args)
    workers = min(len(cases), args.workers or performance_core_count())
    if workers < 1:
        raise ValueError("--workers must be positive")
    # One ONNX Runtime thread per episode, as Scalecar; the episodes are the parallelism.
    os.environ.setdefault("SIM2REAL_ORT_NUM_THREADS", "1")
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("{} case(s), {} worker(s), results in {}", len(cases), workers, output_dir)

    results, errors = {}, {}
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(performance_cpus(),),
    ) as pool:
        futures = {pool.submit(_run_case_logged, case): case for case in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case.name] = future.result()
                print(_summary_line(results[case.name]), flush=True)
            except Exception as error:  # noqa: BLE001 - reported per case, the others keep running
                errors[case.name] = f"{type(error).__name__}: {error}"
                print(f"{case.name}: ERROR {errors[case.name]} (see {case.output_dir / case.name}.log)", flush=True)

    summary = {
        "schema_version": 1,
        "workers": workers,
        "wall_s": time.perf_counter() - started,
        "cases": {
            case.name: {
                "result": str(case.result_path) if case.name in results else None,
                "error": errors.get(case.name),
                **{
                    key: results[case.name][key]
                    for key in ("task_success", "entered", "entry_time_s", "fallen", "completed", "valid")
                    if case.name in results
                },
            }
            for case in cases
        },
        "args": asdict(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    ok = not errors and all(r["task_success"] and r["valid"] for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    import tyro

    raise SystemExit(main(tyro.cli(VehicleSim2SimArgs)))
