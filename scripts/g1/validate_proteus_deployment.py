"""Headless rehearsal of the actual Tracking controller with MuJoCo and BridgeSafety.

No DDS is created. Root is fixed during shaping to approximate gantry support,
then released onto the floor. This is a deployment regression test, not a
model of crane/cable forces or evidence of hardware communication failsafes.
"""
from __future__ import annotations

import json
from collections import deque
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import torch  # noqa: F401
import mujoco
import numpy as np
import tyro
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController, UnitreeJoystickState
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.sim_env.joint_friction import apply_joint_friction, load_joint_friction
from sim2real.rl_policy.robot_io.bridge_safety import PROTEUS_TARGET_MARGIN_RAD, BridgeSafety
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.sim_env.integrated_sim2sim import IntegratedSimRuntime
from sim2real.utils.common import LowCmdMessage, LowStateMessage


class SimJoystick(UnitreeJoystickController):
    def __init__(self):
        self._wc_lock = threading.Lock()
        self._wc_msg = UnitreeJoystickState()
        self._last_wc_msg = UnitreeJoystickState()
        self._pending_extra_keys = []

    def close(self):
        pass


class SimIO(RobotIO):
    def __init__(self, sim, guard, joint_names):
        self.sim, self.guard, self.joint_names = sim, guard, joint_names
        self.now_ns = 1_000_000_000
        self.sequence = 0
        self.mode, self.generation = "zero", 0
        self.rejected = []
        self.measured_root = False
        # A LowState reception gap: ZMQRobotIO serves its cached state until its timeout, then nothing.
        self.gap_since_ns = None
        self.io_timeout_ns = 250_000_000
        self._last = None

    def read_state(self):
        if self.gap_since_ns is not None:
            return self._last if self.now_ns - self.gap_since_ns <= self.io_timeout_ns else None
        state = SimpleNamespace(
            root_pos_w=np.zeros(3), root_quat_w=np.zeros(4), root_lin_vel_w=np.zeros(3),
            root_ang_vel_b=np.zeros(3), joint_pos=np.zeros(len(self.joint_names)), joint_vel=np.zeros(len(self.joint_names)),
            joint_torque=np.zeros(len(self.joint_names)))
        self.sim.sync_policy_state(state)
        root_pos = state.root_pos_w if self.measured_root else np.zeros(3)
        root_vel = state.root_lin_vel_w if self.measured_root else np.zeros(3)
        self._last = RobotState(np.concatenate([root_pos, state.root_quat_w, state.joint_pos]),
                                np.concatenate([root_vel, state.root_ang_vel_b, state.joint_vel]),
                                state.joint_torque, state.low_state_tick, self.guard.stop_epoch)
        return self._last

    def set_control_mode(self, mode, generation):
        self.mode, self.generation = mode, generation

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        self.sequence += 1
        cmd = LowCmdMessage(q_target, dq_target, tau_ff, kp, kd, source_time_ns=self.now_ns,
                            sequence=self.sequence, control_mode=self.mode, control_generation=self.generation)
        accepted = self.guard.accept_command(LowCmdMessage.from_bytes(cmd.to_bytes()), self.now_ns)
        if not accepted and len(self.rejected) < 3:
            invalid = np.where((cmd.q_target < self.guard.lower - .02) | (cmd.q_target > self.guard.upper + .02))[0]
            self.rejected.append(dict(time_s=(self.now_ns - 1_000_000_000) / 1e9, mode=self.mode,
                                      out_of_range={self.joint_names[i]: float(cmd.q_target[i]) for i in invalid}))


class SimTracking(Tracking):
    def __init__(self, args, io):
        self._sim_io = io
        super().__init__(args)

    def _create_robot_io(self):
        return self._sim_io

    def _build_controller(self):
        self.controller_type = "joystick"
        self.keyboard_controller = self.pico_controller = None
        self.joystick_controller = SimJoystick()
        return self.joystick_controller


@dataclass
class Args:
    robot: str = "g1"
    policy_config: str = "checkpoints/proteus/proteus_60/unitree_g1/policy.yaml"
    motion_path: str = "hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz"
    output: str = "outputs/proteus-deployment-rehearsal"
    seed: int = 0
    walking_s: float = 3.0
    npz_playback_limit_s: float | None = None
    scenario: Literal["controls", "walk-hold"] = "controls"
    """walk-hold rehearses natural playback completion with no pause or Select event."""
    hold_s: float = 5.0
    push_velocity_m_s: float = 0.0
    """Seeded horizontal root velocity disturbance 0.5 s after Start; simulation only."""
    max_lateral_error_m: float = 0.2
    """Endpoint lateral error relative to reference allowed by the walk-hold acceptance check."""
    state_gap_at_s: float = 4.6
    """Seconds after Start at which a LowState reception gap begins (the 2026-11-04 hardware gap began at 4.6 s)."""
    state_gap_s: float = 0.0
    """Length of that gap; 0 disables it."""
    io_timeout_s: float = 0.25
    """How long the Tracker keeps inferring on its cached state during a gap (SIM2REAL_IO_TIMEOUT_S)."""
    action_delay_ms: float = 0.0
    """Latency from the Tracker issuing a command to the motors applying it (real PC chain: ZMQ, Bridge, DDS)."""
    joint_friction: str | None = None
    """YAML of joint-name pattern -> dry friction (N m) measured on the robot (sim2real.sim_env.joint_friction)."""
    hold_through_gaps: bool = True
    """Bridge keeps the last PD targets through gaps, as the PC launcher does; False latches damping."""


def main(args):
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="WARNING")
    np.random.seed(args.seed)
    robot_cfg = get_robot_cfg(args.robot)
    sim = IntegratedSimRuntime(robot_cfg, sim_dt=0.002, headless=True, key_callback=None)
    if args.joint_friction:
        apply_joint_friction(sim.mj_model, load_joint_friction(args.joint_friction, robot_cfg.joint_names))
    guard = BridgeSafety(robot_cfg, timeout_s=0.25, target_margin_rad=PROTEUS_TARGET_MARGIN_RAD,
                         hold_through_gaps=args.hold_through_gaps)
    io = SimIO(sim, guard, robot_cfg.joint_names)
    tracking = SimTracking(TrackingArgs(robot=args.robot, policy_config=args.policy_config, motion_path=args.motion_path,
                                       npz_playback_limit_s=args.npz_playback_limit_s,
                                       robot_io="zmq", controller="joystick", inference_backend="onnx-cpu"), io)
    tracking.total_inference_cnt = 0
    io.io_timeout_ns = round(args.io_timeout_s * 1e9)
    io.measured_root = not tracking.policy_config["proteus"]["reference_forcing"]
    # Gantry: hold the root above its standing height, then lower it slightly below.
    stand_height = robot_cfg.default_qpos[2]
    hung, landed = stand_height + 0.15, stand_height - 0.02
    sim.mj_data.qpos[:] = np.array(robot_cfg.default_qpos)
    sim.mj_data.qpos[2] = hung
    mujoco.mj_forward(sim.mj_model, sim.mj_data)
    # Early Start and early second A must be ignored. Then shape, Hold, play,
    # pause, resume, Select, and an attempted Start after stop.
    entry_delay = float(tracking.policy_config.get("reference_entry_s", 0.0))
    start = 13.5 + entry_delay
    pause = start + args.walking_s
    resume = pause + 2.0
    stop = resume + args.walking_s
    end = stop + 1.0
    events = [(0.04, "Start"), (0.2, "A"), (2.0, "A"), (10.24, "A"),
              (start, "Start"), (pause, "Start"), (resume, "Start"), (stop, "Select"), (stop + 0.2, "Start")]
    if args.scenario == "walk-hold":
        if args.npz_playback_limit_s is None:
            raise ValueError("walk-hold requires an explicit npz_playback_limit_s")
        stop = start + args.npz_playback_limit_s
        end = stop + args.hold_s
        events = [(0.04, "Start"), (0.2, "A"), (2.0, "A"), (10.24, "A"), (start, "Start")]
    rng = np.random.default_rng(args.seed)
    push = rng.normal(size=2)
    push *= args.push_velocity_m_s / max(np.linalg.norm(push), 1e-9)
    frames, timings, transitions, faults, reasons = [], [], [], [], []
    pending = deque()
    previous_mode = None
    body_names = tracking.policy_config["proteus"]["body_names"]
    body_ids = [sim.mj_model.body(n).id for n in body_names]
    try:
        for step in range(round(end / 0.02)):
            t = step * 0.02
            io.now_ns = 1_000_000_000 + round(t * 1e9)
            buttons = {name for at, name in events if at - 1e-6 <= t < at + 0.04 - 1e-6}
            tracking.controller._wc_msg = UnitreeJoystickState(A="A" in buttons, Start="Start" in buttons, Select="Select" in buttons)
            gap_start = start + args.state_gap_at_s
            in_gap = gap_start - 1e-6 <= t < gap_start + args.state_gap_s - 1e-6
            io.gap_since_ns = 1_000_000_000 + round(gap_start * 1e9) if in_gap else None
            sensed = io.read_state()
            if not in_gap:
                msg = LowStateMessage(sensed.qpos[3:7], sensed.qvel[3:6], sensed.qpos[7:], sensed.qvel[6:],
                                      sensed.joint_torque, sensed.tick)
                keys = (int("A" in buttons) << 8) | (int("Select" in buttons) << 3)
                guard.observe_state(msg, keys, io.now_ns)
            before = time.perf_counter()
            tracking.step()
            timings.append(time.perf_counter() - before)
            if guard.arm_pending:
                guard.takeover_finished(True, io.now_ns)
            cmd = guard.output(io.now_ns)
            if cmd:
                pending.append((t, cmd))
            mode = (tracking.state_dict["control_mode"], bool(tracking.state_dict["paused"]))
            if mode != previous_mode:
                transitions.append(dict(time_s=t, mode=mode[0], paused=mode[1], motion_t=int(tracking.motion_t[0])))
                previous_mode = mode
            if guard.fault and guard.reason not in faults:
                faults.append(guard.reason)
            if not reasons or reasons[-1]["reason"] != guard.reason:
                reasons.append(dict(time_s=t, reason=guard.reason))
            if abs(t - (start + .5)) < 1e-6:
                sim.mj_data.qvel[sim.root_qvel_adr:sim.root_qvel_adr + 2] += push
            for k in range(10):
                # A command reaches the motors action_delay_ms after the Tracker issues it.
                due = t + k * 0.002 - args.action_delay_ms / 1000 + 1e-9
                while len(pending) > 1 and pending[1][0] <= due:
                    pending.popleft()
                if pending and pending[0][0] <= due:
                    c = pending[0][1]
                    sim.apply_command(c.q_target, c.dq_target, c.tau_ff, c.kp, c.kd)
                if t < 12.1 + entry_delay:
                    # Ideal gantry: fixed horizontal position/orientation and
                    # Enter Hold first, then lower over one second.
                    height = hung - (hung - landed) * np.clip((t - 11.0 - entry_delay) / 1.0, 0, 1)
                    sim.mj_data.qpos[:7] = [0, 0, height, 1, 0, 0, 0]
                    sim.mj_data.qvel[:6] = 0
                    mujoco.mj_forward(sim.mj_model, sim.mj_data)
                sim.sim_step()
            # Compare against the reference as the policy sees it: yaw-aligned to the robot
            # at Hold entry. The raw Motion frame keeps the clip's own heading.
            aligned_ref = tracking._proteus_core._reference()
            ref = aligned_ref[0][0]
            actual = sim.mj_data.xpos[body_ids].copy()
            rel_err = np.linalg.norm((actual - actual[0]) - (ref - ref[0]), axis=-1).mean()
            frames.append(dict(time=t, root=sim.mj_data.qpos[:3].copy(), quat=sim.mj_data.qpos[3:7].copy(),
                               qpos=sim.mj_data.qpos.copy(),
                               joint_pos=sim.mj_data.qpos[sim.qpos_adrs].copy(),
                               joint_vel=sim.mj_data.qvel[sim.qvel_adrs].copy(),
                               joint_torque=sim.mj_data.actuator_force[sim.act_adrs].copy(),
                               cmd_q=cmd.q_target.copy() if cmd is not None else np.full(len(robot_cfg.joint_names), np.nan),
                               reference_root=aligned_ref[0][0, 0].copy(),
                               motion_t=int(tracking.motion_t[0]), mode=mode[0], paused=mode[1], relative_error=rel_err,
                               root_xy_error=np.linalg.norm(actual[0, :2] - ref[0, :2]),
                               finite=bool(np.isfinite(sim.mj_data.qpos).all() and np.isfinite(sim.mj_data.qvel).all())))
            tracking.total_inference_cnt += 1
    finally:
        sim.stop()
        tracking.controller.close()
        tracking.robot_io.close()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path) + ".npz", **{key: np.array([f[key] for f in frames]) for key in frames[0]})
    free = [f for f in frames if 12.1 + entry_delay <= f["time"] < stop]
    report = dict(robot=args.robot, seed=args.seed, simulated_seconds=end, reference_forcing=tracking.policy_config["proteus"]["reference_forcing"], joint_friction=args.joint_friction, action_delay_ms=args.action_delay_ms,
                  npz_playback_limit_s=args.npz_playback_limit_s,
                  scenario=args.scenario, push_velocity_m_s=args.push_velocity_m_s,
                  policy_config=args.policy_config, motion_path=args.motion_path,
                  actor_kind=tracking.policy_config["proteus"].get("actor_kind", "separate"),
                  measured_root=io.measured_root,
                  shaping_target=tracking.policy_config.get("shaping_target", "default_joint_pos"),
                  reference_entry_s=entry_delay,
                  transitions=transitions, bridge_faults=faults, bridge_reasons=reasons, rejected_commands=io.rejected, finite=all(f["finite"] for f in frames),
                  min_free_root_height_m=min(f["root"][2] for f in free),
                  min_free_root_upright=min(1 - 2 * (f["quat"][1]**2 + f["quat"][2]**2) for f in free),
                  mean_free_relative_body_error_m=float(np.mean([f["relative_error"] for f in free])),
                  max_free_root_xy_error_m=max(f["root_xy_error"] for f in free),
                  step_ms_p99=float(np.quantile(timings, .99) * 1000), step_ms_max=max(timings) * 1000,
                  hardware_tested=False, gantry="ideal fixed root during shaping; no cable dynamics")
    expected_modes = ["zero", "init", "policy", "policy", "policy", "policy", "zero"]
    if args.npz_playback_limit_s is not None:
        expected_modes = ["zero", "init", "policy", "policy", "policy", "zero"]
    if args.scenario == "walk-hold":
        expected_modes = ["zero", "init", "policy", "policy", "policy"]
        def segment_metrics(begin, finish):
            segment = [f for f in frames if begin - 1e-6 <= f["time"] < finish - 1e-6]
            first, last = segment[0], segment[-1]
            w, x, y, z = first["quat"]
            heading = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
            forward = np.array([np.cos(heading), np.sin(heading)])
            left = np.array([-np.sin(heading), np.cos(heading)])
            delta = last["root"][:2] - first["root"][:2]
            reference_delta = last["reference_root"][:2] - first["reference_root"][:2]
            bad = [f["time"] for f in segment if f["root"][2] <= .75*stand_height
                   or 1 - 2*(f["quat"][1]**2 + f["quat"][2]**2) <= .7 or not f["finite"]]
            return dict(min_root_height_m=min(f["root"][2] for f in segment),
                        min_upright=min(1 - 2*(f["quat"][1]**2 + f["quat"][2]**2) for f in segment),
                        forward_m=float(delta @ forward), lateral_m=float(delta @ left),
                        reference_forward_m=float(reference_delta @ forward),
                        reference_lateral_m=float(reference_delta @ left),
                        lateral_error_m=float((delta - reference_delta) @ left),
                        mean_relative_body_error_m=float(np.mean([f["relative_error"] for f in segment])),
                        displacement_xy_m=float(np.linalg.norm(delta)),
                        first_instability_time_s=bad[0] if bad else None, upright_all=not bad)
        report["walking"] = segment_metrics(start, stop)
        report["hold"] = segment_metrics(stop, end)
        report["start_time_s"], report["hold_time_s"] = start, stop
    report["passed"] = bool(report["finite"] and not faults and report["min_free_root_height_m"] > .75 * stand_height
                            and report["min_free_root_upright"] > .7
                            and [x["mode"] for x in transitions] == expected_modes)
    if args.scenario == "walk-hold":
        report["walking_passed"] = bool(report["walking"]["upright_all"] and not faults
                                        and report["walking"]["forward_m"] > .5)
        report["max_lateral_error_m"] = args.max_lateral_error_m
        report["straight_tracking_passed"] = abs(report["walking"]["lateral_error_m"]) <= args.max_lateral_error_m
        # A stationary end must stay upright and drift less than 20 cm in five seconds.
        report["hold_passed"] = bool(report["hold"]["upright_all"] and not faults
                                     and report["hold"]["displacement_xy_m"] < .2)
        report["passed"] &= report["walking_passed"] and report["straight_tracking_passed"] and report["hold_passed"]
    Path(str(path) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
