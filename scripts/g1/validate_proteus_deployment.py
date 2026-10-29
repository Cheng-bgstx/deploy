"""Headless rehearsal of the actual Tracking controller with MuJoCo and BridgeSafety.

No DDS is created. Root is fixed during shaping to approximate gantry support,
then released onto the floor. This is a deployment regression test, not a
model of crane/cable forces or evidence of hardware communication failsafes.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch  # noqa: F401
import mujoco
import numpy as np
import tyro
from loguru import logger

from sim2real.config.robots import G1_CFG
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController, UnitreeJoystickState
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.rl_policy.robot_io.bridge_safety import BridgeSafety
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
    def __init__(self, sim, guard):
        self.sim, self.guard = sim, guard
        self.now_ns = 1_000_000_000
        self.sequence = 0
        self.mode, self.generation = "zero", 0
        self.rejected = []

    def read_state(self):
        state = SimpleNamespace(
            root_pos_w=np.zeros(3), root_quat_w=np.zeros(4), root_lin_vel_w=np.zeros(3),
            root_ang_vel_b=np.zeros(3), joint_pos=np.zeros(29), joint_vel=np.zeros(29), joint_torque=np.zeros(29))
        self.sim.sync_policy_state(state)
        return RobotState(np.concatenate([np.zeros(3), state.root_quat_w, state.joint_pos]),
                          np.concatenate([np.zeros(3), state.root_ang_vel_b, state.joint_vel]),
                          state.joint_torque, state.low_state_tick, self.guard.stop_epoch)

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
                                      out_of_range={G1_CFG.joint_names[i]: float(cmd.q_target[i]) for i in invalid}))


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
    policy_config: str = "checkpoints/proteus/proteus_60/unitree_g1/policy.yaml"
    motion_path: str = "hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz"
    output: str = "outputs/proteus-deployment-rehearsal"
    seed: int = 0
    walking_s: float = 3.0
    npz_playback_limit_s: float | None = None


def main(args):
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="WARNING")
    np.random.seed(args.seed)
    sim = IntegratedSimRuntime(G1_CFG, sim_dt=0.002, headless=True, key_callback=None)
    guard = BridgeSafety(G1_CFG)
    io = SimIO(sim, guard)
    tracking = SimTracking(TrackingArgs(policy_config=args.policy_config, motion_path=args.motion_path,
                                       npz_playback_limit_s=args.npz_playback_limit_s,
                                       robot_io="zmq", controller="joystick", inference_backend="onnx-cpu"), io)
    tracking.total_inference_cnt = 0
    sim.mj_data.qpos[:] = np.array(G1_CFG.default_qpos)
    sim.mj_data.qpos[2] = 0.95
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
    frames, timings, transitions, faults = [], [], [], []
    previous_mode = None
    body_names = tracking.policy_config["proteus"]["body_names"]
    body_ids = [sim.mj_model.body(n).id for n in body_names]
    motion_ids = [tracking.motion_body_names.index(n) for n in body_names]
    try:
        for step in range(round(end / 0.02)):
            t = step * 0.02
            io.now_ns = 1_000_000_000 + round(t * 1e9)
            buttons = {name for at, name in events if at - 1e-6 <= t < at + 0.04 - 1e-6}
            tracking.controller._wc_msg = UnitreeJoystickState(A="A" in buttons, Start="Start" in buttons, Select="Select" in buttons)
            sensed = io.read_state()
            msg = LowStateMessage(sensed.qpos[3:7], sensed.qvel[3:6], sensed.qpos[7:], sensed.qvel[6:], sensed.joint_torque, sensed.tick)
            keys = (int("A" in buttons) << 8) | (int("Select" in buttons) << 3)
            guard.observe_state(msg, keys, io.now_ns)
            before = time.perf_counter()
            tracking.step()
            timings.append(time.perf_counter() - before)
            if guard.arm_pending:
                guard.takeover_finished(True, io.now_ns)
            cmd = guard.output(io.now_ns)
            if cmd:
                sim.apply_command(cmd.q_target, cmd.dq_target, cmd.tau_ff, cmd.kp, cmd.kd)
            mode = (tracking.state_dict["control_mode"], bool(tracking.state_dict["paused"]))
            if mode != previous_mode:
                transitions.append(dict(time_s=t, mode=mode[0], paused=mode[1], motion_t=int(tracking.motion_t[0])))
                previous_mode = mode
            if guard.fault and guard.reason not in faults:
                faults.append(guard.reason)
            for _ in range(10):
                if t < 12.1 + entry_delay:
                    # Ideal gantry: fixed horizontal position/orientation and
                    # Enter Hold first, then lower over one second.
                    height = 0.95 - 0.17 * np.clip((t - 11.0 - entry_delay) / 1.0, 0, 1)
                    sim.mj_data.qpos[:7] = [0, 0, height, 1, 0, 0, 0]
                    sim.mj_data.qvel[:6] = 0
                    mujoco.mj_forward(sim.mj_model, sim.mj_data)
                sim.sim_step()
            ref = tracking.motion_data.body_pos_w[0, 0, motion_ids]
            actual = sim.mj_data.xpos[body_ids].copy()
            rel_err = np.linalg.norm((actual - actual[0]) - (ref - ref[0]), axis=-1).mean()
            aligned_ref = tracking._proteus_core._reference()
            frames.append(dict(time=t, root=sim.mj_data.qpos[:3].copy(), quat=sim.mj_data.qpos[3:7].copy(),
                               qpos=sim.mj_data.qpos.copy(),
                               joint_pos=sim.mj_data.qpos[sim.qpos_adrs].copy(),
                               joint_vel=sim.mj_data.qvel[sim.qvel_adrs].copy(),
                               cmd_q=cmd.q_target.copy() if cmd is not None else np.full(29, np.nan),
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
    report = dict(seed=args.seed, simulated_seconds=end, reference_forcing=tracking.policy_config["proteus"]["reference_forcing"],
                  npz_playback_limit_s=args.npz_playback_limit_s,
                  shaping_target=tracking.policy_config.get("shaping_target", "default_joint_pos"),
                  reference_entry_s=entry_delay,
                  transitions=transitions, bridge_faults=faults, rejected_commands=io.rejected, finite=all(f["finite"] for f in frames),
                  min_free_root_height_m=min(f["root"][2] for f in free),
                  min_free_root_upright=min(1 - 2 * (f["quat"][1]**2 + f["quat"][2]**2) for f in free),
                  mean_free_relative_body_error_m=float(np.mean([f["relative_error"] for f in free])),
                  max_free_root_xy_error_m=max(f["root_xy_error"] for f in free),
                  step_ms_p99=float(np.quantile(timings, .99) * 1000), step_ms_max=max(timings) * 1000,
                  hardware_tested=False, gantry="ideal fixed root during shaping; no cable dynamics")
    expected_modes = ["zero", "init", "policy", "policy", "policy", "policy", "zero"]
    if args.npz_playback_limit_s is not None:
        expected_modes = ["zero", "init", "policy", "policy", "policy", "zero"]
    report["passed"] = bool(report["finite"] and not faults and report["min_free_root_height_m"] > .6
                            and report["min_free_root_upright"] > .7
                            and [x["mode"] for x in transitions] == expected_modes)
    Path(str(path) + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Args))
