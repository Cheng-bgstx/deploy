"""Joint response test for a G1 hanging from the gantry, on the robot or in MuJoCo with the same commands.

Shapes from the current pose to the deployment's default pose over 4 s, then drives both legs' hip pitch, knee and
ankle pitch in turn with 1 Hz and 2 Hz sines and a step around that pose, using the deployment PD gains, and
returns to the default pose. Recording q, dq, estimated torque and the commanded target lets the real and simulated
tracking be compared joint by joint (walking showed the real hip pitch lagging about 2.5x more than in simulation).

On the robot it releases the high-level mode and writes LowCmd directly, so no Bridge or Tracker may run. Select on
the remote, Ctrl-C, LowState older than 100 ms, a non-finite state or a joint past its range by 0.1 rad switches to
damping and exits. Run it only with the robot hanging clear of the ground.
"""
from __future__ import annotations

import signal
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import yaml

from sim2real.config.robots import G1_CFG
from sim2real.rl_policy.robot_io.bridge_safety import BridgeSafety

RATE_HZ = 200.0
SHAPE_S, SETTLE_S = 4.0, 2.0
TESTS = (  # joint suffix, sine amplitude (rad); both legs move together
    ("hip_pitch_joint", 0.3),
    ("knee_joint", 0.3),
    ("ankle_pitch_joint", 0.2),
)
SEGMENTS = ((1.0, 4.0), (2.0, 3.0))  # (frequency Hz, duration s)
STEP_S, REST_S = 2.0, 1.0


@dataclass
class Args:
    target: Literal["sim", "real"] = "sim"
    policy_config: str = "checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy-pc.yaml"
    interface: str = "enp5s0"
    output: str = "outputs/joint-response/test"
    joint_friction: str | None = None
    """sim: YAML of joint-name pattern -> dry friction (N m), see sim2real.sim_env.joint_friction."""


def schedule(default: np.ndarray):
    """Yields (time s, phase name, joint targets) after shaping, at RATE_HZ."""
    names = list(G1_CFG.joint_names)
    dt, t = 1.0 / RATE_HZ, 0.0
    for _ in range(int(SETTLE_S * RATE_HZ)):
        yield t, "settle", default.copy()
        t += dt
    for suffix, amplitude in TESTS:
        ids = [i for i, n in enumerate(names) if n.endswith(suffix)]
        for freq, duration in SEGMENTS:
            for k in range(int(duration * RATE_HZ)):
                target = default.copy()
                target[ids] += amplitude * np.sin(2 * np.pi * freq * k * dt)
                yield t, f"{suffix}:{freq:g}Hz", target
                t += dt
        for k in range(int(STEP_S * RATE_HZ)):
            target = default.copy()
            target[ids] += amplitude * (k < STEP_S * RATE_HZ / 2)
            yield t, f"{suffix}:step", target
            t += dt
        for _ in range(int(REST_S * RATE_HZ)):
            yield t, "rest", default.copy()
            t += dt


class SimRobot:
    def __init__(self, joint_friction: str | None = None):
        import mujoco

        from sim2real.sim_env.integrated_sim2sim import IntegratedSimRuntime

        self.mujoco = mujoco
        self.sim = IntegratedSimRuntime(G1_CFG, sim_dt=0.001, headless=True, key_callback=None)
        if joint_friction:
            from sim2real.sim_env.joint_friction import apply_joint_friction, load_joint_friction

            apply_joint_friction(self.sim.mj_model, load_joint_friction(joint_friction, G1_CFG.joint_names))
        d = self.sim.mj_data
        d.qpos[:] = np.array(G1_CFG.default_qpos)
        d.qpos[2] += 0.3  # hanging clear of the ground
        self.root = d.qpos[:7].copy()
        mujoco.mj_forward(self.sim.mj_model, d)

    def read(self):
        d = self.sim.mj_data
        q = d.qpos[self.sim.qpos_adrs].copy()
        dq = d.qvel[self.sim.qvel_adrs].copy()
        tau = d.actuator_force[self.sim.act_adrs].copy()
        return q, dq, tau, False

    def write(self, q, kp, kd):
        zeros = np.zeros_like(q)
        self.sim.apply_command(q, zeros, zeros, kp, kd)
        d = self.sim.mj_data
        for _ in range(int(round(1.0 / RATE_HZ / self.sim.sim_dt))):
            d.qpos[:7], d.qvel[:6] = self.root, 0.0  # the gantry holds the pelvis
            self.sim.sim_step()

    def wait(self, _deadline):
        pass

    def close(self, _damping_kd):
        self.sim.stop()


class RealRobot:
    def __init__(self, interface: str):
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl import default
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(0, interface)
        self.latest = None
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(lambda msg: setattr(self, "latest", (msg, time.monotonic())), 0)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.cmd = default.unitree_hg_msg_dds__LowCmd_()
        self.cmd.mode_machine, self.cmd.mode_pr = 5, 0
        self.crc = CRC()
        deadline = time.monotonic() + 3.0
        while self.latest is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if self.latest is None:
            raise RuntimeError("no LowState from the robot")
        if self.latest[0].mode_machine != 5:
            raise RuntimeError(f"unexpected mode_machine {self.latest[0].mode_machine}")
        client = MotionSwitcherClient()
        client.SetTimeout(1.0)
        client.Init()
        for _ in range(20):
            status, result = client.CheckMode()
            if status != 0 or result is None:
                raise RuntimeError("motion switcher did not answer")
            if not result.get("name"):
                break
            print(f"Releasing high-level mode {result['name']}", flush=True)
            client.ReleaseMode()
            time.sleep(0.2)
        else:
            raise RuntimeError("could not release the high-level mode")

    def read(self):
        msg, received = self.latest
        q = np.array([m.q for m in msg.motor_state[:29]])
        dq = np.array([m.dq for m in msg.motor_state[:29]])
        tau = np.array([m.tau_est for m in msg.motor_state[:29]])
        payload = bytes(msg.wireless_remote)
        keys = struct.unpack_from("<H", payload, 2)[0] if len(payload) == 40 else 0
        stale = time.monotonic() - received > 0.1
        return q, dq, tau, stale or bool(keys & (1 << 3))  # Select

    def write(self, q, kp, kd):
        for i, motor in enumerate(self.cmd.motor_cmd[:29]):
            motor.mode = 1
            motor.q, motor.dq, motor.tau = float(q[i]), 0.0, 0.0
            motor.kp, motor.kd = float(kp[i]), float(kd[i])
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.pub.Write(self.cmd)

    def wait(self, deadline):
        time.sleep(max(0.0, deadline - time.monotonic()))

    def close(self, damping_kd):
        q = self.read()[0]
        for _ in range(int(0.5 * RATE_HZ)):
            self.write(q, np.zeros(29), damping_kd)
            time.sleep(1.0 / RATE_HZ)


def main(args: Args):
    cfg = yaml.safe_load(Path(args.policy_config).read_text())
    names = list(G1_CFG.joint_names)
    default = np.array([cfg["default_joint_pos"][n] for n in names])
    kp = np.array([cfg["joint_kp"][n] for n in names])
    kd = np.array([cfg["joint_kd"][n] for n in names])
    lower = np.array([G1_CFG.joint_pos_lower_limit[n] for n in names])
    upper = np.array([G1_CFG.joint_pos_upper_limit[n] for n in names])
    damping_kd = BridgeSafety(G1_CFG).damping_kd

    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    robot = SimRobot(args.joint_friction) if args.target == "sim" else RealRobot(args.interface)
    rows, reason = [], "completed"
    try:
        start = robot.read()[0]
        steps = int(SHAPE_S * RATE_HZ)
        timeline = [(k / RATE_HZ - SHAPE_S, "shape", start + (default - start) * min(1.0, k / steps))
                    for k in range(steps)]
        t0 = time.monotonic()
        for i, (t, phase, target) in enumerate([*timeline, *schedule(default)]):
            q, dq, tau, stop = robot.read()
            if stop or not np.isfinite(q).all() or np.any(q < lower - 0.1) or np.any(q > upper + 0.1):
                reason = "Select or stale LowState" if stop else "joint state out of range"
                break
            robot.write(target, kp, kd)
            rows.append(np.concatenate([[t], q, dq, tau, target]))
            robot.wait(t0 + (i + 1) / RATE_HZ)
    except KeyboardInterrupt:
        reason = "interrupted"
    finally:
        robot.close(damping_kd)
    data = np.array(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"{output}.npz", t=data[:, 0], q=data[:, 1:30], dq=data[:, 30:59], tau=data[:, 59:88],
                        cmd_q=data[:, 88:117], kp=kp, kd=kd, joint_names=np.array(names), target=args.target)
    print(f"{reason}: {len(rows)} steps -> {output}.npz", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
