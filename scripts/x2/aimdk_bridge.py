"""AgiBot X2 AimDK HAL <-> local ZMQ bridge with deferred takeover and independent stop arbitration.

The X2 counterpart of scripts/et1/hdep_bridge.py, run on the X2 motion-control board (RK3588). It serves the
same ZMQ LowState/LowCmd contract and reuses BridgeSafety: startup only reads, and a ready Tracker plus a
fresh operator A press are needed before the takeover (``aima em stop-app mc``, which releases the native
motion controller) and the first HAL command. Select (space on the operator keyboard), command/state timeouts
or invalid commands latch. Operator keys (scripts/et1/operator_keyboard.py on PORTS["operator"]) are
re-published on PORTS["gamepad"] in the Unitree key layout for the Tracker's bridge_joystick controller.

The HAL publishes joint state at 1000 Hz and the pelvis IMU at 500 Hz; a ROS executor thread keeps the latest
values in X2StateCommand, and this loop sends the arbitrated command to /aima/hal/joint/*/command every cycle,
clamped to the training ∩ vendor joint range. The HAL has no command timeout: after a SIGKILL of this process
or a board power loss the last command stays in force. Keep the gantry and the physical stop ready.
"""
from __future__ import annotations

import json
import shlex
import signal
import struct
import subprocess
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tyro
import zmq
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.config.robots.x2 import X2_AIMDK_GROUPS
from sim2real.rl_policy.robot_io.bridge_safety import BridgeSafety
from sim2real.rl_policy.robot_io.x2 import X2StateCommand
from sim2real.utils.common import PORTS, LowCmdMessage, LowStateMessage
from sim2real.utils.operator_keys import OPERATOR_FORMAT

GAMEPAD_FORMAT = "<QIffff"  # stamp (ns), Unitree keys, lx, ly, rx, ry (sim2real/rl_policy/controllers/bridge_joystick.py)
OPERATOR_FRESH_S = 0.3


class AimdkHal:
    """ROS 2 side: subscriptions fill X2StateCommand from an executor thread; publish() writes one command."""

    def __init__(self, cfg: RobotCfg, *, head_present: bool, head_pitch: bool, imu_topic: str, read_only: bool):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
        from sensor_msgs.msg import Imu

        self.rclpy, self.joint_type, self.array_type = rclpy, JointCommand, JointCommandArray
        # timeout_s here only bounds read_state freshness; BridgeSafety applies the real timeouts.
        self.codec = X2StateCommand(cfg, timeout_s=1.0, head_present=head_present, head_pitch_enabled=head_pitch)
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)  # keep ROS alive for the exit damping
        self.node = Node("sim2real_x2_bridge")
        rx = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        tx = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self.publishers = {}
        for group in self.codec.group_ids:
            prefix = f"/aima/hal/joint/{group}"
            self.node.create_subscription(JointStateArray, prefix + "/state",
                                          lambda m, g=group: self.codec.receive_group(g, m), rx)
            if not read_only:
                self.publishers[group] = self.node.create_publisher(JointCommandArray, prefix + "/command", tx)
        self.node.create_subscription(Imu, imu_topic, self.codec.receive_imu, rx)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, name="x2-aimdk", daemon=True)
        self.thread.start()

    def read(self):
        """(RobotState or None, leg stamp) with the latest feedback; raises on a latched feedback error."""
        state = self.codec.read_state()
        with self.codec.lock:
            leg = self.codec.groups.get("leg")
        return state, (leg[0] if leg else None)

    def publish(self, cmd: LowCmdMessage) -> None:
        codec = self.codec
        q = np.clip(cmd.q_target, codec.lower, codec.upper)
        values = [q, cmd.dq_target, cmd.tau_ff, cmd.kp, cmd.kd]
        if codec.head_present and not codec.head_pitch_enabled:
            i = codec.cfg.joint_names.index("head_pitch_joint")
            values = [np.array(v, dtype=np.float64) for v in values]
            for v in values:
                v[i] = 0.0
        stamp = self.node.get_clock().now().to_msg()
        for group, ids in codec.group_ids.items():
            msg = self.array_type()
            msg.header.stamp = stamp
            for k, name in enumerate(X2_AIMDK_GROUPS[group]):
                joint = self.joint_type()
                joint.name = name
                joint.position, joint.velocity, joint.effort, joint.stiffness, joint.damping = (
                    float(v[ids[k]]) for v in values)
                msg.joints.append(joint)
            self.publishers[group].publish(msg)

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=1.0)
        self.node.destroy_node()
        self.rclpy.shutdown()


class X2Bridge:
    def __init__(self, robot_cfg: RobotCfg, hal: AimdkHal, *, rate_hz=500.0, read_only=False, timeout_s=0.1,
                 takeover_cmd="aima em stop-app mc", takeover_timeout_s=70.0, gamepad_port: int = PORTS["gamepad"],
                 operator_port: int = PORTS["operator"]):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.robot_cfg, self.hal = robot_cfg, hal
        self.dt = 1.0 / rate_hz
        self.read_only = read_only
        self.takeover_cmd = takeover_cmd
        self.takeover_timeout_s = takeover_timeout_s
        # The X2 runs leg and waist kd x2 (knee 15.3) against command delay; see make_damped_config.py.
        self.guard = BridgeSafety(robot_cfg, timeout_s, max_kd=25.0)
        self.tick = 0
        self._stamp = None
        self._last_reason = None
        self._tracker_ready_logged = False
        self.samples = self.writes = 0
        self.takeover_s = None
        context = zmq.Context.instance()
        self.state_pub = context.socket(zmq.PUB)
        self.state_pub.setsockopt(zmq.SNDHWM, 1)
        self.state_pub.setsockopt(zmq.LINGER, 0)
        self.state_pub.bind(f"tcp://{robot_cfg.low_state_bind_addr}:{robot_cfg.low_state_port}")
        self.gamepad_pub = context.socket(zmq.PUB)
        self.gamepad_pub.setsockopt(zmq.SNDHWM, 1)
        self.gamepad_pub.setsockopt(zmq.LINGER, 0)
        self.gamepad_pub.bind(f"tcp://{robot_cfg.low_state_bind_addr}:{gamepad_port}")
        self.cmd_sub = context.socket(zmq.SUB)
        self.cmd_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.cmd_sub.setsockopt(zmq.CONFLATE, 1)
        self.cmd_sub.setsockopt(zmq.LINGER, 0)
        self.cmd_sub.connect(f"tcp://{robot_cfg.low_cmd_host}:{robot_cfg.low_cmd_port}")
        self.operator_sub = context.socket(zmq.SUB)
        self.operator_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.operator_sub.setsockopt(zmq.LINGER, 0)
        self.operator_sub.bind(f"tcp://127.0.0.1:{operator_port}")
        self._operators: dict[int, tuple[int, int]] = {}

    def _operator_keys(self) -> int:
        for _ in range(64):
            try:
                sent_ns, keys, sender = struct.unpack(OPERATOR_FORMAT, self.operator_sub.recv(flags=zmq.DONTWAIT))
            except zmq.Again:
                break
            except struct.error as exc:
                logger.warning("Malformed operator message: {}", exc)
                continue
            self._operators[sender] = (sent_ns, keys)
        now = time.monotonic_ns()
        self._operators = {s: v for s, v in self._operators.items() if 0 <= now - v[0] <= OPERATOR_FRESH_S * 1e9}
        merged = 0
        for _, keys in self._operators.values():
            merged |= keys
        return merged

    def _observe(self) -> None:
        keys = self._operator_keys()
        self.gamepad_pub.send(struct.pack(GAMEPAD_FORMAT, time.monotonic_ns(), keys, 0.0, 0.0, 0.0, 0.0),
                              flags=zmq.DONTWAIT)
        try:
            state, stamp = self.hal.read()
        except RuntimeError as exc:
            self.guard.latch(f"HAL feedback error: {exc.__cause__ or exc}")
            return
        if state is None:
            return  # stale or incomplete feedback: BridgeSafety times out on its own clock
        self.samples += 1
        if stamp != self._stamp:
            self._stamp = stamp
            self.tick = (self.tick + 1) & 0xFFFFFFFF
        message = LowStateMessage(state.qpos[3:7], state.qvel[3:6], state.qpos[7:], state.qvel[6:],
                                  state.joint_torque, self.tick)
        self.guard.observe_state(message, keys, time.monotonic_ns())

    def _write(self, cmd: LowCmdMessage) -> None:
        if self.read_only:
            return
        try:
            self.hal.publish(cmd)
            self.writes += 1
        except Exception as exc:
            self.guard.latch(f"HAL publish failed: {exc}")

    def _take_over(self) -> None:
        begin = time.monotonic()
        logger.info("Operator A with a ready Tracker: releasing the native motion controller ({})", self.takeover_cmd)
        success = True
        if self.takeover_cmd:
            try:
                result = subprocess.run(shlex.split(self.takeover_cmd), capture_output=True, text=True, timeout=self.takeover_timeout_s)
                logger.info("takeover: exit {} {}", result.returncode, (result.stdout + result.stderr).strip()[-300:])
                success = result.returncode == 0
            except Exception as exc:
                logger.error("takeover failed: {}", exc)
                success = False
        self.takeover_s = time.monotonic() - begin
        logger.info("takeover returned after {:.2f} s", self.takeover_s)
        if success and self.takeover_s > self.guard.timeout_ns / 1e9:
            # The loop was blocked, so the Tracker lost LowState and its shaping clock ran on without the robot.
            # mc is released now, so a relaunch takes over at once (stopping mc after a reboot takes ~15 s).
            self.guard.takeover_finished(True, time.monotonic_ns())
            self.guard.latch(f"takeover took {self.takeover_s:.1f} s; mc is released, restart the launcher and "
                             "press a again")
            return
        # The call blocks the loop; judge freshness on new state and the Tracker's latest command.
        self._observe()
        self._receive_command()
        self.guard.takeover_finished(success, time.monotonic_ns())

    def _receive_command(self) -> None:
        try:
            data = self.cmd_sub.recv(flags=zmq.DONTWAIT)
        except zmq.Again:
            return
        try:
            self.guard.accept_command(LowCmdMessage.from_bytes(data), time.monotonic_ns())
        except Exception as exc:
            self.guard.latch(f"malformed command: {exc}")

    def step(self) -> None:
        self._observe()
        self._receive_command()
        if self.guard.ready(time.monotonic_ns()) and not self._tracker_ready_logged:
            logger.info("Tracker and LowState ready. Operator may press A to begin shaping under gantry protection.")
            self._tracker_ready_logged = True
        if self.guard.arm_pending and not self.read_only:
            self._take_over()
        cmd = self.guard.output(time.monotonic_ns())
        if cmd is not None:
            self._write(cmd)
        if self.guard.state is not None:
            self.guard.state.stop_epoch = self.guard.stop_epoch
            try:
                self.state_pub.send(self.guard.state.to_bytes(), flags=zmq.DONTWAIT)
            except zmq.Again:
                pass
        if self.guard.reason != self._last_reason:
            logger.info("Bridge state: {} (stop epoch {})", self.guard.reason, self.guard.stop_epoch)
            self._last_reason = self.guard.reason

    def run(self, max_runtime_s=None, state_log=None) -> None:
        logger.info("X2 bridge; read_only={}; standby emits no HAL command", self.read_only)
        start = next_time = time.monotonic()
        overruns = 0
        try:
            while max_runtime_s is None or time.monotonic() - start < max_runtime_s:
                self.step()
                next_time += self.dt
                delay = next_time - time.monotonic()
                if delay < 0:
                    overruns += 1
                    next_time = time.monotonic()
                time.sleep(max(0.0, delay))
        except KeyboardInterrupt:
            pass
        finally:
            # An orderly local shutdown sends damping before exiting.
            self.guard.latch("bridge shutdown")
            for _ in range(int(0.2 / self.dt) if self.guard.active else 0):
                self._observe()
                self._write(self.guard.output(time.monotonic_ns()))
                time.sleep(self.dt)
            if state_log:
                state = self.guard.state
                report = dict(read_only=self.read_only, samples=self.samples, hal_writes=self.writes,
                              loop_overruns=overruns, takeover_s=self.takeover_s, tick=self.tick,
                              stop_epoch=self.guard.stop_epoch, reason=self.guard.reason,
                              joint_names=list(self.robot_cfg.joint_names),
                              joint_positions=state.joint_positions.tolist() if state else None,
                              quaternion=state.quaternion.tolist() if state else None)
                Path(state_log).parent.mkdir(parents=True, exist_ok=True)
                Path(state_log).write_text(json.dumps(report, indent=2) + "\n")
            if self.guard.active:
                logger.info("Damping sent. Restart the native controller with: aima em start-app mc")
            for socket in (self.state_pub, self.gamepad_pub, self.cmd_sub, self.operator_sub):
                socket.close(linger=0)


@dataclass
class Args:
    robot: str = "x2"
    rate: float = 500.0
    read_only: bool = False
    timeout_s: float = 0.1
    takeover_cmd: str = f"bash {Path(__file__).resolve().parent / 'release_mc.sh'}"
    """Run once on the operator's ready A, before the first HAL command; empty if mc is already stopped."""
    takeover_timeout_s: float = 70.0
    """stop-app mc took over 20 s on the X2 Basic; release_mc.sh waits up to 60 s."""
    no_head: bool = False
    """Robot without a head joint group on the HAL."""
    head_pitch: bool = False
    imu_topic: str = "/aima/hal/imu/torso/state"
    max_runtime_s: float | None = None
    state_log: str | None = None


def main(args: Args) -> None:
    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    cfg = replace(get_robot_cfg(args.robot), low_state_bind_addr="127.0.0.1", low_cmd_host="127.0.0.1")
    hal = AimdkHal(cfg, head_present=not args.no_head, head_pitch=args.head_pitch, imu_topic=args.imu_topic,
                   read_only=args.read_only)
    try:
        X2Bridge(cfg, hal, rate_hz=args.rate, read_only=args.read_only, timeout_s=args.timeout_s,
                 takeover_cmd=args.takeover_cmd, takeover_timeout_s=args.takeover_timeout_s).run(args.max_runtime_s, args.state_log)
    finally:
        hal.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
