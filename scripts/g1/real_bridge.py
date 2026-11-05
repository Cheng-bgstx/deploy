"""SDK2 <-> local ZMQ bridge with deferred takeover and independent stop arbitration."""
from __future__ import annotations

import json
import signal
import struct
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tyro
import zmq
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg, get_unitree_dds_family
from sim2real.rl_policy.controllers.bridge_joystick import GAMEPAD_FORMAT
from sim2real.rl_policy.robot_io.bridge_safety import PROTEUS_TARGET_MARGIN_RAD, BridgeSafety
from sim2real.utils.common import PORTS, LowCmdMessage, LowStateMessage, UNITREE_LEGGED_CONST
from sim2real.utils.sdk_shutdown import close_sdk_channel, exit_after_sdk_timeout

UNITREE_INTERFACE = "eth0"
UNITREE_DOMAIN_ID = 0
RECEPTION_GAP_LOG_NS = 30_000_000  # LowState normally arrives every 1 ms


class RealBridge:
    def __init__(self, robot_cfg: RobotCfg, rate_hz=200, interface=UNITREE_INTERFACE,
                 *, read_only=False, timeout_s=0.1, target_margin_rad=PROTEUS_TARGET_MARGIN_RAD,
                 hold_through_gaps=False, gamepad_port=PORTS["gamepad"]):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.robot_cfg, self.interface = robot_cfg, interface
        self.gamepad_port = gamepad_port
        self.dt = 1.0 / rate_hz
        self.read_only = read_only
        self.guard = BridgeSafety(robot_cfg, timeout_s, target_margin_rad, hold_through_gaps)
        self.lock = threading.Lock()
        self._latest = self._observed = None
        self._reception_gaps = []
        self._takeover_thread = None
        self._takeover_result = None
        self._last_reason = None
        self._tracker_ready_logged = False
        self.samples = 0
        self.writes = 0
        self._closing = False
        self._sdk_close_timed_out = False
        self._init_channels()
        self._init_zmq()

    def _init_channels(self):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.utils.crc import CRC
        from unitree_sdk2py.idl import default
        ChannelFactoryInitialize(UNITREE_DOMAIN_ID, self.interface)
        family = get_unitree_dds_family(self.robot_cfg.name)
        if family == "hg":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_
            self.low_cmd = default.unitree_hg_msg_dds__LowCmd_()
            self.low_cmd.mode_machine, self.low_cmd.mode_pr = 5, 0
        elif family == "go":
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
            self.low_cmd = default.unitree_go_msg_dds__LowCmd_()
            self.low_cmd.head = [0xFE, 0xEF]
            self.low_cmd.level_flag = UNITREE_LEGGED_CONST["LOWLEVEL"]
        else:
            raise NotImplementedError(family)
        for motor in self.low_cmd.motor_cmd:
            motor.mode = 1
            motor.q, motor.dq = UNITREE_LEGGED_CONST["PosStopF"], UNITREE_LEGGED_CONST["VelStopF"]
            motor.kp = motor.kd = motor.tau = 0.0
        self.crc = CRC()
        self.low_cmd_unitree_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_unitree_pub.Init()
        self.low_state_unitree_sub = ChannelSubscriber("rt/lowstate", LowState_)
        # A callback keeps the watchdog runnable even if DDS stops producing state.
        self.low_state_unitree_sub.Init(self._on_low_state, 0)

    def _init_zmq(self):
        context = zmq.Context.instance()
        self.low_state_zmq_pub = context.socket(zmq.PUB)
        self.low_state_zmq_pub.setsockopt(zmq.SNDHWM, 1)
        self.low_state_zmq_pub.setsockopt(zmq.LINGER, 0)
        self.low_state_zmq_pub.bind(f"tcp://{self.robot_cfg.low_state_bind_addr}:{self.robot_cfg.low_state_port}")
        # The handset for the Tracker's bridge_joystick controller, so the Tracker needs no DDS subscription of its
        # own: decoding 1 kHz LowState in Python took most of a core on the G1's Orin.
        self.gamepad_zmq_pub = context.socket(zmq.PUB)
        self.gamepad_zmq_pub.setsockopt(zmq.SNDHWM, 1)
        self.gamepad_zmq_pub.setsockopt(zmq.LINGER, 0)
        self.gamepad_zmq_pub.bind(f"tcp://{self.robot_cfg.low_state_bind_addr}:{self.gamepad_port}")
        self.low_cmd_zmq_sub = context.socket(zmq.SUB)
        self.low_cmd_zmq_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_cmd_zmq_sub.setsockopt(zmq.CONFLATE, 1)
        self.low_cmd_zmq_sub.setsockopt(zmq.LINGER, 0)
        self.low_cmd_zmq_sub.connect(f"tcp://{self.robot_cfg.low_cmd_host}:{self.robot_cfg.low_cmd_port}")

    def _on_low_state(self, msg):
        # The DDS listener only hands over the newest sample (no lock, no decoding), so a busy loop step can
        # never hold up reception; it also notes arrival gaps for the loop to log.
        if self._closing:
            return
        now_ns = time.monotonic_ns()
        if self._latest is not None and now_ns - self._latest[1] > RECEPTION_GAP_LOG_NS:
            self._reception_gaps.append((now_ns - self._latest[1], int(msg.tick) - int(self._latest[0].tick)))
        self.samples += 1
        self._latest = (msg, now_ns)

    def _observe_latest(self):
        """Decode the newest LowState for the guard; called under the lock at every loop step."""
        latest = self._latest
        if latest is None or latest is self._observed:
            return
        self._observed = latest
        msg, received_ns = latest
        count = self.guard.count
        state = LowStateMessage(
            quaternion=np.array(msg.imu_state.quaternion),
            gyroscope=np.array(msg.imu_state.gyroscope),
            joint_positions=np.array([m.q for m in msg.motor_state[:count]]),
            joint_velocities=np.array([m.dq for m in msg.motor_state[:count]]),
            joint_torques=np.array([m.tau_est for m in msg.motor_state[:count]]),
            tick=int(msg.tick),
        )
        payload = bytes(getattr(msg, "wireless_remote", b""))
        keys = struct.unpack_from("<H", payload, 2)[0] if len(payload) == 40 else 0
        self.guard.observe_state(state, keys, received_ns)
        if len(payload) == 40:
            # Unitree SDK layout: keys@2, lx@4, rx@8, ry@12, ly@20.
            lx, rx, ry = struct.unpack_from("<3f", payload, 4)
            ly = struct.unpack_from("<f", payload, 20)[0]
            self.gamepad_zmq_pub.send(struct.pack(GAMEPAD_FORMAT, received_ns, keys, lx, ly, rx, ry), flags=zmq.DONTWAIT)
        if hasattr(msg, "mode_machine") and msg.mode_machine != 5:
            self.guard.latch("unexpected G1 mode_machine")

    def _release_mode(self):
        # Only runs on an explicit, ready A -> init request, before motor writes.
        success = False
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            client = MotionSwitcherClient()
            client.SetTimeout(1.0)
            client.Init()
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                status, result = client.CheckMode()
                if status != 0 or result is None:
                    break
                if not result.get("name"):
                    success = True
                    break
                with self.lock:
                    if self.guard.latched:
                        break
                logger.info("Releasing high-level mode {} after operator A", result["name"])
                release_status, _ = client.ReleaseMode()
                if release_status != 0:
                    break
                time.sleep(0.1)
        except Exception as exc:
            logger.error("Mode release failed: {}", exc)
        with self.lock:
            self._takeover_result = success

    def _write(self, cmd):
        if self.read_only:
            return
        for i in range(self.guard.count):
            motor = self.low_cmd.motor_cmd[i]
            motor.q, motor.dq, motor.tau = float(cmd.q_target[i]), float(cmd.dq_target[i]), float(cmd.tau_ff[i])
            motor.kp, motor.kd = float(cmd.kp[i]), float(cmd.kd[i])
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        if not self.low_cmd_unitree_pub.Write(self.low_cmd):
            self.guard.latch("DDS command write failed")
        self.writes += 1

    def _step(self):
        with self.lock:
            self._observe_latest()
            while self._reception_gaps:
                gap_ns, ticks = self._reception_gaps.pop(0)
                logger.warning("LowState reception gap {:.0f} ms ({} robot ticks)", gap_ns / 1e6, ticks)
            try:
                data = self.low_cmd_zmq_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                data = None
            if data is not None:
                try:
                    self.guard.accept_command(LowCmdMessage.from_bytes(data), time.monotonic_ns())
                except Exception as exc:
                    self.guard.latch(f"malformed command: {exc}")
            if self.guard.ready(time.monotonic_ns()) and not self._tracker_ready_logged:
                logger.info("Tracker and LowState ready. Operator may press A to begin shaping under gantry protection.")
                self._tracker_ready_logged = True
            if self._takeover_result is not None:
                self.guard.takeover_finished(self._takeover_result, time.monotonic_ns())
                self._takeover_result = None
                self._takeover_thread = None
            cmd = self.guard.output(time.monotonic_ns())
            if self.guard.arm_pending and self._takeover_thread is None and not self.read_only:
                logger.info("Tracker ready; operator requested low-level takeover")
                self._takeover_thread = threading.Thread(target=self._release_mode, daemon=True)
                self._takeover_thread.start()
            if cmd is not None:
                self._write(cmd)
            if self.guard.state is not None:
                self.guard.state.stop_epoch = self.guard.stop_epoch
                try:
                    self.low_state_zmq_pub.send(self.guard.state.to_bytes(), flags=zmq.DONTWAIT)
                except zmq.Again:
                    pass
            if self.guard.reason != self._last_reason:
                logger.info("Bridge state: {} (stop epoch {})", self.guard.reason, self.guard.stop_epoch)
                self._last_reason = self.guard.reason

    def run(self, max_runtime_s=None, state_log=None):
        logger.info("Bridge on {}; read_only={}; standby emits no motor command", self.interface, self.read_only)
        start = next_time = time.monotonic()
        try:
            while max_runtime_s is None or time.monotonic() - start < max_runtime_s:
                self._step()
                next_time = max(next_time + self.dt, time.monotonic())
                time.sleep(max(0, next_time - time.monotonic()))
        except KeyboardInterrupt:
            pass
        finally:
            # An orderly local shutdown sends damping before closing DDS.
            with self.lock:
                self.guard.latch("bridge shutdown")
            if self._takeover_thread is not None:
                self._takeover_thread.join(timeout=6.0)
                with self.lock:
                    if self._takeover_result is not None:
                        self.guard.takeover_finished(self._takeover_result, time.monotonic_ns())
            for _ in range(40 if self.guard.active else 0):
                with self.lock:
                    self._write(self.guard.output(time.monotonic_ns()))
                time.sleep(self.dt)
            if state_log:
                state = self.guard.state
                report = dict(interface=self.interface, read_only=self.read_only,
                              samples=self.samples, motor_writes=self.writes,
                              tick=state.tick if state else None, stop_epoch=self.guard.stop_epoch,
                              joint_positions=state.joint_positions.tolist() if state else None,
                              quaternion=state.quaternion.tolist() if state else None)
                Path(state_log).parent.mkdir(parents=True, exist_ok=True)
                Path(state_log).write_text(json.dumps(report, indent=2) + "\n")
            self._closing = True
            reader_closed = close_sdk_channel(self.low_state_unitree_sub)
            writer_closed = close_sdk_channel(self.low_cmd_unitree_pub)
            self._sdk_close_timed_out = not (reader_closed and writer_closed)
            self.low_state_zmq_pub.close(linger=0)
            self.gamepad_zmq_pub.close(linger=0)
            self.low_cmd_zmq_sub.close(linger=0)


@dataclass
class Args:
    robot: str = "g1"
    rate: float = 200.0
    interface: str = UNITREE_INTERFACE
    read_only: bool = False
    timeout_s: float = 0.1
    target_margin_rad: float = PROTEUS_TARGET_MARGIN_RAD
    """How far past the joint range a policy target may point; the policy YAML's soft limits must stay inside."""
    hold_through_gaps: bool = False
    """Keep the last PD targets through LowState / command gaps instead of latching damping."""
    max_runtime_s: float | None = None
    state_log: str | None = None


def main(args: Args):
    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    cfg = replace(get_robot_cfg(args.robot), low_state_bind_addr="127.0.0.1", low_cmd_host="127.0.0.1")
    bridge = RealBridge(cfg, args.rate, args.interface, read_only=args.read_only, timeout_s=args.timeout_s,
                        target_margin_rad=args.target_margin_rad, hold_through_gaps=args.hold_through_gaps)
    bridge.run(args.max_runtime_s, args.state_log)
    exit_after_sdk_timeout(bridge._sdk_close_timed_out)


if __name__ == "__main__":
    main(tyro.cli(Args))
