"""Galbot ET1 HDEP <-> local ZMQ bridge with deferred takeover and independent stop arbitration.

Runs on the ET1 board. It serves the same ZMQ LowState/LowCmd contract as scripts/g1/real_bridge.py
and reuses BridgeSafety: startup only reads, a ready Tracker plus a fresh operator A press is needed
before enter_debug_mode() and the first motor write, and Select (or L1 + Down, Galbot's damping
chord), command/state timeouts or invalid commands latch. Keys from scripts/et1/operator_keyboard.py
(PORTS["operator"], dropped after OPERATOR_FRESH_S) are ORed into the handset, which is re-published
on PORTS["gamepad"] in the Unitree key layout for the Tracker's bridge_joystick controller.

HDEP is reached through scripts/et1/hdep_shim.cpp (built on the board into libet1_hdep_shim.so),
because hdep_py ships only for x86_64 / CPython 3.11. The process exits without HDEP teardown.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import struct
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import tyro
import zmq
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.robot_io.bridge_safety import BridgeSafety
from sim2real.utils.common import PORTS, LowCmdMessage, LowStateMessage
from sim2real.utils.operator_keys import OPERATOR_FORMAT

MAX_SLOTS = 64
# HDEP GamepadButton bit -> Unitree wireless_remote key bit (the layout BridgeSafety
# and UnitreeJoystickController decode).
HDEP_TO_UNITREE_KEYS = {0: 8, 1: 9, 2: 10, 3: 11, 4: 12, 5: 14, 6: 15, 7: 13, 8: 1, 9: 5, 10: 0, 11: 4, 12: 2, 13: 3}
HDEP_L1, HDEP_DOWN = 1 << 8, 1 << 5
UNITREE_SELECT = 1 << 3
# Gamepad message: HDEP gamepad stamp (ns), Unitree keys, lx, ly, rx, ry.
GAMEPAD_FORMAT = "<QIffff"
# Operator keys (sim2real/utils/operator_keys.py) are tracked per sender and expire after this.
OPERATOR_FRESH_S = 0.3
# After enter_debug_mode, how long to wait for the Tracker to resume sending commands.
TAKEOVER_RESUME_S = 0.5


def unitree_keys(hdep_buttons: int) -> int:
    keys = 0
    for hdep_bit, unitree_bit in HDEP_TO_UNITREE_KEYS.items():
        if hdep_buttons & (1 << hdep_bit):
            keys |= 1 << unitree_bit
    if hdep_buttons & HDEP_L1 and hdep_buttons & HDEP_DOWN:
        keys |= UNITREE_SELECT
    return keys


def slot_map(hdep_names: list[str], pr_valid: list[bool], joint_names: tuple[str, ...]) -> np.ndarray:
    """HDEP slot of each robot-config joint; every PR-valid HDEP joint must be one of them."""
    by_name = {f"{name}_joint": i for i, (name, valid) in enumerate(zip(hdep_names, pr_valid)) if valid}
    missing = [n for n in joint_names if n not in by_name]
    extra = sorted(set(by_name) - set(joint_names))
    if missing or extra:
        raise ValueError(f"HDEP PR joints do not match the robot config: missing {missing}, extra {extra}")
    return np.asarray([by_name[n] for n in joint_names], dtype=np.int64)


class Hdep:
    """ctypes view of libet1_hdep_shim.so."""

    def __init__(self, shim: str, bridge_library: str, interface_prefix: str, startup_timeout_ms: int = 5000):
        lib = ctypes.CDLL(shim)
        f64 = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        err = (ctypes.c_char_p, ctypes.c_int)
        lib.et1_open.restype = ctypes.c_void_p
        lib.et1_open.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, *err)
        lib.et1_joint_layout.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
                                         np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
                                         ctypes.POINTER(ctypes.c_uint32), *err)
        lib.et1_read.argtypes = (ctypes.c_void_p, f64, f64, f64, f64, f64, f64,
                                 np.ctypeslib.ndpointer(np.uint64, flags="C_CONTIGUOUS"),
                                 ctypes.POINTER(ctypes.c_uint32),
                                 np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS"), *err)
        lib.et1_enter_debug_mode.argtypes = (ctypes.c_void_p, *err)
        lib.et1_write.argtypes = (ctypes.c_void_p, f64, f64, f64, f64, f64, *err)
        self.lib = lib
        self._err = ctypes.create_string_buffer(512)
        self.handle = lib.et1_open(bridge_library.encode(), interface_prefix.encode(), startup_timeout_ms,
                                   self._err, len(self._err))
        if not self.handle:
            raise RuntimeError(f"HDEP initialize failed: {self._err.value.decode()}")
        self.q, self.dq, self.tau = (np.zeros(MAX_SLOTS) for _ in range(3))
        self.quat, self.gyro, self.accel = np.zeros(4), np.zeros(3), np.zeros(3)
        self.stamps = np.zeros(3, dtype=np.uint64)
        self.buttons = ctypes.c_uint32()
        self.sticks = np.zeros(4, dtype=np.float32)

    def _check(self, code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"HDEP {what} failed: {self._err.value.decode()}")

    def joint_layout(self) -> tuple[list[str], list[bool]]:
        names = ctypes.create_string_buffer(64 * 64)
        valid = np.zeros(MAX_SLOTS, dtype=np.uint8)
        count = ctypes.c_uint32()
        self._check(self.lib.et1_joint_layout(self.handle, names, len(names), valid, ctypes.byref(count),
                                              self._err, len(self._err)), "joint layout")
        return names.value.decode().split("\n")[: count.value], [bool(v) for v in valid[: count.value]]

    def read(self) -> None:
        self._check(self.lib.et1_read(self.handle, self.q, self.dq, self.tau, self.quat, self.gyro, self.accel,
                                      self.stamps, ctypes.byref(self.buttons), self.sticks,
                                      self._err, len(self._err)), "read")

    def enter_debug_mode(self) -> None:
        self._check(self.lib.et1_enter_debug_mode(self.handle, self._err, len(self._err)), "enter_debug_mode")

    def write(self, q, dq, tau, kp, kd) -> None:
        self._check(self.lib.et1_write(self.handle, q, dq, tau, kp, kd, self._err, len(self._err)), "write")


class ET1Bridge:
    def __init__(self, robot_cfg: RobotCfg, hdep: Hdep, *, rate_hz=200.0, read_only=False, timeout_s=0.1,
                 gamepad_port: int = PORTS["gamepad"], operator_port: int = PORTS["operator"]):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.robot_cfg, self.hdep = robot_cfg, hdep
        self.dt = 1.0 / rate_hz
        self.read_only = read_only
        self.guard = BridgeSafety(robot_cfg, timeout_s)
        self.slots = slot_map(*hdep.joint_layout(), robot_cfg.joint_names)
        self.tick = 0
        self._joint_stamp = None
        self._takeover_requested = False
        self._last_reason = None
        self._tracker_ready_logged = False
        self.samples = self.writes = 0
        self.debug_mode_s = None
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
        self._operators: dict[int, tuple[int, int]] = {}  # sender id -> (sent ns, keys)

    def _operator_keys(self) -> int:
        for _ in range(64):  # drain; senders publish at 20 Hz
            try:
                sent_ns, keys, sender = struct.unpack(OPERATOR_FORMAT, self.operator_sub.recv(flags=zmq.DONTWAIT))
            except zmq.Again:
                break
            except struct.error as exc:
                logger.warning("Malformed operator message: {}", exc)
                continue
            self._operators[sender] = (sent_ns, keys)
        now = time.monotonic_ns()
        fresh = {s: v for s, v in self._operators.items() if 0 <= now - v[0] <= OPERATOR_FRESH_S * 1e9}
        self._operators = fresh
        merged = 0
        for _, keys in fresh.values():
            merged |= keys
        return merged

    def _observe(self) -> None:
        h = self.hdep
        h.read()
        self.samples += 1
        if self._joint_stamp != int(h.stamps[0]):
            self._joint_stamp = int(h.stamps[0])
            self.tick = (self.tick + 1) & 0xFFFFFFFF
        state = LowStateMessage(h.quat, h.gyro, h.q[self.slots], h.dq[self.slots], h.tau[self.slots], self.tick)
        keys = unitree_keys(h.buttons.value) | self._operator_keys()
        self.guard.observe_state(state, keys, time.monotonic_ns())
        self.gamepad_pub.send(struct.pack(GAMEPAD_FORMAT, int(h.stamps[2]), keys, *h.sticks.tolist()),
                              flags=zmq.DONTWAIT)

    def _write(self, cmd: LowCmdMessage) -> None:
        if self.read_only:
            return
        arrays = [np.zeros(MAX_SLOTS) for _ in range(5)]  # unlisted slots: kp = kd = 0
        for full, part in zip(arrays, (cmd.q_target, cmd.dq_target, cmd.tau_ff, cmd.kp, cmd.kd)):
            full[self.slots] = part
        try:
            self.hdep.write(*arrays)
            self.writes += 1
        except RuntimeError as exc:
            self.guard.latch(str(exc))

    def _take_over(self) -> None:
        # Only after an explicit, ready operator A and before the first motor write.
        begin = time.monotonic()
        logger.info("Operator A with a ready Tracker: entering ET1 debug mode for low-level control")
        try:
            self.hdep.enter_debug_mode()
            success = True
        except RuntimeError as exc:
            logger.error("{}", exc)
            success = False
        self.debug_mode_s = time.monotonic() - begin
        logger.info("enter_debug_mode returned after {:.2f} s", self.debug_mode_s)
        # A command the Tracker sent just before the block waited in the queue through it and is now
        # as old as the block: drop it rather than latch on it as expired. The Tracker sends nothing
        # during the block (no LowState), so everything queued is from before it.
        self._discard_queued_commands()
        # The call blocks the loop (0.13-0.86 s on hardware). Without fresh LowState the Tracker skips its
        # steps, so its last command is as old as the block: keep publishing state until it sends again.
        # A Tracker that stopped stays stale and fails the takeover. No motor command is written here.
        deadline = time.monotonic() + TAKEOVER_RESUME_S
        while True:
            self._observe()
            self._publish_state()
            self._receive_command()
            if self.guard.ready(time.monotonic_ns()) or time.monotonic() >= deadline:
                break
            time.sleep(self.dt)
        self.guard.takeover_finished(success, time.monotonic_ns())

    def _publish_state(self) -> None:
        if self.guard.state is None:
            return
        self.guard.state.stop_epoch = self.guard.stop_epoch
        try:
            self.state_pub.send(self.guard.state.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass

    def _discard_queued_commands(self) -> None:
        while True:
            try:
                self.cmd_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                return

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
        try:
            self._observe()
        except RuntimeError as exc:
            self.guard.latch(str(exc))
        self._receive_command()
        if self.guard.ready(time.monotonic_ns()) and not self._tracker_ready_logged:
            logger.info("Tracker and LowState ready. Operator may press A to begin shaping under gantry protection.")
            self._tracker_ready_logged = True
        if self.guard.arm_pending and not self.read_only:
            self._take_over()
        cmd = self.guard.output(time.monotonic_ns())
        if cmd is not None:
            self._write(cmd)
        self._publish_state()
        if self.guard.reason != self._last_reason:
            logger.info("Bridge state: {} (stop epoch {})", self.guard.reason, self.guard.stop_epoch)
            self._last_reason = self.guard.reason

    def run(self, max_runtime_s=None, state_log=None) -> None:
        logger.info("ET1 bridge; read_only={}; standby emits no motor command", self.read_only)
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
            for _ in range(40 if self.guard.active else 0):
                try:
                    self._observe()
                except RuntimeError:
                    pass
                self._write(self.guard.output(time.monotonic_ns()))
                time.sleep(self.dt)
            if state_log:
                state = self.guard.state
                report = dict(read_only=self.read_only, samples=self.samples, motor_writes=self.writes,
                              loop_overruns=overruns, debug_mode_s=self.debug_mode_s, tick=self.tick,
                              stop_epoch=self.guard.stop_epoch, reason=self.guard.reason,
                              joint_names=list(self.robot_cfg.joint_names), hdep_slots=self.slots.tolist(),
                              joint_positions=state.joint_positions.tolist() if state else None,
                              quaternion=state.quaternion.tolist() if state else None)
                Path(state_log).parent.mkdir(parents=True, exist_ok=True)
                Path(state_log).write_text(json.dumps(report, indent=2) + "\n")
            self.state_pub.close(linger=0)
            self.gamepad_pub.close(linger=0)
            self.cmd_sub.close(linger=0)
            self.operator_sub.close(linger=0)


@dataclass
class Args:
    shim: str
    """libet1_hdep_shim.so built from scripts/et1/hdep_shim.cpp."""
    bridge_library: str
    """libhdep_bridge_galbot_et1.so.2.2.0 from et1-dep."""
    interface_prefix: str = "192.168.123"
    robot: str = "galbot_et1"
    rate: float = 200.0
    read_only: bool = False
    timeout_s: float = 0.1
    max_runtime_s: float | None = None
    state_log: str | None = None


def main(args: Args) -> None:
    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    cfg = replace(get_robot_cfg(args.robot), low_state_bind_addr="127.0.0.1", low_cmd_host="127.0.0.1")
    code = 0
    try:
        hdep = Hdep(args.shim, args.bridge_library, args.interface_prefix)
        ET1Bridge(cfg, hdep, rate_hz=args.rate, read_only=args.read_only, timeout_s=args.timeout_s).run(
            args.max_runtime_s, args.state_log)
    except Exception:
        logger.exception("ET1 bridge failed")
        code = 1
    # HDEP teardown can race a CycloneDDS callback; leave it to process exit.
    os._exit(code)


if __name__ == "__main__":
    main(tyro.cli(Args))
