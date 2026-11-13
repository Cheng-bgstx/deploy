"""ET1 HDEP bridge: handset mapping, joint slots, and BridgeSafety arbitration with a fake HDEP."""
from __future__ import annotations

import importlib.util
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zmq

from sim2real.config.robots import ET1_CFG
from sim2real.utils.common import LowCmdMessage

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hdep_bridge", REPO_ROOT / "scripts/et1/hdep_bridge.py")
hdep_bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hdep_bridge
spec.loader.exec_module(hdep_bridge)

# HDEP order on the board: 26 PR joints, then the four raw ankle motors.
HDEP_NAMES = [n.removesuffix("_joint") for n in ET1_CFG.joint_names] + [
    "left_ankle_motor_a", "left_ankle_motor_b", "right_ankle_motor_a", "right_ankle_motor_b"]
HDEP_VALID = [True] * 26 + [False] * 4
A, DOWN, L1, START, SELECT = 1 << 0, 1 << 5, 1 << 8, 1 << 12, 1 << 13


class FakeHdep:
    def __init__(self):
        self.q, self.dq, self.tau = (np.zeros(64) for _ in range(3))
        self.q[:26] = ET1_CFG.default_qpos[7:]
        self.quat, self.gyro, self.accel = np.array([1.0, 0, 0, 0]), np.zeros(3), np.array([0, 0, 9.8])
        self.stamps = np.zeros(3, dtype=np.uint64)
        self.buttons = type("B", (), {"value": 0})()
        self.sticks = np.zeros(4, dtype=np.float32)
        self.debug_calls = 0
        self.on_debug = None  # runs inside enter_debug_mode, which blocks the bridge loop
        self.writes = []

    def joint_layout(self):
        return HDEP_NAMES, HDEP_VALID

    def read(self):
        self.stamps[:] += 1_000_000

    def enter_debug_mode(self):
        self.debug_calls += 1
        if self.on_debug is not None:
            self.on_debug()

    def write(self, q, dq, tau, kp, kd):
        self.writes.append((q.copy(), kp.copy(), kd.copy()))


def test_handset_maps_to_unitree_keys():
    assert hdep_bridge.unitree_keys(A) == 1 << 8
    assert hdep_bridge.unitree_keys(START) == 1 << 2
    assert hdep_bridge.unitree_keys(SELECT) == 1 << 3
    # Galbot's damping chord L1 + Down is a Select for the bridge and the Tracker.
    assert hdep_bridge.unitree_keys(L1 | DOWN) & (1 << 3)
    assert not hdep_bridge.unitree_keys(DOWN) & (1 << 3)


def test_slot_map_requires_exactly_the_config_joints():
    slots = hdep_bridge.slot_map(HDEP_NAMES, HDEP_VALID, ET1_CFG.joint_names)
    np.testing.assert_array_equal(slots, np.arange(26))
    with pytest.raises(ValueError):
        hdep_bridge.slot_map(HDEP_NAMES[:25], HDEP_VALID[:25], ET1_CFG.joint_names)


@pytest.fixture
def bridge_factory():
    created = []

    def make(read_only):
        port = 46100 + 10 * len(created)
        cfg = replace(ET1_CFG, low_state_bind_addr="127.0.0.1", low_state_port=port,
                      low_cmd_host="127.0.0.1", low_cmd_port=port + 1)
        fake = FakeHdep()
        bridge = hdep_bridge.ET1Bridge(cfg, fake, read_only=read_only, timeout_s=0.5, gamepad_port=port + 2,
                                       operator_port=port + 3)
        cmd_pub = zmq.Context.instance().socket(zmq.PUB)
        cmd_pub.bind(f"tcp://127.0.0.1:{port + 1}")
        pad_sub = zmq.Context.instance().socket(zmq.SUB)
        pad_sub.setsockopt(zmq.SUBSCRIBE, b"")
        pad_sub.setsockopt(zmq.RCVTIMEO, 500)
        pad_sub.connect(f"tcp://127.0.0.1:{port + 2}")
        time.sleep(0.2)
        created.append((bridge, cmd_pub, pad_sub))
        return bridge, fake, cmd_pub, pad_sub

    yield make
    for bridge, cmd_pub, pad_sub in created:
        for sock in (bridge.state_pub, bridge.gamepad_pub, bridge.cmd_sub, bridge.operator_sub, cmd_pub, pad_sub):
            sock.close(linger=0)


def _send(pub, mode, sequence, generation=1):
    q = np.asarray(ET1_CFG.default_qpos[7:], dtype=np.float32)
    kp, kd = np.full(26, 20.0, np.float32), np.full(26, 1.0, np.float32)
    msg = LowCmdMessage(q, np.zeros(26), np.zeros(26), kp, kd, source_time_ns=time.monotonic_ns(),
                        sequence=sequence, control_mode=mode, control_generation=generation)
    pub.send(msg.to_bytes())
    time.sleep(0.02)


def _press(bridge, fake, buttons):
    fake.buttons.value = buttons
    bridge.step()
    fake.buttons.value = 0
    bridge.step()


@pytest.mark.parametrize("read_only", [True, False])
def test_takeover_needs_ready_tracker_and_a_press(bridge_factory, read_only):
    bridge, fake, cmd_pub, pad_sub = bridge_factory(read_only)
    bridge.step()
    _press(bridge, fake, A)  # no Tracker command yet: not ready, ignored
    assert fake.debug_calls == 0 and not fake.writes
    _send(cmd_pub, "zero", 1)
    bridge.step()
    assert bridge.guard.ready(time.monotonic_ns())
    assert not fake.writes  # standby emits no motor command
    _press(bridge, fake, A)
    _send(cmd_pub, "init", 2, generation=2)
    bridge.step()
    if read_only:
        assert fake.debug_calls == 0 and not fake.writes
        return
    assert fake.debug_calls == 1
    _send(cmd_pub, "init", 3, generation=2)
    bridge.step()
    q, kp, kd = fake.writes[-1]
    np.testing.assert_allclose(q[:26], ET1_CFG.default_qpos[7:], atol=1e-6)
    assert np.all(kp[:26] == 20.0) and np.all(kp[26:] == 0.0) and np.all(kd[26:] == 0.0)
    # Select latches into the stop; L1 + Down does the same through the key mapping.
    _press(bridge, fake, L1 | DOWN)
    assert bridge.guard.latched
    pad = struct.unpack(hdep_bridge.GAMEPAD_FORMAT, pad_sub.recv())
    assert len(pad) == 6


def test_operator_keys_merge_per_sender_and_expire(bridge_factory):
    from sim2real.utils.operator_keys import KEY_A, KEY_SELECT, OperatorKeyPublisher

    bridge, fake, cmd_pub, pad_sub = bridge_factory(False)
    port = int(bridge.operator_sub.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1])
    idle, presser = OperatorKeyPublisher(port=port), OperatorKeyPublisher(port=port)
    try:
        for _ in range(10):  # PUB connect drops messages until subscribed
            idle.send(0)
            presser.send(0)
            time.sleep(0.03)
            bridge._operator_keys()
        # An idle console keeps publishing "released" while another sender holds A:
        # the merge must see one continuous press, i.e. a single rising edge.
        edges, previous = 0, 0
        for i in range(12):
            presser.send(KEY_A if i < 6 else 0)
            idle.send(0)
            time.sleep(0.02)
            keys = bridge._operator_keys()
            edges += bool(keys & KEY_A and not previous & KEY_A)
            previous = keys
        assert edges == 1
        presser.send(KEY_SELECT)
        time.sleep(0.03)
        bridge.step()
        assert bridge.guard.latched and bridge.guard.reason == "Select"
        # A sender that stops publishing is dropped after the freshness window.
        bridge._operators = {7: (time.monotonic_ns() - int(1e9), KEY_A)}
        assert bridge._operator_keys() & KEY_A == 0
    finally:
        idle.close()
        presser.close()


@pytest.mark.parametrize("tracker_alive", [True, False])
def test_takeover_waits_for_the_tracker_to_resume_after_a_blocking_debug_mode_entry(bridge_factory, tracker_alive):
    # enter_debug_mode blocked 0.84 s on hardware. The Tracker sends a command only after a fresh
    # LowState, so it is silent during the block: the bridge must publish state again and wait for
    # its next command. A Tracker that has stopped must still fail the takeover.
    import threading

    bridge, fake, cmd_pub, _ = bridge_factory(False)
    state_port = int(bridge.state_pub.getsockopt(zmq.LAST_ENDPOINT).decode().rsplit(":", 1)[1])
    mode = {"name": "zero", "generation": 1, "alive": True, "sequence": 0}
    stop = threading.Event()

    def tracker():
        sub = zmq.Context.instance().socket(zmq.SUB)
        sub.setsockopt(zmq.SUBSCRIBE, b"")
        sub.setsockopt(zmq.RCVTIMEO, 50)
        sub.connect(f"tcp://127.0.0.1:{state_port}")
        while not stop.is_set():
            try:
                sub.recv()
            except zmq.Again:
                continue  # no fresh state: like ZMQRobotIO, skip the step
            if mode["alive"]:
                mode["sequence"] += 1
                _send(cmd_pub, mode["name"], mode["sequence"], generation=mode["generation"])
        sub.close(linger=0)

    thread = threading.Thread(target=tracker, daemon=True)
    thread.start()
    try:
        for _ in range(30):
            bridge.step()
            time.sleep(0.01)
            if bridge.guard.ready(time.monotonic_ns()):
                break
        assert bridge.guard.ready(time.monotonic_ns())
        mode.update(name="init", generation=2)
        time.sleep(0.05)

        def block():
            # The Tracker's reply to the last state before the block lands in the queue and expires there
            # (hardware 2026-11-11: it latched the takeover as an expired command).
            mode["sequence"] += 1
            _send(cmd_pub, mode["name"], mode["sequence"], generation=mode["generation"])
            time.sleep(0.8)  # > timeout_s = 0.5
            mode["alive"] = tracker_alive

        fake.on_debug = block
        fake.buttons.value = A
        for _ in range(20):
            bridge.step()
            fake.buttons.value = 0
            if fake.debug_calls:
                break
            time.sleep(0.01)
        assert fake.debug_calls == 1
        assert bridge.guard.active
        assert bridge.guard.latched is not tracker_alive
        if tracker_alive:
            assert fake.writes and np.all(fake.writes[-1][1][:26] == 20.0)
    finally:
        stop.set()
        thread.join(timeout=1)
