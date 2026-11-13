"""X2 AimDK bridge: BridgeSafety arbitration, takeover and HAL clamping with a fake HAL (no ROS)."""
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

from sim2real.config.robots import X2_CFG
from sim2real.rl_policy.robot_io.base import RobotState
from sim2real.rl_policy.robot_io.x2 import X2StateCommand
from sim2real.utils.common import LowCmdMessage
from sim2real.utils.operator_keys import KEY_A, KEY_SELECT

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("aimdk_bridge", REPO_ROOT / "scripts/x2/aimdk_bridge.py")
aimdk_bridge = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = aimdk_bridge
spec.loader.exec_module(aimdk_bridge)
N = len(X2_CFG.joint_names)


class FakeHal:
    def __init__(self):
        self.codec = X2StateCommand(X2_CFG, timeout_s=1.0)
        self.q = np.asarray(X2_CFG.default_qpos[7:], dtype=np.float32).copy()
        self.stamp = 0.0
        self.published = []
        self.fail = None

    def read(self):
        if self.fail:
            raise RuntimeError("X2 feedback failed") from ValueError(self.fail)
        self.stamp += 0.001
        qpos = np.zeros(7 + N, np.float32)
        qpos[3] = 1.0
        qpos[7:] = self.q
        return RobotState(qpos=qpos, qvel=np.zeros(6 + N, np.float32), joint_torque=np.zeros(N, np.float32),
                          tick=0), self.stamp

    def publish(self, cmd):
        self.published.append(cmd)


@pytest.fixture
def make_bridge(tmp_path):
    created = []

    def make(read_only=False, takeover_cmd="true"):
        port = 46600 + 10 * len(created)
        cfg = replace(X2_CFG, low_state_bind_addr="127.0.0.1", low_state_port=port,
                      low_cmd_host="127.0.0.1", low_cmd_port=port + 1)
        hal = FakeHal()
        bridge = aimdk_bridge.X2Bridge(cfg, hal, read_only=read_only, timeout_s=0.5, takeover_cmd=takeover_cmd,
                                       gamepad_port=port + 2, operator_port=port + 3)
        keys = {"value": 0}
        bridge._operator_keys = lambda: keys["value"]
        cmd_pub = zmq.Context.instance().socket(zmq.PUB)
        cmd_pub.bind(f"tcp://127.0.0.1:{port + 1}")
        pad_sub = zmq.Context.instance().socket(zmq.SUB)
        pad_sub.setsockopt(zmq.SUBSCRIBE, b"")
        pad_sub.setsockopt(zmq.RCVTIMEO, 500)
        pad_sub.connect(f"tcp://127.0.0.1:{port + 2}")
        time.sleep(0.2)
        created.append((bridge, cmd_pub, pad_sub))
        return bridge, hal, cmd_pub, pad_sub, keys

    yield make
    for bridge, cmd_pub, pad_sub in created:
        for sock in (bridge.state_pub, bridge.gamepad_pub, bridge.cmd_sub, bridge.operator_sub, cmd_pub, pad_sub):
            sock.close(linger=0)


def _send(pub, mode, sequence, generation=1, q=None):
    q = np.asarray(X2_CFG.default_qpos[7:], dtype=np.float32) if q is None else q
    msg = LowCmdMessage(q, np.zeros(N), np.zeros(N), np.full(N, 20.0, np.float32), np.full(N, 1.0, np.float32),
                        source_time_ns=time.monotonic_ns(), sequence=sequence, control_mode=mode,
                        control_generation=generation)
    pub.send(msg.to_bytes())
    time.sleep(0.02)


def _press(bridge, keys, value):
    keys["value"] = value
    bridge.step()
    keys["value"] = 0
    bridge.step()


@pytest.mark.parametrize("read_only", [True, False])
def test_standby_emits_nothing_until_ready_tracker_and_a(make_bridge, read_only):
    bridge, hal, cmd_pub, pad_sub, keys = make_bridge(read_only)
    bridge.step()
    _press(bridge, keys, KEY_A)  # no Tracker yet: ignored
    assert bridge.takeover_s is None and not hal.published
    _send(cmd_pub, "zero", 1)
    bridge.step()
    assert bridge.guard.ready(time.monotonic_ns()) and not hal.published
    _press(bridge, keys, KEY_A)
    _send(cmd_pub, "init", 2, generation=2)
    bridge.step()
    if read_only:
        assert bridge.takeover_s is None and not hal.published
        return
    assert bridge.takeover_s is not None and bridge.guard.active
    _send(cmd_pub, "init", 3, generation=2)
    bridge.step()
    cmd = hal.published[-1]
    np.testing.assert_allclose(cmd.q_target, X2_CFG.default_qpos[7:], atol=1e-6)
    assert np.all(cmd.kp == 20.0)
    _press(bridge, keys, KEY_SELECT)
    assert bridge.guard.latched and bridge.guard.reason == "Select"
    assert len(struct.unpack(aimdk_bridge.GAMEPAD_FORMAT, pad_sub.recv())) == 6


def test_failed_takeover_latches_without_commands(make_bridge):
    bridge, hal, cmd_pub, _, keys = make_bridge(takeover_cmd="false")
    bridge.step()
    _send(cmd_pub, "zero", 1)
    bridge.step()
    _press(bridge, keys, KEY_A)
    _send(cmd_pub, "init", 2, generation=2)
    bridge.step()
    assert bridge.guard.latched and not bridge.guard.active
    assert not hal.published


def test_feedback_error_latches_damping_after_takeover(make_bridge):
    bridge, hal, cmd_pub, _, keys = make_bridge()
    bridge.step()
    _send(cmd_pub, "zero", 1)
    bridge.step()
    _press(bridge, keys, KEY_A)
    _send(cmd_pub, "init", 2, generation=2)
    bridge.step()
    assert bridge.guard.active
    hal.fail = "joint error"
    bridge.step()
    assert bridge.guard.latched and bridge.guard.fault
    cmd = hal.published[-1]
    assert np.all(cmd.kp == 0.0) and np.all(cmd.kd > 0.0)


def test_hal_publish_clamps_to_vendor_range_and_zeroes_head_pitch():
    published = {}

    class Pub:
        def __init__(self, group):
            self.group = group

        def publish(self, msg):
            published[self.group] = msg

    hal = aimdk_bridge.AimdkHal.__new__(aimdk_bridge.AimdkHal)
    hal.codec = X2StateCommand(X2_CFG, timeout_s=1.0)
    hal.publishers = {g: Pub(g) for g in hal.codec.group_ids}
    hal.node = type("Node", (), {"get_clock": lambda self: type("C", (), {"now": lambda s: type(
        "T", (), {"to_msg": lambda t: "stamp"})()})()})()
    hal.joint_type = lambda: type("J", (), {})()
    hal.array_type = lambda: type("A", (), {"header": type("H", (), {})(), "joints": []})()
    q = np.zeros(N, np.float32)
    q[X2_CFG.joint_names.index("waist_yaw_joint")] = 100.0
    ones = np.ones(N, np.float32)
    hal.publish(LowCmdMessage(q, 0 * ones, 0 * ones, 20 * ones, ones))
    waist = {j.name: j for j in published["waist"].joints}
    assert waist["waist_yaw_joint"].position == pytest.approx(hal.codec.upper[X2_CFG.joint_names.index("waist_yaw_joint")])
    head = {j.name: j for j in published["head"].joints}
    assert head["head_pitch_joint"].stiffness == 0.0 and head["head_yaw_joint"].stiffness == 20.0


def test_x2_bridge_accepts_damped_leg_kd_up_to_25(make_bridge):
    bridge, hal, cmd_pub, _, keys = make_bridge()
    assert bridge.guard.max_kd == 25.0
    bridge.step()
    msg_kd = np.full(N, 15.3, np.float32)  # knee kd x2 in policy-pc-damped.yaml
    q = np.asarray(X2_CFG.default_qpos[7:], dtype=np.float32)
    cmd_pub.send(LowCmdMessage(q, np.zeros(N), np.zeros(N), np.full(N, 20.0, np.float32), msg_kd,
                               source_time_ns=time.monotonic_ns(), sequence=1, control_mode="zero",
                               control_generation=1).to_bytes())
    time.sleep(0.02)
    bridge.step()
    assert bridge.guard.ready(time.monotonic_ns()) and not bridge.guard.latched
