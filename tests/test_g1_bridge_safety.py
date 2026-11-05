import numpy as np
import pytest

from sim2real.config.robots import G1_CFG
from sim2real.rl_policy.robot_io.bridge_safety import PROTEUS_TARGET_MARGIN_RAD, BridgeSafety
from sim2real.utils.common import LowCmdMessage, LowStateMessage


def state(tick=1):
    return LowStateMessage(np.array([1, 0, 0, 0]), np.zeros(3), np.array(G1_CFG.default_qpos[7:]), np.zeros(29), tick=tick)


def command(t, seq=1, mode="zero", generation=0):
    return LowCmdMessage(np.array(G1_CFG.default_qpos[7:]), np.zeros(29), np.zeros(29), np.full(29, 20), np.ones(29),
                         source_time_ns=t, sequence=seq, control_mode=mode, control_generation=generation)


def armed(*, timeout_s=0.1, target_margin_rad=0.0, hold_through_gaps=False):
    g = BridgeSafety(G1_CFG, timeout_s, target_margin_rad, hold_through_gaps)
    g.observe_state(state(), 0, 1_000_000_000)
    assert g.accept_command(command(1_000_000_000), 1_000_000_000)
    assert g.output(1_000_000_000) is None
    g.observe_state(state(2), 1 << 8, 1_010_000_000)
    assert g.accept_command(command(1_020_000_000, 2, "init", 1), 1_020_000_000)
    assert g.arm_pending
    assert g.output(1_020_000_000) is None
    g.takeover_finished(True, 1_020_000_000)
    assert g.output(1_020_000_000).control_mode == "init"
    return g


def test_ready_does_not_arm_without_a_and_fresh_init():
    g = BridgeSafety(G1_CFG)
    g.observe_state(state(), 1 << 8, 1_000_000_000)  # Too early: no model output.
    g.accept_command(command(1_010_000_000, mode="init", generation=1), 1_010_000_000)
    assert not g.arm_pending and g.output(1_010_000_000) is None


def test_select_overrides_tracker_and_requires_ack_then_new_a():
    g = armed()
    g.observe_state(state(3), (1 << 3) | (1 << 8), 1_030_000_000)
    out = g.output(1_030_000_000)
    np.testing.assert_array_equal(out.q_target, state().joint_positions)
    assert g.latched and not g.fault
    epoch = g.stop_epoch
    g.accept_command(command(1_040_000_000, 3, "policy", 2), 1_040_000_000)
    assert g.latched
    # Release buttons, acknowledge stop, then request a new shaping session.
    g.observe_state(state(4), 0, 1_050_000_000)
    g.accept_command(command(1_050_000_000, 4, "zero", 3), 1_050_000_000)
    g.observe_state(state(5), 1 << 8, 1_060_000_000)
    g.accept_command(command(1_070_000_000, 5, "init", 4), 1_070_000_000)
    assert not g.latched and g.active and g.stop_epoch == epoch


@pytest.mark.parametrize("failure", ["tracker", "state", "repeated_tick"])
def test_timeouts_use_damping_and_never_resume_automatically(failure):
    g = armed()
    if failure == "tracker":
        g.observe_state(state(3), 0, 1_130_000_001)
    else:
        if failure == "repeated_tick":
            g.observe_state(state(2), 0, 1_130_000_001)
        g.accept_command(command(1_130_000_001, 3, "policy", 2), 1_130_000_001)
    out = g.output(1_130_000_001)
    assert g.latched and g.fault
    np.testing.assert_array_equal(out.kp, 0)
    assert (out.kd > 0).all()
    g.observe_state(state(4), 0, 1_140_000_000)
    g.accept_command(command(1_140_000_000, 4, "policy", 3), 1_140_000_000)
    assert g.latched


@pytest.mark.parametrize("gap_ns", [150_000_000, 200_000_000, 240_000_000])
@pytest.mark.parametrize("delayed", ["state", "command"])
def test_bounded_jitter_continues_without_stop_or_manual_rearm(gap_ns, delayed):
    g = armed(timeout_s=.25)
    now = 1_020_000_000 + gap_ns
    if delayed == "state":
        cmd = command(now, 3, "policy", 2)
        cmd.q_target[0] += .1
        assert g.accept_command(cmd, now)
    else:
        g.observe_state(state(3), 0, now)
    out = g.output(now)
    assert out is g.command and not g.latched and not g.fault
    assert g.stop_epoch == 0
    # The next fresh update continues immediately; no A/init handshake.
    g.observe_state(state(4), 0, now + 1_000_000)
    cmd = command(now + 1_000_000, 4, "policy", 2)
    cmd.q_target[0] -= .1
    assert g.accept_command(cmd, now + 1_000_000)
    assert g.output(now + 1_000_000) is cmd
    assert not g.latched and g.stop_epoch == 0


@pytest.mark.parametrize("missing", ["state", "command"])
def test_sustained_loss_beyond_jitter_window_still_latches(missing):
    g = armed(timeout_s=.25)
    now = 1_280_000_000
    if missing == "state":
        assert g.accept_command(command(now, 3, "policy", 2), now)
    else:
        g.observe_state(state(3), 0, now)
    out = g.output(now)
    assert g.latched and g.fault
    np.testing.assert_array_equal(out.kp, 0)


def test_zmq_cached_state_survives_jitter_but_repeated_tick_cannot_hide_loss(monkeypatch):
    from types import SimpleNamespace
    from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO
    import sim2real.rl_policy.robot_io.zmq as io_module
    pending = []

    def recv(flags):
        if not pending:
            raise io_module.zmq.Again()
        return pending.pop(0)

    socket = SimpleNamespace(setsockopt=lambda *a: None, connect=lambda *a: None,
                             bind=lambda *a: None, recv=recv)
    context = SimpleNamespace(socket=lambda *a: socket)
    monkeypatch.setenv("SIM2REAL_IO_TIMEOUT_S", "0.25")
    io = ZMQRobotIO(G1_CFG, context=context)
    clock = [1_000_000_000]
    monkeypatch.setattr(io_module.time, "monotonic_ns", lambda: clock[0])
    assert io.read_state() is None
    pending.append(state(1).to_bytes())
    fresh = io.read_state()
    clock[0] += 200_000_000
    assert io.read_state() is fresh
    pending.append(state(1).to_bytes())
    assert io.read_state().tick == 1
    clock[0] = 1_250_000_000
    assert io.read_state() is not None
    clock[0] += 1
    assert io.read_state() is None
    pending.append(state(2).to_bytes())
    assert io.read_state().tick == 2


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf", "1.01"])
def test_zmq_rejects_unbounded_or_invalid_jitter_window(monkeypatch, timeout):
    from types import SimpleNamespace
    from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO
    monkeypatch.setenv("SIM2REAL_IO_TIMEOUT_S", timeout)
    with pytest.raises(ValueError, match="timeout_s"):
        ZMQRobotIO(G1_CFG, context=SimpleNamespace())


@pytest.mark.parametrize("failure", ["nan", "size", "limit", "gain", "expired", "sequence", "timestamp", "legacy"])
def test_bad_commands_latch_fault(failure):
    g = armed()
    cmd = command(1_030_000_000, 3, "policy", 2)
    if failure == "nan": cmd.q_target[0] = np.nan
    if failure == "size": cmd.kd = np.zeros(28)
    if failure == "limit": cmd.q_target[0] = 10
    if failure == "gain": cmd.kp[0] = -1
    if failure == "expired": cmd.source_time_ns -= 200_000_000
    if failure == "sequence": cmd.sequence = 2
    if failure == "timestamp": cmd.source_time_ns = 1_020_000_000
    if failure == "legacy": cmd.control_mode = None
    assert not g.accept_command(cmd, 1_030_000_000)
    assert g.latched and g.fault
    assert (g.output(1_030_000_000).kp == 0).all()


def test_policy_targets_may_point_past_the_joint_range_by_the_margin():
    # Proteus drives the G1 ankles up to ~0.8 rad past their range; training never clips targets.
    j = G1_CFG.joint_names.index("left_ankle_pitch_joint")
    upper = G1_CFG.joint_pos_upper_limit["left_ankle_pitch_joint"]
    g = armed(target_margin_rad=PROTEUS_TARGET_MARGIN_RAD)
    cmd = command(1_030_000_000, 3, "policy", 2)
    cmd.q_target[j] = upper + 0.8
    assert g.accept_command(cmd, 1_030_000_000) and not g.latched
    cmd = command(1_040_000_000, 4, "policy", 2)
    cmd.q_target[j] = upper + PROTEUS_TARGET_MARGIN_RAD + 0.1
    assert not g.accept_command(cmd, 1_040_000_000) and g.fault


def test_hold_through_gaps_keeps_the_last_targets_and_resumes():
    g = armed(hold_through_gaps=True)
    cmd = command(1_030_000_000, 3, "policy", 2)
    cmd.q_target[0] += 0.1
    assert g.accept_command(cmd, 1_030_000_000)
    # A 340 ms LowState gap (the Tracker keeps sending on its cached state): no damping, same PD targets.
    out = g.output(1_370_000_000)
    assert not g.latched and g.reason.startswith("LowState gap")
    assert out.q_target[0] == cmd.q_target[0] and (out.kp == 20).all()
    # Commands that arrive late after the gap are dropped without a stop.
    assert not g.accept_command(command(1_030_000_000, 4, "policy", 2), 1_370_000_000)
    assert not g.latched
    # Fresh state and commands resume control without a new A.
    g.observe_state(state(9), 0, 1_380_000_000)
    assert g.accept_command(command(1_380_000_000, 5, "policy", 2), 1_380_000_000)
    assert g.output(1_380_000_000).control_mode == "policy" and g.reason == "active"
    # A Tracker command gap holds the last targets as well.
    g.observe_state(state(10), 0, 1_600_000_000)
    out = g.output(1_600_000_000)
    assert not g.latched and g.reason.startswith("Tracker command gap") and (out.kp == 20).all()


def test_hold_through_gaps_never_resumes_after_a_long_silence_or_robot_restart():
    # A rebooting robot: silent for over a second, then back with a restarted tick. Old targets must not resume.
    g = armed(hold_through_gaps=True)
    assert g.accept_command(command(1_030_000_000, 3, "policy", 2), 1_030_000_000)
    g.output(1_500_000_000)
    assert not g.latched
    g.output(2_100_000_000)
    assert g.latched and g.fault and (g.output(2_100_000_000).kp == 0).all()
    g = armed(hold_through_gaps=True)
    assert g.accept_command(command(1_030_000_000, 3, "policy", 2), 1_030_000_000)
    g.observe_state(state(1), 0, 1_040_000_000)  # tick went back from 2
    assert g.latched and g.fault


def test_hold_through_gaps_still_stops_on_invalid_commands_and_select():
    g = armed(hold_through_gaps=True)
    bad = command(1_030_000_000, 3, "policy", 2)
    bad.q_target[0] = np.nan
    assert not g.accept_command(bad, 1_030_000_000) and g.fault
    g = armed(hold_through_gaps=True)
    g.observe_state(state(3), 1 << 3, 1_030_000_000)
    assert g.latched and not g.fault


def test_g1_overlay_bounds_targets_by_what_the_bridge_accepts(tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / "scripts/g1/prepare_proteus_deployment.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    names = list(G1_CFG.joint_names)
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(dict(
        proteus=dict(reference_forcing=True), joint_names_simulation=names, policy_joint_names=names,
        default_joint_pos={n: 0.0 for n in names}, joint_kp={n: 1.0 for n in names}, joint_kd={n: 0.1 for n in names},
    )))  # fmt: skip
    module.main(module.Args(policy_config=str(policy)))
    limits = yaml.safe_load((tmp_path / "policy-pc.yaml").read_text())["soft_joint_pos_limits"]
    for n in names:
        assert limits[n] == pytest.approx([G1_CFG.joint_pos_lower_limit[n] - PROTEUS_TARGET_MARGIN_RAD,
                                           G1_CFG.joint_pos_upper_limit[n] + PROTEUS_TARGET_MARGIN_RAD])


def test_select_pulse_during_mode_release_prevents_policy_output():
    g = armed()
    g.active = False
    g.arm_pending = True
    g.observe_state(state(3), 1 << 3, 1_030_000_000)
    g.observe_state(state(4), 0, 1_040_000_000)
    g.takeover_finished(True, 1_040_000_000)
    assert g.active and g.latched
    assert (g.output(1_040_000_000).kp == 0).all()


def test_metadata_roundtrip_and_legacy_messages():
    cmd = command(1_000_000_000, 4, "policy", 7)
    decoded = LowCmdMessage.from_bytes(cmd.to_bytes())
    assert decoded.control_mode == "policy" and decoded.control_generation == 7
    cmd.control_mode = None
    assert LowCmdMessage.from_bytes(cmd.to_bytes()).source_time_ns == cmd.source_time_ns
    cmd.source_time_ns = None
    assert LowCmdMessage.from_bytes(cmd.to_bytes()).control_mode is None
    s = state()
    assert LowStateMessage.from_bytes(s.to_bytes()).stop_epoch is None
    s.stop_epoch = 3
    assert LowStateMessage.from_bytes(s.to_bytes()).stop_epoch == 3


def test_zero_following_measured_can_acknowledge_stop_at_a_mechanical_limit():
    g = armed()
    s = state(3)
    s.joint_positions[4] = G1_CFG.joint_pos_lower_limit[G1_CFG.joint_names[4]] - .03
    g.observe_state(s, 1 << 3, 1_030_000_000)
    cmd = command(1_040_000_000, 3, "zero", 2)
    cmd.q_target[:] = s.joint_positions
    assert g.accept_command(cmd, 1_040_000_000)
    assert g.latched and not g.fault
    # Shaping may start at the mechanical limit (hold there or move inward) ...
    cmd = command(1_050_000_000, 4, "init", 3)
    cmd.q_target[4] = s.joint_positions[4]
    assert g.accept_command(cmd, 1_050_000_000)
    # ... but never command further out than the joint already is.
    cmd = command(1_060_000_000, 5, "init", 3)
    cmd.q_target[4] = s.joint_positions[4] - .05
    assert not g.accept_command(cmd, 1_060_000_000)
    assert g.fault


def test_shaping_starts_from_any_pose_past_the_limits():
    # An unpowered robot can rest beyond the model limits (ET1 waist yaw / wrist); A must still
    # bring it back. Each joint may move inward or hold, and the policy stays inside the limits.
    g = BridgeSafety(G1_CFG)
    s = state()
    j = G1_CFG.joint_names.index("waist_yaw_joint")
    past = G1_CFG.joint_pos_upper_limit["waist_yaw_joint"] + 0.5
    s.joint_positions[j] = past
    g.observe_state(s, 0, 1_000_000_000)
    hold = command(1_000_000_000, 1, "init", 1)
    hold.q_target[j] = past
    assert g.accept_command(hold, 1_000_000_000)
    inward = command(1_010_000_000, 2, "init", 1)
    inward.q_target[j] = past - 0.3
    assert g.accept_command(inward, 1_010_000_000) and not g.latched
    policy = command(1_020_000_000, 3, "policy", 2)
    policy.q_target[j] = past
    assert not g.accept_command(policy, 1_020_000_000) and g.fault


def test_actual_zmq_bridge_arms_on_a_and_detects_tracker_exit(monkeypatch):
    import socket
    import struct
    import time
    from dataclasses import replace
    from types import SimpleNamespace
    from scripts.g1.real_bridge import RealBridge
    from sim2real.rl_policy.controllers.bridge_joystick import BridgeJoystickController
    from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO

    def port():
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    cfg = replace(G1_CFG, low_state_port=port(), low_cmd_port=port(),
                  low_state_bind_addr="127.0.0.1", low_cmd_bind_addr="127.0.0.1")
    writes, releases = [], []

    def init_channels(self):
        self.low_cmd = SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(29)])
        self.crc = SimpleNamespace(Crc=lambda cmd: 0)
        self.low_cmd_unitree_pub = SimpleNamespace(Write=lambda cmd: (writes.append([m.kp for m in cmd.motor_cmd]) or True))

    def release(self):
        releases.append(True)
        with self.lock:
            self._takeover_result = True

    monkeypatch.setattr(RealBridge, "_init_channels", init_channels)
    monkeypatch.setattr(RealBridge, "_release_mode", release)
    gamepad_port = port()
    bridge = RealBridge(cfg, gamepad_port=gamepad_port)
    io = ZMQRobotIO(cfg)
    handset = BridgeJoystickController({}, port=gamepad_port)
    tick = 0

    def pump(keys=0, send=True, mode="zero", generation=0, duration=.04):
        nonlocal tick
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            tick += 1
            payload = bytearray(40)
            struct.pack_into("<H", payload, 2, keys)
            msg = SimpleNamespace(tick=tick, mode_machine=5, wireless_remote=payload,
                                  imu_state=SimpleNamespace(quaternion=[1, 0, 0, 0], gyroscope=[0, 0, 0]),
                                  motor_state=[SimpleNamespace(q=q, dq=0, tau_est=0) for q in G1_CFG.default_qpos[7:]])
            bridge._on_low_state(msg)
            if send:
                cmd = command(time.monotonic_ns())
                io.set_control_mode(mode, generation)
                io.write_command(cmd.q_target, cmd.dq_target, cmd.tau_ff, cmd.kp, cmd.kd)
            bridge._step()
            time.sleep(.005)

    try:
        pump(duration=.4)
        assert bridge.guard.command is not None
        assert not writes and not releases
        pump(1 << 8, mode="init", generation=1, duration=.08)
        assert releases == [True] and writes and bridge.guard.active
        # The Bridge re-publishes the handset for a Tracker without its own DDS subscription.
        assert handset.state is not None and handset.state.A
        pump(mode="policy", generation=2)
        pump(send=False, duration=.15)  # Inference process exits; DDS remains fresh.
        assert bridge.guard.latched and bridge.guard.fault
        assert writes[-1] == [0] * 29
        # Stop epoch travels through the actual LowState ZMQ decoder.
        assert io.read_state().stop_epoch == bridge.guard.stop_epoch
        pump(mode="policy", generation=3)
        assert bridge.guard.latched
    finally:
        handset.close()
        io.close()
        bridge.low_state_zmq_pub.close(linger=0)
        bridge.gamepad_zmq_pub.close(linger=0)
        bridge.low_cmd_zmq_sub.close(linger=0)


def test_g1_remote_payload_select_wins_over_a_and_start():
    import struct
    from types import SimpleNamespace
    from scripts.g1.validate_proteus_deployment import SimJoystick
    controller = SimJoystick()
    payload = bytearray(40)
    struct.pack_into("<H", payload, 2, (1 << 3) | (1 << 2) | (1 << 8))
    struct.pack_into("<3f", payload, 4, .1, .2, .3)
    struct.pack_into("<f", payload, 20, .4)
    controller._pending_extra_keys.append("Start")
    controller._on_low_state(SimpleNamespace(wireless_remote=payload))
    assert controller.get_control_mode() == "zero"
    assert controller.get_extra_keys() == []
    np.testing.assert_allclose(controller.state.left_stick, [.1, .4])
    struct.pack_into("<H", payload, 2, 0)
    controller._on_low_state(SimpleNamespace(wireless_remote=payload))
    assert controller.get_control_mode() is None
    struct.pack_into("<H", payload, 2, 1 << 2)
    controller._on_low_state(SimpleNamespace(wireless_remote=payload))
    assert controller.get_control_mode() is None
    assert controller.get_extra_keys() == ["Start"]


def test_native_channel_close_is_bounded():
    import threading
    import time
    from types import SimpleNamespace
    from sim2real.utils.sdk_shutdown import close_sdk_channel
    release = threading.Event()
    before = time.monotonic()
    try:
        assert not close_sdk_channel(SimpleNamespace(Close=lambda: release.wait(1)), timeout_s=.01)
        assert time.monotonic() - before < .5
    finally:
        release.set()


def test_legacy_go_handset_retains_r1_r2_b_controls():
    from scripts.g1.validate_proteus_deployment import SimJoystick
    from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickState
    controller = SimJoystick()
    controller._g1_controls = False
    controller._wc_msg = UnitreeJoystickState(R1=True)
    assert controller.get_control_mode() == "policy"
    controller._wc_msg = UnitreeJoystickState(B=True)
    assert controller.get_control_mode() is None
    assert controller.get_extra_keys() == ["B"]
    controller._wc_msg = UnitreeJoystickState(R2=True)
    assert controller.get_control_mode() == "zero"


def test_reference_entry_is_smooth_and_blocks_start_until_hold(monkeypatch):
    from types import SimpleNamespace
    from sim2real.rl_policy.tracking import Tracking
    from sim2real.rl_policy.base_policy import BasePolicy
    tracker = Tracking.__new__(Tracking)
    tracker.policy_config = {'reference_entry_s': 2.0}
    tracker.motion_backend = 'npz'
    tracker.rl_dt = .02
    tracker.state_dict = {'control_mode': 'init', 'paused': True}
    start, target = np.zeros(29), np.ones(29)
    tracker.state_processor = SimpleNamespace(joint_pos=start)
    tracker.target_postprocessor = None
    tracker.motion_first_frame_joint_pos = lambda: target
    tracker._reference_entry_target = None
    tracker._reference_entry_count = 0
    commits = []

    def commit(self, *, source):
        commits.append(source)
        self.state_dict['control_mode'] = 'policy'
        self.state_dict['paused'] = True

    monkeypatch.setattr(BasePolicy, 'set_policy_mode', commit)
    tracker.set_policy_mode(source='second A')
    tracker.toggle_paused(source='early Start')
    assert tracker.state_dict == {'control_mode': 'init', 'paused': True}
    tracker.set_policy_mode(source='duplicate A')
    assert tracker._reference_entry_count == 0
    rows = [tracker.get_init_target() for _ in range(101)]
    np.testing.assert_array_equal(rows[0], start)
    np.testing.assert_array_equal(rows[-1], target)
    assert np.max(np.abs(np.diff(rows, axis=0))) < .02
    assert commits == ['reference pose transition complete']
    assert tracker.state_dict == {'control_mode': 'policy', 'paused': True}


def test_stop_cancels_reference_entry(monkeypatch):
    from sim2real.rl_policy.tracking import Tracking
    from sim2real.rl_policy.base_policy import BasePolicy
    tracker = Tracking.__new__(Tracking)
    tracker._reference_entry_target = np.ones(29)
    tracker.motion_backend = 'npz'
    tracker.state_dict = {'control_mode': 'init', 'paused': True}
    tracker.set_zero_mode(source='Select')
    assert tracker._reference_entry_target is None
    assert tracker.state_dict['control_mode'] == 'zero'
    assert tracker.state_dict['paused']
