"""Exercise the commissioning controls through the real named ROS serializer."""
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from scripts.q1.commission_joints import Probe
from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.ros2_named import NamedJointCodec, NamedRosRobotIO
from scripts.q1.prepare_deployment import Args as PrepareArgs, prepare


def make_probe(*, read_only=False):
    cfg = get_robot_cfg('q1')
    io = NamedRosRobotIO.__new__(NamedRosRobotIO)
    io.codec = NamedJointCodec(cfg, {'body': cfg.joint_names}, damping=5., allow_init_from_measured=True)
    io._armed, io._preserve_feedback_order, io._sequence = False, True, 0
    io._array_type = lambda: NS(header=NS(stamp=None), joints=[])
    io._joint_type = lambda: NS()
    io._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: None)))
    sent = []
    io._publishers = {'body': NS(publish=sent.append)}
    # A slot receiver must get this layout even though the internal order differs.
    wire = tuple(reversed(cfg.joint_names))
    q = np.array([(cfg.joint_pos_lower_limit[n] + cfg.joint_pos_upper_limit[n]) / 2 for n in cfg.joint_names])
    io.codec.receive_group('body', NS(joints=[NS(name=n, position=q[cfg.joint_names.index(n)], velocity=0., effort=0.) for n in wire]))
    io.codec.receive_imu(NS(orientation=NS(w=1., x=0., y=0., z=0.), angular_velocity=NS(x=0., y=0., z=0.)))
    probe = Probe(io, cfg, 'left_elbow_joint', read_only=read_only)
    probe.started_at = time.monotonic() - 6.
    return probe, io, sent, q, wire


def test_disarmed_and_readonly_cannot_publish_even_on_keys():
    for read_only, keys in ((False, ('', 'i', ']', 'j')), (True, ('h', 'j'))):
        probe, io, sent, _, _ = make_probe(read_only=read_only)
        for key in keys:
            probe.step(time.monotonic(), key, 1)
            io._publish()
        probe.stop(time.monotonic(), 'exit')
        io._publish(force_damping=True)
        assert not sent
        assert not io._armed


def test_first_hold_is_measured_pose_and_only_selected_wire_slot_moves():
    probe, io, sent, baseline, wire = make_probe()
    now = time.monotonic()
    probe.step(now, 'h', 1)
    io._publish()
    np.testing.assert_allclose([j.position for j in sent[-1].joints], baseline[::-1])
    assert all(j.stiffness == 0. for j in sent[-1].joints)
    probe.step(now + 3., 'j', 1)
    for elapsed, offset in ((2., .03), (4., 0.), (6., -.03), (8., 0.)):
        probe.step(now + 3. + elapsed, '', 1)
        io._publish()
        assert [j.name for j in sent[-1].joints] == list(wire)
        expected = baseline.copy()
        expected[probe.index] += offset
        np.testing.assert_allclose([j.position for j in sent[-1].joints], expected[::-1], atol=1e-7)
        assert all(j.stiffness <= 5. and j.damping == 2. for j in sent[-1].joints)
    probe.step(now + 12., '', 1)
    io._publish()
    assert probe.done
    assert all(j.stiffness == 0. and j.damping == 5. for j in sent[-1].joints)


@pytest.mark.parametrize('fault', ['stale', 'moving', 'drift', 'publisher', 'layout'])
def test_fault_damps_through_production_transport(fault):
    probe, io, sent, _, _ = make_probe()
    now = time.monotonic()
    probe.step(now, 'h', 1)
    count = 1
    if fault == 'stale':
        io.codec.imu = (now - 1., *io.codec.imu[1:])
    elif fault == 'moving':
        io.codec.state['body'][1][1][0] = .6
    elif fault == 'drift':
        io.codec.state['body'][1][0][0] += .11
    elif fault == 'publisher':
        count = 2
    else:
        io.codec.feedback_order['body'] = tuple(reversed(io.codec.feedback_order['body']))
    with pytest.raises(RuntimeError):
        probe.step(now + .02, '', count)
    io._publish()
    assert probe.done and io.codec.mode == 'zero'
    assert all(j.stiffness == 0. and j.damping == 5. for j in sent[-1].joints)


def test_busy_publisher_and_joint_motion_refuse_arming():
    probe, io, sent, _, _ = make_probe()
    probe.step(time.monotonic(), 'h', 2)
    io._publish()
    assert not sent and not io._armed
    io.codec.state['body'][1][1][0] = .2
    probe.step(time.monotonic(), 'h', 1)
    io._publish()
    assert not sent and not io._armed


def test_joint_limit_refuses_probe_without_changing_measured_hold():
    probe, io, sent, baseline, _ = make_probe()
    io.codec.state['body'][1][0][probe.index] = probe.cfg.joint_pos_lower_limit[probe.joint] + .01
    now = time.monotonic()
    probe.step(now, 'h', 1)
    probe.step(now + 3., 'j', 1)
    assert probe.mode == 'hold'
    assert probe.pulse_at is None
    io._publish()
    assert all(j.stiffness <= 5. for j in sent[-1].joints)


@pytest.mark.parametrize('key', ['o', ' ', 'q', '\x1b'])
def test_stop_keys_immediately_request_damping(key):
    probe, io, sent, _, _ = make_probe()
    now = time.monotonic()
    probe.step(now, 'h', 1)
    probe.step(now + .02, key, 1)
    io._publish()
    assert probe.done
    assert all(j.stiffness == 0. and j.damping == 5. for j in sent[-1].joints)


def test_manual_overlay_removes_inherited_gates_and_feedforward_without_changing_policy(tmp_path):
    cfg = get_robot_cfg('q1')
    original = {
        'policy_joint_names': list(cfg.joint_names),
        'proteus': {'robot': 'primebot_q1', 'reference_forcing': True},
        'robot_io_options': {'publish_hz': 1000., 'timeout_s': .25},
        'joint_kp': {'left_elbow_joint': 4.2032},
        'joint_kd': {'left_elbow_joint': .204},
        'model_path': 'policy.onnx',
        'policy_entry_max_joint_error_rad': .1,
        'policy_entry_max_joint_velocity_rad_s': .5,
        'shaping_gravity_compensation': {'max_effort_fraction': .1, 'ramp_s': 2.},
    }
    source = tmp_path / 'policy.yaml'
    source.write_text(yaml.safe_dump(original))
    output = prepare(PrepareArgs('/aima/hal/imu/state', (1., 0., 0., 0.),
        policy_config=source, manual_policy_entry=True))
    result = yaml.safe_load(output.read_text())
    for key in ('policy_entry_max_joint_error_rad', 'policy_entry_max_joint_velocity_rad_s', 'shaping_gravity_compensation'):
        assert key not in result
    for key in ('model_path', 'proteus', 'policy_joint_names', 'joint_kp', 'joint_kd', 'robot_io_options'):
        if key != 'robot_io_options':
            assert result[key] == original[key]
    assert result['robot_io_options']['timeout_s'] == .25
    assert result['reference_entry_s'] == 0.
    assert result['max_target_velocity_rad_s'] == 15.
    assert len(result['soft_joint_pos_limits']) == 22
    assert yaml.safe_load(source.read_text()) == original
