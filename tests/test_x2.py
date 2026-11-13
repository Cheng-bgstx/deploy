from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sim2real.config.robots import X2_CFG, get_robot_cfg
from sim2real.config.robots.x2 import X2_AIMDK_GROUPS, X2_VENDOR_LIMITS
from sim2real.rl_policy.robot_io.x2 import X2StateCommand


def group_packet(group, *, reversed_order=False):
    names = list(X2_AIMDK_GROUPS[group])
    if reversed_order:
        names.reverse()
    return NS(joints=[NS(name=n, position=.01 * X2_CFG.joint_names.index(n), velocity=.02,
                         effort=.03, error_code=0) for n in names])


def imu_packet(q=(1., 0., 0., 0.), gyro=(.1, .2, .3)):
    return NS(orientation=NS(w=q[0], x=q[1], y=q[2], z=q[3]),
              angular_velocity=NS(x=gyro[0], y=gyro[1], z=gyro[2]))


def feed(codec):
    for group in X2_AIMDK_GROUPS:
        codec.receive_group(group, group_packet(group, reversed_order=True))
    codec.receive_imu(imu_packet())


def test_model_registration_and_aimdk_waist_permutation():
    assert get_robot_cfg('lingxi_x2') is X2_CFG
    assert get_robot_cfg('agibot_x2') is X2_CFG
    assert len(X2_CFG.joint_names) == 31 and len(X2_CFG.default_qpos) == 38
    assert sum(map(len, X2_AIMDK_GROUPS.values())) == 31
    assert X2_AIMDK_GROUPS['waist'] == ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint')
    assert len(set(n for names in X2_AIMDK_GROUPS.values() for n in names)) == 31


def test_feedback_reorders_groups_and_requires_all_fresh_packets():
    codec = X2StateCommand(X2_CFG)
    assert codec.read_state() is None
    feed(codec)
    state = codec.read_state()
    np.testing.assert_allclose(state.qpos[7:], np.arange(31) * .01)
    np.testing.assert_array_equal(state.qpos[3:7], [1., 0., 0., 0.])
    np.testing.assert_allclose(state.qvel[3:6], [.1, .2, .3])
    t, v = codec.groups['leg']
    codec.groups['leg'] = (t - 1., v)
    assert codec.read_state() is None


def test_command_array_mapping_clipping_head_and_watchdog():
    codec = X2StateCommand(X2_CFG)
    feed(codec)
    codec.mode = 'policy'
    q = np.linspace(0, .15, 31)
    q[X2_CFG.joint_names.index('waist_yaw_joint')] = 100
    kp, kd = np.full(31, 20.), np.full(31, 2.)
    codec.write_command(q, np.zeros(31), np.zeros(31), kp, kd)
    commands = codec.command_groups()
    for g, names in X2_AIMDK_GROUPS.items():
        for i, n in enumerate(names):
            idx = X2_CFG.joint_names.index(n)
            if n == 'head_pitch_joint':
                assert all(v[i] == 0 for v in commands[g])
            else:
                assert commands[g][3][i] == 20
                assert commands[g][0][i] == pytest.approx(np.clip(q[idx], codec.lower[idx], codec.upper[idx]))
    assert commands['waist'][0][0] == pytest.approx(X2_VENDOR_LIMITS['waist_yaw_joint'][1])
    codec.command_time -= 1
    expired = codec.command_groups()
    for values in expired.values():
        np.testing.assert_array_equal(values[1], 0)
        np.testing.assert_array_equal(values[2], 0)
        np.testing.assert_array_equal(values[3], 0)
    assert expired['leg'][4][0] == 2


def test_edu_without_head_uses_default_head_pose_and_never_commands_head():
    codec = X2StateCommand(X2_CFG, head_present=False)
    for group in ('leg', 'waist', 'arm'):
        codec.receive_group(group, group_packet(group, reversed_order=True))
    codec.receive_imu(imu_packet())
    state = codec.read_state()
    assert state is not None
    head = [X2_CFG.joint_names.index(n) for n in X2_AIMDK_GROUPS['head']]
    body = [i for i in range(31) if i not in head]
    np.testing.assert_allclose(state.qpos[7:][body], np.arange(31)[body] * .01)
    np.testing.assert_allclose(state.qpos[7:][head], np.asarray(X2_CFG.default_qpos[7:])[head])
    np.testing.assert_array_equal(state.qvel[6:][head], 0)
    codec.mode = 'policy'
    codec.write_command(np.zeros(31), np.zeros(31), np.zeros(31), np.full(31, 20.), np.full(31, 2.))
    assert set(codec.command_groups()) == {'leg', 'waist', 'arm'}


def test_imu_mount_rotation_applies_to_orientation_and_gyro():
    q = [np.sqrt(.5), 0., 0., np.sqrt(.5)]  # sensor axes turned +90deg about pelvis z
    codec = X2StateCommand(X2_CFG, imu_to_pelvis_wxyz=q)
    feed(codec)
    codec.receive_imu(imu_packet(q=q, gyro=[1., 0., 0.]))
    state = codec.read_state()
    np.testing.assert_allclose(state.qpos[3:7], [1, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(state.qvel[3:6], [0, 1, 0], atol=1e-6)


def test_bad_feedback_latches_and_invalid_commands_are_rejected():
    codec = X2StateCommand(X2_CFG)
    with pytest.raises(ValueError, match='finite'):
        codec.write_command(np.full(31, np.nan), *([np.zeros(31)] * 4))
    msg = group_packet('leg')
    msg.joints[0].error_code = 5
    codec.receive_group('leg', msg)
    with pytest.raises(RuntimeError, match='feedback'):
        codec.read_state()
    codec = X2StateCommand(X2_CFG)
    codec.receive_imu(imu_packet(q=[0., 0., 0., 0.]))
    with pytest.raises(RuntimeError, match='feedback'):
        codec.read_state()


@pytest.mark.skipif(not X2_CFG.mjcf_path.is_file(), reason='X2 runtime assets not staged')
def test_x2_mjcf_has_training_axes_limits_and_named_actuators():
    import json
    import mujoco
    from sim2real.sim_env.utils.mjcf import load_sim_model
    model = load_sim_model(X2_CFG)
    assert model.nq == 38 and model.nu == 31
    assert {model.actuator(i).name for i in range(model.nu)} == set(X2_CFG.joint_names)
    for n in X2_CFG.joint_names:
        np.testing.assert_allclose(model.joint(n).range, [X2_CFG.joint_pos_lower_limit[n], X2_CFG.joint_pos_upper_limit[n]])
    assert model.body_mass.sum() == pytest.approx(41.966473)
    report = json.loads((X2_CFG.mjcf_path.parents[1] / 'validation.json').read_text())
    assert report['max_joint_axis_error'] < 1e-6
    assert report['mass_difference_kg'] != 0.  # The vendor delta must remain explicit.


def test_reference_converter_reads_explicit_joint_order_and_xyzw(tmp_path):
    from scripts.x2.convert_reference import source_joint_names
    path = tmp_path / 'source.py'
    path.write_text("MUJOCO_JOINT_NAMES = ['head_yaw_joint','head_pitch_joint']\nraise RuntimeError('must not import')\n")
    assert source_joint_names(path) == ['head_yaw_joint', 'head_pitch_joint']


def test_ros_command_fields_follow_documented_group_arrays():
    from sim2real.rl_policy.robot_io.x2 import X2RobotIO
    io = X2RobotIO.__new__(X2RobotIO)
    io.codec = X2StateCommand(X2_CFG, head_pitch_enabled=True)
    feed(io.codec)
    io.codec.mode = 'policy'
    io.codec.write_command(np.linspace(-.02, .02, 31), np.zeros(31), np.zeros(31),
                           np.full(31, 12), np.full(31, .5))
    io._joint_type = NS
    io._array_type = lambda: NS(header=NS(stamp=None), joints=[])
    io._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: 'stamp')))
    sent = {}
    io._publishers = {g: NS(publish=lambda msg, group=g: sent.update({group: msg})) for g in X2_AIMDK_GROUPS}
    io._publish()
    assert io.codec.error is None
    assert set(sent) == set(X2_AIMDK_GROUPS)
    for group, names in X2_AIMDK_GROUPS.items():
        msg = sent[group]
        assert msg.header.stamp == 'stamp'
        assert [j.name for j in msg.joints] == list(names)
        assert all(j.stiffness == 12. and j.damping == .5 for j in msg.joints)
    io._publish(force_damping=True)
    assert all(j.stiffness == 0 for msg in sent.values() for j in msg.joints)


@pytest.mark.skipif(not X2_CFG.mjcf_path.is_file(), reason='X2 runtime assets not staged')
def test_reference_conversion_keeps_named_columns_and_absolute_root(tmp_path):
    import joblib
    import mujoco
    from scripts.x2.convert_reference import convert
    root = tmp_path / 'source'
    (root / 'scripts').mkdir(parents=True)
    (root / 'motions').mkdir()
    names = list(reversed(X2_CFG.joint_names))
    (root / 'scripts/eval_x2_mujoco.py').write_text('MUJOCO_JOINT_NAMES = ' + repr(names))
    dof = np.tile(np.arange(31) * .001, (3, 1))
    source = {'clip': {'dof': dof, 'root_trans_offset': np.array([[4, 5, .68], [4.1, 5.2, .69], [4.2, 5.3, .7]]),
                       'root_rot': np.tile([0., 0., np.sqrt(.5), np.sqrt(.5)], (3, 1)), 'fps': 50}}
    joblib.dump(source, root / 'motions/x2_relaxed_walk.pkl')
    path = convert(root, 'relaxed_walk', tmp_path / 'dataset')
    qpos = np.load(path)['qpos']
    np.testing.assert_allclose(qpos[:, :3], [[0, 0, .68], [.1, .2, .69], [.2, .3, .7]], atol=1e-7)
    np.testing.assert_allclose(qpos[0, 3:7], [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-7)
    model = mujoco.MjModel.from_xml_path(str(X2_CFG.mjcf_path))
    for i, name in enumerate(names):
        np.testing.assert_allclose(qpos[:, model.joint(name).qposadr[0]], dof[:, i], atol=1e-7)
