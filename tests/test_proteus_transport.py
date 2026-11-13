"""SDK-free checks of wire layouts, IMU frames, command expiry and vendor gates."""
from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.ros2_named import NamedJointCodec

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = {'q1': 'primebot_q1', 't800': 'engineai_t800', 'pm01': 'engineai_pm01', 'a3': 'agibot_a3'}
ROBOTS = [n for n, c in CONTRACTS.items() if (ROOT / f'sim2real/config/robots/specs/{c}.yaml').is_file()]


def codec_for(name):
    cfg = get_robot_cfg(name)
    if name == 'a3':
        from sim2real.rl_policy.robot_io.a3 import groups
        return NamedJointCodec(cfg, groups(cfg), fixed_joints={'head_pitch_joint': (0, 0, 0, 3, .3)},
            physical_limits={'left_knee_joint': (-.122173, 2.49582)})
    if name in {'t800', 'pm01'}:
        if name == 'pm01':
            from sim2real.rl_policy.robot_io.pm01 import PM01Codec as Codec, PM01_SDK_JOINT_NAMES as sdk_names
        else:
            from sim2real.rl_policy.robot_io.t800 import T800Codec as Codec, T800_SDK_JOINT_NAMES as sdk_names
        # Deliberately reverse runtime order: SDK mapping must remain by name.
        cfg = replace(cfg, joint_names=tuple(reversed(cfg.joint_names)))
        return Codec(cfg, {'body': sdk_names})
    return NamedJointCodec(cfg, {'body': cfg.joint_names}, damping=5.)


def stamp(ns=1):
    return NS(stamp=NS(sec=0, nanosec=ns))


def feed(codec, ns=1):
    for group, names in codec.groups.items():
        positions = [float(i) * .001 for i in range(len(names))]
        if codec.cfg.name in {'t800', 'pm01'}:
            msg = NS(header=stamp(ns), position=positions, velocity=[.02] * len(names), torque=[.03] * len(names))
        else:
            msg = NS(header=stamp(ns), joints=[NS(name=n, position=positions[i], velocity=.02, effort=.03)
                                              for i, n in reversed(list(enumerate(names)))])
        codec.receive_group(group, msg)
    imu = NS(header=stamp(ns), orientation=NS(w=1., x=0., y=0., z=0.),
             angular_velocity=NS(x=.1, y=.2, z=.3))
    if codec.cfg.name in {'t800', 'pm01'}:
        imu.quaternion = imu.orientation
        codec.receive_motion_state(NS(current_motion_task='joint_bridge'))
    codec.receive_imu(imu)


@pytest.mark.parametrize('name', ROBOTS)
def test_feedback_mapping_frames_and_duplicate_stamp_expiry(name):
    codec = codec_for(name)
    assert codec.read_state() is None
    feed(codec)
    state = codec.read_state()
    assert state is not None
    for group, names in codec.groups.items():
        for i, n in enumerate(names):
            if n in codec.cfg.joint_names:
                assert state.qpos[7 + codec.cfg.joint_names.index(n)] == pytest.approx(i * .001)
    np.testing.assert_array_equal(state.qpos[3:7], [1, 0, 0, 0])
    np.testing.assert_allclose(state.qvel[3:6], [.1, .2, .3])
    first_time = codec.imu[0]
    feed(codec)  # Replayed ROS stamps cannot refresh the watchdog.
    assert codec.imu[0] == first_time
    codec.imu = (first_time - 1, *codec.imu[1:])
    assert codec.read_state() is None


@pytest.mark.parametrize('name', ROBOTS)
def test_commands_limits_step_bound_watchdog_and_fixed_motor(name):
    codec = codec_for(name)
    feed(codec)
    codec.mode = 'policy'
    count = len(codec.cfg.joint_names)
    codec.write_command(np.full(count, 100.), np.zeros(count), np.zeros(count), np.full(count, 20.), np.ones(count))
    values = codec.command_groups()
    for g, v in values.items():
        assert np.max(np.abs(v[0] - codec.state[g][1][0])) <= .20001
        assert np.isfinite(v).all()
    for _ in range(80):
        values = codec.command_groups()
    for group, names in codec.groups.items():
        for i, n in enumerate(names):
            if n not in codec.fixed:
                assert values[group][0, i] <= codec.limits[n][1] + 1e-5
                assert values[group][0, i] >= codec.limits[n][0] - 1e-5
    if name == 'a3':
        i = codec.groups['neck'].index('head_pitch_joint')
        assert values['neck'][0, i] == 0
        assert values['neck'][3, i] == 3
        assert codec.limits['left_knee_joint'][1] == 2.49582
    codec.command_time -= 1.
    for v in codec.command_groups().values():
        np.testing.assert_array_equal(v[1:4], 0)
        np.testing.assert_array_equal(v[4], codec.damping)
    with pytest.raises(ValueError, match='finite'):
        codec.write_command(np.full(count, np.nan), *([np.zeros(count)] * 4))


@pytest.mark.parametrize('name', ROBOTS)
def test_bad_feedback_latches_damping(name):
    codec = codec_for(name)
    feed(codec)
    group = next(iter(codec.groups))
    codec.receive_group(group, NS(header=stamp(2), joints=[]))
    with pytest.raises(RuntimeError):
        codec.read_state()
    assert all(np.all(v[1:4] == 0) for v in codec.command_groups().values())


@pytest.mark.parametrize('name', ROBOTS)
def test_factory_dispatches_vendor_options(name, monkeypatch):
    from sim2real.rl_policy.robot_io.factory import create_robot_io
    module = importlib.import_module(f'sim2real.rl_policy.robot_io.{name}')
    attr = {'q1': 'Q1RobotIO', 't800': 'T800RobotIO', 'pm01': 'PM01RobotIO', 'a3': 'A3RobotIO'}[name]
    seen = []
    monkeypatch.setattr(module, attr, lambda cfg, **kw: seen.append((cfg, kw)) or 'created')
    cfg = get_robot_cfg(name)
    assert create_robot_io(mode='inline', robot_name=CONTRACTS[name], robot_cfg=cfg,
                           interface='unused', options={'timeout_s': .1}) == 'created'
    assert seen == [(cfg, {'timeout_s': .1})]


@pytest.mark.parametrize('name', ROBOTS)
def test_prepared_model_axes_contract_policy_and_portability(name):
    import mujoco
    from scripts.proteus.prepare_assets import verify
    from sim2real.sim_env.utils.mjcf import load_sim_model
    cfg = get_robot_cfg(name)
    if not cfg.mjcf_path.is_file():
        pytest.skip('Stage the vendor model')
    spec = yaml.safe_load((ROOT / f'sim2real/config/robots/specs/{CONTRACTS[name]}.yaml').read_text())
    model = load_sim_model(cfg)
    assert model.nq == cfg.qpos_size and model.nu == len(cfg.joint_names)
    axis_atol = 1e-4 if name == 'pm01' else 1e-5
    assert verify(model, cfg, spec, axis_atol=axis_atol) < axis_atol
    assert {model.actuator(i).name for i in range(model.nu)} == set(cfg.joint_names)
    text = cfg.mjcf_path.read_text()
    assert '/home/' not in text
    release = 'wsq61' if name == 'pm01' else 'proteus_wamseq_62'
    policy_path = ROOT / f'checkpoints/proteus/{release}/{CONTRACTS[name]}/policy.yaml'
    if policy_path.is_file():
        policy = yaml.safe_load(policy_path.read_text())
        assert policy['policy_joint_names'] == list(cfg.joint_names)
        assert policy['proteus']['seq_window'] == 10


def test_t800_requires_fresh_joint_bridge_mode():
    if 't800' not in ROBOTS:
        pytest.skip('T800 branch only')
    codec = codec_for('t800')
    feed(codec)
    codec.receive_motion_state(NS(current_motion_task='pd_stand'))
    assert codec.read_state() is None
    codec.mode = 'policy'
    for v in codec.command_groups().values():
        np.testing.assert_array_equal(v[1:4], 0)
    codec.receive_motion_state(NS(current_motion_task='joint_bridge'))
    assert codec.read_state() is not None
    codec.motion_mode_time -= 2
    assert codec.read_state() is None


def test_q1_missing_imu_configuration_fails_before_ros_import():
    if 'q1' not in ROBOTS:
        pytest.skip('Q1 branch only')
    from sim2real.rl_policy.robot_io.q1 import Q1RobotIO
    with pytest.raises(ValueError, match='verified root'):
        Q1RobotIO(get_robot_cfg('q1'))


def test_named_ros_accepts_action_manager_keyword_commands():
    from sim2real.rl_policy.robot_io.ros2_named import NamedRosRobotIO
    from sim2real.rl_policy.utils.command_sender import ActionManager
    codec = codec_for('q1')
    feed(codec)
    io = NamedRosRobotIO.__new__(NamedRosRobotIO)
    io.codec = codec
    manager = ActionManager(codec.cfg, {
        'joint_kp': {'.*': 10.}, 'joint_kd': {'.*': 1.},
        'default_joint_pos': {'.*': 0.},
    }, io)
    n = len(codec.cfg.joint_names)
    manager.send_command(np.full(n, .1), np.zeros(n), np.zeros(n))
    np.testing.assert_allclose(codec.command[0], .1)
    np.testing.assert_allclose(codec.command[3], 10.)


def test_disarmed_ros_transport_does_not_publish_until_shaping():
    from sim2real.rl_policy.robot_io.ros2_named import NamedRosRobotIO
    io = NamedRosRobotIO.__new__(NamedRosRobotIO)
    io.codec = codec_for('q1')
    feed(io.codec)
    io._armed = False
    # A disarmed publish must not even construct a message, including on close.
    io._publish()
    io._publish(force_damping=True)
    assert io.codec.error is None
    io.set_control_mode('zero', 1)
    assert not io._armed
    io.set_control_mode('init', 2)
    assert io._armed


def test_q1_shaping_returns_gradually_from_measured_position_outside_model_range():
    codec = codec_for('q1')
    codec.allow_init_from_measured = True
    feed(codec)
    names = codec.groups['body']
    index = names.index('waist_roll_joint')
    measured = codec.state['body'][1][0].copy()
    measured[index] = .26
    codec.state['body'][1][0][index] = .26
    n = len(names)
    codec.mode = 'init'
    def command(q):
        codec.write_command(q, np.zeros(n), np.zeros(n), np.ones(n), np.ones(n))
        return codec.command_groups()['body'][0, index]
    assert command(measured) == pytest.approx(.26)
    target = measured.copy()
    target[index] = .25
    assert command(target) == pytest.approx(.25)
    target[index] = .4
    assert command(target) == pytest.approx(.26)  # cannot go farther out
    codec.mode = 'policy'
    assert command(target) == pytest.approx(.22)  # policy retains model limits



def test_q1_unclipped_policy_positions_keep_offset_but_shaping_still_clips():
    cfg = get_robot_cfg('q1')
    codec = NamedJointCodec(cfg, {'body': cfg.joint_names}, damping=5., allow_init_from_measured=True,
                            policy_limit_mode='pass')
    feed(codec)
    names = codec.groups['body']
    n, index = len(names), names.index('left_ankle_pitch_joint')
    lo, _ = codec.limits['left_ankle_pitch_joint']
    target = codec.state['body'][1][0].copy()
    target[index] = lo - .15
    def settle():
        codec.write_command(target, np.zeros(n), np.full(n, 1e4), np.ones(n), np.ones(n))
        for _ in range(10):  # the 0.2 rad per-publish step bound still applies
            values = codec.command_groups()['body']
        return values
    codec.mode = 'init'
    assert settle()[0, index] == pytest.approx(lo)  # Shaping keeps the model range
    codec.mode = 'policy'
    values = settle()
    assert values[0, index] == pytest.approx(lo - .15)  # policy keeps the actor's PD offset
    assert values[2, index] == pytest.approx(cfg.joint_effort_limit['left_ankle_pitch_joint'])  # effort still bounded


def test_q1_feedforward_limit_mode_is_a_virtual_stop():
    cfg = get_robot_cfg('q1')
    codec = NamedJointCodec(cfg, {'body': cfg.joint_names}, damping=5., allow_init_from_measured=True,
                            policy_limit_mode='feedforward', limit_feedforward_band_rad=.1)
    feed(codec)
    names = codec.groups['body']
    n, index = len(names), names.index('left_ankle_pitch_joint')
    lo, _ = codec.limits['left_ankle_pitch_joint']
    target = codec.state['body'][1][0].copy()
    target[index] = lo - .15
    def settle(kp=8.4):
        codec.write_command(target, np.zeros(n), np.zeros(n), np.full(n, kp), np.ones(n))
        for _ in range(10):
            values = codec.command_groups()['body']
        return values[0, index], values[2, index]
    codec.mode = 'init'
    assert settle() == pytest.approx((lo, 0.))  # Shaping: clip, no feedforward
    codec.mode = 'policy'
    assert settle() == pytest.approx((lo, 8.4 * -.15))  # well inside the range: the unclipped PD torque
    codec.state['body'][1][0][index] = lo + .05
    assert settle() == pytest.approx((lo, .5 * 8.4 * -.15))  # half way through the band
    codec.state['body'][1][0][index] = lo - .02
    assert settle() == pytest.approx((lo, 0.))  # past the limit: plain clip
    codec.state['body'][1][0][index] = 0.
    limit = cfg.joint_effort_limit['left_ankle_pitch_joint']
    assert settle(kp=1e4) == pytest.approx((lo, -limit))  # effort bound still applies
    with pytest.raises(ValueError, match='policy_limit_mode'):
        NamedJointCodec(cfg, {'body': cfg.joint_names}, policy_limit_mode='unclipped')

def test_q1_full_command_preserves_feedback_slots_and_all_joint_fields(monkeypatch):
    """Replay the observed Q1 layout into the actual serializer and a slot-based HAL.

    The public example sends full commands in feedback order. Merely labelling a
    policy-ordered array by name does not protect a receiver that consumes slots.
    """
    import sys
    from types import ModuleType
    from sim2real.rl_policy.robot_io.ros2_named import NamedRosRobotIO
    from sim2real.rl_policy.robot_io.q1 import Q1RobotIO
    # Run the actual Q1 constructor, replacing only ROS transport creation.
    messages = ModuleType('aimdk_msgs.msg')
    messages.JointCommand = messages.JointCommandArray = messages.JointStateArray = object
    monkeypatch.setitem(sys.modules, 'aimdk_msgs.msg', messages)
    def offline_transport(self, cfg, *, groups, preserve_feedback_order=False, **options):
        self.codec = NamedJointCodec(cfg, groups, damping=options['damping'])
        self._preserve_feedback_order = preserve_feedback_order
    monkeypatch.setattr(NamedRosRobotIO, '__init__', offline_transport)
    cfg = get_robot_cfg('q1')
    wire = [
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint',
        'waist_yaw_joint', 'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint',
        'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'waist_roll_joint',
        'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint',
        'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    ]
    io = Q1RobotIO(cfg, imu_topic='/aima/hal/imu/state')
    codec = io.codec
    feed(codec)
    measured = {n: .5*(cfg.joint_pos_lower_limit[n]+cfg.joint_pos_upper_limit[n]) for n in wire}
    codec.receive_group('body', NS(header=stamp(2), joints=[
        NS(name=n, position=measured[n], velocity=0., effort=0.) for n in wire]))
    io._sequence = 0
    io._array_type = lambda: NS(header=NS(stamp=None), joints=[])
    io._joint_type = lambda: NS()
    io._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: 'stamp')))
    sent = []
    io._publishers = {'body': NS(publish=sent.append)}
    io.set_control_mode('init', 1)
    q = np.array([measured[n] for n in cfg.joint_names])
    dq, tau, kp, kd = [np.arange(22, dtype=float)*.001+base for base in (.01, .1, 10., 1.)]
    io.write_command(q_target=q, dq_target=dq, tau_ff=tau, kp=kp, kd=kd)
    io._publish()
    assert codec.error is None
    assert [j.name for j in sent[-1].joints] == wire
    for slot, name in enumerate(wire):
        command = sent[-1].joints[slot]
        index = cfg.joint_names.index(name)
        np.testing.assert_allclose(
            [command.position, command.velocity, command.effort, command.stiffness, command.damping],
            [v[index] for v in (q, dq, tau, kp, kd)], rtol=1e-6)
    io.set_control_mode('zero', 2)
    io._publish()
    assert [j.name for j in sent[-1].joints] == wire
    np.testing.assert_allclose([j.position for j in sent[-1].joints], [measured[n] for n in wire])
    assert all(j.stiffness == 0. and j.damping == 5. for j in sent[-1].joints)


@pytest.mark.parametrize('name', ROBOTS)
def test_imu_mount_transform(name):
    codec = codec_for(name)
    codec.rotation = np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)], np.float32)
    feed(codec)
    msg = NS(header=stamp(2), orientation=NS(w=np.sqrt(.5), x=0., y=0., z=np.sqrt(.5)),
             angular_velocity=NS(x=1., y=0., z=0.))
    if name in {'t800', 'pm01'}:
        msg.quaternion = msg.orientation
    codec.receive_imu(msg)
    np.testing.assert_allclose(codec.read_state().qpos[3:7], [1, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(codec.read_state().qvel[3:6], [0, 1, 0], atol=1e-6)


@pytest.mark.parametrize('name', ROBOTS)
def test_ros_serialization_and_mode_generation_invalidate_cached_command(name):
    from sim2real.rl_policy.robot_io.ros2_named import NamedRosRobotIO
    codec = codec_for(name)
    feed(codec)
    module = importlib.import_module(f'sim2real.rl_policy.robot_io.{name}')
    cls = getattr(module, {'q1': 'Q1RobotIO', 'a3': 'A3RobotIO', 't800': 'T800RobotIO', 'pm01': 'PM01RobotIO'}[name])
    io = cls.__new__(cls)
    io.codec = codec
    io._sequence = 0
    io._array_type = lambda: NS(header=NS(stamp=None), joints=[])
    io._joint_type = lambda: NS(sequence=0)
    io._node = NS(get_clock=lambda: NS(now=lambda: NS(to_msg=lambda: 'stamp')))
    sent = {}
    io._publishers = {g: NS(publish=lambda msg, group=g: sent.update({group: msg}), get_subscription_count=lambda: 1)
                      for g in codec.groups}
    io.set_control_mode('policy', 1)
    n = len(codec.cfg.joint_names)
    # ActionManager calls RobotIO using these public keyword names.
    io.write_command(q_target=np.zeros(n), dq_target=np.zeros(n), tau_ff=np.zeros(n),
                     kp=np.full(n, 12.), kd=np.full(n, .5))
    io._publish()
    assert codec.error is None
    if name in {'t800', 'pm01'}:
        msg = sent['body']
        assert len(msg.position) == len(codec.cfg.joint_names)
        assert msg.parallel_parser_type == {'t800': 1, 'pm01': 0}[name]
        assert msg.stiffness == [12.] * len(codec.cfg.joint_names) and msg.feed_forward_torque == [0.] * len(codec.cfg.joint_names)
    else:
        for group, names in codec.groups.items():
            assert [j.name for j in sent[group].joints] == list(names)
            assert all(j.sequence == 1 for j in sent[group].joints)
    io.set_control_mode('zero', 2)
    io.set_control_mode('policy', 3)
    assert codec.command is None
    io._publish()
    if name in {'t800', 'pm01'}:
        assert sent['body'].stiffness == [0.] * len(codec.cfg.joint_names)
        codec.write_command(np.zeros(n), np.zeros(n), np.zeros(n), np.full(n, 12.), np.full(n, .5))
        codec.motion_mode_time -= 2.
        io._publish()
        assert sent['body'].stiffness == [0.] * len(codec.cfg.joint_names)
        sent.clear()
        codec.receive_motion_state(NS(current_motion_task='pd_stand'))
        io._publish()
        assert not sent
    else:
        assert all(j.stiffness == 0 for msg in sent.values() for j in msg.joints)
