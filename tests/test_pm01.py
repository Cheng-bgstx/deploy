"""PM01 firmware compatibility and the released 24-joint policy contract."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.pm01 import PM01Codec, PM01_SDK_JOINT_NAMES
from sim2real.rl_policy.robot_io.t800 import T800Codec


def codec():
    cfg = replace(get_robot_cfg('pm01'), joint_names=tuple(reversed(PM01_SDK_JOINT_NAMES)))
    return PM01Codec(cfg, {'body': PM01_SDK_JOINT_NAMES})


@pytest.mark.parametrize('field', ['current_motion_task', 'current_motion_task_name'])
def test_firmware_mode_schema_and_ownership(field):
    c = codec()
    c.receive_motion_state(NS(**{field: 'joint_bridge'}))
    stamp = NS(stamp=NS(sec=1, nanosec=0))
    q = np.arange(24) * .001
    c.receive_group('body', NS(header=stamp, position=q, velocity=q, torque=q))
    c.receive_imu(NS(header=stamp, quaternion=NS(w=1., x=0., y=0., z=0.),
                     angular_velocity=NS(x=0., y=0., z=0.)))
    np.testing.assert_allclose(c.read_state().qpos[7:], q[::-1])
    c.receive_motion_state(NS(**{field: 'pd_stand'}))
    assert c.read_state() is None
    c.receive_motion_state(NS(**{field: 'joint_bridge'}))
    c.motion_mode_time -= 2.
    assert c.read_state() is None


@pytest.mark.parametrize('message', [NS(), NS(current_motion_task=''),
    NS(current_motion_task='joint_bridge', current_motion_task_name='pd_stand')])
def test_bad_mode_schema_latches_error(message):
    c = codec()
    c.receive_motion_state(message)
    with pytest.raises(RuntimeError):
        c.read_state()


def test_zero_stamp_firmware_refreshes_feedback_with_monotonic_watchdog(monkeypatch):
    now = [100.]
    monkeypatch.setattr('time.monotonic', lambda: now[0])
    c = codec()
    header = NS(stamp=NS(sec=0, nanosec=0))
    def feedback():
        c.receive_group('body', NS(header=header, position=[.1]*24, velocity=[0.]*24, torque=[0.]*24))
        c.receive_imu(NS(header=header, quaternion=NS(w=1., x=0., y=0., z=0.),
                         angular_velocity=NS(x=0., y=0., z=0.)))
        c.receive_motion_state(NS(current_motion_task_name='joint_bridge'))
    feedback()
    now[0] += .5
    feedback()
    assert c.read_state() is not None
    c.mode = 'policy'
    z = np.zeros(24)
    c.write_command(z, z, z, np.full(24, 10.), np.full(24, 2.4916))
    values = c.command_groups()['body']
    assert np.all(values[3] == 10.) and np.all(values[4] == 2.4916)
    assert c.parallel_parser_type == 0
    now[0] += .3
    assert c.read_state() is None


def test_pm01_nonzero_stamps_keep_replay_protection():
    c = codec()
    def accept(sec):
        return c._new_stamp('body', NS(header=NS(stamp=NS(sec=sec, nanosec=0))))
    assert accept(0)
    assert accept(0)
    assert accept(10)
    assert not accept(10)
    assert not accept(9)
    assert not accept(0)
    assert accept(11)


@pytest.mark.parametrize('count', [23, 25])
def test_rejects_other_robot_joint_counts(count):
    c = codec()
    c.receive_group('body', NS(header=NS(), position=[0.] * count,
                              velocity=[0.] * count, torque=[0.] * count))
    with pytest.raises(RuntimeError):
        c.read_state()


def test_ws61_contract_preserves_pm01_wrist_signs_and_head_order():
    import json
    from scripts.export_proteus_onnx import policy_yaml
    p = Path('checkpoints/proteus/wsq61/engineai_pm01/engineai_pm01.json')
    if not p.is_file():
        pytest.skip('Prepare PM01 assets first')
    contract = json.loads(p.read_text())
    cfg = policy_yaml(contract, 'pm01', 16)
    assert cfg['policy_joint_names'] == list(PM01_SDK_JOINT_NAMES)
    assert cfg['policy_joint_names'][12] == 'J12_WAIST_YAW'
    assert cfg['policy_joint_names'][-1] == 'J23_HEAD_YAW'
    assert all(cfg['action_scale'][n] < 0 for n in ('J17_ELBOW_YAW_L', 'J22_ELBOW_YAW_R'))
    assert get_robot_cfg('pm_v2') is get_robot_cfg('engineai_pm01')
    from sim2real.rl_policy.base_policy import BasePolicy
    from sim2real.rl_policy.utils.joint_targets import JointTargetPostprocessor
    robot = get_robot_cfg('pm01')
    assert 'soft_joint_pos_limits' not in cfg
    assert 'max_target_velocity_rad_s' not in cfg
    processor = JointTargetPostprocessor.from_policy_config(cfg,
        joint_names=robot.joint_names, policy_joint_names=cfg['policy_joint_names'], control_dt=.02)
    assert processor is None
    rig = NS(policy_config=cfg, target_postprocessor=processor,
             motion_first_frame_joint_pos=lambda: np.full(24, .4))
    np.testing.assert_array_equal(BasePolicy._resolve_shaping_target(rig, 'motion_first_frame'), .4)


def test_pm01_uses_classic_parser_while_t800_keeps_rl_parser():
    assert PM01Codec.parallel_parser_type == 0
    assert T800Codec.parallel_parser_type == 1


def test_pm01_transport_forwards_training_gains_and_uses_measured_pose_on_stop():
    c = codec()  # Reversed runtime order tests name-based SDK mapping too.
    header = NS(stamp=NS(sec=0, nanosec=0))
    c.receive_motion_state(NS(current_motion_task_name='joint_bridge'))
    q = np.linspace(-.3, .3, 24)
    c.receive_group('body', NS(header=header, position=q[::-1].copy(), velocity=[0.]*24, torque=[0.]*24))
    c.receive_imu(NS(header=header, quaternion=NS(w=1., x=0., y=0., z=0.),
                     angular_velocity=NS(x=0., y=0., z=0.)))
    c.mode = 'policy'
    arrays = [q, np.zeros(24), np.zeros(24), np.full(24, 37.701), np.full(24, 2.4916)]
    c.write_command(*arrays)
    actual = c.command_groups()['body']
    np.testing.assert_array_equal(actual, np.array(arrays, dtype=np.float32)[:, ::-1])
    assert c.parallel_parser_type == 0
    # A stale command falls back to the measured pose with the base passive
    # damping; it is a normal stop, not a latched state.
    c.command_time -= 1.
    values = c.command_groups()['body']
    np.testing.assert_allclose(values[0], q[::-1], atol=1e-6)
    np.testing.assert_array_equal(values[1:4], 0)
    np.testing.assert_array_equal(values[4], c.damping)
    assert c.parallel_parser_type == 0
    c.write_command(*arrays)
    c.mode = 'policy'
    np.testing.assert_array_equal(c.command_groups()['body'], np.array(arrays, dtype=np.float32)[:, ::-1])


def test_pm01_runtime_action_targets_match_canonical_training_formula(monkeypatch):
    import json
    from scripts.export_proteus_onnx import policy_yaml
    from sim2real.rl_policy import base_policy
    from sim2real.rl_policy.utils.joint_targets import JointTargetPostprocessor
    path = Path('checkpoints/proteus/wsq61/engineai_pm01/engineai_pm01.json')
    if not path.is_file():
        pytest.skip('Prepare PM01 assets first')
    cfg = policy_yaml(json.loads(path.read_text()), 'pm01', 16)
    names = list(reversed(PM01_SDK_JOINT_NAMES))
    policy = base_policy.BasePolicy.__new__(base_policy.BasePolicy)
    policy.inference_backend = 'onnx-cpu'
    policy.default_dof_angles = np.array([cfg['default_joint_pos'][n] for n in names])
    policy.controlled_joint_indices = [names.index(n) for n in cfg['policy_joint_names']]
    policy.action_scale = np.array([cfg['action_scale'][n] for n in cfg['policy_joint_names']])
    policy.target_postprocessor = JointTargetPostprocessor.from_policy_config(cfg,
        joint_names=names, policy_joint_names=cfg['policy_joint_names'], control_dt=.02)
    assert policy.target_postprocessor is None
    actions = np.linspace(-8., 8., 24, dtype=np.float32)
    monkeypatch.setattr(base_policy, 'build_inference_module', lambda *a: lambda state: {'action': actions.copy()})
    policy.setup_policy('mock-network-only.onnx')
    for sign in (1., -1., 1.):
        actions *= sign
        actual_action, target, state = policy.policy({})
        expected = np.array([cfg['default_joint_pos'][n] +
            cfg['action_scale'][n]*actions[cfg['policy_joint_names'].index(n)] for n in names])
        np.testing.assert_array_equal(actual_action, actions)
        np.testing.assert_array_equal(target, expected)


@pytest.mark.parametrize('zero_header_stamp', [False, True])
def test_ros_round_trip_isolated_from_hardware(monkeypatch, zero_header_stamp):
    """Exercise real DDS and generated schemas, on loopback in a separate domain."""
    import time
    rclpy = pytest.importorskip('rclpy')
    messages = pytest.importorskip('interface_protocol.msg')
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sim2real.rl_policy.robot_io.pm01 import PM01RobotIO

    monkeypatch.setenv('ROS_DOMAIN_ID', '171')
    monkeypatch.setenv('ROS_LOCALHOST_ONLY', '1')
    monkeypatch.setenv('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    # Humble adds the loopback interface itself for ROS_LOCALHOST_ONLY=1.
    # Explicitly selecting lo as well makes Cyclone reject the duplicate.
    monkeypatch.delenv('CYCLONEDDS_URI', raising=False)
    rclpy.init()
    node = rclpy.create_node('pm01_test_firmware')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    joint_pub = node.create_publisher(messages.JointState, '/hardware/joint_state', qos)
    imu_pub = node.create_publisher(messages.ImuInfo, '/hardware/imu_info', qos)
    mode_pub = node.create_publisher(messages.MotionState, '/motion/motion_state', qos)
    received = []
    node.create_subscription(messages.JointCommand, '/hardware/joint_command', received.append, qos)
    io = None
    try:
        io = PM01RobotIO(get_robot_cfg('pm01'), publish_hz=100.)
        mode = messages.MotionState()
        field = 'current_motion_task_name' if hasattr(mode, 'current_motion_task_name') else 'current_motion_task'
        setattr(mode, field, 'joint_bridge')

        def tick(*, command=False):
            stamp = node.get_clock().now().to_msg()
            state = messages.JointState()
            if not zero_header_stamp:
                state.header.stamp = stamp
            state.position = [0.] * 24
            state.velocity = [0.] * 24
            state.torque = [0.] * 24
            imu = messages.ImuInfo()
            if not zero_header_stamp:
                imu.header.stamp = stamp
            imu.quaternion.w = 1.
            joint_pub.publish(state)
            imu_pub.publish(imu)
            mode_pub.publish(mode)
            if command:
                z = np.zeros(24)
                # Match ActionManager's RobotIO keyword-call contract.
                io.write_command(q_target=z, dq_target=z, tau_ff=z,
                                 kp=np.full(24, 12.), kd=np.full(24, .5))
            rclpy.spin_once(node, timeout_sec=.01)
            time.sleep(.005)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and io.read_state() is None:
            tick()
        assert io.read_state() is not None
        io.set_control_mode('policy', 1)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not any(list(m.stiffness) == [12.] * 24 for m in received):
            tick(command=True)
        assert any(list(m.stiffness) == [12.] * 24 and m.parallel_parser_type == 0
                   and len(m.position) == 24 for m in received)
        # Keep feedback fresh, stop policy commands: the independent ROS timer
        # must replace the last active command with damping after 250 ms.
        deadline = time.monotonic() + .4
        while time.monotonic() < deadline:
            tick()
        assert io.read_state() is not None
        assert list(received[-1].stiffness) == [0.] * 24
        assert list(received[-1].damping) == [2.] * 24
        assert list(received[-1].velocity) == [0.] * 24
        assert list(received[-1].feed_forward_torque) == [0.] * 24
    finally:
        if io is not None:
            io.close()
        node.destroy_node()
        rclpy.shutdown()
