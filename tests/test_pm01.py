"""PM01 firmware compatibility and the released 24-joint policy contract."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.pm01 import PM01Codec, PM01_SDK_JOINT_NAMES


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


def test_ros_round_trip_isolated_from_hardware(monkeypatch):
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
            state.header.stamp = stamp
            state.position = [0.] * 24
            state.velocity = [0.] * 24
            state.torque = [0.] * 24
            imu = messages.ImuInfo()
            imu.header.stamp = stamp
            imu.quaternion.w = 1.
            joint_pub.publish(state)
            imu_pub.publish(imu)
            mode_pub.publish(mode)
            if command:
                z = np.zeros(24)
                io.write_command(z, z, z, np.full(24, 12.), np.full(24, .5))
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
        assert any(list(m.stiffness) == [12.] * 24 and m.parallel_parser_type == 1
                   and len(m.position) == 24 for m in received)
        # Keep feedback fresh, stop policy commands: the independent ROS timer
        # must replace the last active command with damping after 250 ms.
        deadline = time.monotonic() + .4
        while time.monotonic() < deadline:
            tick()
        assert list(received[-1].stiffness) == [0.] * 24
        assert list(received[-1].damping) == [2.] * 24
        assert list(received[-1].velocity) == [0.] * 24
        assert list(received[-1].feed_forward_torque) == [0.] * 24
    finally:
        if io is not None:
            io.close()
        node.destroy_node()
        rclpy.shutdown()
