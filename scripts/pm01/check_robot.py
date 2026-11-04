"""Read-only PM01 readiness check: subscribes to feedback, never sends commands."""
import argparse
import json
import math
import time

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.pm01 import PM01Codec, PM01_SDK_JOINT_NAMES


def check_robot(duration=5.):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointState, ImuInfo, MotionState

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('duration must be finite and positive')
    rclpy.init()
    node = rclpy.create_node('sim2real_pm01_check')
    codec = PM01Codec(get_robot_cfg('pm01'), {'body': PM01_SDK_JOINT_NAMES})
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    counts = {'joint_state': 0, 'imu_info': 0, 'motion_state': 0}

    def receive(key, callback):
        def handler(message):
            counts[key] += 1
            callback(message)
        return handler

    try:
        node.create_subscription(JointState, '/hardware/joint_state',
            receive('joint_state', lambda msg: codec.receive_group('body', msg)), qos)
        node.create_subscription(ImuInfo, '/hardware/imu_info', receive('imu_info', codec.receive_imu), qos)
        node.create_subscription(MotionState, '/motion/motion_state',
            receive('motion_state', codec.receive_motion_state), qos)
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(.05, max(0., deadline - time.monotonic())))
        subscribers = node.get_subscriptions_info_by_topic('/hardware/joint_command')
        # The transport sends BEST_EFFORT commands, which require a subscriber
        # that does not request RELIABLE delivery.
        compatible = [s for s in subscribers if s.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT]
        command_publishers = node.get_publishers_info_by_topic('/hardware/joint_command')
        ready = codec.error is None and codec._fresh() and bool(compatible) and not command_publishers
        result = {
            'ready': ready, 'read_only': True, 'expected_joints': 24,
            'motion_state_fields': MotionState.get_fields_and_field_types(),
            'motion_mode': codec.motion_mode, 'received': counts,
            'command_subscribers': len(subscribers), 'compatible_command_subscribers': len(compatible),
            'other_command_publishers': [p.node_name for p in command_publishers],
            'feedback_error': str(codec.error) if codec.error else None,
            'fresh_joint_imu_and_joint_bridge': codec._fresh(),
        }
        if codec.imu is not None:
            result['pelvis_quaternion_wxyz'] = codec.imu[1].tolist()
            result['pelvis_gyro_rad_s'] = codec.imu[2].tolist()
        print(json.dumps(result, indent=2))
        return ready
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration', type=float, default=5.)
    args = parser.parse_args()
    raise SystemExit(0 if check_robot(args.duration) else 2)
