"""EngineAI ROS joint_bridge transport for the standard 25-motor T800.

Schema, SDK array order and parser: engineai-robotics/engineai_ros2_workspace
interface_example/scripts/joint_bridge_example.py and config/t800/joint_test.yaml.
Requires firmware exposing joint_bridge; current native_sdk's read-only ROS bridge
alone is insufficient. The transport never requests a robot mode transition.
"""
import time
from types import SimpleNamespace as NS

from sim2real.rl_policy.robot_io.ros2_named import NamedJointCodec, NamedRosRobotIO

T800_SDK_JOINT_NAMES = (
    'J00_HIP_PITCH_L', 'J01_HIP_ROLL_L', 'J02_HIP_YAW_L', 'J03_KNEE_PITCH_L',
    'J04_ANKLE_PITCH_L', 'J05_ANKLE_ROLL_L', 'J06_HIP_PITCH_R', 'J07_HIP_ROLL_R',
    'J08_HIP_YAW_R', 'J09_KNEE_PITCH_R', 'J10_ANKLE_PITCH_R', 'J11_ANKLE_ROLL_R',
    'J12_TORSO_YAW', 'J13_SHOULDER_PITCH_L', 'J14_SHOULDER_ROLL_L', 'J15_SHOULDER_YAW_L',
    'J16_ELBOW_PITCH_L', 'J17_ELBOW_YAW_L', 'J18_SHOULDER_PITCH_R', 'J19_SHOULDER_ROLL_R',
    'J20_SHOULDER_YAW_R', 'J21_ELBOW_PITCH_R', 'J22_ELBOW_YAW_R', 'J23_HEAD_PITCH', 'J24_HEAD_YAW',
)


class T800Codec(NamedJointCodec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_mode, self.motion_mode_time = None, 0.

    def receive_motion_state(self, msg):
        with self.lock:
            self.motion_mode = str(msg.current_motion_task)
            self.motion_mode_time = time.monotonic()

    def _fresh(self):
        return (super()._fresh() and self.motion_mode == 'joint_bridge'
                and time.monotonic() - self.motion_mode_time <= 1.)

    def receive_group(self, group, msg):
        try:
            arrays = [self.vector(getattr(msg, field), 25) for field in ('position', 'velocity', 'torque')]
            named = NS(header=msg.header, joints=[NS(name=n, position=arrays[0][i], velocity=arrays[1][i], effort=arrays[2][i])
                       for i, n in enumerate(T800_SDK_JOINT_NAMES)])
            super().receive_group(group, named)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def receive_imu(self, msg):
        super().receive_imu(NS(header=msg.header, orientation=msg.quaternion, angular_velocity=msg.angular_velocity))


class T800RobotIO(NamedRosRobotIO):
    def __init__(self, cfg, *, publish_hz=500., imu_topic='/hardware/imu_info', **options):
        if cfg.name != 't800':
            raise ValueError('T800 backend requires robot=t800')
        from interface_protocol.msg import JointCommand, JointState, ImuInfo, MotionState
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        super().__init__(cfg, groups={'body': T800_SDK_JOINT_NAMES},
            topics={'body': ('/hardware/joint_state', '/hardware/joint_command')},
            types=(JointCommand, None, JointState), imu_topic=imu_topic,
            imu_type=ImuInfo, codec_type=T800Codec, publish_hz=publish_hz, **options)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._subscriptions.append(self._node.create_subscription(MotionState, '/motion/motion_state',
            self.codec.receive_motion_state, qos))

    def _publish(self, *, force_damping=False):
        try:
            # Publish only after the documented mode handshake and discovery of
            # a real command subscriber. Native SDK telemetry alone cannot drive.
            with self.codec.lock:
                ready = self.codec.motion_mode == 'joint_bridge'
                # A stale mode heartbeat makes _fresh false and still sends
                # damping; an explicit transition gives ownership back to firmware.
            if not ready or self._publishers['body'].get_subscription_count() == 0:
                return
            groups = self.codec.command_groups(force_damping)
            if 'body' not in groups:
                return
            q, dq, tau, kp, kd = groups['body']
            msg = self._array_type()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.position, msg.velocity = q.tolist(), dq.tolist()
            msg.feed_forward_torque = tau.tolist()
            msg.torque = [0.] * 25
            msg.stiffness, msg.damping = kp.tolist(), kd.tolist()
            # RL_PARSER consumes serial pitch/roll targets and performs the
            # SDK's parallel-ankle transform, matching the vendor RL example.
            msg.parallel_parser_type = 1
            self._publishers['body'].publish(msg)
        except Exception as exc:
            with self.codec.lock:
                self.codec.error = exc
