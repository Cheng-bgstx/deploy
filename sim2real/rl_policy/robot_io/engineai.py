"""EngineAI array transport with mode ownership, feedback and command watchdogs.

Load interface_protocol built from the robot's installed schema. Firmware using
current_motion_task_name and current_motion_task is supported; DDS still requires
matching message definitions at both ends.
"""
import time
from types import SimpleNamespace as NS

from sim2real.rl_policy.robot_io.ros2_named import NamedJointCodec, NamedRosRobotIO


class EngineAICodec(NamedJointCodec):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_mode, self.motion_mode_time = None, 0.

    def receive_motion_state(self, msg):
        try:
            modes = [getattr(msg, key) for key in ('current_motion_task', 'current_motion_task_name')
                     if hasattr(msg, key)]
            if not modes or any(not isinstance(mode, str) or not mode for mode in modes) or len(set(modes)) != 1:
                raise ValueError('Unrecognized or conflicting EngineAI MotionState fields')
            with self.lock:
                self.motion_mode = modes[0]
                self.motion_mode_time = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.error = exc

    def _fresh(self):
        return (super()._fresh() and self.motion_mode == 'joint_bridge'
                and time.monotonic() - self.motion_mode_time <= 1.)

    def receive_group(self, group, msg):
        try:
            names = self.groups[group]
            arrays = [self.vector(getattr(msg, field), len(names)) for field in ('position', 'velocity', 'torque')]
            named = NS(header=msg.header, joints=[NS(name=n, position=arrays[0][i], velocity=arrays[1][i], effort=arrays[2][i])
                       for i, n in enumerate(names)])
            super().receive_group(group, named)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def receive_imu(self, msg):
        super().receive_imu(NS(header=msg.header, orientation=msg.quaternion, angular_velocity=msg.angular_velocity))


class EngineAIRobotIO(NamedRosRobotIO):
    robot_name: str
    sdk_joint_names: tuple[str, ...]
    codec_type = EngineAICodec

    def __init__(self, cfg, *, publish_hz=500., imu_topic='/hardware/imu_info', **options):
        if cfg.name != self.robot_name:
            raise ValueError(f'EngineAI backend requires robot={self.robot_name}')
        from interface_protocol.msg import JointCommand, JointState, ImuInfo, MotionState
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        super().__init__(cfg, groups={'body': self.sdk_joint_names},
            topics={'body': ('/hardware/joint_state', '/hardware/joint_command')},
            types=(JointCommand, None, JointState), imu_topic=imu_topic,
            imu_type=ImuInfo, codec_type=self.codec_type, publish_hz=publish_hz, **options)
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
            msg.torque = [0.] * len(q)
            msg.stiffness, msg.damping = kp.tolist(), kd.tolist()
            # RL_PARSER consumes serial pitch/roll targets and performs the
            # SDK's parallel-ankle transform, matching the vendor RL example.
            msg.parallel_parser_type = 1
            self._publishers['body'].publish(msg)
        except Exception as exc:
            with self.codec.lock:
                self.codec.error = exc
