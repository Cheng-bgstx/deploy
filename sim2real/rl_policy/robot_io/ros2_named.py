"""Named ROS joint groups, root IMU conversion, and independent command watchdog.

Wire message types and topics are supplied by each verified vendor adapter. No
ROS dependency is imported until constructing the transport; the codec is testable.
"""
import threading
import time

import numpy as np

from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.utils.math import quat_conjugate, quat_mul, quat_rotate_numpy


class NamedJointCodec:
    def __init__(self, cfg, groups, *, timeout_s=.25, damping=2., max_position_step=.2,
                 imu_to_pelvis_wxyz=(1., 0., 0., 0.), fixed_joints=None, physical_limits=None):
        self.cfg, self.groups = cfg, groups
        self.timeout_s, self.damping, self.max_step = float(timeout_s), float(damping), float(max_position_step)
        if not all(np.isfinite(v) and v > 0 for v in (self.timeout_s, self.damping, self.max_step)):
            raise ValueError('timeout, damping and max_position_step must be finite and positive')
        self.fixed = dict(fixed_joints or {})
        physical_limits = physical_limits or {}
        self.limits = {n: (max(cfg.joint_pos_lower_limit[n], physical_limits.get(n, (-np.inf, np.inf))[0]),
                           min(cfg.joint_pos_upper_limit[n], physical_limits.get(n, (-np.inf, np.inf))[1]))
                       for n in cfg.joint_names}
        if any(lo >= hi for lo, hi in self.limits.values()):
            raise ValueError('Training and physical joint limits have no intersection')
        wire_names = [n for group in groups.values() for n in group]
        if len(set(wire_names)) != len(wire_names) or set(wire_names) != set(cfg.joint_names) | set(self.fixed):
            raise ValueError('Each policy or fixed joint must appear in exactly one wire group')
        self.rotation = self.vector(imu_to_pelvis_wxyz, 4)
        if np.linalg.norm(self.rotation) < 1e-6:
            raise ValueError('Invalid IMU extrinsic quaternion')
        self.rotation /= np.linalg.norm(self.rotation)
        self.state, self.imu, self.command = {}, None, None
        self.command_time, self.mode, self.error = 0., 'zero', None
        self.last_position, self.stamps = {}, {}
        self.lock = threading.RLock()

    @staticmethod
    def vector(value, count):
        v = np.asarray(value, np.float32)
        if v.shape != (count,) or not np.isfinite(v).all():
            raise ValueError(f'Expected {count} finite values, got {v.shape}')
        return v.copy()

    def _new_stamp(self, key, msg):
        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if stamp is None:
            return True
        ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if ns <= self.stamps.get(key, -1):
            return False
        self.stamps[key] = ns
        return True

    def receive_group(self, group, msg):
        try:
            names = self.groups[group]
            wire = [j.name for j in msg.joints]
            if sorted(wire) != sorted(names):
                raise ValueError(f'{group}: feedback names differ from the vendor layout: {wire}')
            ordered = [msg.joints[wire.index(n)] for n in names]
            values = tuple(self.vector([getattr(j, attr) for j in ordered], len(names))
                           for attr in ('position', 'velocity', 'effort'))
            if any(int(getattr(j, 'error_code', 0)) for j in ordered):
                raise ValueError(f'{group}: motor fault')
            with self.lock:
                if self._new_stamp(group, msg):
                    self.state[group] = (time.monotonic(), values)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def receive_imu(self, msg):
        try:
            if getattr(msg, 'orientation_covariance', [0.])[0] < 0:
                raise ValueError('Root IMU has no orientation estimate')
            o, w = msg.orientation, msg.angular_velocity
            q = self.vector([o.w, o.x, o.y, o.z], 4)
            if np.linalg.norm(q) < 1e-6:
                raise ValueError('Invalid root IMU quaternion')
            q /= np.linalg.norm(q)
            gyro = self.vector([w.x, w.y, w.z], 3)
            q = quat_mul(q[None], quat_conjugate(self.rotation[None]))[0]
            gyro = quat_rotate_numpy(self.rotation[None], gyro[None])[0]
            with self.lock:
                if self._new_stamp('imu', msg):
                    self.imu = (time.monotonic(), q, gyro)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def _fresh(self):
        return (self.imu is not None and set(self.state) == set(self.groups)
                and time.monotonic() - min(self.imu[0], *(s[0] for s in self.state.values())) <= self.timeout_s)

    def read_state(self):
        with self.lock:
            if self.error:
                raise RuntimeError('Vendor feedback/command failed') from self.error
            if not self._fresh():
                return None
            count = len(self.cfg.joint_names)
            qpos, qvel, torque = np.zeros(7 + count, np.float32), np.zeros(6 + count, np.float32), np.zeros(count, np.float32)
            qpos[3:7], qvel[3:6] = self.imu[1:]
            for group, names in self.groups.items():
                for i, n in enumerate(names):
                    if n not in self.cfg.joint_names:
                        continue
                    j = self.cfg.joint_names.index(n)
                    qpos[7 + j], qvel[6 + j], torque[j] = [v[i] for v in self.state[group][1]]
            return RobotState(qpos=qpos, qvel=qvel, joint_torque=torque, tick=round(self.imu[0] * 1000))

    def write_command(self, q, dq, tau, kp, kd):
        values = tuple(self.vector(v, len(self.cfg.joint_names)) for v in (q, dq, tau, kp, kd))
        if np.any(values[3] < 0) or np.any(values[4] < 0):
            raise ValueError('Gains must be non-negative')
        with self.lock:
            if self.error:
                raise RuntimeError('Vendor feedback/command failed') from self.error
            self.command, self.command_time = values, time.monotonic()

    def command_groups(self, force_damping=False):
        with self.lock:
            if set(self.state) != set(self.groups):
                return {}
            damp = (force_damping or self.error is not None or self.mode == 'zero'
                    or not self._fresh() or self.command is None
                    or time.monotonic() - self.command_time > self.timeout_s)
            result = {}
            for group, names in self.groups.items():
                values = np.zeros((5, len(names)), np.float32)
                values[0] = self.state[group][1][0]
                values[4] = self.damping
                if not damp:
                    for i, n in enumerate(names):
                        if n in self.fixed:
                            values[:, i] = self.fixed[n]  # q, dq, tau, kp, kd
                        else:
                            j = self.cfg.joint_names.index(n)
                            values[:, i] = [v[j] for v in self.command]
                            values[0, i] = np.clip(values[0, i], *self.limits[n])
                            values[1, i] = np.clip(values[1, i], -self.cfg.joint_velocity_limit[n], self.cfg.joint_velocity_limit[n])
                            values[2, i] = np.clip(values[2, i], -self.cfg.joint_effort_limit[n], self.cfg.joint_effort_limit[n])
                        previous = self.last_position.get(n, float(self.state[group][1][0][i]))
                        values[0, i] = previous + np.clip(values[0, i] - previous, -self.max_step, self.max_step)
                for i, n in enumerate(names):
                    self.last_position[n] = float(values[0, i])
                result[group] = values
            return result


class NamedRosRobotIO(RobotIO):
    def __init__(self, cfg, *, groups, topics, types, imu_topic, publish_hz=500.,
                 codec_type=NamedJointCodec, imu_type=None, **codec_options):
        if not imu_topic:
            raise ValueError('Configure a verified pelvis IMU sensor_msgs/Imu topic in robot_io_options.imu_topic')
        if not np.isfinite(publish_hz) or publish_hz <= 0:
            raise ValueError('publish_hz must be finite and positive')
        import rclpy
        from rclpy.context import Context
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from sensor_msgs.msg import Imu
        self.codec = codec_type(cfg, groups, **codec_options)
        self._array_type, self._joint_type, state_type = types
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = Node(f'sim2real_{cfg.name}', context=self._context)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self._publishers, self._subscriptions = {}, []
        for group, (state_topic, command_topic) in topics.items():
            self._publishers[group] = self._node.create_publisher(self._array_type, command_topic, qos)
            self._subscriptions.append(self._node.create_subscription(state_type, state_topic,
                lambda msg, g=group: self.codec.receive_group(g, msg), qos))
        self._subscriptions.append(self._node.create_subscription(imu_type or Imu, imu_topic, self.codec.receive_imu, qos))
        self._sequence = 0
        self._timer = self._node.create_timer(1 / publish_hz, self._publish)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name=f'{cfg.name}-hal', daemon=True)
        self._thread.start()

    def read_state(self):
        return self.codec.read_state()

    def write_command(self, *values):
        self.codec.write_command(*values)

    def set_control_mode(self, mode, generation):
        if mode not in {'zero', 'init', 'policy'}:
            raise ValueError(f'Unknown control mode {mode}')
        with self.codec.lock:
            if mode != self.codec.mode or generation != getattr(self.codec, 'generation', None):
                self.codec.command, self.codec.command_time = None, 0.
            self.codec.mode, self.codec.generation = mode, generation

    def _publish(self, *, force_damping=False):
        try:
            self._sequence = (self._sequence + 1) % (2 ** 32)
            for group, values in self.codec.command_groups(force_damping).items():
                msg = self._array_type()
                msg.header.stamp = self._node.get_clock().now().to_msg()
                if hasattr(msg.header, 'sequence'):
                    msg.header.sequence = self._sequence
                for i, name in enumerate(self.codec.groups[group]):
                    joint = self._joint_type()
                    joint.name = name
                    if hasattr(joint, 'sequence'):
                        joint.sequence = self._sequence
                    joint.position, joint.velocity, joint.effort, joint.stiffness, joint.damping = [float(v[i]) for v in values]
                    msg.joints.append(joint)
                self._publishers[group].publish(msg)
        except Exception as exc:
            with self.codec.lock:
                self.codec.error = exc

    def close(self):
        self._timer.cancel()
        self._executor.shutdown(timeout_sec=1.)
        self._thread.join(timeout=1.)
        for _ in range(5):
            self._publish(force_damping=True)
            time.sleep(.002)
        self._node.destroy_node()
        self._context.shutdown()
