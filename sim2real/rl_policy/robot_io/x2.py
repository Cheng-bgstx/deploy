"""AgiBot X2 AimDK ROS 2 HAL transport with group mapping and a command watchdog.

Wire schema/order: x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html
The historical 'torso' IMU topic is the pelvis IMU, not the articulated chest IMU.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from sim2real.config.robots.base import RobotCfg
from sim2real.config.robots.x2 import X2_AIMDK_GROUPS, x2_deploy_joint_limits
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.utils.math import quat_conjugate, quat_mul, quat_rotate_numpy


class X2StateCommand:
    """Pure array/message boundary; no ROS imports, suitable for offline checks."""

    def __init__(self, cfg: RobotCfg, *, timeout_s=.25, head_pitch_enabled=False,
                 imu_to_pelvis_wxyz=(1., 0., 0., 0.)):
        self.cfg, self.timeout_s = cfg, float(timeout_s)
        if cfg.name != 'x2' or self.timeout_s <= 0:
            raise ValueError('X2 backend requires robot=x2 and positive timeout_s')
        self.head_pitch_enabled = bool(head_pitch_enabled)
        self.rotation = self.vector(imu_to_pelvis_wxyz, 4)
        if np.linalg.norm(self.rotation) < 1e-6:
            raise ValueError('Invalid IMU extrinsic quaternion')
        self.rotation /= np.linalg.norm(self.rotation)
        self.group_ids = {g: np.array([cfg.joint_names.index(n) for n in names])
                          for g, names in X2_AIMDK_GROUPS.items()}
        self.groups, self.imu = {}, None
        self.command = None
        self.command_time = 0.
        self.mode = 'zero'
        self.error = None
        self.lock = threading.Lock()
        # Never widen the physical robot limits to the training model's wider bounds.
        limits = x2_deploy_joint_limits(cfg)
        self.lower = np.array([limits[n][0] for n in cfg.joint_names])
        self.upper = np.array([limits[n][1] for n in cfg.joint_names])

    @staticmethod
    def vector(value, count):
        out = np.asarray(value, dtype=np.float32)
        if out.shape != (count,) or not np.isfinite(out).all():
            raise ValueError(f'Expected {count} finite values, got {out.shape}')
        return out.copy()

    def receive_group(self, group, msg):
        try:
            names = X2_AIMDK_GROUPS[group]
            joints = list(msg.joints)
            if len(joints) != len(names):
                raise ValueError(f'X2 {group}: expected {len(names)} feedback joints')
            wire = [j.name for j in joints]
            if any(wire):
                normalized = [n if n.endswith('_joint') else n + '_joint' for n in wire]
                if sorted(normalized) != sorted(names):
                    raise ValueError(f'X2 {group} joint names differ from documented layout: {wire}')
                joints = [joints[normalized.index(n)] for n in names]
            if any(int(getattr(j, 'error_code', 0)) for j in joints):
                raise ValueError(f'X2 {group} reports a joint error')
            values = tuple(self.vector([getattr(j, attr) for j in joints], len(names))
                           for attr in ('position', 'velocity', 'effort'))
            with self.lock:
                self.groups[group] = (time.monotonic(), values)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def receive_imu(self, msg):
        try:
            o, w = msg.orientation, msg.angular_velocity
            q = self.vector([o.w, o.x, o.y, o.z], 4)
            if np.linalg.norm(q) < 1e-6:
                raise ValueError('Invalid X2 IMU orientation')
            q /= np.linalg.norm(q)
            gyro = self.vector([w.x, w.y, w.z], 3)
            # R_WB = R_WI R_BI^-1; omega_B = R_BI omega_I.
            q = quat_mul(q[None], quat_conjugate(self.rotation[None]))[0]
            gyro = quat_rotate_numpy(self.rotation[None], gyro[None])[0]
            with self.lock:
                self.imu = (time.monotonic(), q, gyro)
        except Exception as exc:
            with self.lock:
                self.error = exc

    def _fresh(self, now):
        return (self.imu is not None and all(g in self.groups for g in self.group_ids)
                and now - min(self.imu[0], *(self.groups[g][0] for g in self.group_ids)) <= self.timeout_s)

    def read_state(self):
        with self.lock:
            if self.error:
                raise RuntimeError('X2 feedback failed') from self.error
            now = time.monotonic()
            if not self._fresh(now):
                return None
            qpos = np.zeros(7 + len(self.cfg.joint_names), dtype=np.float32)
            qvel = np.zeros(6 + len(self.cfg.joint_names), dtype=np.float32)
            tau = np.zeros(len(self.cfg.joint_names), dtype=np.float32)
            qpos[3:7], qvel[3:6] = self.imu[1:]
            for group, ids in self.group_ids.items():
                q, dq, effort = self.groups[group][1]
                qpos[7 + ids], qvel[6 + ids], tau[ids] = q, dq, effort
            return RobotState(qpos=qpos, qvel=qvel, joint_torque=tau, tick=round(now * 1000))

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        values = tuple(self.vector(v, len(self.cfg.joint_names)) for v in (q_target, dq_target, tau_ff, kp, kd))
        if np.any(values[3] < 0) or np.any(values[4] < 0):
            raise ValueError('X2 gains must be non-negative')
        values[0][:] = np.clip(values[0], self.lower, self.upper)
        with self.lock:
            if self.error:
                raise RuntimeError('X2 feedback failed') from self.error
            self.command, self.command_time = values, time.monotonic()

    def command_groups(self, *, force_damping=False):
        with self.lock:
            if self.command is None:
                return {}
            now = time.monotonic()
            values = [a.copy() for a in self.command]
            damping = (force_damping or self.mode == 'zero' or self.error is not None
                       or now - self.command_time > self.timeout_s or not self._fresh(now))
            if damping:
                values[1].fill(0)
                values[2].fill(0)
                values[3].fill(0)
            if not self.head_pitch_enabled:
                i = self.cfg.joint_names.index('head_pitch_joint')
                for a in values:
                    a[i] = 0.
            return {g: tuple(a[ids] for a in values) for g, ids in self.group_ids.items()}


class X2RobotIO(RobotIO):
    def __init__(self, cfg: RobotCfg, *, publish_hz=500., timeout_s=.25, head_pitch_enabled=False,
                 imu_to_pelvis_wxyz=(1., 0., 0., 0.), imu_topic='/aima/hal/imu/torso/state'):
        if publish_hz <= 0:
            raise ValueError('publish_hz must be positive')
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
            from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
            from sensor_msgs.msg import Imu
        except ImportError as exc:
            raise ImportError('Source ROS 2 Humble and the X2 AimDK install/setup.bash before running inline X2') from exc
        self.codec = X2StateCommand(cfg, timeout_s=timeout_s, head_pitch_enabled=head_pitch_enabled,
                                   imu_to_pelvis_wxyz=imu_to_pelvis_wxyz)
        self._joint_type, self._array_type = JointCommand, JointCommandArray
        self._context = Context()
        rclpy.init(context=self._context)
        self._node = Node('sim2real_x2', context=self._context)
        rx = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        tx = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
        self._publishers, self._subscriptions = {}, []
        for group in X2_AIMDK_GROUPS:
            prefix = f'/aima/hal/joint/{group}'
            self._publishers[group] = self._node.create_publisher(JointCommandArray, prefix + '/command', tx)
            self._subscriptions.append(self._node.create_subscription(JointStateArray, prefix + '/state',
                lambda msg, g=group: self.codec.receive_group(g, msg), rx))
        self._subscriptions.append(self._node.create_subscription(Imu, imu_topic, self.codec.receive_imu, rx))
        self._timer = self._node.create_timer(1 / publish_hz, self._publish)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._thread = threading.Thread(target=self._executor.spin, name='x2-aimdk', daemon=True)
        self._thread.start()

    def read_state(self):
        return self.codec.read_state()

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        self.codec.write_command(q_target, dq_target, tau_ff, kp, kd)

    def set_control_mode(self, mode, generation):
        if mode not in {'zero', 'init', 'policy'}:
            raise ValueError(f'Unknown X2 control mode: {mode}')
        with self.codec.lock:
            self.codec.mode = mode

    def _publish(self, *, force_damping=False):
        try:
            for group, values in self.codec.command_groups(force_damping=force_damping).items():
                msg = self._array_type()
                msg.header.stamp = self._node.get_clock().now().to_msg()
                for i, name in enumerate(X2_AIMDK_GROUPS[group]):
                    joint = self._joint_type()
                    joint.name = name
                    joint.position, joint.velocity, joint.effort, joint.stiffness, joint.damping = [float(v[i]) for v in values]
                    msg.joints.append(joint)
                self._publishers[group].publish(msg)
        except Exception as exc:
            with self.codec.lock:
                self.codec.error = exc

    def close(self):
        self._timer.cancel()
        # Finish any in-flight normal publish before replacing it with damping.
        self._executor.shutdown(timeout_sec=1.)
        self._thread.join(timeout=1.)
        # HAL has no command timeout: explicitly replace the last command on exit.
        for _ in range(5):
            self._publish(force_damping=True)
            time.sleep(.002)
        self._node.destroy_node()
        self._context.shutdown()
