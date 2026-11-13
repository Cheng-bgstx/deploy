"""LimX HU_D04_01 SDK transport, using virtual pitch/roll (PR) joint coordinates.

The policy runs at 50 Hz; a separate publisher holds the latest command at 1 kHz,
like the official controller. Commands expire into damping if inference or feedback stops.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from sim2real.config.robots.base import RobotCfg
from sim2real.config.robots.oli import OLI_SDK_JOINT_NAMES
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState


class OliRobotIO(RobotIO):
    def __init__(self, robot_cfg: RobotCfg, *, robot_ip: str = '127.0.0.1',
                 publish_hz: float = 1000.0, timeout_s: float = 0.25):
        if robot_cfg.name != 'oli' or os.environ.get('ROBOT_TYPE') != 'HU_D04_01':
            raise ValueError('OLI backend requires robot=oli and ROBOT_TYPE=HU_D04_01')
        if publish_hz <= 0 or timeout_s <= 0:
            raise ValueError('publish_hz and timeout_s must be positive')
        try:
            import limxsdk.datatypes as datatypes
            import limxsdk.robot.Robot as Robot
            import limxsdk.robot.RobotType as RobotType
        except ImportError as exc:
            raise ImportError('Install the official limxsdk-lowlevel Python wheel for this architecture') from exc
        self._command_type = datatypes.RobotCmd
        self._robot = Robot(RobotType.Humanoid)
        if not self._robot.init(robot_ip):
            raise RuntimeError(f'LimX SDK initialization failed at {robot_ip}')
        if self._robot.getMotorNumber() != len(OLI_SDK_JOINT_NAMES):
            raise ValueError('HU_D04_01 requires 31 SDK motors')
        self.cfg = robot_cfg
        self._count = len(robot_cfg.joint_names)
        self._to_sdk = np.array([robot_cfg.joint_names.index(n) for n in OLI_SDK_JOINT_NAMES])
        self._from_sdk = np.argsort(self._to_sdk)
        self._lock = threading.Lock()
        self._state = self._imu = self._command = None
        self._state_time = self._imu_time = self._command_time = 0.0
        self._error = None
        self._control_mode = 'zero'
        self._timeout = timeout_s
        self._period = 1.0 / publish_hz
        self._stop = threading.Event()
        self._robot.subscribeRobotState(self._on_state)
        self._robot.subscribeImuData(self._on_imu)
        self._thread = threading.Thread(target=self._publish_loop, name='oli-command', daemon=True)
        self._thread.start()

    def _vector(self, value, count):
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (count,) or not np.isfinite(array).all():
            raise ValueError(f'Expected {count} finite values, got {array.shape}')
        return array.copy()

    def _on_state(self, state):
        try:
            values = tuple(self._vector(getattr(state, k), self._count) for k in ('q', 'dq', 'tau'))
            names = list(state.motor_names)
            # SDK names may be empty or motor labels, rather than semantic joint names.
            # Nameless/motor-label packets use the documented HU_D04_01 array order.
            order = list(OLI_SDK_JOINT_NAMES)
            if names and set(names) == set(order) and len(names) == self._count:
                ids = [names.index(n) for n in order]
                values = tuple(v[ids] for v in values)
                names = order
            elif names and len(names) != self._count:
                raise ValueError('OLI motor_names length differs from 31')
            with self._lock:
                self._state = (*values, names, int(state.stamp))
                self._state_time = time.monotonic()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def _on_imu(self, imu):
        try:
            quat = self._vector(imu.quat, 4)  # SDK wxyz, pelvis frame; no xyzw permutation.
            norm = np.linalg.norm(quat)
            if norm < 1e-6:
                raise ValueError('Invalid OLI IMU quaternion')
            gyro = self._vector(imu.gyro, 3)
            with self._lock:
                self._imu = (quat / norm, gyro)
                self._imu_time = time.monotonic()
        except Exception as exc:
            with self._lock:
                self._error = exc

    def read_state(self) -> RobotState | None:
        with self._lock:
            if self._error:
                raise RuntimeError('OLI transport failed') from self._error
            if self._state is None or self._imu is None:
                return None
            if time.monotonic() - min(self._state_time, self._imu_time) > self._timeout:
                return None
            q, dq, tau, _, stamp = self._state
            quat, gyro = self._imu
            qpos = np.zeros(7 + self._count, dtype=np.float32)
            qvel = np.zeros(6 + self._count, dtype=np.float32)
            # No world translation from SDK. Proteus reference_forcing supplies the root position.
            qpos[3:7], qpos[7:] = quat, q[self._from_sdk]
            qvel[3:6], qvel[6:] = gyro, dq[self._from_sdk]
            return RobotState(qpos=qpos, qvel=qvel, joint_torque=tau[self._from_sdk].copy(),
                              tick=stamp // 1_000_000)

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        values = tuple(self._vector(v, self._count)[self._to_sdk] for v in (q_target, dq_target, tau_ff, kp, kd))
        if np.any(values[3] < 0) or np.any(values[4] < 0):
            raise ValueError('OLI gains must be non-negative')
        with self._lock:
            if self._error:
                raise RuntimeError('OLI transport failed') from self._error
            self._command = values
            self._command_time = time.monotonic()

    def set_control_mode(self, mode: str, generation: int) -> None:
        if mode not in {'zero', 'init', 'policy'}:
            raise ValueError(f'Unknown OLI control mode: {mode}')
        with self._lock:
            self._control_mode = mode

    def _publish(self, *, force_damping=False):
        with self._lock:
            if self._state is None or self._command is None:
                return
            state, values = self._state, self._command
            zero_mode = self._control_mode == 'zero'
            expired = time.monotonic() - min(self._command_time, self._state_time, self._imu_time) > self._timeout
        q, dq, tau, kp, kd = values
        if force_damping or zero_mode or expired or self._error:
            q = state[0]
            dq = tau = kp = np.zeros(self._count)
        cmd = self._command_type()
        cmd.stamp = time.time_ns()
        cmd.mode = [0] * self._count  # hybrid position / torque
        cmd.parallel_solve_required = [True] * self._count  # SDK converts PR to AB
        cmd.motor_names = state[3]
        cmd.q, cmd.dq, cmd.tau = q.tolist(), dq.tolist(), tau.tolist()
        cmd.Kp, cmd.Kd = kp.tolist(), kd.tolist()
        if self._robot.publishRobotCmd(cmd) is False:
            raise RuntimeError('LimX SDK rejected RobotCmd')

    def _publish_loop(self):
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                self._publish()
            except Exception as exc:
                with self._lock:
                    self._error = exc
            deadline += self._period
            now = time.monotonic()
            if deadline < now:
                deadline = now
            self._stop.wait(deadline - now)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._publish(force_damping=True)
