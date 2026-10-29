from __future__ import annotations

import os
import time

import numpy as np
import zmq
from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.utils.common import LowCmdMessage, LowStateMessage


class ZMQRobotIO(RobotIO):
    """Robot I/O transported over the existing LowState/LowCmd ZMQ ABI."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        context: zmq.Context | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self._joint_count = len(robot_cfg.joint_names)
        self._command_sequence = 0
        self._context = context or zmq.Context.instance()
        self._latest_state: RobotState | None = None
        self._state_updated_ns = 0
        self._control_mode = "zero"
        self._control_generation = 0
        if timeout_s is None:
            timeout_s = float(os.environ.get("SIM2REAL_IO_TIMEOUT_S", "0.1"))
        if not np.isfinite(timeout_s) or not 0 < timeout_s <= 1:
            raise ValueError("timeout_s must be finite and in (0, 1]")
        self._state_timeout_ns = int(timeout_s * 1e9)

        state_endpoint = f"tcp://{robot_cfg.low_state_host}:{robot_cfg.low_state_port}"
        self._state_socket = self._context.socket(zmq.SUB)
        self._state_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._state_socket.setsockopt(zmq.CONFLATE, 1)
        self._state_socket.setsockopt(zmq.RCVTIMEO, 10)
        self._state_socket.connect(state_endpoint)

        command_endpoint = (
            f"tcp://{robot_cfg.low_cmd_bind_addr}:{robot_cfg.low_cmd_port}"
        )
        self._command_socket = self._context.socket(zmq.PUB)
        self._command_socket.setsockopt(zmq.SNDHWM, 1)
        self._command_socket.setsockopt(zmq.LINGER, 0)
        self._command_socket.bind(command_endpoint)

    def read_state(self) -> RobotState | None:
        while True:
            try:
                data = self._state_socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                message = LowStateMessage.from_bytes(data)
                state = self._message_to_state(message)
                if self._latest_state is None or state.tick != self._latest_state.tick:
                    self._state_updated_ns = time.monotonic_ns()
                self._latest_state = state
            except Exception as exc:
                logger.warning("Failed to decode low state message: {}", exc)

        if time.monotonic_ns() - self._state_updated_ns > self._state_timeout_ns:
            return None
        return self._latest_state

    def set_control_mode(self, mode: str, generation: int) -> None:
        self._control_mode = mode
        self._control_generation = generation

    def write_command(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        self._command_sequence += 1
        message = LowCmdMessage(
            q_target,
            dq_target,
            tau_ff,
            kp,
            kd,
            source_time_ns=time.monotonic_ns(),
            sequence=self._command_sequence,
            control_mode=self._control_mode,
            control_generation=self._control_generation,
        )
        try:
            self._command_socket.send(message.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass

    def close(self) -> None:
        self._state_socket.close(linger=0)
        self._command_socket.close(linger=0)

    def _message_to_state(self, message: LowStateMessage) -> RobotState:
        if message.joint_positions.size != self._joint_count:
            raise ValueError(
                "Low state joint count does not match robot config: "
                f"{message.joint_positions.size} != {self._joint_count}"
            )

        qpos = np.zeros(7 + self._joint_count, dtype=np.float32)
        qvel = np.zeros(6 + self._joint_count, dtype=np.float32)
        qpos[3:7] = message.quaternion
        qpos[7:] = message.joint_positions
        qvel[3:6] = message.gyroscope
        qvel[6:] = message.joint_velocities
        if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
            raise ValueError("Low state contains non-finite values")
        if not 0.9 < np.linalg.norm(message.quaternion) < 1.1:
            raise ValueError("Low state quaternion is not normalized")
        torque = message.joint_torques
        if torque is None:
            torque = np.zeros(self._joint_count, dtype=np.float32)

        return RobotState(
            qpos=qpos,
            qvel=qvel,
            joint_torque=np.asarray(torque, dtype=np.float32).copy(),
            tick=int(message.tick),
            stop_epoch=message.stop_epoch,
        )
