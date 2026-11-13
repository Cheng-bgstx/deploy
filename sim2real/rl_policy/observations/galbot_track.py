"""Observations of Galbot's mjlab single-motion tracking actor (et1-dep policy protocol 3.2).

The actor (e.g. ``shanshenbu``) takes one flat 111-dim ``obs``, in the order of its ``inference.toml`` layout:

    command_root_orientation_6d_body  6   reference root orientation in the robot root frame, rotation-matrix
                                          columns 0 and 1 flattened row by row ([r00, r01, r10, r11, r20, r21])
    reference_root_linear_velocity_xy 2   reference root linear velocity, x and y in the reference root frame
    reference_root_angular_velocity_z 1   reference root angular velocity, z in the reference root frame
    reference_joint_positions         A   absolute reference joint positions of the actuated joints
    projected_gravity                 3   gravity in the robot root frame
    base_ang_vel                      3   IMU angular velocity
    joint_pos                         A   joint positions relative to the policy default pose
    joint_vel                         A   joint velocities
    actions                           A   the previous raw action (zero after a reset)

The conventions were checked against the checkpoint's observation normalizer: its mean of the 6D term is the
identity in this flattening, its mean of the reference joints matches the Motion's absolute joint positions, and its
mean of ``joint_pos`` matches them minus the default pose. ``velocity_frame: heading`` uses the reference heading
(yaw only) instead of the full reference root frame for the two velocity terms.

The Motion is aligned once, at reset, to the robot's heading, so the 6D term starts near the identity whatever
the IMU yaw. Every term is a view of one core per runtime configured by the policy YAML's ``galbot_track`` section.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.utils.math import projected_yaw_quat, quat_conjugate, quat_mul, quat_rotate_inverse_numpy

_GRAVITY = np.asarray([[0.0, 0.0, -1.0]])


def _unit(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    return np.asarray([1.0, 0.0, 0.0, 0.0]) if n < 1e-6 else q / n


def rot6d(q: np.ndarray) -> np.ndarray:
    """Columns 0 and 1 of the rotation matrix of wxyz ``q``, flattened row by row."""
    w, x, y, z = _unit(q)
    return np.asarray([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
        2 * (x * z - w * y), 2 * (y * z + w * x),
    ])


def _inverse_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(np.asarray(q, dtype=np.float64)[None], np.asarray(v, dtype=np.float64)[None])[0]


class _GalbotTrackCore:
    def __init__(self, env):
        cfg = dict(env.policy_config.get("galbot_track") or {})
        self.env = env
        self.root_body = str(cfg.get("root_body", "pelvis_link"))
        self.velocity_frame = str(cfg.get("velocity_frame", "body"))
        if self.velocity_frame not in {"body", "heading"}:
            raise ValueError(f"galbot_track.velocity_frame must be 'body' or 'heading', got {self.velocity_frame!r}")
        self.joint_names = [str(n) for n in env.policy_joint_names]
        state_names = list(env.state_processor.joint_names)
        self.state_index = np.asarray([state_names.index(n) for n in self.joint_names])
        sim_names = list(env.joint_names_simulation)
        self.default = np.asarray(env.default_dof_angles, dtype=np.float64)[[sim_names.index(n) for n in self.joint_names]]
        self.align = np.asarray([1.0, 0.0, 0.0, 0.0])
        self.prev_action = np.zeros(len(self.joint_names))
        self._fresh = True
        self._layout = None
        self.outputs: dict[str, np.ndarray] = {}

    def _reference(self):
        """Reference root quaternion (aligned), root velocities (world) and joint positions (policy order)."""
        motion = self.env.motion_data
        layout = (tuple(self.env.motion_body_names), tuple(self.env.motion_joint_names))
        if layout != self._layout:
            body_names, joint_names = layout
            missing = [n for n in self.joint_names if n not in joint_names]
            if self.root_body not in body_names or missing:
                raise ValueError(f"galbot_track needs Motion body {self.root_body!r} and joints {missing}")
            self._root = body_names.index(self.root_body)
            self._joints = [joint_names.index(n) for n in self.joint_names]
            self._layout = layout
        quat = _unit(motion.body_quat_w[0, 0, self._root])
        lin_vel = np.asarray(motion.body_lin_vel_w[0, 0, self._root], dtype=np.float64)
        ang_vel = np.asarray(motion.body_ang_vel_w[0, 0, self._root], dtype=np.float64)
        joint_pos = np.asarray(motion.joint_pos[0, 0], dtype=np.float64)[self._joints]
        return quat, lin_vel, ang_vel, joint_pos

    def reset(self) -> None:
        robot_quat = _unit(self.env.state_processor.root_quat_w)
        ref_quat = self._reference()[0]
        robot_heading = projected_yaw_quat(robot_quat[None])[0]
        ref_heading = projected_yaw_quat(ref_quat[None])[0]
        self.align = quat_mul(robot_heading[None], quat_conjugate(ref_heading[None]))[0]
        self.prev_action[:] = 0.0
        self._fresh = True
        self._compute()

    def update(self, data: Dict[str, Any]) -> None:
        # The action in ``data`` before the first post-reset update comes from Shaping, not from this episode.
        self.prev_action = np.zeros_like(self.prev_action) if self._fresh else np.asarray(data["action"], dtype=np.float64)
        self._fresh = False
        self._compute()

    def _compute(self) -> None:
        state = self.env.state_processor
        robot_quat = _unit(state.root_quat_w)
        ref_quat, lin_vel_w, ang_vel_w, ref_joint_pos = self._reference()
        ref_quat = quat_mul(self.align[None], ref_quat[None])[0]
        frame = ref_quat if self.velocity_frame == "body" else projected_yaw_quat(ref_quat[None])[0]
        # The velocities are world-frame in the Motion; rotating them and the frame by ``align`` cancels out.
        unaligned = quat_mul(quat_conjugate(self.align[None]), frame[None])[0]
        lin_vel = _inverse_rotate(unaligned, lin_vel_w)
        ang_vel = _inverse_rotate(unaligned, ang_vel_w)
        joint_pos = np.asarray(state.joint_pos, dtype=np.float64)[self.state_index]
        joint_vel = np.asarray(state.joint_vel, dtype=np.float64)[self.state_index]
        relative = quat_mul(quat_conjugate(robot_quat[None]), ref_quat[None])[0]
        self.outputs = {
            "root_ori_6d_b": rot6d(relative),
            "ref_root_lin_vel_xy": lin_vel[:2],
            "ref_root_ang_vel_z": ang_vel[2:3],
            "ref_joint_pos": ref_joint_pos,
            "projected_gravity": _inverse_rotate(robot_quat, _GRAVITY[0]),
            "base_ang_vel": np.asarray(state.root_ang_vel_b, dtype=np.float64).reshape(3),
            "joint_pos_rel": joint_pos - self.default,
            "joint_vel": joint_vel,
            "prev_action": self.prev_action.copy(),
        }


class _GalbotTrackObservation(Observation):
    """A view of the runtime's core; the first term created owns its reset and update."""

    namespace = "galbot_track"
    output_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        core = getattr(self.env, "_galbot_track_core", None)
        self.owner = core is None
        if core is None:
            core = _GalbotTrackCore(self.env)
            setattr(self.env, "_galbot_track_core", core)
        self.core = core

    def reset(self) -> None:
        if self.owner:
            self.core.reset()

    def update(self, data: Dict[str, Any]) -> None:
        if self.owner:
            self.core.update(data)

    def compute(self) -> np.ndarray:
        if not self.core.outputs:
            self.core._compute()
        # The exported actor takes a fixed (1, 111) input: keep the batch axis.
        return self.core.outputs[self.output_name].astype(np.float32)[None]


class galbot_root_ori_6d_b(_GalbotTrackObservation):
    output_name = "root_ori_6d_b"


class galbot_ref_root_lin_vel_xy(_GalbotTrackObservation):
    output_name = "ref_root_lin_vel_xy"


class galbot_ref_root_ang_vel_z(_GalbotTrackObservation):
    output_name = "ref_root_ang_vel_z"


class galbot_ref_joint_pos(_GalbotTrackObservation):
    output_name = "ref_joint_pos"


class galbot_projected_gravity(_GalbotTrackObservation):
    output_name = "projected_gravity"


class galbot_base_ang_vel(_GalbotTrackObservation):
    output_name = "base_ang_vel"


class galbot_joint_pos_rel(_GalbotTrackObservation):
    output_name = "joint_pos_rel"


class galbot_joint_vel(_GalbotTrackObservation):
    output_name = "joint_vel"


class galbot_prev_action(_GalbotTrackObservation):
    output_name = "prev_action"
