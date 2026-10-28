"""Observations of the multi-robot Proteus actor (Proteus ScaleTrack/deploy/proteus/observations.py).

Every term is a view of one core per runtime, configured by the policy YAML's ``proteus`` section
(written by scripts/export_proteus_onnx.py from the robot's contract). The ONNX groups are:

    proprioception  (H, 6 + 2 S)   projected gravity, root angular velocity (root frame), slot joint positions
                                   relative to the default pose, slot joint velocities * joint_vel_scale
    task            (F, 18 B + 1)  per reference frame: body positions relative to the anchor and to each robot
                                   body, rotations relative to the anchor and to each robot body (tangent/normal),
                                   the frame offset
    action_history  (H, S)         the actor's last H slot actions (unmasked, as IsaacLab's last_action)

H history steps oldest first, S canonical slots, B tracked bodies (anchor first), F reference frames. Positions are
divided by the robot's length scale. After a reset the proprioception history holds copies of the current state and
the action history zeros, as IsaacLab's history buffers.

The tracked robot bodies come from MuJoCo forward kinematics at the measured root orientation and joint angles. The
root position is the reference anchor's (``reference_forcing``, local tracking: the real robot has no root position)
or the measured one. The Motion is aligned once, at reset, to the robot's heading (and position when not forcing).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.utils.math import (
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse_numpy,
    quat_rotate_numpy,
)

_GRAVITY = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)


def _unit_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-6):
        q = np.where(norm < 1.0e-6, np.asarray([1.0, 0.0, 0.0, 0.0]), q)
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / norm


def _tan_norm(q: np.ndarray) -> np.ndarray:
    """Rotated x axis and rotated z axis (..., 6)."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    tan = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=-1)
    norm = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    return np.concatenate([tan, norm], axis=-1)


def _apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate_inverse_numpy(np.broadcast_to(q, v.shape[:-1] + (4,)), v)


def _mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    shape = np.broadcast_shapes(q1.shape, q2.shape)
    return quat_mul(np.broadcast_to(q1, shape).copy(), np.broadcast_to(q2, shape).copy())


class _ProteusCore:
    def __init__(self, env):
        cfg = dict(env.policy_config["proteus"])
        self.env = env
        self.num_slots = int(cfg["num_slots"])
        self.history = int(cfg["history"])
        self.joint_vel_scale = float(cfg["joint_vel_scale"])
        self.length_scale = float(cfg["length_scale"])
        self.body_names = [str(name) for name in cfg["body_names"]]
        self.future_offsets = np.asarray(cfg["future_offsets"], dtype=np.int64)
        self.reference_forcing = bool(cfg.get("reference_forcing", True))
        if self.reference_forcing and int(self.future_offsets[0]) != 0:
            raise ValueError("Proteus reference_forcing places the root at the first reference frame: offset 0")

        slots = {str(name): int(slot) for name, slot in cfg["joint_slots"].items()}
        signs = {str(name): float(sign) for name, sign in cfg.get("joint_signs", {}).items()}
        defaults = {str(name): float(value) for name, value in env.policy_config["default_joint_pos"].items()}
        self.joint_names = list(slots)  # the contract's joints: driven ones in their slot, locked ones (-1) only FK
        state_joints = list(env.state_processor.joint_names)
        missing = [name for name in self.joint_names if name not in state_joints or name not in defaults]
        if missing:
            raise ValueError(f"Proteus joints missing from the robot state or default_joint_pos: {missing}")
        slotted = [name for name in self.joint_names if slots[name] >= 0]
        self.state_index = np.asarray([state_joints.index(name) for name in slotted], dtype=np.int64)
        self.slot = np.asarray([slots[name] for name in slotted], dtype=np.int64)
        self.sign = np.asarray([signs.get(name, 1.0) for name in slotted], dtype=np.float32)
        self.default = np.asarray([defaults[name] for name in slotted], dtype=np.float32)
        if len(set(self.slot.tolist())) != len(self.slot) or self.slot.max(initial=-1) >= self.num_slots:
            raise ValueError(f"Proteus joint slots must be distinct and below {self.num_slots}: {slots}")

        self._init_kinematics(cfg.get("mjcf_path"), [state_joints.index(name) for name in self.joint_names])

        self.prop_dim = 6 + 2 * self.num_slots
        self.prop_hist = np.zeros((self.history, self.prop_dim), dtype=np.float32)
        self.action_hist = np.zeros((self.history, self.num_slots), dtype=np.float32)
        F, B = len(self.future_offsets), len(self.body_names)
        self.target_body_pos = np.zeros((F, 3 * B), dtype=np.float32)
        self.target_body_pos_rel = np.zeros((F, 3 * B), dtype=np.float32)
        self.target_body_rot = np.zeros((F, 6 * B), dtype=np.float32)
        self.target_body_rot_rel = np.zeros((F, 6 * B), dtype=np.float32)
        self.frame_offsets = self.future_offsets.astype(np.float32)[:, None]

        self._align_quat = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._align_src = np.zeros(3)  # Motion xy at reset
        self._align_dst = np.zeros(3)  # robot xy at reset (global tracking)
        self._fill_on_update = True
        self._motion_layout: tuple[tuple[str, ...], tuple[int, ...]] | None = None

    def _init_kinematics(self, mjcf_path: str | None, fk_state_index: list[int]) -> None:
        import mujoco

        if mjcf_path is None:
            mjcf_path = self.env.robot_cfg.resolve_mjcf_path()
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        free = [i for i in range(self.model.njnt) if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1:
            raise ValueError(f"Proteus FK needs exactly one free joint in {mjcf_path}")
        self.root_qpos = int(self.model.jnt_qposadr[free[0]])
        if self.model.body(self.body_names[0]).id != int(self.model.jnt_bodyid[free[0]]):
            raise ValueError(f"Proteus anchor body {self.body_names[0]!r} must be the MJCF root body")
        self.fk_qpos_adr = np.asarray([self.model.joint(name).qposadr[0] for name in self.joint_names])
        self.fk_state_index = np.asarray(fk_state_index, dtype=np.int64)
        self.body_ids = np.asarray([self.model.body(name).id for name in self.body_names])

    # ------------------------------------------------------------------ robot state

    def _robot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = self.env.state_processor
        root_quat = _unit_quat(np.asarray(state.root_quat_w).reshape(4))
        joint_pos = np.asarray(state.joint_pos, dtype=np.float64).reshape(-1)
        joint_vel = np.asarray(state.joint_vel, dtype=np.float64).reshape(-1)
        return root_quat, joint_pos, joint_vel

    def _proprioception(self, root_quat: np.ndarray, joint_pos: np.ndarray, joint_vel: np.ndarray) -> np.ndarray:
        state = self.env.state_processor
        pos = np.zeros(self.num_slots, dtype=np.float32)
        vel = np.zeros(self.num_slots, dtype=np.float32)
        pos[self.slot] = self.sign * (joint_pos[self.state_index] - self.default)
        vel[self.slot] = self.sign * joint_vel[self.state_index] * self.joint_vel_scale
        gravity = quat_rotate_inverse_numpy(root_quat[None], _GRAVITY[None].astype(np.float64))[0]
        ang_vel = np.asarray(state.root_ang_vel_b, dtype=np.float32).reshape(3)
        return np.concatenate([gravity, ang_vel, pos, vel]).astype(np.float32)

    def _bodies(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray):
        """World positions (B, 3) and orientations (B, 4) wxyz of the tracked bodies."""
        d = self.data
        d.qpos[:] = self.model.qpos0
        d.qpos[self.root_qpos : self.root_qpos + 3] = root_pos
        d.qpos[self.root_qpos + 3 : self.root_qpos + 7] = root_quat
        d.qpos[self.fk_qpos_adr] = joint_pos[self.fk_state_index]
        self._mujoco.mj_kinematics(self.model, d)
        return d.xpos[self.body_ids].copy(), d.xquat[self.body_ids].copy()

    # ------------------------------------------------------------------ reference

    def _reference(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Aligned reference bodies (F, B, 3/4) at the future offsets, or None without a Motion."""
        motion_data = getattr(self.env, "motion_data", None)
        if motion_data is None:
            return None
        body_names = tuple(getattr(self.env, "motion_body_names", ()))
        steps = tuple(int(step) for step in np.asarray(getattr(self.env, "motion_future_steps", [])).tolist())
        if self._motion_layout != (body_names, steps):
            missing_bodies = [name for name in self.body_names if name not in body_names]
            missing_steps = [int(o) for o in self.future_offsets if int(o) not in steps]
            if missing_bodies or missing_steps:
                raise ValueError(
                    f"Proteus needs Motion bodies {missing_bodies} and motion.future_steps {missing_steps} "
                    f"(available: {list(body_names)}, {list(steps)})"
                )
            self._body_index = [body_names.index(name) for name in self.body_names]
            self._step_index = [steps.index(int(o)) for o in self.future_offsets]
            self._motion_layout = (body_names, steps)
        pos = np.asarray(motion_data.body_pos_w[0], dtype=np.float64)[self._step_index][:, self._body_index]
        quat = _unit_quat(np.asarray(motion_data.body_quat_w[0], dtype=np.float64)[self._step_index][:, self._body_index])
        align = np.broadcast_to(self._align_quat, quat.shape)
        pos = quat_rotate_numpy(align, pos - self._align_src) + self._align_dst
        return pos, _mul(self._align_quat, quat)

    def _reset_alignment(self, root_quat: np.ndarray) -> None:
        self._align_quat = np.asarray([1.0, 0.0, 0.0, 0.0])
        self._align_src[:] = 0.0
        self._align_dst[:] = 0.0
        reference = self._reference()
        if reference is None:
            return
        ref_pos, ref_quat = reference
        robot_heading = projected_yaw_quat(root_quat[None])[0]
        ref_heading = projected_yaw_quat(ref_quat[0, 0][None])[0]
        self._align_quat = quat_mul(robot_heading[None], quat_conjugate(ref_heading[None]))[0]
        self._align_src[:2] = ref_pos[0, 0, :2]
        if not self.reference_forcing:
            self._align_dst[:2] = np.asarray(self.env.state_processor.root_pos_w, dtype=np.float64).reshape(3)[:2]

    def _update_task(self, root_quat: np.ndarray, joint_pos: np.ndarray) -> None:
        reference = self._reference()
        if self.reference_forcing and reference is not None:
            root_pos = reference[0][0, 0]
        else:
            root_pos = np.asarray(self.env.state_processor.root_pos_w, dtype=np.float64).reshape(3)
        body_pos, body_quat = self._bodies(root_pos, root_quat, joint_pos)
        if reference is None:  # no Motion yet: hold the current pose
            F = len(self.future_offsets)
            reference = (np.broadcast_to(body_pos, (F,) + body_pos.shape), np.broadcast_to(body_quat, (F,) + body_quat.shape))
        ref_pos, ref_quat = reference
        F = ref_pos.shape[0]
        r = self.length_scale
        a_pos, a_quat = body_pos[0], body_quat[0]
        a_inv = quat_conjugate(a_quat[None])[0]
        self.target_body_pos[:] = (_apply_inverse(a_quat, ref_pos - a_pos) / r).reshape(F, -1)
        self.target_body_pos_rel[:] = (_apply_inverse(a_quat, ref_pos - body_pos[None]) / r).reshape(F, -1)
        self.target_body_rot[:] = _tan_norm(_mul(a_inv, ref_quat)).reshape(F, -1)
        relative = _mul(_mul(a_inv, _mul(ref_quat, quat_conjugate(body_quat)[None])), a_quat)
        self.target_body_rot_rel[:] = _tan_norm(relative).reshape(F, -1)

    # ------------------------------------------------------------------ lifecycle

    def reset(self) -> None:
        root_quat, joint_pos, joint_vel = self._robot()
        self._reset_alignment(root_quat)
        self.prop_hist[:] = self._proprioception(root_quat, joint_pos, joint_vel)
        self.action_hist.fill(0.0)
        self._update_task(root_quat, joint_pos)
        # The runtimes read a new state after the reset; the first update restarts the history from it.
        self._fill_on_update = True

    def update(self, data: Dict[str, Any]) -> None:
        root_quat, joint_pos, joint_vel = self._robot()
        prop = self._proprioception(root_quat, joint_pos, joint_vel)
        if self._fill_on_update:
            self.prop_hist[:] = prop
            self.action_hist.fill(0.0)
            self._fill_on_update = False
        else:
            self.prop_hist = np.roll(self.prop_hist, -1, axis=0)
            self.prop_hist[-1] = prop
            self.action_hist = np.roll(self.action_hist, -1, axis=0)
            self.action_hist[-1] = np.asarray(data.get("slot_action", 0.0), dtype=np.float32)
        self._update_task(root_quat, joint_pos)


class _ProteusObservation(Observation):
    """A view of the runtime's Proteus core; the first term created owns its reset and update."""

    namespace = "proteus"
    output_name: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        core = getattr(self.env, "_proteus_core", None)
        self.owner = core is None
        if core is None:
            core = _ProteusCore(self.env)
            setattr(self.env, "_proteus_core", core)
        self.core = core

    def reset(self) -> None:
        if self.owner:
            self.core.reset()

    def update(self, data: Dict[str, Any]) -> None:
        if self.owner:
            self.core.update(data)

    def compute(self) -> np.ndarray:
        return getattr(self.core, self.output_name)


class proteus_projected_gravity_history(_ProteusObservation):
    def compute(self) -> np.ndarray:
        return self.core.prop_hist[:, 0:3]


class proteus_root_ang_vel_history(_ProteusObservation):
    def compute(self) -> np.ndarray:
        return self.core.prop_hist[:, 3:6]


class proteus_slot_joint_pos_history(_ProteusObservation):
    def compute(self) -> np.ndarray:
        return self.core.prop_hist[:, 6 : 6 + self.core.num_slots]


class proteus_slot_joint_vel_history(_ProteusObservation):
    def compute(self) -> np.ndarray:
        return self.core.prop_hist[:, 6 + self.core.num_slots :]


class proteus_target_body_pos(_ProteusObservation):
    output_name = "target_body_pos"


class proteus_target_body_pos_rel(_ProteusObservation):
    output_name = "target_body_pos_rel"


class proteus_target_body_rot(_ProteusObservation):
    output_name = "target_body_rot"


class proteus_target_body_rot_rel(_ProteusObservation):
    output_name = "target_body_rot_rel"


class proteus_frame_offsets(_ProteusObservation):
    output_name = "frame_offsets"


class proteus_slot_action_history(_ProteusObservation):
    output_name = "action_hist"
