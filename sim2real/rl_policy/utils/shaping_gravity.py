"""Bounded model gravity feedforward for supported Shaping, never policy control."""
import numpy as np


class ShapingGravity:
    def __init__(self, cfg, *, max_effort_fraction=.1, ramp_s=2.):
        import mujoco
        if not np.isfinite(max_effort_fraction) or not 0 < max_effort_fraction <= 1:
            raise ValueError('Shaping gravity max_effort_fraction must be in (0, 1]')
        if not np.isfinite(ramp_s) or ramp_s <= 0:
            raise ValueError('Shaping gravity ramp_s must be finite and positive')
        self.mujoco, self.ramp_s = mujoco, float(ramp_s)
        self.model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or self.model.jnt_qposadr[0] != 0:
            raise ValueError('Shaping gravity requires a floating pelvis at qpos[0:7]')
        self.data = mujoco.MjData(self.model)
        self.qpos_indices = [int(self.model.joint(n).qposadr[0]) for n in cfg.joint_names]
        self.dof_indices = [int(self.model.joint(n).dofadr[0]) for n in cfg.joint_names]
        self.effort_cap = np.array([cfg.joint_effort_limit[n] for n in cfg.joint_names]) * max_effort_fraction

    def compute(self, joint_pos, root_quat_w, elapsed_s):
        q = np.asarray(joint_pos, dtype=float)
        quat = np.asarray(root_quat_w, dtype=float)
        if (q.shape != (len(self.qpos_indices),) or quat.shape != (4,)
                or not np.isfinite(q).all() or not np.isfinite(quat).all()
                or np.linalg.norm(quat) < 1e-6 or not np.isfinite(elapsed_s)):
            raise ValueError('Invalid measured pose for Shaping gravity')
        self.data.qpos[:3] = [0., 0., 1.]
        self.data.qpos[3:7] = quat / np.linalg.norm(quat)
        self.data.qpos[self.qpos_indices] = q
        self.data.qvel[:] = 0.
        # With zero velocity, qfrc_bias contains gravity only. Do not include
        # contact forces: this compensator is intended for gantry-supported entry.
        self.mujoco.mj_forward(self.model, self.data)
        tau = self.data.qfrc_bias[self.dof_indices].copy()
        if not np.isfinite(tau).all():
            raise ValueError('Non-finite Shaping gravity torque')
        return np.clip(tau, -self.effort_cap, self.effort_cap) * np.clip(elapsed_s / self.ramp_s, 0., 1.)
