"""X2 GMR task solve, following TaiE64/AgibotX2-Retargeter (Apache-2.0).

Keep the shoulder Euler branch and unconstrained head near neutral; use each IK
stage's own position/rotation offsets and pass Mink limits by keyword. This is
the same solver for PICO and offline LAFAN, on the prepared training-contract MJCF.
"""
import mink
import mujoco
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting


class X2Retargeter(GeneralMotionRetargeting):
    reference_dt = 1 / 30
    max_joint_speed = 12.0

    def setup_retarget_configuration(self):
        super().setup_retarget_configuration()
        cost = np.full(self.model.nv, .05)
        cost[:6] = 0
        self.hinge_qpos = []
        for joint_id in range(self.model.njnt):
            joint = self.model.joint(joint_id)
            if joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            self.hinge_qpos.append(int(joint.qposadr[0]))
            dof = int(joint.dofadr[0])
            if 'wrist' in joint.name or 'head' in joint.name:
                cost[dof] = .5
            if 'shoulder_' in joint.name:
                cost[dof] = 5
            if joint.name == 'waist_pitch_joint':
                cost[dof] = 10
        posture = mink.PostureTask(self.model, cost=cost)
        posture.set_target(self.model.qpos0.copy())
        self.tasks1.append(posture)
        self.tasks2.append(posture)

    def update_targets(self, human_data, offset_to_ground=False):
        scaled = self.scale_human_data(self.to_numpy(human_data), self.human_root_name, self.human_scale_table)
        for stage in (1, 2):
            human = self.offset_human_data(scaled, getattr(self, f'pos_offsets{stage}'),
                                          getattr(self, f'rot_offsets{stage}'))
            human = self.apply_ground_offset(human)
            if offset_to_ground:
                human = self.offset_human_data_to_ground(human)
            if stage == 1:
                self.scaled_human_data = human
            if getattr(self, f'use_ik_match_table{stage}'):
                for name, task in getattr(self, f'human_body_to_task{stage}').items():
                    pos, quat = human[name]
                    task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(quat), pos))

    def retarget(self, human_data, offset_to_ground=False):
        previous = self.configuration.data.qpos.copy()
        self.update_targets(human_data, offset_to_ground)
        dt = self.model.opt.timestep
        for stage in (1, 2):
            if not getattr(self, f'use_ik_match_table{stage}'):
                continue
            tasks = getattr(self, f'tasks{stage}')
            error_fn = getattr(self, f'error{stage}')
            error = error_fn()
            for _ in range(self.max_iter + 1):
                velocity = mink.solve_ik(self.configuration, tasks, dt, self.solver,
                                         damping=self.damping, limits=self.ik_limits)
                self.configuration.integrate_inplace(velocity, dt)
                next_error = error_fn()
                if error - next_error <= .001:
                    break
                error = next_error
        q = self.configuration.data.qpos.copy()
        indices = self.hinge_qpos
        bound = self.max_joint_speed * self.reference_dt
        q[indices] = previous[indices] + np.clip(q[indices] - previous[indices], -bound, bound)
        if not np.isfinite(q).all():
            raise ValueError('X2 IK produced non-finite qpos')
        self.configuration.update(q)
        return q
