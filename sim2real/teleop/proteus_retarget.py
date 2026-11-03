"""Bounded two-stage GMR for Proteus bodies, with stage-specific target offsets."""
import mink
import mujoco
import numpy as np
from general_motion_retargeting import GeneralMotionRetargeting


class ProteusRetargeter(GeneralMotionRetargeting):
    reference_dt = 1 / 30
    max_joint_speed = 12.

    def setup_retarget_configuration(self):
        super().setup_retarget_configuration()
        self.hinge_qpos = [int(self.model.joint(i).qposadr[0]) for i in range(self.model.njnt)
                           if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        cost = np.full(self.model.nv, .05)
        cost[:6] = 0
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
            tasks, error_fn = getattr(self, f'tasks{stage}'), getattr(self, f'error{stage}')
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
        ids = self.hinge_qpos
        bound = self.max_joint_speed * self.reference_dt
        q[ids] = previous[ids] + np.clip(q[ids] - previous[ids], -bound, bound)
        if not np.isfinite(q).all():
            raise ValueError('Proteus IK produced non-finite qpos')
        self.configuration.update(q)
        return q
