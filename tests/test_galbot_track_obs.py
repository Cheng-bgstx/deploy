"""Galbot single-motion tracking observations (et1-dep protocol 3.2 actor)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sim2real.rl_policy.observations import Observation
from sim2real.rl_policy.observations.galbot_track import rot6d

ORDER = ["root_ori_6d_b", "ref_root_lin_vel_xy", "ref_root_ang_vel_z", "ref_joint_pos", "projected_gravity",
         "base_ang_vel", "joint_pos_rel", "joint_vel", "prev_action"]


def yaw(angle):
    return np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)])


def make_env(robot_quat, ref_quat, *, lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)):
    names = ["a", "b", "head"]
    motion = SimpleNamespace(
        joint_pos=np.array([[[0.3, -0.2, 0.1]]]),
        body_quat_w=np.array([[[ref_quat]]], dtype=float),
        body_lin_vel_w=np.array([[[lin_vel]]], dtype=float),
        body_ang_vel_w=np.array([[[ang_vel]]], dtype=float),
    )
    state = SimpleNamespace(joint_names=names, root_quat_w=np.asarray(robot_quat, dtype=float),
                            root_ang_vel_b=np.array([0.1, 0.2, 0.3]), joint_pos=np.array([0.5, 0.0, 0.0]),
                            joint_vel=np.array([1.0, -1.0, 0.0]))
    return SimpleNamespace(policy_config={"galbot_track": {"root_body": "pelvis_link"}}, state_processor=state,
                           policy_joint_names=["a", "b"], joint_names_simulation=names,
                           default_dof_angles=np.array([0.2, 0.0, 0.0]), motion_data=motion,
                           motion_body_names=["pelvis_link"], motion_joint_names=names)


def build(env):
    terms = [Observation.resolve(f"galbot_track.galbot_{name}")(env=env) for name in ORDER]
    for term in terms:
        term.reset()
    return terms


def obs(terms, action=(0.0, 0.0)):
    for term in terms:
        term.update({"action": np.asarray(action)})
    return np.concatenate([t.compute() for t in terms], axis=-1)[0]


def test_rot6d_flattens_columns_zero_and_one_row_by_row():
    np.testing.assert_allclose(rot6d([1, 0, 0, 0]), [1, 0, 0, 1, 0, 0])
    # 90 degrees about z: x -> y, y -> -x, so column 0 = (0, 1, 0) and column 1 = (-1, 0, 0).
    np.testing.assert_allclose(rot6d(yaw(np.pi / 2)), [0, -1, 1, 0, 0, 0], atol=1e-12)


def test_layout_and_heading_alignment():
    # Any robot heading: the reference is turned to it at reset, so the orientation term starts at identity.
    env = make_env(yaw(1.0), yaw(-0.5), lin_vel=(0.0, 0.4, 0.0), ang_vel=(0.0, 0.0, 0.7))
    terms = build(env)
    x = obs(terms, action=(9.0, 9.0))
    assert x.shape == (6 + 2 + 1 + 2 + 3 + 3 + 2 + 2 + 2,)
    np.testing.assert_allclose(x[:6], [1, 0, 0, 1, 0, 0], atol=1e-6)
    # World +y velocity of a reference facing -0.5 rad, in its own frame: slightly backward, mostly left.
    np.testing.assert_allclose(x[6:8], [-0.4 * np.sin(0.5), 0.4 * np.cos(0.5)], atol=1e-6)
    np.testing.assert_allclose(x[8], 0.7, atol=1e-6)
    np.testing.assert_allclose(x[9:11], [0.3, -0.2])
    np.testing.assert_allclose(x[11:14], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(x[14:17], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(x[17:19], [0.3, 0.0], atol=1e-6)
    np.testing.assert_allclose(x[19:21], [1.0, -1.0])
    # The first update after a reset ignores the Shaping action in the runtime state.
    np.testing.assert_allclose(x[21:23], [0.0, 0.0])
    np.testing.assert_allclose(obs(terms, action=(0.25, -0.5))[21:23], [0.25, -0.5])


def test_robot_turning_after_reset_shows_in_the_orientation_term():
    env = make_env(yaw(0.0), yaw(0.0))
    terms = build(env)
    obs(terms)
    env.state_processor.root_quat_w = yaw(0.3)  # the robot turned left: the reference is now to its right
    np.testing.assert_allclose(obs(terms)[:6], rot6d(yaw(-0.3)), atol=1e-6)
