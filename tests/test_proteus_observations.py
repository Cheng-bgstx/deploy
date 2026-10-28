from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.rl_policy.observations import Observation
from sim2real.utils.math import quat_from_yaw

# A free pelvis, a hip hinge to the leg and a waist hinge to the torso.
MJCF = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.1"/>
      <body name="leg" pos="0 0.1 -0.2">
        <joint name="hip_joint" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05"/>
      </body>
      <body name="torso" pos="0 0 0.3">
        <joint name="waist_joint" axis="0 0 1"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

GROUPS = {
    "proprioception": ["projected_gravity_history", "root_ang_vel_history", "slot_joint_pos_history",
                       "slot_joint_vel_history"],
    "task": ["target_body_pos", "target_body_pos_rel", "target_body_rot", "target_body_rot_rel", "frame_offsets"],
    "action_history": ["slot_action_history"],
}  # fmt: skip


def _env(tmp_path, *, reference_forcing=True, waist_sign=-1.0):
    mjcf = tmp_path / "robot.xml"
    mjcf.write_text(MJCF)
    state = SimpleNamespace(
        joint_names=["waist_joint", "hip_joint", "ankle_joint"],
        root_pos_w=np.array([0.3, -0.2, 0.9]),
        root_quat_w=np.array([1.0, 0.0, 0.0, 0.0]),
        root_ang_vel_b=np.array([0.1, 0.2, 0.3]),
        joint_pos=np.array([0.5, 0.2, 0.7]),
        joint_vel=np.array([2.0, -4.0, 1.0]),
    )
    policy_config = {
        "default_joint_pos": {"hip_joint": 0.1, "waist_joint": 0.0},
        "proteus": {
            "num_slots": 4,
            "history": 3,
            "joint_vel_scale": 0.05,
            "length_scale": 0.5,
            "body_names": ["pelvis", "leg", "torso"],
            "future_offsets": [0, 1, 16],
            "reference_forcing": reference_forcing,
            "joint_slots": {"hip_joint": 0, "waist_joint": 2},
            "joint_signs": {"hip_joint": 1.0, "waist_joint": waist_sign},
            "mjcf_path": str(mjcf),
        },
    }
    env = SimpleNamespace(state_processor=state, policy_config=policy_config, motion_data=None,
                          motion_body_names=["torso", "pelvis", "leg"], motion_future_steps=[0, 1, 2, 16])  # fmt: skip
    return env


def _observations(env):
    return {
        group: [Observation.resolve(f"proteus.proteus_{name}")(env=env) for name in names]
        for group, names in GROUPS.items()
    }


def _compute(obs):
    return {group: np.concatenate([o.compute() for o in terms], axis=-1) for group, terms in obs.items()}


def _reset(obs):
    for terms in obs.values():
        for o in terms:
            o.reset()


def _update(obs, data):
    for terms in obs.values():
        for o in terms:
            o.update(data)


def _robot_bodies(env):
    core = env._proteus_core
    state = env.state_processor
    root_pos = state.root_pos_w if not core.reference_forcing else core._reference()[0][0, 0]
    return core._bodies(root_pos, state.root_quat_w, state.joint_pos)


def _set_motion(env, body_pos, body_quat):
    """Motion bodies (B, 3/4) in the observation's body order, the same at every future step."""
    order = [["pelvis", "leg", "torso"].index(name) for name in env.motion_body_names]
    steps = len(env.motion_future_steps)
    env.motion_data = SimpleNamespace(
        body_pos_w=np.broadcast_to(body_pos[order], (1, steps) + body_pos[order].shape).copy(),
        body_quat_w=np.broadcast_to(body_quat[order], (1, steps) + body_quat[order].shape).copy(),
    )


def test_group_shapes_and_proprioception(tmp_path):
    env = _env(tmp_path)
    obs = _observations(env)
    _reset(obs)
    _update(obs, {})
    out = _compute(obs)
    assert out["proprioception"].shape == (3, 6 + 2 * 4)
    assert out["task"].shape == (3, 18 * 3 + 1)
    assert out["action_history"].shape == (3, 4)

    # slots: hip in 0, waist (sign -1) in 2, the others empty
    expected = np.concatenate([
        [0.0, 0.0, -1.0], [0.1, 0.2, 0.3],
        [0.2 - 0.1, 0.0, -(0.5 - 0.0), 0.0],
        [-4.0 * 0.05, 0.0, -2.0 * 0.05, 0.0],
    ])  # fmt: skip
    np.testing.assert_allclose(out["proprioception"], np.tile(expected, (3, 1)), atol=1e-6)
    np.testing.assert_array_equal(out["task"][:, -1], [0.0, 1.0, 16.0])


def test_history_is_oldest_first_and_resets_like_isaaclab(tmp_path):
    env = _env(tmp_path)
    obs = _observations(env)
    state = env.state_processor
    _reset(obs)
    _update(obs, {"slot_action": np.full(4, 9.0)})  # first update after a reset: fill, no stale action
    assert not _compute(obs)["action_history"].any()

    for step in range(1, 4):
        state.root_ang_vel_b = np.array([step, 0.0, 0.0])
        _update(obs, {"slot_action": np.full(4, float(step))})
    out = _compute(obs)
    np.testing.assert_array_equal(out["proprioception"][:, 3], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(out["action_history"][:, 0], [1.0, 2.0, 3.0])

    _reset(obs)
    state.root_ang_vel_b = np.array([7.0, 0.0, 0.0])
    _update(obs, {"slot_action": np.full(4, 5.0)})
    out = _compute(obs)
    np.testing.assert_array_equal(out["proprioception"][:, 3], [7.0, 7.0, 7.0])
    assert not out["action_history"].any()


def test_reference_at_the_robot_pose_gives_identity_task(tmp_path):
    env = _env(tmp_path, reference_forcing=False)
    env.state_processor.root_quat_w = quat_from_yaw(np.array(0.4))
    obs = _observations(env)
    _reset(obs)  # no Motion: the reference holds the robot's pose
    _update(obs, {})
    body_pos, body_quat = _robot_bodies(env)
    _set_motion(env, body_pos, body_quat)
    _reset(obs)
    _update(obs, {})
    out = _compute(obs)

    B = 3
    pos, pos_rel = out["task"][:, : 3 * B], out["task"][:, 3 * B : 6 * B]
    rot, rot_rel = out["task"][:, 6 * B : 12 * B], out["task"][:, 12 * B : 18 * B]
    assert np.abs(pos_rel).max() < 1e-6
    np.testing.assert_allclose(rot_rel, np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], (3, B)), atol=1e-6)
    # the anchor relative to itself, the torso 0.3 m above it (divided by the length scale 0.5)
    np.testing.assert_allclose(pos[:, 0:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(pos[:, 6:9], np.tile([0.0, 0.0, 0.6], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(rot[:, 0:6], np.tile([1.0, 0.0, 0.0, 0.0, 0.0, 1.0], (3, 1)), atol=1e-6)


def test_motion_is_aligned_to_the_robot_heading_once(tmp_path):
    env = _env(tmp_path, reference_forcing=True)
    obs = _observations(env)
    env.state_processor.root_quat_w = quat_from_yaw(np.array(np.pi / 2))
    motion_pos = np.array([[5.0, 3.0, 0.8], [5.0, 3.1, 0.6], [5.3, 3.0, 1.1]])  # torso ahead of the pelvis (+x)
    motion_quat = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    _set_motion(env, motion_pos, motion_quat)
    _reset(obs)
    _update(obs, {})

    # Reference forcing: the root stands at the reference anchor, whose heading now matches the robot's.
    ref_pos, ref_quat = env._proteus_core._reference()
    np.testing.assert_allclose(ref_pos[0, 0], [0.0, 0.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(ref_pos[0, 2] - ref_pos[0, 0], [0.0, 0.3, 0.3], atol=1e-6)
    np.testing.assert_allclose(np.abs(ref_quat[0, 0]), np.abs(quat_from_yaw(np.array(np.pi / 2))), atol=1e-6)
    task = _compute(obs)["task"]
    np.testing.assert_allclose(task[:, 0:3], 0.0, atol=1e-6)

    # Later robot turns do not re-align the Motion.
    env.state_processor.root_quat_w = quat_from_yaw(np.array(0.0))
    _update(obs, {})
    np.testing.assert_allclose(env._proteus_core._reference()[0][0, 0], [0.0, 0.0, 0.8], atol=1e-6)


def test_global_tracking_aligns_the_motion_to_the_robot_position(tmp_path):
    env = _env(tmp_path, reference_forcing=False)
    obs = _observations(env)
    _set_motion(env, np.array([[5.0, 3.0, 0.8], [5.0, 3.1, 0.6], [5.0, 3.0, 1.1]]), np.tile([1.0, 0, 0, 0], (3, 1)))
    _reset(obs)
    _update(obs, {})
    np.testing.assert_allclose(env._proteus_core._reference()[0][0, 0], [0.3, -0.2, 0.8], atol=1e-6)


def test_forcing_needs_offset_zero_first(tmp_path):
    env = _env(tmp_path)
    env.policy_config["proteus"]["future_offsets"] = [1, 2]
    with pytest.raises(ValueError, match="offset 0"):
        _observations(env)
