"""Galbot ET1 robot config against the et1-dep MJCF and the Proteus test-motion conversion."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sim2real.config.robots import ET1_CFG, get_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
MJCF = Path(ET1_CFG.mjcf_path)


def _load_script(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


def test_et1_is_registered_under_both_names():
    assert get_robot_cfg("et1") is ET1_CFG
    assert get_robot_cfg("galbot_et1") is ET1_CFG
    assert len(ET1_CFG.joint_names) == 26
    assert ET1_CFG.qpos_size == 33


def test_et1_limits_and_stand_pose_are_consistent():
    names = ET1_CFG.joint_names
    stand = np.asarray(ET1_CFG.default_qpos[7:])
    lower = np.asarray([ET1_CFG.joint_pos_lower_limit[n] for n in names])
    upper = np.asarray([ET1_CFG.joint_pos_upper_limit[n] for n in names])
    assert np.all(lower < upper)
    assert np.all((stand > lower) & (stand < upper))
    # The hardware model limits waist yaw more than the Proteus training spec.
    assert ET1_CFG.joint_pos_upper_limit["waist_yaw_joint"] == pytest.approx(0.7854)
    assert all(ET1_CFG.safe_joint_kd[n] > 0 for n in names)


@pytest.mark.skipif(not MJCF.exists(), reason="et1-dep checkout not found (set ET1_DEP_ROOT)")
def test_et1_config_matches_et1_dep_mjcf():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert [model.joint(i).name for i in range(1, model.njnt)] == list(ET1_CFG.joint_names)
    for name in ET1_CFG.joint_names:
        low, high = model.joint(name).range
        assert low == pytest.approx(ET1_CFG.joint_pos_lower_limit[name], abs=1e-3), name
        assert high == pytest.approx(ET1_CFG.joint_pos_upper_limit[name], abs=1e-3), name
    for body in ET1_CFG.body_names:
        assert model.body(body).id >= 0


def test_clip_qpos_reorders_joints_into_mjcf_order(tmp_path):
    module = _load_script("scripts/prepare_proteus_test_motions.py")
    mjcf_names = ["a_joint", "b_joint", "c_joint"]
    frames = 4
    joint_pos = np.arange(frames * 3, dtype=np.float32).reshape(frames, 3)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (frames, 2, 1))
    pos = np.zeros((frames, 2, 3))
    pos[:, 0, 2] = 0.7
    path = tmp_path / "clip.npz"
    np.savez(path, joint_pos=joint_pos, joint_names=np.array(["c_joint", "a_joint", "b_joint"]),
             body_pos_w=pos, body_quat_w=quat, body_names=np.array(["pelvis_link", "hand"]), fps=50)  # fmt: skip
    qpos, fps = module.clip_qpos(path, mjcf_names, "pelvis_link")
    assert fps == 50
    np.testing.assert_allclose(qpos[:, :3], pos[:, 0])
    np.testing.assert_allclose(qpos[:, 3:7], quat[:, 0])
    np.testing.assert_allclose(qpos[:, 7:], joint_pos[:, [1, 2, 0]])
    with pytest.raises(ValueError):
        module.clip_qpos(path, mjcf_names, "torso")


def test_et1_overlay_locks_the_head_at_its_default(tmp_path):
    import yaml

    from sim2real.rl_policy.utils.joint_targets import JointTargetPostprocessor

    module = _load_script("scripts/g1/prepare_proteus_deployment.py")
    names = list(ET1_CFG.joint_names)
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(dict(
        proteus=dict(reference_forcing=True), joint_names_simulation=names, policy_joint_names=names,
        default_joint_pos={n: 0.0 for n in names}, joint_kp={n: 1.0 for n in names}, joint_kd={n: 0.1 for n in names},
    )))  # fmt: skip
    module.main(module.Args(policy_config=str(policy), robot="galbot_et1"))
    cfg = yaml.safe_load((tmp_path / "policy-pc.yaml").read_text())
    assert cfg["soft_joint_pos_limits"]["head_yaw_joint"] == [0.0, 0.0]
    assert cfg["soft_joint_pos_limits"]["head_pitch_joint"] == [0.0, 0.0]
    assert cfg["soft_joint_pos_limits"]["waist_yaw_joint"][1] < ET1_CFG.joint_pos_upper_limit["waist_yaw_joint"]

    post = JointTargetPostprocessor.from_policy_config(cfg, joint_names=names, policy_joint_names=names, control_dt=0.02)
    raw = np.zeros(len(names))
    raw[names.index("head_yaw_joint")] = 1.05
    raw[names.index("head_pitch_joint")] = 0.4
    target = post(raw, np.zeros(len(names)))
    assert target[names.index("head_yaw_joint")] == 0.0
    assert target[names.index("head_pitch_joint")] == 0.0

    module.main(module.Args(policy_config=str(policy), robot="galbot_et1", locked_joints=(), output_name="free.yaml"))
    free = yaml.safe_load((tmp_path / "free.yaml").read_text())
    assert free["soft_joint_pos_limits"]["head_yaw_joint"][1] > 1.0
