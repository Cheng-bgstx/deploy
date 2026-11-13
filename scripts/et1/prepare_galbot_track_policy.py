"""Turn a Galbot et1-dep tracking checkpoint (policy protocol 3.2) into a Tracker policy for ET1.

The checkpoint folder (e.g. the 26-1111 addon's ``storage/ckpt-shanshenbu``) holds ``policy.toml``,
``inference.toml``, ``policy.onnx`` (obs normalizer baked in) and ``motion.npz``. This writes:

    <output>/policy.onnx       the same actor
    <output>/policy.yaml       observation group ``obs`` (galbot_track.* terms), joints, gains, action scale
    <output>/policy-pc.yaml    the deployment overlay: 5% inset soft limits, the head locked at 0, 15 rad/s slew,
                               2 s reference entry
    <output>/motions/<name>    the Motion as a qpos dataset (pelvis root + ET1 joints)

    python scripts/et1/prepare_galbot_track_policy.py \
        --checkpoint checkpoints/et1-dep-addon/26-1111/storage/ckpt-shanshenbu \
        --output checkpoints/galbot_track/galbot_et1/shanshenbu
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset

LAYOUT = [  # inference.toml term -> galbot_track observation
    ("command_root_orientation_6d_body", "galbot_root_ori_6d_b", 6),
    ("reference_root_linear_velocity_xy", "galbot_ref_root_lin_vel_xy", 2),
    ("reference_root_angular_velocity_z", "galbot_ref_root_ang_vel_z", 1),
    ("reference_joint_positions", "galbot_ref_joint_pos", None),
    ("projected_gravity", "galbot_projected_gravity", 3),
    ("base_ang_vel", "galbot_base_ang_vel", 3),
    ("joint_pos", "galbot_joint_pos_rel", None),
    ("joint_vel", "galbot_joint_vel", None),
    ("actions", "galbot_prev_action", None),
]


def read_toml(path: Path) -> dict:
    """Parse TOML with a system Python >= 3.11 (tomllib); the project runs on 3.10 without a TOML parser."""
    code = "import json, sys, tomllib; print(json.dumps(tomllib.load(open(sys.argv[1], 'rb'))))"
    for python in ("python3.12", "python3.11", "python3"):
        try:
            return json.loads(subprocess.run([python, "-I", "-c", code, str(path)], check=True,
                                             capture_output=True, text=True).stdout)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("Reading TOML needs a system python3 >= 3.11")


@dataclass
class Args:
    checkpoint: Path
    output: Path
    robot: str = "galbot_et1"
    motion_name: str | None = None
    """Motion folder name (default: the checkpoint folder name without ``ckpt-``)."""
    velocity_frame: str = "body"
    reference_entry_s: float = 2.0
    max_target_velocity_rad_s: float = 15.0


def main(args: Args) -> None:
    robot_cfg = get_robot_cfg(args.robot)
    policy = read_toml(args.checkpoint / "policy.toml")
    inference = read_toml(args.checkpoint / "inference.toml")
    if policy.get("protocol_version") != "3.2" or float(policy["policy_hz"]) != 50.0:
        raise ValueError("Expected a 50 Hz protocol 3.2 policy.toml")
    joints = policy["joint"]
    active = list(joints["names"])
    fixed = policy.get("fixed_joint", {"names": [], "default_positions": [], "kp": [], "kd": []})
    names = list(robot_cfg.joint_names)
    if sorted(active + list(fixed["names"])) != sorted(names):
        raise ValueError(f"policy joints {active} + fixed {fixed['names']} != robot joints {names}")
    n = len(active)
    layout = [term.split(":") for term in inference["policy"]["observation"]["layout"]]
    expected = [(name, str(dim or n)) for name, _, dim in LAYOUT]
    if [tuple(t) for t in layout] != expected:
        raise ValueError(f"Unexpected observation layout {layout}")
    if not np.allclose(inference["policy"]["action"]["action_scale"], policy["action"]["scale"]):
        raise ValueError("inference.toml and policy.toml disagree on the action scale")
    if policy["action"].get("nominal_mode") != "default":
        raise ValueError("Only nominal_mode = 'default' (target = default + scale * action) is supported")

    def per_joint(key):
        return {**dict(zip(active, joints[key])), **dict(zip(fixed["names"], fixed[key]))}

    default = per_joint("default_positions")
    model = mujoco.MjModel.from_xml_path(str(robot_cfg.resolve_mjcf_path()))
    bodies = [model.body(i).name for i in range(1, model.nbody)]
    mjcf_joints = [model.joint(i).name for i in range(1, model.njnt)]
    if mjcf_joints != names:
        raise ValueError(f"MJCF joint order {mjcf_joints} != robot config {names}")

    motion_name = args.motion_name or args.checkpoint.name.removeprefix("ckpt-")
    cfg = {
        "model_path": "policy.onnx",
        "observation": {"obs": {name: {"_target_": f"galbot_track.{target}"} for name, target, _ in LAYOUT}},
        "joint_names_simulation": names,
        "body_names_simulation": bodies,
        "policy_joint_names": active,
        "default_joint_pos": {k: float(default[k]) for k in names},
        "joint_kp": {k: float(v) for k, v in per_joint("kp").items()},
        "joint_kd": {k: float(v) for k, v in per_joint("kd").items()},
        "action_scale": {k: float(v) for k, v in zip(active, policy["action"]["scale"])},
        "galbot_track": {
            "root_body": policy["observation"][0]["root_body"],
            "velocity_frame": args.velocity_frame,
            "source": str(args.checkpoint),
        },
        "motion": {"motion_backend": "npz", "future_steps": [0], "root_body_name": "pelvis_link", "motion_dt_s": 0.02},
        "shaping_target": "default_joint_pos",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint / "policy.onnx", args.output / "policy.onnx")
    (args.output / "policy.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    overlay = dict(cfg)
    overlay["reference_entry_s"] = args.reference_entry_s
    limits = {}
    for name in names:
        low, high = robot_cfg.joint_pos_lower_limit[name], robot_cfg.joint_pos_upper_limit[name]
        inset = 0.05 * (high - low)
        limits[name] = [float(low + inset), float(high - inset)]
    for name in fixed["names"]:
        limits[name] = [float(default[name])] * 2
    overlay["soft_joint_pos_limits"] = limits
    overlay["max_target_velocity_rad_s"] = args.max_target_velocity_rad_s
    header = (f"# Deployment overlay: {', '.join(fixed['names'])} locked, 5% joint-limit margins, "
              f"{args.max_target_velocity_rad_s:g} rad/s target slew, {args.reference_entry_s:g} s reference entry.\n")
    (args.output / "policy-pc.yaml").write_text(header + yaml.safe_dump(overlay, sort_keys=False))

    with np.load(args.checkpoint / "motion.npz", allow_pickle=False) as data:
        joint_names = [str(x) for x in data["joint_names"]]
        if str(data["body_names"][0]) != "pelvis_link":
            raise ValueError("Expected pelvis_link as the Motion's first body")
        root_quat = data["body_quat_w"][:, 0]
        qpos = np.concatenate([data["body_pos_w"][:, 0], root_quat,
                               data["joint_pos"][:, [joint_names.index(k) for k in names]]], axis=1)
        fps = float(data["fps"])
    if not np.isfinite(qpos).all() or np.abs(np.linalg.norm(root_quat, axis=1) - 1).max() > 1e-3:
        raise ValueError("non-finite qpos or non-unit root quaternion")
    output = args.output / "motions" / motion_name
    if output.exists():
        shutil.rmtree(output)
    write_single_motion_qpos_dataset(
        output, robot_cfg=robot_cfg, qpos=qpos, fps=fps,
        dataset_name=f"Galbot {motion_name} reference motion",
        source=dict(checkpoint=str(args.checkpoint), fps=fps),
    )  # fmt: skip
    (args.output / "source.json").write_text(json.dumps(
        {"checkpoint": str(args.checkpoint), "motion": motion_name, "frames": len(qpos), "fps": fps}, indent=2))
    print(f"{motion_name}: {len(qpos) / fps:.1f} s -> {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
