"""Create a PC deployment overlay without modifying the exported actor contract.

G1 by default; ``--robot et1 --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml`` for the
Galbot ET1, which has a waist roll but no waist pitch joint.
"""
from dataclasses import dataclass
from pathlib import Path

import tyro
import yaml

from sim2real.config.robots import get_robot_cfg


@dataclass
class Args:
    policy_config: str = "checkpoints/proteus/proteus_60/unitree_g1/policy.yaml"
    output_name: str = "policy-pc.yaml"
    robot: str = "g1"
    limit_inset: float = 0.05
    """Soft joint limits sit this fraction of the range inside the hard limits (the ET1 ankles need 0.01)."""


def main(args):
    if not 0.0 <= args.limit_inset < 0.5:
        raise ValueError("limit_inset must be in [0, 0.5)")
    robot_cfg = get_robot_cfg(args.robot)
    path = Path(args.policy_config)
    cfg = yaml.safe_load(path.read_text())
    if not cfg["proteus"]["reference_forcing"]:
        raise ValueError("PC G1 deployment requires local reference forcing")
    if cfg["joint_names_simulation"] != list(robot_cfg.joint_names):
        raise ValueError(f"Deployment joint order must match {robot_cfg.name} DDS")
    for name in ("waist_roll_joint", "waist_pitch_joint"):
        if name not in cfg["joint_kp"]:
            continue
        cfg["joint_kp"][name] = 100.0
        cfg["joint_kd"][name] = 2.5
    cfg["shaping_target"] = "default_joint_pos"
    cfg["reference_entry_s"] = 2.0
    cfg["soft_joint_pos_limits"] = {}
    for n in robot_cfg.joint_names:
        low, high = robot_cfg.joint_pos_lower_limit[n], robot_cfg.joint_pos_upper_limit[n]
        inset = args.limit_inset * (high - low)
        cfg["soft_joint_pos_limits"][n] = [low + inset, high - inset]
    cfg["max_target_velocity_rad_s"] = 15.0
    output = path.with_name(args.output_name)
    output.write_text(f"# Deployment overlay: waist roll/pitch PD 100/2.5, arms-down default standing pose, "
                      f"{args.limit_inset:.0%} joint-limit margins, 15 rad/s target slew.\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(output.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))
