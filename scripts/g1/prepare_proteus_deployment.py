"""Create a PC deployment overlay without modifying the exported actor contract."""
from dataclasses import dataclass
from pathlib import Path

import tyro
import yaml

from sim2real.config.robots import G1_CFG


@dataclass
class Args:
    policy_config: str = "checkpoints/proteus/proteus_60/unitree_g1/policy.yaml"
    output_name: str = "policy-pc.yaml"


def main(args):
    path = Path(args.policy_config)
    cfg = yaml.safe_load(path.read_text())
    if not cfg["proteus"]["reference_forcing"]:
        raise ValueError("PC G1 deployment requires local reference forcing")
    if cfg["joint_names_simulation"] != list(G1_CFG.joint_names):
        raise ValueError("Deployment joint order must match G1 DDS")
    for name in ("waist_roll_joint", "waist_pitch_joint"):
        cfg["joint_kp"][name] = 100.0
        cfg["joint_kd"][name] = 2.5
    cfg["shaping_target"] = "default_joint_pos"
    cfg["reference_entry_s"] = 2.0
    cfg["soft_joint_pos_limits"] = {}
    for n in G1_CFG.joint_names:
        low, high = G1_CFG.joint_pos_lower_limit[n], G1_CFG.joint_pos_upper_limit[n]
        inset = .05 * (high - low)
        cfg["soft_joint_pos_limits"][n] = [low + inset, high - inset]
    cfg["max_target_velocity_rad_s"] = 15.0
    output = path.with_name(args.output_name)
    output.write_text("# Deployment overlay: waist roll/pitch PD 100/2.5, arms-down default standing pose, 5% joint-limit margins, 15 rad/s target slew.\n" + yaml.safe_dump(cfg, sort_keys=False))
    print(output.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))
