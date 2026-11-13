"""Create a PC deployment overlay without modifying the exported actor contract."""
from dataclasses import dataclass
from pathlib import Path

import tyro
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.bridge_safety import PROTEUS_TARGET_MARGIN_RAD


@dataclass
class Args:
    policy_config: str = "checkpoints/proteus/proteus_60/unitree_g1/policy.yaml"
    output_name: str = "policy-pc.yaml"
    robot: str = "g1"
    locked_joints: tuple[str, ...] | None = None
    """Joints held at their default_joint_pos; None uses the robot's default (ET1: head)."""


# The Proteus ET1 actor tracks no head body and drives the head to about 1 rad yaw.
DEFAULT_LOCKED_JOINTS = {"galbot_et1": ("head_pitch_joint", "head_yaw_joint"),
                         # X2 Basic reports a constant zero head; the Ultra drives head yaw only.
                         "x2": ("head_pitch_joint", "head_yaw_joint")}


def main(args):
    path = Path(args.policy_config)
    cfg = yaml.safe_load(path.read_text())
    if not cfg["proteus"]["reference_forcing"]:
        raise ValueError("PC G1 deployment requires local reference forcing")
    robot_cfg = get_robot_cfg(args.robot)
    if cfg["joint_names_simulation"] != list(robot_cfg.joint_names):
        raise ValueError(f"Deployment joint order must match the {robot_cfg.name} robot config")
    if robot_cfg.name == "g1":
        # G1 hardware: the trained waist gains caused high-frequency waist tremor.
        for name in ("waist_roll_joint", "waist_pitch_joint"):
            cfg["joint_kp"][name] = 100.0
            cfg["joint_kd"][name] = 2.5
    cfg["shaping_target"] = "default_joint_pos"
    cfg["reference_entry_s"] = 2.0
    cfg["soft_joint_pos_limits"] = {}
    for n in robot_cfg.joint_names:
        low, high = robot_cfg.joint_pos_lower_limit[n], robot_cfg.joint_pos_upper_limit[n]
        if robot_cfg.name == "x2":
            # Never wider than the vendor's physical range.
            from sim2real.config.robots.x2 import X2_VENDOR_LIMITS
            low, high = max(low, X2_VENDOR_LIMITS[n][0]), min(high, X2_VENDOR_LIMITS[n][1])
        if robot_cfg.name == "g1":
            # Proteus trains without target clipping; bound targets only by what the G1 Bridge accepts.
            cfg["soft_joint_pos_limits"][n] = [low - PROTEUS_TARGET_MARGIN_RAD, high + PROTEUS_TARGET_MARGIN_RAD]
        else:
            inset = .05 * (high - low)
            cfg["soft_joint_pos_limits"][n] = [low + inset, high - inset]
    # A zero-width soft limit clips the policy's target to the default pose; the
    # observation still reports the measured joint, as for any other joint.
    locked = DEFAULT_LOCKED_JOINTS.get(robot_cfg.name, ()) if args.locked_joints is None else args.locked_joints
    for n in locked:
        cfg["soft_joint_pos_limits"][n] = [cfg["default_joint_pos"][n]] * 2
    cfg["max_target_velocity_rad_s"] = 15.0
    output = path.with_name(args.output_name)
    gains = "waist roll/pitch PD 100/2.5, arms-down " if robot_cfg.name == "g1" else ""
    lock = f"{', '.join(locked)} locked, " if locked else ""
    margins = (f"targets up to {PROTEUS_TARGET_MARGIN_RAD:g} rad past the joint limits" if robot_cfg.name == "g1"
               else "5% joint-limit margins")
    output.write_text(f"# Deployment overlay: {gains}default standing pose, {lock}{margins}, 15 rad/s target slew.\n"
                      + yaml.safe_dump(cfg, sort_keys=False))
    print(output.resolve())


if __name__ == "__main__":
    main(tyro.cli(Args))
