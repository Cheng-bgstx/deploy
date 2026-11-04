"""Dry joint friction for the MuJoCo G1, measured on the hanging robot.

The G1 MJCF carries almost none (frictionloss 0.01 N m), and Proteus trained without it, but the real hip and
ankle pitch joints stick within about +-0.06 rad of their PD targets (scripts/g1/joint_response_test.py). A YAML
maps joint-name patterns to Coulomb friction in N m:

    left_hip_pitch_joint|right_hip_pitch_joint: 2.5
    .*_ankle_pitch_joint: 1.3
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import yaml

from sim2real.utils.strings import resolve_matching_names_values


def load_joint_friction(path: str | Path, joint_names: Sequence[str]) -> dict[str, float]:
    spec = yaml.safe_load(Path(path).read_text()) or {}
    _, names, values = resolve_matching_names_values(dict(spec), list(joint_names), preserve_order=True)
    return {name: float(value) for name, value in zip(names, values)}


def apply_joint_friction(model, friction: dict[str, float]) -> None:
    """Set MuJoCo frictionloss (N m) on the named hinge joints."""
    for name, value in friction.items():
        if value < 0:
            raise ValueError(f"joint friction must be non-negative: {name}={value}")
        model.dof_frictionloss[model.joint(name).dofadr[0]] = value
