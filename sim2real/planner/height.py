"""Mesh-free G1 leg FK for a one-time feet-on-ground initialization estimate.

Ported from Scalecar_v2 ``hardware/height.py`` (SOURCES.json); only the profile path differs.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def standing_height(joints: np.ndarray, names: list[str], quat: np.ndarray) -> dict:
    """Estimate pelvis height from foot collision soles; no motion odometry implied."""
    q = np.asarray(joints, float)
    if q.shape != (29,) or not np.isfinite(q).all() or len(set(names)) != 29:
        raise ValueError("Standing FK requires 29 named finite joint positions")
    profile = json.loads((Path(__file__).resolve().parent / "profiles/g1_leg_kinematics.json").read_text())
    angles = dict(zip(names, q))
    root_rotation = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
    heights = []

    def visit(body, position, rotation):
        position = position + rotation @ np.asarray(body["pos"])
        rotation = rotation @ Rotation.from_quat(np.array(body["quat"])[[1, 2, 3, 0]]).as_matrix()
        if "joint" in body:
            joint = body["joint"]
            rotation = (
                rotation
                @ Rotation.from_rotvec(
                    np.asarray(joint["axis"]) * angles[joint["name"]]
                ).as_matrix()
            )
        if body["sole_points"]:
            heights.append(
                -min(
                    (position + rotation @ np.array(point[:3]))[2] - point[3]
                    for point in body["sole_points"]
                )
            )
        for child in body["children"]:
            visit(child, position, rotation)

    for leg in profile["legs"]:
        visit(leg, np.zeros(3), root_rotation)
    if len(heights) != 2 or not np.isfinite(heights).all():
        raise ValueError("Standing FK did not resolve two feet")
    height = float(np.mean(heights))
    difference = float(abs(heights[0] - heights[1]))
    if not 0.45 < height < 1.0 or difference > 0.08:
        raise ValueError(f"Feet-on-ground FK assumption inconsistent: heights={heights}")
    return dict(
        height_m=height,
        left_m=heights[0],
        right_m=heights[1],
        difference_m=difference,
        source="G1 leg FK, feet-on-ground assumption; startup only",
        profile_scene_sha256=profile["source_scene_sha256"],
    )
