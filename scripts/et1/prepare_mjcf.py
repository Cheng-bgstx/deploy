#!/usr/bin/env python3
"""Prepare the Galbot ET1 MJCF for sim2real from et1-dep's vendor model.

et1-dep (``storage/humanoid_description/galbot_et1``) ships ``galbot_et1_26dof.xml``, whose elbow and wrist body
frames are turned by -90 deg about y relative to the Unitree R1 model the ET1 derives from (the frames of
Galbot's retargeted motions and of the R1 Proteus was trained on). The policy observes tracked-body orientations,
so sim2real uses the vendor model with those bodies re-expressed in the Unitree frames: each body gets
``quat * FRAME_ROTATION`` and everything defined in its frame (inertial, joints, geoms, sites, child bodies) the
inverse rotation. The physics is unchanged; the script checks body positions, the frame change, masses and the
mass matrix against the vendor model at random configurations.

    uv run python scripts/et1/prepare_mjcf.py --source ../et1-dep/storage/humanoid_description/galbot_et1

Writes ``<output-dir>/galbot_et1_26dof_unitree_frames.xml`` next to copies of the meshes and the DDS joint table.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from sim2real.config.robots.et1 import ET1_MJCF_PATH

VENDOR_MJCF = "galbot_et1_26dof.xml"
EXTRA_FILES = ("galbot_et1_joint_index_dds.md",)
REFRAMED_BODIES = ("left_elbow_link", "left_wrist_roll_link", "right_elbow_link", "right_wrist_roll_link")
FRAME_ROTATION = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])  # +90 deg about y: vendor frame -> Unitree frame
TOLERANCE = 1.0e-9


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


def _conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return _quat_mul(_quat_mul(q, np.concatenate([[0.0], v])), _conj(q))[1:]


def _vec(node: ET.Element, key: str, default: str) -> np.ndarray:
    return np.array([float(x) for x in node.get(key, default).split()])


def _fmt(values: np.ndarray) -> str:
    values = np.where(np.abs(values) < 1.0e-12, 0.0, values)
    return " ".join(f"{v:.9g}" for v in values)


def reframe_body(body: ET.Element, rotation: np.ndarray) -> None:
    """Turn ``body``'s frame by ``rotation`` (in its own frame), keeping every element where it is."""
    inverse = _conj(rotation)
    unsupported = [k for k in ("axisangle", "xyaxes", "zaxis", "euler") if k in body.attrib]
    if unsupported:
        raise ValueError(f"body {body.get('name')}: orientation given as {unsupported}, only quat is supported")
    body.set("quat", _fmt(_quat_mul(_vec(body, "quat", "1 0 0 0"), rotation)))
    for child in body:
        if child.tag not in ("inertial", "joint", "geom", "site", "body", "camera"):
            continue
        if any(k in child.attrib for k in ("fromto", "axisangle", "xyaxes", "zaxis", "euler", "fullinertia")):
            raise ValueError(f"{child.tag} {child.get('name', '')} of {body.get('name')}: unsupported placement")
        if "pos" in child.attrib or child.tag in ("body", "geom", "site"):
            child.set("pos", _fmt(_rotate(inverse, _vec(child, "pos", "0 0 0"))))
        if child.tag == "joint":
            child.set("axis", _fmt(_rotate(inverse, _vec(child, "axis", "0 0 1"))))
        else:
            child.set("quat", _fmt(_quat_mul(inverse, _vec(child, "quat", "1 0 0 0"))))


def reframe_mjcf(source: Path, output: Path, bodies: tuple[str, ...] = REFRAMED_BODIES) -> None:
    tree = ET.parse(source)
    found = {b.get("name"): b for b in tree.getroot().iter("body") if b.get("name") in bodies}
    if missing := sorted(set(bodies) - set(found)):
        raise ValueError(f"{source} has no bodies {missing}")
    for name in bodies:
        reframe_body(found[name], FRAME_ROTATION)
    comment = ET.Comment(
        f" sim2real scripts/et1/prepare_mjcf.py: {source.name} with {', '.join(bodies)} in Unitree R1 frames "
    )
    tree.getroot().insert(0, comment)
    tree.write(output, encoding="unicode")


def _mass_matrix(mujoco, model, data) -> np.ndarray:
    out, unit, column = np.zeros((model.nv, model.nv)), np.zeros(model.nv), np.zeros(model.nv)
    for i in range(model.nv):
        unit[:] = 0.0
        unit[i] = 1.0
        mujoco.mj_mulM(model, data, column, unit)
        out[:, i] = column
    return out


def check_equivalent(vendor_path: Path, prepared_path: Path, bodies: tuple[str, ...] = REFRAMED_BODIES,
                     samples: int = 16, seed: int = 0) -> dict[str, float]:  # fmt: skip
    """Largest differences between the two models at random configurations (raises above TOLERANCE)."""
    import mujoco

    vendor = mujoco.MjModel.from_xml_path(str(vendor_path))
    prepared = mujoco.MjModel.from_xml_path(str(prepared_path))
    names = [vendor.body(i).name for i in range(vendor.nbody)]
    if names != [prepared.body(i).name for i in range(prepared.nbody)] or vendor.nq != prepared.nq:
        raise ValueError("the prepared model's bodies or joints differ from the vendor model")
    reframed = np.array([name in bodies for name in names])
    rng = np.random.default_rng(seed)
    errors = {"body_pos": 0.0, "body_quat": 0.0, "mass": float(np.abs(vendor.body_mass - prepared.body_mass).max()),
              "mass_matrix": 0.0, "bias_force": 0.0}  # fmt: skip
    dv, dp = mujoco.MjData(vendor), mujoco.MjData(prepared)
    for _ in range(samples):
        lo, hi = vendor.jnt_range[1:].T
        q = np.concatenate([[0.0, 0.0, 0.8], [1.0, 0.0, 0.0, 0.0], rng.uniform(lo, hi)])
        v = rng.normal(size=vendor.nv)
        for model, data in ((vendor, dv), (prepared, dp)):
            data.qpos[:], data.qvel[:] = q, v
            mujoco.mj_forward(model, data)
        expected_quat = np.array([_quat_mul(qv, FRAME_ROTATION) if r else qv for qv, r in zip(dv.xquat, reframed)])
        quat_err = 1.0 - np.abs(np.sum(expected_quat * dp.xquat, axis=1))
        mv, mp = _mass_matrix(mujoco, vendor, dv), _mass_matrix(mujoco, prepared, dp)
        errors["body_pos"] = max(errors["body_pos"], float(np.abs(dv.xpos - dp.xpos).max()))
        errors["body_quat"] = max(errors["body_quat"], float(quat_err.max()))
        errors["mass_matrix"] = max(errors["mass_matrix"], float(np.abs(mv - mp).max()))
        errors["bias_force"] = max(errors["bias_force"], float(np.abs(dv.qfrc_bias - dp.qfrc_bias).max()))
    if bad := {k: v for k, v in errors.items() if v > TOLERANCE}:
        raise RuntimeError(f"prepared model differs from the vendor model: {bad}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="et1-dep's storage/humanoid_description/galbot_et1")
    parser.add_argument("--output-dir", type=Path, default=ET1_MJCF_PATH.parent)
    args = parser.parse_args()

    source, output_dir = args.source.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if source != output_dir:
        shutil.copytree(source / "meshes", output_dir / "meshes", dirs_exist_ok=True)
        for name in (VENDOR_MJCF, *EXTRA_FILES):
            shutil.copy2(source / name, output_dir / name)
    output = output_dir / ET1_MJCF_PATH.name
    reframe_mjcf(output_dir / VENDOR_MJCF, output)
    errors = check_equivalent(output_dir / VENDOR_MJCF, output)
    print(f"[et1] {output}")
    print("  max |prepared - vendor|: " + ", ".join(f"{k} {v:.2g}" for k, v in errors.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
