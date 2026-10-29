#!/usr/bin/env python3
"""Convert Galbot ET1 motions from et1-dep into an any4hdmi qpos dataset for sim2real.

et1-dep keeps ET1 motions (``storage/motions/galbot_et1``) in two layouts, both 50 Hz with the joint and body
names of the ET1 model:

- a ``.npz`` with ``fps``, ``joint_names``, ``body_names``, ``joint_pos``, ``body_pos_w`` and ``body_quat_w``;
- a directory with ``meta.json`` (``fps``, ``joint_names``, ``body_names``) and the same arrays as ``.npy``.

Each motion becomes ``<output-dir>/motions/<name>.npz``: the qpos root is the ``pelvis_link`` world pose
(``body_quat_w`` is WXYZ), the joints are copied by name. The sources are checked against the sim2real ET1
MJCF (``scripts/et1/prepare_mjcf.py``): the forward kinematics of every converted frame must reproduce the stored
positions and orientations of the bodies both models have. The arrays are loaded with ``allow_pickle=False``;
run the script with ``python -I``.

    uv run --frozen python -I scripts/et1/convert_motions.py \\
        --output-dir /abs/path/checkpoints/galbot_et1/motions \\
        --motions "/abs/path/et1-dep/storage/motions/galbot_et1/1010test/2-太极大师-20000轮.npz=taichi" \\
            /abs/path/et1-dep/storage/motions/galbot_et1/liedown_getup_data/0814getup_liedown/liedown_002/liedown_002_yanwei

A source without ``=<name>`` is named after its file or directory, which must then be a valid name.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ET1_FPS = 50
ROOT_BODY_NAME = "pelvis_link"
ROOT_QPOS_NAMES = ("root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz")
DATASET_NAME = "galbot_et1"
POS_TOLERANCE = 1.0e-4  # m
QUAT_TOLERANCE = 1.0e-6  # 1 - |<q_fk, q_stored>|
FK_CHECK_STRIDE = 10

_NAME = re.compile(r"[A-Za-z0-9_]+")
_ARRAYS = ("joint_pos", "body_pos_w", "body_quat_w")


@dataclass(frozen=True)
class SourceMotion:
    path: Path
    fps: float
    joint_names: list[str]
    body_names: list[str]
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray


def parse_source(spec: str) -> tuple[str, Path]:
    """``path`` or ``path=name`` -> (name, path)."""
    path_text, _, name = spec.rpartition("=") if "=" in spec else (spec, "", "")
    path = Path(path_text).expanduser()
    if not name:
        name = path.stem if path.suffix == ".npz" else path.name
    if not _NAME.fullmatch(name):
        raise ValueError(f"Invalid motion name {name!r} for {path}: append =<name> with letters, digits and '_'")
    return name, path


def load_source(path: str | Path) -> SourceMotion:
    path = Path(path)
    if path.is_dir():
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        arrays = {key: np.load(path / f"{key}.npy", allow_pickle=False) for key in _ARRAYS}
        fps, joint_names, body_names = meta["fps"], meta["joint_names"], meta["body_names"]
    else:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {key: np.asarray(payload[key]) for key in _ARRAYS}
            fps = payload["fps"]
            joint_names, body_names = payload["joint_names"].tolist(), payload["body_names"].tolist()
    motion = SourceMotion(
        path=path,
        fps=float(fps),
        joint_names=[str(name) for name in joint_names],
        body_names=[str(name) for name in body_names],
        **{key: np.asarray(value, dtype=np.float64) for key, value in arrays.items()},
    )
    frames = motion.joint_pos.shape[0]
    expected = {
        "joint_pos": (frames, len(motion.joint_names)),
        "body_pos_w": (frames, len(motion.body_names), 3),
        "body_quat_w": (frames, len(motion.body_names), 4),
    }
    for key, shape in expected.items():
        if getattr(motion, key).shape != shape:
            raise ValueError(f"{path}: {key} has shape {getattr(motion, key).shape}, expected {shape}")
    if not np.isfinite(motion.joint_pos).all() or not np.isfinite(motion.body_pos_w).all():
        raise ValueError(f"{path}: non-finite values")
    return motion


def motion_to_qpos(motion: SourceMotion, qpos_names: list[str]) -> np.ndarray:
    if round(motion.fps) != ET1_FPS or abs(motion.fps - ET1_FPS) > 1.0e-6:
        raise ValueError(f"{motion.path}: {motion.fps} fps, expected {ET1_FPS}")
    if tuple(qpos_names[:7]) != ROOT_QPOS_NAMES:
        raise ValueError(f"qpos must start with a floating root, got {qpos_names[:7]}")
    if ROOT_BODY_NAME not in motion.body_names:
        raise ValueError(f"{motion.path}: no {ROOT_BODY_NAME} body")
    missing = [name for name in qpos_names[7:] if name not in motion.joint_names]
    if missing:
        raise ValueError(f"{motion.path}: missing joints {missing}")
    root = motion.body_names.index(ROOT_BODY_NAME)
    quat = motion.body_quat_w[:, root]
    qpos = np.empty((motion.joint_pos.shape[0], len(qpos_names)), dtype=np.float64)
    qpos[:, 0:3] = motion.body_pos_w[:, root]
    qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    qpos[:, 7:] = motion.joint_pos[:, [motion.joint_names.index(name) for name in qpos_names[7:]]]
    return qpos


def fk_errors(model, motion: SourceMotion, qpos: np.ndarray, stride: int = FK_CHECK_STRIDE) -> tuple[float, float]:
    """Largest body position and orientation errors of ``model``'s FK against the stored bodies."""
    import mujoco

    data = mujoco.MjData(model)
    shared = []
    for index, name in enumerate(motion.body_names):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            shared.append((index, body_id))
    source_index, body_ids = map(list, zip(*shared))
    pos_err = quat_err = 0.0
    for frame in range(0, qpos.shape[0], stride):
        data.qpos[:] = qpos[frame]
        mujoco.mj_kinematics(model, data)
        pos_err = max(pos_err, float(np.abs(data.xpos[body_ids] - motion.body_pos_w[frame, source_index]).max()))
        stored = motion.body_quat_w[frame, source_index]
        stored = stored / np.linalg.norm(stored, axis=-1, keepdims=True)
        quat_err = max(quat_err, float((1.0 - np.abs(np.sum(data.xquat[body_ids] * stored, axis=-1))).max()))
    return pos_err, quat_err


def _sha256_of(path: Path) -> str:
    if path.is_dir():
        hasher = hashlib.sha256()
        for name in ("meta.json", *(f"{key}.npy" for key in _ARRAYS)):
            hasher.update((path / name).read_bytes())
        return hasher.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def convert_motions(sources: dict[str, Path], output_dir: str | Path, *, robot: str = "et1") -> dict[str, Path]:
    """Write each source as ``<output_dir>/motions/<name>.npz``; motions already in the dataset are kept."""
    import mujoco
    from any4hdmi.core.format import MOTIONS_SUBDIR, save_motion, write_manifest
    from any4hdmi.utils.mjcf import qpos_names_from_model

    from sim2real.config.robots import get_robot_cfg
    from sim2real.config.robots.base import resolve_mjcf_root_body_name

    robot_cfg = get_robot_cfg(robot)
    mjcf_path = robot_cfg.resolve_mjcf_path()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    if (root_body := resolve_mjcf_root_body_name(model)) != ROOT_BODY_NAME:
        raise ValueError(f"{mjcf_path}: root body is {root_body!r}, expected {ROOT_BODY_NAME!r}")
    qpos_names = qpos_names_from_model(model)

    output_root = Path(output_dir).expanduser().resolve()
    motions_dir = output_root / MOTIONS_SUBDIR
    manifest_path = output_root / "manifest.json"
    info: dict[str, Any] = {}
    if manifest_path.is_file():
        info = dict(json.loads(manifest_path.read_text(encoding="utf-8")).get("source", {}).get("motions", {}))

    written = {}
    for name, path in sources.items():
        motion = load_source(path)
        qpos = motion_to_qpos(motion, qpos_names)
        pos_err, quat_err = fk_errors(model, motion, qpos)
        if pos_err > POS_TOLERANCE or quat_err > QUAT_TOLERANCE:
            raise ValueError(
                f"{path}: {mjcf_path.name} does not reproduce the stored bodies (position error {pos_err:.3g} m, "
                f"orientation 1-|dot| {quat_err:.3g}); was the motion retargeted to another model?"
            )
        written[name] = save_motion(motions_dir / f"{name}.npz", qpos)
        info[name] = {
            "source": path.name,
            "source_sha256": _sha256_of(path),
            "frames": int(qpos.shape[0]),
            "fk_max_position_error_m": pos_err,
        }

    present = sorted(path.stem for path in motions_dir.glob("*.npz"))
    info = {name: info[name] for name in present if name in info}
    write_manifest(
        output_root,
        dataset_name=DATASET_NAME,
        mjcf=mjcf_path,
        timestep=1.0 / ET1_FPS,
        qpos_names=qpos_names,
        num_motions=len(present),
        total_hours=sum(entry["frames"] for entry in info.values()) / ET1_FPS / 3600.0,
        source={
            "kind": "et1-dep motions (npz or meta.json + npy), 50 Hz",
            "root": f"{ROOT_BODY_NAME} world pose from body_pos_w/body_quat_w (WXYZ), no re-centring",
            "joints": "copied by name from joint_pos",
            "motions": info,
        },
    )
    return written


@dataclass
class ConvertArgs:
    """Convert et1-dep ET1 motions into an any4hdmi dataset."""

    motions: tuple[str, ...]
    """Motion .npz files or meta.json directories, each optionally followed by =<name>."""
    output_dir: Path
    """any4hdmi dataset root; existing motions in it are kept."""
    robot: str = "et1"


def main() -> None:
    import tyro

    args = tyro.cli(ConvertArgs)
    sources = dict(parse_source(spec) for spec in args.motions)
    written = convert_motions(sources, args.output_dir, robot=args.robot)
    for name, path in written.items():
        print(f"[convert] {name} -> {path}")
    print(f"[convert] manifest: {Path(args.output_dir).expanduser().resolve() / 'manifest.json'}")


if __name__ == "__main__":
    main()
