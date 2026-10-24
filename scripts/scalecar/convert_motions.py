#!/usr/bin/env python3
"""Convert Scalecar 50 Hz motion NPZ files into an any4hdmi qpos dataset.

The input tar is verified against its SHA256SUMS before anything is read, and
only the requested takes are extracted, into a temporary directory. The NPZ
files are treated as untrusted data (``allow_pickle=False``); run this script
with ``python -I``.

Each take becomes ``<output_dir>/motions/<take>.npz``. The qpos root is the
pelvis world pose from ``body_pos_w``/``body_quat_w`` (WXYZ), unchanged, so the
Motion keeps the vehicle scene's world frame. Joints are copied by name.

    uv run --frozen python -I scripts/scalecar/convert_motions.py \\
        --output-dir /abs/path/checkpoints/scalecar_d1146/motions \\
        --takes take1512 take0595
"""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

MODELSCOPE_ASSETS = Path(
    "/home/galbot/.cache/modelscope/models/Hzzzz001--scalecar_v2/snapshots/master/assets"
)
DEFAULT_TAR = MODELSCOPE_ASSETS / "motions_merged1146_converted_50hz_v1.tar"
DEFAULT_SHA256SUMS = MODELSCOPE_ASSETS / "SHA256SUMS"
TAR_MOTIONS_DIR = "data/scalecar_g1_merged1146_20261020_v1/converted_50hz_v1/motions"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "checkpoints/scalecar_d1146/motions"
SCALECAR_FPS = 50
ROOT_BODY_NAME = "pelvis"
ROOT_QPOS_NAMES = ("root_tx", "root_ty", "root_tz", "root_qw", "root_qx", "root_qy", "root_qz")

_TAKE_NAME = re.compile(r"[A-Za-z0-9_]+")
_HASH_CHUNK_BYTES = 8 * 1024 * 1024
_REQUIRED_KEYS = ("fps", "robot_joint_names", "robot_body_names", "joint_pos", "body_pos_w", "body_quat_w")


def _sha256_of(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


def _expected_sha256(sums_path: Path, file_name: str) -> str:
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == file_name:
            return parts[0].lower()
    raise ValueError(f"{file_name} is not listed in SHA256SUMS {sums_path}")


def verify_sha256(path: str | Path, sums_path: str | Path) -> str:
    """Return the file's sha256 after checking it against a SHA256SUMS file."""
    path = Path(path)
    expected = _expected_sha256(Path(sums_path), path.name)
    actual = _sha256_of(path)
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def extract_takes(
    tar_path: str | Path,
    sha256sums_path: str | Path,
    takes: Iterable[str],
    out_dir: str | Path,
    *,
    member_dir: str = TAR_MOTIONS_DIR,
) -> dict[str, Path]:
    """Verify the tar, then extract ``<member_dir>/<take>.npz`` for each take into out_dir."""
    takes = list(takes)
    for take in takes:
        if not _TAKE_NAME.fullmatch(take):
            raise ValueError(f"Invalid take name {take!r}: use letters, digits and '_' only")
    tar_path = Path(tar_path)
    out_dir = Path(out_dir)
    verify_sha256(tar_path, sha256sums_path)

    with tarfile.open(tar_path, "r:") as tar:
        members = {}
        for take in takes:
            try:
                member = tar.getmember(f"{member_dir}/{take}.npz")
            except KeyError:
                raise KeyError(f"{take} is not in {tar_path} under {member_dir}") from None
            if not member.isfile():
                raise ValueError(f"{member.name} in {tar_path} is not a regular file")
            members[take] = member

        out_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for take, member in members.items():
            source = tar.extractfile(member)
            assert source is not None
            target = out_dir / f"{take}.npz"
            target.write_bytes(source.read())
            paths[take] = target
    return paths


def scalecar_motion_to_qpos(npz_path: str | Path, qpos_names: list[str]) -> np.ndarray:
    """Build any4hdmi qpos from a Scalecar NPZ: world pelvis pose as the root, joints by name."""
    with np.load(npz_path, allow_pickle=False) as data:
        missing = [key for key in _REQUIRED_KEYS if key not in data.files]
        if missing:
            raise KeyError(f"{npz_path} is missing Scalecar motion keys {missing}")
        fps = float(data["fps"])
        joint_names = [str(name) for name in data["robot_joint_names"]]
        body_names = [str(name) for name in data["robot_body_names"]]
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float64)

    if abs(fps / SCALECAR_FPS - 1.0) > 1e-8:
        raise ValueError(f"{npz_path}: expected {SCALECAR_FPS} Hz, got fps={fps}")
    if ROOT_BODY_NAME not in body_names:
        raise KeyError(f"{npz_path}: robot_body_names has no {ROOT_BODY_NAME!r}")
    if tuple(qpos_names[:7]) != ROOT_QPOS_NAMES:
        raise ValueError(f"qpos must start with a floating root, got {qpos_names[:7]}")
    target_joints = list(qpos_names[7:])
    if sorted(target_joints) != sorted(joint_names):
        raise ValueError(
            f"{npz_path}: joint names differ from the MJCF; "
            f"only in NPZ {sorted(set(joint_names) - set(target_joints))}, "
            f"only in MJCF {sorted(set(target_joints) - set(joint_names))}"
        )

    frames = joint_pos.shape[0]
    root = body_names.index(ROOT_BODY_NAME)
    pelvis_pos = body_pos[:, root]
    pelvis_quat = body_quat[:, root]
    if frames == 0 or joint_pos.shape != (frames, len(joint_names)):
        raise ValueError(f"{npz_path}: joint_pos has shape {joint_pos.shape}")
    if pelvis_pos.shape != (frames, 3) or pelvis_quat.shape != (frames, 4):
        raise ValueError(f"{npz_path}: body_pos_w/body_quat_w frame count differs from joint_pos")
    if not all(np.isfinite(a).all() for a in (joint_pos, pelvis_pos, pelvis_quat)):
        raise ValueError(f"{npz_path}: non-finite motion values")
    if not np.allclose(np.linalg.norm(pelvis_quat, axis=-1), 1.0, atol=1e-4):
        raise ValueError(f"{npz_path}: pelvis body_quat_w must be unit WXYZ quaternions")

    qpos = np.empty((frames, len(qpos_names)), dtype=np.float64)
    qpos[:, 0:3] = pelvis_pos
    qpos[:, 3:7] = pelvis_quat
    qpos[:, 7:] = joint_pos[:, [joint_names.index(name) for name in target_joints]]
    return qpos


def convert_motion_files(
    npz_paths: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    robot: str = "g1",
    source: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write each Scalecar NPZ as ``<output_dir>/motions/<take>.npz`` of an any4hdmi dataset.

    Motions already in the dataset are kept, and manifest.json is rewritten to
    cover every motion file present.
    """
    import mujoco
    from any4hdmi.core.format import MOTIONS_SUBDIR, save_motion, write_manifest
    from any4hdmi.utils.mjcf import qpos_names_from_model
    from sim2real.config.robots import get_robot_cfg
    from sim2real.config.robots.base import resolve_mjcf_root_body_name

    for take in npz_paths:
        if not _TAKE_NAME.fullmatch(take):
            raise ValueError(f"Invalid take name {take!r}: use letters, digits and '_' only")
    robot_cfg = get_robot_cfg(robot)
    mjcf_path = robot_cfg.resolve_mjcf_path()
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    root_body = resolve_mjcf_root_body_name(model)
    if root_body != ROOT_BODY_NAME:
        raise ValueError(f"{mjcf_path}: root body is {root_body!r}, expected {ROOT_BODY_NAME!r}")
    qpos_names = qpos_names_from_model(model)

    output_root = Path(output_dir).expanduser().resolve()
    motions_dir = output_root / MOTIONS_SUBDIR
    manifest_path = output_root / "manifest.json"
    takes_info: dict[str, Any] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        takes_info = dict(previous.get("source", {}).get("takes", {}))

    written = {}
    for take, npz_path in npz_paths.items():
        npz_path = Path(npz_path)
        qpos = scalecar_motion_to_qpos(npz_path, qpos_names)
        written[take] = save_motion(motions_dir / f"{take}.npz", qpos)
        takes_info[take] = {
            "source_npz": npz_path.name,
            "source_npz_sha256": _sha256_of(npz_path),
            "frames": int(qpos.shape[0]),
        }

    present = sorted(path.stem for path in motions_dir.glob("*.npz"))
    takes_info = {take: takes_info[take] for take in present if take in takes_info}
    frames_total = sum(info["frames"] for info in takes_info.values())
    write_manifest(
        output_root,
        dataset_name="scalecar_d1146",
        mjcf=robot_cfg.mjcf_path if robot_cfg.mjcf_path is not None else mjcf_path,
        timestep=1.0 / SCALECAR_FPS,
        qpos_names=qpos_names,
        num_motions=len(present),
        total_hours=frames_total / SCALECAR_FPS / 3600.0,
        source={
            "kind": "scalecar_v2_50hz_npz",
            "root": f"{ROOT_BODY_NAME} world pose from body_pos_w/body_quat_w (WXYZ), no re-centring",
            "joints": "copied by name from joint_pos",
            **dict(source or {}),
            "takes": takes_info,
        },
    )
    return written


@dataclass
class ConvertArgs:
    """Convert Scalecar takes from the verified ModelScope tar into an any4hdmi dataset."""

    takes: tuple[str, ...] = ("take1512", "take0595")
    """Take names, e.g. take1512 (the tar holds 1146 takes)."""
    tar: Path = DEFAULT_TAR
    sha256sums: Path = DEFAULT_SHA256SUMS
    output_dir: Path = DEFAULT_OUTPUT_DIR
    """any4hdmi dataset root; existing motions in it are kept."""
    robot: str = "g1"


def run_convert(args: ConvertArgs) -> dict[str, Path]:
    with tempfile.TemporaryDirectory(prefix="scalecar_takes_") as tmp:
        sources = extract_takes(args.tar, args.sha256sums, args.takes, tmp)
        written = convert_motion_files(
            sources,
            args.output_dir,
            robot=args.robot,
            source={
                "tar": args.tar.name,
                "tar_sha256": _expected_sha256(args.sha256sums, args.tar.name),
                "tar_member_dir": TAR_MOTIONS_DIR,
            },
        )
    for take, path in written.items():
        print(f"[convert] {take} -> {path}")
    print(f"[convert] manifest: {Path(args.output_dir).expanduser().resolve() / 'manifest.json'}")
    return written


def main() -> None:
    import tyro

    run_convert(tyro.cli(ConvertArgs))


if __name__ == "__main__":
    main()
