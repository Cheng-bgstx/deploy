"""Convert the Proteus ScaleBridge test clips (G1 / Galbot ET1) into sim2real qpos datasets.

Proteus ships ten short clips per robot under ScaleBridge/scalebridge/data/motion/<robot>/proteus/
(``joint_pos`` in IsaacLab order with ``joint_names``, all MJCF bodies in ``body_pos_w``/``body_quat_w``
with ``body_names``, ``fps``). The root body pose and the named joints are reordered into the MJCF
qpos of the sim2real robot and written as one any4hdmi dataset per clip:
<output-root>/<clip>/{manifest.json, motions/motion.npz}.

Of the walks, only 10_walk_b walks forward; 07, 08 and 09 walk backwards relative to the pelvis.

    python scripts/prepare_proteus_test_motions.py --robot galbot_et1
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

from sim2real.config.robots import get_robot_cfg
from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
# sim2real robot -> (Proteus contract, ScaleBridge motion directory)
ROBOTS = {
    "g1": ("unitree_g1", "g1_29dof"),
    "galbot_et1": ("galbot_et1", "galbot_et1"),
}


@dataclass
class Args:
    robot: str = "galbot_et1"
    proteus_root: Path = REPO_ROOT.parent / "Proteus"
    output_root: Path | None = None
    """Defaults to checkpoints/proteus/proteus_60/<contract>/motions."""
    clips: tuple[str, ...] = ()
    """Clip stems such as 01_idle; all clips when empty."""


def clip_qpos(path: Path, mjcf_joint_names: list[str], root_body: str) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        joint_names = [str(n) for n in data["joint_names"]]
        body_names = [str(n) for n in data["body_names"]]
        missing = sorted(set(mjcf_joint_names) - set(joint_names))
        if missing or body_names[0] != root_body:
            raise ValueError(f"{path.name}: missing joints {missing} or root {body_names[0]!r} != {root_body!r}")
        order = [joint_names.index(name) for name in mjcf_joint_names]
        root_quat = data["body_quat_w"][:, 0]  # w, x, y, z
        qpos = np.concatenate([data["body_pos_w"][:, 0], root_quat, data["joint_pos"][:, order]], axis=1)
        fps = float(data["fps"])
    norm = np.linalg.norm(root_quat, axis=1)
    if not np.isfinite(qpos).all() or np.abs(norm - 1).max() > 1e-3:
        raise ValueError(f"{path.name}: non-finite qpos or non-unit root quaternion")
    return qpos, fps


def main(args: Args) -> None:
    contract, motion_dir = ROBOTS[args.robot]
    robot_cfg = get_robot_cfg(args.robot)
    model = mujoco.MjModel.from_xml_path(str(robot_cfg.resolve_mjcf_path()))
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Expected the MJCF free joint first")
    mjcf_joint_names = [model.joint(i).name for i in range(1, model.njnt)]
    root_body = model.body(int(model.jnt_bodyid[0])).name
    source_dir = args.proteus_root / "ScaleBridge/scalebridge/data/motion" / motion_dir / "proteus"
    output_root = args.output_root or REPO_ROOT / "checkpoints/proteus/proteus_60" / contract / "motions"
    paths = sorted(source_dir.glob("*.npz"))
    if args.clips:
        paths = [p for p in paths if p.stem in args.clips]
        if len(paths) != len(args.clips):
            raise FileNotFoundError(f"Clips {args.clips} not all found in {source_dir}")
    for path in paths:
        output = output_root / path.stem
        if (output / "manifest.json").exists():
            print(f"{path.stem}: exists, skipped")
            continue
        qpos, fps = clip_qpos(path, mjcf_joint_names, root_body)
        write_single_motion_qpos_dataset(
            output, robot_cfg=robot_cfg, qpos=qpos, fps=fps,
            dataset_name=f"Proteus {contract} test motion {path.stem}",
            source=dict(proteus_file=str(path.relative_to(args.proteus_root)), fps=fps),
        )  # fmt: skip
        print(f"{path.stem}: {len(qpos) / fps:.1f} s -> {output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
