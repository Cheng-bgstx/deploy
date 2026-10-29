"""Reproduce the eight-second original-speed G1 forward-walking test clip."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from huggingface_hub import hf_hub_download

from sim2real.config.robots import G1_CFG
from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset

DATASET = "elijahgalahad/any4hdmi-g1-lafan"
REVISION = "7937b18a2cbe8a2d354a57524f25c7799134cf15"
SOURCE_FPS = 30


@dataclass
class Args:
    output_dir: str = "checkpoints/proteus/proteus_60/unitree_g1/motions/walk_forward_original"
    source_path: str | None = None


def main(args: Args):
    output = Path(args.output_dir)
    if (output / "manifest.json").exists() or (output / "motions/motion.npz").exists():
        raise FileExistsError(f"Refusing to overwrite existing motion artifacts: {output}")
    source = args.source_path or hf_hub_download(
        repo_id=DATASET, repo_type="dataset", revision=REVISION,
        filename="motions/walk1_subject1.npz",
    )
    with np.load(source, allow_pickle=False) as data:
        qpos = data["qpos"]
        if qpos.ndim != 2 or qpos.shape[1] != G1_CFG.qpos_size:
            raise ValueError("Source motion must contain G1 qpos with 36 coordinates")
        if len(qpos) < 11 * SOURCE_FPS + 1:
            raise ValueError("Source motion must include the complete seconds 3-11")
        clip = qpos[3 * SOURCE_FPS:11 * SOURCE_FPS + 1].copy()
    if not np.isfinite(clip).all():
        raise ValueError("Motion contains non-finite qpos")
    metadata = dict(
        dataset=DATASET, revision=REVISION, motion="walk1_subject1.npz",
        source_start_s=3.0, source_end_s=11.0, source_fps=SOURCE_FPS,
        playback_speed=1.0,
        status="protected test candidate; earlier Proteus G1 hardware test became unstable",
    )
    motion, manifest = write_single_motion_qpos_dataset(
        output, robot_cfg=G1_CFG, qpos=clip, fps=SOURCE_FPS,
        dataset_name="Proteus G1 forward walking at original speed", source=metadata,
    )
    print(f"Motion: {motion}\nManifest: {manifest}\nDuration: 8 s; playback speed: 1.0")


if __name__ == "__main__":
    main(tyro.cli(Args))
