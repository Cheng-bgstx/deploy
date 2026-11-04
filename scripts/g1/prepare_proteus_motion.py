"""Reproduce the original-speed G1 forward-walking test clips, each from a stand to a double-support stand."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from huggingface_hub import hf_hub_download

from sim2real.config.robots import G1_CFG
from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset

DATASET = "elijahgalahad/any4hdmi-g1-lafan"
REVISION = "7937b18a2cbe8a2d354a57524f25c7799134cf15"
SOURCE_FPS = 30
MOTION_ROOT = "checkpoints/proteus/proteus_60/unitree_g1/motions"
# name -> (source motion, first frame, last frame). Hold freezes the last frame, so every clip ends standing on
# both feet. A scan of the dataset's 12 walk motions found four stand -> straight forward walk -> stand segments;
# in simulation (WAM actor, local forcing, waist 100/2.5, 0.15 m/s pushes) walk_straight_short drifted 0.05 m
# sideways on average against 0.53 m for walk_forward_to_stand.
CLIPS = {
    # 5.1 s, 2.2 m at 0.52 m/s, straight.
    "walk_straight_short": ("walk3_subject1", 6670, 6823),
    # 8.6 s, 3.6 m at 0.47 m/s. Ends at 11.63 s, stopped before the turn at 12.2 s; the former 11 s end froze the
    # right foot mid-step.
    "walk_forward_to_stand": ("walk1_subject1", 90, 349),
    # In-place dances from a stand to a stand, chosen by scanning the 8 dance motions for windows within 0.6 m and
    # without flight and with arm movement in every second, then simulated with the measured joint friction and
    # 30 ms command delay (no falls in 3 runs).
    # dance_in_place: 6 s, feet down, the richest arm movement (0.08 m range).
    "dance_in_place": ("dance1_subject3", 345, 525),
    # dance_steps: 8 s with two steps (0.21 m range); Hold afterwards moved up to 0.55 m in simulation.
    "dance_steps": ("dance1_subject3", 165, 405),
}


@dataclass
class Args:
    clip: Literal["walk_straight_short", "walk_forward_to_stand", "dance_in_place", "dance_steps"] = \
        "walk_straight_short"
    output_dir: str | None = None
    """Defaults to MOTION_ROOT/<clip>."""
    source_path: str | None = None


def main(args: Args):
    name, start, end = CLIPS[args.clip]
    output = Path(args.output_dir or f"{MOTION_ROOT}/{args.clip}")
    if (output / "manifest.json").exists() or (output / "motions/motion.npz").exists():
        raise FileExistsError(f"Refusing to overwrite existing motion artifacts: {output}")
    source = args.source_path or hf_hub_download(
        repo_id=DATASET, repo_type="dataset", revision=REVISION,
        filename=f"motions/{name}.npz",
    )
    with np.load(source, allow_pickle=False) as data:
        qpos = data["qpos"]
        if qpos.ndim != 2 or qpos.shape[1] != G1_CFG.qpos_size:
            raise ValueError("Source motion must contain G1 qpos with 36 coordinates")
        if len(qpos) <= end:
            raise ValueError(f"Source motion must include frames {start}-{end}")
        clip = qpos[start:end + 1].copy()
    if not np.isfinite(clip).all():
        raise ValueError("Motion contains non-finite qpos")
    duration = (len(clip) - 1) / SOURCE_FPS
    metadata = dict(
        dataset=DATASET, revision=REVISION, motion=f"{name}.npz",
        source_start_s=start / SOURCE_FPS, source_end_s=end / SOURCE_FPS,
        source_fps=SOURCE_FPS, playback_speed=1.0,
        status="protected test candidate; ends standing on both feet",
    )
    motion, manifest = write_single_motion_qpos_dataset(
        output, robot_cfg=G1_CFG, qpos=clip, fps=SOURCE_FPS,
        dataset_name=f"Proteus G1 {args.clip} at original speed", source=metadata,
    )
    print(f"Motion: {motion}\nManifest: {manifest}\nDuration: {duration:.2f} s; playback speed: 1.0")


if __name__ == "__main__":
    main(tyro.cli(Args))
