"""Slow down the end of a qpos motion dataset and append a still end, for a calmer stop.

ET1 on 11_stand_walk_straight squats while it closes its feet at original speed; slowing everything
after 10.4 s (walking has ended) by 2x and holding the last frame for 1 s lets it stop standing:

    python scripts/et1/slow_motion_tail.py \
        --source checkpoints/proteus/proteus_60/galbot_et1/motions/11_stand_walk_straight \
        --output checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from sim2real.config.robots import get_robot_cfg
from sim2real.teleop.qpos_dataset import write_single_motion_qpos_dataset


@dataclass
class Args:
    source: Path
    output: Path
    robot: str = "galbot_et1"
    after_s: float = 10.4
    factor: float = 2.0
    still_s: float = 1.0


def slerp(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    theta = np.arccos(dot)
    return (np.sin((1 - u) * theta) * a + np.sin(u * theta) * b) / np.sin(theta)


def slow_tail(qpos: np.ndarray, start: int, factor: float) -> np.ndarray:
    """Resample qpos[start:] at 1/factor speed: positions linearly, the root quaternion (3:7) by slerp."""
    tail = qpos[start:]
    n = len(tail)
    rows = []
    for x in np.linspace(0, n - 1, int(round((n - 1) * factor)) + 1):
        i = int(np.floor(x))
        j, u = min(i + 1, n - 1), x - i
        row = (1 - u) * tail[i] + u * tail[j]
        row[3:7] = slerp(tail[i, 3:7], tail[j, 3:7], u)
        rows.append(row)
    return np.concatenate([qpos[:start], np.asarray(rows)])


def main(args: Args) -> None:
    if args.factor < 1 or args.after_s < 0 or args.still_s < 0:
        raise ValueError("factor >= 1, after_s >= 0 and still_s >= 0 are required")
    if (args.output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    manifest = json.loads((args.source / "manifest.json").read_text())
    fps = round(1.0 / float(manifest["timestep"]))
    qpos = np.asarray(np.load(args.source / "motions" / "motion.npz")["qpos"], dtype=np.float64)
    start = int(round(args.after_s * fps))
    if not 0 <= start < len(qpos) - 1:
        raise ValueError(f"after_s {args.after_s} is outside the {len(qpos) / fps:.1f} s motion")
    out = slow_tail(qpos, start, args.factor)
    out = np.concatenate([out, np.repeat(out[-1:], int(round(args.still_s * fps)), axis=0)])
    write_single_motion_qpos_dataset(
        args.output, robot_cfg=get_robot_cfg(args.robot), qpos=out, fps=fps,
        dataset_name=f"{manifest['dataset_name']}, slowed {args.factor}x after {args.after_s} s, still end",
        source=dict(motion=str(args.source), after_s=args.after_s, factor=args.factor, still_s=args.still_s),
    )  # fmt: skip
    print(f"{args.output}: {len(qpos) / fps:.1f} s -> {len(out) / fps:.1f} s")


if __name__ == "__main__":
    main(tyro.cli(Args))
