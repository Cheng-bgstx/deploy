#!/usr/bin/env python3
"""Create a smooth synthetic OLI arm-wave reference for dynamic sim2sim smoke testing."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import OLI_CFG


def make_wave(output: Path, duration: float = 8.0):
    if duration < 6:
        raise ValueError('duration must be at least 6 seconds')
    output = output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Dataset already exists: {output}')
    model = mujoco.MjModel.from_xml_path(str(OLI_CFG.resolve_mjcf_path()))
    q0 = model.qpos0.copy()
    q0[:7] = OLI_CFG.default_qpos[:7]
    for i, name in enumerate(OLI_CFG.joint_names):
        q0[model.joint(name).qposadr[0]] = OLI_CFG.default_qpos[7 + i]
    t = np.arange(round(duration * 50)) / 50
    qpos = np.tile(q0, (len(t), 1))

    def smooth(x):
        x = np.clip(x, 0, 1)
        return x**3 * (10 - 15 * x + 6 * x**2)
    envelope = smooth((t - .5) / 1.5) * smooth((duration - .5 - t) / 1.5)
    wave = np.sin(2 * np.pi * .75 * (t - 2))
    targets = {
        'left_shoulder_pitch_joint': -1.15,
        'left_shoulder_roll_joint': .65 + .18 * wave,
        'left_shoulder_yaw_joint': -.25,
        'left_elbow_joint': -1.35 + .23 * wave,
        'left_wrist_pitch_joint': .1 + .22 * wave,
        'right_shoulder_pitch_joint': -.10,
    }
    for name, target in targets.items():
        adr = model.joint(name).qposadr[0]
        qpos[:, adr] = q0[adr] + envelope * (target - q0[adr])
    for name in OLI_CFG.joint_names:
        adr = model.joint(name).qposadr[0]
        lo, hi = model.joint(name).range
        if np.any((qpos[:, adr] < lo) | (qpos[:, adr] > hi)):
            raise ValueError(f'Reference exceeds joint limits: {name}')
    motion = save_motion(output / 'motions' / 'synthetic_arm_wave.npz', qpos)
    write_manifest(output, dataset_name='oli_synthetic_arm_wave', mjcf=OLI_CFG.resolve_mjcf_path(),
        timestep=.02, qpos_names=qpos_names_from_model(model), num_motions=1,
        source={'kind': 'synthetic dynamic reference; not motion capture',
                'action': 'raise left arm, wave, lower arm', 'duration_s': duration,
                'generator': 'scripts/oli/make_wave_motion.py', 'wave_frequency_hz': .75},
        total_hours=duration / 3600)
    print(motion)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/oli/wave_motion'))
    p.add_argument('--duration', type=float, default=8)
    args = p.parse_args()
    make_wave(args.output, args.duration)


if __name__ == '__main__':
    main()
