#!/usr/bin/env python3
"""Write a synthetic X2 arm-wave reference: stand, raise one arm, wave, lower it, stand.

The legs, pelvis and waist stay at the default standing pose; only the waving arm moves. Every joint stays inside
the deployed soft limits (training ∩ vendor range minus 5%). The wrist pitch/roll joints are left at their defaults
because they read exactly 0 on the X2 Basic.

  python scripts/x2/make_wave_motion.py --output outputs/x2/wave/21_wave_right
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import X2_CFG

POLICY_CFG = Path('checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc.yaml')
FPS = 50

# Raised pose (right arm): upper arm out to the side, rotated so the forearm points straight up (a "hello" wave
# seen from the front). The wave rocks the forearm side to side with the elbow. The arm stays at least 12 cm from
# the torso and head over the whole clip (checked with mj_geomDistance on the vendor collision geoms).
RAISED = {'shoulder_pitch': -0.3, 'shoulder_roll': -1.3, 'shoulder_yaw': -1.3, 'elbow': -1.5}
WAVE = {'elbow': 0.4}
MIRRORED = ('shoulder_roll', 'shoulder_yaw')


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def make_wave(output: Path, arm: str, duration: float, freq: float, ramp: float, hold: float):
    if duration < 2 * (hold + ramp) + 1.0:
        raise ValueError('duration too short for the holds and ramps')
    output = output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(output)
    model = mujoco.MjModel.from_xml_path(str(X2_CFG.resolve_mjcf_path()))
    q0 = model.qpos0.copy()
    q0[:7] = X2_CFG.default_qpos[:7]
    for i, name in enumerate(X2_CFG.joint_names):
        q0[model.joint(name).qposadr[0]] = X2_CFG.default_qpos[7 + i]
    t = np.arange(round(duration * FPS)) / FPS
    qpos = np.tile(q0, (len(t), 1))
    # Raise over `ramp` after `hold`, lower symmetrically at the end. The wave starts at zero phase once the arm is
    # up and fades with the same envelope, so velocities stay continuous.
    envelope = smoothstep((t - hold) / ramp) * smoothstep((duration - hold - t) / ramp)
    wave = np.sin(2 * np.pi * freq * (t - hold - ramp)) * smoothstep((t - hold - 0.5 * ramp) / ramp) \
        * smoothstep((duration - hold - 0.5 * ramp - t) / ramp)
    sign = 1.0 if arm == 'right' else -1.0
    for key, raised in RAISED.items():
        name = f'{arm}_{key}_joint'
        adr = model.joint(name).qposadr[0]
        mirror = sign if key in MIRRORED else 1.0
        target = mirror * raised + mirror * WAVE.get(key, 0.0) * wave
        qpos[:, adr] = q0[adr] + envelope * (target - q0[adr])
    limits = yaml.safe_load(POLICY_CFG.read_text())['soft_joint_pos_limits'] if POLICY_CFG.exists() else None
    for name in X2_CFG.joint_names:
        adr = model.joint(name).qposadr[0]
        lo, hi = limits[name] if limits else model.joint(name).range
        if np.any((qpos[:, adr] < lo) | (qpos[:, adr] > hi)):
            raise ValueError(f'reference leaves the soft limits: {name} [{qpos[:, adr].min():.3f}, '
                             f'{qpos[:, adr].max():.3f}] vs [{lo:.3f}, {hi:.3f}]')
    peak = np.abs(np.diff(qpos[:, 7:], axis=0)).max() * FPS
    motion = save_motion(output / 'motions' / f'wave_{arm}.npz', qpos)
    write_manifest(output, dataset_name=f'x2_synthetic_wave_{arm}', mjcf=X2_CFG.resolve_mjcf_path(), timestep=1 / FPS,
                   qpos_names=qpos_names_from_model(model), num_motions=1,
                   source={'kind': 'synthetic reference; not motion capture',
                           'action': f'stand, raise {arm} arm, wave, lower arm, stand',
                           'generator': 'scripts/x2/make_wave_motion.py', 'duration_s': duration,
                           'wave_frequency_hz': freq, 'ramp_s': ramp, 'hold_s': hold,
                           'raised_pose': RAISED, 'wave_amplitude': WAVE,
                           'peak_joint_speed_rad_s': round(float(peak), 3)},
                   total_hours=duration / 3600)
    print(f'{motion}  peak joint speed {peak:.2f} rad/s')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, default=Path('outputs/x2/wave/21_wave_right'))
    p.add_argument('--arm', choices=['right', 'left'], default='right')
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--freq', type=float, default=1.0, help='wave frequency in Hz')
    p.add_argument('--ramp', type=float, default=1.5, help='seconds to raise and to lower the arm')
    p.add_argument('--hold', type=float, default=1.0, help='seconds of still standing at each end')
    args = p.parse_args()
    make_wave(args.output, args.arm, args.duration, args.freq, args.ramp, args.hold)


if __name__ == '__main__':
    main()
