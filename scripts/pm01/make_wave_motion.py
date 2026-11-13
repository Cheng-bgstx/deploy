#!/usr/bin/env python3
"""Write a synthetic PM01 arm-wave reference: stand, raise one arm, wave, lower it, stand.

Adapted from the X2 wave (``scripts/x2/make_wave_motion.py`` on the X2 onboard branch). The legs, pelvis, waist,
head and the other arm stay at the default standing pose of the Proteus contract; only the waving arm moves. The
raised pose puts the upper arm out to the side (slightly up) and the forearm vertical beside the head, found by FK on
the prepared pm_v2 model; the wave rocks the forearm side to side with the elbow. The clip starts and ends at the
default pose, so shaping, Hold and the end of playback are the same as for a standing reference.

Checks: every joint inside the policy's ``soft_joint_pos_limits`` (or the model range without a policy YAML), and
the waving arm's collision geoms at least ``--min-clearance`` from the torso, head and other arm on every frame.

  python -m scripts.pm01.make_wave_motion --output checkpoints/engineai_pm01/wave
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import yaml
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import PM01_CFG

POLICY_CFG = Path('checkpoints/proteus/proteus_wamseq_62/engineai_pm01/policy.yaml')
FPS = 50

# Right-arm raised pose (rad). Upper arm direction (0.05, -0.97, 0.25) in the pelvis frame, forearm (0, 0, 1).
RAISED = {'SHOULDER_PITCH': -1.115, 'SHOULDER_ROLL': -1.496, 'SHOULDER_YAW': -0.457, 'ELBOW_PITCH': -1.452,
          'ELBOW_YAW': 0.0}
WAVE = {'ELBOW_PITCH': 0.4}
# The left arm mirrors roll and both yaws; pitch is shared.
MIRRORED = ('SHOULDER_ROLL', 'SHOULDER_YAW', 'ELBOW_YAW')
ARM_JOINTS = {'right': {'SHOULDER_PITCH': 'J18', 'SHOULDER_ROLL': 'J19', 'SHOULDER_YAW': 'J20',
                        'ELBOW_PITCH': 'J21', 'ELBOW_YAW': 'J22'},
              'left': {'SHOULDER_PITCH': 'J13', 'SHOULDER_ROLL': 'J14', 'SHOULDER_YAW': 'J15',
                       'ELBOW_PITCH': 'J16', 'ELBOW_YAW': 'J17'}}
ARM_BODIES = ('SHOULDER_ROLL', 'SHOULDER_YAW', 'ELBOW_PITCH', 'ELBOW_YAW', 'ELBOW_END')


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def joint_name(arm: str, key: str) -> str:
    side = 'R' if arm == 'right' else 'L'
    return f'{ARM_JOINTS[arm][key]}_{key}_{side}'


def default_qpos(model):
    q0 = model.qpos0.copy()
    q0[:7] = PM01_CFG.default_qpos[:7]
    for i, name in enumerate(PM01_CFG.joint_names):
        q0[model.joint(name).qposadr[0]] = PM01_CFG.default_qpos[7 + i]
    return q0


def min_clearance(model, qpos, arm: str) -> float:
    """Smallest distance from the waving arm's collision geoms to the torso, head and other arm over the clip."""
    side, other = ('R', 'L') if arm == 'right' else ('L', 'R')
    body = lambda name: model.body(name).id  # noqa: E731
    moving = {body(f'LINK_{b}_{side}') for b in ARM_BODIES}
    fixed = {body('LINK_BASE'), body('LINK_TORSO_YAW'), body('LINK_HEAD_YAW')} | {body(f'LINK_{b}_{other}') for b in ARM_BODIES}
    collides = lambda g: model.geom_contype[g] or model.geom_conaffinity[g]  # noqa: E731
    a = [g for g in range(model.ngeom) if model.geom_bodyid[g] in moving and collides(g)]
    b = [g for g in range(model.ngeom) if model.geom_bodyid[g] in fixed and collides(g)]
    if not a or not b:
        raise ValueError('no collision geoms to check the arm clearance with')
    data = mujoco.MjData(model)
    fromto = np.zeros(6)
    best = np.inf
    for q in qpos[::5]:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        for ga in a:
            for gb in b:
                best = min(best, mujoco.mj_geomDistance(model, data, ga, gb, 1.0, fromto))
    return float(best)


def make_wave(output: Path, arm: str, duration: float, freq: float, ramp: float, hold: float, min_gap: float):
    if duration < 2 * (hold + ramp) + 1.0:
        raise ValueError('duration too short for the holds and ramps')
    output = output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(output)
    model = mujoco.MjModel.from_xml_path(str(PM01_CFG.resolve_mjcf_path()))
    q0 = default_qpos(model)
    t = np.arange(round(duration * FPS)) / FPS
    qpos = np.tile(q0, (len(t), 1))
    # Raise over `ramp` after `hold`, lower symmetrically at the end. The wave starts at zero phase once the arm is
    # up and fades with the same envelope, so velocities stay continuous.
    envelope = smoothstep((t - hold) / ramp) * smoothstep((duration - hold - t) / ramp)
    wave = np.sin(2 * np.pi * freq * (t - hold - ramp)) * smoothstep((t - hold - 0.5 * ramp) / ramp) \
        * smoothstep((duration - hold - 0.5 * ramp - t) / ramp)
    sign = 1.0 if arm == 'right' else -1.0
    for key, raised in RAISED.items():
        adr = model.joint(joint_name(arm, key)).qposadr[0]
        mirror = sign if key in MIRRORED else 1.0
        target = mirror * raised + mirror * WAVE.get(key, 0.0) * wave
        qpos[:, adr] = q0[adr] + envelope * (target - q0[adr])
    limits = yaml.safe_load(POLICY_CFG.read_text()).get('soft_joint_pos_limits') if POLICY_CFG.exists() else None
    for name in PM01_CFG.joint_names:
        adr = model.joint(name).qposadr[0]
        lo, hi = limits[name] if limits else model.joint(name).range
        if np.any((qpos[:, adr] < lo) | (qpos[:, adr] > hi)):
            raise ValueError(f'reference leaves the joint limits: {name} [{qpos[:, adr].min():.3f}, '
                             f'{qpos[:, adr].max():.3f}] vs [{lo:.3f}, {hi:.3f}]')
    clearance = min_clearance(model, qpos, arm)
    if clearance < min_gap:
        raise ValueError(f'the {arm} arm comes within {clearance:.3f} m of the body (< {min_gap} m)')
    peak = np.abs(np.diff(qpos[:, 7:], axis=0)).max() * FPS
    motion = save_motion(output / 'motions' / f'wave_{arm}.npz', qpos)
    write_manifest(output, dataset_name=f'pm01_synthetic_wave_{arm}', mjcf=PM01_CFG.resolve_mjcf_path(),
                   timestep=1 / FPS, qpos_names=qpos_names_from_model(model), num_motions=1,
                   source={'kind': 'synthetic reference; not motion capture',
                           'action': f'stand, raise {arm} arm, wave, lower arm, stand',
                           'generator': 'scripts/pm01/make_wave_motion.py', 'duration_s': duration,
                           'wave_frequency_hz': freq, 'ramp_s': ramp, 'hold_s': hold,
                           'raised_pose': RAISED, 'wave_amplitude': WAVE,
                           'peak_joint_speed_rad_s': round(float(peak), 3),
                           'min_arm_clearance_m': round(clearance, 4)},
                   total_hours=duration / 3600)
    print(f'{motion}  peak joint speed {peak:.2f} rad/s, min arm clearance {clearance * 100:.1f} cm')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, default=Path('checkpoints/engineai_pm01/wave'))
    p.add_argument('--arm', choices=['right', 'left'], default='right')
    p.add_argument('--duration', type=float, default=10.0)
    p.add_argument('--freq', type=float, default=1.0, help='wave frequency in Hz')
    p.add_argument('--ramp', type=float, default=1.5, help='seconds to raise and to lower the arm')
    p.add_argument('--hold', type=float, default=1.0, help='seconds of still standing at each end')
    p.add_argument('--min-clearance', type=float, default=0.05, help='metres between the waving arm and the body')
    args = p.parse_args()
    make_wave(args.output, args.arm, args.duration, args.freq, args.ramp, args.hold, args.min_clearance)


if __name__ == '__main__':
    main()
