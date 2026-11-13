#!/usr/bin/env python3
"""Shorten the retargeted LAFAN forward walk (clip 11) to a few steps that end standing.

Clip 11 stands for about 3.5 s, then walks 3.4 m straight forward and slows down in its last second, but it ends
with the rear heel lifting for the next step, so a tracker that holds that frame keeps walking. This keeps the start
and the slow-down and drops the middle: it splices the walk at frame i onto the slow-down at frame j, where i and j
are the pair with the closest gait phase (joint positions and velocities, ankle heights). The root after the cut is
shifted so it continues from frame i, and a short crossfade hides the remaining joint mismatch. The clip then ends
on the last frame where both feet are flat and the centre of mass is closest to the midpoint between them; the last
--slow seconds are time-warped so playback speed falls smoothly to zero there, and that frame is held.

  python scripts/x2/make_short_walk.py --cut-after 6.2 --output outputs/x2/lafan/15_walk_short
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R, Slerp
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import X2_CFG

SOURCE = Path('outputs/x2/lafan/11_stand_walk_straight')
POLICY_CFG = Path('checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc.yaml')


def kinematics(model, qpos):
    """Ankle positions (N, 2, 3) and whole-body centre of mass (N, 3) for each frame."""
    data = mujoco.MjData(model)
    ids = [model.body(f'{s}_ankle_roll_link').id for s in ('left', 'right')]
    ankles, com = [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        ankles.append(data.xpos[ids].copy())
        com.append(data.subtree_com[0].copy())
    return np.array(ankles), np.array(com)


def slow_to_stop(seq, frames):
    """Time-warp the last `frames` source frames of seq so playback speed falls smoothly from 1 to 0 at its end."""
    if frames <= 0:
        return seq
    start = len(seq) - 1 - frames
    # Rate 1 - smoothstep(x) over 2 * frames output frames covers exactly `frames` source frames.
    x = np.arange(1, 2 * frames + 1) / (2 * frames)
    pos = start + 2 * frames * (x - 2.5 * x**4 + 3 * x**5 - x**6)
    lo = np.floor(pos).astype(int).clip(max=len(seq) - 2)
    w = (pos - lo)[:, None]
    out = (1 - w) * seq[lo] + w * seq[lo + 1]
    out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True)
    return np.concatenate([seq[:start + 1], out])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--cut-after', type=float, default=6.2, help='approximate source time (s) to leave the walk')
    p.add_argument('--search', type=float, default=0.5, help='search +- this many seconds around --cut-after')
    p.add_argument('--stop-from', type=float, default=8.6, help='earliest source time (s) to join the deceleration')
    p.add_argument('--stop-to', type=float, default=9.4, help='latest source time (s) to join the deceleration')
    p.add_argument('--blend', type=float, default=0.2, help='crossfade seconds after the splice')
    p.add_argument('--stop-at', type=float, help='source time (s) of the final frame (default: chosen automatically)')
    p.add_argument('--slow', type=float, default=1.0, help='seconds over which playback slows to zero at the end')
    p.add_argument('--hold', type=float, default=3.0, help='seconds to hold the final standing frame')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    manifest = json.loads((args.source / 'manifest.json').read_text())
    dt = manifest['timestep']
    fps = round(1 / dt)
    src = np.load(next((args.source / 'motions').glob('*.npz')))['qpos'].astype(np.float64)
    model = mujoco.MjModel.from_xml_path(str(X2_CFG.resolve_mjcf_path()))
    ankles, com = kinematics(model, src)
    feet = ankles[:, :, 2]
    vel = np.gradient(src[:, 7:], dt, axis=0)
    root_vel = np.gradient(src[:, :2], dt, axis=0)

    lo, hi = round((args.cut_after - args.search) * fps), round((args.cut_after + args.search) * fps)
    jlo, jhi = round(args.stop_from * fps), min(round(args.stop_to * fps), len(src) - 2)
    best = None
    for i in range(lo, hi + 1):
        for j in range(jlo, jhi + 1):
            cost = (np.linalg.norm(src[i, 7:] - src[j, 7:]) + 0.1 * np.linalg.norm(vel[i] - vel[j])
                    + 10 * np.linalg.norm(feet[i] - feet[j]) + np.linalg.norm(root_vel[i] - root_vel[j]))
            if best is None or cost < best[0]:
                best = (cost, i, j)
    cost, i, j = best

    # Final frame: both feet flat, centre of mass nearest the midpoint between the ankles.
    if args.stop_at is not None:
        stop = round(args.stop_at * fps)
    else:
        flat = np.all(feet < feet.min(axis=0) + 0.012, axis=1)
        offset = np.linalg.norm(com[:, :2] - ankles[:, :, :2].mean(axis=1), axis=1)
        candidates = [k for k in range(j + 1 + round(args.blend * fps), len(src)) if flat[k]]
        if not candidates:
            raise ValueError('no frame after the splice has both feet flat')
        stop = min(candidates, key=lambda k: offset[k])
    if stop <= j + round(args.slow * fps / 2):
        raise ValueError(f'stop frame {stop / fps:.2f}s is too close to the splice at {j / fps:.2f}s')

    # Splice: frames [0, i] then (j, stop], with the tail's root shifted to continue from frame i.
    head, tail = src[:i + 1].copy(), src[j + 1:stop + 1].copy()
    shift = src[i, :2] - src[j, :2]
    tail[:, :2] += shift
    # Heading: rotate the tail about frame i's root so its yaw continues from frame i.
    yaw_i = R.from_quat(src[i, [4, 5, 6, 3]]).as_euler('zyx')[0]
    yaw_j = R.from_quat(src[j, [4, 5, 6, 3]]).as_euler('zyx')[0]
    rot = R.from_euler('z', yaw_i - yaw_j)
    tail[:, :2] = rot.apply(np.c_[tail[:, :2] - src[i, :2], np.zeros(len(tail))])[:, :2] + src[i, :2]
    tail[:, [4, 5, 6, 3]] = (rot * R.from_quat(tail[:, [4, 5, 6, 3]])).as_quat()
    # Crossfade the joints and root height/orientation from the walk's continuation into the tail.
    n = min(round(args.blend * fps), len(tail))
    if n and i + 1 + n <= len(src):
        cont = src[i + 1:i + 1 + n]
        w = (np.arange(1, n + 1) / (n + 1))[:, None]
        w = w * w * (3 - 2 * w)
        tail[:n, 7:] = (1 - w) * cont[:, 7:] + w * tail[:n, 7:]
        tail[:n, 2] = (1 - w[:, 0]) * cont[:, 2] + w[:, 0] * tail[:n, 2]
        for k in range(n):
            pair = R.from_quat([cont[k, [4, 5, 6, 3]], tail[k, [4, 5, 6, 3]]])
            tail[k, [4, 5, 6, 3]] = Slerp([0, 1], pair)([w[k, 0]]).as_quat()[0]
    qpos = slow_to_stop(np.concatenate([head, tail]), round(args.slow * fps / 2))
    qpos = np.concatenate([qpos, np.repeat(qpos[-1:], round(args.hold * fps), axis=0)])

    # The retargeted source already crosses a few soft limits (hip roll, wrists; the runtime clips the targets), so
    # only reject joints that leave both the soft limits and the source clip's own range.
    limits = yaml.safe_load(POLICY_CFG.read_text())['soft_joint_pos_limits']
    for name in X2_CFG.joint_names:
        adr = model.joint(name).qposadr[0]
        lo_, hi_ = min(limits[name][0], src[:, adr].min()), max(limits[name][1], src[:, adr].max())
        if np.any((qpos[:, adr] < lo_ - 1e-6) | (qpos[:, adr] > hi_ + 1e-6)):
            raise ValueError(f'{name} leaves the soft limits [{qpos[:, adr].min():.3f}, {qpos[:, adr].max():.3f}] '
                             f'vs soft limits and source range [{lo_:.3f}, {hi_:.3f}]')
    peak = float(np.abs(np.diff(qpos[:, 7:], axis=0)).max() * fps)
    distance = float(np.linalg.norm(qpos[-1, :2] - qpos[0, :2]))
    yaw = np.degrees(np.unwrap(R.from_quat(qpos[:, [4, 5, 6, 3]]).as_euler('zyx')[:, 0]))
    out = args.output.expanduser().resolve()
    if (out / 'manifest.json').exists():
        raise FileExistsError(out)
    save_motion(out / 'motions' / 'walk_short.npz', qpos)
    source = dict(manifest['source'])
    source.update({'derived_from': str(args.source), 'generator': 'scripts/x2/make_short_walk.py',
                   'splice_source_frames': [int(i), int(j)], 'splice_source_s': [i / fps, j / fps],
                   'splice_cost': round(float(cost), 3), 'blend_s': args.blend, 'stop_source_s': stop / fps,
                   'slow_to_stop_s': args.slow, 'final_hold_s': args.hold,
                   'forward_distance_m': round(distance, 3), 'yaw_change_deg': round(float(yaw[-1] - yaw[0]), 1),
                   'peak_joint_speed_rad_s': round(peak, 3)})
    write_manifest(out, dataset_name='x2_lafan_walk1_short', mjcf=X2_CFG.resolve_mjcf_path(), timestep=dt,
                   qpos_names=qpos_names_from_model(model), num_motions=1, source=source,
                   total_hours=len(qpos) * dt / 3600)
    print(f'splice {i / fps:.2f}s -> {j / fps:.2f}s (cost {cost:.2f}), stop at {stop / fps:.2f}s; {len(qpos) * dt:.1f} s, walks {distance:.2f} m, '
          f'yaw change {yaw[-1] - yaw[0]:+.1f} deg, peak joint speed {peak:.2f} rad/s -> {out}')


if __name__ == '__main__':
    main()
