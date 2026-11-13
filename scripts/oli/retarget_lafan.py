#!/usr/bin/env python3
"""Retarget a real LAFAN1 BVH clip to OLI using LimX's official GMR, without a viewer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model
from tqdm import tqdm

from sim2real.config.robots import OLI_CFG


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmr-root', type=Path, default=Path('../GMR-limx'))
    p.add_argument('--bvh', required=True, type=Path)
    p.add_argument('--start-s', type=float, default=0)
    p.add_argument('--duration', type=float, default=8)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.start_s < 0 or args.duration <= 0:
        raise ValueError('start-s must be non-negative and duration positive')
    output = args.output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Dataset already exists: {output}')
    sys.path.insert(0, str(args.gmr_root.resolve()))
    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.params import IK_CONFIG_DICT, ROBOT_XML_DICT
    from general_motion_retargeting.utils.lafan1 import load_bvh_file
    # Exactly the model verified against the Proteus contract; no second robot asset tree.
    ROBOT_XML_DICT['limx_oli_edu'] = OLI_CFG.resolve_mjcf_path()
    frames, height = load_bvh_file(str(args.bvh), format='lafan1')
    match = re.search(r'Frame Time:\s*([\d.]+)', args.bvh.read_text())
    if match is None:
        raise ValueError('BVH has no Frame Time')
    dt = float(match[1])
    start = round(args.start_s / dt)
    count = round(args.duration / dt)
    selected = frames[start:start + count]
    if len(selected) < 2:
        raise ValueError('Selected motion interval has fewer than two frames')
    retargeter = GeneralMotionRetargeting(src_human='bvh_lafan1', tgt_robot='limx_oli_edu',
        actual_human_height=height, verbose=False)
    for _ in range(20):
        retargeter.retarget(selected[0])
    qpos = np.stack([retargeter.retarget(frame) for frame in tqdm(selected, desc='LAFAN1 -> OLI')])
    raw = qpos.copy()
    # Place the lowest sole on the floor while preserving the source's foot swing.
    # GMR targets ankle frames; the physical sole extends below them.
    model = retargeter.model
    data = mujoco.MjData(model)
    feet = [model.geom(n).id for n in ('left_foot', 'right_foot')]
    shifts = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        bottoms = [data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
                   for g in feet]
        shift = -min(bottoms)
        q[2] += shift
        shifts.append(shift)
    qpos[:, :2] -= qpos[0, :2].copy()
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    if not np.isfinite(qpos).all():
        raise ValueError('Retargeting produced non-finite poses')
    motion = save_motion(output / 'motions' / f'{args.bvh.stem}.npz', qpos)
    np.savez_compressed(output / 'gmr_raw.npz', qpos=raw)
    config = Path(IK_CONFIG_DICT['bvh_lafan1']['limx_oli_edu'])
    source = {'kind': 'LAFAN1 human mocap retargeted by official limxdynamics/GMR',
        'bvh': str(args.bvh.resolve()), 'bvh_sha256': hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
        'start_s': args.start_s, 'requested_duration_s': args.duration,
        'gmr_commit': subprocess.check_output(['git', '-C', str(args.gmr_root), 'rev-parse', 'HEAD'], text=True).strip(),
        'ik_config': config.name, 'ik_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'grounding': 'per-frame vertical shift of lowest sole box to z=0',
        'ground_shift_min_m': float(min(shifts)), 'ground_shift_max_m': float(max(shifts)),
        'root_xy': 'first frame recentered to zero'}
    write_manifest(output, dataset_name=f'oli_lafan_{args.bvh.stem}', mjcf=OLI_CFG.resolve_mjcf_path(),
        timestep=dt, qpos_names=qpos_names_from_model(model), num_motions=1,
        source=source, total_hours=len(qpos) * dt / 3600)
    print(json.dumps(source, indent=2))
    print(motion)


if __name__ == '__main__':
    main()
