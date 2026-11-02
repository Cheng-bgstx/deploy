#!/usr/bin/env python3
"""Retarget real LAFAN BVH to the X2 contract with a calibrated, bounded IK solve."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim2real.config.robots import X2_CFG


def load_frames(path):
    # Match the upstream bb1bbe4 LAFAN convention used to calibrate this table.
    # LimX's newer loader uses a different basis AND conjugates body rotations.
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh
    from general_motion_retargeting.utils.lafan_vendor import utils
    data = read_bvh(str(path))
    rotations, positions = utils.quat_fk(data.quats, data.pos, data.parents)
    basis = R.from_euler('x', 90, degrees=True)
    frames = []
    for t in range(len(positions)):
        frame = {name: [basis.apply(positions[t, j]) / 100,
                        (basis * R.from_quat(rotations[t, j], scalar_first=True)).as_quat(scalar_first=True)]
                 for j, name in enumerate(data.bones)}
        for side in ('Left', 'Right'):
            frame[f'{side}FootMod'] = [frame[f'{side}Foot'][0], frame[f'{side}Toe'][1]]
        frames.append(frame)
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gmr-root', type=Path, default=Path('../GMR-limx'))
    p.add_argument('--bvh', required=True, type=Path)
    p.add_argument('--start-s', type=float, default=0)
    p.add_argument('--duration', type=float, default=8)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.start_s < 0 or args.duration <= 0:
        raise ValueError('Invalid motion interval')
    output = args.output.resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(output)
    sys.path.insert(0, str(args.gmr_root.resolve()))
    from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
    from sim2real.teleop.x2_retarget import X2Retargeter
    config = Path(__file__).resolve().parents[2] / 'sim2real/teleop/configs/bvh_lafan1_to_x2.json'
    ROBOT_XML_DICT['lingxi_x2'] = X2_CFG.resolve_mjcf_path()
    IK_CONFIG_DICT.setdefault('bvh_lafan1', {})['lingxi_x2'] = config
    match = re.search(r'Frame Time:\s*([\d.]+)', args.bvh.read_text())
    if match is None:
        raise ValueError('Missing BVH Frame Time')
    dt = float(match[1])
    frames = load_frames(args.bvh)
    first, count = round(args.start_s / dt), round(args.duration / dt)
    selected = frames[first:first + count]
    if len(selected) < 2:
        raise ValueError('Motion interval has fewer than two frames')
    # Calibrated at 1.8 m, upstream LAFAN skeleton assumption is 1.75 m.
    ik = X2Retargeter('bvh_lafan1', 'lingxi_x2', actual_human_height=1.8**2 / 1.75, verbose=False)
    ik.reference_dt = dt
    for _ in range(20):
        ik.retarget(selected[0])
    qpos = np.stack([ik.retarget(frame) for frame in tqdm(selected, desc='LAFAN1 -> X2')])
    raw = qpos.copy()
    data = mujoco.MjData(ik.model)
    soles = [g for g in range(ik.model.ngeom)
             if ik.model.body(ik.model.geom_bodyid[g]).name in ('left_ankle_roll_link', 'right_ankle_roll_link')
             and ik.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and ik.model.geom_contype[g]]
    if not soles:
        raise ValueError('Prepared X2 model has no sole contact spheres')
    shifts = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(ik.model, data)
        bottoms = data.geom_xpos[soles, 2] - ik.model.geom_size[soles, 0]
        shift = -float(bottoms.min())
        q[2] += shift
        shifts.append(shift)
    qpos[:, :2] -= qpos[0, :2].copy()
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / 'gmr_raw.npz', qpos=raw)
    save_motion(output / 'motions' / f'{args.bvh.stem}.npz', qpos)
    source = {'kind': 'LAFAN1 human mocap retargeted to X2; not G1 joint playback',
              'bvh': str(args.bvh.resolve()), 'bvh_sha256': hashlib.sha256(args.bvh.read_bytes()).hexdigest(),
              'start_s': args.start_s, 'requested_duration_s': args.duration,
              'gmr_commit': subprocess.check_output(['git', '-C', str(args.gmr_root), 'rev-parse', 'HEAD'], text=True).strip(),
              'calibration_source': 'TaiE64/AgibotX2-Retargeter@bb582d97b43cd0df08953d3347fc4472dad8fa9e',
              'ik_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
              'lafan_frame_basis': 'upstream Rx(90) left multiplication, no conjugation',
              'height_normalization': '1.8 / 1.75', 'joint_speed_bound_rad_s': ik.max_joint_speed,
              'grounding': 'per-frame lowest vendor sole contact sphere to z=0',
              'ground_shift_min_m': min(shifts), 'ground_shift_max_m': max(shifts),
              'root_xy': 'first frame recentered to zero'}
    write_manifest(output, dataset_name=f'x2_lafan_{args.bvh.stem}', mjcf=X2_CFG.resolve_mjcf_path(),
                   timestep=dt, qpos_names=qpos_names_from_model(ik.model), num_motions=1,
                   source=source, total_hours=len(qpos) * dt / 3600)
    print(json.dumps(source, indent=2))


if __name__ == '__main__':
    main()
