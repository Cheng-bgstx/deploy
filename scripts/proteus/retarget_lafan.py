#!/usr/bin/env python3
"""Retarget several real LAFAN1 BVHs into one named any4hdmi motion library."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model
from tqdm import tqdm

from scripts.x2.retarget_lafan import load_frames
from sim2real.config.robots import get_robot_cfg
from sim2real.teleop.proteus_retarget import ProteusRetargeter


def retarget(robot, bvhs, output, start_s, duration):
    from general_motion_retargeting.params import ROBOT_XML_DICT, IK_CONFIG_DICT
    cfg = get_robot_cfg(robot)
    if output.exists():
        raise FileExistsError(f'Choose a new dataset directory: {output}')
    for path in bvhs:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = f'sim2real_{cfg.name}'
    profile = Path(__file__).resolve().parents[2] / f'sim2real/teleop/configs/bvh_lafan1_to_{cfg.name}.json'
    ROBOT_XML_DICT[target] = cfg.resolve_mjcf_path()
    IK_CONFIG_DICT.setdefault('bvh_lafan1', {})[target] = profile
    records = []
    timestep = None
    for bvh in bvhs:
        match = re.search(r'Frame Time:\s*([\d.]+)', bvh.read_text())
        if match is None:
            raise ValueError(f'BVH has no Frame Time: {bvh}')
        dt = float(match[1])
        if timestep is not None and not np.isclose(dt, timestep):
            raise ValueError('Mixed BVH frame rates need separate libraries')
        timestep = dt
        frames = load_frames(bvh)
        frames = frames[round(start_s / dt):round((start_s + duration) / dt)]
        if len(frames) < 2:
            raise ValueError(f'Motion interval has fewer than two frames: {bvh}')
        ik = ProteusRetargeter('bvh_lafan1', target, actual_human_height=None, verbose=False)
        ik.reference_dt = dt
        # Start IK at the training pose rather than a zero pose outside its limits.
        q = ik.configuration.data.qpos.copy()
        q[:7] = cfg.default_qpos[:7]
        for i, name in enumerate(cfg.joint_names):
            q[ik.model.joint(name).qposadr] = cfg.default_qpos[7 + i]
        ik.configuration.update(q)
        for _ in range(40):
            ik.retarget(frames[0])
        qpos = np.stack([ik.retarget(frame) for frame in tqdm(frames, desc=f'{cfg.name}: {bvh.stem}')])
        # Ground with the compiled contact mesh AABBs, boxes and spheres. The
        # per-frame vertical correction preserves the mocap's horizontal path.
        data = mujoco.MjData(ik.model)
        feet = [ik.model.body(n).id for n in cfg.body_names
                if 'ankle_roll' in n.lower()]
        for body in range(1, ik.model.nbody):
            if int(ik.model.body_parentid[body]) in feet and body not in feet:
                feet.append(body)
        geoms = [g for g in range(ik.model.ngeom) if ik.model.geom_bodyid[g] in feet
                 and (ik.model.geom_contype[g] or ik.model.geom_conaffinity[g])]
        if not geoms:
            raise ValueError('No foot contact geometry to ground the reference')
        shifts = []
        for pose in qpos:
            data.qpos[:] = pose
            mujoco.mj_kinematics(ik.model, data)
            bottoms = []
            for g in geoms:
                kind = ik.model.geom_type[g]
                if kind == mujoco.mjtGeom.mjGEOM_MESH:
                    mesh = ik.model.geom_dataid[g]
                    start, count = ik.model.mesh_vertadr[mesh], ik.model.mesh_vertnum[mesh]
                    verts = ik.model.mesh_vert[start:start + count]
                    bottoms.append(float((verts @ data.geom_xmat[g].reshape(3, 3)[2]).min() + data.geom_xpos[g, 2]))
                elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                    bottoms.append(float(data.geom_xpos[g, 2] - ik.model.geom_size[g, 0]))
                elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                    bottoms.append(float(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ ik.model.geom_size[g]))
                else:
                    raise ValueError(f'Unsupported sole geometry {kind}; add exact grounding')
            shift = -min(bottoms)
            pose[2] += shift
            shifts.append(shift)
        qpos[:, :2] -= qpos[0, :2].copy()
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=1, keepdims=True)
        if not np.isfinite(qpos).all():
            raise ValueError('Non-finite retargeted reference')
        save_motion(output / 'motions' / f'{bvh.stem}.npz', qpos)
        records.append({'file': f'motions/{bvh.stem}.npz', 'frames': len(qpos),
            'bvh': bvh.name, 'bvh_sha256': hashlib.sha256(bvh.read_bytes()).hexdigest(),
            'start_s': start_s, 'duration_s': len(qpos) * dt,
            'ground_shift_range_m': [min(shifts), max(shifts)]})
    source = {'kind': 'LAFAN1 human mocap solved by target-model IK', 'motions': records,
              'profile_sha256': hashlib.sha256(profile.read_bytes()).hexdigest(),
              'max_joint_speed_rad_s': ik.max_joint_speed,
              'root_xy': 'recentered at first frame', 'quaternion': 'wxyz'}
    write_manifest(output, dataset_name=f'{cfg.name}_lafan', mjcf=cfg.resolve_mjcf_path(),
        timestep=timestep, qpos_names=qpos_names_from_model(ik.model), num_motions=len(records),
        source=source, total_hours=sum(r['duration_s'] for r in records) / 3600)
    print(json.dumps(source, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True)
    p.add_argument('--bvh', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--start-s', type=float, default=0.)
    p.add_argument('--duration', type=float, default=8.)
    a = p.parse_args()
    if a.start_s < 0 or a.duration <= 0 or len({b.stem for b in a.bvh}) != len(a.bvh):
        raise ValueError('Invalid interval or duplicate motion names')
    retarget(a.robot, a.bvh, a.output, a.start_s, a.duration)
