#!/usr/bin/env python3
"""Convert a named X2 reference from sonic-x2's published PKL to any4hdmi.

Only reference poses are imported; the controller remains the Proteus WAMSeq actor.
The upstream schema and quaternion convention are recorded in its eval_x2_mujoco.py.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import X2_CFG


def source_joint_names(script: Path) -> list[str]:
    # Read the published explicit column order without importing another controller.
    for stmt in ast.parse(script.read_text()).body:
        if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MUJOCO_JOINT_NAMES' for t in stmt.targets):
            return list(ast.literal_eval(stmt.value))
    raise ValueError(f'No explicit MUJOCO_JOINT_NAMES in {script}')


def convert(source_root: Path, clip: str, output: Path, motion_key: str | None = None):
    path = source_root / 'motions' / f'x2_{clip}.pkl'
    data = joblib.load(path)
    key = motion_key or next(iter(data))
    motion = data[key]
    names = source_joint_names(source_root / 'scripts/eval_x2_mujoco.py')
    if sorted(names) != sorted(X2_CFG.joint_names):
        raise ValueError('Source references are not the expected 31 X2 joints')
    model = mujoco.MjModel.from_xml_path(str(X2_CFG.resolve_mjcf_path()))
    dof = np.asarray(motion['dof'], dtype=np.float64)
    if dof.ndim != 2 or dof.shape[1] != len(names):
        raise ValueError(f'Invalid reference joint shape {dof.shape}')
    qpos = np.tile(model.qpos0, (len(dof), 1))
    qpos[:, :3] = motion['root_trans_offset']
    # Published source: root_rot is scipy xyzw; MuJoCo and any4hdmi use wxyz.
    qpos[:, 3:7] = np.asarray(motion['root_rot'])[:, [3, 0, 1, 2]]
    if np.any(np.linalg.norm(qpos[:, 3:7], axis=-1) < 1e-6):
        raise ValueError('Invalid reference quaternion')
    qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
    for n in X2_CFG.joint_names:
        qpos[:, model.joint(n).qposadr[0]] = dof[:, names.index(n)]
    if not np.isfinite(qpos).all() or np.median(qpos[:, 2]) < .4:
        raise ValueError('Invalid absolute root height; root-relative clips need explicit reconstruction')
    qpos[:, :2] -= qpos[0, :2].copy()
    output = output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Dataset exists: {output}')
    result = save_motion(output / 'motions' / f'{clip}.npz', qpos)
    source = {'kind': 'published X2 mocap reference', 'repository': 'https://github.com/meetsitaram/sonic-x2',
              'source_file': path.name, 'motion_key': key,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'root_quaternion': 'source xyzw -> wxyz', 'joint_columns': names,
              'root_xy': 'first frame recentered; z unchanged'}
    fps = float(motion['fps'])
    write_manifest(output, dataset_name=f'x2_{clip}', mjcf=X2_CFG.resolve_mjcf_path(), timestep=1 / fps,
                   qpos_names=qpos_names_from_model(model), num_motions=1, source=source,
                   total_hours=len(qpos) / fps / 3600)
    print(json.dumps({'frames': len(qpos), 'fps': fps, 'root_z': [float(qpos[:, 2].min()), float(qpos[:, 2].max())]}))
    print(result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('../sonic-x2-reference'))
    p.add_argument('--clip', choices=['relaxed_walk', 'gangam_dance', 'idle_stand'], required=True)
    p.add_argument('--motion-key')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    convert(args.source_root, args.clip, args.output, args.motion_key)


if __name__ == '__main__':
    main()
