#!/usr/bin/env python3
"""Convert one Proteus packed OLI clip to any4hdmi; or create a stationary smoke-test clip."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import OLI_CFG


def convert(output: Path, *, library: Path | None = None, motion_index: int = 0, duration: float = 4.0):
    model = mujoco.MjModel.from_xml_path(str(OLI_CFG.resolve_mjcf_path()))
    names = qpos_names_from_model(model)
    q0 = model.qpos0.copy()
    q0[:7] = OLI_CFG.default_qpos[:7]
    for i, n in enumerate(OLI_CFG.joint_names):
        q0[model.joint(n).qposadr] = OLI_CFG.default_qpos[7 + i]
    if library is None:
        if duration <= 0:
            raise ValueError('duration must be positive')
        qpos = np.tile(q0, (max(2, round(duration * 50)), 1))
        name, source = 'standing_smoke', {'kind': 'synthetic stationary reference; smoke test only'}
    else:
        index = json.loads((library / 'index.json').read_text())
        if not np.isclose(index['fps'], 50):
            raise ValueError('Proteus OLI deployment expects 50 Hz motions')
        if not 0 <= motion_index < len(index['lengths']):
            raise ValueError('motion-index outside library')
        start = int(sum(index['lengths'][:motion_index]))
        count = int(index['lengths'][motion_index])
        sl = slice(start, start + count)
        def array(key):
            return np.asarray(np.load(library / f'{key}.npy', mmap_mode='r')[sl])
        source_joints = list(index['joint_names'])
        if set(source_joints) != set(OLI_CFG.joint_names):
            raise ValueError('Library must contain the 31 OLI joints by name')
        qpos = np.tile(q0, (count, 1))
        qpos[:, :3], qpos[:, 3:7] = array('root_pos_w'), array('root_quat_w')
        if not np.isfinite(qpos).all() or np.any(np.linalg.norm(qpos[:, 3:7], axis=-1) < 1e-6):
            raise ValueError('Invalid root poses')
        qpos[:, 3:7] /= np.linalg.norm(qpos[:, 3:7], axis=-1, keepdims=True)
        jp = array('joint_pos')
        for n in OLI_CFG.joint_names:
            qpos[:, model.joint(n).qposadr[0]] = jp[:, source_joints.index(n)]
        body_names = list(index['body_names'])
        ids = [model.body(n).id for n in body_names]
        pos, quat = array('body_pos_w'), array('body_quat_w')
        data = mujoco.MjData(model)
        max_pos_error = max_quat_error = 0.0
        for t in range(0, count, max(1, count // 32)):
            data.qpos[:] = qpos[t]
            mujoco.mj_kinematics(model, data)
            max_pos_error = max(max_pos_error, float(np.linalg.norm(data.xpos[ids] - pos[t], axis=-1).max()))
            max_quat_error = max(max_quat_error, float(np.abs(1 - np.abs(np.sum(data.xquat[ids] * quat[t], axis=-1))).max()))
        if max_pos_error > 1e-3 or max_quat_error > 1e-4:
            raise ValueError(f'Library FK differs from OLI: {max_pos_error} m, quaternion error {max_quat_error}')
        # Keep file names independent of arbitrary source path characters.
        name = f'clip_{motion_index:05d}'
        source = {'kind': 'Proteus packed OLI library', 'library': str(library.resolve()),
                  'motion': index['motion_names'][motion_index], 'fk_position_error_m': max_pos_error,
                  'fk_quaternion_error': max_quat_error}
    if not np.isfinite(qpos).all():
        raise ValueError('Motion contains non-finite joint positions')
    output = output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'{output} already contains a dataset; select a new output directory')
    motion = save_motion(output / 'motions' / f'{name}.npz', qpos)
    write_manifest(output, dataset_name=f'oli_{name}', mjcf=OLI_CFG.resolve_mjcf_path(), timestep=.02,
                   qpos_names=names, num_motions=1, source=source, total_hours=len(qpos) / 50 / 3600)
    print(motion)
    return motion


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--library', type=Path)
    mode.add_argument('--standing-smoke', action='store_true')
    p.add_argument('--motion-index', type=int, default=0)
    p.add_argument('--duration', type=float, default=4.0)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    convert(args.output, library=args.library, motion_index=args.motion_index, duration=args.duration)


if __name__ == '__main__':
    main()
