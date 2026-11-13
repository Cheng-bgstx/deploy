#!/usr/bin/env python3
"""Write a named neutral reference for shaping before a live PICO stream."""
import argparse
from pathlib import Path

import mujoco
import numpy as np
from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.utils.mjcf import qpos_names_from_model

from sim2real.config.robots import get_robot_cfg


def make_hold(robot: str, output: Path):
    cfg = get_robot_cfg(robot)
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
    q = model.qpos0.copy()
    q[:7] = cfg.default_qpos[:7]
    for i, name in enumerate(cfg.joint_names):
        q[model.joint(name).qposadr[0]] = cfg.default_qpos[7 + i]
    output = output.expanduser().resolve()
    if (output / 'manifest.json').exists():
        raise FileExistsError(output)
    save_motion(output / 'motions' / 'hold.npz', np.tile(q, (100, 1)))
    write_manifest(output, dataset_name=f'{cfg.name}_pico_hold', mjcf=cfg.resolve_mjcf_path(),
                   timestep=.02, qpos_names=qpos_names_from_model(model), num_motions=1,
                   source={'kind': 'RobotCfg default pose; PICO startup shaping reference'},
                   total_hours=2 / 3600)
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    make_hold(args.robot, args.output)
