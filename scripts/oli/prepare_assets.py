#!/usr/bin/env python3
"""Stage OLI's serial PR model and verify it against the Proteus training spec.

uv run --no-sync python scripts/oli/prepare_assets.py --description-root ../humanoid-description
Alternatively --asset-root points at SCALETRACK_ROBOT_ASSETS (the trained robot asset tree).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import yaml

from sim2real.config.robots.oli import OLI_CFG, OLI_MJCF_PATH

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path('ScaleTrack/source/scaletrack/scaletrack/robots/spec/configs/limx_oli_edu.yaml')


def validate(path: Path, spec: dict) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    scalar_names = [model.joint(i).name for i in range(model.njnt)
                    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    if set(scalar_names) != set(OLI_CFG.joint_names) or model.nq != OLI_CFG.qpos_size:
        raise ValueError('Expected the serial HU_D04_01 model with exactly 31 virtual PR joints')
    for name, joint in spec['joints'].items():
        if not np.allclose(model.joint(name).range, [joint['lower'], joint['upper']], atol=1e-6):
            raise ValueError(f'{name}: joint limits differ from training')
        data.qpos[model.joint(name).qposadr] = joint['default']
    mujoco.mj_forward(model, data)
    error = max(float(np.abs(data.xaxis[model.joint(n).id] - j['axis']).max())
                for n, j in spec['joints'].items())
    if error > 1e-5 or not np.isclose(model.body_mass.sum(), spec['mass'], atol=1e-5):
        raise ValueError(f'Model differs from training: axis error={error}, mass={model.body_mass.sum()}')
    for name in OLI_CFG.body_names:
        model.body(name)  # Fail before staging if any runtime body is missing.
    return {'mass': float(model.body_mass.sum()), 'max_joint_axis_error': error,
            'nq': model.nq, 'nv': model.nv,
            'mjcf_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--description-root', type=Path)
    source.add_argument('--asset-root', type=Path)
    parser.add_argument('--proteus-root', type=Path, default=ROOT.parent / 'Proteus')
    args = parser.parse_args()
    spec = yaml.safe_load((args.proteus_root / SPEC).read_text())
    model_dir = (args.description_root / 'HU_D04_description' if args.description_root
                 else args.asset_root / 'limx_oli_edu/model')
    source_xml = model_dir / 'xml/HU_D04_01_vis.xml'
    report = validate(source_xml, spec)
    destination = OLI_MJCF_PATH.parents[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if model_dir.resolve() != destination.resolve():
        shutil.copytree(model_dir, destination, dirs_exist_ok=True)
    validate(OLI_MJCF_PATH, spec)
    (destination.parent / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(OLI_MJCF_PATH)


if __name__ == '__main__':
    main()
