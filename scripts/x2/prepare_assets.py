#!/usr/bin/env python3
"""Stage the public X2 Ultra v1.3 model for the Proteus lingxi_x2 contract.

The vendor geometry is retained; training joint limits, armatures and zero passive
friction are applied explicitly. Every model difference is recorded in validation.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from sim2real.config.robots.x2 import X2_CFG, X2_MJCF_PATH

ROOT = Path(__file__).resolve().parents[2]
SPEC = Path('ScaleTrack/source/scaletrack/scaletrack/robots/spec/configs/lingxi_x2.yaml')


def verify(model, spec):
    if model.nq != X2_CFG.qpos_size:
        raise ValueError(f'Expected 31 articulated joints, got nq={model.nq}')
    data = mujoco.MjData(model)
    for name, joint in spec['joints'].items():
        data.qpos[model.joint(name).qposadr] = joint['default']
    mujoco.mj_kinematics(model, data)
    errors = {n: float(np.abs(data.xaxis[model.joint(n).id] - j['axis']).max())
              for n, j in spec['joints'].items()}
    if max(errors.values()) > 1e-5:
        raise ValueError(f'Joint axes differ from training: {errors}')
    for name in X2_CFG.body_names:
        model.body(name)
    return max(errors.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--description-root', type=Path, default=ROOT.parent / 'agibot_x2_urdf')
    p.add_argument('--proteus-root', type=Path, default=ROOT.parent / 'Proteus-wamseq')
    args = p.parse_args()
    source = args.description_root / 'X2_URDF-v1.3.0'
    xml = source / 'x2_ultra.xml'
    spec = yaml.safe_load((args.proteus_root / SPEC).read_text())
    vendor = mujoco.MjModel.from_xml_path(str(xml))
    axis_error = verify(vendor, spec)
    report = {'source_xml': str(xml.resolve()), 'vendor_xml_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(),
              'vendor_mass_kg': float(vendor.body_mass.sum()), 'training_spec_mass_kg': spec['mass'],
              'mass_difference_kg': float(vendor.body_mass.sum() - spec['mass']),
              'max_joint_axis_error': axis_error, 'joint_limit_overrides': {}}
    tree = ET.parse(xml)
    root = tree.getroot()
    for element in list(root):
        if element.tag in {'actuator', 'sensor', 'keyframe'}:
            root.remove(element)
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        if name not in spec['joints']:
            continue
        cfg = spec['joints'][name]
        trained_range = [cfg['lower'], cfg['upper']]
        actual_range = vendor.joint(name).range.tolist()
        if not np.allclose(actual_range, trained_range):
            report['joint_limit_overrides'][name] = {'vendor': actual_range, 'training': trained_range}
        joint.set('range', ' '.join(map(str, trained_range)))
        joint.set('armature', str(cfg['armature']))
        joint.set('damping', '0')
        joint.set('frictionloss', '0')
    destination = X2_MJCF_PATH.parent
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The vendor tree contains meshes/meshes -> '.'; do not follow this cycle.
    shutil.copytree(source, destination, dirs_exist_ok=True,
                    ignore=lambda folder, names: [n for n in names if (Path(folder) / n).is_symlink()])
    # Preserve the upstream license alongside the runtime assets.
    license_path = args.description_root / 'LICENSE'
    if license_path.is_file():
        shutil.copy2(license_path, destination.parent / 'LICENSE')
    tree.write(X2_MJCF_PATH, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(X2_MJCF_PATH))
    verify(model, spec)
    report['prepared_xml_sha256'] = hashlib.sha256(X2_MJCF_PATH.read_bytes()).hexdigest()
    (destination.parent / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(X2_MJCF_PATH)


if __name__ == '__main__':
    main()
