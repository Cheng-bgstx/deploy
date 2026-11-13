#!/usr/bin/env python3
"""Stage an official MJCF and verify its kinematics against the frozen training spec.

--model-dir is copied in full to third_party/<contract>/model. --xml is relative
to that directory. Runtime actuators/sensors are replaced by sim2real. Differences
in mass and limits remain explicit in validation.json; joint-axis differences fail.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots import proteus_spec


def verify(model, cfg, spec, *, axis_atol=1e-5):
    scalar = [model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
    free = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1 or set(scalar) != set(cfg.joint_names) or model.nq != cfg.qpos_size:
        raise ValueError(f'Asset joints differ from {cfg.name}: {scalar}')
    data = mujoco.MjData(model)
    # Spec axes are expressed with an identity root, at the training default pose.
    data.qpos[model.joint(free[0]).qposadr[0] + 3:][:4] = [1., 0., 0., 0.]
    for n, j in spec['joints'].items():
        data.qpos[model.joint(n).qposadr] = j['default']
    mujoco.mj_kinematics(model, data)
    errors = {n: float(np.abs(data.xaxis[model.joint(n).id] - j['axis']).max())
              for n, j in spec['joints'].items()}
    if max(errors.values()) > axis_atol:
        raise ValueError(f'Asset axes differ from training: {errors}')
    for body in cfg.body_names:
        model.body(body)
    return max(errors.values())


def prepare(robot, contract, model_dir, xml, license_path=None, fixed_joints=(), *, axis_atol=1e-5):
    cfg = get_robot_cfg(robot)
    spec_path = Path(proteus_spec.__file__).with_name('specs') / f'{contract}.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    source = model_dir.resolve() / xml
    expanded = mujoco.MjSpec.from_file(str(source))
    expanded.compile()
    tree = ET.ElementTree(ET.fromstring(expanded.to_xml()))
    root = tree.getroot()
    for block in list(root):
        if block.tag in {'actuator', 'sensor', 'keyframe'}:
            root.remove(block)
    # The A3 contract has no head-pitch DOF. Preserve the link and inertia,
    # explicitly fixing only requested joints at zero; reject every other extra.
    for name in fixed_joints:
        matches = [(body, j) for body in root.iter() for j in body.findall('joint') if j.get('name') == name]
        if len(matches) != 1 or name in spec['joints']:
            raise ValueError(f'Cannot fix a missing or contract-driven joint: {name}')
        matches[0][0].remove(matches[0][1])
    for joint in root.iter():
        if joint.tag == 'freejoint' or (joint.tag == 'joint' and joint.get('type') == 'free'):
            if not joint.get('name'):
                joint.set('name', 'floating_base_joint')
    with tempfile.NamedTemporaryFile(suffix='.xml', dir=source.parent) as temporary:
        tree.write(temporary.name, encoding='utf-8', xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(temporary.name)
    error = verify(model, cfg, spec, axis_atol=axis_atol)
    report = {'robot': robot, 'contract': contract, 'source_xml': str(source),
              'source_xml_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
              'vendor_mass_kg': float(model.body_mass.sum()), 'training_mass_kg': spec['mass'],
              'mass_difference_kg': float(model.body_mass.sum() - spec['mass']),
              'max_joint_axis_error': error, 'joint_axis_tolerance': axis_atol, 'joint_limit_overrides': {}}
    report['joints_fixed_at_zero'] = list(fixed_joints)
    target = cfg.mjcf_path
    model_target = target.parents[len(Path(spec['mjcf']).parts) - 3]
    if model_dir.resolve() != model_target.resolve():
        shutil.copytree(model_dir, model_target, dirs_exist_ok=True,
                        ignore=lambda folder, names: [n for n in names
                            if n == '.git' or (Path(folder) / n).is_symlink()])
    for j in root.findall('.//joint'):
        n = j.get('name')
        if n not in spec['joints']:
            continue
        trained = spec['joints'][n]
        limits = [trained['lower'], trained['upper']]
        if not np.allclose(model.joint(n).range, limits):
            report['joint_limit_overrides'][n] = {'vendor': model.joint(n).range.tolist(), 'training': limits}
        j.set('range', ' '.join(map(str, limits)))
        j.set('armature', str(trained['armature']))
        j.set('damping', '0')
        j.set('frictionloss', '0')
    target.parent.mkdir(parents=True, exist_ok=True)
    # MuJoCo expands nested includes and makes resource filenames absolute.
    # Bind every resource to the copied tree so the result is portable.
    for compiler in root.findall('compiler'):
        for key in ('meshdir', 'texturedir'):
            compiler.attrib.pop(key, None)
    for element in root.iter():
        filename = element.get('file')
        if filename:
            original = Path(filename)
            directory = expanded.meshdir if element.tag == 'mesh' else expanded.texturedir if element.tag == 'texture' else ''
            original = original if original.is_absolute() else source.parent / directory / original
            copied = model_target / original.resolve().relative_to(model_dir.resolve())
            element.set('file', os.path.relpath(copied, target.parent))
    tree.write(target, encoding='utf-8', xml_declaration=True)
    final = mujoco.MjModel.from_xml_path(str(target))
    verify(final, cfg, spec, axis_atol=axis_atol)
    if license_path:
        shutil.copy2(license_path, model_target.parent / license_path.name)
    report['prepared_xml_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    (model_target.parent / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True)
    p.add_argument('--contract', required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--xml', type=Path, required=True)
    p.add_argument('--license', type=Path)
    p.add_argument('--fix-joint-at-zero', action='append', default=[])
    a = p.parse_args()
    print(prepare(a.robot, a.contract, a.model_dir, a.xml, a.license, a.fix_joint_at_zero))
