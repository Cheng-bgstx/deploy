#!/usr/bin/env python3
"""Transfer G1's calibrated human axes to another morphology using neutral-pose FK.

No G1 joint trajectory is played on the target. The source table supplies the human
body axes and task weights; target offsets, limb/root scales and link frames come
from the two MJCFs. Real motion is solved by IK on the target model. Preserve the
generated profiles with their source hashes; confirm tracking before hardware use.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml

from sim2real.config.robots import get_robot_cfg, proteus_spec


def neutral(cfg):
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
    data = mujoco.MjData(model)
    data.qpos[:7] = cfg.default_qpos[:7]
    for n in cfg.joint_names:
        data.qpos[model.joint(n).qposadr] = np.clip(0., cfg.joint_pos_lower_limit[n], cfg.joint_pos_upper_limit[n])
    mujoco.mj_kinematics(model, data)
    return model, data


def build(robot, contract, source, kind):
    cfg = get_robot_cfg(robot)
    spec_path = Path(proteus_spec.__file__).with_name('specs') / f'{contract}.yaml'
    spec = yaml.safe_load(spec_path.read_text())
    target_model, target_data = neutral(cfg)
    source_model, source_data = neutral(get_robot_cfg('g1'))
    table = json.loads(source.read_text())
    semantic = spec['semantic_bodies']
    humans = {'pelvis': 'pelvis', 'spine3': 'torso', **{
        f'{side}_{human}': f'{side}_{body}' for side in ('left', 'right')
        for human, body in [('hip', 'hip'), ('knee', 'knee'), ('foot', 'ankle'),
                            ('shoulder', 'shoulder'), ('elbow', 'elbow'), ('wrist', 'wrist')]}}
    def key(human):
        n = human.lower()
        if kind == 'bvh_lafan1':
            for side in ('left', 'right'):
                if n.startswith(side):
                    part = n[len(side):].lower()
                    part = {'upleg': 'hip', 'leg': 'knee', 'footmod': 'foot', 'toe': 'foot',
                            'arm': 'shoulder', 'forearm': 'elbow', 'hand': 'wrist'}.get(part, part)
                    return f'{side}_{part}'
            return {'hips': 'pelvis', 'spine2': 'spine3'}.get(n, n)
        return n
    # Target end-of-ball frames constrain all hip/shoulder axes. The semantic
    # descriptor's intermediate roll frame alone does not constrain yaw.
    mapping = dict(semantic)
    for side in ('left', 'right'):
        for part in ('hip', 'shoulder'):
            candidates = [n for n in cfg.body_names if side in n.lower() and part in n.lower() and 'yaw' in n.lower()]
            if candidates:
                mapping[f'{side}_{part}'] = candidates[-1]
    root_scale = float(cfg.default_qpos[2] / get_robot_cfg('g1').default_qpos[2])
    src_root = source_data.xpos[source_model.body(table['robot_root_name']).id]
    dst_root = target_data.xpos[target_model.body(semantic['pelvis']).id]
    root_name = table['human_root_name']
    old_root_scale = table['human_scale_table'][root_name]
    new_root_scale = old_root_scale * root_scale
    scales = {}
    for stage in ('ik_match_table1', 'ik_match_table2'):
        entries = {}
        for old_link, entry in table[stage].items():
            human, pc, rc, offset, quat = entry
            body_key = humans[key(human)]
            link = mapping[body_key]
            old_id, new_id = source_model.body(old_link).id, target_model.body(link).id
            old_rot = R.from_quat(source_data.xquat[old_id], scalar_first=True)
            new_rot = R.from_quat(target_data.xquat[new_id], scalar_first=True)
            old_offset_rot = R.from_quat(quat, scalar_first=True)
            human_rot = old_rot * old_offset_rot.inv()
            new_quat = (human_rot.inv() * new_rot).as_quat(scalar_first=True)
            old_len = np.linalg.norm(source_data.xpos[old_id] - src_root)
            new_len = np.linalg.norm(target_data.xpos[new_id] - dst_root)
            scale = new_root_scale if human == root_name else table['human_scale_table'][human] * new_len / max(old_len, 1e-6)
            scales[human] = float(scale)
            # Reconstruct the same source human neutral skeleton, then express
            # the target's neutral residual in its own body frame.
            old_human = source_data.xpos[old_id] - old_rot.apply(offset)
            human_root = src_root / old_root_scale
            old_human_local = (old_human - src_root) / table['human_scale_table'][human]
            new_human = new_root_scale * human_root + scale * old_human_local
            pos = new_rot.inv().apply(target_data.xpos[new_id] - new_human)
            entries[link] = [human, pc, rc, pos.tolist(), new_quat.tolist()]
        table[stage] = entries
    table['robot_root_name'] = semantic['pelvis']
    table['human_scale_table'] = scales
    table['source'] = {'method': 'neutral FK morphology transfer of calibrated human frames; target IK',
        'source_config': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'target_spec_sha256': hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'generator': 'scripts/proteus/build_retarget_profiles.py'}
    return table


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True)
    p.add_argument('--contract', required=True)
    p.add_argument('--gmr-configs', type=Path, required=True)
    a = p.parse_args()
    folder = Path(proteus_spec.__file__).parents[2] / 'teleop/configs'
    for kind, filename in [('xrobot', 'xrobot_to_g1.json'), ('bvh_lafan1', 'bvh_lafan1_to_g1.json')]:
        table = build(a.robot, a.contract, a.gmr_configs / filename, kind)
        path = folder / f'{kind}_to_{a.robot}.json'
        path.write_text(json.dumps(table, indent=2) + '\n')
        print(path)
