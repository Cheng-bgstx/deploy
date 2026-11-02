#!/usr/bin/env python3
"""Convert a calibrated SMPL-X GMR profile to PICO/XRobot's body frames.

GMR's xrobot_to_g1 and smplx_to_g1 agree on a right-side Ry(pi)
change of source body axes. It changes orientation offsets, not world positions
or target-link-local position offsets. Preserve the target morphology's weights
and scales; no G1 joint angles enter the target profile.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scipy.spatial.transform import Rotation as R


def xrobot_name(name):
    return name.title() if name != 'spine3' else 'Spine3'


def convert(config):
    c = json.loads(json.dumps(config))
    c['human_root_name'] = xrobot_name(c['human_root_name'])
    c['human_scale_table'] = {xrobot_name(k): v for k, v in c['human_scale_table'].items()}
    for table in ('ik_match_table1', 'ik_match_table2'):
        for entry in c[table].values():
            entry[0] = xrobot_name(entry[0])
            entry[4] = (R.from_quat([0, 0, 1, 0], scalar_first=True)
                        * R.from_quat(entry[4], scalar_first=True)).as_quat(scalar_first=True).tolist()
    return c


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smplx-config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    c = convert(json.loads(args.smplx_config.read_text()))
    c['source'] = {'smplx_config': args.smplx_config.name,
                   'sha256': hashlib.sha256(args.smplx_config.read_bytes()).hexdigest(),
                   'generator': 'scripts/build_xrobot_profile.py',
                   'source_axes': 'R_smplx = R_xrobot * Ry(pi); world positions unchanged'}
    args.output.write_text(json.dumps(c, indent=2) + '\n')
    print(args.output)
