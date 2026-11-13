#!/usr/bin/env python3
"""Adapt the public X2 calibration to the vendor model's actual ankle frames."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.build_xrobot_profile import convert


def build(smplx, bvh, gmr_configs):
    config = convert(json.loads(smplx.read_text()))
    lafan = json.loads(bvh.read_text())
    g1_lafan = json.loads((gmr_configs / 'bvh_lafan1_to_g1.json').read_text())
    g1_xr = json.loads((gmr_configs / 'xrobot_to_g1.json').read_text())
    for stage in ('ik_match_table1', 'ik_match_table2'):
        for side in ('left', 'right'):
            human = f'{side.title()}FootMod'
            # G1's fixed toe and ankle-roll frames have identity relative rotation.
            # FootMod is an ankle position with the toe's orientation, in both sources.
            xr_entry = next(v for v in g1_xr[stage].values() if v[0] == f'{side.title()}_Foot')
            bv_entry = next(v for v in g1_lafan[stage].values() if v[0] == human)
            entry = json.loads(json.dumps(lafan[stage][f'{side}_ankle_roll_link']))
            entry[0] = human
            entry[4] = (R.from_quat(xr_entry[4], scalar_first=True)
                        * R.from_quat(bv_entry[4], scalar_first=True).inv()
                        * R.from_quat(entry[4], scalar_first=True)).as_quat(scalar_first=True).tolist()
            del config[stage][f'{side}_toe_link']
            config[stage][f'{side}_ankle_roll_link'] = entry
            config['human_scale_table'].pop(f'{side.title()}_Foot', None)
            config['human_scale_table'][human] = lafan['human_scale_table'][human]
    config['source'] = {
        'repository': 'TaiE64/AgibotX2-Retargeter',
        'commit': 'bb582d97b43cd0df08953d3347fc4472dad8fa9e',
        'generator': 'scripts/x2/build_pico_profile.py',
        'body_axes': 'SMPL-X offset composed with Ry(pi)',
        'feet': 'Use calibrated LAFAN ankle-roll tasks; FootMod position=XR Ankle, orientation=XR Foot',
        'input_sha256': {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (smplx, bvh, gmr_configs / 'xrobot_to_g1.json',
                                   gmr_configs / 'bvh_lafan1_to_g1.json')},
    }
    return config


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retarget-root', type=Path, default=Path('../AgibotX2-Retargeter'))
    p.add_argument('--gmr-configs', type=Path, default=Path('../GMR/general_motion_retargeting/ik_configs'))
    p.add_argument('--output', type=Path, default=Path('sim2real/teleop/configs/xrobot_to_x2.json'))
    args = p.parse_args()
    configs = args.retarget_root / 'gmr_x2_overlay/ik_configs'
    args.output.write_text(json.dumps(build(configs / 'smplx_to_x2.json',
                                           configs / 'bvh_lafan1_to_x2.json', args.gmr_configs), indent=2) + '\n')
