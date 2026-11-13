#!/usr/bin/env python3
"""Write policy-pc-damped.yaml next to policy-pc.yaml: arm kd x5 and leg/waist kd x2 by default, kp unchanged.

On the X2 Basic the arms oscillated at 4.4 Hz with growing amplitude once the policy took over with the training
gains (arm kp 17.5 / kd 1.1); the vendor's own mimic policy runs the arms at kp 50 / kd 3. sim2sim reproduces it
(scripts/x2/eval_sim2sim.py --sim-dt 0.001 --cmd-delay-ms 40: growing 4.3 Hz arm oscillation with the training
gains). Arm kd x5 removes it at 40 ms, keeps the wave standing at 50 ms where the training gains fall, and leaves the
tracking error unchanged.

With the arms fixed, the hips and pelvis then rocked at 3.7 Hz with a steady amplitude (second live run). With 50 ms
of delay sim2sim shows the same standing leg oscillation (0.72 rad/s); leg and waist kd x2 removes it (0.01) and holds
standing at 60 ms, with walking and wave errors unchanged. The X2 bridge accepts kd up to 25 for this.

  python scripts/x2/make_damped_config.py checkpoints/proteus/proteus_wamseq_62/lingxi_x2
"""
import argparse
from pathlib import Path

import yaml

ARM_KEYS = ('shoulder', 'elbow', 'wrist')
LEG_KEYS = ('hip', 'knee', 'ankle', 'waist')

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('policy_dir', type=Path)
p.add_argument('--kd-scale', type=float, default=5.0, help='arm kd multiplier')
p.add_argument('--leg-kd-scale', type=float, default=2.0, help='leg and waist kd multiplier')
args = p.parse_args()
cfg = yaml.safe_load((args.policy_dir / 'policy-pc.yaml').read_text())
for name in cfg['joint_kd']:
    if any(k in name for k in ARM_KEYS):
        cfg['joint_kd'][name] = round(cfg['joint_kd'][name] * args.kd_scale, 4)
    elif any(k in name for k in LEG_KEYS):
        cfg['joint_kd'][name] = round(cfg['joint_kd'][name] * args.leg_kd_scale, 4)
out = args.policy_dir / 'policy-pc-damped.yaml'
out.write_text(f'# policy-pc.yaml with arm kd x{args.kd_scale:g}, leg/waist kd x{args.leg_kd_scale:g}, kp unchanged '
               '(scripts/x2/make_damped_config.py).\n'
               + yaml.safe_dump(cfg, sort_keys=False))
print(out)
