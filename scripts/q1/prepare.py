#!/usr/bin/env python3
"""Prepare the verified vendor model, WAMSeq ONNX, PICO hold and three real motions."""
import argparse
import os
from pathlib import Path
import subprocess

from scripts.proteus.prepare_assets import prepare
from scripts.proteus.build_retarget_profiles import build
from scripts.export_proteus_onnx import export_policy
from scripts.make_hold_motion import make_hold
import json

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=ROOT, text=True).strip())
    p.add_argument('--sources-root', type=Path, default=common.parent.parent)
    p.add_argument('--proteus-root', type=Path)
    p.add_argument('--release', default='proteus_wamseq_62')
    p.add_argument('--skip-motions', action='store_true')
    a = p.parse_args()
    sources = a.sources_root.resolve()
    source = sources / 'primebot-q1-urdf'
    license_path = None
    prepare('q1', 'primebot_q1', source, Path('xml/q1_v3.xml'), license_path, ())
    export_policy('primebot_q1', proteus_root=a.proteus_root or sources / 'Proteus-wamseq',
                  robot='q1', release=a.release)
    configs = ROOT / 'sim2real/teleop/configs'
    for kind, filename in [('xrobot', 'xrobot_to_g1.json'), ('bvh_lafan1', 'bvh_lafan1_to_g1.json')]:
        profile = build('q1', 'primebot_q1', sources / 'GMR/general_motion_retargeting/ik_configs' / filename, kind)
        (configs / f'{kind}_to_q1.json').write_text(json.dumps(profile, indent=2) + '\n')
    hold = ROOT / 'checkpoints/primebot_q1/pico_hold'
    if not (hold / 'manifest.json').is_file():
        make_hold('q1', hold)
    motions = ROOT / 'checkpoints/primebot_q1/motions'
    if (motions / 'manifest.json').is_file():
        import hashlib
        manifest = json.loads((motions / 'manifest.json').read_text())
        expected = hashlib.sha256((configs / 'bvh_lafan1_to_q1.json').read_bytes()).hexdigest()
        if manifest['source']['profile_sha256'] != expected:
            raise ValueError('Existing motion library uses another IK profile; move it aside before regenerating')
    if not a.skip_motions and not (motions / 'manifest.json').is_file():
        bvh_root = sources / 'ubisoft-laforge-animation-dataset/lafan1/data'
        command = ['uv', 'run', '--project', 'venv/pico', '--no-sync', 'python', '-m',
                   'scripts.proteus.retarget_lafan', '--robot', 'q1', '--duration', '8',
                   '--output', str(motions), '--bvh',
                   *(str(bvh_root / name) for name in ['walk1_subject1.bvh', 'dance1_subject1.bvh', 'fightAndSports1_subject4.bvh'])]
        subprocess.run(command, cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}, check=True)
    print(f'Policy: {ROOT / "checkpoints/proteus" / a.release / "primebot_q1/policy.yaml"}')
    print(f'Motions: {motions}')
    print(f'PICO hold: {hold}')


if __name__ == '__main__':
    main()
