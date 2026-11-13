#!/usr/bin/env python3
"""Prepare the 24-joint PM01 model, wsq61 ONNX, PICO hold and LAFAN motions."""
import argparse
import os
from pathlib import Path
import subprocess

from scripts.proteus.prepare_assets import prepare
from scripts.export_proteus_onnx import export_policy
from scripts.make_hold_motion import make_hold
import json

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    common = Path(subprocess.check_output(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=ROOT, text=True).strip())
    p.add_argument('--sources-root', type=Path, default=common.parent.parent)
    p.add_argument('--proteus-root', type=Path)
    p.add_argument('--release', default='wsq61')
    p.add_argument('--skip-motions', action='store_true')
    a = p.parse_args()
    sources = a.sources_root.resolve()
    source = sources / 'GMR-engineai/assets/engineai_pm01'
    license_path = sources / 'GMR-engineai/LICENSE'
    # The released PM01 spec rounds world axes to four decimals.
    prepare('pm01', 'engineai_pm01', source, Path('pm_v2.xml'), license_path, (), axis_atol=1e-4)
    export_policy('engineai_pm01', proteus_root=a.proteus_root or sources / 'Proteus-wamseq',
                  robot='pm01', release=a.release)
    configs = ROOT / 'sim2real/teleop/configs'
    from scripts.build_xrobot_profile import convert
    import hashlib
    for kind, filename in [('xrobot', 'smplx_to_pm01.json'), ('bvh_lafan1', 'bvh_lafan1_to_pm01.json')]:
        source_config = sources / 'GMR-engineai/general_motion_retargeting/ik_configs' / filename
        profile = json.loads(source_config.read_text())
        if kind == 'xrobot':
            profile = convert(profile)
        profile['source'] = {'repository': 'engineai-robotics/GMR',
            'commit': subprocess.check_output(['git', '-C', str(sources / 'GMR-engineai'), 'rev-parse', 'HEAD'], text=True).strip(),
            'profile': filename, 'sha256': hashlib.sha256(source_config.read_bytes()).hexdigest(),
            'generator': 'scripts/pm01/prepare.py',
            'source_axes': 'R_smplx = R_xrobot * Ry(pi)' if kind == 'xrobot' else 'upstream LAFAN Rx(90), no conjugation'}
        (configs / f'{kind}_to_pm01.json').write_text(json.dumps(profile, indent=2) + '\n')
    hold = ROOT / 'checkpoints/engineai_pm01/pico_hold'
    if not (hold / 'manifest.json').is_file():
        make_hold('pm01', hold)
    motions = ROOT / 'checkpoints/engineai_pm01/motions'
    if (motions / 'manifest.json').is_file():
        import hashlib
        manifest = json.loads((motions / 'manifest.json').read_text())
        expected = hashlib.sha256((configs / 'bvh_lafan1_to_pm01.json').read_bytes()).hexdigest()
        if manifest['source']['profile_sha256'] != expected:
            raise ValueError('Existing motion library uses another IK profile; move it aside before regenerating')
    if not a.skip_motions and not (motions / 'manifest.json').is_file():
        bvh_root = sources / 'ubisoft-laforge-animation-dataset/lafan1/data'
        command = ['uv', 'run', '--project', 'venv/pico', '--no-sync', 'python', '-m',
                   'scripts.proteus.retarget_lafan', '--robot', 'pm01', '--duration', '8',
                   '--output', str(motions), '--bvh',
                   *(str(bvh_root / name) for name in ['walk1_subject1.bvh', 'dance2_subject1.bvh', 'fightAndSports1_subject4.bvh', 'aiming1_subject1.bvh'])]
        subprocess.run(command, cwd=ROOT, env={**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}, check=True)
    print(f'Policy: {ROOT / "checkpoints/proteus" / a.release / "engineai_pm01/policy.yaml"}')
    print(f'Motions: {motions}')
    print(f'PICO hold: {hold}')


if __name__ == '__main__':
    main()
