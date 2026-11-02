#!/usr/bin/env python3
"""Build reproducible robot-specific LAFAN references, optionally run sim2sim."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from sim2real.config.robots import get_robot_cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True)
    p.add_argument('--lafan-data', type=Path, default=Path('../ubisoft-laforge-animation-dataset/lafan1/data'))
    p.add_argument('--gmr-root', type=Path, default=Path('../GMR-limx'))
    p.add_argument('--output', type=Path)
    p.add_argument('--render', action='store_true')
    args = p.parse_args()
    robot = get_robot_cfg(args.robot).name
    root = Path(__file__).resolve().parents[1]
    output = (args.output or Path(f'outputs/{robot}/lafan_suite')).resolve()
    recipe = json.loads(Path(__file__).with_suffix('.json').read_text())
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
           'SIM2REAL_ORT_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MUJOCO_GL': 'egl'}
    def run(*argv):
        print('+', ' '.join(map(str, argv)), flush=True)
        subprocess.run([sys.executable, *map(str, argv)], cwd=root, env=env, check=True)
    for clip in recipe['clips']:
        bvh = (args.lafan_data / clip['bvh']).resolve()
        if not bvh.is_file():
            raise FileNotFoundError(bvh)
        motion = output / clip['name']
        if not (motion / 'manifest.json').exists():
            run(f'scripts/{robot}/retarget_lafan.py', '--gmr-root', args.gmr_root.resolve(),
                '--bvh', bvh, '--start-s', clip['start_s'], '--duration', clip['duration_s'], '--output', motion)
        if args.render:
            video = output / f'{robot}_wamseq_lafan_{clip["name"]}.mp4'
            run(f'scripts/{robot}/render_sim2sim.py', '--motion-path', motion,
                '--duration', clip['duration_s'] + 2, '--output', video,
                '--label', f'LAFAN1 {clip["bvh"]} | first {clip["duration_s"]} s | measured world root')
            run('scripts/summarize_proteus_sim2sim.py', video)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')


if __name__ == '__main__':
    main()
