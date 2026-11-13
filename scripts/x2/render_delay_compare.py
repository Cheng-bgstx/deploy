#!/usr/bin/env python3
"""Render X2 sim2sim videos with a command delay, to compare damping configs as the robot sees them.

Same recorder as scripts/x2/render_sim2sim.py (reference beside policy + MuJoCo physics), plus the physics-time
command delay of scripts/x2/eval_sim2sim.py --cmd-delay-ms: each policy command reaches the motors that long after
the state the policy saw. Renders one video per config and stacks them vertically into one comparison video.

  python scripts/x2/render_delay_compare.py --motion-path outputs/x2/wave/21_wave_right --delay-ms 50 \
      --config "training gains=checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc.yaml" \
      --config "damped=checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc-damped.yaml" \
      --output outputs/x2/videos/delay50_compare.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def render_one(label: str, config: str, motion: str, delay_ms: float, duration: float, sim_dt: float,
               output: Path) -> dict:
    from scripts.oli.render_sim2sim import VideoSim
    from sim2real.sim_env.integrated_sim2sim import IntegratedSim2SimArgs

    args = IntegratedSim2SimArgs(robot='x2', policy_config=config, motion_path=motion, headless=True,
                                 initial_pause_s=1.0, max_runtime_s=duration, seed=0, sim_dt=sim_dt)
    sim = VideoSim(args, output, 25, f'{label} | {delay_ms:g} ms command delay')
    pending = []
    apply, step = sim.sim.apply_command, sim.sim.sim_step

    def apply_command(*command, **kw):
        pending.append((sim.sim.mj_data.time + delay_ms / 1000 - 1e-9, command, kw))

    def sim_step():
        while pending and pending[0][0] <= sim.sim.mj_data.time:
            _, command, kw = pending.pop(0)
            apply(*command, **kw)
        return step()

    if delay_ms > 0:
        sim.sim.apply_command, sim.sim.sim_step = apply_command, sim_step
    sim.record_video()
    return json.loads(output.with_suffix('.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--motion-path', required=True)
    p.add_argument('--config', action='append', required=True, help='"label=path/to/policy-pc*.yaml"')
    p.add_argument('--delay-ms', type=float, default=50.0)
    p.add_argument('--duration', type=float, default=13.0)
    p.add_argument('--sim-dt', type=float, default=0.001)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    with ProcessPoolExecutor(max_workers=min(len(args.config), 4)) as pool:
        for i, item in enumerate(args.config):
            label, config = item.split('=', 1)
            part = args.output.with_name(f'{args.output.stem}.{i}.mp4')
            jobs.append(pool.submit(render_one, label, config, args.motion_path, args.delay_ms, args.duration,
                                    args.sim_dt, part))
        parts = [j.result() for j in jobs]
    inputs = sum((['-i', r['video']] for r in parts), [])
    scaled = ''.join(f'[{i}:v]scale=960:540[v{i}];' for i in range(len(parts)))
    stack = ''.join(f'[v{i}]' for i in range(len(parts))) + f'vstack=inputs={len(parts)}[out]'
    if len(parts) == 1:  # vstack needs two inputs; a single config is just the scaled video
        stack = '[v0]null[out]'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *inputs, '-filter_complex', scaled + stack,
                    '-map', '[out]', '-c:v', 'libx264', '-crf', '22', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart', str(args.output)], check=True)
    print(json.dumps({'video': str(args.output), 'parts': [
        {k: r[k] for k in ('video', 'label', 'policy', 'final_root_height_m')} for r in parts]}, indent=2))


if __name__ == '__main__':
    main()
