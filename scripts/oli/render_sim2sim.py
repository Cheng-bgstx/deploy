#!/usr/bin/env python3
"""Record actual integrated OLI sim2sim beside its reference, using MuJoCo EGL and ffmpeg."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('SIM2REAL_ORT_NUM_THREADS', '1')

import torch  # noqa: F401  # native library ordering on ARM
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim2real.config.robots import get_robot_cfg
from sim2real.sim_env.integrated_sim2sim import IntegratedSim2Sim, IntegratedSim2SimArgs


class VideoSim(IntegratedSim2Sim):
    def __init__(self, args, output: Path, fps: int, label: str):
        args.root_trajectory_output = str(output.with_suffix('.root.npz'))
        args.trajectory_output = str(output.with_suffix('.trajectory.npz'))
        super().__init__(args)
        self.output = output
        self.fps, self.label = fps, label
        stride = 1.0 / fps / args.sim_dt
        if not np.isclose(stride, round(stride)):
            raise ValueError('fps must divide the physics frequency')
        self.stride = round(stride)
        self.frames = 0
        self.sim.mj_model.vis.global_.offwidth = 640
        self.sim.mj_model.vis.global_.offheight = 600
        self.renderer = mujoco.Renderer(self.sim.mj_model, height=600, width=640)
        self.reference = mujoco.MjData(self.sim.mj_model)
        self.camera = mujoco.MjvCamera()
        scale = float(self.robot_cfg.default_qpos[2]) / .9282088749221369
        self.camera.lookat[:] = [0, 0, .85 * scale]
        self.camera.distance, self.camera.azimuth, self.camera.elevation = 3.1 * scale, 145, -12
        font = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        self.font = ImageFont.truetype(font, 24) if Path(font).exists() else ImageFont.load_default()
        self.small = ImageFont.truetype(font, 18) if Path(font).exists() else ImageFont.load_default()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.encoder = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', '1280x720', '-framerate', str(fps),
            '-i', '-', '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'fast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output),
        ], stdin=subprocess.PIPE)

    def _append_trajectory_frame(self):
        super()._append_trajectory_frame()
        tick = round(self.sim.mj_data.time / self.args.sim_dt)
        if tick % self.stride:
            return
        state = self.state_processor
        frame = self._motion_frame(int(state.motion_t[0]))
        root_name = self.policy.policy_config['motion']['root_body_name']
        root = state.motion_body_names.index(root_name)
        adr = self.sim.root_qpos_adr
        self.reference.qpos[adr:adr + 3] = frame.body_pos_w[0, 0, root]
        self.reference.qpos[adr + 3:adr + 7] = frame.body_quat_w[0, 0, root]
        for robot_id, qadr in zip(self.sim.joint_indices_unitree, self.sim.qpos_adrs):
            name = self.robot_cfg.joint_names[robot_id]
            self.reference.qpos[qadr] = frame.joint_pos[0, 0, state.motion_joint_names.index(name)]
        mujoco.mj_forward(self.sim.mj_model, self.reference)
        # One shared camera follows both roots, retaining visible tracking drift.
        self.camera.lookat[:2] = .5 * (self.reference.qpos[adr:adr + 2] + self.sim.mj_data.qpos[adr:adr + 2])
        canvas = Image.new('RGB', (1280, 720), '#16222e')
        # Same camera, timing and scale on both panes.
        for x, data in [(0, self.reference), (640, self.sim.mj_data)]:
            self.renderer.update_scene(data, camera=self.camera)
            canvas.paste(Image.fromarray(self.renderer.render()), (x, 72))
        draw = ImageDraw.Draw(canvas)
        robot_label = {'x2': 'AgiBot Lingxi X2', 'pm01': 'EngineAI PM01'}.get(self.robot_cfg.name, 'LimX OLI HU_D04_01')
        draw.text((20, 8), f'{robot_label} | Proteus WAMSeq | sim2sim', font=self.font, fill='white')
        draw.text((20, 42), self.label, font=self.small, fill='#9ecbdc')
        for x, text in [(16, 'REFERENCE'), (656, 'POLICY + MuJoCo physics')]:
            draw.rectangle((x - 4, 84, x + 335, 117), fill='#16222e')
            draw.text((x + 4, 88), text, font=self.small, fill='white')
        t = self.sim.mj_data.time
        phase = 'Initial hold' if not self.playback_started else ('Final-frame hold' if self._at_last_paused_frame() else 'Reference playback')
        height = self.sim.mj_data.qpos[adr + 2]
        draw.text((20, 686), f't={t:4.2f}s | {phase} | policy 50 Hz | root z={height:.3f} m | 1x speed',
                  font=self.small, fill='white')
        if self.frames == 0:
            canvas.save(self.output.with_suffix('.jpg'))
        self.encoder.stdin.write(canvas.tobytes())
        self.frames += 1

    def record_video(self):
        try:
            self.run()
        finally:
            self.encoder.stdin.close()
            code = self.encoder.wait()
            self.renderer.close()
        if code:
            raise RuntimeError(f'ffmpeg failed: {code}')
        report = {'video': str(self.output), 'fps': self.fps, 'frames': self.frames,
                  'duration_s': self.frames / self.fps, 'motion': self.args.motion_path,
                  'policy': self.args.policy_config, 'label': self.label, 'seed': self.args.seed,
                  'final_root_height_m': float(self.sim.mj_data.qpos[self.sim.root_qpos_adr + 2])}
        report['root_trajectory'] = self.args.root_trajectory_output
        report['trajectory'] = self.args.trajectory_output
        self.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


def main(default_robot='oli'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-path', required=True)
    parser.add_argument('--robot', default=default_robot)
    parser.add_argument('--output', type=Path, default=Path('outputs/oli/sim2sim.mp4'))
    parser.add_argument('--policy-config')
    parser.add_argument('--duration', type=float, default=6)
    parser.add_argument('--initial-pause-s', type=float, default=1)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--label', default='OLI motion reference')
    parser.add_argument('--sim-dt', type=float, help='Physics timestep (default: the simulator default)')
    args = parser.parse_args()
    robot_name = get_robot_cfg(args.robot).name
    default_policy = ('checkpoints/proteus/wsq61/lingxi_x2/policy-sim.yaml' if robot_name == 'x2'
                      else 'checkpoints/proteus/wsq61/limx_oli_edu/policy-sim.yaml')
    sim_args = IntegratedSim2SimArgs(robot=args.robot, policy_config=args.policy_config or default_policy,
        motion_path=args.motion_path, headless=True, initial_pause_s=args.initial_pause_s,
        max_runtime_s=args.duration, seed=0, **({'sim_dt': args.sim_dt} if args.sim_dt else {}))
    VideoSim(sim_args, args.output, args.fps, args.label).record_video()


if __name__ == '__main__':
    main()
