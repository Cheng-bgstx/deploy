#!/usr/bin/env python3
"""Evaluate every named reference with integrated sim2sim, initial and final holds.

Writes per-motion root/whole-body trajectories and summary.json with exact inputs,
seed and metrics. Uses measured root position (policy-sim.yaml), and fails if a
reference does not finish, any value is non-finite, or body error exceeds 0.5 m.
"""
import argparse
import json
import hashlib
import os
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('SIM2REAL_ORT_NUM_THREADS', '1')

import numpy as np
import yaml

from sim2real.sim_env.integrated_sim2sim import IntegratedSim2Sim, IntegratedSim2SimArgs
from sim2real.utils.math import quat_rotate_inverse_numpy


def evaluate(robot, policy, dataset, output, seed=0):
    cfg = yaml.safe_load(policy.read_text())
    if cfg['proteus']['reference_forcing']:
        raise ValueError('Use policy-sim.yaml with measured world root for evaluation')
    if output.exists():
        raise FileExistsError(f'Choose a new evaluation directory: {output}')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    onnx_path = policy.parent / cfg['model_path']
    inputs = {'policy_yaml_sha256': digest(policy), 'onnx_sha256': digest(onnx_path),
              'manifest_sha256': digest(dataset / 'manifest.json')}
    dt = float(manifest['timestep'])
    output.mkdir(parents=True)
    records = []
    for motion in sorted((dataset / 'motions').glob('*.npz')):
        length = len(np.load(motion)['qpos'])
        prefix = output / motion.stem
        args = IntegratedSim2SimArgs(robot=robot, policy_config=str(policy.resolve()),
            motion_path=str(motion.resolve()), headless=True, initial_pause_s=1.,
            max_runtime_s=(length - 1) * dt + 2., seed=seed,
            root_trajectory_output=str(prefix.with_suffix('.root.npz')),
            trajectory_output=str(prefix.with_suffix('.trajectory.npz')))
        sim = IntegratedSim2Sim(args)
        sim.run()
        root = np.load(prefix.with_suffix('.root.npz'))
        traj = np.load(prefix.with_suffix('.trajectory.npz'))
        indices = [traj['body_names'].tolist().index(n) for n in cfg['proteus']['body_names']]
        robot_body = traj['robot_body_pos_w'][:, indices]
        ref_body = traj['motion_body_pos_w'][:, indices]
        def local(body, name):
            centered = body - traj[f'{name}_root_pos_w'][:, None]
            quat = np.broadcast_to(traj[f'{name}_root_quat_w'][:, None], (*centered.shape[:2], 4)).copy()
            return quat_rotate_inverse_numpy(quat, centered)
        error = np.linalg.norm(robot_body - ref_body, axis=-1).mean(axis=-1)
        local_error = np.linalg.norm(local(robot_body, 'robot') - local(ref_body, 'motion'), axis=-1).mean(axis=-1)
        final_index = int(sim.state_processor.motion_t[0])
        # MotionLoader resamples to env_dt; its final-frame index is authoritative.
        complete = bool(sim._at_last_paused_frame())
        record = {'motion': str(motion.resolve()), 'policy': str(policy.resolve()), 'seed': seed,
            'motion_sha256': digest(motion), **inputs,
            'prepared_mjcf_sha256': digest(sim.robot_cfg.resolve_mjcf_path()),
            'sim_dt_s': args.sim_dt, 'policy_dt_s': args.env_dt,
            'initial_hold_s': 1., 'final_hold_s': 1., 'last_motion_index': final_index,
            'completed': complete, 'finite': bool(np.isfinite(robot_body).all() and np.isfinite(sim.sim.mj_data.qpos).all()),
            'mean_world_body_error_m': float(error.mean()), 'max_frame_world_body_error_m': float(error.max()),
            'mean_local_body_error_m': float(local_error.mean()),
            'world_root_endpoint_error_m': float(np.linalg.norm(root['robot_end_pos_w'] - root['motion_end_pos_w'])),
            'separate_start_frame_root_error_m': float(root['root_final_error_norm']),
            'min_playback_root_height_m': float(traj['robot_root_pos_w'][:, 2].min()),
            'root_trajectory': str(prefix.with_suffix('.root.npz')),
            'trajectory': str(prefix.with_suffix('.trajectory.npz'))}
        record['passed'] = complete and record['finite'] and record['max_frame_world_body_error_m'] <= .5
        prefix.with_suffix('.metrics.json').write_text(json.dumps(record, indent=2) + '\n')
        records.append(record)
    if not records:
        raise ValueError('Dataset has no motion NPZs')
    summary = {'robot': robot, 'policy': str(policy.resolve()), 'dataset': str(dataset.resolve()),
               'passed': all(r['passed'] for r in records), 'motions': records}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    if not summary['passed']:
        raise SystemExit('One or more motions failed; see summary.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot', required=True)
    p.add_argument('--policy-config', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    evaluate(a.robot, a.policy_config, a.dataset, a.output, a.seed)
