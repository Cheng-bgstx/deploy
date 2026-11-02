#!/usr/bin/env python3
"""Summarize the exact trajectories written while rendering an X2 sim2sim video."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from sim2real.utils.math import quat_rotate_inverse_numpy


def summarize(video: Path):
    trajectory = np.load(video.with_suffix('.trajectory.npz'))
    root = np.load(video.with_suffix('.root.npz'))
    cfg = yaml.safe_load(Path(str(trajectory['policy_config'])).read_text())
    all_names = trajectory['body_names'].tolist()
    names = cfg['proteus']['body_names']
    indices = [all_names.index(n) for n in names]
    robot = trajectory['robot_body_pos_w'][:, indices]
    motion = trajectory['motion_body_pos_w'][:, indices]
    def local(body, prefix):
        quat = trajectory[f'{prefix}_root_quat_w']
        centered = body - trajectory[f'{prefix}_root_pos_w'][:, None]
        return quat_rotate_inverse_numpy(np.broadcast_to(quat[:, None], (*centered.shape[:2], 4)).copy(), centered)
    world_error = np.linalg.norm(robot - motion, axis=-1).mean(axis=-1)
    local_error = np.linalg.norm(local(robot, 'robot') - local(motion, 'motion'), axis=-1).mean(axis=-1)
    report = {'video': str(video), 'policy_config': str(trajectory['policy_config']),
              'motion_path': str(trajectory['motion_path']), 'seed': int(trajectory['seed']),
              'reference_forcing': cfg['proteus']['reference_forcing'],
              'tracked_body_names': names,
              'mean_tracked_body_world_error_m': float(world_error.mean()),
              'max_frame_mean_tracked_body_world_error_m': float(world_error.max()),
              'mean_tracked_body_local_error_m': float(local_error.mean()),
              'max_frame_mean_tracked_body_local_error_m': float(local_error.max()),
              'separate_start_frame_root_endpoint_error_m': float(root['root_final_error_norm']),
              'world_root_endpoint_error_m': float(np.linalg.norm(root['robot_end_pos_w'] - root['motion_end_pos_w'])),
              'common_frame_root_displacement_error_m': float(np.linalg.norm(
                  (root['robot_end_pos_w'] - root['robot_start_pos_w'])
                  - (root['motion_end_pos_w'] - root['motion_start_pos_w']))),
              'robot_relative_final_pos': root['robot_relative_final_pos'].tolist(),
              'motion_relative_final_pos': root['motion_relative_final_pos'].tolist(),
              'min_playback_root_height_m': float(trajectory['robot_root_pos_w'][:, 2].min()),
              'max_playback_root_height_m': float(trajectory['robot_root_pos_w'][:, 2].max())}
    video.with_suffix('.metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', type=Path)
    summarize(p.parse_args().video)
