"""Create a Q1 hardware overlay for measured-pose Shaping followed by policy Hold."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
import yaml

from sim2real.config.robots import get_robot_cfg


@dataclass
class Args:
    imu_topic: str
    """Verified sensor_msgs/Imu topic. Observed on the local Q1: /aima/hal/imu/state."""
    imu_to_pelvis_wxyz: tuple[float, float, float, float]
    """Rotation from sensor axes to pelvis axes; identity requires aligned mounting."""
    policy_config: Path = Path('checkpoints/proteus/wsq61/primebot_q1/policy.yaml')
    output_name: str = 'policy-pc.yaml'
    manual_policy_entry: bool = False
    """Allow ] to enter paused policy Hold without Shaping completion/pose gates."""
    enable_shaping_gravity: bool = False
    """Opt in to experimental Shaping feedforward; hardware effort execution is unverified."""
    policy_limit_mode: Literal['clip', 'pass', 'feedforward'] = 'clip'
    """Policy targets beyond the joint range (Shaping targets are always clipped). clip: send the
    bound. pass (experimental): send unchanged. On Q1 use it only with motions whose first frames keep
    the arms away from their limits: a LAFAN T-pose start (arms held at the shoulder-roll limit) made
    the right arm swing violently on 2026-11-06. feedforward (experimental): send the bound plus
    effort kp * (target - bound), faded to zero over the last 0.1 rad before the limit, a virtual
    stop. Earlier evidence suggests the Q1 hardware path may ignore effort; it then acts as clip."""


def prepare(args: Args) -> Path:
    cfg = get_robot_cfg('q1')
    policy = yaml.safe_load(args.policy_config.read_text())
    if policy['policy_joint_names'] != list(cfg.joint_names) or policy['proteus']['robot'] != 'primebot_q1':
        raise ValueError('Expected the 22-joint Primebot Q1 policy')
    if not policy['proteus']['reference_forcing']:
        raise ValueError('Use the hardware policy.yaml with reference_forcing=true')
    rotation = np.asarray(args.imu_to_pelvis_wxyz, dtype=float)
    if rotation.shape != (4,) or not np.isfinite(rotation).all() or np.linalg.norm(rotation) < 1e-6:
        raise ValueError('IMU extrinsic must be a finite nonzero WXYZ quaternion')
    if not args.imu_topic.startswith('/'):
        raise ValueError('Provide an absolute ROS IMU topic')
    policy['robot_io_options'].update(
        imu_topic=args.imu_topic, imu_to_pelvis_wxyz=(rotation / np.linalg.norm(rotation)).tolist())
    policy['shaping_target'] = 'motion_first_frame'
    if args.enable_shaping_gravity:
        policy['shaping_gravity_compensation'] = {'max_effort_fraction': .1, 'ramp_s': 2.}
    else:
        policy.pop('shaping_gravity_compensation', None)
    policy['reference_entry_s'] = 0.0
    limits = {
        name: [float(cfg.joint_pos_lower_limit[name]), float(cfg.joint_pos_upper_limit[name])]
        for name in cfg.joint_names
    }
    policy.pop('soft_joint_pos_limits', None)
    policy.pop('shaping_joint_pos_limits', None)
    # clip: the YAML postprocessor clips as the codec does (an unclipped postprocessor over a
    # clipping codec lets the rate-limit state drift past what the robot receives). Otherwise the
    # codec alone handles out-of-range policy targets and the YAML limits only bound Shaping.
    clip = args.policy_limit_mode == 'clip'
    policy['soft_joint_pos_limits' if clip else 'shaping_joint_pos_limits'] = limits
    policy['robot_io_options'].pop('clip_policy_positions', None)
    policy['robot_io_options']['policy_limit_mode'] = args.policy_limit_mode
    policy['max_target_velocity_rad_s'] = 15.0
    if args.manual_policy_entry:
        policy.pop('policy_entry_max_joint_error_rad', None)
        policy.pop('policy_entry_max_joint_velocity_rad_s', None)
    else:
        policy['policy_entry_max_joint_error_rad'] = 0.10
        policy['policy_entry_max_joint_velocity_rad_s'] = 0.5
    output = args.policy_config.with_name(args.output_name)
    if output.resolve() == args.policy_config.resolve():
        raise ValueError('The overlay must not overwrite the exported policy')
    output.write_text(
        '# Q1 deployment overlay; exported actor, observation contract and gains retained.\n'
        '# Official Q1 model: aligned pelvis IMU; supplied extrinsic must match the driver frame.\n'
        f'# Shaping gravity feedforward enabled: {args.enable_shaping_gravity}.\n'
        + {'clip': '# Policy targets clipped to the joint range (YAML and codec).\n',
           'pass': '# EXPERIMENTAL: policy targets passed beyond the joint range unclipped. On Q1 use only with motions '
                   'that start with the arms down (a T-pose start made the right arm swing, 2026-11-06).\n',
           'feedforward': '# EXPERIMENTAL: out-of-range policy targets sent as range bound + effort feedforward '
                          '(virtual stop at the limit).\n'}[args.policy_limit_mode]
        + ('# Manual entry: ] enters paused policy Hold directly; i is optional; o damps; Space plays.\n'
           if args.manual_policy_entry else
           '# Guarded entry: i shapes for 10 s; ] checks measured pose then enters Hold; o damps.\n')
        + yaml.safe_dump(policy, sort_keys=False))
    return output


if __name__ == '__main__':
    print(prepare(tyro.cli(Args)).resolve())
