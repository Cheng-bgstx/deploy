"""Load a frozen Proteus robot spec without needing the training checkout or assets."""
from pathlib import Path

import yaml

from sim2real.config.robots.base import PROJECT_ROOT, RobotCfg


def from_proteus_spec(name: str, contract: str, body_names: tuple[str, ...]) -> RobotCfg:
    spec = yaml.safe_load((Path(__file__).with_name('specs') / f'{contract}.yaml').read_text())
    joints = spec['joints']
    root = spec['semantic_bodies']['pelvis']
    return RobotCfg(
        name=name, joint_names=tuple(joints), body_names=body_names,
        joint_pos_lower_limit={n: v['lower'] for n, v in joints.items()},
        joint_pos_upper_limit={n: v['upper'] for n, v in joints.items()},
        joint_velocity_limit={n: v['velocity'] for n, v in joints.items()},
        joint_effort_limit={n: v['effort'] for n, v in joints.items()},
        joint_armature={n: v['armature'] for n, v in joints.items()},
        safe_joint_kp={n: v['stiffness'] for n, v in joints.items()},
        safe_joint_kd={n: v['damping'] for n, v in joints.items()},
        joint_frictionloss={n: 0. for n in joints}, self_collisions=False, ground_friction=1.,
        mjcf_path=PROJECT_ROOT / 'third_party' / spec['mjcf'],
        default_qpos=(0., 0., spec['root_height'], 1., 0., 0., 0.,
                      *(v['default'] for v in joints.values())),
        root_joint_names=('floating_base_joint', 'root', 'free_joint', 'floating_base'),
        viewer_track_body_names=(root,),
        elastic_band_attach_body_names=(spec['semantic_bodies']['torso'], root),
    )
