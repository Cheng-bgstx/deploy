"""A3 body_drive ROS transport, mapped by names independently of policy order.

Wire schema/topics: AgibotTech/sonic_for_a3 gear_sonic_deploy thirdparty/joint_msgs
and a3_aimrt_config.ros2.yaml. Head pitch is absent from Proteus's 30-DOF contract;
the physical 31st motor is explicitly held at zero and damped on stop.
"""
from sim2real.rl_policy.robot_io.ros2_named import NamedRosRobotIO


def groups(cfg):
    return {
        'waist': tuple(f'waist_{axis}_joint' for axis in ('yaw', 'roll', 'pitch')),
        'neck': ('head_yaw_joint', 'head_pitch_joint'),
        'arm': tuple(f'{s}_{j}_joint' for s in ('left', 'right') for j in
                     ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                      'wrist_roll', 'wrist_pitch', 'wrist_yaw')),
        'leg': tuple(f'{s}_{j}_joint' for s in ('left', 'right') for j in
                     ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')),
    }


class A3RobotIO(NamedRosRobotIO):
    def __init__(self, cfg, *, imu_topic='/body_drive/pelvis_imu/data', publish_hz=500., **options):
        if cfg.name != 'a3':
            raise ValueError('A3 backend requires robot=a3')
        from joint_msgs.msg import JointCommand, Command, JointState
        super().__init__(cfg, groups=groups(cfg),
            topics={g: (f'/body_drive/{g}_joint_state', f'/body_drive/{g}_joint_command') for g in groups(cfg)},
            types=(JointCommand, Command, JointState), imu_topic=imu_topic, publish_hz=publish_hz,
            physical_limits={'left_knee_joint': (-.122173, 2.49582), 'right_knee_joint': (-.122173, 2.49582)},
            fixed_joints={'head_pitch_joint': (0., 0., 0., 3., .3)}, **options)
