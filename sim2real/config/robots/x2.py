"""Proteus lingxi_x2: 31 joint X2 Ultra, vendor v1.3 model with training limits.

Joint table comes from Proteus wamseq's spec and contract. Assets are staged by
scripts/x2/prepare_assets.py; importing this module needs no vendor files.
"""
from sim2real.config.robots.base import PROJECT_ROOT, RobotCfg

X2_MJCF_PATH = PROJECT_ROOT / 'third_party/lingxi_x2/model/scene_tracking.xml'
# lower, upper, velocity, effort, armature, kp, kd, default.
_JOINTS = {
    'left_hip_pitch_joint': (-2.704, 2.556, 32.3508, 120.0, 0.03, 49.2884, 3.1038, -0.08132395153125747),
    'left_hip_roll_joint': (-0.235, 2.906, 20.2193, 120.0, 0.03, 121.5654, 7.6552, 0.00834874456668256),
    'left_hip_yaw_joint': (-1.684, 3.43, 32.3508, 120.0, 0.03, 49.2884, 3.1038, 0.09378353417357031),
    'left_knee_joint': (0.0, 2.4073, 20.2193, 120.0, 0.03, 121.5654, 7.6552, 0.2901824921941415),
    'left_ankle_pitch_joint': (-0.803, 0.453, 37.4057, 36.0, 0.03, 34.9629, 2.2017, -0.13343288520374186),
    'left_ankle_roll_joint': (-0.262, 0.262, 37.4057, 24.0, 0.03, 34.9629, 2.2017, 0.005907513006630611),
    'right_hip_pitch_joint': (-2.704, 2.556, 32.3508, 120.0, 0.03, 49.2884, 3.1038, -0.04242681915842901),
    'right_hip_roll_joint': (-2.906, 0.235, 20.2193, 120.0, 0.03, 121.5654, 7.6552, 0.026747118450244953),
    'right_hip_yaw_joint': (-3.43, 1.684, 32.3508, 120.0, 0.03, 49.2884, 3.1038, -0.17058119859866486),
    'right_knee_joint': (0.0, 2.4073, 20.2193, 120.0, 0.03, 121.5654, 7.6552, 0.2624317885690089),
    'right_ankle_pitch_joint': (-0.803, 0.453, 37.4057, 36.0, 0.03, 34.9629, 2.2017, -0.17586325623500032),
    'right_ankle_roll_joint': (-0.2625, 0.2625, 37.4057, 24.0, 0.03, 34.9629, 2.2017, -0.004506753493659759),
    'waist_yaw_joint': (-3.43, 2.382, 32.3508, 120.0, 0.03, 122.6713, 2.4268, 0.02980913936192675),
    'waist_pitch_joint': (-0.314, 0.314, 37.4057, 48.0, 0.03, 368.014, 6.067, 0.1327515359270567),
    'waist_roll_joint': (-0.488, 0.488, 37.4057, 48.0, 0.03, 368.014, 6.067, 0.01753147356728804),
    'left_shoulder_pitch_joint': (-3.08, 2.04, 37.4057, 36.0, 0.03, 17.4814, 1.1008, -0.3001376962073018),
    'left_shoulder_roll_joint': (-0.061, 2.993, 37.4057, 36.0, 0.03, 17.4814, 1.1008, 0.14278627458116303),
    'left_shoulder_yaw_joint': (-2.556, 2.556, 37.4057, 24.0, 0.03, 17.4814, 1.1008, -0.37570623265347036),
    'left_elbow_joint': (-2.3556, 0.0, 37.4057, 24.0, 0.03, 17.4814, 1.1008, -0.5397383782826883),
    'left_wrist_yaw_joint': (-2.556, 2.556, 22.2412, 24.0, 0.03, 20.5822, 1.2961, -0.30141807997063075),
    'left_wrist_pitch_joint': (-0.558, 0.558, 22.2412, 4.8, 0.03, 20.5822, 1.2961, 0.0),
    'left_wrist_roll_joint': (-1.571, 0.724, 37.4057, 4.8, 0.03, 17.4814, 1.1008, 0.0),
    'right_shoulder_pitch_joint': (-3.08, 2.04, 37.4057, 36.0, 0.03, 17.4814, 1.1008, 0.027099450198416593),
    'right_shoulder_roll_joint': (-2.993, 0.061, 37.4057, 36.0, 0.03, 17.4814, 1.1008, -0.14501399806355392),
    'right_shoulder_yaw_joint': (-2.556, 2.556, 37.4057, 24.0, 0.03, 17.4814, 1.1008, 0.48725540098972797),
    'right_elbow_joint': (-2.3556, 0.0, 37.4057, 24.0, 0.03, 17.4814, 1.1008, -0.528216281340766),
    'right_wrist_yaw_joint': (-2.556, 2.556, 22.2412, 24.0, 0.03, 20.5822, 1.2961, 0.15573446416603554),
    'right_wrist_pitch_joint': (-0.558, 0.558, 22.2412, 4.8, 0.03, 20.5822, 1.2961, 0.0),
    'right_wrist_roll_joint': (-0.724, 1.571, 37.4057, 4.8, 0.03, 17.4814, 1.1008, 0.0),
    'head_yaw_joint': (-0.366, 0.366, 22.2412, 2.6, 0.03, 20.5822, 1.2961, 0.0),
    'head_pitch_joint': (-0.3838, 0.3838, 22.2412, 0.6, 0.03, 20.5822, 1.2961, 0.0),
}
X2_JOINT_NAMES = tuple(_JOINTS)
X2_BODY_NAMES = (
    'pelvis',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'waist_yaw_link',
    'waist_pitch_link',
    'torso_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_yaw_link',
    'left_wrist_pitch_link',
    'left_wrist_roll_link',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_yaw_link',
    'right_wrist_pitch_link',
    'right_wrist_roll_link',
    'head_yaw_link',
    'head_pitch_link',
)

# AimDK arrays use this documented order, independently of MJCF / policy order.
X2_AIMDK_GROUPS = {
    'leg': tuple(f'{s}_{j}_joint' for s in ('left', 'right') for j in
                 ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')),
    'waist': ('waist_yaw_joint', 'waist_pitch_joint', 'waist_roll_joint'),
    'arm': tuple(f'{s}_{j}_joint' for s in ('left', 'right') for j in
                 ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_yaw', 'wrist_pitch', 'wrist_roll')),
    'head': ('head_yaw_joint', 'head_pitch_joint'),
}

X2_CFG = RobotCfg(
    name='x2', joint_names=X2_JOINT_NAMES, body_names=X2_BODY_NAMES,
    joint_pos_lower_limit={n: v[0] for n, v in _JOINTS.items()},
    joint_pos_upper_limit={n: v[1] for n, v in _JOINTS.items()},
    joint_velocity_limit={n: v[2] for n, v in _JOINTS.items()},
    joint_effort_limit={n: v[3] for n, v in _JOINTS.items()},
    joint_armature={n: v[4] for n, v in _JOINTS.items()},
    safe_joint_kp={n: v[5] for n, v in _JOINTS.items()},
    safe_joint_kd={n: v[6] for n, v in _JOINTS.items()},
    joint_frictionloss={n: 0.0 for n in _JOINTS}, mjcf_path=X2_MJCF_PATH,
    default_qpos=(0., 0., 0.6734864235648271, 1., 0., 0., 0., *(v[7] for v in _JOINTS.values())),
    root_joint_names=('floating_base_joint',), viewer_track_body_names=('pelvis',),
    elastic_band_attach_body_names=('torso_link', 'pelvis'),
)

# Physical v1.3 limits stay distinct from the reconstructed training scene.
X2_VENDOR_LIMITS = {
    'left_hip_pitch_joint': (-2.704, 2.556),
    'left_hip_roll_joint': (-0.235, 2.906),
    'left_hip_yaw_joint': (-1.8588, 3.6041),
    'left_knee_joint': (0.0, 2.4073),
    'left_ankle_pitch_joint': (-0.803, 0.453),
    'left_ankle_roll_joint': (-0.2967, 0.2967),
    'right_hip_pitch_joint': (-2.704, 2.556),
    'right_hip_roll_joint': (-2.906, 0.235),
    'right_hip_yaw_joint': (-3.6041, 1.8588),
    'right_knee_joint': (0.0, 2.4073),
    'right_ankle_pitch_joint': (-0.803, 0.453),
    'right_ankle_roll_joint': (-0.2967, 0.2967),
    'waist_yaw_joint': (-3.43, 2.2078),
    'waist_pitch_joint': (-0.314, 0.314),
    'waist_roll_joint': (-0.488, 0.488),
    'left_shoulder_pitch_joint': (-3.08, 2.04),
    'left_shoulder_roll_joint': (-0.061, 3.0456),
    'left_shoulder_yaw_joint': (-2.556, 2.556),
    'left_elbow_joint': (-2.3556, 0.0),
    'left_wrist_yaw_joint': (-2.556, 2.556),
    'left_wrist_pitch_joint': (-0.5236, 0.5236),
    'left_wrist_roll_joint': (-1.5097, 0.724),
    'right_shoulder_pitch_joint': (-3.08, 2.04),
    'right_shoulder_roll_joint': (-3.0456, 0.061),
    'right_shoulder_yaw_joint': (-2.556, 2.556),
    'right_elbow_joint': (-2.3556, 0.0),
    'right_wrist_yaw_joint': (-2.556, 2.556),
    'right_wrist_pitch_joint': (-0.5236, 0.5236),
    'right_wrist_roll_joint': (-0.724, 1.5097),
    'head_yaw_joint': (-0.349, 0.349),
    'head_pitch_joint': (-0.3838, 0.3838),
}


def x2_deploy_joint_limits(cfg: RobotCfg = X2_CFG) -> dict[str, tuple[float, float]]:
    """Training limits narrowed to the physical v1.3 limits: the widest targets the hardware accepts."""
    return {n: (max(cfg.joint_pos_lower_limit[n], X2_VENDOR_LIMITS[n][0]),
                min(cfg.joint_pos_upper_limit[n], X2_VENDOR_LIMITS[n][1])) for n in cfg.joint_names}
