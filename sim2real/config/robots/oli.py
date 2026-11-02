"""LimX OLI EDU / HU_D04_01 (31 virtual PR joints).

Joint parameters and reference pose: Proteus wamseq limx_oli_edu spec / contract.
MJCF: official humanoid-description/HU_D04_description/xml/HU_D04_01_vis.xml.
Prepare the model with scripts/oli/prepare_assets.py; imports need no external files.
"""

from sim2real.config.robots.base import PROJECT_ROOT, RobotCfg

OLI_MJCF_PATH = PROJECT_ROOT / "third_party/limx_oli_edu/model/xml/HU_D04_01_vis.xml"
# lower, upper, velocity, effort, armature, kp, kd, default (radians / SI units).
_JOINTS = {
    'left_hip_pitch_joint': (-2.618, 2.618, 27.4487, 188.4995, 0.159221, 86.0655, 6.3876, -0.2152299306449577),
    'left_hip_roll_joint': (-0.8727, 2.094, 17.1555, 297.7435, 0.159221, 212.2727, 15.7544, -0.012409628451430142),
    'left_hip_yaw_joint': (-2.618, 2.618, 27.4487, 188.4995, 0.159221, 86.0655, 6.3876, -0.07469176021935806),
    'left_knee_joint': (-0.087, 2.583, 17.1555, 297.7435, 0.159221, 212.2727, 15.7544, 0.20239833317813333),
    'left_ankle_pitch_joint': (-1.012, 0.611, 31.7376, 107.102, 0.083508, 61.0508, 4.5311, 0.054395573584187264),
    'left_ankle_roll_joint': (-0.436, 0.436, 31.7376, 107.102, 0.083508, 61.0508, 4.5311, -0.019139711419921074),
    'right_hip_pitch_joint': (-2.618, 2.618, 27.4487, 188.4995, 0.159221, 86.0655, 6.3876, -0.16911575732951475),
    'right_hip_roll_joint': (-2.094, 0.8727, 17.1555, 297.7435, 0.159221, 212.2727, 15.7544, -0.025242483781646494),
    'right_hip_yaw_joint': (-2.618, 2.618, 27.4487, 188.4995, 0.159221, 86.0655, 6.3876, -0.13607069278208678),
    'right_knee_joint': (-0.087, 2.583, 17.1555, 297.7435, 0.159221, 212.2727, 15.7544, 0.17689448269278923),
    'right_ankle_pitch_joint': (-1.012, 0.611, 31.7376, 107.102, 0.083508, 61.0508, 4.5311, 0.026004541185623127),
    'right_ankle_roll_joint': (-0.436, 0.436, 31.7376, 107.102, 0.083508, 61.0508, 4.5311, 0.08597450878472773),
    'waist_yaw_joint': (-2.618, 2.618, 27.4487, 188.4995, 0.083508, 214.204, 4.9944, 0.018966887411214214),
    'waist_roll_joint': (-0.524, 0.524, 31.7376, 107.102, 0.083508, 642.6119, 12.4861, 0.03319488111162146),
    'waist_pitch_joint': (-0.524, 0.785, 31.7376, 107.102, 0.083508, 642.6119, 12.4861, 0.10473259371329396),
    'head_yaw_joint': (-0.785, 0.785, 18.871, 10.7102, 0.016608, 35.9398, 2.6674, 0.0),
    'head_pitch_joint': (-0.524, 0.785, 18.871, 10.7102, 0.016608, 35.9398, 2.6674, 0.0),
    'left_shoulder_pitch_joint': (-2.792, 2.792, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, 0.002633487368502903),
    'left_shoulder_roll_joint': (-0.087, 3.229, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, 0.4130260026259984),
    'left_shoulder_yaw_joint': (-3.403, 1.571, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, -0.6813304335299093),
    'left_elbow_joint': (-2.443, 1.571, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, -0.7560585932690065),
    'left_wrist_yaw_joint': (-2.356, 0.785, 18.871, 10.7102, 0.016608, 35.9398, 2.6674, -0.3139477406041372),
    'left_wrist_pitch_joint': (-1.571, 1.571, 18.871, 10.7102, 0.016608, 35.9398, 2.6674, 0.0769812076514341),
    'left_wrist_roll_joint': (-1.571, 1.571, 31.7376, 53.551, 0.016608, 30.5254, 2.2655, 0.32689465153092034),
    'right_shoulder_pitch_joint': (-2.792, 2.792, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, 0.04166652018987524),
    'right_shoulder_roll_joint': (-3.229, 0.087, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, -0.46315897158266767),
    'right_shoulder_yaw_joint': (-1.571, 3.403, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, 0.554498064453121),
    'right_elbow_joint': (-2.443, 1.571, 31.7376, 53.551, 0.040771, 30.5254, 2.2655, -0.7617507442003947),
    'right_wrist_yaw_joint': (-0.785, 2.356, 18.871, 10.7102, 0.016608, 35.9398, 2.6674, 0.2220739714376042),
    'right_wrist_pitch_joint': (-1.571, 1.571, 18.871, 10.7102, 0.016608, 35.9398, 2.6674, 0.07205642688528566),
    'right_wrist_roll_joint': (-1.571, 1.571, 31.7376, 53.551, 0.016608, 30.5254, 2.2655, -0.2730601343058918),
}

OLI_JOINT_NAMES = tuple(_JOINTS)
OLI_BODY_NAMES = (
    'base_link',
    'left_hip_pitch_link',
    'left_hip_roll_link',
    'left_hip_yaw_link',
    'left_knee_link',
    'left_ankle_pitch_link',
    'left_ankle_roll_link',
    'contact_foot_heel_L',
    'contact_foot_center_L',
    'contact_foot_tip_L',
    'right_hip_pitch_link',
    'right_hip_roll_link',
    'right_hip_yaw_link',
    'right_knee_link',
    'right_ankle_pitch_link',
    'right_ankle_roll_link',
    'contact_foot_heel_R',
    'contact_foot_center_R',
    'contact_foot_tip_R',
    'waist_yaw_link',
    'waist_roll_link',
    'waist_pitch_link',
    'head_yaw_link',
    'head_pitch_link',
    'left_shoulder_pitch_link',
    'left_shoulder_roll_link',
    'left_shoulder_yaw_link',
    'left_elbow_link',
    'left_wrist_yaw_link',
    'left_wrist_pitch_link',
    'left_wrist_roll_link',
    'left_hand_contact',
    'left_hand_manip',
    'right_shoulder_pitch_link',
    'right_shoulder_roll_link',
    'right_shoulder_yaw_link',
    'right_elbow_link',
    'right_wrist_yaw_link',
    'right_wrist_pitch_link',
    'right_wrist_roll_link',
    'right_hand_contact',
    'right_hand_manip',
)

# Official controller array order. Head pitch precedes yaw here; MJCF has yaw before pitch.
OLI_SDK_JOINT_NAMES = (
    *OLI_JOINT_NAMES[:15], "head_pitch_joint", "head_yaw_joint",
    *(f"{side}_{part}_joint" for side in ("left", "right") for part in
      ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")),
)

OLI_CFG = RobotCfg(
    name="oli",
    joint_names=OLI_JOINT_NAMES,
    body_names=OLI_BODY_NAMES,
    joint_pos_lower_limit={n: v[0] for n, v in _JOINTS.items()},
    joint_pos_upper_limit={n: v[1] for n, v in _JOINTS.items()},
    joint_velocity_limit={n: v[2] for n, v in _JOINTS.items()},
    joint_effort_limit={n: v[3] for n, v in _JOINTS.items()},
    joint_armature={n: v[4] for n, v in _JOINTS.items()},
    safe_joint_kp={n: v[5] for n, v in _JOINTS.items()},
    safe_joint_kd={n: v[6] for n, v in _JOINTS.items()},
    joint_frictionloss={n: 0.0 for n in _JOINTS},
    mjcf_path=OLI_MJCF_PATH,
    default_qpos=(0.0, 0.0, 0.9282088749221369, 1.0, 0.0, 0.0, 0.0, *(v[7] for v in _JOINTS.values())),
    root_joint_names=("floating_base_joint",),
    viewer_track_body_names=("base_link",),
    elastic_band_attach_body_names=("waist_pitch_link", "base_link"),
)
