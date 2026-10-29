from __future__ import annotations

import os

from sim2real.config.robots.base import RobotCfg

# The ET1 model ships with Galbot's internal et1-dep delivery and is not vendored here.
ET1_DEP_ROOT = os.path.expanduser(os.environ.get("ET1_DEP_ROOT", "~/et1-dep"))

# Joint order follows et1-dep (configs/robots/models.toml), which is also the MJCF order.
ET1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
)

# Motor groups by reflected inertia; values match Proteus galbot_et1 (ScaleTrack spec)
# and Galbot's own ET1 tracker gains, which were derived from the same armature.
_LEG_ARMATURE = 0.01889931491
_ANKLE_PITCH_ARMATURE = 0.01491796575
_ANKLE_ROLL_ARMATURE = 0.0061870700719
_SMALL_ARMATURE = 0.0039785307809


def _group(joint_name: str) -> str:
    if "ankle_pitch" in joint_name:
        return "ankle_pitch"
    if "ankle_roll" in joint_name:
        return "ankle_roll"
    if any(key in joint_name for key in ("shoulder_yaw", "elbow", "wrist", "head")):
        return "small"
    return "large"


_ARMATURE = {
    "large": _LEG_ARMATURE,
    "ankle_pitch": _ANKLE_PITCH_ARMATURE,
    "ankle_roll": _ANKLE_ROLL_ARMATURE,
    "small": _SMALL_ARMATURE,
}
_EFFORT = {"large": 60.0, "ankle_pitch": 78.1286081571, "ankle_roll": 50.3149967424, "small": 31.7}
_VELOCITY = {"large": 38.125, "ankle_pitch": 24.053, "ankle_roll": 24.053, "small": 54.2}
_KP = {"large": 74.61150464, "ankle_pitch": 58.89376817, "ankle_roll": 24.4255736, "small": 15.70661}
_KD = {"large": 4.74991591, "ankle_pitch": 3.74929373, "ankle_roll": 1.55498031, "small": 0.999914}

# Ranges from et1-dep galbot_et1_26dof.xml. waist_yaw is +-0.785 there, tighter than
# the +-pi/2 in the Proteus training spec; the hardware model wins.
_LIMITS = {
    "left_hip_pitch_joint": (-2.9322, 2.5482),
    "left_hip_roll_joint": (-1.0123, 1.7454),
    "left_hip_yaw_joint": (-2.7402, 2.7402),
    "left_knee_joint": (-0.1745, 2.4260),
    "left_ankle_pitch_joint": (-0.8727, 0.5760),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.9322, 2.5482),
    "right_hip_roll_joint": (-1.7454, 1.0123),
    "right_hip_yaw_joint": (-2.7402, 2.7402),
    "right_knee_joint": (-0.1745, 2.4260),
    "right_ankle_pitch_joint": (-0.8727, 0.5760),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_roll_joint": (-0.4363, 0.4363),
    "waist_yaw_joint": (-0.7854, 0.7854),
    "left_shoulder_pitch_joint": (-3.1416, 2.0944),
    "left_shoulder_roll_joint": (-0.2600, 2.4400),
    "left_shoulder_yaw_joint": (-1.9100, 1.9100),
    "left_elbow_joint": (-0.4400, 2.2600),
    "left_wrist_roll_joint": (-1.9100, 1.9100),
    "right_shoulder_pitch_joint": (-3.1416, 2.0944),
    "right_shoulder_roll_joint": (-2.4400, 0.2600),
    "right_shoulder_yaw_joint": (-1.9100, 1.9100),
    "right_elbow_joint": (-0.4400, 2.2600),
    "right_wrist_roll_joint": (-1.9100, 1.9100),
    "head_pitch_joint": (-0.5200, 0.5200),
    "head_yaw_joint": (-1.2600, 1.2600),
}

# Galbot's ET1 walker/tracker standing pose; head centred. Root height puts the
# foot capsules on the floor.
_STAND_JOINT_POS = (
    -0.2, 0.13, 0.0, 0.33, -0.15, -0.12,
    -0.2, -0.13, 0.0, 0.33, -0.15, 0.12,
    0.0, 0.0,
    0.0, 0.2, 0.0, 1.5, 0.0,
    0.0, -0.2, 0.0, 1.5, 0.0,
    0.0, 0.0,
)  # fmt: skip

ET1_CFG = RobotCfg(
    name="galbot_et1",
    joint_names=ET1_JOINT_NAMES,
    body_names=(
        "pelvis_link",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_roll_link",
        "waist_yaw_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "head_pitch_link",
        "head_yaw_link",
    ),
    joint_pos_lower_limit={name: _LIMITS[name][0] for name in ET1_JOINT_NAMES},
    joint_pos_upper_limit={name: _LIMITS[name][1] for name in ET1_JOINT_NAMES},
    joint_velocity_limit={name: _VELOCITY[_group(name)] for name in ET1_JOINT_NAMES},
    joint_effort_limit={name: _EFFORT[_group(name)] for name in ET1_JOINT_NAMES},
    safe_joint_kp={name: _KP[_group(name)] for name in ET1_JOINT_NAMES},
    safe_joint_kd={name: _KD[_group(name)] for name in ET1_JOINT_NAMES},
    joint_armature={name: _ARMATURE[_group(name)] for name in ET1_JOINT_NAMES},
    # et1-dep's MJCF value; Proteus does not specify joint friction.
    joint_frictionloss={name: 0.1 for name in ET1_JOINT_NAMES},
    mjcf_path=f"{ET1_DEP_ROOT}/storage/humanoid_description/galbot_et1/galbot_et1_26dof.xml",
    default_qpos=(0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0, *_STAND_JOINT_POS),
    publish_hz=50.0,
    domain_id=0,
    interface="eth0",
    mocap_ip="localhost",
    viewer_track_body_names=("pelvis_link",),
    elastic_band_attach_body_names=("waist_yaw_link", "pelvis_link"),
)
