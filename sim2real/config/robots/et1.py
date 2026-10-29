"""Galbot ET1: a 26-DoF humanoid derived from the Unitree R1 (same joint names and kinematics).

The MJCF is et1-dep's vendor model (``storage/humanoid_description/galbot_et1``) with the elbow and wrist frames
of the Unitree R1, written to ``third_party/galbot_et1/`` (a runtime artifact) by ``scripts/et1/prepare_mjcf.py``.
The values below are copied from it so that importing the config never needs the file. The robot speaks Unitree
HG DDS (``rt/lowcmd`` / ``rt/lowstate``) with 26 of the 35 motor slots in use, see
``galbot_et1_joint_index_dds.md`` next to the MJCF.
"""

from __future__ import annotations

import math

from sim2real.config.robots.base import PROJECT_ROOT, RobotCfg


ET1_MJCF_PATH = PROJECT_ROOT / "third_party" / "galbot_et1" / "galbot_et1_26dof_unitree_frames.xml"

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

ET1_BODY_NAMES = (
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
)

# Unitree HG motor slot of every joint (IDL slots 14, 20, 21, 27 and 28 are empty on the ET1).
ET1_UNITREE_MOTOR_INDICES = (*range(0, 14), *range(15, 20), *range(22, 27), 29, 30)

ET1_JOINT_RANGES = {
    "left_hip_pitch_joint": (-2.9322, 2.5482),
    "left_hip_roll_joint": (-1.0123, 1.7454),
    "left_hip_yaw_joint": (-2.7402, 2.7402),
    "left_knee_joint": (-0.1745, 2.426),
    "left_ankle_pitch_joint": (-0.8726, 0.576),
    "left_ankle_roll_joint": (-0.2618, 0.2618),
    "right_hip_pitch_joint": (-2.9322, 2.5482),
    "right_hip_roll_joint": (-1.7454, 1.0123),
    "right_hip_yaw_joint": (-2.7402, 2.7402),
    "right_knee_joint": (-0.1745, 2.426),
    "right_ankle_pitch_joint": (-0.8726, 0.576),
    "right_ankle_roll_joint": (-0.2618, 0.2618),
    "waist_roll_joint": (-0.4364, 0.4364),
    "waist_yaw_joint": (-0.785, 0.785),
    "left_shoulder_pitch_joint": (-3.14159, 2.0944),
    "left_shoulder_roll_joint": (-0.26, 2.44),
    "left_shoulder_yaw_joint": (-1.91, 1.91),
    "left_elbow_joint": (-0.44, 2.26),
    "left_wrist_roll_joint": (-1.91, 1.91),
    "right_shoulder_pitch_joint": (-3.14159, 2.0944),
    "right_shoulder_roll_joint": (-2.44, 0.26),
    "right_shoulder_yaw_joint": (-1.91, 1.91),
    "right_elbow_joint": (-0.44, 2.26),
    "right_wrist_roll_joint": (-1.91, 1.91),
    "head_pitch_joint": (-0.52, 0.52),
    "head_yaw_joint": (-1.26, 1.26),
}


def _by_group(*, leg: float, ankle: float, waist: float, shoulder: float, arm: float, head_pitch: float,
              head_yaw: float) -> dict[str, float]:
    def value(name: str) -> float:
        if "_ankle_" in name:
            return ankle
        if name.startswith(("left_hip", "right_hip", "left_knee", "right_knee")):
            return leg
        if name.startswith("waist_"):
            return waist
        if "_shoulder_pitch" in name or "_shoulder_roll" in name:
            return shoulder
        if name == "head_pitch_joint":
            return head_pitch
        if name == "head_yaw_joint":
            return head_yaw
        return arm  # shoulder yaw, elbow, wrist roll

    return {name: value(name) for name in ET1_JOINT_NAMES}


# Motor torque limits (the MJCF actuator ctrlrange).
ET1_JOINT_EFFORT_LIMIT = _by_group(leg=60.0, ankle=50.0, waist=60.0, shoulder=60.0, arm=33.0, head_pitch=33.0,
                                   head_yaw=3.1)  # fmt: skip

# The vendor MJCF gives every joint a placeholder armature of 0.01. The ET1 walker and tracker shipped in
# et1-dep were trained with kp = armature * (2 pi * 10 Hz)^2 (and kd = 4 * armature * 2 pi * 10 Hz), so their
# gains give the armatures used here; the head takes Loco's uncovered-joint kp.
_NATURAL_FREQ = 10.0 * 2.0 * math.pi
_ARMATURE_LARGE = 74.61150464105879 / _NATURAL_FREQ**2  # hips, knees, waist, shoulder pitch / roll
_ARMATURE_ANKLE_PITCH = 58.893768166012336 / _NATURAL_FREQ**2
_ARMATURE_ANKLE_ROLL = 24.425573603154323 / _NATURAL_FREQ**2
_ARMATURE_ARM = 15.706609961074328 / _NATURAL_FREQ**2  # shoulder yaw, elbow, wrist roll, head


def _joint_armature() -> dict[str, float]:
    armature = _by_group(leg=_ARMATURE_LARGE, ankle=_ARMATURE_ANKLE_PITCH, waist=_ARMATURE_LARGE,
                         shoulder=_ARMATURE_LARGE, arm=_ARMATURE_ARM, head_pitch=_ARMATURE_ARM,
                         head_yaw=_ARMATURE_ARM)  # fmt: skip
    for side in ("left", "right"):
        armature[f"{side}_ankle_roll_joint"] = _ARMATURE_ANKLE_ROLL
    return armature


# Standing pose of the et1-dep Walker (head straight); the root height puts the feet on the ground.
ET1_DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.2,
    "left_hip_roll_joint": 0.13,
    "left_knee_joint": 0.33,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": -0.12,
    "right_hip_pitch_joint": -0.2,
    "right_hip_roll_joint": -0.13,
    "right_knee_joint": 0.33,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.12,
    "left_shoulder_roll_joint": 0.2,
    "left_elbow_joint": 1.5,
    "right_shoulder_roll_joint": -0.2,
    "right_elbow_joint": 1.5,
}
ET1_DEFAULT_ROOT_HEIGHT = 0.74

# Hold gains of galbot_et1_joint_index_dds.md.
SAFE_JOINT_KP = {
    ".*_hip_.*_joint": 80.0,
    ".*_knee_joint": 80.0,
    ".*_ankle_.*_joint": 40.0,
    "waist_.*_joint": 80.0,
    ".*_shoulder_(pitch|roll)_joint": 60.0,
    ".*_(shoulder_yaw|elbow|wrist_roll)_joint": 40.0,
    "head_pitch_joint": 40.0,
    "head_yaw_joint": 10.0,
}
SAFE_JOINT_KD = {
    ".*_hip_.*_joint": 5.0,
    ".*_knee_joint": 5.0,
    ".*_ankle_.*_joint": 3.0,
    "waist_.*_joint": 5.0,
    ".*_shoulder_(pitch|roll)_joint": 4.0,
    ".*_(shoulder_yaw|elbow|wrist_roll)_joint": 3.0,
    "head_pitch_joint": 3.0,
    "head_yaw_joint": 0.1,
}


ET1_CFG = RobotCfg(
    name="et1",
    joint_names=ET1_JOINT_NAMES,
    body_names=ET1_BODY_NAMES,
    joint_pos_lower_limit={name: lower for name, (lower, _) in ET1_JOINT_RANGES.items()},
    joint_pos_upper_limit={name: upper for name, (_, upper) in ET1_JOINT_RANGES.items()},
    # Unitree R1 velocities (the G1 ones of the same motors), as in Proteus's R1 description.
    joint_velocity_limit=_by_group(leg=32.0, ankle=37.0, waist=32.0, shoulder=37.0, arm=37.0, head_pitch=22.0,
                                   head_yaw=22.0),  # fmt: skip
    joint_effort_limit=ET1_JOINT_EFFORT_LIMIT,
    safe_joint_kp=SAFE_JOINT_KP,
    safe_joint_kd=SAFE_JOINT_KD,
    joint_armature=_joint_armature(),
    joint_frictionloss={name: 0.1 for name in ET1_JOINT_NAMES},  # the MJCF's joint default
    mjcf_path=ET1_MJCF_PATH,
    default_qpos=(
        0.0,
        0.0,
        ET1_DEFAULT_ROOT_HEIGHT,
        1.0,
        0.0,
        0.0,
        0.0,
        *(ET1_DEFAULT_JOINT_POS.get(name, 0.0) for name in ET1_JOINT_NAMES),
    ),
    unitree_motor_indices=ET1_UNITREE_MOTOR_INDICES,
    unitree_mode_machine=None,
    root_joint_names=("floating_base_joint",),
    viewer_track_body_names=("pelvis_link",),
    elastic_band_attach_body_names=("waist_yaw_link", "pelvis_link"),
)


__all__ = ["ET1_BODY_NAMES", "ET1_CFG", "ET1_JOINT_NAMES", "ET1_MJCF_PATH", "ET1_UNITREE_MOTOR_INDICES"]
