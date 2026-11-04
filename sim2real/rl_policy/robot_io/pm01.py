"""Standard 24-motor PM01/pm_v2 joint_bridge interface.

Wire order: vendor config/pm01/joint_test.yaml, matched against the connected
robot's serial_pm_v2.urdf. J12 is WAIST_YAW; J23 is the sole head joint (yaw).
"""
from sim2real.rl_policy.robot_io.engineai import EngineAICodec, EngineAIRobotIO

PM01_SDK_JOINT_NAMES = (
    'J00_HIP_PITCH_L', 'J01_HIP_ROLL_L', 'J02_HIP_YAW_L', 'J03_KNEE_PITCH_L',
    'J04_ANKLE_PITCH_L', 'J05_ANKLE_ROLL_L', 'J06_HIP_PITCH_R', 'J07_HIP_ROLL_R',
    'J08_HIP_YAW_R', 'J09_KNEE_PITCH_R', 'J10_ANKLE_PITCH_R', 'J11_ANKLE_ROLL_R',
    'J12_WAIST_YAW', 'J13_SHOULDER_PITCH_L', 'J14_SHOULDER_ROLL_L', 'J15_SHOULDER_YAW_L',
    'J16_ELBOW_PITCH_L', 'J17_ELBOW_YAW_L', 'J18_SHOULDER_PITCH_R', 'J19_SHOULDER_ROLL_R',
    'J20_SHOULDER_YAW_R', 'J21_ELBOW_PITCH_R', 'J22_ELBOW_YAW_R', 'J23_HEAD_YAW',
)


class PM01Codec(EngineAICodec):
    pass


class PM01RobotIO(EngineAIRobotIO):
    robot_name = 'pm01'
    sdk_joint_names = PM01_SDK_JOINT_NAMES
    codec_type = PM01Codec
