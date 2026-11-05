"""Standard 24-motor PM01/pm_v2 joint_bridge interface.

Wire order: vendor config/pm01/joint_test.yaml, matched against the connected
robot's serial_pm_v2.urdf. J12 is WAIST_YAW; J23 is the sole head joint (yaw).

The transport is the same as T800: canonical training targets (position,
velocity, feed-forward torque, kp, kd) are forwarded unchanged, the mode and
feedback watchdogs stay in the base codec, and a zero mode sends the measured
pose with zero stiffness. The PM01-specific handling is the unset header stamp
in release 76445d09 and the classic ankle parser.
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
    # RL_PARSER closes ankle PD outside the motor driver, across a measured
    # 8-9 ms loop. On a light free foot, kd=2 and the training kd=2.4916 are
    # unstable there (whole-robot resets on 2026-11-04 and 2026-11-05). Classic
    # keeps PD in the driver; the gains then act motor-side, so joint-side
    # ankle pitch is about 4.09x and roll about 1.11x the commanded values.
    parallel_parser_type = 0

    def _new_stamp(self, key, msg):
        # PM01 release 76445d09 publishes unset (zero) JointState/ImuInfo
        # headers. An all-zero stamp means "no stamp": accept it and let the
        # base codec's monotonic receive time drive the freshness watchdog.
        # Once a real stamp has been seen, zero resets are rejected again.
        stamp = getattr(getattr(msg, 'header', None), 'stamp', None)
        if (stamp is not None and stamp.sec == 0 and stamp.nanosec == 0
                and self.stamps.get(key, 0) == 0):
            self.stamps[key] = 0
            return True
        return super()._new_stamp(key, msg)


class PM01RobotIO(EngineAIRobotIO):
    robot_name = 'pm01'
    sdk_joint_names = PM01_SDK_JOINT_NAMES
    codec_type = PM01Codec
