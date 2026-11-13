"""Standard 24-motor PM01/pm_v2 joint_bridge interface.

Wire order: vendor config/pm01/joint_test.yaml, matched against the connected
robot's serial_pm_v2.urdf. J12 is WAIST_YAW; J23 is the sole head joint (yaw).

The transport is the same as T800: canonical training targets (position,
velocity, feed-forward torque, kp, kd) are forwarded unchanged, the mode and
feedback watchdogs stay in the base codec, and a zero mode sends the measured
pose with zero stiffness. The PM01-specific handling is the unset header stamp
in release 76445d09, the classic ankle parser and optional scales on the
commanded ankle-pitch gains, arm kp and ankle kd (PM01_ANKLE_PITCH_GAIN_SCALE, PM01_ARM_KP_SCALE, PM01_ANKLE_KD_SCALE).
"""

import os

import numpy as np

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
    ANKLE_PITCH_JOINTS = ('J04_ANKLE_PITCH_L', 'J10_ANKLE_PITCH_R')
    ANKLE_JOINTS = ('J04_ANKLE_PITCH_L', 'J05_ANKLE_ROLL_L', 'J10_ANKLE_PITCH_R', 'J11_ANKLE_ROLL_R')
    ARM_JOINTS = PM01_SDK_JOINT_NAMES[13:23]  # shoulders and elbows, J13-J22

    def __init__(self, *args, ankle_pitch_gain_scale=None, arm_kp_scale=None, ankle_kd_scale=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Both scales apply to policy commands only; the damping fallback is untouched. 1 sends the training gains.
        # Ankle pitch (kp and kd): the training gains act about 4.09x stiffer through the classic parser, which makes
        # the policy rock the pelvis at about 2 Hz and step during the wave in simulation. The onboard launcher sends
        # 1.25 / 4.09, which stays within 0.8-2x of the training joint-side stiffness if the 4.09 estimate is off.
        # Arm kp: with a command delay the soft arm servos (kp 18.9, kd 1.25) start a whole-body oscillation; in
        # simulation it appears at about 20 ms observation-to-target delay, and arm kp x0.7 moves it to about 30 ms.
        # Ankle kd (pitch and roll, after the pitch scale): with the feet lightly loaded (a taut gantry rope carrying
        # 70-80% of the weight on 2026-11-13) the policy rocked the pelvis at 4.3-4.7 Hz; in simulation x1.5 cut
        # that wave rocking from 8.8 to 0.4 deg/s and left full-load standing, the wave and the walk unchanged.
        # PM01_ANKLE_PITCH_GAIN_SCALE, PM01_ARM_KP_SCALE and PM01_ANKLE_KD_SCALE override.
        self.ankle_pitch_gain_scale = self._scale(ankle_pitch_gain_scale, 'PM01_ANKLE_PITCH_GAIN_SCALE', .1)
        self.arm_kp_scale = self._scale(arm_kp_scale, 'PM01_ARM_KP_SCALE', .5)
        self.ankle_kd_scale = self._scale(ankle_kd_scale, 'PM01_ANKLE_KD_SCALE', 1., 3.)
        self._ankle_pitch_index = [self.cfg.joint_names.index(n) for n in self.ANKLE_PITCH_JOINTS]
        self._ankle_index = [self.cfg.joint_names.index(n) for n in self.ANKLE_JOINTS]
        self._arm_index = [self.cfg.joint_names.index(n) for n in self.ARM_JOINTS]

    @staticmethod
    def _scale(value, env, low, high=1.):
        value = float(os.environ.get(env, '1') if value is None else value)
        if not low <= value <= high:
            raise ValueError(f'{env} must be in [{low}, {high}], got {value}')
        return value

    def write_command(self, q, dq, tau, kp, kd):
        if self.ankle_pitch_gain_scale != 1.0 or self.arm_kp_scale != 1.0 or self.ankle_kd_scale != 1.0:
            kp = np.array(kp, dtype=np.float32, copy=True)
            kd = np.array(kd, dtype=np.float32, copy=True)
            kp[self._ankle_pitch_index] *= self.ankle_pitch_gain_scale
            kd[self._ankle_pitch_index] *= self.ankle_pitch_gain_scale
            kd[self._ankle_index] *= self.ankle_kd_scale
            kp[self._arm_index] *= self.arm_kp_scale
        super().write_command(q, dq, tau, kp, kd)

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

    def __init__(self, cfg, *, spin_hz=None, spin_cpu=None, **options):
        # The firmware streams joint and IMU state faster than Python decodes them on the Nezha, so an unpaced
        # callback thread runs nonstop; on the policy's CPU it doubled inference to about 24 ms. PM01_HAL_HZ paces
        # it and PM01_HAL_CPU moves it off the policy CPU (the onboard launcher sets 200 and 0).
        env_hz, env_cpu = os.environ.get('PM01_HAL_HZ', ''), os.environ.get('PM01_HAL_CPU', '')
        spin_hz = float(env_hz) if spin_hz is None and env_hz else spin_hz
        spin_cpu = int(env_cpu) if spin_cpu is None and env_cpu else spin_cpu
        super().__init__(cfg, spin_hz=spin_hz, spin_cpu=spin_cpu, **options)
