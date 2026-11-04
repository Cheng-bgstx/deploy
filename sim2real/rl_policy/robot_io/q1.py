"""Primebot Q1 HAL: PrimebotTech/primebot-q1-sdk's named full-body ROS interface."""
import os

from sim2real.rl_policy.robot_io.ros2_named import NamedRosRobotIO


class Q1RobotIO(NamedRosRobotIO):
    def __init__(self, cfg, *, imu_topic=None, publish_hz=1000., **options):
        if cfg.name != 'q1':
            raise ValueError('Q1 backend requires robot=q1')
        # The public SDK does not document a root IMU topic. Require deployment
        # configuration rather than assuming that X2's IMU address is reusable.
        imu_topic = imu_topic or os.environ.get('SIM2REAL_Q1_IMU_TOPIC')
        if not imu_topic:
            raise ValueError('Q1 requires the verified root sensor_msgs/Imu topic in robot_io_options.imu_topic')
        from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
        super().__init__(cfg, groups={'body': cfg.joint_names},
            topics={'body': ('/aima/hal/joint/state', '/aima/hal/joint/command')},
            types=(JointCommandArray, JointCommand, JointStateArray), imu_topic=imu_topic,
            publish_hz=publish_hz, damping=5., max_position_step=.2,
            start_disarmed=True, allow_init_from_measured=True, preserve_feedback_order=True, **options)
