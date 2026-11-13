"""Read-only PM01 sway monitor: subscribes to the IMU and joint feedback, publishes nothing.

Every 0.5 s it prints the pelvis pitch / roll (deg), the oscillating part of the pelvis pitch and roll rates over the
last second (deg/s rms after removing the mean), the fastest joint and the ankle-pitch speeds, and the firmware mode.
Run beside a Tracking session to judge a hold:
  python3 -m scripts.pm01.watch_sway [--seconds 120]
"""
import argparse
import collections
import math
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sim2real.rl_policy.robot_io.pm01 import PM01_SDK_JOINT_NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=120.0)
    a = p.parse_args()
    from interface_protocol.msg import ImuInfo, JointState, MotionState
    rclpy.init()
    node = rclpy.create_node('pm01_watch_sway')
    # Depth 1 and a 100 Hz loop: the firmware streams faster than Python decodes, so take only the newest sample.
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    gyro = collections.deque(maxlen=400)
    state = {'quat': None, 'qd': None, 'mode': '?'}

    def on_imu(m):
        q = m.quaternion
        state['quat'] = (q.w, q.x, q.y, q.z)
        gyro.append((time.monotonic(), m.angular_velocity.x, m.angular_velocity.y))

    def on_joint(m):
        state['qd'] = np.asarray(m.velocity, dtype=float)

    def on_mode(m):
        state['mode'] = m.current_motion_task_name

    node.create_subscription(ImuInfo, '/hardware/imu_info', on_imu, qos)
    node.create_subscription(JointState, '/hardware/joint_state', on_joint, qos)
    node.create_subscription(MotionState, '/motion/motion_state', on_mode, qos)
    ankle = [PM01_SDK_JOINT_NAMES.index(n) for n in ('J04_ANKLE_PITCH_L', 'J10_ANKLE_PITCH_R')]
    end, next_print = time.monotonic() + a.seconds, time.monotonic() + 0.5
    print('time   mode          pitch   roll | pitch-rate roll-rate osc (deg/s rms, 1 s) | max joint speed | ankle pitch speeds',
          flush=True)
    try:
        while time.monotonic() < end:
            for _ in range(3):
                rclpy.spin_once(node, timeout_sec=0)
            time.sleep(0.01)
            now = time.monotonic()
            if now < next_print or state['quat'] is None or state['qd'] is None:
                continue
            next_print = now + 0.5
            w, x, y, z = state['quat']
            pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
            roll = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            recent = np.array([(gx, gy) for t, gx, gy in gyro if now - t <= 1.0])
            osc = np.degrees(recent.std(0)) if len(recent) > 10 else (float('nan'), float('nan'))
            qd = np.abs(state['qd'])
            i = int(qd.argmax())
            print(f'{time.strftime("%H:%M:%S")} {state["mode"]:<12s} {pitch:+6.1f} {roll:+6.1f} | {osc[1]:6.2f} {osc[0]:6.2f}'
                  f' | {qd[i]:5.2f} rad/s {PM01_SDK_JOINT_NAMES[i]:<20s} | {qd[ankle[0]]:5.2f} {qd[ankle[1]]:5.2f}',
                  flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
