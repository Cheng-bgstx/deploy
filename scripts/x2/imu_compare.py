#!/usr/bin/env python3
"""Read-only: compare the two X2 IMU topics with the waist joints to tell which IMU sits in the pelvis.

If topic A is the pelvis IMU and topic B the torso IMU, the rotation from A to B follows the waist joints
(yaw, pitch, roll); the pelvis IMU's tilt does not change when only the waist moves.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

TOPICS = ('/aima/hal/imu/torso/state', '/aima/hal/imu/chest/state')


def main(seconds=2.0):
    import rclpy
    from aimdk_msgs.msg import JointStateArray
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu
    rclpy.init()
    node = rclpy.create_node('x2_imu_compare')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    last = {}
    for t in TOPICS:
        node.create_subscription(Imu, t, lambda m, t=t: last.__setitem__(t, m), qos)
    node.create_subscription(JointStateArray, '/aima/hal/joint/waist/state', lambda m: last.__setitem__('waist', m), qos)
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    rots = {}
    for t in TOPICS:
        m = last.get(t)
        if m is None:
            print(f'{t}: no messages')
            continue
        o = m.orientation
        rots[t] = R.from_quat([o.x, o.y, o.z, o.w])
        g = rots[t].inv().apply([0, 0, -1])
        print(f'{t}: wxyz=({o.w:+.3f},{o.x:+.3f},{o.y:+.3f},{o.z:+.3f}) projected gravity {np.round(g, 3)} '
              f'tilt {np.degrees(np.arccos(-g[2])):.1f} deg  frame={m.header.frame_id}')
    w = last.get('waist')
    if w is not None:
        print('waist:', {j.name: round(j.position, 3) for j in w.joints})
    if len(rots) == 2:
        rel = rots[TOPICS[0]].inv() * rots[TOPICS[1]]
        print('rotation torso-topic -> chest-topic, intrinsic ZYX (yaw, pitch, roll) deg:',
              np.round(np.degrees(rel.as_euler('ZYX')), 1))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
