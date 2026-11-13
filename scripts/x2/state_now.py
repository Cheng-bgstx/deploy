#!/usr/bin/env python3
"""Read-only snapshot (or --watch, once per second) for the operator: pelvis tilt, leg joint error from standing, and leg torques (foot load).

Runs on the non-isolated cores so it does not disturb a live bridge (core 5) or Tracking (core 7).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from sim2real.config.robots import X2_CFG


def watch_line(last, default):
    o = last['imu'].orientation
    g = R.from_quat([o.x, o.y, o.z, o.w]).inv().apply([0, 0, -1])
    pitch, roll = np.degrees(np.arcsin(np.clip(g[0], -1, 1))), np.degrees(np.arcsin(np.clip(-g[1], -1, 1)))
    leg = {j.name: j for j in last['leg'].joints}
    ankle = [leg[f'{s}_ankle_pitch_joint'].position - default[f'{s}_ankle_pitch_joint'] for s in ('left', 'right')]
    knee = [leg[f'{s}_knee_joint'].effort for s in ('left', 'right')]
    ok = abs(pitch - 4.6) < 4 and abs(roll) < 4 and max(abs(a) for a in ankle) < 0.12
    return (f'pitch {pitch:+5.1f} deg (stand +4.6)  roll {roll:+5.1f} deg  ankle pitch err L {ankle[0]:+.2f} R {ankle[1]:+.2f} rad'
            f'  knee tau L {knee[0]:+5.1f} R {knee[1]:+5.1f} Nm  -> {"UPRIGHT, ok for the second a" if ok else "not upright"}')


def watch_bridge():
    """While a bridge runs: read its LowState (ZMQ, latest message only) once per second; no ROS subscription."""
    import zmq
    from sim2real.utils.common import LowStateMessage
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b'')
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.RCVTIMEO, 2000)
    sock.connect(f'tcp://127.0.0.1:{X2_CFG.low_state_port}')
    default = dict(zip(X2_CFG.joint_names, X2_CFG.default_qpos[7:]))
    shim = type('J', (), {})
    try:
        while True:
            time.sleep(1.0)
            try:
                msg = LowStateMessage.from_bytes(sock.recv())
            except zmq.Again:
                print(time.strftime('%H:%M:%S'), 'no LowState from the bridge', flush=True)
                continue
            imu = shim()
            w, x, y, z = msg.quaternion
            imu.orientation = type('Q', (), dict(w=w, x=x, y=y, z=z))()
            joints = []
            torques = msg.joint_torques if msg.joint_torques is not None else np.zeros(len(X2_CFG.joint_names))
            for name, q, tau in zip(X2_CFG.joint_names, msg.joint_positions, torques):
                j = shim()
                j.name, j.position, j.effort = name, float(q), float(tau)
                joints.append(j)
            leg = shim()
            leg.joints = joints
            line = watch_line({'imu': imu, 'leg': leg}, default)
            if msg.stop_epoch:  # the bridge latched a stop (space, timeout or fault): damping, not standing
                line = line.split('  -> ')[0] + f'  -> BRIDGE STOPPED (stop {msg.stop_epoch}), damping'
            print(time.strftime('%H:%M:%S'), line, flush=True)
    except KeyboardInterrupt:
        return


def main(seconds=1.0, watch=False):
    import rclpy
    from aimdk_msgs.msg import JointStateArray
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu
    rclpy.init()
    node = rclpy.create_node('x2_state_now')
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)
    last = {}
    node.create_subscription(Imu, '/aima/hal/imu/torso/state', lambda m: last.__setitem__('imu', m), qos)
    for g in ('leg', 'waist', 'arm'):
        node.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state', lambda m, g=g: last.__setitem__(g, m), qos)
    default = dict(zip(X2_CFG.joint_names, X2_CFG.default_qpos[7:]))
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    if watch:  # one line per second until Ctrl-C
        try:
            while True:
                print(time.strftime('%H:%M:%S'), watch_line(last, default), flush=True)
                end = time.monotonic() + 1.0
                while time.monotonic() < end:
                    rclpy.spin_once(node, timeout_sec=0.05)
        except KeyboardInterrupt:
            return
    o = last['imu'].orientation
    g = R.from_quat([o.x, o.y, o.z, o.w]).inv().apply([0, 0, -1])
    print(f'pelvis projected gravity {np.round(g, 3)}: pitch {np.degrees(np.arcsin(np.clip(g[0], -1, 1))):+.1f} deg '
          f'(+ forward), roll {np.degrees(np.arcsin(np.clip(-g[1], -1, 1))):+.1f} deg, tilt {np.degrees(np.arccos(-g[2])):.1f} deg')
    for grp in ('leg', 'waist', 'arm'):
        for j in last[grp].joints:
            if j.name in default and not j.name.startswith(('left_wrist_p', 'left_wrist_r', 'right_wrist_p', 'right_wrist_r')):
                print(f'  {j.name:28s} q={j.position:+.3f} (stand {default[j.name]:+.3f}, err {j.position - default[j.name]:+.3f})'
                      f' dq={j.velocity:+.2f} tau={j.effort:+6.1f}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys
    if '--watch-bridge' in sys.argv:
        watch_bridge()
    else:
        main(watch='--watch' in sys.argv)
