"""Read-only AgiBot X2 check: AimDK joint/IMU feedback rates and values, plus ONNX timing.

Publishes nothing, so it is safe while the native mc controller still owns the robot.
Run on the X2 board after sourcing ROS 2 Humble and the AimDK environment:
  python3 scripts/x2/state_probe.py --seconds 5 --policy checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy.onnx
"""
from __future__ import annotations

import argparse
import time

import numpy as np

GROUPS = ('leg', 'waist', 'arm', 'head')


def probe_ros(seconds, imu_topic):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from aimdk_msgs.msg import JointStateArray
    from sensor_msgs.msg import Imu

    rclpy.init()
    node = Node('sim2real_x2_state_probe')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    count, last = {g: 0 for g in (*GROUPS, 'imu')}, {}

    def on_group(group, msg):
        count[group] += 1
        last[group] = msg

    def on_imu(msg):
        count['imu'] += 1
        last['imu'] = msg

    subs = [node.create_subscription(JointStateArray, f'/aima/hal/joint/{g}/state',
                                     lambda m, g=g: on_group(g, m), qos) for g in GROUPS]
    subs.append(node.create_subscription(Imu, imu_topic, on_imu, qos))
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        rclpy.spin_once(node, timeout_sec=0.05)
    elapsed = time.monotonic() - start
    print(f'--- AimDK feedback over {elapsed:.1f} s (read-only) ---')
    for group in GROUPS:
        msg = last.get(group)
        if msg is None:
            print(f'{group:5s}: no messages')
            continue
        state = getattr(getattr(msg, 'state', None), 'value', None)
        print(f'{group:5s}: {count[group] / elapsed:6.1f} Hz, {len(msg.joints)} joints, group state={state}')
        for j in msg.joints:
            print(f'    {j.name or "?":28s} q={j.position:+.3f} dq={j.velocity:+.3f} '
                  f'tau={j.effort:+.2f} err={getattr(j, "error_code", 0)}')
    msg = last.get('imu')
    if msg is None:
        print(f'imu  : no messages on {imu_topic}')
    else:
        o, w = msg.orientation, msg.angular_velocity
        print(f'imu  : {count["imu"] / elapsed:6.1f} Hz, wxyz=({o.w:+.3f}, {o.x:+.3f}, {o.y:+.3f}, {o.z:+.3f}) '
              f'gyro=({w.x:+.3f}, {w.y:+.3f}, {w.z:+.3f}) frame={msg.header.frame_id}')
    for s in subs:
        node.destroy_subscription(s)
    node.destroy_node()
    rclpy.shutdown()


def probe_onnx(path, runs):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(path, opts, providers=['CPUExecutionProvider'])
    feeds = {i.name: np.zeros(i.shape, dtype=np.float32) for i in session.get_inputs()}
    for _ in range(10):
        session.run(None, feeds)
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        session.run(None, feeds)
        times.append((time.perf_counter() - t) * 1e3)
    times = np.array(times)
    print(f'--- ONNX ({path}), 1 thread, {runs} runs ---')
    print(f'mean {times.mean():.2f} ms, p99 {np.percentile(times, 99):.2f} ms, max {times.max():.2f} ms '
          f'(policy step budget 20 ms)')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seconds', type=float, default=5.)
    p.add_argument('--imu-topic', default='/aima/hal/imu/torso/state')
    p.add_argument('--policy', default=None, help='policy.onnx to time; omitted skips the timing')
    p.add_argument('--runs', type=int, default=300)
    p.add_argument('--skip-ros', action='store_true')
    args = p.parse_args()
    if not args.skip_ros:
        probe_ros(args.seconds, args.imu_topic)
    if args.policy:
        probe_onnx(args.policy, args.runs)


if __name__ == '__main__':
    main()
