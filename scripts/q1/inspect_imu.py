"""Print Q1 IMU axes and gravity consistency using subscriptions only.

This does not estimate a mounting rotation, write configuration or send commands.
The reported pelvis axes assume the supplied sensor-to-pelvis quaternion is correct.
"""
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from scipy.spatial.transform import Rotation


@dataclass
class Args:
    topic: str = '/aima/hal/imu/state'
    duration_s: float = 15.
    imu_to_pelvis_wxyz: tuple[float, float, float, float] = (1., 0., 0., 0.)
    output: Path | None = None


def main(args):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Imu

    mount = np.asarray(args.imu_to_pelvis_wxyz, dtype=float)
    if not np.isfinite(mount).all() or np.linalg.norm(mount) < 1e-6:
        raise ValueError('Supply a finite nonzero mounting quaternion')
    if not np.isfinite(args.duration_s) or args.duration_s <= 0:
        raise ValueError('duration_s must be positive')
    sensor_to_pelvis = Rotation.from_quat(mount[[1, 2, 3, 0]])
    rclpy.init()
    node = rclpy.create_node('q1_readonly_imu_axes', enable_rosout=False)
    rows, latest = [], None
    latest_arrival = 0.
    count = 0

    def receive(message):
        nonlocal latest, count, latest_arrival
        latest = message
        latest_arrival = time.monotonic()
        count += 1

    node.create_subscription(Imu, args.topic, receive,
        QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    end, next_print = time.monotonic() + args.duration_s, 0.
    print('READ ONLY. Pelvis output assumes the specified mounting; this is not a calibration.')
    try:
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)
            if latest is None or time.monotonic() < next_print:
                continue
            if time.monotonic() - latest_arrival > .25:
                raise RuntimeError('IMU feedback became stale')
            m = latest
            q = np.array([m.orientation.x, m.orientation.y, m.orientation.z, m.orientation.w])
            if m.orientation_covariance[0] < 0 or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
                raise ValueError('IMU orientation is unavailable or invalid')
            rotation = Rotation.from_quat(q) * sensor_to_pelvis.inv()
            gyro = sensor_to_pelvis.apply([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            acceleration = sensor_to_pelvis.apply([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
            up = rotation.inv().apply([0, 0, 1])
            magnitude = np.linalg.norm(acceleration)
            mismatch = float(np.degrees(np.arccos(np.clip(np.dot(acceleration, up) / magnitude, -1, 1)))) if magnitude > 1e-6 else None
            row = {
                'stamp_s': m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                'frame_id': m.header.frame_id,
                'assumed_pelvis_rpy_deg': rotation.as_euler('xyz', degrees=True).tolist(),
                'assumed_pelvis_gyro_xyz_rad_s': gyro.tolist(),
                'assumed_pelvis_specific_force_xyz_m_s2': acceleration.tolist(),
                'stationary_gravity_mismatch_deg': mismatch,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            next_print = time.monotonic() + .5
        if not rows:
            raise RuntimeError(f'No IMU samples received on {args.topic}')
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({'mounting_verified': False, 'received': count,
                'imu_to_pelvis_wxyz': list(args.imu_to_pelvis_wxyz), 'samples': rows}, indent=2)+'\n')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(tyro.cli(Args))
