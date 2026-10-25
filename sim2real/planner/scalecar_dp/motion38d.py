"""ECHO-compatible 38D motion encoding and conversion to the Scalecar reference-only tracker token.

38D layout (ECHO_CODE generator/utils/robot_process.py, 50 fps, joints in the contract/"Isaac" order):
  [0:29]  joint_pos (rad)
  [29:31] root_vel_xy: per-frame pelvis XY displacement (m/frame) in the aligned frame; row 0 is 0
  [31]    root_z: pelvis height above the ground plane (m)
  [32:38] root_rot_6d: columns 0 and 1 of the aligned pelvis rotation matrix, [col0, col1]
The aligned frame puts frame 0's pelvis XY at the origin and its heading (yaw of the
pelvis x axis) on +X; z stays the world/vehicle z (ground plane at z = 0 in Scalecar).

All inputs are numpy arrays; quaternions are WXYZ. No simulator imports.
"""
from __future__ import annotations

import numpy as np

FPS = 50
DIM = 38
TRACKER_OFFSETS = (0, 2, 4, 8, 16, 32)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit WXYZ with w >= 0 (Shepperd's method)."""
    R = np.asarray(R, dtype=np.float64)
    m = R.reshape(-1, 3, 3)
    out = np.empty((len(m), 4))
    for i, r in enumerate(m):
        t = np.trace(r)
        if t > 0:
            s = 2 * np.sqrt(t + 1)
            out[i] = [s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        else:
            k = int(np.argmax(np.diag(r)))
            a, b = (k + 1) % 3, (k + 2) % 3
            s = 2 * np.sqrt(1 + r[k, k] - r[a, a] - r[b, b])
            v = np.empty(3)
            v[k] = s / 4
            v[a] = (r[a, k] + r[k, a]) / s
            v[b] = (r[b, k] + r[k, b]) / s
            out[i] = [(r[b, a] - r[a, b]) / s, *v]
    out *= np.where(out[:, :1] < 0, -1, 1)
    return out.reshape(*R.shape[:-2], 4)


def rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def matrix_from_rot6d(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt, same as ECHO rot6d_to_rotation_matrix (columns b1, b2, b1 x b2)."""
    a1, a2 = r6[..., :3], r6[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def heading(q: np.ndarray) -> np.ndarray:
    """Yaw of the pelvis x axis projected on the ground plane (rad)."""
    R = quat_to_matrix(q)
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def encode(joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    """(T,29) joints, (T,3) pelvis position (z above ground), (T,4) pelvis WXYZ -> (T,38), aligned at frame 0."""
    joint_pos, root_pos, root_quat = (np.asarray(a, dtype=np.float64) for a in (joint_pos, root_pos, root_quat))
    T = len(joint_pos)
    if joint_pos.shape != (T, 29) or root_pos.shape != (T, 3) or root_quat.shape != (T, 4) or T == 0:
        raise ValueError(f'encode expects (T,29),(T,3),(T,4) with T>0; got {joint_pos.shape},{root_pos.shape},{root_quat.shape}')
    yaw0 = heading(root_quat[0])
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    rel = root_pos[:, :2] - root_pos[0, :2]
    xy = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], -1)
    vel = np.zeros((T, 2))
    vel[1:] = xy[1:] - xy[:-1]
    q = quat_mul(np.broadcast_to(yaw_quat(-yaw0), root_quat.shape), root_quat)
    return np.concatenate([joint_pos, vel, root_pos[:, 2:3], rot6d_from_matrix(quat_to_matrix(q))], -1)


def decode(feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T,38) -> joints (T,29), pelvis position (T,3) and WXYZ (T,4) in the aligned frame."""
    feat = np.asarray(feat, dtype=np.float64)
    if feat.ndim != 2 or feat.shape[1] != DIM:
        raise ValueError(f'decode expects (T,38); got {feat.shape}')
    xy = np.cumsum(np.concatenate([np.zeros((1, 2)), feat[1:, 29:31]]), 0)
    pos = np.concatenate([xy, feat[:, 31:32]], -1)
    return feat[:, :29].copy(), pos, matrix_to_quat(matrix_from_rot6d(feat[:, 32:38]))


def joint_velocity(joint_pos: np.ndarray, fps: int = FPS) -> np.ndarray:
    """Central difference (one-sided at the ends), rad/s."""
    return np.gradient(np.asarray(joint_pos, dtype=np.float64), 1.0 / fps, axis=0)


def reference_only_features(joint_pos, joint_vel, root_pos, root_quat, offsets) -> np.ndarray:
    """NumPy port of scalecar.utils.reference_only.reference_only_features for one sample.

    Inputs are F frames (F,29),(F,29),(F,3),(F,4) and offsets (F,); returns (F,69):
    q29, dq29, displacement3 in frame-0 heading, height1, rotation x/z axes6, offset1.
    """
    root_pos, root_quat = np.asarray(root_pos, np.float64), np.asarray(root_quat, np.float64)
    w, x, y, z = root_quat[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    d = root_pos - root_pos[:1]
    disp = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]], -1)
    a, b = np.cos(yaw / 2), -np.sin(yaw / 2)
    rw, rx, ry, rz = root_quat.T
    w, x, y, z = a * rw - b * rz, a * rx - b * ry, a * ry + b * rx, a * rz + b * rw
    rot = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y),
                    2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], -1)
    return np.concatenate([joint_pos, joint_vel, disp, root_pos[:, 2:3], rot,
                           np.asarray(offsets, np.float64)[:, None]], -1)


def tracker_tokens(chunk: np.ndarray, offsets=TRACKER_OFFSETS, fps: int = FPS) -> np.ndarray:
    """38D chunk whose row 0 is the current frame -> tracker reference token (len(offsets), 69).

    Joint velocity comes from central differences of the chunk; the chunk must cover max(offsets).
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    if len(chunk) <= max(offsets):
        raise ValueError(f'chunk has {len(chunk)} frames; offsets need {max(offsets) + 1}')
    q, pos, quat = decode(chunk)
    dq = joint_velocity(q, fps)
    idx = list(offsets)
    return reference_only_features(q[idx], dq[idx], pos[idx], quat[idx], idx)
