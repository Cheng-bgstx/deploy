"""dp_v1 input/output contract of the depth -> reference DP (docs/DP_TRAINING_PLAN.md).

Frames: everything is expressed in the *t0 frame*: origin = pelvis XY at the observation time t0,
x axis = pelvis heading at t0, z = height above the ground (not shifted). Quaternions WXYZ, metres,
radians, 50 Hz control rate, joints in spec.JOINT_NAMES order.

Internal motion row ("abs38", what the network predicts): q29 | xy in t0 frame (2) | z (1) |
rot6d = columns 0,1 of R_z(-yaw0) @ R_pelvis (6). ``to_echo38`` turns a chunk whose row 0 is t0 into
ECHO 38D (per-frame xy displacement, row 0 = 0), identical to ``motion38d.encode`` on the same rows.
History row (41) = abs38 of a past state + projected gravity (3).
Depth input: N_DEPTH frames of (DEPTH_H, DEPTH_W) uint16 mm (0 = invalid), see depth_preprocess.
"""
from __future__ import annotations

import numpy as np

from .. import motion38d as m38

SCHEMA = 'dp_v1'
FPS = 50
DT = 1.0 / FPS
CHUNK = 128                 # output frames (2.56 s)
REPLAN_STEPS = 10           # 5 Hz replanning
MAX_PREFIX = 10             # hard prefix frames d in [0, MAX_PREFIX]
HISTORY_N = 9
HISTORY_STRIDE = 6          # 0.12 s
HISTORY_DIM = 41
MOTION_DIM = 38
N_DEPTH = 4
DEPTH_STRIDE = 7            # 30 Hz frames between history depth frames (~0.24 s)
DEPTH_H, DEPTH_W = 120, 160
DEPTH_BLOCK = 4             # 480x640 -> 120x160
DEPTH_MIN_M, DEPTH_MAX_M = 0.15, 4.0
GO_CLOCK_S = 3.0
TRACKER_OFFSETS = m38.TRACKER_OFFSETS

# slices of an abs38 / history row
Q, XY, Z, ROT = slice(0, 29), slice(29, 31), slice(31, 32), slice(32, 38)
GRAV = slice(38, 41)
# joint groups for per-part random walks (indices into spec.JOINT_NAMES)
JOINT_GROUPS = {
    'left_leg': (0, 3, 6, 9, 13, 17), 'right_leg': (1, 4, 7, 10, 14, 18), 'waist': (2, 5, 8),
    'left_arm': (11, 15, 19, 21, 23, 25, 27), 'right_arm': (12, 16, 20, 22, 24, 26, 28)}


def yaw_of(quat: np.ndarray) -> np.ndarray:
    return m38.heading(quat)


def abs38(joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, origin_pos: np.ndarray,
          origin_yaw: float) -> np.ndarray:
    """(T,29),(T,3),(T,4) states -> (T,38) in the frame of (origin_pos xy, origin_yaw)."""
    c, s = np.cos(-origin_yaw), np.sin(-origin_yaw)
    rel = np.asarray(root_pos, np.float64)[:, :2] - np.asarray(origin_pos, np.float64)[:2]
    xy = np.stack([c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1]], -1)
    q = m38.quat_mul(np.broadcast_to(m38.yaw_quat(-origin_yaw), np.shape(root_quat)), root_quat)
    return np.concatenate([joint_pos, xy, np.asarray(root_pos)[:, 2:3], m38.rot6d_from_matrix(m38.quat_to_matrix(q))], -1)


def state_of_abs38(rows: np.ndarray, origin_pos: np.ndarray, origin_yaw: float):
    """Inverse of ``abs38``: (T,38) rows in the frame (origin_pos xy, origin_yaw) -> states in the outer frame:
    joint_pos (T,29), root_pos (T,3) (z unchanged), root_quat (T,4) WXYZ, projected_gravity (T,3)."""
    rows = np.asarray(rows, np.float64)
    c, s = np.cos(origin_yaw), np.sin(origin_yaw)
    x, y = rows[:, 29], rows[:, 30]
    pos = np.stack([origin_pos[0] + c * x - s * y, origin_pos[1] + s * x + c * y, rows[:, 31]], -1)
    R = m38.quat_to_matrix(m38.yaw_quat(origin_yaw)) @ m38.matrix_from_rot6d(rows[:, 32:38])
    gravity = np.einsum('tji,j->ti', R, np.array([0., 0., -1.]))
    return rows[:, :29], pos, m38.matrix_to_quat(R), gravity


def to_echo38(chunk_abs: np.ndarray) -> np.ndarray:
    """abs38 chunk with row 0 at t0 -> ECHO 38D (per-frame xy displacement; row 0 displacement 0)."""
    out = np.array(chunk_abs, dtype=np.float64, copy=True)
    xy = out[:, XY].copy()
    out[:, XY] = 0.
    out[1:, XY] = xy[1:] - xy[:-1]
    return out


def tracker_tokens(chunk_abs: np.ndarray, start: int = 0) -> np.ndarray:
    """Reference-only tracker tokens (6,69) for the frame ``start`` of an abs38 chunk."""
    return m38.tracker_tokens(to_echo38(chunk_abs[start:]))
