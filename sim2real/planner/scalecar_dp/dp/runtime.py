"""Online dp_v1 inference: input buffers, the 5 Hz planner with training-time soft RTC, and the
tracker-side chunk buffer (docs/DP_TRAINING_PLAN.md 4.5). Pure torch/numpy, no simulator.

Time is one monotonic clock in seconds shared by all sensors (caller's responsibility). Poses come from
the robot's odometry frame ("odom"): only relative poses enter the model, plus the pelvis height above
the ground (FK estimate). Data flow:

  sensors --push_depth(z16 480x640, t)--> DepthBuffer (downsampled frames + cached encoder features)
          --push_state(t, q29, pelvis pos/quat in odom, projected gravity)--> StateBuffer (50 Hz)
  Planner.plan(now) every REPLAN_STEPS*DT -> Chunk(t0, origin pose, (CHUNK,38) abs38 rows, ready time)
  ChunkBuffer.set(chunk); ChunkBuffer.tokens(t) at 50 Hz -> (6,69) reference-only tracker token

History source (deployment choice, ``history_state``): the StateBuffer is fed either the measured robot
state (``robot``) or the reference frame commanded at that instant (``self``: the DP runs autoregressively on
its own output, as in the offline data generation).

The observation assembled by ``Observer.observe(t0)`` is identical to the training sample of the same
instant (tests/test_dp_runtime.py replays dataset episodes through it).
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import motion38d as m38
from . import contract as C
from .depth_preprocess import downsample_z16, network_input


# ----------------------------------------------------------------------------------------------- inputs
class DepthBuffer:
    """Recent depth frames (downsampled uint16 mm) with timestamps; optional per-frame feature cache."""

    def __init__(self, maxlen: int = 64):
        self.t = deque(maxlen=maxlen)
        self.frames = deque(maxlen=maxlen)
        self.feats = deque(maxlen=maxlen)       # torch (20,d) or None until encoded

    def push(self, z16: np.ndarray, t_s: float, downsampled: bool = False):
        if self.t and t_s <= self.t[-1]:
            raise ValueError(f'depth timestamps must increase ({t_s} <= {self.t[-1]})')
        f = np.asarray(z16, np.uint16) if downsampled else downsample_z16(np.asarray(z16, np.uint16))
        if f.shape != (C.DEPTH_H, C.DEPTH_W):
            raise ValueError(f'depth frame shape {f.shape}')
        self.t.append(float(t_s)); self.frames.append(f); self.feats.append(None)

    def select(self, t0: float) -> list[int]:
        """Indices (oldest first) of N_DEPTH frames: newest with t <= t0, then every DEPTH_STRIDE-th earlier frame."""
        if not self.t:
            raise RuntimeError('no depth frame yet')
        ts = np.asarray(self.t)
        newest = max(int(np.searchsorted(ts, t0 + 1e-9, side='right')) - 1, 0)
        idx = [newest]
        for _ in range(C.N_DEPTH - 1):
            idx.append(max(idx[-1] - C.DEPTH_STRIDE, 0))
        return idx[::-1]


class StateBuffer:
    """50 Hz robot states: joints, pelvis pose in odom (z = height above ground), projected gravity."""

    def __init__(self, maxlen: int = 400):
        self.t = deque(maxlen=maxlen)
        self.rows = deque(maxlen=maxlen)

    def push(self, t_s: float, joint_pos, pelvis_pos, pelvis_quat_wxyz, projected_gravity):
        if self.t and t_s <= self.t[-1]:
            raise ValueError(f'state timestamps must increase ({t_s} <= {self.t[-1]})')
        row = np.concatenate([np.asarray(joint_pos, np.float64).reshape(29), np.asarray(pelvis_pos, np.float64).reshape(3),
                              np.asarray(pelvis_quat_wxyz, np.float64).reshape(4), np.asarray(projected_gravity, np.float64).reshape(3)])
        if not np.isfinite(row).all():
            raise ValueError('non-finite state')
        self.t.append(float(t_s)); self.rows.append(row)

    def at(self, t: float) -> np.ndarray | None:
        """Latest state at or before t (None if t is before the buffer)."""
        ts = np.asarray(self.t)
        i = int(np.searchsorted(ts, t + 1e-6, side='right')) - 1
        return None if i < 0 else self.rows[i]

    def latest_time(self) -> float:
        return self.t[-1]


def pose_of(row: np.ndarray) -> tuple[np.ndarray, float]:
    return row[29:32], float(m38.heading(row[32:36]))


@dataclass
class Observation:
    t0: float
    origin: tuple                      # (pelvis pos (3,), yaw) in odom
    depth_mm: np.ndarray               # (N_DEPTH,120,160) uint16
    depth_idx: list                    # indices into DepthBuffer
    depth_age_s: np.ndarray            # (N_DEPTH,)
    history: np.ndarray                # (HISTORY_N,41)
    history_mask: np.ndarray           # (HISTORY_N,) bool
    go_clock: float


class Observer:
    def __init__(self, depth: DepthBuffer, state: StateBuffer):
        self.depth, self.state = depth, state
        self.t_go = None

    def start(self, t_go: float):
        """Start signal (robot standing). Training time 0 corresponds to t_go."""
        self.t_go = float(t_go)

    def observe(self, t0: float) -> Observation:
        if self.t_go is None:
            raise RuntimeError('planner not started (no start signal)')
        now = self.state.at(t0)
        if now is None:
            raise RuntimeError(f'no state at t0={t0}')
        origin = pose_of(now)
        rows, mask = [], []
        for m in range(C.HISTORY_N - 1, -1, -1):
            t = t0 - m * C.HISTORY_STRIDE * C.DT
            r = self.state.at(t) if t >= self.t_go - 1e-6 else None
            if r is None:                                       # before the start signal: first state, masked
                r = self.state.at(self.t_go) if self.state.at(self.t_go) is not None else now
                mask.append(False)
            else:
                mask.append(True)
            rows.append(r)
        rows = np.stack(rows)
        hist = np.concatenate([C.abs38(rows[:, :29], rows[:, 29:32], rows[:, 32:36], origin[0], origin[1]), rows[:, 36:39]], -1)
        idx = self.depth.select(t0)
        return Observation(t0=t0, origin=origin, depth_mm=np.stack([self.depth.frames[i] for i in idx]), depth_idx=idx,
                           depth_age_s=np.array([t0 - self.depth.t[i] for i in idx], np.float32),
                           history=hist.astype(np.float32), history_mask=np.array(mask),
                           go_clock=min(max(t0 - self.t_go, 0.), C.GO_CLOCK_S) / C.GO_CLOCK_S)


# ----------------------------------------------------------------------------------------------- planner
@dataclass
class Chunk:
    t0: float
    origin: tuple                      # (pelvis pos (3,), yaw) in odom at t0
    rows: np.ndarray                   # (CHUNK,38) abs38 in the t0 frame
    t_ready: float
    prefix: int


def move_rows(rows: np.ndarray, old: tuple, new: tuple) -> np.ndarray:
    """abs38 rows expressed in the frame ``old`` -> frame ``new`` (both (pos, yaw) in odom)."""
    (po, yo), (pn, yn) = old, new
    c, s = math.cos(-yn), math.sin(-yn)
    d = (po - pn)[:2]
    dyaw = yo - yn
    out = rows.copy()
    cy, sy = math.cos(dyaw), math.sin(dyaw)
    x, y = rows[:, 29], rows[:, 30]
    out[:, 29] = cy * x - sy * y + (c * d[0] - s * d[1])
    out[:, 30] = sy * x + cy * y + (s * d[0] + c * d[1])
    for col in (32, 35):
        ax, ay = rows[:, col], rows[:, col + 1]
        out[:, col], out[:, col + 1] = cy * ax - sy * ay, sy * ax + cy * ay
    return out


def resample(rows: np.ndarray, start: float, n: int) -> np.ndarray:
    """Rows at fractional indices start, start+1, ... (linear, edge-clamped)."""
    i = np.clip(start + np.arange(n), 0, len(rows) - 1)
    lo = np.minimum(np.floor(i).astype(int), len(rows) - 2)
    w = (i - lo)[:, None]
    return rows[lo] * (1 - w) + rows[lo + 1] * w


class Planner:
    """Loads a training checkpoint (EMA weights + normalizer) and plans chunks with training-time soft RTC."""

    def __init__(self, ckpt: str, device: str = 'cuda', steps: int = 10, w0: float = 0.45, lam_s: float = 0.6,
                 latency_margin_s: float = 0.01, initial_latency_s: float = 0.08):
        from .model import DPModel
        from .normalizer import Normalizer
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        self.cfg = sd['config']
        self.dev = torch.device(device)
        self.model = DPModel(**self.cfg['model']).to(self.dev).eval()
        self.model.load_state_dict(sd['ema'])
        self.norm = Normalizer(**{k: v.numpy() for k, v in sd['normalizer'].items()}).to(self.dev)
        self.steps, self.w0, self.lam = steps, w0, lam_s / C.DT
        self.margin = latency_margin_s
        self.latencies = deque([initial_latency_s], maxlen=50)
        self.prev: Chunk | None = None

    def reset(self):
        self.prev = None

    def expected_latency(self) -> float:
        return float(np.percentile(self.latencies, 95)) + self.margin

    @torch.no_grad()
    def encode_missing(self, depth: DepthBuffer, idx: list[int]):
        todo = [i for i in idx if depth.feats[i] is None]
        if todo:
            x = network_input(torch.as_tensor(np.stack([depth.frames[i] for i in todo]), device=self.dev).float() * 1e-3)
            with torch.autocast(self.dev.type, dtype=torch.bfloat16, enabled=self.dev.type == 'cuda'):
                f = self.model.depth.encode_frames(x).float()
            for i, fi in zip(todo, f):
                depth.feats[i] = fi

    def prior(self, obs: Observation, d: int):
        """Prior rows (CHUNK,38) in the new t0 frame and tau_bar (CHUNK,) from the previous chunk."""
        tau_bar = np.ones(C.CHUNK, np.float32)
        rows = np.zeros((C.CHUNK, C.MOTION_DIM))
        if self.prev is None:
            return rows, tau_bar
        shift = (obs.t0 - self.prev.t0) / C.DT
        avail = int(np.floor(C.CHUNK - 1 - shift))           # previous-chunk frames that still cover new frames
        rows = move_rows(resample(self.prev.rows, shift, C.CHUNK), self.prev.origin, obs.origin)
        i = np.arange(C.CHUNK)
        tau_bar = np.where(i < d, 0., 1 - self.w0 * np.exp(-(i - d) / self.lam)).astype(np.float32)
        tau_bar[i > avail] = 1.
        return rows, tau_bar

    @torch.no_grad()
    def plan(self, obs: Observation, depth: DepthBuffer, now_fn=time.monotonic) -> Chunk:
        t_start = now_fn()
        lat = self.expected_latency() + max(t_start - obs.t0, 0.)    # t0 may already be in the past
        d = 0 if self.prev is None else int(min(math.ceil(lat / C.DT), C.MAX_PREFIX))
        prior, tau_bar = self.prior(obs, d)
        if self.prev is None:                                          # first chunk: standing pose prefix
            stand = np.repeat(obs.history[-1:, :38], C.CHUNK, 0)
            d = int(min(math.ceil(lat / C.DT), C.MAX_PREFIX))
            prior, tau_bar = stand, np.where(np.arange(C.CHUNK) < d, 0., 1.).astype(np.float32)
        self.encode_missing(depth, obs.depth_idx)
        dev = self.dev
        feats = torch.stack([depth.feats[i] for i in obs.depth_idx])[None]
        hist = torch.as_tensor(obs.history, device=dev)[None]
        hmask = torch.as_tensor(obs.history_mask, device=dev)[None]
        tb = torch.as_tensor(tau_bar, device=dev)[None]
        prefix, pmask = tb == 0, tb < 1
        pn = self.norm.norm_target(torch.as_tensor(prior, device=dev, dtype=torch.float32)[None])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            tokens, g = self.model.observe(None, torch.as_tensor(obs.depth_age_s, device=dev)[None],
                                           self.norm.norm_history(hist, hmask), hmask,
                                           torch.tensor([obs.go_clock], device=dev), depth_feats=feats)
            from .flow import sample
            out = sample(lambda x, t: self.model.denoise(tokens, g, x, t, prefix, pmask).float(), pn, tb, self.steps)
        rows = self.norm.denorm_target(out)[0].double().cpu().numpy()
        rows[:d] = prior[:d]                                            # exact committed prefix
        t_ready = now_fn()
        self.latencies.append(t_ready - obs.t0)
        chunk = Chunk(t0=obs.t0, origin=obs.origin, rows=rows, t_ready=t_ready, prefix=d)
        self.prev = chunk
        return chunk


# ----------------------------------------------------------------------------------------------- tracker side
@dataclass
class ChunkBuffer:
    """Thread-safe current chunk for the 50 Hz tracker loop.

    ``set`` switches at the chunk's ready time. If the new chunk arrived later than its hard prefix covers,
    the first ``blend_frames`` frames after the switch are blended from the previous chunk (expressed in
    the new frame) to the new one. ``tokens(t)`` returns the reference-only token at time t; past the end of
    the chunk it holds the last frame and counts ``stale``.
    """
    blend_frames: int = 3
    current: Chunk | None = None
    late_switches: int = 0
    stale: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, chunk: Chunk):
        with self._lock:
            old = self.current
            if old is not None:
                k = int(math.ceil((chunk.t_ready - chunk.t0) / C.DT))
                if k > chunk.prefix:                              # late: prefix does not cover the switch frame
                    self.late_switches += 1
                    prev = move_rows(resample(old.rows, (chunk.t0 - old.t0) / C.DT, C.CHUNK), old.origin, chunk.origin)
                    rows = chunk.rows.copy()
                    for j in range(self.blend_frames):
                        f = k + j
                        if f < C.CHUNK:
                            w = (j + 1) / (self.blend_frames + 1)
                            rows[f] = (1 - w) * prev[f] + w * rows[f]
                    chunk = Chunk(chunk.t0, chunk.origin, rows, chunk.t_ready, chunk.prefix)
            self.current = chunk

    def rows_at(self, t: float, n: int = max(C.TRACKER_OFFSETS) + 1) -> np.ndarray | None:
        with self._lock:
            c = self.current
        if c is None:
            return None
        start = (t - c.t0) / C.DT
        if start + n - 1 > C.CHUNK - 1:
            self.stale += 1
        return resample(c.rows, start, n)

    def tokens(self, t: float) -> np.ndarray | None:
        rows = self.rows_at(t)
        return None if rows is None else C.tracker_tokens(rows)

    def reference_state(self, t: float):
        """The reference frame commanded at t as a StateBuffer.push state in odom: (q29, pelvis pos, quat, gravity);
        None before the first chunk. Used for the ``self`` history source (see ``history_state``)."""
        with self._lock:
            c = self.current
        if c is None:
            return None
        row = resample(c.rows, (t - c.t0) / C.DT, 1)
        q, pos, quat, gravity = C.state_of_abs38(row, c.origin[0], c.origin[1])
        return q[0], pos[0], quat[0], gravity[0]


HISTORY_SOURCES = ('robot', 'self')


def history_state(source: str, buffer: ChunkBuffer, t: float, robot_state: tuple) -> tuple:
    """State pushed into the planner's StateBuffer at t (50 Hz), by deployment mode:

    * ``robot``: the measured robot state (closed loop through the robot);
    * ``self``: the reference frame the tracker is commanded at t (DP autoregressive on its own output; the robot
      state is used only before the first chunk, i.e. the standing pose at the start signal).
    ``robot_state`` = (q29, pelvis pos, quat WXYZ, projected gravity) in odom."""
    if source not in HISTORY_SOURCES:
        raise ValueError(f'history source {source!r} not in {HISTORY_SOURCES}')
    if source == 'self':
        ref = buffer.reference_state(t)
        if ref is not None:
            return ref
    return robot_state


class PlannerThread(threading.Thread):
    """Runs Planner.plan every REPLAN_STEPS*DT on the newest state and publishes into a ChunkBuffer."""

    def __init__(self, planner: Planner, observer: Observer, buffer: ChunkBuffer, now_fn=time.monotonic):
        super().__init__(daemon=True)
        self.planner, self.observer, self.buffer, self.now = planner, observer, buffer, now_fn
        self.stop_event = threading.Event()
        self.errors = 0

    def run(self):
        period = C.REPLAN_STEPS * C.DT
        while not self.stop_event.is_set():
            t_next = self.now() + period
            try:
                obs = self.observer.observe(self.observer.state.latest_time())
                self.buffer.set(self.planner.plan(obs, self.observer.depth, self.now))
            except Exception:
                self.errors += 1
                raise
            self.stop_event.wait(max(t_next - self.now(), 0.))
