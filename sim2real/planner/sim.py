"""The Planner in the vehicle sim2sim: simulated depth and the simulation clock.

- ``SimulatedTimeWorker``: the Planner worker on simulation time. A plan requested at
  simulation time t0 is ready at simulation time t0 + its measured wall-clock latency.
- ``DepthCamera`` / ``SimulatedDepthSource``: the head D435i rendered from the physics
  model at 30 Hz of simulation time, delivered like the Host's depth receiver.

``DepthCamera`` is ported from Scalecar_v2 ``simulation/depth.py`` (SOURCES.json),
with the pieces of ``scalecar_dp/spec.py`` it uses (frozen intrinsics, mount).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .scalecar_dp.dp.depth_preprocess import downsample_z16

DEPTH_HZ = 30


class SimulatedTimeWorker:
    """Runs a Planner worker on the simulation clock, injecting each plan's measured latency.

    The worker stamps a Chunk's ``t_ready`` with ``time.monotonic() + clock_offset_s``;
    ``submit`` sets the offset so that the request time maps to its simulation time t0.
    ``t_ready - t0`` is then the wall-clock time from request to result. The simulation
    loop never sleeps, so ``poll`` waits while a requested plan is still out and could
    be ready by now: until the result arrives, or until as much wall time has passed
    since the request as simulation time.
    """

    def __init__(self, worker, *, clock: Callable[[], float], wall: Callable[[], float] = time.monotonic,
                 poll_interval_s: float = 0.0005) -> None:
        self.worker = worker
        self.clock = clock
        self.wall = wall
        self.poll_interval_s = poll_interval_s
        self._pending: tuple | None = None  # (identity, t0, wall at submit, latest result before it)
        self.wait_s = 0.0

    def submit(self, item: dict) -> bool:
        before = self.worker.poll()
        submitted_at = self.wall()
        t0 = float(item["observation"].t0)
        published = self.worker.submit(dict(item, clock_offset_s=t0 - submitted_at))
        if published:
            self._pending = ((item["generation"], item["request_id"]), t0, submitted_at, before)
        return published

    def poll(self):
        item = self.worker.poll()
        if self._pending is None:
            return item
        identity, t0, submitted_at, before = self._pending
        budget = self.clock() - t0
        started = self.wall()
        while not self._answered(item, identity, before) and self.wall() - submitted_at < budget:
            time.sleep(self.poll_interval_s)
            item = self.worker.poll()
        self.wait_s += self.wall() - started
        if self._answered(item, identity, before):
            self._pending = None
        return item

    @staticmethod
    def _answered(item, identity, before) -> bool:
        if item is None or item is before:
            return False
        return item.get("status") == "error" or (item.get("generation"), item.get("request_id")) == identity

    def close(self) -> None:
        self.worker.close()


# ---------------------------------------------------------------------------
# Depth: Scalecar's real D435i stream contract (scalecar_dp/spec.py) and its MuJoCo camera
# ---------------------------------------------------------------------------

CAMERA_CONFIG = Path(__file__).with_name("camera_config.json")
CAMERA_CONFIG_SHA256 = "ab1636f686218c903e4e2aea87e2461120b0b03691c61cd058cd9d56ecef3974"


@dataclass(frozen=True)
class DepthSpec:
    """The real D435i depth stream (OpenCV pixel convention): Z16 mm, 0 = invalid, no distortion."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float = 1e-3
    max_distance_m: float = 10.0
    min_distance_m: float = 0.05


@dataclass(frozen=True)
class MountSpec:
    """G1 built-in D435 mount on torso_link (Unitree URDF d435_link); camera x forward, z up."""

    parent_link: str = "torso_link"
    pos: tuple = (0.0576235, 0.01753, 0.42987)
    rpy: tuple = (0.0, 0.8307767, 0.0)


def load_depth_spec(path: Path = CAMERA_CONFIG) -> DepthSpec:
    """The frozen camera snapshot (``spec.load_depth_spec``); refuses a modified file or distortion."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != CAMERA_CONFIG_SHA256:
        raise ValueError(f"{path} sha256 {digest} != pinned {CAMERA_CONFIG_SHA256}")
    config = json.loads(data)
    k = config["depth_intrinsics"]
    if any(float(c) != 0.0 for c in k["coefficients"]):
        raise ValueError(f"{path}: non-zero depth distortion is not supported")
    if abs(float(config["depth_scale_m_per_unit"]) - 1e-3) > 1e-9:
        raise ValueError(f"{path}: depth scale is not 1 mm")
    return DepthSpec(width=int(k["width"]), height=int(k["height"]), fx=float(k["fx"]), fy=float(k["fy"]),
                     cx=float(k["cx"]), cy=float(k["cy"]))


def metres_to_z16(depth_m: np.ndarray, spec: DepthSpec) -> np.ndarray:
    """Clean metric depth -> Z16. NaN/inf, <=0, below min or above max distance become 0."""
    d = np.asarray(depth_m, dtype=np.float64)
    ok = np.isfinite(d) & (d >= spec.min_distance_m) & (d <= spec.max_distance_m)
    out = np.zeros(d.shape, dtype=np.uint16)
    out[ok] = np.rint(d[ok] / spec.depth_scale_m).astype(np.uint16)
    return out


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX intrinsic (yaw-pitch-roll) to WXYZ."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


class DepthCamera:
    """MuJoCo optical-axis depth at the D435 mount and calibrated pixel grid (no noise).

    Scalecar's ``simulation/depth.py``: renders straight from the model being simulated
    (its visual geoms, group 3 hidden) at the real camera's intrinsics. It raises the
    model's offscreen buffer to 640x480 and sets the depth near/far planes, which only
    affect rendering.
    """

    def __init__(self, model):
        import mujoco

        from .scalecar_dp.motion38d import quat_to_matrix

        self.spec = load_depth_spec()
        self.mount = MountSpec()
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.mount.parent_link)
        if self.body < 0:
            raise ValueError(f"Missing camera parent {self.mount.parent_link}")
        self.model = model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.spec.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.spec.height)
        # Renderer depth linearization uses the same near/far stored in the model.
        model.vis.map.znear = 0.05 / model.stat.extent
        model.vis.map.zfar = 100.0 / model.stat.extent
        self.renderer = mujoco.Renderer(model, self.spec.height, self.spec.width)
        self.renderer.enable_depth_rendering()
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3] = 0
        R = quat_to_matrix(quat_from_rpy(*self.mount.rpy))
        # Mount convention is x forward, y left, z up; optical is right/down/forward.
        self.forward = R[:, 0]
        self.up = R[:, 2]
        self.times = []

    def render(self, data) -> np.ndarray:
        """(480, 640) uint16 Z16 depth in millimetres."""
        started = time.perf_counter()
        R = data.xmat[self.body].reshape(3, 3)
        position = data.xpos[self.body] + R @ np.asarray(self.mount.pos)
        self.renderer.update_scene(data, scene_option=self.option)
        d = self.spec
        near = self.model.vis.map.znear * self.model.stat.extent
        far = self.model.vis.map.zfar * self.model.stat.extent
        for camera in self.renderer.scene.camera:
            camera.pos[:] = position
            camera.forward[:] = R @ self.forward
            camera.up[:] = R @ self.up
            camera.orthographic = 0
            camera.frustum_near, camera.frustum_far = near, far
            camera.frustum_width = d.width / (2 * d.fx) * near
            camera.frustum_center = (d.width / 2 - 0.5 - d.cx) / d.fx * near
            camera.frustum_top = (d.cy + 0.5) / d.fy * near
            camera.frustum_bottom = -(d.height - d.cy - 0.5) / d.fy * near
        depth = metres_to_z16(self.renderer.render(), d)
        self.times.append(time.perf_counter() - started)
        return depth

    def close(self) -> None:
        self.renderer.close()


class SimulatedDepthSource:
    """The depth source of the vehicle sim2sim, with the Host receiver's ``poll`` interface.

    At the first poll at or after each 1/30 s of simulation time it renders the current
    physics state (Scalecar's ``DpProvider.observe``) and returns the downsampled frame,
    stamped with the simulation time; otherwise ``None``.
    """

    def __init__(self, model, data, *, clock: Callable[[], float]) -> None:
        self.camera = DepthCamera(model)
        self.data = data
        self.clock = clock
        self.next_frame_time = None
        self.frame = -1

    def poll(self) -> dict | None:
        now = self.clock()
        if self.next_frame_time is None:
            self.next_frame_time = now
        if now + 1e-8 < self.next_frame_time:
            return None
        depth = downsample_z16(self.camera.render(self.data))
        self.next_frame_time += 1.0 / DEPTH_HZ
        self.frame += 1
        return dict(session=0, frame=self.frame, acquired=now, depth_mm=depth)

    def close(self) -> None:
        self.camera.close()
