"""CUDA Planner process with generation-isolated latest request and result slots.

Ported from Scalecar_v2 ``dp_worker.py`` (SOURCES.json). Differences:

- the checkpoint is checked against its pinned sha256 and loaded with
  ``weights_only=True`` (``checkpoint.load_planner``);
- cuDNN is disabled in this process: the root env's torch 2.13 cuDNN 9.20 wheel lacks a
  sub-library and picks up a mismatching system cuDNN on some Hosts; without cuDNN the
  outputs are bitwise those of Scalecar's torch 2.7 build;
- the Planner's clock is ``time.monotonic() + request["clock_offset_s"]`` (0 on the robot;
  the vehicle sim2sim maps it onto simulation time, ``sim.SimulatedTimeWorker``);
- no TraceWriter or summary file: results carry their timing, the 50 Hz side logs it.

Options: ``checkpoint``, ``checkpoint_sha256``, ``device`` (CUDA only), ``seed``.
"""

from __future__ import annotations

from collections import deque
import multiprocessing as mp
import os
import time
from typing import Any

import numpy as np

from .checkpoint import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, file_sha256
from .mailbox import LatestMailbox

DEFAULT_OPTIONS = dict(
    checkpoint=str(DEFAULT_CHECKPOINT),
    checkpoint_sha256=CHECKPOINT_SHA256,
    device="cuda:0",
    seed=4242,
)
RESTART_DELAY_S = 5.0


def worker_options(options: dict | None = None) -> dict:
    unknown = set(options or {}) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError(f"Unknown Planner options {sorted(unknown)}; allowed {sorted(DEFAULT_OPTIONS)}")
    return {**DEFAULT_OPTIONS, **(options or {})}


def _plan(options: dict, request: LatestMailbox, result: LatestMailbox, stop: Any, cpu: int | None) -> None:
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    try:
        import torch

        from .checkpoint import load_planner
        from .scalecar_dp.dp import runtime

        torch.backends.cudnn.enabled = False
        digest = file_sha256(options["checkpoint"])
        if digest != options["checkpoint_sha256"]:
            raise ValueError(
                f"Planner checkpoint {options['checkpoint']} has sha256 {digest}, "
                f"expected {options['checkpoint_sha256']}"
            )
        device = torch.device(options["device"])
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("The Planner requires CUDA; CPU fallback is disabled")
        torch.set_num_threads(1)
        torch.manual_seed(options["seed"])
        planner = load_planner(options["checkpoint"], sha256=None, device=options["device"])
        if next(planner.model.parameters()).device.type != "cuda":
            raise RuntimeError("Planner parameters are not on CUDA")
        result.publish(
            dict(
                status="loaded",
                device=str(device),
                gpu=torch.cuda.get_device_name(device),
                cudnn_enabled=bool(torch.backends.cudnn.enabled),
                torch=torch.__version__,
            )
        )
        generation = None
        features = {}
        version = 0
        while not stop.is_set():
            new_version, item = request.read(version)
            if item is None:
                stop.wait(0.001)
                continue
            skipped = max(0, new_version - version - 1)
            version = new_version
            try:
                if item["generation"] != generation:
                    planner.reset()
                    features.clear()
                    generation = item["generation"]
                depth = runtime.DepthBuffer()
                depth.t = deque(item["depth_times"], maxlen=64)
                depth.frames = deque(item["observation"].depth_mm, maxlen=64)
                depth.feats = deque([features.get(t) for t in depth.t], maxlen=64)
                obs = item["observation"]
                obs.depth_idx = list(range(len(depth.t)))
                offset = float(item.get("clock_offset_s", 0.0))
                begin = time.monotonic()
                chunk = planner.plan(obs, depth, lambda: time.monotonic() + offset)
                if chunk.rows.shape != (128, 38) or not np.isfinite(chunk.rows).all():
                    raise ValueError("DP produced invalid 128×38 chunk")
                features = {t: f for t, f in zip(depth.t, depth.feats) if f is not None}
                record = dict(
                    generation=generation,
                    request_id=item["request_id"],
                    t0=chunk.t0,
                    t_ready=chunk.t_ready,
                    prefix=chunk.prefix,
                    inference_ms=(time.monotonic() - begin) * 1000,
                    age_ms=(chunk.t_ready - chunk.t0) * 1000,
                    replaced_requests=skipped,
                    warmup=item["warmup"],
                )
                result.publish(record | dict(status="ready", chunk=chunk, device=str(device)))
            except Exception as error:
                # A failed request never poisons subsequent requests/prior history.
                planner.reset()
                features.clear()
                result.publish(dict(status="error", error=f"{type(error).__name__}: {error}"))
                stop.wait(0.05)
    except Exception as error:
        result.publish(dict(status="error", error=f"{type(error).__name__}: {error}"))


class GpuPlannerWorker:
    """Single CUDA owner; submit/poll never wait for inference to finish. Restarts 5 s after a crash."""

    def __init__(self, options: dict | None = None, cpu: int | None = None):
        context = mp.get_context("spawn")
        self.request = LatestMailbox(context)
        self.result = LatestMailbox(context)
        self.stop = context.Event()
        self.context = context
        self.options, self.cpu = worker_options(options), cpu
        self.restarts = 0
        self.next_restart = 0.0
        self.version = 0
        self.last = None
        self.process_error = None
        self.dropped = 0
        self.process = self._spawn()

    def _spawn(self):
        process = self.context.Process(
            target=_plan, args=(self.options, self.request, self.result, self.stop, self.cpu), daemon=False
        )
        process.start()
        return process

    def submit(self, item: dict) -> bool:
        published = self.request.publish(item)
        self.dropped += not published
        return published

    def poll(self) -> dict | None:
        self.version, item = self.result.read(self.version)
        if item is not None:
            self.last = item
            if item.get("status") == "error":
                self.process_error = item["error"]
        if not self.process.is_alive() and not self.stop.is_set():
            # Keep the process's own last error visible (e.g. a checkpoint sha256 mismatch).
            cause = f" after: {self.process_error}" if self.process_error else ""
            self.last = dict(status="error", error=f"GPU planner exited{cause}; restarting on CUDA")
            now = time.monotonic()
            if now >= self.next_restart:
                self.process.join(timeout=0)
                self.next_restart = now + RESTART_DELAY_S
                self.restarts += 1
                self.process_error = None
                self.process = self._spawn()
        return self.last

    def close(self) -> None:
        self.stop.set()
        self.process.join(timeout=5)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)
