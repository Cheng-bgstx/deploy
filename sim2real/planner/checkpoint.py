"""The Planner checkpoint: the ModelScope d1146 file, pinned by sha256 and loaded with ``weights_only=True``."""

from __future__ import annotations

import hashlib
from collections import deque
from pathlib import Path

MODELSCOPE_ROOT = Path.home() / ".cache/modelscope/models/Hzzzz001--scalecar_v2/snapshots/master"
DEFAULT_CHECKPOINT = MODELSCOPE_ROOT / "dp/d1146_v1/ckpt_40000.pt"
CHECKPOINT_SHA256 = "d5e12b4a1fd27ad2eeb0d3a4cbd33aeb1715c8fe0b24472032fe49693429863c"


def file_sha256(path: Path, chunk_bytes: int = 1 << 22) -> str:
    """sha256 of a file (Python 3.10 has no ``hashlib.file_digest``)."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def load_planner(path: Path | str, *, sha256: str | None = CHECKPOINT_SHA256, device: str = "cuda:0"):
    """A ``scalecar_dp.dp.runtime.Planner`` from a checkpoint whose sha256 must equal ``sha256``
    (``None``: the caller has just checked it).

    Scalecar's ``Planner.__init__`` unpickles with ``weights_only=False``. This builds the
    same object (``runtime.py:180-193``: EMA weights, normaliser, default sampling and
    latency settings) from a ``weights_only=True`` load, after checking the hash.
    """
    import torch

    from .scalecar_dp.dp import runtime
    from .scalecar_dp.dp import contract as C
    from .scalecar_dp.dp.model import DPModel
    from .scalecar_dp.dp.normalizer import Normalizer

    path = Path(path)
    if sha256 is not None:
        digest = file_sha256(path)
        if digest != sha256:
            raise ValueError(f"Planner checkpoint {path} has sha256 {digest}, expected {sha256}")
    state = torch.load(path, map_location="cpu", weights_only=True)

    planner = runtime.Planner.__new__(runtime.Planner)
    # Mirrors runtime.Planner.__init__ with its default arguments.
    steps, w0, lam_s, latency_margin_s, initial_latency_s = 10, 0.45, 0.6, 0.01, 0.08
    planner.cfg = state["config"]
    planner.dev = torch.device(device)
    planner.model = DPModel(**planner.cfg["model"]).to(planner.dev).eval()
    planner.model.load_state_dict(state["ema"])
    planner.norm = Normalizer(**{k: v.numpy() for k, v in state["normalizer"].items()}).to(planner.dev)
    planner.steps, planner.w0, planner.lam = steps, w0, lam_s / C.DT
    planner.margin = latency_margin_s
    planner.latencies = deque([initial_latency_s], maxlen=50)
    planner.prev = None
    return planner
