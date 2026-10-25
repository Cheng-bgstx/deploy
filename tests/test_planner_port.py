"""The ported Planner core (dixiatielu/sim2real_scalecar#7): provenance, ResNet-18, checkpoint, Go height.

Seams under test:
- the vendored files against the sha256 recorded in ``sim2real/planner/SOURCES.json``
  (and against Scalecar_v2 itself when that checkout is present);
- the torchvision-free ResNet-18 against torchvision (bitwise; needs the overlay below);
- loading the pinned checkpoint (``weights_only=True``, sha256 lock) against Scalecar's own loader;
- the Go-time pelvis height estimate against Scalecar's ``standing_height``.

    env -u PYTHONPATH HF_HUB_OFFLINE=1 UV_FROZEN=1 uv run --with pytest python -m pytest tests/test_planner_port.py
    # ResNet-18 vs torchvision (bitwise):
    env -u PYTHONPATH HF_HUB_OFFLINE=1 UV_FROZEN=1 uv run --with pytest --with torchvision==0.28.0 \
        python -m pytest tests/test_planner_port.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER = REPO_ROOT / "sim2real" / "planner"
SCALECAR = Path("/home/galbot/projects/Scalecar_v2")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources() -> dict:
    return json.loads((PLANNER / "SOURCES.json").read_text(encoding="utf-8"))


def test_vendored_files_match_their_recorded_sha256():
    files = sources()["files"]
    assert files, "SOURCES.json lists no files"
    for vendored, entry in files.items():
        assert sha256(PLANNER / vendored) == entry["sha256"], vendored


def test_verbatim_files_are_byte_identical_to_their_source():
    for vendored, entry in sources()["files"].items():
        if entry["kind"] == "verbatim":
            assert entry["sha256"] == entry["source_sha256"], vendored


def test_model_differs_from_its_source_only_by_the_torchvision_import():
    entry = sources()["files"]["scalecar_dp/dp/model.py"]
    vendored = (PLANNER / "scalecar_dp/dp/model.py").read_text(encoding="utf-8")
    replaced = [line for line in vendored.splitlines() if "torchvision" in line and line.startswith(("import", "from"))]
    assert len(replaced) == 1
    original = vendored.replace(replaced[0], "import torchvision", 1)
    assert hashlib.sha256(original.encode("utf-8")).hexdigest() == entry["source_sha256"]


@pytest.mark.skipif(not SCALECAR.is_dir(), reason="Scalecar_v2 checkout not present")
def test_recorded_source_sha256_match_scalecar():
    data = sources()
    for entry in data["files"].values():
        assert sha256(SCALECAR / entry["source"]) == entry["source_sha256"], entry["source"]
    for entry in data["ported"].values():
        assert sha256(SCALECAR / entry["source"]) == entry["source_sha256"], entry["source"]


# ---------------------------------------------------------------------------
# torchvision-free ResNet-18
# ---------------------------------------------------------------------------


def _torch_model_modules():
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision", reason="run with --with torchvision==0.28.0")
    return torch, torchvision


def _model_with_real_torchvision():
    """The vendored model.py with its one changed line reverted: Scalecar's model.py, on real torchvision."""
    import types

    import sim2real.planner.scalecar_dp.dp as dp_package

    vendored = (PLANNER / "scalecar_dp/dp/model.py").read_text(encoding="utf-8")
    line = next(l for l in vendored.splitlines() if "torchvision" in l and l.startswith(("import", "from")))
    module = types.ModuleType("sim2real.planner.scalecar_dp.dp._model_torchvision")
    module.__package__ = dp_package.__name__
    exec(compile(vendored.replace(line, "import torchvision", 1), "model_torchvision.py", "exec"), module.__dict__)
    return module


def test_resnet18_matches_torchvision_bitwise():
    torch, torchvision = _torch_model_modules()
    from torch import nn

    from sim2real.planner import resnet

    def norm(c):
        return nn.GroupNorm(c // 16, c)

    torch.manual_seed(7)
    reference = torchvision.models.resnet18(weights=None, norm_layer=norm).eval()
    reference_rng = torch.get_rng_state()
    torch.manual_seed(7)
    ours = resnet.models.resnet18(weights=None, norm_layer=norm).eval()

    assert torch.equal(torch.get_rng_state(), reference_rng), "construction must draw the same random numbers"
    expected, actual = reference.state_dict(), ours.state_dict()
    assert list(actual) == list(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
    x = torch.randn(3, 3, 120, 160, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        assert torch.equal(ours(x), reference(x))


def test_planner_model_matches_scalecar_model_on_torchvision_bitwise():
    torch, _ = _torch_model_modules()
    from sim2real.planner.scalecar_dp.dp import model as ours_module

    original_module = _model_with_real_torchvision()
    config = dict(d=384, layers=8, heads=6)
    torch.manual_seed(4242)
    original = original_module.DPModel(**config).eval()
    original_rng = torch.get_rng_state()
    torch.manual_seed(4242)
    ours = ours_module.DPModel(**config).eval()

    assert torch.equal(torch.get_rng_state(), original_rng)
    expected, actual = original.state_dict(), ours.state_dict()
    assert list(actual) == list(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key
    frames = torch.rand(4, 2, 120, 160, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(ours.depth.encode_frames(frames), original.depth.encode_frames(frames))


# ---------------------------------------------------------------------------
# Checkpoint: sha256 lock, weights_only=True
# ---------------------------------------------------------------------------


def test_checkpoint_with_another_sha256_is_refused_before_unpickling(tmp_path):
    from sim2real.planner.checkpoint import load_planner

    fake = tmp_path / "ckpt.pt"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="sha256"):
        load_planner(fake, sha256="0" * 64, device="cpu")


def test_pinned_checkpoint_loads_like_scalecars_planner():
    torch = pytest.importorskip("torch")
    from sim2real.planner.checkpoint import CHECKPOINT_SHA256, DEFAULT_CHECKPOINT, load_planner
    from sim2real.planner.scalecar_dp.dp import runtime

    if not DEFAULT_CHECKPOINT.is_file():
        pytest.skip(f"{DEFAULT_CHECKPOINT} not present")
    assert CHECKPOINT_SHA256 == "d5e12b4a1fd27ad2eeb0d3a4cbd33aeb1715c8fe0b24472032fe49693429863c"

    ours = load_planner(DEFAULT_CHECKPOINT, sha256=CHECKPOINT_SHA256, device="cpu")
    scalecar = runtime.Planner(str(DEFAULT_CHECKPOINT), device="cpu")  # weights_only=False inside

    assert ours.cfg == scalecar.cfg
    for name in ("steps", "w0", "lam", "margin", "prev"):
        assert getattr(ours, name) == getattr(scalecar, name), name
    assert list(ours.latencies) == list(scalecar.latencies) and ours.latencies.maxlen == scalecar.latencies.maxlen
    assert ours.dev == scalecar.dev and not ours.model.training
    for module in ("model", "norm"):
        expected, actual = getattr(scalecar, module).state_dict(), getattr(ours, module).state_dict()
        assert list(actual) == list(expected)
        for key in expected:
            assert torch.equal(actual[key], expected[key]), (module, key)


# ---------------------------------------------------------------------------
# Go-time pelvis height (leg FK, feet on the ground)
# ---------------------------------------------------------------------------

UPRIGHT = np.array([1.0, 0.0, 0.0, 0.0])


def _tracker_pose(**by_joint: float) -> np.ndarray:
    from sim2real.planner.provider import JOINT_NAMES

    q = np.zeros(len(JOINT_NAMES))
    for name, value in by_joint.items():
        q[JOINT_NAMES.index(f"{name}_joint")] = value
    return q


def test_standing_height_of_the_zero_pose():
    from sim2real.planner.height import standing_height
    from sim2real.planner.provider import JOINT_NAMES

    result = standing_height(np.zeros(29), list(JOINT_NAMES), UPRIGHT)

    assert result["height_m"] == pytest.approx(0.7918958624002903, abs=1e-12)
    assert result["difference_m"] == pytest.approx(0.0, abs=1e-12)


def test_standing_height_refuses_one_foot_lifted():
    from sim2real.planner.height import standing_height
    from sim2real.planner.provider import JOINT_NAMES

    lifted = _tracker_pose(left_hip_pitch=-0.8, left_knee=1.6, left_ankle_pitch=-0.8)
    with pytest.raises(ValueError, match="Feet-on-ground"):
        standing_height(lifted, list(JOINT_NAMES), UPRIGHT)


@pytest.mark.skipif(not SCALECAR.is_dir(), reason="Scalecar_v2 checkout not present")
def test_standing_height_matches_scalecar():
    from sim2real.planner.height import standing_height
    from sim2real.planner.provider import JOINT_NAMES

    scalecar_height = _scalecar_module("scalecar_deploy.hardware.height").standing_height
    rng = np.random.default_rng(3)
    poses = [np.zeros(29), _tracker_pose(left_hip_pitch=-0.2, right_hip_pitch=-0.2, left_knee=0.42, right_knee=0.42,
                                         left_ankle_pitch=-0.23, right_ankle_pitch=-0.23)]
    poses += [poses[1] + rng.normal(0.0, 0.03, 29) for _ in range(20)]
    for q in poses:
        quat = np.array([1.0, *rng.normal(0.0, 0.02, 3)])
        quat /= np.linalg.norm(quat)
        try:
            expected = scalecar_height(q, list(JOINT_NAMES), quat)
        except ValueError as error:
            with pytest.raises(ValueError, match="Feet-on-ground"):
                standing_height(q, list(JOINT_NAMES), quat)
            continue
        assert standing_height(q, list(JOINT_NAMES), quat) == expected


def _scalecar_module(name: str):
    """Import a module from the Scalecar_v2 checkout without writing bytecode into it."""
    import importlib

    sys.dont_write_bytecode = True
    for path in (SCALECAR / "source/scalecar_dp", SCALECAR / "source/scalecar_deploy"):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return importlib.import_module(name)


# ---------------------------------------------------------------------------
# The CUDA Planner worker process
# ---------------------------------------------------------------------------


def _wait_for(worker, predicate, timeout_s: float):
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        item = worker.poll()
        if item is not None and predicate(item):
            return item
        time.sleep(0.005)
    raise AssertionError(f"worker gave no matching result within {timeout_s} s (last: {worker.poll()})")


def test_worker_reports_a_checkpoint_with_another_sha256(tmp_path):
    from sim2real.planner.worker import GpuPlannerWorker

    fake = tmp_path / "ckpt.pt"
    fake.write_bytes(b"not a checkpoint")
    worker = GpuPlannerWorker(dict(checkpoint=str(fake), checkpoint_sha256="0" * 64))
    try:
        item = _wait_for(worker, lambda i: i.get("status") == "error", 60.0)
        assert "sha256" in item["error"]
    finally:
        worker.close()


def test_worker_plans_on_cuda_without_cudnn_on_the_requesters_clock():
    torch = pytest.importorskip("torch")
    from sim2real.planner.checkpoint import DEFAULT_CHECKPOINT
    from sim2real.planner.provider import JOINT_NAMES
    from sim2real.planner.scalecar_dp.dp import runtime
    from sim2real.planner.worker import GpuPlannerWorker

    if not torch.cuda.is_available() or not DEFAULT_CHECKPOINT.is_file():
        pytest.skip("needs CUDA and the Planner checkpoint")
    depth, states = runtime.DepthBuffer(), runtime.StateBuffer()
    rng = np.random.default_rng(0)
    for i in range(30):
        depth.push(rng.integers(300, 3000, (120, 160), dtype=np.uint16), i / 30, downsampled=True)
    q = np.zeros(29)
    q[[JOINT_NAMES.index(f"{s}_knee_joint") for s in ("left", "right")]] = 0.4
    states.push(1.0, q, [0.0, 0.0, 0.78], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0])
    observer = runtime.Observer(depth, states)
    observer.start(1.0)
    obs = observer.observe(1.0)
    offset = 500.0  # the requester's clock runs 500 s ahead of the worker's monotonic clock

    worker = GpuPlannerWorker(dict())
    try:
        loaded = _wait_for(worker, lambda i: i.get("status") in ("loaded", "error"), 120.0)
        assert loaded["status"] == "loaded", loaded
        assert loaded["cudnn_enabled"] is False and loaded["device"].startswith("cuda")
        import time

        submitted = time.monotonic() + offset
        assert worker.submit(dict(generation=1, request_id=1, observation=obs, warmup=False,
                                  depth_times=[depth.t[i] for i in obs.depth_idx],
                                  clock_offset_s=obs.t0 - (submitted - offset)))
        item = _wait_for(worker, lambda i: i.get("status") in ("ready", "error"), 60.0)
        received = time.monotonic() + offset
    finally:
        worker.close()
    assert item["status"] == "ready", item
    chunk = item["chunk"]
    assert chunk.rows.shape == (128, 38) and np.isfinite(chunk.rows).all()
    assert chunk.t0 == obs.t0
    # t_ready is on the requester's clock: t0 plus the wall time the plan took.
    assert 0.0 < chunk.t_ready - chunk.t0 <= received - submitted
    assert item["inference_ms"] / 1000 <= chunk.t_ready - chunk.t0
