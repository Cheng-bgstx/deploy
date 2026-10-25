#!/usr/bin/env python3
"""Compare Reference Tokens of a converted any4hdmi motion with Scalecar's own.

For every frame t the six tokens use frames ``min(t + offset, F - 1)``.

- Scalecar side: the source NPZ read the way Scalecar's ``MotionReference``
  reads it (contract joint order, last-frame ``joint_vel`` zeroed), fed to
  Scalecar's ``reference_only_features``.
- sim2real side: the converted motion loaded through
  ``MotionDataset.create_from_path`` and sliced with ``get_slice``, the same
  path ``tracking.py`` uses, fed to the same function.

any4hdmi keeps loaded motion fields in a float16 backing, so the script also
reports the comparison against any4hdmi's float32 FK cache (the data just
before float16 storage) and checks that the loader output equals that cache
rounded to float16.

    uv run --frozen python -I scripts/scalecar/compare_reference_tokens.py \\
        --dataset-root /abs/path/checkpoints/scalecar_d1146/motions \\
        --takes take1512 take0595
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sim2real.planner.checkpoint import MODELSCOPE_ROOT  # noqa: E402

REFERENCE_OFFSETS = (0, 2, 4, 8, 16, 32)
TOKEN_COMPONENTS = {
    "q": slice(0, 29),
    "dq": slice(29, 58),
    "displacement": slice(58, 61),
    "height": slice(61, 62),
    "rotation": slice(62, 68),
    "offset": slice(68, 69),
}
ROOT_BODY_NAME = "pelvis"
SCALECAR_ROOT = Path.home() / "projects/Scalecar_v2"
SCALECAR_REFERENCE_PY = SCALECAR_ROOT / "source/scalecar_policy/scalecar_policy/reference.py"
TRACKER_CONTRACT = MODELSCOPE_ROOT / "trackers/d1146_dpsd_ft/ft500ema/deployment.json"

ReferenceFeatures = Callable[..., np.ndarray]


def load_reference_only_features(reference_py: str | Path) -> ReferenceFeatures:
    """Load Scalecar's ``reference_only_features`` from a copy of its source text.

    The file is compiled in memory, so nothing is written next to it.
    """
    path = Path(reference_py)
    module = types.ModuleType("scalecar_reference_copy")
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), module.__dict__)
    return module.reference_only_features


@dataclass
class _Reference:
    """One motion as the token formula needs it: joints in tracker order, pelvis pose."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    pelvis_pos: np.ndarray
    pelvis_quat: np.ndarray

    def rounded_to_float16(self) -> "_Reference":
        """The same data as stored by the any4hdmi float16 backing."""
        return _Reference(
            *(a.astype(np.float16) for a in (self.joint_pos, self.joint_vel, self.pelvis_pos, self.pelvis_quat))
        )

    def upcast(self) -> "_Reference":
        """float64 copies with the pelvis quaternion renormalised.

        sim2real's MotionDataset hands the any4hdmi float16 backing out as
        float16 numpy arrays. Arithmetic on them stays in float16, and their
        quaternions are off unit norm by up to ~3e-4, beyond the 1e-4 that
        Scalecar's ``reference_only_features`` accepts.
        """
        quat = self.pelvis_quat.astype(np.float64)
        return _Reference(
            self.joint_pos.astype(np.float64),
            self.joint_vel.astype(np.float64),
            self.pelvis_pos.astype(np.float64),
            quat / np.linalg.norm(quat, axis=-1, keepdims=True),
        )


def _tokens(
    ref: _Reference, features: ReferenceFeatures, offsets: np.ndarray
) -> np.ndarray:
    """Scalecar sampling: future frames clamped to the last frame (``kinematics.py:79-83``)."""
    frames = len(ref.joint_pos)
    out = np.empty((frames, len(offsets), 69), dtype=np.float32)
    for t in range(frames):
        idx = np.minimum(t + offsets, frames - 1)
        out[t] = features(
            ref.joint_pos[idx], ref.joint_vel[idx], ref.pelvis_pos[idx], ref.pelvis_quat[idx], offsets
        )
    return out


def _scalecar_reference(source_npz: Path, joint_names: list[str]) -> _Reference:
    """Mirror Scalecar ``MotionReference``: reorder by name, zero last-frame joint_vel."""
    with np.load(source_npz, allow_pickle=False) as data:
        src_joints = [str(n) for n in data["robot_joint_names"]]
        bodies = [str(n) for n in data["robot_body_names"]]
        ji = [src_joints.index(n) for n in joint_names]
        root = bodies.index(ROOT_BODY_NAME)
        joint_vel = data["joint_vel"][:, ji].copy()
        joint_vel[-1] = 0.0
        return _Reference(
            joint_pos=data["joint_pos"][:, ji].copy(),
            joint_vel=joint_vel,
            pelvis_pos=data["body_pos_w"][:, root].copy(),
            pelvis_quat=data["body_quat_w"][:, root].copy(),
        )


def _sim2real_loader_tokens(
    motion_path: Path, joint_names: list[str], features: ReferenceFeatures, offsets: np.ndarray, robot: str
) -> tuple[np.ndarray, _Reference]:
    """Tokens from ``MotionDataset.get_slice(motion_ids, starts=t, steps=offsets)``, as tracking.py slices."""
    from sim2real.config.robots import get_robot_cfg
    from sim2real.rl_policy.utils.motion import MotionDataset

    dataset = MotionDataset.create_from_path(str(motion_path), robot_cfg=get_robot_cfg(robot))
    if dataset.num_motions != 1:
        raise ValueError(f"{motion_path} resolved to {dataset.num_motions} motions, expected 1")
    frames = dataset.num_steps
    ji = [dataset.joint_names.index(n) for n in joint_names]
    root = dataset.body_names.index(ROOT_BODY_NAME)

    sliced = dataset.get_slice(np.zeros(frames, dtype=np.int64), np.arange(frames), offsets)
    tokens = np.empty((frames, len(offsets), 69), dtype=np.float32)
    for t in range(frames):
        rows = _Reference(
            sliced.joint_pos[t][:, ji],
            sliced.joint_vel[t][:, ji],
            sliced.body_pos_w[t, :, root],
            sliced.body_quat_w[t, :, root],
        ).upcast()
        tokens[t] = features(rows.joint_pos, rows.joint_vel, rows.pelvis_pos, rows.pelvis_quat, offsets)
    whole = dataset.get_slice(np.array([0]), np.array([0]), np.arange(frames))
    ref = _Reference(
        joint_pos=whole.joint_pos[0][:, ji],
        joint_vel=whole.joint_vel[0][:, ji],
        pelvis_pos=whole.body_pos_w[0, :, root],
        pelvis_quat=whole.body_quat_w[0, :, root],
    )
    return tokens, ref


def _fp32_fk_cache_reference(motion_path: Path, joint_names: list[str]) -> _Reference:
    """any4hdmi's float32 FK cache for the motion: what the loader stores before float16."""
    import sim2real
    from any4hdmi.dataset.fk_cache import FKCache

    base_dir = Path(sim2real.__file__).parent.parent
    entry = FKCache.from_inputs(
        input_paths=[Path(motion_path).resolve()], target_fps=50, base_dir=base_dir
    ).get_or_build()
    if len(entry.starts) != 1:
        raise ValueError(f"{motion_path} resolved to {len(entry.starts)} motions, expected 1")
    start, end = entry.starts[0], entry.ends[0]
    ji = [entry.joint_names.index(n) for n in joint_names]
    root = entry.body_names.index(ROOT_BODY_NAME)
    fields = {k: v[start:end].numpy() for k, v in entry.storage_fields.items()}
    return _Reference(
        joint_pos=fields["joint_pos"][:, ji].copy(),
        joint_vel=fields["joint_vel"][:, ji].copy(),
        pelvis_pos=fields["body_pos_w"][:, root].copy(),
        pelvis_quat=fields["body_quat_w"][:, root].copy(),
    )


def _component_max_errors(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {name: float(diff[..., s].max()) for name, s in TOKEN_COMPONENTS.items()}


def compare_reference_tokens(
    motion_path: str | Path,
    source_npz: str | Path,
    *,
    reference_only_features: ReferenceFeatures,
    offsets: tuple[int, ...] = REFERENCE_OFFSETS,
    joint_names: list[str] | None = None,
    robot: str = "g1",
) -> dict[str, Any]:
    """Per-component max |Δ| between sim2real and Scalecar Reference Tokens for one motion.

    ``joint_names`` is the tracker joint order; it defaults to the source NPZ's
    ``robot_joint_names``.
    """
    motion_path, source_npz = Path(motion_path), Path(source_npz)
    offsets_arr = np.asarray(offsets, dtype=np.int64)
    if joint_names is None:
        with np.load(source_npz, allow_pickle=False) as data:
            joint_names = [str(n) for n in data["robot_joint_names"]]

    scalecar = _scalecar_reference(source_npz, joint_names)
    scalecar_tokens = _tokens(scalecar, reference_only_features, offsets_arr)
    loader_tokens, loader = _sim2real_loader_tokens(
        motion_path, joint_names, reference_only_features, offsets_arr, robot
    )
    fp32 = _fp32_fk_cache_reference(motion_path, joint_names)
    fp32_tokens = _tokens(fp32.upcast(), reference_only_features, offsets_arr)
    fp16_rounded_tokens = _tokens(
        fp32.rounded_to_float16().upcast(), reference_only_features, offsets_arr
    )

    frames = len(scalecar.joint_pos)
    if loader_tokens.shape != scalecar_tokens.shape:
        raise ValueError(
            f"frame count differs: sim2real {loader_tokens.shape[0]}, Scalecar {frames}"
        )

    # dq: Scalecar uses np.gradient (central) and zeroes the last frame;
    # any4hdmi recomputes a forward difference and repeats it on the last frame.
    dq_s = TOKEN_COMPONENTS["dq"]
    touches_last = (np.arange(frames)[:, None] + offsets_arr[None, :]) >= frames - 1
    dq_diff = np.abs(
        loader_tokens[..., dq_s].astype(np.float64) - scalecar_tokens[..., dq_s].astype(np.float64)
    )
    fp32_dq_diff = np.abs(fp32_tokens[..., dq_s].astype(np.float64) - scalecar_tokens[..., dq_s])
    return {
        "motion": str(motion_path),
        "source_npz": str(source_npz),
        "frames": frames,
        "offsets": list(map(int, offsets_arr)),
        "loader": _component_max_errors(loader_tokens, scalecar_tokens),
        "fp32_fk_cache": _component_max_errors(fp32_tokens, scalecar_tokens),
        "loader_vs_fp16_rounded_fp32": _component_max_errors(loader_tokens, fp16_rounded_tokens),
        "pelvis_world": {
            "loader_max_abs_xy": float(np.abs(loader.pelvis_pos[:, :2] - scalecar.pelvis_pos[:, :2]).max()),
            "loader_max_abs_yaw": float(np.abs(_wrap(_yaw(loader.pelvis_quat) - _yaw(scalecar.pelvis_quat))).max()),
            "fp32_max_abs_xy": float(np.abs(fp32.pelvis_pos[:, :2] - scalecar.pelvis_pos[:, :2]).max()),
            "fp32_max_abs_yaw": float(np.abs(_wrap(_yaw(fp32.pelvis_quat) - _yaw(scalecar.pelvis_quat))).max()),
            "frame0_scalecar_xyz": scalecar.pelvis_pos[0].astype(float).tolist(),
            "frame0_loader_xyz": loader.pelvis_pos[0].astype(float).tolist(),
        },
        "loader_arrays": {
            "dtype": str(loader.pelvis_pos.dtype),
            "max_abs_quat_norm_deviation": float(
                np.abs(np.linalg.norm(loader.pelvis_quat.astype(np.float64), axis=-1) - 1.0).max()
            ),
        },
        "dq": {
            "loader_max_abs": float(dq_diff.max()),
            "loader_max_abs_excluding_last_frame_rows": float(dq_diff[~touches_last].max()),
            "fp32_max_abs_excluding_last_frame_rows": float(fp32_dq_diff[~touches_last].max()),
            "loader_rms_excluding_last_frame_rows": float(np.sqrt(np.mean(dq_diff[~touches_last] ** 2))),
            "loader_p50": float(np.percentile(dq_diff, 50)),
            "loader_p99": float(np.percentile(dq_diff, 99)),
            "loader_worst": _worst_entry(dq_diff, offsets_arr, joint_names),
            "scalecar_max_abs": float(np.abs(scalecar_tokens[..., dq_s]).max()),
        },
        "dq_last_frame": {
            "scalecar_max_abs": float(np.abs(scalecar.joint_vel[-1]).max()),
            "sim2real_max_abs": float(np.abs(loader.joint_vel[-1]).max()),
            "sim2real_equals_second_last": bool(np.array_equal(loader.joint_vel[-1], loader.joint_vel[-2])),
            "loader_max_abs_in_last_frame_rows": float(dq_diff[touches_last].max()),
        },
    }


def _worst_entry(diff: np.ndarray, offsets: np.ndarray, joint_names: list[str]) -> dict[str, Any]:
    t, k, j = np.unravel_index(int(diff.argmax()), diff.shape)
    return {"frame": int(t), "offset": int(offsets[k]), "reference_frame": int(t + offsets[k]), "joint": joint_names[j]}


def _yaw(quat_wxyz: np.ndarray) -> np.ndarray:
    """Heading of a WXYZ quaternion; independent of its norm."""
    w, x, y, z = np.moveaxis(np.asarray(quat_wxyz, dtype=np.float64), -1, 0)
    return np.arctan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class CompareArgs:
    """Compare sim2real and Scalecar Reference Tokens for converted takes."""

    dataset_root: Path = Path(__file__).resolve().parents[2] / "checkpoints/scalecar_d1146/motions"
    takes: tuple[str, ...] = ("take1512", "take0595")
    tar: Path | None = None
    """Scalecar motions tar (default: the ModelScope cache path used by convert_motions)."""
    sha256sums: Path | None = None
    scalecar_reference: Path = SCALECAR_REFERENCE_PY
    """Scalecar's reference.py (default: in the Scalecar_v2 checkout at ~/projects/Scalecar_v2)."""
    contract: Path = TRACKER_CONTRACT
    """deployment.json whose joint_names give the tracker joint order (default: in the ModelScope cache)."""
    output_json: Path | None = None
    """Default: <dataset_root>/../motion_token_comparison.json"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_compare(args: CompareArgs) -> dict[str, Any]:
    from scripts.scalecar.convert_motions import DEFAULT_SHA256SUMS, DEFAULT_TAR, extract_takes

    tar = args.tar or DEFAULT_TAR
    sums = args.sha256sums or DEFAULT_SHA256SUMS
    features = load_reference_only_features(args.scalecar_reference)
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    joint_names = list(contract["joint_names"])
    offsets = tuple(int(o) for o in contract.get("reference_offsets", REFERENCE_OFFSETS))
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    report: dict[str, Any] = {
        "scalecar_reference": str(args.scalecar_reference),
        "scalecar_reference_sha256": _sha256(args.scalecar_reference),
        "contract": str(args.contract),
        "contract_sha256": _sha256(args.contract),
        "takes": {},
    }
    with tempfile.TemporaryDirectory(prefix="scalecar_takes_") as tmp:
        sources = extract_takes(tar, sums, args.takes, tmp)
        for take, source in sources.items():
            with np.load(source, allow_pickle=False) as data:
                npz_joints = [str(n) for n in data["robot_joint_names"]]
            result = compare_reference_tokens(
                dataset_root / "motions" / f"{take}.npz",
                source,
                reference_only_features=features,
                offsets=offsets,
                joint_names=joint_names,
            )
            result["npz_joint_order_equals_contract"] = npz_joints == joint_names
            result["source_npz"] = f"{tar.name}:{take}.npz"
            report["takes"][take] = result
            _print_result(take, result)

    output_json = args.output_json or dataset_root.parent / "motion_token_comparison.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[compare] wrote {output_json}")
    return report


def _print_result(take: str, result: dict[str, Any]) -> None:
    print(f"== {take}: {result['frames']} frames, offsets {result['offsets']}")
    print(f"   {'component':<14}{'loader (fp16)':>16}{'fp32 FK cache':>16}{'loader - fp16(fp32)':>22}")
    for name in TOKEN_COMPONENTS:
        print(
            f"   {name:<14}{result['loader'][name]:>16.3e}{result['fp32_fk_cache'][name]:>16.3e}"
            f"{result['loader_vs_fp16_rounded_fp32'][name]:>22.3e}"
        )
    print(f"   pelvis world: {result['pelvis_world']}")
    print(f"   dq: {result['dq']}")
    print(f"   dq last frame: {result['dq_last_frame']}")


def main() -> None:
    import tyro

    run_compare(tyro.cli(CompareArgs))


if __name__ == "__main__":
    main()
