"""Scalecar d1146 Tracker re-export (dixiatielu/sim2real_scalecar#2).

The exporter rebuilds the Tracker from Scalecar's ``actor.onnx`` and re-exports
it for sim2real. These tests run it on the pinned ModelScope inputs and compare
the artifact with the original Scalecar ONNX. The exporter needs torch 2.7.0:

    UV_FROZEN=1 uv run --with pytest --with torch==2.7.0 \
        --index https://download.pytorch.org/whl/cpu \
        python -m pytest tests/test_scalecar_tracker_export.py

(``python -m`` puts the repo root first on sys.path; a dependency installs a
top-level ``scripts`` package that would otherwise shadow this repo's.)
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import yaml

from scripts.export_scalecar_tracker_onnx import DEFAULT_MODELSCOPE_ROOT, export_tracker
from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.base_policy import BasePolicy
from sim2real.rl_policy.inference.onnx_module import ONNXModule
from sim2real.rl_policy.utils.motion import MotionData

TRACKERS_DIR = DEFAULT_MODELSCOPE_ROOT / "trackers" / "d1146_dpsd_ft"
VARIANTS = ("ft500ema", "ft1000ema")
TOLERANCE = 5e-6

def _torch_version() -> str:
    try:
        import torch
    except ImportError:
        return "missing"
    return torch.__version__


pytestmark = [
    pytest.mark.skipif(
        not TRACKERS_DIR.is_dir(),
        reason=f"ModelScope Scalecar trackers not downloaded: {TRACKERS_DIR}",
    ),
    pytest.mark.skipif(
        not _torch_version().startswith("2.7.0"),
        reason=f"the exporter needs torch 2.7.0 (found {_torch_version()}); see the module docstring",
    ),
]


@pytest.fixture(scope="module", params=VARIANTS)
def exported(request, tmp_path_factory):
    variant = request.param
    output_dir = tmp_path_factory.mktemp(variant)
    export_tracker(variant, output_dir=output_dir)
    source_dir = TRACKERS_DIR / variant
    return SimpleNamespace(
        variant=variant,
        output_dir=output_dir,
        onnx_path=output_dir / "policy.onnx",
        policy_config=yaml.safe_load((output_dir / "policy.yaml").read_text()),
        comparison=json.loads((output_dir / "comparison.json").read_text()),
        original_onnx=source_dir / "actor.onnx",
        deployment=json.loads((source_dir / "deployment.json").read_text()),
    )


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------


def _dims(value_info) -> list:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def test_artifact_is_unbatched_static_and_loadable_by_ort_116(exported):
    model = onnx.load(str(exported.onnx_path))
    onnx.checker.check_model(model, full_check=True)

    assert model.ir_version <= 9
    default_opsets = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    assert default_opsets and max(default_opsets) <= 19

    inputs = {i.name: _dims(i) for i in model.graph.input}
    assert all(len(dims) == 1 and isinstance(dims[0], int) and dims[0] > 0 for dims in inputs.values()), inputs
    assert "action" not in inputs
    assert set(inputs) == set(exported.policy_config["observation"])

    outputs = {o.name: _dims(o) for o in model.graph.output}
    assert outputs == {"action": [29]}


def test_export_records_comparison_against_original(exported):
    comparison = exported.comparison
    assert comparison["num_samples"] >= 200
    assert comparison["max_abs_error"] <= TOLERANCE
    assert comparison["passed"] is True

    pinned = {}
    for line in (TRACKERS_DIR / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split()
        pinned[rel.removeprefix("./")] = digest
    for name in ("actor.onnx", "deployment.json"):
        assert comparison["sources"][name]["sha256"] == pinned[f"{exported.variant}/{name}"]


def test_policy_yaml_carries_tracker_contract(exported):
    cfg = exported.policy_config
    dep = exported.deployment
    names = dep["joint_names"]

    assert cfg["model_path"] == "policy.onnx"
    assert cfg["policy_joint_names"] == names
    assert cfg["joint_names_simulation"] == list(get_robot_cfg("g1").joint_names)
    for key, dep_key in [
        ("default_joint_pos", "default_joint_pos"),
        ("action_scale", "action_scale"),
        ("joint_kp", "stiffness"),
        ("joint_kd", "damping"),
    ]:
        assert list(cfg[key]) == names, key
        np.testing.assert_allclose([cfg[key][n] for n in names], dep[dep_key], err_msg=key)

    future_steps = cfg["motion"]["future_steps"]
    assert future_steps[0] == 0
    assert set(dep["reference_offsets"]) <= set(future_steps)
    assert "pelvis" in cfg["motion"]["body_names"]


def test_policy_yaml_carries_tracker_action_post_processing(exported):
    """#3's runtime keys, filled from deployment.json by name (Tracker order)."""
    cfg = exported.policy_config
    dep = exported.deployment

    limits = cfg["soft_joint_pos_limits"]
    assert list(limits) == dep["joint_names"]
    assert limits["left_knee_joint"] == [-0.013090252876281738, 2.8056230545043945]
    for name, bounds in zip(dep["joint_names"], dep["soft_joint_pos_limits"], strict=True):
        assert limits[name] == bounds, name
    assert cfg["max_target_velocity_rad_s"] == 15.0
    assert cfg["shaping_target"] == "motion_first_frame"


def test_deployment_contract_ships_next_to_the_policy(exported):
    """The vehicle sim2sim takes its physics (kp/kd, armature, effort limits) from the contract."""
    shipped = exported.output_dir / "deployment.json"

    assert shipped.read_bytes() == (TRACKERS_DIR / exported.variant / "deployment.json").read_bytes()


def test_refuses_source_files_that_do_not_match_pinned_sha256(tmp_path):
    modelscope_root = tmp_path / "modelscope"
    source_dir = modelscope_root / "trackers" / "d1146_dpsd_ft" / "ft500ema"
    shutil.copytree(TRACKERS_DIR / "ft500ema", source_dir)
    deployment = source_dir / "deployment.json"
    deployment.write_text(deployment.read_text() + "\n")

    with pytest.raises(ValueError, match="sha256"):
        export_tracker("ft500ema", modelscope_root=modelscope_root, output_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "policy.onnx").exists()


# ---------------------------------------------------------------------------
# Runtime equivalence: sim2real observation helpers + exported ONNX
#   vs. Scalecar's observation layout + original ONNX
# ---------------------------------------------------------------------------
#
# The expected side follows Scalecar's deploy code directly:
#   S:source/scalecar_deploy/scalecar_deploy/observations.py (ObservationBuilder)
#   S:source/scalecar_policy/scalecar_policy/reference.py (reference_only_features)


def _sc_rotate(q, v):
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def _sc_multiply(a, b):
    w = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, v), axis=-1)


def _sc_conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _sc_reference_only_features(q, dq, pos, quat, offsets):
    w, x, y, z = quat[0]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    inv = np.array([np.cos(yaw / 2), 0.0, 0.0, -np.sin(yaw / 2)])
    rel = _sc_multiply(inv, quat)
    tangent_normal = np.concatenate(
        (_sc_rotate(rel, np.array([1.0, 0.0, 0.0])), _sc_rotate(rel, np.array([0.0, 0.0, 1.0]))), -1
    )
    return np.concatenate(
        (q, dq, _sc_rotate(inv, pos - pos[0]), pos[:, 2:3], tangent_normal, np.asarray(offsets, float)[:, None]),
        -1,
    ).astype(np.float32)


def _sc_proprioception(state, sdk_index, default):
    return np.concatenate(
        (
            _sc_rotate(_sc_conjugate(state.root_quat_w), np.array([0.0, 0.0, -1.0])),
            state.root_ang_vel_b,
            state.joint_pos[sdk_index] - default,
            state.joint_vel[sdk_index] * 0.05,
        )
    ).astype(np.float32)


def _random_quat(rng, n, tilt):
    roll, pitch = rng.uniform(-tilt, tilt, (2, n))
    yaw = rng.uniform(-np.pi, np.pi, n)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        -1,
    )


def _random_robot_state(rng, sdk_names):
    return SimpleNamespace(
        joint_names=list(sdk_names),
        root_quat_w=_random_quat(rng, 1, 0.3)[0],
        root_ang_vel_b=rng.normal(size=3),
        joint_pos=rng.normal(size=29) * 0.5,
        joint_vel=rng.normal(size=29) * 2.0,
    )


def _random_motion_window(rng, num_frames, motion_body_names):
    num_bodies = len(motion_body_names)
    body_pos = rng.normal(size=(num_frames, num_bodies, 3))
    body_pos[..., 2] = rng.uniform(0.3, 1.0, (num_frames, num_bodies))
    body_quat = _random_quat(rng, num_frames * num_bodies, 0.6).reshape(num_frames, num_bodies, 4)
    return dict(
        joint_pos=rng.normal(size=(num_frames, 29)) * 0.5,
        joint_vel=rng.normal(size=(num_frames, 29)),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
    )


def test_sim2real_runtime_matches_original_tracker(exported):
    cfg = exported.policy_config
    dep = exported.deployment
    rng = np.random.default_rng(2)

    sdk_names = list(get_robot_cfg("g1").joint_names)
    tracker_names = dep["joint_names"]
    sdk_index = np.array([sdk_names.index(n) for n in tracker_names])
    default = np.asarray(dep["default_joint_pos"])
    offsets = dep["reference_offsets"]

    motion_future_steps = list(cfg["motion"]["future_steps"])
    reference_frames = [motion_future_steps.index(o) for o in offsets]
    motion_body_names = ["pelvis", "torso_link"]

    # Minimal stand-in for Tracking: real helpers are built from the YAML via
    # BasePolicy.setup_observations, exactly as at runtime.
    env = SimpleNamespace(
        state_processor=_random_robot_state(rng, sdk_names),
        joint_names_simulation=cfg["joint_names_simulation"],
        body_names_simulation=cfg["body_names_simulation"],
        policy_joint_names=cfg["policy_joint_names"],
        num_actions=len(cfg["policy_joint_names"]),
        motion_config=cfg["motion"],
        motion_future_steps=np.asarray(motion_future_steps),
        motion_joint_names=sdk_names,
        motion_body_names=motion_body_names,
        motion_data=None,
        state_dict={"action": np.zeros(29, dtype=np.float32)},
    )

    def set_motion(window):
        zeros = np.zeros_like(window["body_pos_w"])
        env.motion_data = MotionData(
            joint_pos=window["joint_pos"][None],
            joint_vel=window["joint_vel"][None],
            body_pos_w=window["body_pos_w"][None],
            body_lin_vel_w=zeros[None],
            body_quat_w=window["body_quat_w"][None],
            body_ang_vel_w=zeros[None],
        )

    set_motion(_random_motion_window(rng, len(motion_future_steps), motion_body_names))
    BasePolicy.setup_observations(env, cfg["observation"])
    BasePolicy.reset(env)

    module = ONNXModule(str(exported.onnx_path), providers="cpu")
    original = ort.InferenceSession(str(exported.original_onnx), providers=["CPUExecutionProvider"])

    states, prev_actions = [], []
    max_err, compared = 0.0, 0
    for tick in range(203):
        state = _random_robot_state(rng, sdk_names)
        vars(env.state_processor).update(vars(state))
        window = _random_motion_window(rng, len(motion_future_steps), motion_body_names)
        set_motion(window)
        env.state_dict["action"] = rng.normal(size=29).astype(np.float32)  # a_{t-1}

        states.append(state)
        prev_actions.append(env.state_dict["action"])

        BasePolicy.update(env)
        obs_dict, _ = BasePolicy.prepare_obs_for_rl(env)
        env.state_dict.update(obs_dict)
        action = module(env.state_dict)["action"]
        assert action.shape == (29,)
        if tick < 2:
            continue

        pelvis = motion_body_names.index("pelvis")
        task = _sc_reference_only_features(
            window["joint_pos"][reference_frames][:, sdk_index],
            window["joint_vel"][reference_frames][:, sdk_index],
            window["body_pos_w"][reference_frames, pelvis],
            window["body_quat_w"][reference_frames, pelvis],
            offsets,
        )
        expected = original.run(
            None,
            {
                "policy": np.stack([_sc_proprioception(s, sdk_index, default) for s in states[-3:]])[None],
                "policy_task": task[None],
                "action": np.stack(prev_actions[-3:])[None],
            },
        )[0][0]
        max_err = max(max_err, float(np.abs(action - expected).max()))
        compared += 1

    assert compared >= 200
    assert max_err <= TOLERANCE, max_err
