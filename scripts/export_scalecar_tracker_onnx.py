#!/usr/bin/env python3
"""Rebuild the Scalecar d1146 Tracker from its ONNX and re-export it for sim2real.

Scalecar ships only ``actor.onnx`` and ``deployment.json`` per Tracker, so the
PyTorch actor is rebuilt from the ONNX initializers (ADR-0001) and re-exported
through a wrapper that follows sim2real's conventions:

- unbatched, static, flat inputs, each the output of an existing sim2real
  observation helper (the observation groups in the generated ``policy.yaml``);
- Reference Tokens computed inside the graph from the raw Reference;
- a single ``action[29]`` output in Tracker joint order (``policy_joint_names``).

Each export is compared with the original ONNX on random inputs in sim2real
layout, mapped to Scalecar layout for the original. Output directory:
``policy.onnx``, ``policy.yaml`` and ``comparison.json``.

Run in the root project with torch pinned to the version that produced the
original ONNX (CPU wheel is enough)::

    UV_FROZEN=1 uv run --with torch==2.7.0 --index https://download.pytorch.org/whl/cpu \\
        python scripts/export_scalecar_tracker_onnx.py --variant ft500ema ft1000ema
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import yaml
from onnx import numpy_helper

from sim2real.config.robots import get_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELSCOPE_ROOT = (
    Path.home() / ".cache" / "modelscope" / "models" / "Hzzzz001--scalecar_v2" / "snapshots" / "master"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "checkpoints" / "scalecar_d1146" / "tracker"
TRACKERS_SUBDIR = Path("trackers") / "d1146_dpsd_ft"

# sha256 of the inputs, as listed in ModelScope trackers/d1146_dpsd_ft/SHA256SUMS.
PINNED_SOURCES = {
    "ft500ema": {
        "actor.onnx": "46435688c56ef97122d2bfd8f262dd96c85c7415d10c764ed21ea9d8c5c4fcd1",
        "deployment.json": "b2977eb3d4057a6e64acc3332f01e672dc442f1d6023f068ee94cf25ab8a8c56",
    },
    "ft1000ema": {
        "actor.onnx": "8cb0eaeb2718290cb93cd879985b6a4d6fca782782ebd63bf783eaae8aa1cbd6",
        "deployment.json": "b4ba4372c1e2d9ce41f30e47b2aea5588864d1e12000ce9dacd65f4d61aee606",
    },
}

REQUIRED_TORCH = "2.7.0"  # producer_version of the original actor.onnx
OPSET = 17  # same as the original; torch 2.7 writes IR 8, loadable by onnxruntime 1.16
TOLERANCE = 5e-6
NUM_SAMPLES = 256

# HumanoidTransformerAdapt(64, 29, 29, 128, 4, 128, 4) + TaskEmbedder(69, 128)
PROP_DIM = 64
NUM_JOINTS = 29
TOKEN_DIM = 69
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 4
NUM_PARAMETERS = 752_157  # the ONNX initializers hold 752,221 values: these plus one 64-value rope.inv_freq
CONTEXT = 3  # rows of state / action history; row 0 is the oldest

HISTORY_STEPS = [0, 1, 2]  # sim2real history helpers, newest first
PREV_ACTION_STEPS = 3
DQ_SCALE = 0.05
INPUT_NAMES = ["policy", "prev_actions", "ref_command", "ref_pelvis_pos", "ref_pelvis_ori"]


# ---------------------------------------------------------------------------
# Scalecar network: inference path of
# S:source/my_rsl_rl/my_rsl_rl/networks/humanoid_transformer_adapt.py (Scalecar_v2 7b8f370).
# Module names match the original so named ONNX initializers load by name. The
# deployed actor passes no extra task tokens, so that branch is left out.
# ---------------------------------------------------------------------------


class RoPEPositionalEncoding(nn.Module):
    def __init__(self, dim: int, base: float = 10000):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(position, self.inv_freq)
        cos_freqs = torch.cos(freqs).unsqueeze(0)
        sin_freqs = torch.sin(freqs).unsqueeze(0)
        x_even = x[:, :, 0::2]
        x_odd = x[:, :, 1::2]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_even * cos_freqs - x_odd * sin_freqs
        x_rotated[..., 1::2] = x_even * sin_freqs + x_odd * cos_freqs
        return x_rotated


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return self.scale * x / (norm + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, hidden_dim, bias=False)
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, input_dim, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.silu(self.w(x)) * self.v(x))


class HumanoidTransformerAdaptBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.feed_forward = SwiGLU(embed_dim, ff_dim)
        self.rmsnorm1 = RMSNorm(embed_dim)
        self.rmsnorm2 = RMSNorm(embed_dim)
        self.rmsnorm3 = RMSNorm(embed_dim)
        self.cond_norm = RMSNorm(embed_dim)
        self.rope = RoPEPositionalEncoding(embed_dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor, self_attn_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.rmsnorm1(x)
        x_rope = self.rope(x_norm)
        attn_output, _ = self.self_attention(x_rope, x_rope, x_norm, attn_mask=self_attn_mask)
        x = x + attn_output
        x_norm2 = self.rmsnorm2(x)
        c_norm = self.cond_norm(c)
        cross_output, _ = self.cross_attention(query=x_norm2, key=c_norm, value=c_norm)
        x = x + cross_output
        return x + self.feed_forward(self.rmsnorm3(x))


class HumanoidTransformerAdapt(nn.Module):
    def __init__(self, prop_obs_dim, action_dim, output_dim, embed_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.prop_projection = nn.Linear(prop_obs_dim, embed_dim)
        self.action_projection = nn.Linear(action_dim, embed_dim)
        self.empty_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer_blocks = nn.ModuleList(
            [HumanoidTransformerAdaptBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        )
        self.final_norm = RMSNorm(embed_dim)
        self.projection_head = nn.Linear(embed_dim, output_dim)

    def forward(self, prop_obs: torch.Tensor, action_obs: torch.Tensor, task_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, context_size = prop_obs.shape[:2]
        prop_obs = self.prop_projection(prop_obs)
        action_obs = self.action_projection(action_obs)
        # Sequence [s_{t-2}, a_{t-2}, s_{t-1}, a_{t-1}, s_t, E]; action row 0 is dropped.
        context = prop_obs.new_empty(batch_size, 2 * context_size - 1, self.embed_dim)
        context[:, 0::2] = prop_obs
        context[:, 1::2] = action_obs[:, 1:]
        x = torch.cat([context, self.empty_embedding.expand(batch_size, -1, -1)], dim=1)
        self_attn_mask = torch.zeros(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device)
        self_attn_mask[:-1, -1] = True
        for block in self.transformer_blocks:
            x = block(x, task_tokens, self_attn_mask)
        x = self.final_norm(x)
        return self.projection_head(x[:, -1, :])


class TaskEmbedder(nn.Module):
    def __init__(self, task_obs_dim: int, embedding_dim: int):
        super().__init__()
        self.task_projection = nn.Linear(task_obs_dim, embedding_dim)

    def forward(self, task_obs: torch.Tensor) -> torch.Tensor:
        return self.task_projection(task_obs)


class ScalecarActor(nn.Module):
    """The deployed actor in Scalecar layout: policy (B,3,64), policy_task (B,6,69), action (B,3,29)."""

    def __init__(self):
        super().__init__()
        self.transformer = HumanoidTransformerAdapt(
            PROP_DIM, NUM_JOINTS, NUM_JOINTS, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS
        )
        self.reference_encoder = TaskEmbedder(TOKEN_DIM, EMBED_DIM)

    def forward(self, policy: torch.Tensor, policy_task: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.transformer(policy, action, self.reference_encoder(policy_task))


# ---------------------------------------------------------------------------
# Weights from the original ONNX
# ---------------------------------------------------------------------------


def rebuild_actor(model: onnx.ModelProto) -> ScalecarActor:
    """Load every ONNX initializer back into a ScalecarActor."""
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    node_inits = {n.name: [i for i in n.input if i in inits] for n in model.graph.node}
    used: set[str] = set()

    def take(name: str) -> np.ndarray:
        used.add(name)
        return inits[name]

    def anonymous(node: str) -> np.ndarray:
        # torch names folded weights "onnx::MatMul_*" / "onnx::Add_*"; the consumer node identifies them.
        (name,) = [i for i in node_inits[node] if i.startswith("onnx::")]
        return take(name)

    def linear_weight(node: str) -> np.ndarray:
        return anonymous(node).T  # exported as x @ W.T

    actor = ScalecarActor()
    own = {k: v.numpy() for k, v in actor.state_dict().items()}
    state = {key: take(key) for key in own if key in inits}

    state["reference_encoder.task_projection.weight"] = linear_weight("/reference_encoder/task_projection/MatMul")
    state["transformer.prop_projection.weight"] = linear_weight("/transformer/prop_projection/MatMul")
    state["transformer.action_projection.weight"] = linear_weight("/transformer/action_projection/MatMul")
    for b in range(NUM_LAYERS):
        node = f"/transformer/transformer_blocks.{b}"
        key = f"transformer.transformer_blocks.{b}"
        qkv = ("", "_1", "_2")  # q, k, v
        state[f"{key}.self_attention.in_proj_weight"] = np.concatenate(
            [linear_weight(f"{node}/self_attention/MatMul{s}") for s in qkv]
        )
        state[f"{key}.self_attention.in_proj_bias"] = np.concatenate(
            [anonymous(f"{node}/self_attention/Add{s}") for s in qkv]
        )
        for proj in ("w", "v", "output"):
            state[f"{key}.feed_forward.{proj}.weight"] = linear_weight(f"{node}/feed_forward/{proj}/MatMul")
        # The exporter keeps one rope.inv_freq and aliases the rest; it is a constant of embed_dim.
        inv_freq = f"{key}.rope.inv_freq"
        state.setdefault(inv_freq, own[inv_freq])
        if not np.allclose(state[inv_freq], own[inv_freq], rtol=1e-6, atol=0):
            raise ValueError(f"{inv_freq} in the ONNX does not match embed_dim={EMBED_DIM}")

    unused = sorted(set(inits) - used)
    if unused:
        raise ValueError(f"ONNX initializers not mapped to actor parameters: {unused}")
    actor.load_state_dict({k: torch.from_numpy(np.array(v, dtype=np.float32)) for k, v in state.items()}, strict=True)
    num_parameters = sum(p.numel() for p in actor.parameters())
    if num_parameters != NUM_PARAMETERS:
        raise ValueError(f"Rebuilt actor has {num_parameters} parameters, expected {NUM_PARAMETERS}")
    return actor.eval()


# ---------------------------------------------------------------------------
# sim2real layout -> Scalecar layout
# ---------------------------------------------------------------------------


def policy_layout(
    history_steps: Sequence[int], sigma: Sequence[int], default_joint_pos: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather/offset/scale tables turning the flat ``policy`` group into Scalecar's (3, 64).

    The group concatenates projected_gravity_history, root_ang_vel_history,
    joint_pos_history and joint_vel_history; each emits buffer[history_steps]
    (buffer[0] newest) flattened time-major, joints in joint_names_simulation
    order. Scalecar row r holds lag 2 - r as [gravity, ang_vel, q - default,
    0.05 * dq] in Tracker joint order; sigma[j] is Tracker joint j's position in
    joint_names_simulation.
    """
    history_steps = list(history_steps)
    if sorted(history_steps) != list(range(CONTEXT)):
        raise ValueError(f"history_steps must be a permutation of {list(range(CONTEXT))}")
    h = len(history_steps)
    sigma = np.asarray(sigma, dtype=np.int64)
    index = np.zeros((CONTEXT, PROP_DIM), dtype=np.int64)
    offset = np.zeros((CONTEXT, PROP_DIM), dtype=np.float32)
    scale = np.ones((CONTEXT, PROP_DIM), dtype=np.float32)
    gravity, ang_vel, joint_pos, joint_vel = 0, 3 * h, 6 * h, 6 * h + NUM_JOINTS * h
    for r in range(CONTEXT):
        i = history_steps.index(CONTEXT - 1 - r)
        index[r, 0:3] = gravity + 3 * i + np.arange(3)
        index[r, 3:6] = ang_vel + 3 * i + np.arange(3)
        index[r, 6:35] = joint_pos + NUM_JOINTS * i + sigma
        index[r, 35:64] = joint_vel + NUM_JOINTS * i + sigma
        offset[r, 6:35] = default_joint_pos
        scale[r, 35:64] = DQ_SCALE
    return index, offset, scale


def action_layout(steps: int) -> np.ndarray:
    """Gather table turning ``prev_actions(steps)`` (newest first) into Scalecar's (3, 29), oldest first."""
    if steps < CONTEXT - 1:
        raise ValueError(f"prev_actions needs at least {CONTEXT - 1} steps")
    index = np.zeros((CONTEXT, NUM_JOINTS), dtype=np.int64)
    for r in range(CONTEXT):
        lag = min(CONTEXT - 1 - r, steps - 1)  # row 0 never reaches the output
        index[r] = NUM_JOINTS * lag + np.arange(NUM_JOINTS)
    return index


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
            a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
            a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
        ],
        dim=-1,
    )


class Sim2RealToScalecar(nn.Module):
    """Maps the flat sim2real inputs to Scalecar's (1,3,64), (1,6,69), (1,3,29).

    Reference Tokens (reference_only_v1, one row per offset) are assembled from
    sonic_command_multi_future_nonflat ([q; dq] per offset, Tracker order) and
    ref_body_pos/ori_future_local with body = anchor = pelvis, i.e. pelvis
    position and rotation-matrix rows 0-1 in the frame of the offset-0 pelvis
    position and heading (position z stays absolute). A token is
    [q, dq, dx, dy, z - z_0, z, R'[:, 0], R'[:, 2], offset].
    """

    def __init__(self, policy_index, policy_offset, policy_scale, action_index, reference_offsets):
        super().__init__()
        self.register_buffer("policy_index", torch.as_tensor(policy_index))
        self.register_buffer("policy_offset", torch.as_tensor(policy_offset))
        self.register_buffer("policy_scale", torch.as_tensor(policy_scale))
        self.register_buffer("action_index", torch.as_tensor(action_index))
        self.register_buffer(
            "reference_offsets", torch.as_tensor(reference_offsets, dtype=torch.float32).reshape(-1, 1)
        )
        self.num_frames = len(reference_offsets)

    def reference_tokens(self, ref_command, ref_pelvis_pos, ref_pelvis_ori) -> torch.Tensor:
        n = self.num_frames
        q = ref_command[: n * NUM_JOINTS].reshape(n, NUM_JOINTS)
        dq = ref_command[n * NUM_JOINTS :].reshape(n, NUM_JOINTS)
        pos = ref_pelvis_pos.reshape(n, 3)
        rows = ref_pelvis_ori.reshape(n, 2, 3)
        r0, r1 = rows[:, 0], rows[:, 1]
        r2 = _cross(r0, r1)
        displacement = torch.cat([pos[:, :2], pos[:, 2:3] - pos[0:1, 2:3]], dim=-1)
        column0 = torch.stack([r0[:, 0], r1[:, 0], r2[:, 0]], dim=-1)
        column2 = torch.stack([r0[:, 2], r1[:, 2], r2[:, 2]], dim=-1)
        return torch.cat([q, dq, displacement, pos[:, 2:3], column0, column2, self.reference_offsets], dim=-1)

    def forward(self, policy, prev_actions, ref_command, ref_pelvis_pos, ref_pelvis_ori):
        prop = (policy[self.policy_index] - self.policy_offset) * self.policy_scale
        action = prev_actions[self.action_index]
        tokens = self.reference_tokens(ref_command, ref_pelvis_pos, ref_pelvis_ori)
        return prop.unsqueeze(0), tokens.unsqueeze(0), action.unsqueeze(0)


class Sim2RealTracker(nn.Module):
    def __init__(self, layout: Sim2RealToScalecar, actor: ScalecarActor):
        super().__init__()
        self.layout = layout
        self.actor = actor

    def forward(self, policy, prev_actions, ref_command, ref_pelvis_pos, ref_pelvis_ori):
        prop, tokens, action = self.layout(policy, prev_actions, ref_command, ref_pelvis_pos, ref_pelvis_ori)
        return self.actor(prop, tokens, action)[0]


def input_sizes(num_frames: int) -> dict[str, int]:
    return {
        "policy": len(HISTORY_STEPS) * PROP_DIM,
        "prev_actions": PREV_ACTION_STEPS * NUM_JOINTS,
        "ref_command": 2 * num_frames * NUM_JOINTS,
        "ref_pelvis_pos": num_frames * 3,
        "ref_pelvis_ori": num_frames * 6,
    }


# ---------------------------------------------------------------------------
# Policy YAML
# ---------------------------------------------------------------------------


class _PolicyYamlDumper(yaml.SafeDumper):
    """No anchors/aliases; scalar lists inline, mappings as blocks."""

    def ignore_aliases(self, data):
        return True


_PolicyYamlDumper.add_representer(
    list, lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)
)


def policy_yaml(deployment: dict[str, Any]) -> dict[str, Any]:
    robot_cfg = get_robot_cfg("g1")
    names = list(deployment["joint_names"])
    offsets = [int(o) for o in deployment["reference_offsets"]]
    history = {"history_steps": HISTORY_STEPS}
    pelvis = {"future_steps": offsets, "body_names": ["pelvis"], "anchor_body_name": "pelvis"}

    def per_joint(values):
        return {name: float(v) for name, v in zip(names, values, strict=True)}

    return {
        "model_path": "policy.onnx",
        # Group names are the ONNX input names; group order inside "policy" fixes its layout.
        "observation": {
            "policy": {
                "projected_gravity_history": {"_target_": "mimic_lite.projected_gravity_history", **history},
                "root_ang_vel_history": {"_target_": "mimic_lite.root_ang_vel_history", **history},
                "joint_pos_history": {"_target_": "mimic_lite.joint_pos_history", **history},
                "joint_vel_history": {"_target_": "mimic_lite.joint_vel_history", **history},
            },
            "prev_actions": {
                "prev_actions": {"_target_": "mimic_lite.prev_actions", "steps": PREV_ACTION_STEPS},
            },
            "ref_command": {
                "sonic_command_multi_future_nonflat": {
                    "_target_": "sonic.sonic_command_multi_future_nonflat",
                    "future_steps": offsets,
                    "joint_names": names,
                },
            },
            "ref_pelvis_pos": {
                "ref_body_pos_future_local": {"_target_": "mimic_lite.ref_body_pos_future_local", **pelvis},
            },
            "ref_pelvis_ori": {
                "ref_body_ori_future_local": {"_target_": "mimic_lite.ref_body_ori_future_local", **pelvis},
            },
        },
        "joint_names_simulation": list(robot_cfg.joint_names),
        "body_names_simulation": list(robot_cfg.body_names),
        "policy_joint_names": names,
        "default_joint_pos": per_joint(deployment["default_joint_pos"]),
        "joint_kp": per_joint(deployment["stiffness"]),
        "joint_kd": per_joint(deployment["damping"]),
        "action_scale": per_joint(deployment["action_scale"]),
        "motion": {
            "motion_backend": "npz",
            "future_steps": offsets,
            "joint_names": names,
            "body_names": ["pelvis"],
            "root_body_name": "pelvis",
            "motion_dt_s": float(deployment["control_dt"]),
        },
    }


def layout_from_policy_yaml(cfg: dict[str, Any], deployment: dict[str, Any]) -> Sim2RealToScalecar:
    """Build the in-graph layout from the YAML it will run with."""
    obs = cfg["observation"]
    history_steps = obs["policy"]["joint_pos_history"]["history_steps"]
    if any(item.get("history_steps") != history_steps for item in obs["policy"].values()):
        raise ValueError("All policy history helpers must share history_steps")
    future_steps = obs["ref_command"]["sonic_command_multi_future_nonflat"]["future_steps"]
    if future_steps[0] != 0 or future_steps != list(deployment["reference_offsets"]):
        raise ValueError("Reference future_steps must equal the Tracker offsets and start at 0")
    sim_names = cfg["joint_names_simulation"]
    sigma = [sim_names.index(name) for name in deployment["joint_names"]]
    return Sim2RealToScalecar(
        *policy_layout(history_steps, sigma, deployment["default_joint_pos"]),
        action_layout(obs["prev_actions"]["prev_actions"]["steps"]),
        future_steps,
    )


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(source_dir: Path, variant: str) -> dict[str, dict[str, str]]:
    sources = {}
    for name, expected in PINNED_SOURCES[variant].items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing Scalecar input {path}; download the ModelScope repo Hzzzz001/scalecar_v2")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{path}: sha256 {actual} does not match pinned {expected}")
        sources[name] = {"path": str(path), "sha256": actual}
    return sources


def _dims(value_info) -> list:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def check_artifact(model: onnx.ModelProto, expected_inputs: dict[str, int]) -> dict[str, Any]:
    onnx.checker.check_model(model, full_check=True)
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    inputs = {i.name: _dims(i) for i in model.graph.input}
    outputs = {o.name: _dims(o) for o in model.graph.output}
    problems = []
    if model.ir_version > 9 or opset > 19:
        problems.append(f"IR {model.ir_version} / opset {opset} exceeds onnxruntime 1.16 (IR 9 / opset 19)")
    if inputs != {name: [size] for name, size in expected_inputs.items()}:
        problems.append(f"inputs {inputs} != {expected_inputs}")
    if outputs != {"action": [NUM_JOINTS]}:
        problems.append(f"outputs {outputs} != action[{NUM_JOINTS}]")
    if problems:
        raise RuntimeError("Exported ONNX violates the sim2real contract: " + "; ".join(problems))
    return {"ir_version": model.ir_version, "opset": opset, "inputs": inputs, "outputs": outputs}


def _random_rotation_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    w, x, y, z = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        ],
        axis=1,
    )


def random_sim2real_inputs(rng: np.random.Generator, num_frames: int) -> dict[str, np.ndarray]:
    h = len(HISTORY_STEPS)
    gravity = rng.normal(size=(h, 3)) * [0.2, 0.2, 1.0]
    gravity /= np.linalg.norm(gravity, axis=-1, keepdims=True)
    pelvis_pos = rng.normal(size=(num_frames, 3)) * 0.5
    pelvis_pos[:, 2] = rng.uniform(0.3, 1.0, num_frames)
    inputs = {
        "policy": np.concatenate(
            [
                gravity.reshape(-1),
                rng.normal(size=3 * h),
                rng.normal(size=NUM_JOINTS * h) * 0.5,
                rng.normal(size=NUM_JOINTS * h) * 2.0,
            ]
        ),
        "prev_actions": rng.normal(size=PREV_ACTION_STEPS * NUM_JOINTS),
        "ref_command": rng.normal(size=2 * num_frames * NUM_JOINTS) * 0.5,
        "ref_pelvis_pos": pelvis_pos.reshape(-1),
        "ref_pelvis_ori": _random_rotation_rows(rng, num_frames).reshape(-1),
    }
    return {k: v.astype(np.float32) for k, v in inputs.items()}


def compare_with_original(
    exported_path: Path, original_path: Path, layout: Sim2RealToScalecar, num_samples: int, seed: int
) -> float:
    """Max |exported - original| over random sim2real inputs mapped to Scalecar layout."""
    exported = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
    original = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(num_samples):
        inputs = random_sim2real_inputs(rng, layout.num_frames)
        with torch.inference_mode():
            prop, tokens, action = layout(*(torch.from_numpy(inputs[name]) for name in INPUT_NAMES))
        expected = original.run(
            None, {"policy": prop.numpy(), "policy_task": tokens.numpy(), "action": action.numpy()}
        )[0][0]
        actual = exported.run(None, inputs)[0]
        max_err = max(max_err, float(np.abs(actual - expected).max()))
    return max_err


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_tracker(
    variant: str,
    *,
    modelscope_root: Path = DEFAULT_MODELSCOPE_ROOT,
    output_dir: Path | None = None,
    num_samples: int = NUM_SAMPLES,
    seed: int = 0,
) -> Path:
    """Export one Tracker; returns the output directory."""
    if variant not in PINNED_SOURCES:
        raise ValueError(f"Unknown Tracker {variant!r}; choose from {sorted(PINNED_SOURCES)}")
    if torch.__version__.split("+")[0] != REQUIRED_TORCH:
        raise RuntimeError(
            f"Export needs torch {REQUIRED_TORCH} (found {torch.__version__}); run with "
            "`uv run --with torch==2.7.0 --index https://download.pytorch.org/whl/cpu ...`"
        )
    source_dir = Path(modelscope_root) / TRACKERS_SUBDIR / variant
    sources = verify_sources(source_dir, variant)
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / variant

    deployment = json.loads((source_dir / "deployment.json").read_text(encoding="utf-8"))
    if deployment.get("input_interface") != "reference_only_v1":
        raise ValueError(f"Unsupported Tracker input interface {deployment.get('input_interface')!r}")
    original_path = source_dir / "actor.onnx"
    actor = rebuild_actor(onnx.load(str(original_path)))
    cfg = policy_yaml(deployment)
    layout = layout_from_policy_yaml(cfg, deployment).eval()
    wrapper = Sim2RealTracker(layout, actor).eval()
    sizes = input_sizes(layout.num_frames)

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / cfg["model_path"]
    with torch.inference_mode(), warnings.catch_warnings():
        # RoPE's stride-2 slices cannot be folded; harmless.
        warnings.filterwarnings("ignore", message="Constant folding - Only steps=1")
        torch.onnx.export(
            wrapper,
            tuple(torch.zeros(sizes[name]) for name in INPUT_NAMES),
            str(onnx_path),
            input_names=INPUT_NAMES,
            output_names=["action"],
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    artifact = check_artifact(onnx.load(str(onnx_path)), sizes)
    artifact["sha256"] = sha256_file(onnx_path)

    header = (
        f"# Scalecar d1146 Tracker {variant}, generated by scripts/export_scalecar_tracker_onnx.py.\n"
        "# Regenerate instead of editing; comparison.json records the check against the original ONNX.\n"
    )
    (output_dir / "policy.yaml").write_text(
        header + yaml.dump(cfg, Dumper=_PolicyYamlDumper, sort_keys=False, width=100),
        encoding="utf-8",
    )

    max_err = compare_with_original(onnx_path, original_path, layout, num_samples, seed)
    comparison = {
        "variant": variant,
        "passed": max_err <= TOLERANCE,
        "max_abs_error": max_err,
        "tolerance": TOLERANCE,
        "num_samples": num_samples,
        "seed": seed,
        "method": (
            "random inputs in sim2real layout; the original ONNX gets them mapped to "
            "Scalecar layout (policy[1,3,64], policy_task[1,6,69], action[1,3,29])"
        ),
        "sources": sources,
        "artifact": {"policy.onnx": artifact},
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    if not comparison["passed"]:
        raise RuntimeError(f"{variant}: max |exported - original| = {max_err:.3g} > {TOLERANCE}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", nargs="+", choices=sorted(PINNED_SOURCES), default=["ft500ema"])
    parser.add_argument("--modelscope-root", type=Path, default=DEFAULT_MODELSCOPE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Each Tracker is written to <output-root>/<variant>/",
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for variant in args.variant:
        output_dir = export_tracker(
            variant,
            modelscope_root=args.modelscope_root,
            output_dir=args.output_root / variant,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
        print(
            f"{variant}: {output_dir} max|d|={comparison['max_abs_error']:.3g} "
            f"over {comparison['num_samples']} samples"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
