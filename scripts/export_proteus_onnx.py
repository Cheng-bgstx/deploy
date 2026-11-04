#!/usr/bin/env python3
"""Specialize the Proteus multi-robot ONNX actor to one robot for sim2real.

Proteus (``ScaleTrack/checkpoints/proteus_60`` in the Proteus repo) ships one ONNX actor for 60 robots and one
contract ``<robot>.json`` per robot. The actor is batched and robot-agnostic: a morphology input and 31 canonical
joint slots. sim2real's BasePolicy feeds unbatched observation groups and reads ``action`` in
``policy_joint_names`` order, so the released graph is wrapped, not re-exported:

- inputs ``proprioception[H, 6+2S]``, ``task[F, 18B+1]`` and ``action_history[H, S]``, the observation groups of
  the generated ``policy.yaml`` (``sim2real/rl_policy/observations/proteus.py``), unsqueezed into the actor's
  ``policy``, ``policy_task`` and ``action``;
- the contract's morphology descriptor as the constant ``morph``;
- outputs ``action[J]``, the slot actions of the robot's driven joints in ``policy_joint_names`` order, and
  ``next_slot_action[S]``, every slot action (the action history holds the unmasked actor output, as IsaacLab's
  ``last_action`` does in training).

The joint target is ``default + action_scale * action`` with the contract's signed ``action_scale``, as in Proteus.
The wrapper keeps the actor's ONNX IR and opset (ORT 1.16 on the G1 loads them) and is compared with the released
actor on random inputs. Output directory: ``policy.onnx``, ``policy.yaml``, ``comparison.json`` and a verbatim
copy of the contract.

For ``actor_kind=wam_seq``, ``seq_window`` replaces the legacy three-frame
history. The wrapper also forwards ``valid_history[W]`` to ``valid`` and maps
the observation/action windows to ``obs_window`` / ``prev_actions``. Use
``--checkpoint-dir`` to select a WAM release; the original export stays available.

    uv run python scripts/export_proteus_onnx.py --proteus-root ../Proteus --contract unitree_g1 --robot g1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnx import TensorProto, helper, numpy_helper

from sim2real.config.robots import get_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTEUS_ROOT = REPO_ROOT.parent / "Proteus"
EXPORT_SUBDIR = Path("ScaleTrack") / "checkpoints" / "proteus_60"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "checkpoints" / "proteus" / "proteus_60"
DEFAULT_FAR_OFFSET = 16  # the far reference frame of deployment (ScaleBridge's proteus.yaml)
NUM_SAMPLES = 64
TOLERANCE = 1.0e-6

# released actor I/O -> wrapper I/O
SOURCE_INPUTS = {"policy": "proprioception", "policy_task": "task", "action": "action_history"}
WAM_SOURCE_INPUTS = {"obs_window": "proprioception", "policy_task": "task",
                     "prev_actions": "action_history", "valid": "valid_history"}
SOURCE_MORPH = "morph"
SOURCE_OUTPUT = "actions"
INTERNAL_PREFIX = "proteus_actor/"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if contract["control_dt"] != 0.02:
        raise ValueError(f"Proteus control_dt {contract['control_dt']} != 0.02 (sim2real runs the actor at 50 Hz)")
    offsets = list(contract["future_offsets"])
    if offsets[-1] != -1 or any(o < 0 for o in offsets[:-1]):
        raise ValueError(f"Expected near offsets and a far placeholder -1, got future_offsets={offsets}")
    return contract


def resolved_offsets(contract: dict[str, Any], far_offset: int) -> list[int]:
    lo, hi = contract["far_offset_range"]
    if not lo <= far_offset <= hi:
        raise ValueError(f"far_offset {far_offset} outside the trained range [{lo}, {hi}]")
    return [int(o) for o in contract["future_offsets"][:-1]] + [int(far_offset)]


def driven_joints(contract: dict[str, Any]) -> list[dict[str, Any]]:
    """The joints the actor drives, in contract order (policy_joint_names)."""
    return [j for j in contract["joints"] if j["slot"] >= 0]


def input_shapes(contract: dict[str, Any], far_offset: int) -> dict[str, tuple[int, ...]]:
    seq = contract.get("actor_kind", "separate") == "wam_seq"
    H = int(contract["seq_window"] if seq else contract["history"])
    S = int(contract["num_slots"])
    F, B = len(resolved_offsets(contract, far_offset)), len(contract["body_names"])
    shapes = {"proprioception": (H, 6 + 2 * S), "task": (F, 18 * B + 1), "action_history": (H, S)}
    if seq:
        shapes["valid_history"] = (H,)
    return shapes


def source_inputs(contract: dict[str, Any]) -> dict[str, str]:
    kind = contract.get("actor_kind", "separate")
    if kind not in {"separate", "wam_seq"}:
        raise ValueError(f"Unsupported Proteus actor_kind: {kind}")
    return WAM_SOURCE_INPUTS if kind == "wam_seq" else SOURCE_INPUTS


# ---------------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------------


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    return [d.dim_param or d.dim_value for d in value_info.type.tensor_type.shape.dim]


def wrap_actor(source: onnx.ModelProto, contract: dict[str, Any], far_offset: int) -> onnx.ModelProto:
    model = onnx.ModelProto()
    model.CopyFrom(source)
    graph = model.graph
    S = int(contract["num_slots"])
    shapes = input_shapes(contract, far_offset)
    mapping = source_inputs(contract)
    morph = np.asarray(contract["morph"], dtype=np.float32)[None]

    inputs = {i.name: i for i in graph.input}
    expected = {old: shapes[new] for old, new in mapping.items()} | {SOURCE_MORPH: morph.shape[1:]}
    if set(inputs) != set(expected):
        raise ValueError(f"Proteus actor inputs {set(inputs)} != {set(expected)}")
    for name, shape in expected.items():
        if name not in inputs or tuple(_shape(inputs[name])[1:]) != tuple(shape):
            raise ValueError(f"Proteus actor input {name} != [batch, {shape}]: {[(i.name, _shape(i)) for i in graph.input]}")
    if [o.name for o in graph.output] != [SOURCE_OUTPUT] or _shape(graph.output[0])[1:] != [S]:
        raise ValueError(f"Proteus actor output != {SOURCE_OUTPUT}[batch, {S}]")
    tensor_names = {t for n in graph.node for t in (*n.input, *n.output)} | {i.name for i in graph.initializer}
    clashes = sorted(tensor_names & {*mapping.values(), "action", "next_slot_action"} - set(mapping))
    if clashes or any(n.op_type in {"If", "Loop", "Scan"} for n in graph.node):
        raise ValueError(f"Cannot rename the actor's inputs safely (clashes: {clashes})")

    # the released inputs become internal tensors
    rename = {name: INTERNAL_PREFIX + name for name in (*mapping, SOURCE_MORPH)}
    for node in graph.node:
        node.input[:] = [rename.get(name, name) for name in node.input]
    for value_info in graph.value_info:
        value_info.name = rename.get(value_info.name, value_info.name)
    del graph.input[:]

    slots = np.asarray([j["slot"] for j in driven_joints(contract)], dtype=np.int64)
    graph.initializer.extend([
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), INTERNAL_PREFIX + "batch_axis"),
        numpy_helper.from_array(morph, rename[SOURCE_MORPH]),
        numpy_helper.from_array(slots, INTERNAL_PREFIX + "driven_slots"),
    ])  # fmt: skip
    unsqueeze = [
        helper.make_node("Unsqueeze", [new, INTERNAL_PREFIX + "batch_axis"], [rename[old]], name=f"unsqueeze_{new}")
        for old, new in mapping.items()
    ]
    nodes = unsqueeze + list(graph.node) + [
        helper.make_node("Squeeze", [SOURCE_OUTPUT, INTERNAL_PREFIX + "batch_axis"], ["next_slot_action"],
                         name="squeeze_actions"),
        helper.make_node("Gather", ["next_slot_action", INTERNAL_PREFIX + "driven_slots"], ["action"], axis=0,
                         name="gather_driven_slots"),
    ]  # fmt: skip
    del graph.node[:]
    graph.node.extend(nodes)
    graph.input.extend(helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name]) for name in mapping.values())
    del graph.output[:]
    graph.output.extend([
        helper.make_tensor_value_info("action", TensorProto.FLOAT, [len(slots)]),
        helper.make_tensor_value_info("next_slot_action", TensorProto.FLOAT, [S]),
    ])  # fmt: skip
    graph.name = f"proteus_{contract['robot']}"
    model.doc_string = (
        f"Proteus actor for {contract['robot']}, generated by sim2real scripts/export_proteus_onnx.py "
        f"(far_offset={far_offset})"
    )
    onnx.checker.check_model(model, full_check=True)
    return onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


def compare_with_source(
    wrapped: onnx.ModelProto, source_path: Path, contract: dict[str, Any], far_offset: int, num_samples: int, seed: int
) -> float:
    """Max |wrapper - released actor| over random inputs (the released actor gets the batch axis and the morph)."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    wrapper = ort.InferenceSession(wrapped.SerializeToString(), options, providers=["CPUExecutionProvider"])
    source = ort.InferenceSession(str(source_path), options, providers=["CPUExecutionProvider"])
    slots = [j["slot"] for j in driven_joints(contract)]
    morph = np.asarray(contract["morph"], dtype=np.float32)[None]
    rng = np.random.default_rng(seed)
    max_err = 0.0
    for sample in range(num_samples):
        feeds = {name: rng.normal(size=shape).astype(np.float32) for name, shape in input_shapes(contract, far_offset).items()}
        if "valid_history" in feeds:
            # Cover startup, partially populated episodes and full windows.
            count = sample % (len(feeds["valid_history"]) + 1)
            valid = (np.arange(len(feeds["valid_history"])) >= len(feeds["valid_history"]) - count)
            feeds["valid_history"] = valid.astype(np.float32)
            feeds["proprioception"][~valid] = 0.0
            prev_valid = np.concatenate([[False], valid[:-1]])
            feeds["action_history"][~prev_valid] = 0.0
        action, next_slot_action = wrapper.run(["action", "next_slot_action"], feeds)
        source_feeds = {old: feeds[new][None] for old, new in source_inputs(contract).items()} | {SOURCE_MORPH: morph}
        (actions,) = source.run([SOURCE_OUTPUT], source_feeds)
        if not all(np.isfinite(value).all() for value in (actions, action, next_slot_action)):
            raise RuntimeError("Proteus export comparison produced non-finite actions")
        max_err = max(max_err, float(np.abs(next_slot_action - actions[0]).max()),
                      float(np.abs(action - actions[0, slots]).max()))  # fmt: skip
    return max_err


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


def policy_yaml(contract: dict[str, Any], robot: str, far_offset: int) -> dict[str, Any]:
    robot_cfg = get_robot_cfg(robot)
    joints = contract["joints"]
    names = [j["name"] for j in joints]
    missing = [name for name in names if name not in robot_cfg.joint_names]
    if missing:
        raise ValueError(f"sim2real robot {robot!r} lacks the {contract['robot']} contract joints {missing}")
    offsets = resolved_offsets(contract, far_offset)

    def term(name: str) -> dict[str, str]:
        return {"_target_": f"proteus.proteus_{name}"}

    cfg = {
        "model_path": "policy.onnx",
        "observation": {
            "proprioception": {
                "projected_gravity": term("projected_gravity_history"),
                "root_ang_vel": term("root_ang_vel_history"),
                "slot_joint_pos": term("slot_joint_pos_history"),
                "slot_joint_vel": term("slot_joint_vel_history"),
            },
            "task": {
                "target_body_pos": term("target_body_pos"),
                "target_body_pos_rel": term("target_body_pos_rel"),
                "target_body_rot": term("target_body_rot"),
                "target_body_rot_rel": term("target_body_rot_rel"),
                "frame_offsets": term("frame_offsets"),
            },
            "action_history": {"slot_actions": term("slot_action_history")},
        },
        "joint_names_simulation": list(robot_cfg.joint_names),
        "body_names_simulation": list(robot_cfg.body_names),
        "policy_joint_names": [j["name"] for j in driven_joints(contract)],
        "default_joint_pos": {j["name"]: float(j["default"]) for j in joints},
        "joint_kp": {j["name"]: float(j["kp"]) for j in joints},
        "joint_kd": {j["name"]: float(j["kd"]) for j in joints},
        "action_scale": {j["name"]: float(j["action_scale"]) for j in driven_joints(contract)},
        "proteus": {
            "robot": contract["robot"],
            "contract": f"{contract['robot']}.json",
            "num_slots": int(contract["num_slots"]),
            "history": int(contract["history"]),
            "joint_vel_scale": float(contract["joint_vel_scale"]),
            "length_scale": float(contract["length_scale"]),
            "body_names": list(contract["body_names"]),
            "future_offsets": offsets,
            # Local tracking: the real G1 has no root position, the reference anchor stands in for it.
            "reference_forcing": True,
            "joint_slots": {j["name"]: int(j["slot"]) for j in joints},
            "joint_signs": {j["name"]: float(j["sign"]) for j in joints},
        },
        "motion": {
            "motion_backend": "npz",
            "future_steps": offsets,
            "root_body_name": contract["body_names"][0],
            "motion_dt_s": float(contract["control_dt"]),
        },
    }
    if contract.get("actor_kind", "separate") == "wam_seq":
        cfg["proteus"].update(actor_kind="wam_seq", seq_window=int(contract["seq_window"]))
        cfg["observation"]["valid_history"] = {"valid": term("valid_history")}
    return cfg


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def export_policy(
    contract_name: str,
    *,
    proteus_root: Path = DEFAULT_PROTEUS_ROOT,
    robot: str = "g1",
    far_offset: int = DEFAULT_FAR_OFFSET,
    output_dir: Path | None = None,
    num_samples: int = NUM_SAMPLES,
    seed: int = 0,
    checkpoint_dir: Path | None = None,
) -> Path:
    """Export one robot's policy; returns the output directory."""
    export_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path(proteus_root) / EXPORT_SUBDIR
    contract_path = export_dir / f"{contract_name}.json"
    contract_bytes = contract_path.read_bytes()
    contract = load_contract(contract_path)
    source_path = export_dir / contract["actor"]
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_ROOT / contract_name

    source = onnx.load(str(source_path))
    wrapped = wrap_actor(source, contract, far_offset)
    max_err = compare_with_source(wrapped, source_path, contract, far_offset, num_samples, seed)
    if max_err > TOLERANCE:
        raise RuntimeError(f"{contract_name}: max |wrapper - released actor| = {max_err:.3g} > {TOLERANCE}")

    cfg = policy_yaml(contract, robot, far_offset)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / cfg["model_path"]
    onnx.save(wrapped, str(onnx_path))
    header = (
        f"# Proteus 60-robot actor for {contract_name} (sim2real robot {robot}), generated by\n"
        "# scripts/export_proteus_onnx.py. Regenerate instead of editing; comparison.json records the check\n"
        "# against the released actor. proteus.reference_forcing: false tracks the measured root position.\n"
    )
    (output_dir / "policy.yaml").write_text(
        header + yaml.dump(cfg, Dumper=_PolicyYamlDumper, sort_keys=False, width=110),
        encoding="utf-8",
    )
    (output_dir / cfg["proteus"]["contract"]).write_bytes(contract_bytes)
    comparison = {
        "contract": contract_name,
        "robot": robot,
        "actor_kind": contract.get("actor_kind", "separate"),
        "far_offset": far_offset,
        "passed": True,
        "max_abs_error": max_err,
        "tolerance": TOLERANCE,
        "num_samples": num_samples,
        "seed": seed,
        "method": (
            "standard-normal inputs in sim2real layout; the released actor gets them with a batch axis and the "
            "contract's morph; WAM includes binary startup/partial/full validity masks and zero episode padding; "
            "action is compared with actions[driven slots], next_slot_action with actions"
        ),
        "sources": {
            contract["actor"]: sha256_file(source_path),
            contract_path.name: hashlib.sha256(contract_bytes).hexdigest(),
        },
        "artifact": {
            "policy.onnx": {
                "sha256": sha256_file(onnx_path),
                "ir_version": int(wrapped.ir_version),
                "opset": {o.domain or "ai.onnx": int(o.version) for o in wrapped.opset_import},
                "inputs": {i.name: _shape(i) for i in wrapped.graph.input},
                "outputs": {o.name: _shape(o) for o in wrapped.graph.output},
            }
        },
        "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--proteus-root", type=Path, default=DEFAULT_PROTEUS_ROOT)
    parser.add_argument("--checkpoint-dir", type=Path, help="Released actor and contracts directory (e.g. proteus_wamseq_62)")
    parser.add_argument("--contract", nargs="+", default=["unitree_g1"], help="contract names in proteus_60/")
    parser.add_argument("--robot", default="g1", help="sim2real robot config of the contracts")
    parser.add_argument("--far-offset", type=int, default=DEFAULT_FAR_OFFSET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Each policy is written to <output-root>/<contract>/",
    )
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for name in args.contract:
        output_dir = export_policy(
            name,
            proteus_root=args.proteus_root,
            robot=args.robot,
            far_offset=args.far_offset,
            output_dir=args.output_root / name,
            num_samples=args.num_samples,
            seed=args.seed,
            checkpoint_dir=args.checkpoint_dir,
        )
        comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
        print(f"{name}: {output_dir} max|d|={comparison['max_abs_error']:.3g} over {comparison['num_samples']} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
