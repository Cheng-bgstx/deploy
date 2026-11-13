#!/usr/bin/env python3
"""Build a Proteus contract for a robot outside the released 60, from its MJCF and a released robot.

Proteus (``ScaleTrack/checkpoints/proteus_60`` in the Proteus repo) conditions one actor on a per-robot contract:
joint slots and signs, default pose, PD gains, efforts, tracked bodies, length scale and a 722-dimensional
morphology descriptor. Proteus builds a robot's description from its MJCF and retargeted LAFAN1 walking clips
(``ScaleTrack/scripts/multi_robot/build_robot_spec.py``) and the descriptor from that description
(``supermimic/morphology.py``). This script runs the same code on a sim2real robot, with a released robot of the
same kinematics (``--template``) standing in for the walking clips:

- the reference pose is the template's default pose, clipped inside the robot's joint limits (2 % margin);
- the standing hip height ``h`` is the template's (``--stand-height`` overrides it); the forward-kinematics hip
  height of the reference pose is printed next to it;
- slots, signs, tracked bodies and actuators come from Proteus's own functions on the robot's MJCF: efforts from
  its actuators, armatures from the sim2real robot config (``joint_armature``), stiffness, damping and velocity
  limits the G1 values of the slot rescaled to the robot's mass and ``h`` (as for every released robot);
- every slot and every joint axis in the reference pose must match the template's, so that the template's pose and
  ``h`` apply (``--allow-kinematic-mismatch`` turns the axis check into warnings);
- ``--lock`` joints get no slot and are held at their default with ``--locked-gains`` (Proteus's locked joints).

The Galbot ET1 is a Unitree R1 with a heavier body and narrower waist, arm and head limits; et1-dep's Loco holds
its head with fixed gains, which the default ``--lock`` and ``--locked-gains`` reproduce:

    uv run python scripts/build_proteus_contract.py --robot et1 --name galbot_et1 --template unitree_r1

Output: ``<output-dir>/<name>.json`` (read by ``scripts/export_proteus_onnx.py --contract-dir``) and the
robot description ``<name>.yaml`` for review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import yaml

from sim2real.config.robots import get_robot_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTEUS_ROOT = REPO_ROOT.parent / "Proteus"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "checkpoints" / "proteus" / "contracts"
SCALETRACK = Path("ScaleTrack")
EXPORT_SUBDIR = SCALETRACK / "checkpoints" / "proteus_60"
SPEC_PACKAGE = SCALETRACK / "source" / "scaletrack" / "scaletrack" / "robots" / "spec"
SUPERMIMIC_PACKAGE = SCALETRACK / "source" / "scaletrack" / "scaletrack" / "tasks" / "tracking" / "supermimic"
SPEC_BUILDER = SCALETRACK / "scripts" / "multi_robot" / "build_robot_spec.py"
LIMIT_MARGIN = 0.02  # build_robot_spec.reference_pose: defaults strictly inside the limits
AXIS_TOLERANCE = 1.0e-3
HEIGHT_TOLERANCE = 0.05  # gross-error guard: the FK pose is not a walking median
DEFAULT_LOCK = ("head_pitch_joint", "head_yaw_joint")
DEFAULT_LOCKED_GAINS = (15.706609961074328, 0.999913845837132)  # et1-dep Loco's uncovered_joint_gains
# Contract fields that hold for every robot of the release, copied from the template's contract.
SHARED_FIELDS = ("control_dt", "physics_dt", "num_slots", "ground_friction", "joint_vel_scale", "history",
                 "future_offsets", "far_offset_range", "actor")  # fmt: skip


class Proteus:
    """The Proteus modules this script needs, loaded without importing scaletrack (which needs IsaacLab)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.spec = self._package("proteus_robot_spec", self.root / SPEC_PACKAGE)
        self._package("proteus_supermimic", self.root / SUPERMIMIC_PACKAGE)
        from proteus_supermimic.kinematics import KinematicTree
        from proteus_supermimic.morphology import flat_descriptor

        self.KinematicTree = KinematicTree
        self.flat_descriptor = flat_descriptor
        self.builder = self._module("proteus_build_robot_spec", self.root / SPEC_BUILDER)

    @staticmethod
    def _package(name: str, path: Path) -> ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _module(name: str, path: Path) -> ModuleType:
        if name in sys.modules:
            return sys.modules[name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def description(self, name: str) -> dict[str, Any]:
        return yaml.safe_load((self.spec.spec.CONFIG_DIR / f"{name}.yaml").read_text(encoding="utf-8"))

    def contract(self, name: str) -> dict[str, Any]:
        return json.loads((self.root / EXPORT_SUBDIR / f"{name}.json").read_text(encoding="utf-8"))


def reference_pose(robot, template: dict[str, Any], lock: set[str], overrides: dict[str, float]) -> np.ndarray:
    """Template defaults by joint name, clipped inside the robot's limits; locked joints at 0."""
    lo, hi = robot.model.jnt_range[robot.joints].T
    q = np.zeros(len(robot.joints))
    for i, name in enumerate(robot.joint_names):
        if name in overrides:
            q[i] = overrides[name]
        elif name in lock:
            q[i] = 0.0
        elif name in template["joints"]:
            margin = LIMIT_MARGIN * (hi[i] - lo[i])
            q[i] = np.clip(template["joints"][name]["default"], lo[i] + margin, hi[i] - margin)
        else:
            raise ValueError(f"{name} is not a joint of the template; give its default with --default {name}=<rad>")
    return q


def standing_heights(robot, lb) -> tuple[float, float]:
    """Hip height (first joint of each leg) and root height above the feet's lowest collision point, current FK."""
    m, d = robot.model, robot.data
    feet = {b for side in ("left", "right") for b in robot.subtree(lb.feet[side])}
    lowest = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in feet and (m.geom_contype[g] or m.geom_conaffinity[g]):
            z_row = d.geom_xmat[g].reshape(3, 3)[2]  # world z of the geom's axes
            center, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            lowest = min(lowest, d.geom_xpos[g, 2] + z_row @ center - np.abs(z_row) @ half)
    hips = [robot.joints[lb.joints[f"{side}_leg"][0]] for side in ("left", "right")]
    return float(d.xanchor[hips, 2].mean() - lowest), float(d.xpos[robot.root, 2] - lowest)


def build_contract(
    robot_name: str,
    contract_name: str,
    *,
    template_name: str,
    proteus_root: Path = DEFAULT_PROTEUS_ROOT,
    lock: tuple[str, ...] = DEFAULT_LOCK,
    locked_gains: tuple[float, float] = DEFAULT_LOCKED_GAINS,
    defaults: dict[str, float] | None = None,
    stand_height: float | None = None,
    allow_kinematic_mismatch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Returns the contract, the robot description (Proteus spec YAML layout) and the warnings."""
    proteus = Proteus(proteus_root)
    rs, brs = proteus.spec, proteus.builder
    robot_cfg = get_robot_cfg(robot_name)
    mjcf = Path(robot_cfg.resolve_mjcf_path())
    template = proteus.description(template_name)
    template_contract = proteus.contract(template_name)
    reference = proteus.description(rs.REFERENCE_ROBOT)
    lock_set = set(lock)

    robot = brs.Robot(str(mjcf))
    if unknown := lock_set - set(robot.joint_names):
        raise ValueError(f"--lock names joints {sorted(unknown)} that {mjcf.name} does not have")
    missing = [name for name in robot.joint_names if name not in robot_cfg.joint_armature]
    if missing:
        raise ValueError(f"sim2real robot {robot_name!r} has no joint_armature for {missing}")
    for name, j in zip(robot.joint_names, robot.joints):  # the armatures the robot is simulated with
        robot.model.dof_armature[robot.model.jnt_dofadr[j]] = robot_cfg.joint_armature[name]

    q_ref = reference_pose(robot, template, lock_set, dict(defaults or {}))
    robot.fk(q_ref)
    lb = brs.limbs(robot)
    slots, warnings = brs.assign_slots(robot, brs.preferences(robot, lb), {})
    axes = {n: robot.data.xaxis[j].copy() for n, j in zip(robot.joint_names, robot.joints)}

    wrong_slots = {n: (s, template["joints"][n]["slot"]) for n, s in slots.items()
                   if n in template["joints"] and template["joints"][n]["slot"] != s}  # fmt: skip
    if wrong_slots:
        raise ValueError(f"slots differ from {template_name}'s (joint: (robot, template)): {wrong_slots}")
    for name, axis in axes.items():
        if name not in template["joints"]:
            continue
        off = float(np.linalg.norm(axis - np.asarray(template["joints"][name]["axis"])))
        if off > AXIS_TOLERANCE:
            message = f"{name}: axis in the reference pose {off:.3g} away from {template_name}'s"
            if not allow_kinematic_mismatch:
                raise ValueError(message + " (the template's pose and height may not apply; see --help)")
            warnings.append(message)

    signs, sign_warnings = brs.joint_signs(axes, slots, reference, brs.pelvis_side_joints(robot, lb.pelvis, slots))
    warnings += sign_warnings
    bodies = brs.semantic_bodies(robot, lb, slots)
    if bodies["pelvis"] == bodies["torso"] or any(bodies[f"{s}_elbow"] == bodies[f"{s}_wrist"] for s in ("left", "right")):
        raise ValueError(f"{mjcf.name} needs Proteus marker bodies (no waist joint or hand link): {bodies}")
    if robot.root != robot.model.body(bodies["pelvis"]).id:
        raise ValueError(f"the anchor {bodies['pelvis']} must be the floating base of {mjcf.name}")

    fk_height, root_height = standing_heights(robot, lb)
    height = float(template["stand_height"] if stand_height is None else stand_height)
    if abs(fk_height - height) > HEIGHT_TOLERANCE * height:
        warnings.append(f"stand height {height:.4f} m vs {fk_height:.4f} m from the reference pose")
    mass = float(robot.model.body_mass.sum())
    actuators = brs.actuators(robot, slots, mass, height, (reference["mass"], reference["stand_height"]))
    lo, hi = robot.model.jnt_range[robot.joints].T
    description = {
        "name": contract_name,
        "mjcf": f"{contract_name}/{mjcf.name}",
        "usd": "",
        "tree": "",
        "stand_height": round(height, 4),
        "root_height": round(root_height, 4),
        "self_collisions": False,
        "mass": round(mass, 3),
        "semantic_bodies": bodies,
        "markers": {},
        "feet": brs.contact_feet(robot, lb),
        "joints": {
            n: {"slot": slots[n], "sign": signs[n], "axis": [round(float(v), 4) for v in axes[n]],
                "lower": round(float(lo[i]), 4), "upper": round(float(hi[i]), 4), "default": round(float(q_ref[i]), 4),
                **actuators[n], "actuated": n not in lock_set}
            for i, n in enumerate(robot.joint_names)
        },  # fmt: skip
    }

    spec = rs.RobotSpec(
        name=contract_name, mjcf=description["mjcf"], usd="", tree="", stand_height=description["stand_height"],
        root_height=description["root_height"], self_collisions=False, mass=description["mass"],
        semantic_bodies=dict(bodies), markers={}, feet=tuple(description["feet"]),
        joints=tuple(rs.spec._joint(n, j) for n, j in description["joints"].items()),
    )  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        tree_path = Path(tmp) / "tree.npz"
        brs.export_tree(mjcf, tree_path)
        tree = proteus.KinematicTree.load(str(tree_path))
    reference_spec = rs.load_spec(rs.REFERENCE_ROBOT)
    morph = proteus.flat_descriptor(spec, tree, reference_spec, rs.layout).numpy()

    scale = rs.action_scale(spec)
    joints = []
    for j in spec.joints:
        joints.append({
            "name": j.name, "slot": rs.SLOTS.index(j.slot) if j.actuated else -1, "sign": float(j.sign),
            "default": j.default, "action_scale": scale.get(j.name, 0.0),
            "kp": j.stiffness if j.actuated else float(locked_gains[0]),
            "kd": j.damping if j.actuated else float(locked_gains[1]),
            "effort": j.effort, "armature": j.armature,
        })  # fmt: skip
    shared = {k: template_contract[k] for k in SHARED_FIELDS}
    contract = {  # the key order of the released contracts
        "robot": contract_name,
        "mjcf": description["mjcf"],
        "control_dt": shared["control_dt"],
        "physics_dt": shared["physics_dt"],
        "num_slots": shared["num_slots"],
        "joints": joints,
        "body_names": spec.body_names,
        "length_scale": spec.stand_height / reference_spec.stand_height,
        "self_collisions": False,
        "ground_friction": shared["ground_friction"],
        "joint_vel_scale": shared["joint_vel_scale"],
        "history": shared["history"],
        "future_offsets": shared["future_offsets"],
        "far_offset_range": shared["far_offset_range"],
        "morph": [float(v) for v in morph],
        "actor": shared["actor"],
        "source": {
            "builder": "sim2real scripts/build_proteus_contract.py",
            "robot": robot_name,
            "template": template_name,
            "mjcf_sha256": hashlib.sha256(mjcf.read_bytes()).hexdigest(),
            "locked": sorted(lock_set),
            "fk_stand_height": round(fk_height, 4),
        },
    }
    return contract, description, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", default="et1", help="sim2real robot config (MJCF, armatures)")
    parser.add_argument("--name", default="galbot_et1", help="contract name, <name>.json")
    parser.add_argument("--template", default="unitree_r1", help="released robot with the same kinematics")
    parser.add_argument("--proteus-root", type=Path, default=DEFAULT_PROTEUS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lock", nargs="*", default=list(DEFAULT_LOCK), help="joints without a slot")
    parser.add_argument("--locked-gains", nargs=2, type=float, default=list(DEFAULT_LOCKED_GAINS), metavar=("KP", "KD"))
    parser.add_argument("--default", nargs="*", default=[], help="joint=angle, replaces the template's default")
    parser.add_argument("--stand-height", type=float, help="standing hip height h in m (default: the template's)")
    parser.add_argument("--allow-kinematic-mismatch", action="store_true")
    args = parser.parse_args()

    contract, description, warnings = build_contract(
        args.robot,
        args.name,
        template_name=args.template,
        proteus_root=args.proteus_root,
        lock=tuple(args.lock),
        locked_gains=tuple(args.locked_gains),
        defaults={k: float(v) for k, v in (o.split("=") for o in args.default)},
        stand_height=args.stand_height,
        allow_kinematic_mismatch=args.allow_kinematic_mismatch,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / f"{args.name}.json"
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    with open(args.output_dir / f"{args.name}.yaml", "w", encoding="utf-8") as f:
        f.write(f"# generated by sim2real scripts/build_proteus_contract.py (template {args.template})\n")
        yaml.safe_dump(description, f, sort_keys=False, width=120)

    source = contract["source"]
    print(f"[{args.name}] {contract_path}")
    print(f"  h={description['stand_height']} m (reference-pose FK {source['fk_stand_height']} m), "
          f"M={description['mass']} kg, length_scale={contract['length_scale']:.4f}")  # fmt: skip
    for j in sorted(contract["joints"], key=lambda j: (j["slot"] < 0, j["slot"])):
        role = f"slot {j['slot']:2d}" if j["slot"] >= 0 else "locked "
        print(f"  {role} {j['name']:26s} sign {j['sign']:+.0f} default {j['default']:+.4f} kp {j['kp']:8.4f} "
              f"kd {j['kd']:.4f} effort {j['effort']:6.2f} scale {j['action_scale']:.4f}")  # fmt: skip
    print("  bodies: " + ", ".join(contract["body_names"]))
    for w in warnings:
        print(f"  WARNING {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
