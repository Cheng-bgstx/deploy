# /// script
# requires-python = "==3.10.*"
# dependencies = [
#     "mujoco==3.11.0",
#     "numpy==1.26.4",
#     "rtree==1.4.1",
#     "scipy==1.15.3",
#     "trimesh==4.5.1",
#     "usd-core==26.8",
# ]
# ///
"""Build the Vehicle Scene: Scalecar's car scene rebuilt offline and cropped to the Task Envelope.

Rebuilds Scalecar's scene from the ModelScope sources with Scalecar's own ``build_scene`` (vendored
verbatim under ``vendor/scalecar_deploy``, default options), keeps only the vehicle collision hulls
whose world AABB intersects the Task Envelope, and writes ``vehicle_scene.mjb`` plus ``build.json``.
Rationale: docs/adr/0004. The USD toolchain is declared inline and never enters the root env:

    uv run --script scripts/scalecar/build_vehicle_scene.py --out <abs path>/checkpoints/scalecar_d1146/scene

The full rebuild takes about 12 minutes and peaks near 8 GB RSS. ``--work-dir`` keeps the rebuild
and reuses it on later runs with identical inputs (e.g. to re-crop with another ``--margin``).
The outputs derive from private Isaac assets: keep them local, never commit or upload them.
Load the MJB with MuJoCo 3.11.0 only, and set ``model.vis.global_.bvactive = 0`` after loading.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import resource
import shutil
import sys
import tarfile
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

ENVELOPE_REGIONS = ("cabin_feet_region", "sill_region", "footwell_region")
POSE_ATTRIBUTES = ("pos", "quat", "axisangle", "euler", "xyaxes", "zaxis", "fromto")
HULL_BATCH = 1000  # Scalecar's MjSpec.attach batch size
MJB_NAME = "vehicle_scene.mjb"

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = Path(__file__).resolve().parent / "vendor"
SCALECAR_COMMIT = "7b8f370"
DEFAULT_MODELSCOPE = Path.home() / ".cache/modelscope/models/Hzzzz001--scalecar_v2/snapshots/master"
ISAAC_TAR = "isaac_assets_v1.tar"
SURFACE_TAR = "mujoco_reference_only_surface_v2.tar"
MOTIONS_TAR = "motions_merged1146_converted_50hz_v1.tar"
VEHICLE_MEMBER = "data/reference_only_surface_v2/vehicle.json"
ROBOT_DIR = "source/scalecar/scalecar/assets/robots/g1_29dof/"
ROBOT_XML = ROBOT_DIR + "g1_29dof.xml"
MOTION_DIR = "data/scalecar_g1_merged1146_20261020_v1/converted_50hz_v1/motions/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha256sums(directory: Path, names: Iterable[str]) -> dict[str, str]:
    """Check ``names`` in ``directory`` against its ``SHA256SUMS``; return their sha256s."""
    expected = {}
    for line in (Path(directory) / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest
    verified = {}
    for name in names:
        if name not in expected:
            raise ValueError(f"{name} is not listed in {directory}/SHA256SUMS")
        actual = sha256_file(Path(directory) / name)
        if actual != expected[name]:
            raise ValueError(f"{name}: sha256 {actual} does not match SHA256SUMS {expected[name]}")
        verified[name] = actual
    return verified


def task_envelope(vehicle: dict, motions: Iterable[np.ndarray], margin_m: float) -> dict:
    """Union AABB of the vehicle task regions and every Motion link origin, grown by ``margin_m``.

    ``motions`` yields ``body_pos_w`` arrays (any leading shape, last axis xyz, metres, world frame).
    """
    boxes = {name: np.asarray(vehicle[name], dtype=np.float64) for name in ENVELOPE_REGIONS}
    seat = np.asarray(vehicle["seat_target"], dtype=np.float64)
    boxes["seat_target"] = np.stack([seat, seat])
    boxes["steering_support.bounds"] = np.asarray(vehicle["steering_support"]["bounds"], dtype=np.float64)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for positions in motions:
        points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        lo = np.minimum(lo, points.min(0))
        hi = np.maximum(hi, points.max(0))
    boxes["motion_link_origins"] = np.stack([lo, hi])
    lo = np.min([box[0] for box in boxes.values()], axis=0) - margin_m
    hi = np.max([box[1] for box in boxes.values()], axis=0) + margin_m
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "margin_m": margin_m,
        "sources": {name: box.tolist() for name, box in boxes.items()},
    }


def intersects_envelope(aabb_min: np.ndarray, aabb_max: np.ndarray, envelope: dict) -> np.ndarray:
    """Crop rule: keep a hull when its world AABB ``[N,3]`` intersects the envelope (closed boxes)."""
    lo = np.asarray(envelope["min"], dtype=np.float64)
    hi = np.asarray(envelope["max"], dtype=np.float64)
    return np.all(np.asarray(aabb_max) >= lo, axis=1) & np.all(np.asarray(aabb_min) <= hi, axis=1)


def _obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray([line.split()[1:4] for line in path.read_text().splitlines() if line.startswith("v ")],
                          dtype=np.float64)
    return vertices.min(0), vertices.max(0)


def _is_hull(geom: ET.Element) -> bool:
    return "_collision_" in geom.get("mesh", "")


def _require_identity_pose(element: ET.Element) -> None:
    # Hull OBJ vertices are world coordinates only while body and geom carry no transform.
    posed = [key for key in POSE_ATTRIBUTES if key in element.attrib]
    if posed:
        raise ValueError(f"{element.tag} {element.get('name') or element.get('mesh')} has a pose ({posed}); "
                         "hull AABBs assume identity transforms")


def compile_cropped_scene(scene_xml: Path, envelope: dict):
    """Compile Scalecar's rebuilt scene.xml keeping only the hulls that intersect the envelope.

    Mirrors Scalecar's ``_load_surface_xml`` (vendor/scalecar_deploy/simulation/physics.py): hulls are
    attached in batches of 1000 through ``MjSpec.attach`` to their original static body, keeping their
    attributes and relative order, so contact ordering, and with it bitwise physics, is unchanged.
    The mesh directory goes into ``meshdir`` rather than the file names, so the MJB stores relative
    mesh paths and its bytes do not depend on where the rebuild lives.
    Returns the compiled model and per-component ``{"total", "kept"}`` hull counts.
    """
    import mujoco

    root = ET.parse(scene_xml).getroot()
    compiler = root.find("compiler")
    meshdir = (scene_xml.parent / compiler.get("meshdir", ".")).resolve()
    compiler.set("meshdir", str(meshdir))
    asset = root.find("asset")
    meshes = {mesh.get("name") or Path(mesh.get("file")).stem: mesh for mesh in asset.findall("mesh")}
    bodies = [b for b in root.findall("./worldbody/body") if b.get("name", "").startswith("vehicle_")]
    counts, batches, hull_meshes = {}, [], set()
    for body in bodies:
        _require_identity_pose(body)
        hulls = [g for g in body.findall("geom") if _is_hull(g)]
        body[:] = [element for element in body if not (element.tag == "geom" and _is_hull(element))]
        bounds = []
        for geom in hulls:
            _require_identity_pose(geom)
            bounds.append(_obj_bounds(meshdir / meshes[geom.get("mesh")].get("file")))
        keep = intersects_envelope(np.array([b[0] for b in bounds]).reshape(-1, 3),
                                   np.array([b[1] for b in bounds]).reshape(-1, 3), envelope)
        kept = [geom for geom, k in zip(hulls, keep) if k]
        counts[body.get("name").removeprefix("vehicle_")] = {"total": len(hulls), "kept": len(kept)}
        hull_meshes.update(geom.get("mesh") for geom in hulls)
        for start in range(0, len(kept), HULL_BATCH):
            batches.append((body.get("name"), kept[start:start + HULL_BATCH]))
    if hull_meshes & {g.get("mesh") for g in root.iter("geom")}:
        raise ValueError("Hull mesh assets must not be shared with non-hull geometry")
    asset[:] = [e for e in asset
                if not (e.tag == "mesh" and (e.get("name") or Path(e.get("file")).stem) in hull_meshes)]
    base = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for owner, geoms in batches:
        child_root = ET.Element("mujoco")
        ET.SubElement(child_root, "compiler", angle="radian", meshdir=str(meshdir))
        child_asset = ET.SubElement(child_root, "asset")
        child_world = ET.SubElement(child_root, "worldbody")
        for geom in geoms:
            child_asset.append(copy.deepcopy(meshes[geom.get("mesh")]))
            child_world.append(copy.deepcopy(geom))
        child = mujoco.MjSpec.from_string(ET.tostring(child_root, encoding="unicode"))
        base.attach(child, frame=base.body(owner).add_frame(), prefix="", suffix="")
    return base.compile(), counts


def write_cropped_scene(scene_xml: Path, vehicle: dict, motions: Iterable[np.ndarray], out_dir: Path, *,
                        margin_m: float, inputs: dict) -> dict:
    """Compute the Task Envelope, compile the cropped scene, write ``vehicle_scene.mjb`` and ``build.json``.

    ``inputs`` (input sha256s and provenance) is recorded verbatim. Returns the build.json record.
    """
    import mujoco

    envelope = task_envelope(vehicle, motions, margin_m)
    model, counts = compile_cropped_scene(Path(scene_xml), envelope)
    out_dir.mkdir(parents=True, exist_ok=True)
    mjb = out_dir / MJB_NAME
    mujoco.mj_saveModel(model, str(mjb), None)
    record = {
        "schema_version": 1,
        "mujoco_version": mujoco.__version__,
        "scene": {
            "file": MJB_NAME,
            "sha256": sha256_file(mjb),
            "bytes": mjb.stat().st_size,
            "load": {"vis.global_.bvactive": 0},
        },
        "task_envelope": {"frame": "world", **envelope},
        "hulls": {
            "total": sum(c["total"] for c in counts.values()),
            "kept": sum(c["kept"] for c in counts.values()),
            "per_component": counts,
        },
        "model": {key: int(getattr(model, key)) for key in ("nbody", "ngeom", "nmesh", "nq", "nv", "nu")},
        "inputs": inputs,
    }
    (out_dir / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def verify_vendored_sources() -> dict[str, str]:
    """Refuse to build with an edited copy of Scalecar's scene code; return the verified sha256s."""
    expected = {}
    for line in (VENDOR / "scalecar_deploy" / "SOURCE_SHA256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(maxsplit=1)
            expected[name] = digest
    for name, digest in expected.items():
        if sha256_file(VENDOR / "scalecar_deploy" / name) != digest:
            raise ValueError(f"vendor/scalecar_deploy/{name} differs from Scalecar_v2 @ {SCALECAR_COMMIT}")
    return expected


def _extract(tar_path: Path, members: list[str], dest: Path) -> None:
    """Extract only the named members (or member-prefixes ending in '/') of an untrusted tar."""
    with tarfile.open(tar_path) as tar:
        wanted = [m for m in tar.getmembers() if m.isfile() and any(
            m.name == name or (name.endswith("/") and m.name.startswith(name)) for name in members)]
        missing = [name for name in members if not any(
            m.name == name or (name.endswith("/") and m.name.startswith(name)) for m in wanted)]
        if missing:
            raise ValueError(f"{tar_path.name} lacks {missing}")
        tar.extractall(dest, members=wanted, filter="data")


def motion_link_boxes(tar_path: Path) -> tuple[list[np.ndarray], dict]:
    """Per-Motion AABB of all link origins over all frames, read from the motion tar with allow_pickle=False."""
    boxes, frames, links = [], 0, set()
    with tarfile.open(tar_path) as tar:
        for member in tar:
            name = member.name
            if not (member.isfile() and name.startswith(MOTION_DIR) and name.endswith(".npz")):
                continue
            with np.load(io.BytesIO(tar.extractfile(member).read()), allow_pickle=False) as motion:
                positions = np.asarray(motion["body_pos_w"], dtype=np.float64)
            if positions.ndim != 3 or positions.shape[2] != 3 or not np.isfinite(positions).all():
                raise ValueError(f"{name}: body_pos_w must be finite [frames, links, 3]")
            frames += positions.shape[0]
            links.add(positions.shape[1])
            points = positions.reshape(-1, 3)
            boxes.append(np.stack([points.min(0), points.max(0)]))
    if len(links) != 1:
        raise ValueError(f"Motions disagree on the link count: {sorted(links)}")
    return boxes, {"count": len(boxes), "links": links.pop(), "frames": frames}


def _toolchain() -> dict[str, str]:
    import importlib.metadata as metadata
    import platform

    names = ("mujoco", "numpy", "scipy", "trimesh", "rtree", "usd-core")
    return {"python": platform.python_version(), **{name: metadata.version(name) for name in names}}


def rebuild_scene(assets: Path, vehicle_json: Path, rebuild: Path, stamp: dict) -> dict:
    """Run Scalecar's build_scene (defaults) into ``rebuild``; reuse it when its stamp matches."""
    stamp_path = rebuild.parent / (rebuild.name + ".inputs.json")
    if rebuild.exists():
        if stamp_path.exists() and json.loads(stamp_path.read_text()) == stamp:
            print(f"REUSE rebuild {rebuild}", flush=True)
            return json.loads((rebuild / "conversion.json").read_text())
        raise ValueError(f"{rebuild} exists but was built from other inputs; delete it or pick another --work-dir")
    sys.path.insert(0, str(VENDOR))
    from scalecar_deploy.scene import build_scene

    report = build_scene(assets, vehicle_json, assets / ROBOT_XML, rebuild)
    stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
    return report


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelscope-dir", type=Path, default=DEFAULT_MODELSCOPE,
                        help="local snapshot of the ModelScope repo Hzzzz001/scalecar_v2")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "checkpoints/scalecar_d1146/scene")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="keep the full rebuild here and reuse it when inputs match (default: temporary)")
    parser.add_argument("--margin", type=float, default=0.3, help="Task Envelope margin in metres")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    vendored = verify_vendored_sources()
    asset_dir = args.modelscope_dir / "assets"
    tars = verify_sha256sums(asset_dir, [ISAAC_TAR, SURFACE_TAR, MOTIONS_TAR])
    toolchain = _toolchain()
    with tempfile.TemporaryDirectory(prefix="vehicle_scene_") as scratch:
        scratch = Path(scratch)
        work = args.work_dir.resolve() if args.work_dir else scratch
        work.mkdir(parents=True, exist_ok=True)
        inputs_dir = scratch / "inputs"  # fresh, empty: tar contents are untrusted
        _extract(asset_dir / SURFACE_TAR, [VEHICLE_MEMBER], inputs_dir)
        vehicle_json = inputs_dir / VEHICLE_MEMBER
        vehicle = json.loads(vehicle_json.read_text())
        _extract(asset_dir / ISAAC_TAR, [vehicle["static_usd"], vehicle["driver_door_usd"], ROBOT_DIR], inputs_dir)
        stamp = {"tars": tars, "vendored": vendored, "toolchain": toolchain}
        t0 = time.perf_counter()
        report = rebuild_scene(inputs_dir, vehicle_json, work / "rebuild", stamp)
        rebuild_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        boxes, motions = motion_link_boxes(asset_dir / MOTIONS_TAR)
        inputs = {
            "modelscope_tars": tars,
            "vehicle_json": {"member": f"{SURFACE_TAR}:{VEHICLE_MEMBER}", "sha256": sha256_file(vehicle_json)},
            "robot_xml": {"member": f"{ISAAC_TAR}:{ROBOT_XML}", "sha256": sha256_file(inputs_dir / ROBOT_XML)},
            "motions": {"member_prefix": f"{MOTIONS_TAR}:{MOTION_DIR}", **motions},
            "scalecar_sources": {"commit": SCALECAR_COMMIT, "sha256": vendored},
            "rebuild": {
                "build_scene": "default options",
                "ngeom": report["ngeom"],
                "hulls": {c["name"]: c["hulls"] for c in report["components"]},
            },
            "toolchain": toolchain,
        }
        record = write_cropped_scene(work / "rebuild" / "scene.xml", vehicle, boxes, args.out.resolve(),
                                     margin_m=args.margin, inputs=inputs)
        crop_s = time.perf_counter() - t0
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(json.dumps({
        "out": str(args.out.resolve()),
        "envelope_min": [round(v, 3) for v in record["task_envelope"]["min"]],
        "envelope_max": [round(v, 3) for v in record["task_envelope"]["max"]],
        "hulls_kept": record["hulls"]["kept"], "hulls_total": record["hulls"]["total"],
        "mjb_sha256": record["scene"]["sha256"], "mjb_bytes": record["scene"]["bytes"],
        "rebuild_s": round(rebuild_s, 1), "crop_s": round(crop_s, 1),
        "total_s": round(time.perf_counter() - started, 1), "peak_rss_gb": round(peak_gb, 2),
    }, indent=2), flush=True)
    return record


if __name__ == "__main__":
    main()
