import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.scalecar.build_vehicle_scene import (
    intersects_envelope,
    task_envelope,
    verify_sha256sums,
    write_cropped_scene,
)


VEHICLE = {
    "cabin_feet_region": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    "sill_region": [[-1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
    "footwell_region": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    "seat_target": [2.0, 0.5, 0.5],
    "steering_support": {"bounds": [[0.0, 0.0, 2.0], [0.5, 0.5, 3.0]]},
}


def _motion(*points):
    # body_pos_w layout: [frames, links, 3]
    return np.asarray(points, dtype=np.float64).reshape(len(points), 1, 3)


def test_task_envelope_is_union_of_vehicle_regions_and_all_motion_links_plus_margin():
    motions = [
        _motion([0.2, -2.0, 0.4], [0.3, 0.1, 0.5]),
        _motion([0.1, 0.2, -0.5]),
    ]

    envelope = task_envelope(VEHICLE, motions, margin_m=0.3)

    assert envelope["min"] == pytest.approx([-1.3, -2.3, -0.8])
    assert envelope["max"] == pytest.approx([2.3, 1.3, 3.3])
    assert envelope["margin_m"] == 0.3


def test_crop_rule_keeps_every_hull_whose_world_aabb_touches_the_envelope():
    envelope = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    hulls = {
        "inside": ([0.2, 0.2, 0.2], [0.4, 0.4, 0.4]),
        "straddles_x_face": ([0.9, 0.2, 0.2], [1.5, 0.4, 0.4]),
        "touches_z_face": ([0.2, 0.2, 1.0], [0.4, 0.4, 1.2]),
        "encloses_envelope": ([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0]),
        "beyond_x": ([1.01, 0.2, 0.2], [1.5, 0.4, 0.4]),
        "below_z": ([0.2, 0.2, -0.5], [0.4, 0.4, -0.01]),
        "overlaps_x_only": ([0.2, 2.0, 2.0], [0.4, 3.0, 3.0]),
    }
    lo = np.array([box[0] for box in hulls.values()])
    hi = np.array([box[1] for box in hulls.values()])

    keep = intersects_envelope(lo, hi, envelope)

    assert dict(zip(hulls, keep.tolist())) == {
        "inside": True,
        "straddles_x_face": True,
        "touches_z_face": True,
        "encloses_envelope": True,
        "beyond_x": False,
        "below_z": False,
        "overlaps_x_only": False,
    }


TETRAHEDRON = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)]
# Hull x offsets per vehicle component; the synthetic envelope below spans [-1, 1] on every axis.
HULL_OFFSETS = {"carbody": [0.0, 5.0, 0.5, 10.0], "Seat": [-0.95, 20.0]}


def _write_obj(path: Path, x_offset: float) -> None:
    lines = [f"v {x + x_offset} {y} {z}" for x, y, z in TETRAHEDRON]
    lines += ["f 1 2 3", "f 1 2 4", "f 1 3 4", "f 2 3 4"]
    path.write_text("\n".join(lines) + "\n")


def _write_rebuilt_scene(root: Path) -> Path:
    """A miniature of Scalecar's build_scene output: robot, ground and static vehicle_* bodies."""
    meshes = root / "meshes"
    meshes.mkdir(parents=True)
    assets, bodies = [], []
    for component, offsets in HULL_OFFSETS.items():
        _write_obj(meshes / f"{component}_visual.obj", 0.0)
        assets.append(f'<mesh name="{component}_visual" file="{component}_visual.obj"/>')
        geoms = [f'<geom name="{component}_visual" type="mesh" mesh="{component}_visual" '
                 'contype="0" conaffinity="0" group="2" density="0"/>']
        for index, offset in enumerate(offsets):
            _write_obj(meshes / f"{component}_collision_{index}.obj", offset)
            assets.append(f'<mesh file="{component}_collision_{index}.obj"/>')
            geoms.append(f'<geom type="mesh" mesh="{component}_collision_{index}" group="3" '
                         'contype="1" conaffinity="1" density="0"/>')
        bodies.append(f'<body name="vehicle_{component}">{"".join(geoms)}</body>')
    xml = f"""<mujoco model="mini">
  <compiler angle="radian" meshdir="meshes"/>
  <asset>{"".join(assets)}</asset>
  <worldbody>
    <body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body>
    <geom name="ground" type="plane" size="0 0 .1"/>
    {"".join(bodies)}
  </worldbody>
</mujoco>"""
    scene = root / "scene.xml"
    scene.write_text(xml)
    return scene


MINI_VEHICLE = {
    "cabin_feet_region": [[-0.7, -0.7, -0.7], [0.0, 0.0, 0.0]],
    "sill_region": [[0.0, 0.0, 0.0], [0.7, 0.7, 0.7]],
    "footwell_region": [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]],
    "seat_target": [0.2, 0.2, 0.2],
    "steering_support": {"bounds": [[0.0, 0.0, 0.0], [0.3, 0.3, 0.3]]},
}
MINI_MOTIONS = [_motion([0.5, 0.5, 0.5], [-0.5, -0.5, -0.5])]


def _vehicle_geoms(model):
    import mujoco

    out = {}
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
        if body.startswith("vehicle_"):
            mesh = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[geom]))
            out.setdefault(body, []).append(mesh)
    return out


def test_cropped_scene_keeps_enveloped_hulls_in_their_body_and_order_plus_all_visuals(tmp_path):
    import mujoco

    scene = _write_rebuilt_scene(tmp_path / "rebuild")

    write_cropped_scene(scene, MINI_VEHICLE, MINI_MOTIONS, tmp_path / "out", margin_m=0.3, inputs={})

    model = mujoco.MjModel.from_binary_path(str(tmp_path / "out" / "vehicle_scene.mjb"))
    assert _vehicle_geoms(model) == {
        "vehicle_carbody": ["carbody_visual", "carbody_collision_0", "carbody_collision_2"],
        "vehicle_Seat": ["Seat_visual", "Seat_collision_0"],
    }


def test_build_json_records_inputs_mujoco_version_envelope_and_kept_hulls(tmp_path):
    scene = _write_rebuilt_scene(tmp_path / "rebuild")
    inputs = {"isaac_assets_v1.tar": {"sha256": "ab" * 32}}

    write_cropped_scene(scene, MINI_VEHICLE, MINI_MOTIONS, tmp_path / "out", margin_m=0.3, inputs=inputs)

    record = json.loads((tmp_path / "out" / "build.json").read_text())
    mjb = (tmp_path / "out" / "vehicle_scene.mjb").read_bytes()
    assert record["inputs"] == inputs
    assert record["mujoco_version"] == "3.11.0"
    assert record["scene"] == {
        "file": "vehicle_scene.mjb",
        "sha256": hashlib.sha256(mjb).hexdigest(),
        "bytes": len(mjb),
        "load": {"vis.global_.bvactive": 0},
    }
    assert record["task_envelope"]["min"] == pytest.approx([-1.0, -1.0, -1.0])
    assert record["task_envelope"]["max"] == pytest.approx([1.0, 1.0, 1.0])
    assert record["task_envelope"]["margin_m"] == 0.3
    assert record["hulls"] == {
        "total": 6,
        "kept": 3,
        "per_component": {"carbody": {"total": 4, "kept": 2}, "Seat": {"total": 2, "kept": 1}},
    }


def test_mjb_bytes_do_not_depend_on_where_the_rebuild_lives(tmp_path):
    first = _write_rebuilt_scene(tmp_path / "a" / "rebuild")
    second = _write_rebuilt_scene(tmp_path / "somewhere" / "else" / "rebuild")

    write_cropped_scene(first, MINI_VEHICLE, MINI_MOTIONS, tmp_path / "out_a", margin_m=0.3, inputs={})
    write_cropped_scene(second, MINI_VEHICLE, MINI_MOTIONS, tmp_path / "out_b", margin_m=0.3, inputs={})

    assert (tmp_path / "out_a" / "vehicle_scene.mjb").read_bytes() == (tmp_path / "out_b" / "vehicle_scene.mjb").read_bytes()
    assert (tmp_path / "out_a" / "build.json").read_bytes() == (tmp_path / "out_b" / "build.json").read_bytes()


def _sums(directory: Path, files: dict) -> None:
    lines = []
    for name, data in files.items():
        (directory / name).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
    (directory / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def test_inputs_are_verified_against_sha256sums(tmp_path):
    _sums(tmp_path, {"a.tar": b"alpha", "b.tar": b"beta"})

    assert verify_sha256sums(tmp_path, ["a.tar"]) == {
        "a.tar": "8ed3f6ad685b959ead7022518e1af76cd816f8e8ec7ccdda1ed4018e8f2223f8"
    }

    (tmp_path / "b.tar").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="b.tar"):
        verify_sha256sums(tmp_path, ["a.tar", "b.tar"])
    with pytest.raises(ValueError, match="c.tar"):
        verify_sha256sums(tmp_path, ["c.tar"])
