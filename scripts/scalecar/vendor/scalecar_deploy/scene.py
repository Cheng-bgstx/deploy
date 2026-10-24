"""Convert the training USD vehicle into a portable MuJoCo scene.

Local convex shells approximate static triangle contacts. Rendering keeps the
original triangles by default; bidirectional distance reports gate publication.
"""
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np


def replace_world_ground(root: ET.Element) -> list[str]:
    """Replace inherited world planes with one ground at z=0 (metres).

    Robot MJCFs may contain multiple worldbody sections. Leaving their planes
    while adding the deployment ground creates duplicate contact constraints.
    Non-plane and body-local geometry is retained.
    """
    worlds = root.findall('worldbody')
    if not worlds:
        raise ValueError('Robot MJCF must contain a worldbody')
    removed = []
    for world in worlds:
        for geom in list(world.findall('geom')):
            if geom.get('type') == 'plane':
                removed.append(geom.get('name', '<unnamed>'))
                world.remove(geom)
    ET.SubElement(worlds[0], 'geom', name='ground', type='plane', size='0 0 .1',
                  friction='1 .005 .0001', rgba='.18 .20 .22 1', contype='1', conaffinity='1')
    return removed


def match_robot_collision_shapes(root: ET.Element, robot_usd: Path) -> list[str]:
    """Transfer sphere/capsule collision dimensions and body-local poses from USD.

    The robot XML and training USD can contain different authored dimensions.
    Instance proxies must be traversed: the training colliders are instanced.
    Mesh collision approximation remains the separately documented conversion.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(robot_usd))
    if stage is None:
        raise ValueError(f'Cannot read robot USD: {robot_usd}')
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError(f'Robot USD must use metre units: {robot_usd}')
    geoms = {g.get('name'): g for g in root.iter('geom') if g.get('name')}
    owners = {g.get('name'): b.get('name') for b in root.iter('body')
              for g in b.findall('geom') if g.get('name')}
    matched = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetTypeName() not in ('Sphere', 'Capsule'):
            continue
        name = prim.GetName()
        if name not in geoms:
            raise ValueError(f'USD collision {name} has no named MJCF geometry')
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError(f'USD collision {name} has no rigid body ancestor')
        if owners.get(name) != body.GetName():
            raise ValueError(f'USD/MJCF collision body mismatch for {name}')
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        matrix = matrix @ np.linalg.inv(np.asarray(
            UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())))
        if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-5):
            raise ValueError(f'USD collision {name} has unsupported non-unit scale')
        geom = geoms[name]
        if geom.get('type') != prim.GetTypeName().lower():
            raise ValueError(f'USD/MJCF collision type mismatch for {name}')
        for key in ('pos', 'quat', 'axisangle', 'euler', 'xyaxes', 'zaxis', 'fromto'):
            geom.attrib.pop(key, None)
        shape = UsdGeom.Sphere(prim) if prim.GetTypeName() == 'Sphere' else UsdGeom.Capsule(prim)
        geom.set('size', format(float(shape.GetRadiusAttr().Get()), '.17g'))
        if prim.GetTypeName() == 'Sphere':
            geom.set('pos', ' '.join(format(v, '.17g') for v in matrix[3, :3]))
        else:
            endpoints = np.zeros((2, 4))
            endpoints[:, 3] = 1.
            axis = {'X': 0, 'Y': 1, 'Z': 2}[shape.GetAxisAttr().Get()]
            endpoints[:, axis] = np.array([-1., 1.]) * float(shape.GetHeightAttr().Get()) / 2
            geom.set('fromto', ' '.join(format(v, '.17g') for v in (endpoints @ matrix)[:, :3].ravel()))
        matched.append(name)
    if not matched:
        raise ValueError(f'Robot USD contains no supported collision shapes: {robot_usd}')
    return matched


def apply_robot_collision_filters(root: ET.Element, robot_usd: Path) -> list[list[str]]:
    """Carry body and articulation exclusions into MJCF body pairs.

    A prim can be both a rigid body and an articulation root. PhysX applies
    its filteredPairs to the entire articulation, not just that root body.
    MJCF has no articulation exclusion, so expand it to named body pairs.
    """
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(robot_usd))
    if stage is None:
        raise ValueError(f'Cannot read robot USD: {robot_usd}')
    bodies = {b.get('name') for b in root.iter('body')}

    def members(prim):
        if not prim:
            raise ValueError('USD collision exclusion references an invalid prim')
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            # The USD stores links as siblings below the articulation's
            # container; the actual robot tree is expressed by its joints.
            if prim.GetName() not in bodies:
                raise ValueError(f'USD articulation root has no MJCF body: {prim.GetPath()}')
            body = next(b for b in root.iter('body') if b.get('name') == prim.GetName())
            return {b.get('name') for b in body.iter('body')}
        if prim.GetName() not in bodies:
            raise ValueError(f'USD collision exclusion references missing MuJoCo body: {prim.GetPath()}')
        return {prim.GetName()}

    pairs = set()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for other in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                for a in members(prim):
                    for b in members(stage.GetPrimAtPath(other)):
                        if a != b:
                            pairs.add(tuple(sorted((a, b))))
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    existing = {tuple(sorted((e.get('body1'), e.get('body2')))) for e in contact.findall('exclude')}
    for a, b in sorted(pairs):
        if a not in bodies or b not in bodies:
            raise ValueError(f'USD collision exclusion references missing MuJoCo body: {a}, {b}')
        if (a, b) not in existing:
            ET.SubElement(contact, 'exclude', body1=a, body2=b)
    return [list(pair) for pair in sorted(pairs)]


def _usd_meshes(path: Path):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        matrix = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix)[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        if not np.all(counts == 3):
            raise ValueError(f'Expected triangulated source mesh: {prim.GetPath()}')
        yield (str(prim.GetPath()).split('/')[2], np.array(vertices, dtype=np.float64, order='C'),
               np.array(indices.reshape(-1, 3), dtype=np.int32, order='C'))


def build_scene(asset_root: Path, vehicle_path: Path, robot_xml: Path,
                output: Path, collision_error_m: float = .01, *,
                carbody_collision: str = 'spatial_surface', spatial_cell_m: float = .10,
                vehicle_collision: str | None = 'spatial_surface', visual_geometry: str = 'raw',
                surface_error_m: float = .01, surface_samples: int = 4096) -> dict:
    import trimesh
    import mujoco

    if not 0 < collision_error_m <= .02:
        raise ValueError('collision_error_m must be in (0, 0.02] meters')
    if carbody_collision not in ('coacd', 'spatial_surface'):
        raise ValueError('carbody_collision must be coacd or spatial_surface')
    if vehicle_collision not in (None, 'coacd', 'spatial_surface'):
        raise ValueError('vehicle_collision must be coacd or spatial_surface')
    if visual_geometry not in ('raw', 'simplified'):
        raise ValueError('visual_geometry must be raw or simplified')
    output.mkdir(parents=True, exist_ok=False)
    meshes = output / 'meshes'
    meshes.mkdir()
    vehicle = json.loads(vehicle_path.read_text())
    tree = ET.parse(robot_xml)
    root = tree.getroot()
    root.set('model', 'scalecar_chevrolet_deployment')
    robot_usd = robot_xml.with_suffix('.usda')
    excluded_pairs = apply_robot_collision_filters(root, robot_usd)
    matched_shapes = match_robot_collision_shapes(root, robot_usd)
    compiler = root.find('compiler')
    compiler.set('meshdir', 'meshes')
    for mesh in root.findall('./asset/mesh'):
        source = robot_xml.parent / 'meshes' / mesh.get('file')
        shutil.copy2(source, meshes / source.name)
    option = root.find('option')
    if option is None:
        option = ET.SubElement(root, 'option')
    option.attrib.update(timestep='.005', integrator='implicitfast', solver='Newton',
                         iterations='100', tolerance='1e-8', cone='elliptic')
    world = root.find('worldbody')
    removed_planes = replace_world_ground(root)
    ET.SubElement(world, 'light', pos='0 -3 5', dir='0 0 -1', diffuse='.8 .8 .8')
    asset = root.find('asset')
    report = dict(schema_version=3, door_degrees=70., collision='Per-component convex surface approximation',
        carbody_collision=carbody_collision, spatial_cell_m=spatial_cell_m,
        vehicle_collision=vehicle_collision, visual_geometry=visual_geometry,
        collision_geom_naming='Left seat names retained; other vehicle hulls use body membership and mesh filename stems',
        surface_error_m=surface_error_m, surface_samples=surface_samples,
        replaced_world_planes=removed_planes, usd_matched_robot_collision_shapes=matched_shapes,
        requested_concavity_m=collision_error_m, seed=101, components=[], excluded_body_pairs=excluded_pairs,
        limitation='Collision approximation differs from Isaac triangle meshes; tolerance is a decomposition request, not a certified surface-error bound.',
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [vehicle_path, robot_xml, robot_usd, asset_root / vehicle['static_usd'], asset_root / vehicle['driver_door_usd']]})
    sources = [(asset_root / vehicle['static_usd'], False), (asset_root / vehicle['driver_door_usd'], True)]
    for source, door in sources:
        for name, vertices, faces in _usd_meshes(source):
            if door:
                name = 'DriverDoor'
                theta = np.deg2rad(180. - 70.)
                rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
                pivot = np.array(vehicle['driver_door']['pivot']) * [-1, -1, 1] + vehicle['translation']
                vertices = vertices @ rotation.T + pivot
            print(f'CONVERT {name}: {len(faces)} source triangles', flush=True)
            # Decimation affects only the explicitly reported MuJoCo conversion.
            visual_v, visual_f = vertices, faces
            if visual_geometry == 'simplified' and len(faces) > 50000:
                import fast_simplification
                visual_v, visual_f = fast_simplification.simplify(vertices, faces, target_count=50000)
            from .surface_validation import surface_distance_report, enforce_surface_distance
            visual_report = surface_distance_report(vertices, faces, visual_v, visual_f,
                threshold_m=surface_error_m, samples=surface_samples)
            enforce_surface_distance(visual_report, f'{name} visual', output / f'{name}_visual_distance.json')
            filename = f'{name}_visual.obj'
            trimesh.Trimesh(visual_v, visual_f, process=False).export(meshes / filename)
            ET.SubElement(asset, 'mesh', name=f'{name}_visual', file=filename)
            body = ET.SubElement(world, 'body', name=f'vehicle_{name}')
            ET.SubElement(body, 'geom', name=f'{name}_visual', type='mesh', mesh=f'{name}_visual',
                          contype='0', conaffinity='0', group='2', density='0', rgba='.65 .68 .72 1')
            collision_v, collision_f = vertices, faces
            method = vehicle_collision or (carbody_collision if name == 'carbody' else 'coacd')
            spatial = method == 'spatial_surface'
            if spatial:
                from .vehicle_collision import spatial_triangle_hulls
                parts = spatial_triangle_hulls(vertices, faces, cell_m=spatial_cell_m)
            else:
                import coacd
                import fast_simplification
                coacd.set_log_level('warn')
                if len(faces) > 20000:
                    collision_v, collision_f = fast_simplification.simplify(vertices, faces, target_count=20000)
                parts = coacd.run_coacd(coacd.Mesh(collision_v, collision_f),
                    threshold=collision_error_m, real_metric=True, preprocess_mode='auto',
                    preprocess_resolution=80, resolution=1500, mcts_nodes=15,
                    mcts_iterations=80, mcts_max_depth=3, seed=101, max_ch_vertex=64)
            count = 0
            all_vertices, all_faces = [], []
            vertex_offset = 0
            for index, (part_v, part_f) in enumerate(parts):
                count += 1
                all_vertices.append(part_v)
                all_faces.append(np.asarray(part_f) + vertex_offset)
                vertex_offset += len(part_v)
                part_name = f'{name}_collision_{index}'
                trimesh.Trimesh(part_v, part_f, process=False).export(meshes / f'{part_name}.obj')
                # Filename stem supplies the mesh name during compile. Avoid
                # MuJoCo 3.11 sorting all prior names at every XML insertion.
                ET.SubElement(asset, 'mesh', file=f'{part_name}.obj')
                geom = ET.SubElement(body, 'geom', type='mesh', mesh=part_name,
                    group='3', contype='1', conaffinity='1', friction='1 .005 .0001', density='0')
                if name == 'VehicleSeat_Left':
                    geom.set('name', part_name)
            if not count:
                raise RuntimeError(f'No collision parts generated for {name}')
            collision_report = surface_distance_report(vertices, faces, np.concatenate(all_vertices),
                np.concatenate(all_faces), threshold_m=surface_error_m, samples=surface_samples)
            enforce_surface_distance(collision_report, f'{name} collision', output / f'{name}_collision_distance.json')
            report['components'].append(dict(name=name, source_triangles=len(faces),
                collision_method='spatial_surface' if spatial else 'coacd',
                decomposition_triangles=len(collision_f), hulls=count,
                source_bounds=[vertices.min(0).tolist(), vertices.max(0).tolist()]))
            print(f'DONE {name}: {count} convex parts', flush=True)
    ET.indent(root)
    scene_path = output / 'scene.xml'
    tree.write(scene_path, encoding='utf-8', xml_declaration=True)
    from .simulation.physics import load_model
    model = load_model(scene_path)
    report.update(compiled=True, ngeom=model.ngeom, njnt=model.njnt,
                  nq=model.nq, nu=model.nu, mujoco_version=mujoco.__version__)
    report['asset_hashes'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'conversion.json').write_text(json.dumps(report, indent=2))
    return report
