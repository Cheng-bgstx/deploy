"""Shared MuJoCo actuator setup matching the exported implicit PD contract."""

import numpy as np
from pathlib import Path


def load_model(path: Path):
    """Load XML or a same-MuJoCo-version MJB cache into a fresh mutable model."""
    import mujoco

    path = Path(path)
    if path.suffix.lower() == '.mjb':
        return mujoco.MjModel.from_binary_path(str(path))
    if path.suffix.lower() == '.xml':
        return _load_surface_xml(path)
    raise ValueError(f'MuJoCo scene must be XML or MJB: {path}')


def configure_actuators(model, mapper, kp, kd, contract, *, drive_integration='velocity_implicit'):
    import mujoco

    ids = mapper.actuators
    if drive_integration not in ('velocity_implicit', 'backward_euler', 'explicit_pd'):
        raise ValueError('drive_integration must be velocity_implicit, backward_euler or explicit_pd')
    model.actuator_gaintype[ids] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[ids] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[ids] = 0
    model.actuator_biasprm[ids] = 0
    model.actuator_gainprm[ids, 0] = kp
    model.actuator_biasprm[ids, 1] = -np.asarray(kp)
    model.actuator_biasprm[ids, 2] = -np.asarray(kd)
    if drive_integration == 'backward_euler':
        # MuJoCo updates velocity by an implicit increment. Including kp*dt
        # in both the force and its velocity derivative yields the backward
        # Euler position spring: (M+dt*kd+dt²*kp)v_next=Mv+dt*kp(q_target-q).
        # This is a numerical integration term; exported hardware kd is unchanged.
        model.actuator_biasprm[ids, 2] -= np.asarray(kp) * model.opt.timestep
    elif drive_integration == 'explicit_pd':
        model.actuator_gainprm[ids, 0] = 1.
        model.actuator_biasprm[ids] = 0.
    model.actuator_ctrllimited[ids] = False
    model.actuator_forcelimited[ids] = True
    effort = np.asarray(contract["effort_limit_sim"])
    model.actuator_forcerange[ids] = np.stack((-effort, effort), axis=1)


def step_physics(model, data, mapper, kp, kd, *, drive_integration='velocity_implicit'):
    """Step the configured drive; preserve caller-applied generalized forces.

    backward_euler integrates position stiffness as well as velocity damping.
    It approximates PhysX implicit drives, not the complete TGS solver.
    """
    import mujoco

    if drive_integration in ('velocity_implicit', 'backward_euler'):
        mujoco.mj_step(model, data)
        return
    if drive_integration != 'explicit_pd':
        raise ValueError('drive_integration must be velocity_implicit, backward_euler or explicit_pd')
    target = data.ctrl[mapper.actuators].copy()
    try:
        data.ctrl[mapper.actuators] = (np.asarray(kp) * (target - data.qpos[mapper.qpos])
                                      - np.asarray(kd) * data.qvel[mapper.qvel])
        mujoco.mj_step(model, data)
    finally:
        data.ctrl[mapper.actuators] = target


def _load_surface_xml(path: Path):
    """Batch authored static vehicle geoms to avoid quadratic MjSpec insertion.

    Only generated vehicle hulls are batched, preserving the owning static body,
    geom attributes, mesh file paths and all robot ordering. Small/general XMLs
    keep MuJoCo's standard reader. Compiled caches bypass this path entirely.
    """
    import copy
    import xml.etree.ElementTree as ET
    import mujoco

    tree = ET.parse(path)
    root = tree.getroot()
    bodies = [b for b in root.findall('./worldbody/body')
              if b.get('name', '').startswith('vehicle_')]
    total = sum(len(b.findall('geom')) for b in bodies)
    if total < 2000:
        return mujoco.MjModel.from_xml_path(str(path))
    compiler = root.find('compiler')
    meshdir = (path.parent / compiler.get('meshdir', '.')).resolve()
    asset = root.find('asset')
    meshes = {}
    for mesh in asset.findall('mesh'):
        name = mesh.get('name') or Path(mesh.get('file')).stem
        meshes[name] = mesh
        mesh.set('file', str(meshdir / mesh.get('file')))
    batches = []
    used = set()
    for body in bodies:
        geoms = [g for g in body.findall('geom') if '_collision_' in g.get('mesh', '')]
        removed = set(geoms)
        body[:] = [element for element in body if element not in removed]
        used.update(geom.get('mesh') for geom in geoms)
        for start in range(0, len(geoms), 1000):
            batches.append((body.get('name'), geoms[start:start+1000]))
    # Shared mesh references must not be removed from the base model.
    remaining = {g.get('mesh') for g in root.iter('geom')}
    if used & remaining:
        raise ValueError('Generated surface hull assets must not be shared with nonvehicle geometry')
    removed_assets = {meshes[name] for name in used}
    asset[:] = [element for element in asset if element not in removed_assets]
    base = mujoco.MjSpec.from_string(ET.tostring(root, encoding='unicode'))
    for owner, geoms in batches:
        child_root = ET.Element('mujoco')
        ET.SubElement(child_root, 'compiler', angle='radian')
        child_asset = ET.SubElement(child_root, 'asset')
        world = ET.SubElement(child_root, 'worldbody')
        for geom in geoms:
            child_asset.append(copy.deepcopy(meshes[geom.get('mesh')]))
            world.append(copy.deepcopy(geom))
        child = mujoco.MjSpec.from_string(ET.tostring(child_root, encoding='unicode'))
        base.attach(child, frame=base.body(owner).add_frame(), prefix='', suffix='')
    return base.compile()
