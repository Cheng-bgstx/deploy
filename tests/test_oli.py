from types import SimpleNamespace
import sys
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from sim2real.config.robots import OLI_CFG, get_robot_cfg
from sim2real.config.robots.oli import OLI_SDK_JOINT_NAMES
from sim2real.rl_policy.robot_io.oli import OliRobotIO
from sim2real.sim_env.utils.mjcf import _inject_floor_scene_xml


@pytest.fixture
def backend(monkeypatch):
    robot = SimpleNamespace(init=lambda ip: True, getMotorNumber=lambda: 31,
                            subscribeRobotState=lambda cb: None, subscribeImuData=lambda cb: None,
                            publishRobotCmd=Mock(return_value=True))
    # Exercise the real constructor without networking or a nondeterministic publisher.
    monkeypatch.setenv('ROBOT_TYPE', 'HU_D04_01')
    dt = SimpleNamespace(RobotCmd=SimpleNamespace)
    sdk_robot = SimpleNamespace(Robot=lambda kind: robot, RobotType=SimpleNamespace(Humanoid=1))
    monkeypatch.setitem(sys.modules, 'limxsdk', SimpleNamespace(datatypes=dt, robot=sdk_robot))
    monkeypatch.setitem(sys.modules, 'limxsdk.datatypes', dt)
    monkeypatch.setitem(sys.modules, 'limxsdk.robot', sdk_robot)
    monkeypatch.setitem(sys.modules, 'limxsdk.robot.Robot', sdk_robot.Robot)
    monkeypatch.setitem(sys.modules, 'limxsdk.robot.RobotType', sdk_robot.RobotType)
    monkeypatch.setattr(threading.Thread, 'start', lambda self: None)
    io = OliRobotIO(OLI_CFG)
    yield io, robot


def feed(io):
    io._on_state(SimpleNamespace(q=np.arange(31), dq=np.arange(31) + 100,
                                tau=np.arange(31) + 200, stamp=12_000_000, motor_names=[]))
    io._on_imu(SimpleNamespace(quat=[1, 0, 0, 0], gyro=[0.1, 0.2, 0.3]))


def test_sdk_joint_permutation_and_pr_commands(backend):
    io, robot = backend
    assert io.read_state() is None
    feed(io)
    state = io.read_state()
    assert state.tick == 12
    # SDK: head pitch=15, yaw=16. Runtime/MJCF: yaw=15, pitch=16.
    np.testing.assert_array_equal(state.qpos[7 + 15:7 + 17], [16, 15])
    np.testing.assert_array_equal(state.qpos[3:7], [1, 0, 0, 0])
    np.testing.assert_allclose(state.qvel[3:6], [.1, .2, .3])
    assert state.joint_torque[15] == 216
    q = np.arange(31, dtype=float) / 10
    io.set_control_mode('policy', 1)
    io.write_command(q, q + 1, q + 2, q + 3, q + 4)
    io._publish()
    cmd = robot.publishRobotCmd.call_args.args[0]
    assert cmd.parallel_solve_required == [True] * 31
    assert cmd.mode == [0] * 31
    np.testing.assert_allclose(cmd.q[15:17], [q[16], q[15]])
    np.testing.assert_allclose(cmd.Kp[15:17], [q[16] + 3, q[15] + 3])
    io.set_control_mode('zero', 2)
    io._publish()
    np.testing.assert_array_equal(robot.publishRobotCmd.call_args.args[0].Kp, 0)
    io.set_control_mode('policy', 3)
    io._command_time -= 1
    io._publish()
    expired = robot.publishRobotCmd.call_args.args[0]
    np.testing.assert_array_equal(expired.Kp, 0)
    np.testing.assert_array_equal(expired.tau, 0)
    assert np.all(np.array(expired.Kd) > 0)
    io._state_time -= 1
    assert io.read_state() is None


def test_bad_feedback_and_commands_fail(backend):
    io, _ = backend
    with pytest.raises(ValueError, match='finite'):
        io.write_command(np.full(31, np.nan), *([np.zeros(31)] * 4))
    io._on_imu(SimpleNamespace(quat=[0, 0, 0, 0], gyro=[0, 0, 0]))
    with pytest.raises(RuntimeError, match='transport'):
        io.read_state()


def test_config_and_existing_vendor_floor():
    assert get_robot_cfg('limx_oli_edu') is OLI_CFG
    assert len(OLI_CFG.default_qpos) == 38
    assert len(set(OLI_SDK_JOINT_NAMES)) == 31
    xml = '<mujoco><asset></asset><worldbody><geom name="floor" type="plane" size="0 0 .1"/></worldbody></mujoco>'
    import xml.etree.ElementTree as ET
    root = ET.fromstring(_inject_floor_scene_xml(xml))
    assert len(root.findall('worldbody/geom')) == 1


@pytest.mark.skipif(not OLI_CFG.mjcf_path.is_file(), reason='OLI runtime model not staged')
def test_packed_motion_conversion_maps_joints_and_root_by_name(tmp_path):
    import json
    import mujoco
    from scripts.oli.convert_motion import convert
    model = mujoco.MjModel.from_xml_path(str(OLI_CFG.mjcf_path))
    data = mujoco.MjData(model)
    joints = list(reversed(OLI_CFG.joint_names))
    bodies = list(reversed(OLI_CFG.body_names))
    qpos = np.tile(model.qpos0, (3, 1))
    qpos[:, 0] = [.1, .2, .3]
    qpos[:, 3:7] = [np.cos(.15), 0, 0, np.sin(.15)]  # wxyz
    for i, n in enumerate(joints):
        qpos[:, model.joint(n).qposadr[0]] = .001 * i
    positions, rotations = [], []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        positions.append(data.xpos[[model.body(n).id for n in bodies]].copy())
        rotations.append(data.xquat[[model.body(n).id for n in bodies]].copy())
    library = tmp_path / 'library'
    library.mkdir()
    (library / 'index.json').write_text(json.dumps({'fps': 50, 'lengths': [3], 'motion_names': ['test'],
                                                  'joint_names': joints, 'body_names': bodies}))
    fields = {'root_pos_w': qpos[:, :3], 'root_quat_w': qpos[:, 3:7],
              'joint_pos': qpos[:, [model.joint(n).qposadr[0] for n in joints]],
              'body_pos_w': positions, 'body_quat_w': rotations}
    for key, value in fields.items():
        np.save(library / f'{key}.npy', value)
    path = convert(tmp_path / 'converted', library=library)
    np.testing.assert_allclose(np.load(path)['qpos'], qpos, atol=1e-7)


@pytest.mark.skipif(not OLI_CFG.mjcf_path.is_file(), reason='OLI runtime model not staged')
def test_integrated_model_uses_floor_contacts_without_self_collisions():
    import mujoco
    from sim2real.sim_env.utils.mjcf import load_sim_model
    model = load_sim_model(OLI_CFG)
    assert model.nu == 31
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            assert (model.geom_contype[i], model.geom_conaffinity[i]) == (2, 1)
        elif model.geom_contype[i] or model.geom_conaffinity[i]:
            assert (model.geom_contype[i], model.geom_conaffinity[i]) == (1, 2)
    np.testing.assert_array_equal(model.dof_damping, 0)


def test_wamseq_export_preserves_semantic_inputs_and_slot_output(tmp_path):
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from scripts.export_proteus_onnx import input_shapes, source_inputs, wrap_actor, compare_with_source, policy_yaml
    # Tiny source actor which returns the last raw slot action, with reversed joint slots.
    contract = {'robot': 'limx_oli_edu', 'actor_kind': 'wam_seq', 'seq_window': 3, 'history': 3,
                'num_slots': 31, 'morph': [0.0], 'future_offsets': [0, -1], 'far_offset_range': [5, 33],
                'body_names': ['base_link'], 'joint_vel_scale': .05, 'length_scale': 1., 'control_dt': .02,
                'joints': [{'name': n, 'slot': 30 - i, 'sign': 1., 'default': 0., 'kp': 1., 'kd': 1.,
                            'action_scale': .5} for i, n in enumerate(OLI_CFG.joint_names)]}
    shapes = input_shapes(contract, 16)
    inputs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, *shapes[v]])
              for n, v in source_inputs(contract).items()]
    inputs.append(helper.make_tensor_value_info('morph', TensorProto.FLOAT, [1, 1]))
    graph = helper.make_graph([helper.make_node('Gather', ['prev_actions', 'last'], ['actions'], axis=1)],
                              'test_actor', inputs, [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 31])],
                              [numpy_helper.from_array(np.array(2, dtype=np.int64), 'last')])
    source = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    source.ir_version = 8
    path = tmp_path / 'actor.onnx'
    onnx.save(source, path)
    wrapped = wrap_actor(source, contract, 16)
    assert {i.name: [d.dim_value for d in i.type.tensor_type.shape.dim] for i in wrapped.graph.input} == {
        n: list(v) for n, v in shapes.items()}
    assert compare_with_source(wrapped, path, contract, 16, 4, 0) == 0.
    yaml = policy_yaml(contract, 'oli', 16)
    assert set(yaml['observation']) == set(shapes)
    assert yaml['shaping_target'] == 'motion_first_frame'
