"""Frame-convention checks plus an optional local, SDK-free PICO/ZMQ integration."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from sim2real.config.robots import get_robot_cfg

ROOT = Path(__file__).resolve().parents[1]
PROFILES = sorted((ROOT / 'sim2real/teleop/configs').glob('xrobot_to_*.json'))
PROFILES = [p for p in PROFILES if p.stem.removeprefix('xrobot_to_') in {'oli', 'x2', 'q1'}]


@pytest.mark.parametrize('path', PROFILES, ids=lambda p: p.stem)
def test_xrobot_profile_is_valid_for_target_contract(path):
    cfg = get_robot_cfg(path.stem.removeprefix('xrobot_to_'))
    profile = json.loads(path.read_text())
    assert profile['robot_root_name'] == cfg.body_names[0]
    assert profile['human_root_name'] == 'Pelvis'
    for table in ('ik_match_table1', 'ik_match_table2'):
        for link, entry in profile[table].items():
            assert link in cfg.body_names
            assert entry[0] in profile['human_scale_table']
            assert np.linalg.norm(entry[4]) == pytest.approx(1, abs=1e-6)


def test_source_axis_conversion_preserves_world_targets():
    spec = importlib.util.spec_from_file_location('build_profile', ROOT / 'scripts/build_xrobot_profile.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rng = np.random.default_rng(42)
    src = R.random(random_state=rng)
    offset = R.random(random_state=rng)
    position_offset = rng.normal(size=3)
    smpl = {'human_root_name': 'pelvis', 'human_scale_table': {'pelvis': .8},
            **{t: {'base': ['pelvis', 100, 10, position_offset.tolist(),
                           offset.as_quat(scalar_first=True).tolist()]}
               for t in ('ik_match_table1', 'ik_match_table2')}}
    target = module.convert(smpl)['ik_match_table1']['base']
    # XRobot body axes are Ry(pi) relative to SMPL-X, in the same world frame.
    xrobot = src * R.from_euler('y', 180, degrees=True)
    actual = xrobot * R.from_quat(target[4], scalar_first=True)
    expected = src * offset
    np.testing.assert_allclose(actual.as_matrix(), expected.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(actual.apply(target[3]), expected.apply(position_offset), atol=1e-12)


@pytest.mark.parametrize('path', PROFILES, ids=lambda p: p.stem)
@pytest.mark.parametrize('human_height', [1.6, 1.8, 2.0])
def test_pico_retarget_fk_and_named_zmq_buffer(path, human_height, monkeypatch):
    pytest.importorskip('general_motion_retargeting')
    pytest.importorskip('cv2')
    import mujoco
    import zmq
    from general_motion_retargeting import XRobotStreamer
    from sim2real.teleop import pico_retarget_pub as pico
    from sim2real.rl_policy.controllers.pico import PicoController
    from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer

    cfg = get_robot_cfg(path.stem.removeprefix('xrobot_to_'))
    if not Path(cfg.resolve_mjcf_path()).is_file():
        pytest.skip('Prepare robot MJCF assets for the local integration check')
    profile = json.loads(path.read_text())
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
    data = mujoco.MjData(model)
    data.qpos[:7] = cfg.default_qpos[:7]
    for i, name in enumerate(cfg.joint_names):
        data.qpos[model.joint(name).qposadr[0]] = cfg.default_qpos[7 + i]
    mujoco.mj_kinematics(model, data)
    # Invert the calibrated task transform to obtain an exactly known XR fixture.
    frames = {}
    root = data.xpos[model.body(profile['robot_root_name']).id]
    root_scale = profile['human_scale_table']['Pelvis']
    for link, (human, _, _, pos, quat) in profile['ik_match_table1'].items():
        body = model.body(link).id
        target_rot = R.from_quat(data.xquat[body], scalar_first=True)
        target_pos = data.xpos[body] - target_rot.apply(pos)
        scale = profile['human_scale_table'][human]
        human_pos = (root / root_scale + (target_pos - root) / scale) * (
            human_height / profile['human_height_assumption'])
        human_rot = target_rot * R.from_quat(quat, scalar_first=True).inv()
        frames[human] = [human_pos, human_rot]
    if cfg.name == 'x2':
        # Real SDK packets have ankle and foot entries; the publisher constructs
        # the calibrated virtual FootMod frame from those two real measurements.
        for side in ('Left', 'Right'):
            pos, rot = frames.pop(f'{side}FootMod')
            frames[f'{side}_Ankle'] = [pos.copy(), rot]
            frames[f'{side}_Foot'] = [pos + np.array([.1, 0, 0]), rot]
    unity_to_world = R.from_euler('x', 90, degrees=True)

    class Streamer:
        body_joint_names = list(frames)
        x = False
        a = False
        b = False
        connected = True
        coordinate_transform_unity_data = XRobotStreamer.coordinate_transform_unity_data

        def get_controller_data(self):
            return {'LeftController': {'key_one': self.x},
                    'RightController': {'key_one': self.a, 'key_two': self.b},
                    'timestamp': time.time_ns()}

        def get_raw_body_data(self):
            if not self.connected:
                return None, None, None, None, 0
            poses = [np.r_[unity_to_world.inv().apply(pos),
                           (unity_to_world.inv() * rot).as_quat()]
                     for pos, rot in frames.values()]
            return poses, None, None, None, time.time_ns()

    monkeypatch.setattr(pico, 'XRobotStreamer', Streamer)
    worker = pico.LiveRetargetPublisher(pico.PublisherArgs(
        robot=cfg.name, viewer=False, actual_human_height=human_height,
        min_link_height_align_strategy='none', controller_bind='tcp://127.0.0.1:*',
        hand_grip_bind='tcp://127.0.0.1:*'))
    pub = zmq.Context.instance().socket(zmq.PUB)
    controller = PicoController(connect=worker._controller_sock.getsockopt(zmq.LAST_ENDPOINT).decode())
    port = pub.bind_to_random_port('tcp://127.0.0.1')
    buffer = RealtimeMotionBuffer(robot_cfg=cfg, future_steps=[0, 1, 2, 3, 4, 16],
                                  root_body_name=cfg.body_names[0],
                                  motion_zmq_connect=f'tcp://127.0.0.1:{port}')
    try:
        for human, scale in worker.retarget.human_scale_table.items():
            assert scale == pytest.approx(profile['human_scale_table'][human]
                                          * profile['human_height_assumption'] / human_height)
        paused = worker.sample_and_retarget()
        assert paused['paused'] and not paused['motion_first_frame']
        np.testing.assert_allclose(paused['joint_pos'], cfg.default_qpos[7:], atol=1e-6)
        worker.streamer.x = True
        payload = worker.sample_and_retarget()
        assert not payload['paused'] and payload['motion_first_frame']
        worker.streamer.x = False
        for step in range(10):
            if step == 3:
                frames['Left_Wrist'][0] += np.array([0, 0, .25])
            previous = worker.latest_qpos.copy()
            payload = worker.sample_and_retarget()
            if cfg.name in {'x2', 'q1', 't800', 'a3'}:
                indices = [model.joint(n).qposadr[0] for n in cfg.joint_names]
                assert np.max(np.abs(np.asarray(payload['qpos'])[indices] - previous[indices])) <= .40001
        assert not payload['motion_first_frame']
        q = np.asarray(payload['qpos'])
        assert np.isfinite(q).all()
        assert np.linalg.norm(q[3:7]) == pytest.approx(1, abs=1e-5)
        assert payload['joint_names'] == list(cfg.joint_names)
        for i, name in enumerate(cfg.joint_names):
            assert payload['joint_pos'][i] == pytest.approx(q[model.joint(name).qposadr[0]])
        deadline = time.monotonic() + 3
        while buffer.latest_timestamp_ns is None and time.monotonic() < deadline:
            payload['publish_t_ns'] = time.time_ns()
            pub.send_json(payload)
            time.sleep(.02)
        assert buffer.latest_timestamp_ns is not None
        obs = buffer.get_obs()
        np.testing.assert_allclose(obs.joint_pos[0, 0], payload['joint_pos'], atol=1e-5)
        if cfg.name in {'q1', 't800', 'a3'}:
            _run_streamed_proteus_policy(cfg, buffer, obs, q)
        assert np.isfinite(obs.body_pos_w).all()
        worker.streamer.connected = False
        assert worker.sample_and_retarget() is None
        worker.streamer.connected = True
        assert worker.sample_and_retarget()['motion_first_frame']
        worker.streamer.x = True
        assert worker.sample_and_retarget()['paused']
        def buttons(a, b, expected):
            worker.streamer.a, worker.streamer.b = a, b
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                worker._poll_pause_toggle()
                mode = controller.get_control_mode()
                if mode is not None:
                    assert mode == expected
                    return
                if expected is None and controller._pico_msg.A == a and controller._pico_msg.B == b:
                    return
                time.sleep(.01)
            pytest.fail(f'PICO buttons {a, b} were not delivered')
        buttons(True, False, 'init')
        buttons(False, False, None)
        buttons(True, True, 'policy')
        buttons(False, False, None)
        buttons(False, True, 'zero')
    finally:
        controller.close()
        worker.close()
        buffer._motion_stream_stop.set()
        buffer._motion_stream_thread.join(timeout=1)
        buffer._motion_stream_socket.close(0)
        pub.close(0)


def _run_streamed_proteus_policy(cfg, buffer, motion, qpos):
    # Real publisher -> ZMQ receiver -> semantic observations -> complete ONNX.
    # XR hardware is replaced by the known-pose fixture above; the rest is real.
    from types import SimpleNamespace as NS
    import yaml
    import mujoco
    import subprocess
    from sim2real.rl_policy.observations import Observation
    contracts = {'q1': 'primebot_q1', 't800': 'engineai_t800', 'a3': 'agibot_a3'}
    folder = ROOT / 'checkpoints/proteus/proteus_wamseq_62' / contracts[cfg.name]
    if not (folder / 'policy.onnx').is_file():
        pytest.skip('Export the complete ONNX for the streamed policy integration')
    policy = yaml.safe_load((folder / 'policy.yaml').read_text())
    model = mujoco.MjModel.from_xml_path(str(cfg.resolve_mjcf_path()))
    indices = [model.joint(n).qposadr[0] for n in cfg.joint_names]
    env = NS(robot_cfg=cfg, policy_config=policy, motion_data=motion,
        motion_body_names=buffer.body_names, motion_future_steps=buffer.future_steps,
        state_processor=NS(joint_names=list(cfg.joint_names), root_pos_w=qpos[:3],
            root_quat_w=qpos[3:7], root_ang_vel_b=np.zeros(3), joint_pos=qpos[indices],
            joint_vel=np.zeros(len(indices))))
    groups = {key: [Observation.resolve(value['_target_'])(env=env) for value in terms.values()]
              for key, terms in policy['observation'].items()}
    # Keep ONNX Runtime in the root project; the teleop environment runs only IK.
    command = [str(ROOT / '.venv/bin/python'), '-u', '-c', r"""
import sys,json,numpy as np,onnxruntime as ort
options=ort.SessionOptions();options.intra_op_num_threads=1
session=ort.InferenceSession(sys.argv[1],options,providers=['CPUExecutionProvider'])
for line in sys.stdin:
    feed={k:np.asarray(v,dtype=np.float32) for k,v in json.loads(line).items()}
    print(json.dumps([v.tolist() for v in session.run(['action','next_slot_action'],feed)]),flush=True)
""", str(folder / 'policy.onnx')]
    actor = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        for terms in groups.values():
            for term in terms:
                term.reset()
        previous = np.zeros(31, np.float32)
        for step in range(12):
            for terms in groups.values():
                for term in terms:
                    term.update({'slot_action': previous})
            feed = {key: np.concatenate([term.compute() for term in terms], axis=-1).astype(np.float32)
                    for key, terms in groups.items()}
            actor.stdin.write(json.dumps({k: v.tolist() for k, v in feed.items()}) + '\n')
            actor.stdin.flush()
            action, previous = [np.asarray(v, np.float32) for v in json.loads(actor.stdout.readline())]
            assert action.shape == (len(cfg.joint_names),) and np.isfinite(action).all()
            assert previous.shape == (31,) and np.isfinite(previous).all()
            assert feed['history_valid'].sum() == min(step + 1, 10)
    finally:
        actor.stdin.close()
        actor.wait(timeout=10)
        actor.stdout.close()
    assert actor.returncode == 0
