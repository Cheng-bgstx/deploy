import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from sim2real.config.robots import ET1_CFG, G1_CFG, get_robot_cfg
from sim2real.config.robots.base import get_unitree_dds_family
from sim2real.utils.common import UNITREE_LEGGED_CONST

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTEUS_ROOT = REPO_ROOT.parent / "Proteus"
EMPTY_ET1_SLOTS = (14, 20, 21, 27, 28)
HG_MOTORS = 35


def _load_script(name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module while it executes
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------- robot config


def test_et1_config_is_registered_as_a_unitree_hg_robot() -> None:
    assert get_robot_cfg("galbot_et1") is ET1_CFG
    assert get_robot_cfg("ET1") is ET1_CFG
    assert get_unitree_dds_family(ET1_CFG.name) == "hg"
    assert len(ET1_CFG.joint_names) == 26
    assert len(ET1_CFG.default_qpos) == ET1_CFG.qpos_size
    for table in (ET1_CFG.joint_pos_lower_limit, ET1_CFG.joint_effort_limit, ET1_CFG.joint_armature):
        assert set(table) == set(ET1_CFG.joint_names)


def test_et1_motor_slots_skip_the_empty_hg_slots() -> None:
    indices = ET1_CFG.resolve_unitree_motor_indices()
    assert len(indices) == len(ET1_CFG.joint_names)
    assert sorted(indices) == [i for i in range(31) if i not in EMPTY_ET1_SLOTS]
    by_joint = dict(zip(ET1_CFG.joint_names, indices))
    assert by_joint["waist_yaw_joint"] == 13
    assert by_joint["left_shoulder_pitch_joint"] == 15
    assert by_joint["right_shoulder_pitch_joint"] == 22
    assert (by_joint["head_pitch_joint"], by_joint["head_yaw_joint"]) == (29, 30)
    assert ET1_CFG.unitree_mode_machine is None


def test_motor_slots_default_to_identity_and_reject_bad_maps() -> None:
    assert G1_CFG.resolve_unitree_motor_indices() == tuple(range(len(G1_CFG.joint_names)))
    assert G1_CFG.unitree_mode_machine == 5
    with pytest.raises(ValueError, match="one distinct slot"):
        ET1_CFG.__class__(**{**ET1_CFG.__dict__, "unitree_motor_indices": (0,) * 26}).resolve_unitree_motor_indices()


@pytest.mark.skipif(not ET1_CFG.mjcf_path.is_file(), reason="third_party/galbot_et1 not present")
def test_et1_config_matches_its_mjcf() -> None:
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ET1_CFG.resolve_mjcf_path()))
    assert [model.joint(i).name for i in range(1, model.njnt)] == list(ET1_CFG.joint_names)
    assert [model.body(i).name for i in range(1, model.nbody)] == list(ET1_CFG.body_names)
    for name in ET1_CFG.joint_names:
        lower, upper = model.jnt_range[model.joint(name).id]
        assert (lower, upper) == pytest.approx((ET1_CFG.joint_pos_lower_limit[name], ET1_CFG.joint_pos_upper_limit[name]))
        actuator = next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == model.joint(name).id)
        assert model.actuator_ctrlrange[actuator, 1] == pytest.approx(ET1_CFG.joint_effort_limit[name])
    assert model.site("imu").bodyid == model.body("pelvis_link").id  # Proteus reads the root (pelvis) IMU


# ---------------------------------------------------------------------------- MJCF frames

ARM_MJCF = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="left_elbow_link" pos="0.1 0.2 -0.1" quat="0.707107 0 -0.707107 0">
        <inertial pos="-0.01 -0.02 -0.06" quat="0.99 -0.07 0.01 -0.01" mass="0.7" diaginertia="0.0008 0.0007 0.0005"/>
        <joint name="elbow" pos="0 0 0" axis="0 1 0" range="-0.4 2.2"/>
        <geom type="capsule" size="0.02 0.05" pos="0 0 -0.05"/>
        <body name="left_wrist_roll_link" pos="0.003 -0.03 -0.1">
          <inertial pos="0.002 -0.0005 -0.07" quat="0.86 -0.04 0.001 0.5" mass="0.3" diaginertia="0.0006 0.0006 0.0001"/>
          <joint name="wrist" pos="0 0 0" axis="0 0 -1" range="-1.9 1.9"/>
          <site name="palm" pos="0.003 -0.015 -0.1"/>
          <geom type="box" size="0.02 0.03 0.04" pos="0 0 -0.08"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_reframing_keeps_the_physics_and_turns_the_frames(tmp_path: Path) -> None:
    prepare = _load_script("et1_prepare_mjcf", "scripts/et1/prepare_mjcf.py")
    vendor, prepared = tmp_path / "vendor.xml", tmp_path / "prepared.xml"
    vendor.write_text(ARM_MJCF)
    bodies = ("left_elbow_link", "left_wrist_roll_link")
    prepare.reframe_mjcf(vendor, prepared, bodies)
    errors = prepare.check_equivalent(vendor, prepared, bodies)
    assert max(errors.values()) < 1.0e-9

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(prepared))
    # In the Unitree frames the wrist rolls about +x (the forearm) and the elbow keeps its y axis.
    assert model.jnt_axis[model.joint("wrist").id] == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-9)
    assert model.jnt_axis[model.joint("elbow").id] == pytest.approx([0.0, 1.0, 0.0], abs=1.0e-9)


# ---------------------------------------------------------------------------- motions


def _write_npz_motion(path: Path, *, quat_scale: float = 1.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    frames = 4
    arrays = {
        "fps": np.int32(50),
        "joint_names": np.array(["b_joint", "a_joint"]),
        "body_names": np.array(["pelvis_link", "leg"]),
        "joint_pos": rng.normal(size=(frames, 2)),
        "body_pos_w": rng.normal(size=(frames, 2, 3)),
        "body_quat_w": np.tile([2.0 * quat_scale, 0.0, 0.0, 0.0], (frames, 2, 1)),
    }
    np.savez(path, **arrays)
    return arrays


def test_motion_sources_parse_names_and_both_layouts(tmp_path: Path) -> None:
    convert = _load_script("et1_convert_motions", "scripts/et1/convert_motions.py")
    assert convert.parse_source("/a/2-太极.npz=taichi") == ("taichi", Path("/a/2-太极.npz"))
    assert convert.parse_source("/a/getup_000") == ("getup_000", Path("/a/getup_000"))
    with pytest.raises(ValueError, match="=<name>"):
        convert.parse_source("/a/2-太极.npz")

    arrays = _write_npz_motion(tmp_path / "clip.npz")
    from_npz = convert.load_source(tmp_path / "clip.npz")
    directory = tmp_path / "clip_dir"
    directory.mkdir()
    (directory / "meta.json").write_text(json.dumps({"fps": 50.0, "joint_names": ["b_joint", "a_joint"],
                                                      "body_names": ["pelvis_link", "leg"]}))  # fmt: skip
    for key in ("joint_pos", "body_pos_w", "body_quat_w"):
        np.save(directory / f"{key}.npy", arrays[key])
    from_dir = convert.load_source(directory)
    for motion in (from_npz, from_dir):
        qpos = convert.motion_to_qpos(motion, [*convert.ROOT_QPOS_NAMES, "a_joint", "b_joint"])
        assert qpos[:, :3] == pytest.approx(arrays["body_pos_w"][:, 0])
        assert qpos[:, 3:7] == pytest.approx(np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)))  # normalised WXYZ
        assert qpos[:, 7:] == pytest.approx(arrays["joint_pos"][:, [1, 0]])  # by name

    motion = convert.load_source(tmp_path / "clip.npz")
    with pytest.raises(ValueError, match="missing joints"):
        convert.motion_to_qpos(motion, [*convert.ROOT_QPOS_NAMES, "c_joint"])


@pytest.mark.skipif(not ET1_CFG.mjcf_path.is_file(), reason="third_party/galbot_et1 not present")
def test_motion_conversion_checks_the_forward_kinematics(tmp_path: Path) -> None:
    import mujoco

    convert = _load_script("et1_convert_motions", "scripts/et1/convert_motions.py")
    model = mujoco.MjModel.from_xml_path(str(ET1_CFG.resolve_mjcf_path()))
    data = mujoco.MjData(model)
    frames, body_names = 3, ["pelvis_link", "left_wrist_roll_link"]
    joint_pos = np.zeros((frames, 26))
    joint_pos[:, list(ET1_CFG.joint_names).index("left_elbow_joint")] = [0.2, 0.6, 1.0]
    body_pos, body_quat = np.zeros((frames, 2, 3)), np.zeros((frames, 2, 4))
    for t in range(frames):
        data.qpos[:] = [0.0, 0.0, 0.7, 1.0, 0.0, 0.0, 0.0, *joint_pos[t]]
        mujoco.mj_kinematics(model, data)
        for i, name in enumerate(body_names):
            body_pos[t, i], body_quat[t, i] = data.xpos[model.body(name).id], data.xquat[model.body(name).id]
    np.savez(tmp_path / "good.npz", fps=50, joint_names=np.array(ET1_CFG.joint_names), body_names=np.array(body_names),
             joint_pos=joint_pos, body_pos_w=body_pos, body_quat_w=body_quat)  # fmt: skip
    written = convert.convert_motions({"good": tmp_path / "good.npz"}, tmp_path / "dataset")
    assert np.load(written["good"])["qpos"].shape == (frames, 33)
    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    assert manifest["num_motions"] == 1 and manifest["timestep"] == pytest.approx(0.02)

    turned = body_quat.copy()  # q * (+90 deg about y): the wrist in another model's frame
    turned[:, 1] = [[q[0] - q[2], q[1] - q[3], q[0] + q[2], q[1] + q[3]] for q in turned[:, 1] / np.sqrt(2.0)]
    np.savez(tmp_path / "bad.npz", fps=50, joint_names=np.array(ET1_CFG.joint_names), body_names=np.array(body_names),
             joint_pos=joint_pos, body_pos_w=body_pos, body_quat_w=turned)  # fmt: skip
    with pytest.raises(ValueError, match="does not reproduce"):
        convert.convert_motions({"bad": tmp_path / "bad.npz"}, tmp_path / "dataset")


# ---------------------------------------------------------------------------- Proteus contract


@pytest.mark.skipif(
    not (ET1_CFG.mjcf_path.is_file() and (PROTEUS_ROOT / "ScaleTrack" / "checkpoints" / "proteus_60").is_dir()),
    reason="needs third_party/galbot_et1 and the Proteus repo next to sim2real",
)
def test_et1_contract_follows_the_r1_template() -> None:
    builder = _load_script("et1_build_proteus_contract", "scripts/build_proteus_contract.py")
    contract, description, _ = builder.build_contract("et1", "galbot_et1", template_name="unitree_r1",
                                                      proteus_root=PROTEUS_ROOT)  # fmt: skip
    r1 = json.loads((PROTEUS_ROOT / "ScaleTrack/checkpoints/proteus_60/unitree_r1.json").read_text())
    r1_joints = {j["name"]: j for j in r1["joints"]}
    joints = {j["name"]: j for j in contract["joints"]}
    assert set(joints) == set(ET1_CFG.joint_names)
    for name, joint in joints.items():
        if name.startswith("head_"):
            assert joint["slot"] == -1 and joint["action_scale"] == 0.0
            assert (joint["kp"], joint["kd"]) == pytest.approx(builder.DEFAULT_LOCKED_GAINS)
            continue
        assert (joint["slot"], joint["sign"]) == (r1_joints[name]["slot"], r1_joints[name]["sign"])
        assert joint["default"] == pytest.approx(r1_joints[name]["default"], abs=1.0e-4)  # 4 decimals, as Proteus
        assert joint["armature"] == pytest.approx(ET1_CFG.joint_armature[name], rel=1.0e-4)
        # Gains are the G1 ones of the slot rescaled by M g h: the R1's times the mass ratio.
        assert joint["kp"] == pytest.approx(r1_joints[name]["kp"] * description["mass"] / 28.931929, rel=2.0e-3)
    assert contract["body_names"][0] == "pelvis_link"
    assert [b.replace("toe", "ankle_roll") for b in r1["body_names"]] == contract["body_names"]
    assert len(contract["morph"]) == len(r1["morph"]) == 722
    assert contract["length_scale"] == pytest.approx(r1["length_scale"], abs=1.0e-4)
    for key in ("control_dt", "num_slots", "history", "future_offsets", "far_offset_range", "actor"):
        assert contract[key] == r1[key]


# ---------------------------------------------------------------------------- Unitree HG bridge


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _hg_state(tick: int, *, keys: int = 0, mode_machine: int = 7) -> SimpleNamespace:
    import struct

    payload = bytearray(40)
    struct.pack_into("<H", payload, 2, keys)
    # Slot s reports q = s / 100 (inside every ET1 limit), so the joint order shows which slot was read.
    motors = [SimpleNamespace(q=0.01 * s, dq=0.0, tau_est=float(s)) for s in range(HG_MOTORS)]
    imu = SimpleNamespace(quaternion=[1.0, 0.0, 0.0, 0.0], gyroscope=[0.0, 0.0, 0.0])
    return SimpleNamespace(tick=tick, mode_machine=mode_machine, wireless_remote=payload, imu_state=imu,
                           motor_state=motors)  # fmt: skip


def test_bridge_maps_et1_joints_to_hg_slots_and_keeps_the_lowstate_mode_machine(monkeypatch) -> None:
    import time
    from dataclasses import replace

    from scripts.g1.real_bridge import RealBridge
    from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO

    cfg = replace(ET1_CFG, low_state_port=_free_port(), low_cmd_port=_free_port(),
                  low_state_bind_addr="127.0.0.1", low_cmd_bind_addr="127.0.0.1")  # fmt: skip
    writes = []

    def init_channels(self):
        self.low_cmd = SimpleNamespace(mode_machine=0, motor_cmd=[
            SimpleNamespace(q=UNITREE_LEGGED_CONST["PosStopF"], dq=0.0, tau=0.0, kp=0.0, kd=0.0) for _ in range(HG_MOTORS)
        ])  # fmt: skip
        self.crc = SimpleNamespace(Crc=lambda _cmd: 0)

        def write(cmd):
            writes.append((cmd.mode_machine, [(m.q, m.kp) for m in cmd.motor_cmd]))
            return True

        self.low_cmd_unitree_pub = SimpleNamespace(Write=write)

    def release(self):
        with self.lock:
            self._takeover_result = True

    monkeypatch.setattr(RealBridge, "_init_channels", init_channels)
    monkeypatch.setattr(RealBridge, "_release_mode", release)
    bridge = RealBridge(cfg)
    io = ZMQRobotIO(cfg)
    indices = np.asarray(ET1_CFG.resolve_unitree_motor_indices())
    measured = 0.01 * indices
    tick = 0

    def pump(keys=0, mode="zero", generation=0, duration=0.04, mode_machine=7):
        nonlocal tick
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            tick += 1
            bridge._on_low_state(_hg_state(tick, keys=keys, mode_machine=mode_machine))
            io.set_control_mode(mode, generation)
            q = measured + 0.001 * np.arange(26) * (mode != "zero")
            io.write_command(q, np.zeros(26), np.zeros(26), np.full(26, 20.0), np.ones(26))
            bridge._step()
            time.sleep(0.005)

    try:
        pump(duration=0.3)
        assert bridge.mode_machine == 7 and not writes  # standby: learnt, nothing written
        state = io.read_state()
        assert state.qpos[7:] == pytest.approx(measured)  # joint i reads its own HG slot
        pump(1 << 8, mode="init", generation=1, duration=0.08)
        assert bridge.guard.active and writes
        mode_machine, motors = writes[-1]
        assert mode_machine == 7
        for joint, slot in enumerate(indices):
            assert motors[slot] == pytest.approx((0.01 * slot + 0.001 * joint, 20.0))
        for slot in (*EMPTY_ET1_SLOTS, *range(31, HG_MOTORS)):
            assert motors[slot] == (UNITREE_LEGGED_CONST["PosStopF"], 0.0)  # unused motors stay passive
        pump(mode="policy", generation=2, mode_machine=5)  # another robot model on the bus
        assert bridge.guard.latched and bridge.guard.fault
        assert "mode_machine" in bridge.guard.reason
    finally:
        io.close()
        bridge.low_state_zmq_pub.close(linger=0)
        bridge.low_cmd_zmq_sub.close(linger=0)


def test_bridge_keeps_the_g1_slots_and_mode_machine() -> None:
    assert G1_CFG.resolve_unitree_motor_indices() == tuple(range(29))
    assert G1_CFG.unitree_mode_machine == 5
