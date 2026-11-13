"""Opt-in full launcher regression with real ROS/ONNX and fake loopback feedback.

Run after preparing PM01 assets and sourcing scripts/pm01/env.sh:
PM01_RUN_LAUNCH_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync \
    python -m pytest tests/test_pm01_launch.py -q
This checks software integration, not physical robot stability.
"""
import os
from pathlib import Path
import pty
import signal
import subprocess
import time

import numpy as np
import pytest
import yaml


@pytest.mark.skipif(os.environ.get('PM01_RUN_LAUNCH_SMOKE') != '1',
                    reason='Requires prepared PM01 assets and matching ROS messages')
def test_actual_launcher_keyboard_inference_and_recording(monkeypatch, tmp_path):
    # Set both the parent node and launcher environment before initializing DDS.
    # env.sh honors PM01_ROS_DOMAIN_ID and this explicit loopback-only URI.
    for name, value in {
        'PM01_ROS_DOMAIN_ID': '171', 'ROS_DOMAIN_ID': '171',
        'PM01_NETWORK_INTERFACE': 'lo', 'ROS_LOCALHOST_ONLY': '0',
        'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
        'CYCLONEDDS_URI': '<CycloneDDS><Domain><General><Interfaces>'
                         '<NetworkInterface name="lo"/></Interfaces></General></Domain></CycloneDDS>',
        'PYTHONUNBUFFERED': '1',
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('PROTEUS_POLICY_CONFIG', raising=False)

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointState, ImuInfo, MotionState, JointCommand
    from sim2real.config.robots import get_robot_cfg

    root = Path(__file__).resolve().parents[1]
    record_path = tmp_path / 'tracking.npz'
    log_path = tmp_path / 'launcher.log'
    cfg = get_robot_cfg('pm01')
    policy_cfg = yaml.safe_load((root / 'checkpoints/proteus/wsq61/engineai_pm01/policy.yaml').read_text())
    q = [float(policy_cfg['default_joint_pos'][name]) for name in cfg.joint_names]
    rclpy.init()
    node = rclpy.create_node('pm01_loopback_launch_fixture')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    joint_pub = node.create_publisher(JointState, '/hardware/joint_state', qos)
    imu_pub = node.create_publisher(ImuInfo, '/hardware/imu_info', qos)
    mode_pub = node.create_publisher(MotionState, '/motion/motion_state', qos)
    commands = []

    def command(msg):
        commands.append(msg)
        # Instant position follower, deliberately not a dynamics simulation.
        if any(msg.stiffness):
            q[:] = list(msg.position)

    node.create_subscription(JointCommand, '/hardware/joint_command', command, qos)
    mode = MotionState()
    field = 'current_motion_task_name' if hasattr(mode, 'current_motion_task_name') else 'current_motion_task'
    setattr(mode, field, 'joint_bridge')

    def feedback():
        state = JointState()  # Installed firmware leaves sensor stamps at zero.
        state.position = q
        state.velocity = [0.] * 24
        state.torque = [0.] * 24
        imu = ImuInfo()
        imu.quaternion.w = 1.
        joint_pub.publish(state)
        imu_pub.publish(imu)

    node.create_timer(.002, feedback)
    node.create_timer(.2, lambda: mode_pub.publish(mode))
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    proc = None
    output = bytearray()

    def pump(seconds, until=None, allow_exit=False):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.002)
            try:
                output.extend(os.read(master, 65536))
            except (BlockingIOError, OSError):
                pass
            if until is not None and until():
                return
            if not allow_exit:
                assert proc.poll() is None, output.decode(errors='replace')
        if until is not None:
            pytest.fail('Launcher timed out; output:\n' + output.decode(errors='replace'))

    try:
        proc = subprocess.Popen(
            ['bash', 'scripts/pm01/start_proteus.sh', '--record',
             '--record-output', str(record_path)],
            cwd=root, stdin=slave, stdout=slave, stderr=slave,
            start_new_session=True,
        )
        os.close(slave)
        slave = None
        pump(60, until=lambda: b'total_inference_cnt: 100' in output)
        assert b'"ready": true' in output
        os.write(master, b'i')
        pump(3, until=lambda: b'Control mode set to init' in output)
        pump(11)  # Exercise all 500 shaping steps at the real 50 Hz rate.
        assert any(any(msg.stiffness) for msg in commands)
        os.write(master, b']')
        pump(3, until=lambda: b'Control mode set to policy' in output)
        pump(2)
        os.write(master, b' ')
        pump(3, until=lambda: b'Paused state toggled to False' in output)
        pump(2)
        os.write(master, b'o')
        pump(3, until=lambda: b'Control mode set to zero' in output)
        pump(.5)
        assert list(commands[-1].stiffness) == [0.] * 24
        assert list(commands[-1].damping) == [2.] * 24
        os.killpg(proc.pid, signal.SIGINT)
        pump(15, until=lambda: proc.poll() is not None, allow_exit=True)
        assert proc.returncode == 0, output.decode(errors='replace')
        assert b'Traceback' not in output
        assert b'Error in policy inference' not in output
        assert b'terminate called' not in output

        data = np.load(record_path, allow_pickle=True)['data'].item()
        assert data['metadata']['frame_count'] > 750
        for name in ('action', 'q_target', 'cmd_q'):
            assert np.isfinite(data['policy'][name]).all(), name
        modes = data['runtime']['control_mode']
        paused = data['runtime']['paused']
        assert set(modes) == {'zero', 'init', 'policy'}
        assert np.sum(modes == 'init') >= 500
        assert np.sum((modes == 'policy') & paused) >= 50
        assert np.sum((modes == 'policy') & ~paused) >= 50
        assert np.ptp(data['runtime']['motion_t'][(modes == 'policy') & ~paused]) > 0
        low = np.array([cfg.joint_pos_lower_limit[n] for n in cfg.joint_names])
        high = np.array([cfg.joint_pos_upper_limit[n] for n in cfg.joint_names])
        for msg in commands:
            assert len(msg.position) == 24 and msg.parallel_parser_type == 0
            if any(msg.stiffness):
                assert np.all(np.asarray(msg.position) >= low - 1e-6)
                assert np.all(np.asarray(msg.position) <= high + 1e-6)
        print(f'{data["metadata"]["frame_count"]} recorded frames, '
              f'{len(commands)} commands; log={log_path}; record={record_path}')
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        log_path.write_bytes(output)
        os.close(master)
        if slave is not None:
            os.close(slave)
        node.destroy_node()
        rclpy.shutdown()
