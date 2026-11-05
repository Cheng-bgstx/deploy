"""Bounded damping tool tests. ROS cases are restricted to loopback domain 171."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from scripts.pm01.bench_damping import ANKLES, DampingTrial, NATIVE_DAMPING, temperatures


def command(trial, now, **kw):
    args = dict(q=np.zeros(24), dq=np.zeros(24), tau=np.zeros(24),
                feedback_age=.001, mode='joint_bridge', mode_age=.01)
    args.update(kw)
    return trial.command(now, **args)


def test_requires_classic_ack_and_never_rearms_after_abort():
    trial = DampingTrial(.05, .2)
    assert command(trial, 1)[0] == 0
    assert command(trial, 1.51)[0] == 0
    assert 'not acknowledged' in trial.reason
    assert command(trial, 1.52, classic_ack=True)[0] == 0


def test_bounded_rl_and_confirmed_completion():
    trial = DampingTrial(.05, .2)
    command(trial, 1)
    parser, kd = command(trial, 1.5, classic_ack=True)
    assert parser == 1
    assert np.all(kd[ANKLES] == .05)
    assert command(trial, 1.6, rl_ack=True)[0] == 1
    assert command(trial, 1.71, rl_ack=True)[0] == 0
    assert trial.reason == 'completed'


@pytest.mark.parametrize('fault', [dict(feedback_age=float('nan')), dict(feedback_age=.03),
    dict(mode_age=-1), dict(q=np.zeros(23)), dict(dq=np.ones(24)), dict(tau=np.ones(24)*3)])
def test_bad_preflight_never_publishes(fault):
    assert command(DampingTrial(.05, .2), 1, **fault) is None


def test_failure_returns_to_classic_and_mode_change_stops_output():
    trial = DampingTrial(.05, .2)
    command(trial, 1)
    command(trial, 1.5, classic_ack=True)
    assert command(trial, 1.56)[0] == 0
    assert 'acknowledgement lost' in trial.reason
    assert command(trial, 1.57, mode='passive') is None


def test_old_firmware_temperature_trailer_is_not_a_motor():
    assert temperatures(NS(mos_temperature=[39.]*24+[2572131.],
                           motor_temperature=[35.]*24+[2572131.])) == 39.
    assert temperatures(NS(mos_temperature=[39.]*24, motor_temperature=[35.]*24)) is None


def test_native_damping_matches_installed_readback():
    assert NATIVE_DAMPING.shape == (24,)
    assert np.all(NATIVE_DAMPING > 0)


@pytest.mark.skipif(os.environ.get('PM01_RUN_BENCH_SMOKE') != '1', reason='Requires ROS schemas')
@pytest.mark.parametrize('case', ['read_only', 'idle', 'normal', 'lost_echo', 'worker_hang', 'hold_pose'])
def test_real_ros_loopback(case, monkeypatch, tmp_path):
    for name, value in {
        'ROS_DOMAIN_ID': '171', 'ROS_LOCALHOST_ONLY': '0', 'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp',
        'CYCLONEDDS_URI': '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/>'
                         '</Interfaces></General></Domain></CycloneDDS>',
    }.items():
        monkeypatch.setenv(name, value)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointState, JointCommand, MotionState, MotorDebug
    rclpy.init()
    node = rclpy.create_node('pm01_bench_test_fixture')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    pubs = {key: node.create_publisher(typ, topic, qos) for key, typ, topic in [
        ('joint', JointState, '/hardware/joint_state'), ('mode', MotionState, '/motion/motion_state'),
        ('echo', JointCommand, '/hardware/joint_command_feedback'), ('debug', MotorDebug, '/hardware/motor_debug')]}
    commands = []
    last = JointCommand()
    last.position = last.velocity = last.torque = last.stiffness = last.damping = last.feed_forward_torque = [0.]*24
    hung = False
    proc = None
    def receive(msg):
        nonlocal last, hung
        commands.append((time.monotonic(), msg))
        if case != 'lost_echo' or msg.parallel_parser_type == 0:
            last = msg
        if case == 'worker_hang' and msg.parallel_parser_type == 1 and not hung:
            children = Path(f'/proc/{proc.pid}/task/{proc.pid}/children').read_text().split()
            assert len(children) == 1
            os.kill(int(children[0]), signal.SIGSTOP)
            hung = True
    node.create_subscription(JointCommand, '/hardware/joint_command', receive, qos)
    def feedback():
        joint = JointState()
        joint.position = joint.velocity = joint.torque = [0.]*24
        pubs['joint'].publish(joint)
        pubs['echo'].publish(last)
    def slow_feedback():
        mode = MotionState()
        setattr(mode, 'current_motion_task_name' if hasattr(mode, 'current_motion_task_name') else 'current_motion_task',
                'idle' if case == 'idle' else 'joint_bridge')
        pubs['mode'].publish(mode)
        debug = MotorDebug()
        debug.mos_temperature = debug.motor_temperature = [30.]*24+[time.monotonic()]
        pubs['debug'].publish(debug)
    node.create_timer(.002, feedback)
    node.create_timer(.02, slow_feedback)
    output = tmp_path/'trial.jsonl'
    try:
        with (tmp_path/'console.txt').open('w+') as console:
            proc = subprocess.Popen([sys.executable, '-m', 'scripts.pm01.bench_damping',
                '--duration', '.2', '--output', str(output)] + ([] if case == 'read_only' else ['--execute'])
                + (['--ankle-kp', '37.701'] if case == 'hold_pose' else []),
                stdout=console, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+12
            while proc.poll() is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.001)
            assert proc.poll() is not None, 'supervisor did not exit'
            console.seek(0)
            log = console.read()
        if case in ('read_only', 'idle'):
            assert not commands, log
            assert proc.returncode == (0 if case == 'read_only' else 2), log
        else:
            assert any(m.parallel_parser_type == 1 for _, m in commands), log
            assert commands[-1][1].parallel_parser_type == 0, log
            assert list(commands[-1][1].damping) == NATIVE_DAMPING.tolist()
            for _, m in commands:
                expected = np.zeros(24)
                if case == 'hold_pose' and m.parallel_parser_type == 1:
                    expected[ANKLES] = 37.701
                assert np.allclose(m.stiffness, expected)
                assert not any(m.feed_forward_torque)
            report = json.loads(output.with_suffix('.supervisor.json').read_text())
            assert report['classic_fallback_confirmed'], log
            assert report['watchdog_triggered'] == (case == 'worker_hang'), log
            assert proc.returncode == (0 if case in ('normal', 'hold_pose') else 2), log
            first_rl = next(t for t, m in commands if m.parallel_parser_type == 1)
            last_rl = max(t for t, m in commands if m.parallel_parser_type == 1)
            assert last_rl-first_rl < .3
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        node.destroy_node()
        rclpy.shutdown()
