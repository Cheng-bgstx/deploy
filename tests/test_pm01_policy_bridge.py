import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

from scripts.pm01.policy_bridge import PolicyGate, REQUEST_TOPIC, STATUS_TOPIC, WALK_START_WAIT_S
from scripts.pm01.bench_damping import ANKLES, NATIVE_DAMPING


def config():
    kd = np.ones(24)
    kd[ANKLES] = .3
    return dict(kp=[10.]*24, kd=kd.tolist(), lower=[-3.]*24, upper=[3.]*24)


def values(c, target=None):
    return [np.zeros(24) if target is None else target, np.zeros(24), np.zeros(24), c['kp'], c['kd']]


def armed():
    c = config()
    gate = PolicyGate(c, 'session')
    gate.accept(10., 100., 'zero', [], np.zeros(24), np.zeros(24))
    assert gate.accept(10.02, 100.02, 'init', values(c), np.zeros(24), np.zeros(24))
    return gate, c


def test_deadline_latches_and_late_packets_cannot_restart():
    gate, c = armed()
    gate.observe(10.13, np.zeros(24), np.zeros(24), np.zeros(24), .001)
    assert gate.reason == 'PC command timeout'
    assert not gate.accept(10.14, 100.14, 'init', values(c), np.zeros(24), np.zeros(24))


@pytest.mark.parametrize('kind', ['replay', 'nan', 'gain', 'early_policy'])
def test_invalid_request_latches(kind):
    gate, c = armed()
    now, source, mode, v, dq = 10.04, 100.04, 'init', values(c), np.zeros(24)
    if kind == 'replay': source = 100.02
    if kind == 'nan': source = float('nan')
    if kind == 'gain': v[4] = np.ones(24)*2
    if kind == 'early_policy': mode = 'policy'
    assert not gate.accept(now, source, mode, v, np.zeros(24), dq)
    assert gate.reason


def test_measured_speed_beyond_training_limit_latches():
    gate, c = armed()
    dq = np.zeros(24)
    dq[4] = 40.
    gate.observe(10.04, np.zeros(24), dq, np.zeros(24), .001)
    assert gate.reason == 'joint speed limit'


def test_expired_packet_is_discarded_without_unloading_but_cannot_extend_deadline():
    gate,c=armed()
    previous=gate.command[0].copy()
    assert not gate.accept(10.10,100.04,'init',values(c),np.zeros(24),np.zeros(24))
    assert gate.reason is None
    np.testing.assert_array_equal(gate.command[0],previous)
    gate.observe(10.13,np.zeros(24),np.zeros(24),np.zeros(24),.001)
    assert gate.reason=='PC command timeout'


def test_short_walk_feedback_gap_holds_target_and_rejects_new_requests_until_fresh():
    gate,c=walk_holding()
    previous=[v.copy() for v in gate.command]
    for age in (.051,.06,.08,.096):
        gate.observe(20.22+age,np.zeros(24),np.zeros(24),np.zeros(24),age)
        assert gate.reason is None and gate.feedback_hold
        assert not gate.accept(20.22+age,110.22+age,'policy',values(c,np.ones(24)),np.zeros(24),np.zeros(24))
        for a,b in zip(previous,gate.command):np.testing.assert_array_equal(a,b)
    gate.observe(20.305,np.zeros(24),np.zeros(24),np.zeros(24),.001)
    assert not gate.feedback_hold and gate.reason is None
    assert gate.feedback_epoch == 1
    assert gate.accept(20.306,110.306,'policy',values(c),np.zeros(24),np.zeros(24))
    gate.observe(20.42,np.zeros(24),np.zeros(24),np.zeros(24),.114)
    assert gate.reason=='joint feedback timeout'


def test_recovery_target_maps_sdk_ack_back_to_robot_order():
    from scripts.pm01.trial_policy import RequestIO
    io=RequestIO.__new__(RequestIO)
    io.indices=list(reversed(range(24)))
    io.status={'last_target_sdk':list(range(24))}
    np.testing.assert_array_equal(io.resume_target(),list(reversed(range(24))))


def test_feedback_gap_during_play_resumes_without_requiring_stationary_pose():
    gate,c=walk_holding()
    for i in range(1,251):
        t=20.22+i*.02
        gate.accept(t,t+90,'policy',values(c),np.zeros(24),np.zeros(24))
        gate.observe(t,np.zeros(24),np.zeros(24),np.zeros(24),0.)
    assert gate.accept(25.24,115.24,'play',values(c),np.zeros(24),np.zeros(24))
    gate.observe(25.27,np.zeros(24),np.zeros(24),np.zeros(24),.06)
    assert gate.feedback_hold and gate.reason is None
    gate.observe(25.30,np.zeros(24),np.zeros(24),np.zeros(24),.001)
    assert gate.accept(25.302,115.302,'play',values(c),np.zeros(24),np.zeros(24))


@pytest.mark.parametrize('hold_seconds', [1., 3.])
def test_shapes_before_policy_then_stops_at_deadline(hold_seconds):
    gate, c = armed()
    gate.hold_seconds = hold_seconds
    for step in range(2, 511):
        assert gate.accept(10+step*.02, 100+step*.02, 'init', values(c), np.zeros(24), np.zeros(24))
    assert gate.accept(20.22, 110.22, 'policy', values(c), np.zeros(24), np.zeros(24))
    for step in range(1, int(hold_seconds/.02)+1):
        gate.accept(20.22+step*.02, 110.22+step*.02, 'policy', values(c), np.zeros(24), np.zeros(24))
        gate.observe(20.22+step*.02-.001, np.zeros(24), np.zeros(24), np.zeros(24), .001)
        assert gate.reason is None
    gate.observe(20.23+hold_seconds, np.zeros(24), np.zeros(24), np.zeros(24), .001)
    assert gate.reason == 'completed first-frame hold'


@pytest.mark.parametrize('duration', [0., -1., 2., 10., float('inf'), float('nan')])
def test_rejects_unbounded_or_unsupported_policy_duration(duration):
    with pytest.raises(ValueError):
        PolicyGate(config(), 'session', hold_seconds=duration)


def test_walk_shaping_only_prohibits_policy():
    with pytest.raises(ValueError):
        PolicyGate(config(), 'session', walk_shaping=True)
    gate = PolicyGate(config(), 'session', shaping_only=10., walk_shaping=True)
    np.testing.assert_array_equal(gate.torque_limit, PolicyGate(config(), 'default').torque_limit)
    gate.accept(10., 100., 'zero', [], np.zeros(24), np.zeros(24))
    assert not gate.accept(10.02, 100.02, 'policy', values(config()), np.zeros(24), np.zeros(24))
    assert gate.reason == 'policy disabled during shaping probe'




def walk_holding():
    c = config()
    gate = PolicyGate(c, 'walk', walk_once=True)
    gate.accept(10., 100., 'zero', [], np.zeros(24), np.zeros(24))
    for step in range(1, 510):
        assert gate.accept(10.+step*.02, 100.+step*.02, 'init', values(c), np.zeros(24), np.zeros(24))
    assert gate.accept(20.20, 110.20, 'hold', values(c), np.zeros(24), np.zeros(24))
    assert gate.accept(20.22, 110.22, 'policy', values(c), np.zeros(24), np.zeros(24))
    return gate, c


def test_walk_requires_hold_and_stops_after_one_playback():
    gate, c = walk_holding()
    assert not gate.accept(20.24, 110.24, 'play', values(c), np.zeros(24), np.zeros(24))
    assert gate.reason == 'invalid walk phase transition'
    gate, c = walk_holding()
    for i in range(1, 251):
        gate.accept(20.22+i*.02, 110.22+i*.02, 'policy', values(c), np.zeros(24), np.zeros(24))
        gate.observe(20.22+i*.02, np.zeros(24), np.zeros(24), np.zeros(24), 0.)
    assert gate.hold_stable
    assert gate.accept(25.24, 115.24, 'play', values(c), np.zeros(24), np.zeros(24))
    for i in range(1, 426):
        t = 25.24+i*.02
        gate.accept(t, t+90, 'play', values(c), np.zeros(24), np.zeros(24))
        gate.observe(t, np.zeros(24), np.zeros(24), np.zeros(24), 0.)
    gate.observe(33.75, np.zeros(24), np.zeros(24), np.zeros(24), 0.)
    assert gate.reason == 'completed walk'
    assert not gate.accept(33.76, 123.76, 'init', values(c), np.zeros(24), np.zeros(24))


def test_walk_waiting_is_bounded_and_torque_guard_remains():
    gate, c = walk_holding()
    for elapsed in (1., 45., WALK_START_WAIT_S-.01, WALK_START_WAIT_S+.02):
        t = 20.22+elapsed
        gate.accept(t, t+90, 'policy', values(c), np.zeros(24), np.zeros(24))
        gate.observe(t, np.zeros(24), np.zeros(24), np.zeros(24), 0.)
        if elapsed < WALK_START_WAIT_S:
            assert gate.reason is None
    assert gate.reason == 'walk start timeout'
    gate, c = walk_holding()
    tau = np.zeros(24)
    tau[4] = 70.
    gate.observe(20.23, np.zeros(24), np.zeros(24), tau, 0.)
    assert gate.reason == 'measured joint torque limit'


def test_wide_target_step_is_forwarded_without_a_slew_guard():
    gate, c = armed()
    for step in range(2, 511):
        assert gate.accept(10+step*.02, 100+step*.02, 'init', values(c), np.zeros(24), np.zeros(24))
    target = np.full(24, .3)
    assert gate.accept(20.22, 110.22, 'policy', values(c, target), np.zeros(24), np.zeros(24))
    assert gate.reason is None
    np.testing.assert_array_equal(gate.command[0], target)




def test_stationary_policy_detects_sustained_oscillation_below_peak_limits():
    gate, c = walk_holding()
    for i in range(1, 220):
        t = 20.22+i*.02
        q, dq = np.zeros(24), np.zeros(24)
        q[14] = .2*np.sin(2*np.pi*2*i*.02)
        dq[14] = .2*2*np.pi*2*np.cos(2*np.pi*2*i*.02)
        gate.accept(t, t+90, 'policy', values(c), q, dq)
        gate.observe(t, q, dq, np.zeros(24), 0.)
        if gate.reason:
            break
    assert gate.reason == 'stationary shoulder oscillation'
    assert not gate.accept(t+.02, t+90.02, 'policy', values(c), q, dq)




def test_gate_accepts_training_ankle_damping():
    c = config()
    c['kd'][4] = c['kd'][5] = c['kd'][10] = c['kd'][11] = 2.4916
    gate = PolicyGate(c, 'session')
    gate.accept(10., 100., 'zero', [], np.zeros(24), np.zeros(24))
    assert gate.accept(10.02, 100.02, 'init', values(c), np.zeros(24), np.zeros(24))
    assert gate.reason is None


@pytest.mark.parametrize('duration', [1., 3., 5., 10.])
def test_shaping_probe_duration_and_policy_lockout(duration):
    c = config()
    gate = PolicyGate(c, 'probe', shaping_only=duration)
    gate.accept(10, 100, 'zero', [], np.zeros(24), np.zeros(24))
    for step in range(int(duration/.02)+1):
        now = 10.02+step*.02
        gate.accept(now, now+90, 'init', values(c), np.zeros(24), np.zeros(24))
        gate.observe(now, np.zeros(24), np.zeros(24), np.zeros(24), 0.)
    gate.observe(10.021+duration, np.zeros(24), np.zeros(24), np.zeros(24), 0.)
    assert gate.reason == 'completed shaping probe'
    gate = PolicyGate(c, 'probe', shaping_only=duration)
    gate.accept(10, 100, 'zero', [], np.zeros(24), np.zeros(24))
    assert not gate.accept(10.02, 100.02, 'policy', values(c), np.zeros(24), np.zeros(24))
    assert gate.reason == 'policy disabled during shaping probe'


@pytest.mark.skipif(os.environ.get('PM01_RUN_BENCH_SMOKE') != '1', reason='Requires ROS schemas')
@pytest.mark.parametrize('case', ['normal', 'three_second_hold', 'client_loss', 'shaping_only', 'walk_shaping', 'walk_once', 'walk_feedback_gap', 'walk_feedback_loss'])
def test_onboard_relay_loopback(case, monkeypatch, tmp_path):
    for k, v in {'ROS_DOMAIN_ID': '171', 'ROS_LOCALHOST_ONLY': '0',
        'RMW_IMPLEMENTATION': 'rmw_cyclonedds_cpp', 'CYCLONEDDS_URI':
        '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/>'
        '</Interfaces></General></Domain></CycloneDDS>'}.items():
        monkeypatch.setenv(k, v)
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointState, JointCommand, MotionState, MotorDebug, ImuInfo
    from std_msgs.msg import String
    c = config()
    conf = tmp_path/'config.json'
    conf.write_text(json.dumps(c))
    rclpy.init()
    node = rclpy.create_node('bounded_policy_fixture')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    request_qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
    pubs = {k: node.create_publisher(t, topic, request_qos if k == 'request' else qos) for k, t, topic in [
        ('joint', JointState, '/hardware/joint_state'), ('echo', JointCommand, '/hardware/joint_command_feedback'),
        ('mode', MotionState, '/motion/motion_state'), ('debug', MotorDebug, '/hardware/motor_debug'),
        ('imu', ImuInfo, '/hardware/imu_info'),
        ('request', JointCommand, REQUEST_TOPIC)]}
    state = {'status': None, 'ready_at': None, 'last': JointCommand()}
    is_walk = case in ('walk_once', 'walk_feedback_gap', 'walk_feedback_loss')
    feedback_holds = []
    last = state['last']
    last.position = last.velocity = last.torque = last.feed_forward_torque = last.stiffness = [0.]*24
    last.damping = NATIVE_DAMPING.tolist()
    commands = []
    def received(msg):
        commands.append((time.monotonic(), msg))
        state['last'] = msg
    def status(msg):
        state['status'] = json.loads(msg.data)
        if state['status'].get('feedback_hold'):feedback_holds.append(time.monotonic())
        if state['ready_at'] is None: state['ready_at'] = time.monotonic()
    node.create_subscription(JointCommand, '/hardware/joint_command', received, qos)
    node.create_subscription(String, STATUS_TOPIC, status, qos)
    def feedback():
        elapsed = time.monotonic()-state['ready_at'] if state['ready_at'] is not None else -1.
        short_gap = case == 'walk_feedback_gap' and (12.3 <= elapsed < 12.36 or 13.3 <= elapsed < 13.366)
        long_gap = case == 'walk_feedback_loss' and 12.3 <= elapsed < 12.6
        j = JointState()
        j.position = j.velocity = j.torque = [0.]*24
        if not (short_gap or long_gap):pubs['joint'].publish(j)
        pubs['echo'].publish(state['last'])
        imu = ImuInfo()
        imu.quaternion.w = 1.
        if not short_gap:pubs['imu'].publish(imu)
    def slow():
        m = MotionState()
        setattr(m, 'current_motion_task_name' if hasattr(m, 'current_motion_task_name') else 'current_motion_task', 'joint_bridge')
        pubs['mode'].publish(m)
        d = MotorDebug()
        d.mos_temperature = d.motor_temperature = [30.]*24+[time.monotonic()]
        pubs['debug'].publish(d)
        s = state['status']
        if s is None or not s['ready']: return
        elapsed = time.monotonic()-state['ready_at']
        if case == 'client_loss' and elapsed > .5: return
        mode = 'zero' if elapsed < .3 else ('init' if elapsed < 10.6 else 'policy')
        if is_walk and elapsed >= 10.6:
            mode = 'hold' if elapsed < 11.3 else 'policy' if elapsed < 16.5 else 'play'
        cmd = JointCommand()
        epoch=s['feedback_epoch']
        stale_epoch=case=='walk_feedback_gap' and 12.4<=elapsed<12.42
        cmd.header.frame_id = s['token']+'/'+str(max(0,epoch-1) if stale_epoch else epoch)+'/'+mode
        cmd.header.stamp.sec, cmd.header.stamp.nanosec = divmod(time.monotonic_ns(), 10**9)
        cmd.position = cmd.velocity = cmd.torque = cmd.feed_forward_torque = [0.]*24
        if stale_epoch:
            cmd.position=[0.]*23+[.5]  # Must be ignored, never applied on recovery.
        cmd.stiffness, cmd.damping = c['kp'], c['kd']
        pubs['request'].publish(cmd)
    node.create_timer(.002, feedback)
    node.create_timer(.02, slow)
    output = tmp_path/'trial.jsonl'
    proc = None
    try:
        with (tmp_path/'console.txt').open('w+') as console:
            proc = subprocess.Popen([sys.executable, '-m', 'scripts.pm01.policy_bridge', '--execute',
                '--config', str(conf), '--output', str(output)]
                + (['--shaping-only'] if case == 'shaping_only' else [])
                + (['--shaping-only', '--shaping-seconds', '10', '--walk-shaping'] if case == 'walk_shaping' else [])
                + (['--walk-once'] if is_walk else [])
                + (['--hold-seconds', '3'] if case == 'three_second_hold' else []), stdout=console, stderr=subprocess.STDOUT)
            deadline = time.monotonic()+(32 if is_walk else 22)
            while proc.poll() is None and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=.001)
            assert proc.poll() is not None
            console.seek(0)
            log = console.read()
        assert proc.returncode == (0 if case in ('normal', 'three_second_hold', 'shaping_only', 'walk_shaping', 'walk_once', 'walk_feedback_gap') else 2), log
        assert any(m.parallel_parser_type == 1 for _, m in commands), log
        assert commands[-1][1].parallel_parser_type == 0, log
        assert list(commands[-1][1].damping) == NATIVE_DAMPING.tolist()
        report = json.loads(output.with_suffix('.supervisor.json').read_text())
        assert report['classic_fallback_confirmed'], log
        assert not report['watchdog_triggered'], log
        if case == 'three_second_hold':
            assert '"hold_seconds": 3.0' in log
            active = [t for t, m in commands if m.parallel_parser_type == 1]
            assert 13.1 < max(active)-min(active) < 13.6
        if case == 'shaping_only':
            active = [t for t, m in commands if m.parallel_parser_type == 1]
            assert max(active)-min(active) < 1.05
        if case == 'walk_shaping':
            assert 'completed shaping probe' in log
            active = [t for t, m in commands if m.parallel_parser_type == 1]
            assert 9.9 < max(active)-min(active) < 10.1
        if case in ('walk_once','walk_feedback_gap'):
            assert 'completed walk' in log
        if case == 'walk_feedback_gap':
            assert feedback_holds
            during_gap=[m for t,m in commands if 12.2<t-state['ready_at']<13.5]
            assert during_gap and all(m.parallel_parser_type==1 and list(m.stiffness)==c['kp'] for m in during_gap)
        if case == 'walk_feedback_loss':
            assert 'joint feedback timeout' in log
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        node.destroy_node()
        rclpy.shutdown()


def test_retired_client_profile_is_rejected_before_ros_creation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.pm01.trial_policy import TrialTracking
    from scripts.pm01.shadow_policy import ShadowTracking
    config = tmp_path/'policy.yaml'
    config.write_text('max_target_velocity_rad_s: 6.0\npm01_policy_blend_seconds: 2.0\n')
    def unexpected(*args):
        raise AssertionError('Do not construct ROS with a retired profile')
    monkeypatch.setattr(ShadowTracking, '__init__', unexpected)
    with pytest.raises(ValueError, match='Retired PM01'):
        TrialTracking(SimpleNamespace(policy_config=str(config)))
