"""Bounded PM01 policy relay for supervised trials.

The relay is the only hardware command writer. A PC sends candidate requests to a
different topic. Each run allows shaping and a short first-frame policy hold, then
latches classic damping. Gains, feed-forward and speeds are passed through
unchanged; the guards only reject non-finite input, stale commands, out-of-range
torque and motion beyond the training actuator limits.
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import numpy as np

try:
    from .bench_damping import ANKLES, NATIVE_DAMPING, motor_readback, matches, supervised, temperatures
except ImportError:  # Standalone copied scripts on the motion computer.
    from bench_damping import ANKLES, NATIVE_DAMPING, motor_readback, matches, supervised, temperatures

REQUEST_TOPIC = '/sim2real/pm01/request'
STATUS_TOPIC = '/sim2real/pm01/status'
WALK_START_WAIT_S = 180.
WALK_ACTIVE_LIMIT_S = 250.
WALK_WORKER_LIMIT_S = 260.
WALK_SUPERVISOR_LIMIT_S = 270.
WALK_CLIENT_LIMIT_S = 285.
JOINT_FRESH_S = .05
WALK_FEEDBACK_HOLD_LIMIT_S = .1


# Training actuator limits from the released engineai_pm01 contract. These are
# the same values the policy was trained against; nothing tighter is applied.
PM01_EFFORT = np.array([164., 164., 61., 164., 61., 61., 164., 164., 61., 164., 61., 61.,
                        61., 61., 61., 61., 61., 61., 61., 61., 61., 61., 61., 61.])
PM01_VELOCITY = np.array([30.8247, 19.2654, 30.8247, 19.2654, 35.641, 35.641,
                          30.8247, 19.2654, 30.8247, 19.2654, 35.641, 35.641,
                          30.8247] + [35.641]*10 + [21.192])


class PolicyGate:
    """Single-session request validation; a stop cannot be cleared by a packet."""
    def __init__(self, config, token, *, shaping_only=False, hold_seconds=1., walk_shaping=False, walk_once=False):
        self.config, self.token = config, token
        self.shaping_only = shaping_only
        self.walk_once = walk_once
        if walk_once and (shaping_only or walk_shaping):
            raise ValueError('Walk execution cannot be combined with shaping probes')
        if walk_shaping and shaping_only != 10.:
            raise ValueError('Walk shaping requires the ten-second shaping-only probe')
        if hold_seconds not in (1., 3.):
            raise ValueError('Unsupported policy hold duration')
        self.hold_seconds = float(hold_seconds)
        if shaping_only and shaping_only not in (1., 3., 5., 10.):
            raise ValueError('Unsupported shaping probe duration')
        for k in ('kp', 'kd', 'lower', 'upper'):
            a = np.asarray(config[k], float)
            if a.shape != (24,) or not np.isfinite(a).all():
                raise ValueError('Invalid bridge configuration')
            setattr(self, k, a)
        if np.any(self.kp <= 0) or np.any(self.kd < 0) or np.any(self.lower >= self.upper):
            raise ValueError('Invalid gain/limit configuration')
        self.torque_limit = np.asarray(config.get('effort', PM01_EFFORT), float)
        self.velocity_limit = np.asarray(config.get('velocity', PM01_VELOCITY), float)
        for name, a in (('effort', self.torque_limit), ('velocity', self.velocity_limit)):
            if a.shape != (24,) or not np.isfinite(a).all() or np.any(a <= 0):
                raise ValueError(f'Invalid {name} limit configuration')
        self.last_source = self.source_origin = self.arrival_origin = None
        self.at = self.started = self.policy_at = None
        self.play_at = None
        self.active_mode = 'zero'
        self.command = self.previous_q = None
        self.reason = None
        self.hold_samples = deque()
        self.hold_checked_at = -math.inf
        self.hold_stable = False
        self.feedback_hold = False
        self.feedback_epoch = 0

    def stop(self, reason):
        if self.reason is None:
            self.reason = reason

    def observe(self, now, q, dq, tau, age):
        if self.started is None or self.reason:
            return
        vectors = [np.asarray(a, float) for a in (q, dq, tau)]
        was_hold = self.feedback_hold
        self.feedback_hold = False
        if any(a.shape != (24,) or not np.isfinite(a).all() for a in vectors):
            self.stop('invalid feedback')
        elif not math.isfinite(age) or not 0 <= age <= (WALK_FEEDBACK_HOLD_LIMIT_S if self.walk_once else JOINT_FRESH_S):
            self.stop('joint feedback timeout')
        elif age > JOINT_FRESH_S:
            # The inspected firmware recomputes joint PD in its independent
            # 2 ms motor task. Retain the last validated target/gains during a
            # bounded ROS feedback gap; never advance targets from stale q/dq.
            self.feedback_hold = True
            if not was_hold:
                self.feedback_epoch += 1
            self.hold_stable = False
            self.hold_samples.clear()
            if self.at is None or now-self.at > .1:
                self.stop('PC command timeout')
        elif np.any(np.abs(vectors[1]) > self.velocity_limit):
            self.stop('joint speed limit')
        elif np.any(np.abs(vectors[2]) > self.torque_limit):
            self.stop('measured joint torque limit')
        elif self.at is None or now-self.at > .1:
            self.stop('PC command timeout')
        elif self.shaping_only and now-self.started >= float(self.shaping_only):
            self.stop('completed shaping probe')
        elif now-self.started > (WALK_ACTIVE_LIMIT_S if self.walk_once else 11.5+self.hold_seconds):
            self.stop('trial time limit')
        elif self.walk_once and self.policy_at is None and now-self.started > 55.:
            self.stop('policy activation timeout')
        elif self.walk_once and self.play_at is not None and now-self.play_at >= 8.5:
            self.stop('completed walk')
        elif self.walk_once and self.policy_at is not None and self.play_at is None and now-self.policy_at > WALK_START_WAIT_S:
            self.stop('walk start timeout')
        elif not self.walk_once and self.policy_at is not None and now-self.policy_at >= self.hold_seconds:
            self.stop('completed first-frame hold')
        elif self.command is not None:
            torque = self.kp*(self.command[0]-vectors[0])-self.kd*vectors[1]
            if np.any(np.abs(torque) > self.torque_limit):
                self.stop('estimated joint PD torque limit')
        if self.walk_once and self.active_mode == 'policy' and not self.reason and not self.feedback_hold:
            self.hold_samples.append((now, vectors[0][[14, 19]].copy(), vectors[1][[14, 19]].copy()))
            while self.hold_samples and now-self.hold_samples[0][0] > 2.:
                self.hold_samples.popleft()
            if (self.policy_at is not None and now-self.policy_at >= 4.
                    and now-self.hold_checked_at >= .1 and len(self.hold_samples) >= 20
                    and now-self.hold_samples[0][0] >= 1.8):
                self.hold_checked_at = now
                q_window = np.array([s[1] for s in self.hold_samples])
                dq_window = np.array([s[2] for s in self.hold_samples])
                if np.max(np.ptp(q_window, axis=0)) > .25 or np.max(np.sqrt(np.mean(dq_window**2, axis=0))) > .6:
                    self.stop('stationary shoulder oscillation')
                else:
                    self.hold_stable = True

    def accept(self, now, source, mode, values, q, dq):
        if self.reason or self.feedback_hold:
            return False
        if (not math.isfinite(source) or not math.isfinite(now)
                or source < 0 or now < 0 or (self.last_source is not None and source <= self.last_source)):
            self.stop('invalid or out-of-order clock')
            return False
        if mode == 'zero':
            if self.started is not None:
                self.stop('PC stop')
                return False
            if self.source_origin is None:
                self.source_origin, self.arrival_origin = source, now
            self.last_source = source
            return False
        if self.source_origin is None or mode not in (('init', 'hold', 'policy', 'play') if self.walk_once else ('init', 'policy')):
            self.stop('missing standby handshake or invalid mode')
            return False
        if self.shaping_only and mode == 'policy':
            self.stop('policy disabled during shaping probe')
            return False
        if self.walk_once and ((mode == 'init' and self.active_mode in ('hold', 'policy', 'play'))
                or (mode == 'hold' and (self.started is None or now-self.started < 9.5 or self.policy_at is not None))
                or (mode == 'policy' and self.active_mode not in ('hold', 'policy'))
                or (mode == 'policy' and self.play_at is not None)
                or (mode == 'play' and (self.policy_at is None or now-self.policy_at < 4.
                    or (self.play_at is None and not self.hold_stable)))):
            self.stop('invalid walk phase transition')
            return False
        # Source times are monotonic, with a per-session baseline from standby.
        # This catches backlog/clock jumps; it is not a synchronized wire clock.
        delay = (now-self.arrival_origin)-(source-self.source_origin)
        if delay > .05:
            # An old queued packet is not a fresh command. Discard it without
            # unloading the robot; the unchanged 100 ms command deadline still
            # latches if no current packet follows.
            return False
        if not math.isfinite(source) or source <= self.last_source or delay < -.02:
            self.stop('expired or out-of-order request')
            return False
        self.last_source = source
        arrays = [np.asarray(a, float) for a in values]
        if len(arrays) != 5 or any(a.shape != (24,) or not np.isfinite(a).all() for a in arrays):
            self.stop('invalid command arrays')
            return False
        target, vel, ff, kp, kd = arrays
        q, dq = np.asarray(q), np.asarray(dq)
        if any(a.shape != (24,) or not np.isfinite(a).all() for a in (q, dq)):
            self.stop('invalid feedback')
            return False
        if (not np.allclose(kp, self.kp, rtol=0, atol=1e-5)
                or not np.allclose(kd, self.kd, rtol=0, atol=1e-5)):
            self.stop('unexpected gains')
        elif self.started is None and (mode != 'init' or np.max(np.abs(target-q)) > .02):
            self.stop('first command must shape from the measured pose')
        elif (np.any(target < self.lower-.01) or np.any(target > self.upper+.01)) and not (
                mode in ('init', 'hold') and np.max(np.abs(target-q)) <= .02):
            self.stop('target position limit')
        elif np.any(np.abs(kp*(target-q)-kd*dq) > self.torque_limit):
            self.stop('requested joint PD torque limit')
        elif mode == 'policy' and now-self.started < 9.5:
            self.stop('policy selected before shaping')
        if self.reason:
            return False
        if self.started is None:
            self.started = now
        if mode == 'policy' and self.policy_at is None:
            self.policy_at = now
        if mode == 'play' and self.play_at is None:
            self.play_at = now
        self.active_mode = mode
        self.at, self.command, self.previous_q = now, arrays, target.copy()
        return True


def worker(args):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointState, JointCommand, MotionState, MotorDebug, ImuInfo
    from std_msgs.msg import String
    from rosidl_runtime_py.convert import message_to_ordereddict
    config = json.loads(args.config.read_text())
    gate = PolicyGate(config, uuid.uuid4().hex,
                      shaping_only=args.shaping_seconds if args.shaping_only else False,
                      hold_seconds=args.hold_seconds, walk_shaping=args.walk_shaping, walk_once=args.walk_once)
    motor = motor_readback(True)
    rclpy.init()
    node = rclpy.create_node('pm01_bounded_policy_bridge')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    state = dict(joint=None, joint_at=-math.inf, mode=None, mode_at=-math.inf,
                 debug=None, debug_at=-math.inf, echo=None, echo_at=-math.inf, imu=None, imu_at=-math.inf)
    started, sent, fallback_at = time.monotonic(), 0, None
    log = args.output.open('x')
    status_pub = node.create_publisher(String, STATUS_TOPIC, qos)
    publisher = None

    def record(kind, msg):
        log.write(json.dumps({'elapsed_s': time.monotonic()-started, 'kind': kind,
                              'message': message_to_ordereddict(msg)})+'\n')

    def receive(kind, msg):
        state[kind], state[kind+'_at'] = msg, time.monotonic()

    for kind, typ, topic in [('joint', JointState, '/hardware/joint_state'),
                             ('echo', JointCommand, '/hardware/joint_command_feedback'),
                             ('debug', MotorDebug, '/hardware/motor_debug')]:
        node.create_subscription(typ, topic, lambda m, k=kind: receive(k, m), qos)
    def mode(msg):
        state.update(mode=getattr(msg, 'current_motion_task_name',
                                 getattr(msg, 'current_motion_task', None)), mode_at=time.monotonic())
    node.create_subscription(MotionState, '/motion/motion_state', mode, qos)
    if args.walk_once:
        node.create_subscription(ImuInfo, '/hardware/imu_info', lambda m: receive('imu', m), qos)

    def valid_walk_imu(now):
        m = state['imu']
        if m is None or now-state['imu_at'] > WALK_FEEDBACK_HOLD_LIMIT_S:
            return False
        q = np.array([m.quaternion.w, m.quaternion.x, m.quaternion.y, m.quaternion.z])
        w = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        if not np.isfinite(q).all() or not np.isfinite(w).all() or not .95 < np.linalg.norm(q) < 1.05:
            return False
        q /= np.linalg.norm(q)
        return np.arccos(np.clip(1-2*(q[1]**2+q[2]**2), -1, 1)) <= .35 and np.max(abs(w)) <= 3.

    def request(msg):
        prefix = gate.token+'/'+str(gate.feedback_epoch)+'/'
        if msg.header.frame_id not in tuple(prefix+m for m in ('zero', 'init', 'hold', 'policy', 'play')):
            return
        j = state['joint']
        if (j is None or time.monotonic()-state['joint_at'] > JOINT_FRESH_S
                or (args.walk_once and time.monotonic()-state['imu_at'] > JOINT_FRESH_S)):
            return
        record('request', msg)
        gate.accept(time.monotonic(), msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            msg.header.frame_id.split('/')[-1],
            [msg.position, msg.velocity, msg.feed_forward_torque, msg.stiffness, msg.damping], j.position, j.velocity)

    def classic_confirmed():
        return time.monotonic()-state['echo_at'] < .05 and matches(state['echo'], 0, NATIVE_DAMPING, motor)

    def publish(classic):
        nonlocal sent
        msg = JointCommand()
        if classic:
            msg.position = msg.velocity = msg.feed_forward_torque = msg.torque = msg.stiffness = [0.]*24
            msg.damping = NATIVE_DAMPING.tolist()
            msg.parallel_parser_type = 0
        else:
            msg.position, msg.velocity, msg.feed_forward_torque, msg.stiffness, msg.damping = [a.tolist() for a in gate.command]
            msg.torque = [0.]*24
            msg.parallel_parser_type = 1
        os.write(args.worker_fd, b'A' if not sent else b'.')
        publisher.publish(msg)
        sent += 1
        record('sent', msg)

    try:
        until = time.monotonic()+2.
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.001)
        if (state['mode'] != 'joint_bridge' or (motor and not motor.enabled_without_faults())
                or not classic_confirmed() or node.get_publishers_info_by_topic('/hardware/joint_command')
                or (args.walk_once and not valid_walk_imu(time.monotonic()))):
            raise RuntimeError('Require exclusive bridge, enabled motors and confirmed native classic damping')
        # Preserve adjacent policy samples on the private PC-to-relay link.
        # Stale/reordered requests still latch the existing clock/deadline guard.
        request_qos = QoSProfile(depth=8, reliability=ReliabilityPolicy.RELIABLE)
        node.create_subscription(JointCommand, REQUEST_TOPIC, request, request_qos)
        publisher = node.create_publisher(JointCommand, '/hardware/joint_command', qos)
        next_pub = next_status = time.monotonic()
        while time.monotonic()-started < (WALK_WORKER_LIMIT_S if args.walk_once else 25.):
            rclpy.spin_once(node, timeout_sec=.001)
            now = time.monotonic()
            if state['mode'] != 'joint_bridge' or now-state['mode_at'] > .4:
                gate.stop('bridge ownership lost')
            if args.walk_once and not valid_walk_imu(now):
                gate.stop('IMU freshness, tilt or angular speed limit')
            if args.walk_once and motor and gate.started is not None:
                if any(not np.isfinite(v).all() for v in motor.snapshot().values()):
                    gate.stop('invalid parallel motor feedback')
            temp = temperatures(state['debug']) if state['debug'] else None
            if temp is None or temp > 60 or now-state['debug_at'] > .1 or (motor and not motor.enabled_without_faults()):
                gate.stop('motor feedback, temperature or enable fault')
            j = state['joint']
            if j is not None:
                feedback_age = max(now-state['joint_at'], now-state['imu_at']) if args.walk_once else now-state['joint_at']
                gate.observe(now, j.position, j.velocity, j.torque, feedback_age)
            if gate.started is None and now-started > 10.:
                gate.stop('no client start')
            if now >= next_status:
                status = String()
                status.data = json.dumps({'token': gate.token, 'ready': gate.reason is None,
                    'active': gate.started is not None, 'reason': gate.reason, 'phase': gate.active_mode,
                    'stationary_hold_stable': gate.hold_stable and gate.reason is None,
                    'feedback_hold': gate.feedback_hold, 'feedback_epoch': gate.feedback_epoch,
                    'last_target_sdk': gate.command[0].tolist() if gate.command is not None else None})
                status_pub.publish(status)
                next_status = now+.02
            if gate.reason and fallback_at is None:
                fallback_at = now
            if gate.started is not None and now >= next_pub and state['mode'] == 'joint_bridge':
                publish(gate.reason is not None)
                record('joint', j)
                if args.walk_once and motor:
                    log.write(json.dumps({'elapsed_s': now-started, 'kind': 'motor_command',
                        'message': {k: v.tolist() for k, v in motor.snapshot().items()}})+'\n')
                next_pub = now+.005
            if fallback_at is not None and now-fallback_at > .5:
                break
        result = {'reason': gate.reason, 'sent': sent, 'hold_seconds': gate.hold_seconds,
                  'classic_fallback_confirmed': classic_confirmed() if sent else False}
        print(json.dumps(result, indent=2), flush=True)
        return 0 if gate.reason in ('completed first-frame hold', 'completed shaping probe', 'completed walk') and result['classic_fallback_confirmed'] else 2
    finally:
        if sent:
            until = time.monotonic()+.3
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=.002)
                if state['mode'] == 'joint_bridge':
                    publish(True)
            if classic_confirmed():
                os.write(args.worker_fd, b'C')
        log.close()
        node.destroy_node()
        rclpy.shutdown()
        if motor:
            motor.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--shaping-only', action='store_true', help='Stop after --shaping-seconds, before policy mode')
    p.add_argument('--shaping-seconds', type=float, choices=[1., 3., 5., 10.], default=1.)
    p.add_argument('--hold-seconds', type=float, choices=[1., 3.], default=1.)
    p.add_argument('--walk-shaping', action='store_true',
                   help='Original walk first-frame shoulder envelope; requires --shaping-only --shaping-seconds 10')
    p.add_argument('--walk-once', action='store_true', help='Guarded original walk; waits for the client play barrier')
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if not args.execute:
        raise SystemExit('No output: experimental execution requires --execute')
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    if args.worker_fd is None:
        raise SystemExit(supervised(args, worker_argv=[sys.executable, str(Path(__file__).resolve()),
            '--execute', '--config', str(args.config), '--output', str(args.output),
            '--hold-seconds', str(args.hold_seconds)]
            + (['--walk-shaping'] if args.walk_shaping else [])
            + (['--walk-once'] if args.walk_once else [])
            + (['--shaping-only', '--shaping-seconds', str(args.shaping_seconds)] if args.shaping_only else []),
            timeout_s=WALK_SUPERVISOR_LIMIT_S if args.walk_once else 30.))
    raise SystemExit(worker(args))
