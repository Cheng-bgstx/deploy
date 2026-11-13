"""Restricted PM01 ankle investigation; never runs a policy or moving target.

Default is read-only. --execute permits one at-most-one-second ankle damping
trial after 0.5 s of classic damping, followed by classic damping and command
feedback verification. Keep the robot supported with hardware emergency stop
available. This is an incident investigation tool, not deployment approval.
An optional --ankle-kp holds ONLY the measured initial ankle pose, never a
standing pose or reference motion. The same excursion/torque limits apply.
"""
import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import struct
import sys
import time
from pathlib import Path

import numpy as np

ANKLES = np.array([4, 5, 10, 11])
# Read from the connected 76445d09 firmware's passive/default configuration.
NATIVE_DAMPING = np.array([4, 4, 4, 4, 3, 3, 4, 4, 4, 4, 3, 3,
                          2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], dtype=float)


def temperatures(msg):
    """76445d09 DataMonitor::Add appends its timestamp after 24 motor values."""
    arrays = [np.asarray(getattr(msg, key), dtype=float)
              for key in ('mos_temperature', 'motor_temperature')]
    if any(a.shape != (25,) or not np.isfinite(a).all() or a[-1] < 0
           or np.any(a[:24] < -20) or np.any(a[:24] > 150) for a in arrays):
        return None
    return max(float(a[:24].max()) for a in arrays)


def matches(msg, parser, kd, motor=None, ankle_kp=0.):
    # This installed release leaves the feedback parser byte at its default.
    # Hardware acknowledgement therefore also requires actual motor-side gains.
    kp = np.zeros(24)
    if parser:
        kp[ANKLES] = ankle_kp
    return (msg is not None and (motor.matches(parser) if motor else msg.parallel_parser_type == parser)
            and np.asarray(msg.stiffness).shape == (24,)
            and np.allclose(msg.stiffness, kp, atol=1e-6, rtol=0)
            and all(list(getattr(msg, key)) == [0.] * 24
                    for key in ('feed_forward_torque', 'velocity'))
            and np.asarray(msg.damping).shape == (24,)
            and np.allclose(msg.damping, kd, atol=1e-6, rtol=0))


class MotorReadback:
    """Read-only ABI probe, restricted to the inspected libmotor binary.

    No ptrace, calls into the driver, memory writes or process suspension.
    Kernel access checks require running the bench on the motion host as root.
    The sampled arrays are not atomic; mismatches fail closed.
    """
    def __init__(self):
        pid = int(subprocess.check_output(['pgrep', '-n', '-x', 'src_executor']))
        lines = Path(f'/proc/{pid}/maps').read_text().splitlines()
        line = next(l for l in lines if l.endswith('/libmotor.so') and l.split()[2] == '00000000')
        if hashlib.sha256(Path(line.split()[-1]).read_bytes()).hexdigest() != 'd4e8777b09e487a678b60dc10f9a1eca7b955d2a74dedeaf66443d9382260c00':
            raise RuntimeError('Unrecognized motor ABI; refuse execution')
        self.base = int(line.split('-')[0], 16)
        self.fd = os.open(f'/proc/{pid}/mem', os.O_RDONLY)
        self.cmd = self.ptr(self.base + 0x3fcb0)
        self.errors = self.ptr(self.base + 0x3fcc8)
        self.power = self.ptr(self.base + 0x3fcd8)
        if any(struct.unpack('<i', self.read(a, 4))[0] != 24 for a in (self.cmd, self.errors)):
            raise RuntimeError('Unexpected motor dimension')

    def read(self, address, count):
        if address <= 4096 or not 0 < count <= 4096:
            raise RuntimeError('Invalid ABI address')
        data = os.pread(self.fd, count, address)
        if len(data) != count:
            raise RuntimeError('Incomplete ABI read')
        return data

    def ptr(self, address):
        return struct.unpack('<Q', self.read(address, 8))[0]

    def snapshot(self):
        return {key: np.array(struct.unpack('<24f', self.read(self.ptr(self.cmd+offset), 96)))
                for key, offset in [('kp', 24), ('kd', 32), ('tau_ff', 40)]}

    def enabled_without_faults(self):
        # Offline bits on this firmware were set even with fresh live telemetry;
        # they are not used as evidence of enable. All other errors are checked.
        return (self.read(self.power, 1) == b'\1'
                and all(not any(self.read(self.ptr(self.errors+8+8*i), 24))
                        for i in (0, *range(2, 15))))

    def matches(self, parser):
        data = self.snapshot()
        expected = NATIVE_DAMPING.copy()
        if parser == 1:
            expected[ANKLES] = 0.
        return (all(np.isfinite(a).all() for a in data.values())
                and np.allclose(data['kp'], 0, atol=1e-6, rtol=0)
                and np.allclose(data['kd'], expected, atol=1e-6, rtol=0)
                and np.max(np.abs(data['tau_ff'])) <= (2. if parser else 1e-6))

    def close(self):
        os.close(self.fd)


def motor_readback(execute):
    isolated = (os.environ.get('ROS_DOMAIN_ID') == '171' and os.environ.get('CYCLONEDDS_URI') ==
        '<CycloneDDS><Domain><General><Interfaces><NetworkInterface name="lo"/>'
        '</Interfaces></General></Domain></CycloneDDS>')
    return MotorReadback() if execute and not isolated else None


class DampingTrial:
    """Bounded experiment state machine, independent of ROS and real clocks."""
    def __init__(self, ankle_kd, duration):
        if ankle_kd not in (.05, .1, .2, .3):
            raise ValueError('Only ankle kd 0.05, 0.1, 0.2 or 0.3 is permitted')
        if not math.isfinite(duration) or not 0 < duration <= 1.:
            raise ValueError('Trial duration must be in (0, 1] seconds')
        self.ankle_kd, self.duration = ankle_kd, duration
        self.started = None
        self.origin = None
        self.reason = None
        self.fallback_at = None
        self.rl_at = None
        self.rl_confirmed = False

    def abort(self, now, reason):
        if self.reason is None:
            self.reason = reason
            self.fallback_at = now

    def command(self, now, q, dq, tau, *, feedback_age, mode, mode_age,
                classic_ack=False, rl_ack=False):
        q, dq, tau = (np.asarray(a, dtype=float) for a in (q, dq, tau))
        valid = all(a.shape == (24,) and np.isfinite(a).all() for a in (q, dq, tau))
        reason = None
        if not all(math.isfinite(t) and t >= 0 for t in (now, feedback_age, mode_age)):
            reason = 'invalid clock or feedback age'
        elif not valid:
            reason = 'invalid feedback'
        elif mode != 'joint_bridge' or mode_age > .4:
            reason = 'bridge ownership lost'
        elif feedback_age > .02:
            reason = 'joint feedback older than 20 ms'
        elif np.max(np.abs(dq[ANKLES])) > .5:
            reason = 'ankle speed exceeds 0.5 rad/s'
        elif np.max(np.abs(tau[ANKLES])) > 2.:
            reason = 'ankle feedback torque exceeds 2 Nm'
        elif self.origin is not None and np.max(np.abs(q[ANKLES] - self.origin)) > .04:
            reason = 'ankle moved more than 0.04 rad'
        if reason:
            self.abort(now, reason)
        if self.started is None:
            if reason:
                return None
            self.started = now
            self.origin = q[ANKLES].copy()
        # Do not publish into another native motion. Its controller owns output.
        if mode != 'joint_bridge':
            return None
        elapsed = now - self.started
        if self.reason is None and elapsed >= .5 and self.rl_at is None:
            if classic_ack:
                self.rl_at = now
            else:
                self.abort(now, 'classic damping was not acknowledged')
        if self.rl_at is not None and self.reason is None:
            self.rl_confirmed |= rl_ack
            if now - self.rl_at >= .05 and not rl_ack:
                self.abort(now, 'RL damping acknowledgement lost')
            elif now - self.rl_at >= self.duration:
                self.abort(now, 'completed' if self.rl_confirmed else 'RL damping was not acknowledged')
        parser = 0 if self.reason is not None or self.rl_at is None else 1
        kd = NATIVE_DAMPING.copy()
        if parser == 1:
            kd[ANKLES] = self.ankle_kd
        # Position, velocity, stiffness and feed-forward torque are ALWAYS zero.
        # With kp=0, q_target has no control effect in either firmware parser.
        return parser, kd


def run(args):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointCommand, JointState, MotionState, MotorDebug
    from rosidl_runtime_py.convert import message_to_ordereddict

    trial = DampingTrial(args.ankle_kd, args.duration)
    motor = motor_readback(args.execute)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves earlier incident/trial evidence.
    log = args.output.open('x')
    rclpy.init()
    node = rclpy.create_node('pm01_damping_investigation')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    state = {'joint': None, 'joint_at': -math.inf, 'mode': None, 'mode_at': -math.inf,
             'echo': None, 'echo_at': -math.inf, 'temperature': None, 'debug_at': -math.inf}
    count = {'joint': 0, 'echo': 0, 'mode': 0, 'debug': 0, 'sent': 0}
    started = time.monotonic()

    def record(kind, msg):
        log.write(json.dumps({'elapsed_s': time.monotonic() - started, 'kind': kind,
                              'message': message_to_ordereddict(msg)}) + '\n')

    def receive(kind, msg):
        now = time.monotonic()
        count[kind] += 1
        record(kind, msg)
        state[kind + '_at'] = now
        if kind == 'mode':
            state['mode'] = getattr(msg, 'current_motion_task_name',
                                    getattr(msg, 'current_motion_task', None))
        elif kind == 'debug':
            state['temperature'] = temperatures(msg)
        else:
            state[kind] = msg

    for kind, typ, topic in [('joint', JointState, '/hardware/joint_state'),
                             ('echo', JointCommand, '/hardware/joint_command_feedback'),
                             ('mode', MotionState, '/motion/motion_state'),
                             ('debug', MotorDebug, '/hardware/motor_debug')]:
        node.create_subscription(typ, topic, lambda m, k=kind: receive(k, m), qos)
    publisher = None

    def publish(command):
        if command is None:
            return
        parser, kd = command
        if args.worker_fd is not None:
            os.write(args.worker_fd, b'A' if count['sent'] == 0 else b'.')
        msg = JointCommand()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.position = msg.velocity = msg.feed_forward_torque = msg.torque = msg.stiffness = [0.] * 24
        if parser == 1 and args.ankle_kp:
            q, kp = np.zeros(24), np.zeros(24)
            q[ANKLES], kp[ANKLES] = trial.origin, args.ankle_kp
            msg.position, msg.stiffness = q.tolist(), kp.tolist()
        msg.damping = kd.tolist()
        msg.parallel_parser_type = parser
        publisher.publish(msg)
        count['sent'] += 1
        record('sent', msg)

    def classic_confirmed():
        m = state['echo']
        return time.monotonic() - state['echo_at'] < .05 and matches(m, 0, NATIVE_DAMPING, motor)

    try:
        # Collect actual telemetry before any publisher exists.
        until = time.monotonic() + 2.
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.002)
        if not args.execute:
            print(json.dumps({'read_only': True, 'mode': state['mode'], 'counts': count,
                              'max_temperature_C': state['temperature'], 'output': str(args.output)}, indent=2))
            return 0
        others = node.get_publishers_info_by_topic('/hardware/joint_command')
        subscribers = node.get_subscriptions_info_by_topic('/hardware/joint_command')
        if others or not any(s.qos_profile.reliability == ReliabilityPolicy.BEST_EFFORT for s in subscribers):
            raise RuntimeError('Require exclusive command ownership and a compatible subscriber')
        now = time.monotonic()
        if motor and not motor.enabled_without_faults():
            raise RuntimeError('Require powered, enabled motors without reported faults')
        if (state['joint'] is None or state['echo'] is None or state['temperature'] is None
                or state['temperature'] > 60 or now - state['debug_at'] > .1):
            raise RuntimeError('Missing fresh joint/command/temperature feedback, or temperature exceeds 60 C')
        j = state['joint']
        first = trial.command(now, j.position, j.velocity, j.torque,
            feedback_age=now-state['joint_at'], mode=state['mode'], mode_age=now-state['mode_at'])
        if first is None:
            raise RuntimeError('Preflight failed: ' + str(trial.reason))
        # The preflight above must not consume the command/acknowledgement window.
        trial = DampingTrial(args.ankle_kd, args.duration)
        publisher = node.create_publisher(JointCommand, '/hardware/joint_command', qos)
        discovery_deadline = time.monotonic() + 1.
        while publisher.get_subscription_count() == 0 and time.monotonic() < discovery_deadline:
            rclpy.spin_once(node, timeout_sec=.002)
        if publisher.get_subscription_count() == 0:
            raise RuntimeError('Command subscriber discovery failed')
        now = time.monotonic()
        deadline = now + .5 + args.duration + .6
        next_publish = now
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.001)
            now = time.monotonic()
            if now < next_publish:
                continue
            next_publish = now + .002
            j = state['joint']
            if state['temperature'] is None or state['temperature'] > 60 or now-state['debug_at'] > .1:
                trial.abort(now, 'motor temperature feedback fault')
            if motor and not motor.enabled_without_faults():
                trial.abort(now, 'motor power, enable or fault changed')
            if count['sent'] and now - next_command_at > .02:
                trial.abort(now, 'command loop missed 20 ms deadline')
            rl_kd = NATIVE_DAMPING.copy()
            rl_kd[ANKLES] = args.ankle_kd
            publish(trial.command(now, j.position, j.velocity, j.torque,
                feedback_age=now-state['joint_at'], mode=state['mode'], mode_age=now-state['mode_at'],
                classic_ack=classic_confirmed(),
                rl_ack=now-state['echo_at'] < .05 and matches(state['echo'], 1, rl_kd, motor, args.ankle_kp)))
            if motor:
                log.write(json.dumps({'elapsed_s': time.monotonic()-started, 'kind': 'motor_command',
                    'message': {k: v.tolist() for k, v in motor.snapshot().items()}})+'\n')
            next_command_at = now
            if trial.fallback_at is not None and now-trial.fallback_at > .5:
                break
        confirmed = classic_confirmed()
        print(json.dumps({'reason': trial.reason, 'classic_fallback_confirmed': confirmed,
                          'rl_damping_confirmed': trial.rl_confirmed,
                          'counts': count, 'output': str(args.output)}, indent=2))
        return 0 if trial.reason == 'completed' and confirmed else 2
    finally:
        # Only a trial that actually sent output owns cleanup. No startup output.
        if publisher is not None and count['sent']:
            until = time.monotonic() + .3
            while time.monotonic() < until:
                if state['mode'] == 'joint_bridge':
                    publish((0, NATIVE_DAMPING))
                rclpy.spin_once(node, timeout_sec=.002)
        log.flush()
        log.close()
        if args.worker_fd is not None and count['sent'] and classic_confirmed():
            os.write(args.worker_fd, b'C')
        node.destroy_node()
        rclpy.shutdown()
        if motor:
            motor.close()


def supervised(args, *, worker_argv=None, timeout_s=8.):
    """Separate process enforces a wall-time limit even if the worker hangs.

    On hardware run BOTH processes on the motion computer. This covers worker
    failure and PC/SSH disconnection, not failure of Linux, ROS or the motor bus.
    Only after reaping the worker may the supervisor create its command writer.
    """
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from interface_protocol.msg import JointCommand, MotionState

    motor = motor_readback(True)
    rclpy.init()
    node = rclpy.create_node('pm01_damping_supervisor')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    state = {'mode': None, 'at': -math.inf, 'echo': None, 'echo_at': -math.inf}
    def mode(m):
        state.update(mode=getattr(m, 'current_motion_task_name',
                                 getattr(m, 'current_motion_task', None)), at=time.monotonic())
    def echo(m):
        state.update(echo=m, echo_at=time.monotonic())
    node.create_subscription(MotionState, '/motion/motion_state', mode, qos)
    node.create_subscription(JointCommand, '/hardware/joint_command_feedback', echo, qos)
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    worker = None
    armed = False
    watchdog = False
    cleaned = False
    confirmed = False
    started = heartbeat = time.monotonic()
    try:
        if worker_argv is None:
            worker_argv = [sys.executable, str(Path(__file__).resolve()), '--execute',
                '--ankle-kd', str(args.ankle_kd), '--ankle-kp', str(args.ankle_kp),
                '--duration', str(args.duration), '--output', str(args.output)]
        worker = subprocess.Popen([*worker_argv, '--worker-fd', str(write_fd)], pass_fds=(write_fd,))
        os.close(write_fd)
        write_fd = None
        while worker.poll() is None:
            rclpy.spin_once(node, timeout_sec=.005)
            try:
                data = os.read(read_fd, 4096)
            except BlockingIOError:
                data = b''
            if data:
                armed |= b'A' in data
                cleaned |= b'C' in data
                heartbeat = time.monotonic()
            if time.monotonic()-started > timeout_s or (armed and time.monotonic()-heartbeat > (2 if cleaned else .1)):
                watchdog = True
                worker.kill()
                break
    finally:
        if worker is not None:
            if worker.poll() is None:
                worker.kill()
            worker.wait(timeout=2)
        # Drain activation bytes if the worker crashed immediately after output.
        try:
            armed |= b'A' in os.read(read_fd, 65536)
        except BlockingIOError:
            pass
        if armed:
            publisher = node.create_publisher(JointCommand, '/hardware/joint_command', qos)
            until = time.monotonic() + .7
            while time.monotonic() < until:
                rclpy.spin_once(node, timeout_sec=.002)
                if state['mode'] == 'joint_bridge' and time.monotonic()-state['at'] <= .4:
                    msg = JointCommand()
                    msg.position = msg.velocity = msg.feed_forward_torque = msg.torque = msg.stiffness = [0.] * 24
                    msg.damping = NATIVE_DAMPING.tolist()
                    msg.parallel_parser_type = 0
                    publisher.publish(msg)
            confirmed = (time.monotonic()-state['echo_at'] < .05
                         and matches(state['echo'], 0, NATIVE_DAMPING, motor))
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
        node.destroy_node()
        rclpy.shutdown()
        if motor:
            motor.close()
    result = {'supervisor': True, 'watchdog_triggered': watchdog, 'output_started': armed,
              'classic_fallback_confirmed': confirmed, 'worker_exit': worker.returncode}
    args.output.with_suffix('.supervisor.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if worker.returncode == 0 and not watchdog and confirmed else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='Execute the bounded damping trial (no policy)')
    p.add_argument('--ankle-kd', type=float, choices=[.05, .1, .2, .3], default=.05)
    p.add_argument('--ankle-kp', type=float, choices=[0., 2., 5., 10., 20., 30., 37.701], default=0.,
                   help='Hold the measured initial ankle pose with bounded stiffness; default is pure damping')
    p.add_argument('--duration', type=float, default=1.)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker-fd', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    DampingTrial(args.ankle_kd, args.duration)
    if args.execute:
        # A dropped SSH session must not terminate the onboard exit supervisor.
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(supervised(args) if args.execute and args.worker_fd is None else run(args))
