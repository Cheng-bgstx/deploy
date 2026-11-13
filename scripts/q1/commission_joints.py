"""Operator-triggered Q1 measured-pose hold and one-joint commissioning probe.

Uses the production Q1 transport and serializer, without loading a policy.
Startup is disarmed. Low gains do not support a standing robot: use a gantry.
"""
import json
import os
import select
import sys
import termios
import time
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tyro

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.robot_io.q1 import Q1RobotIO


@dataclass
class Args:
    joint: str = 'left_elbow_joint'
    """One joint to probe. Left/right always refer to the robot."""
    read_only: bool = False
    """Observe only; never arm or send motor commands."""
    duration_s: float = 60.
    """Overall process deadline; an armed session also has a fixed 25 s deadline."""
    output_dir: Path = Path('outputs/q1/commission')


class Probe:
    amplitude = .03
    stiffness = 5.
    damping = 2.
    max_displacement = .10
    max_speed = .5

    def __init__(self, io, cfg, joint, *, read_only=False):
        self.io, self.cfg, self.joint = io, cfg, joint
        self.index = list(cfg.joint_names).index(joint)
        self.read_only = read_only
        self.mode = 'disarmed'
        self.baseline = self.armed_at = self.pulse_at = None
        self.events, self.samples = [], []
        self.done = False
        self.started_at = time.monotonic()
        self.last_sample = -np.inf
        self.wire_order = None

    def event(self, now, message):
        self.events.append({'time_s': now - self.started_at, 'message': message})
        print(message, flush=True)

    def stop(self, now, reason):
        if self.done:
            return
        if self.armed_at is not None:
            self.io.set_control_mode('zero', 2)
        self.done = True
        self.mode = 'stopped'
        self.event(now, reason)

    def step(self, now, key, publisher_count):
        try:
            self._step(now, key, publisher_count)
        except Exception as exc:
            self.stop(now, 'STOP: ' + str(exc))
            raise

    def _step(self, now, key, publisher_count):
        if self.done:
            return
        if key in ('o', ' ', 'q', '\x1b'):
            self.stop(now, 'STOP: damping requested; exiting.')
            return
        state = self.io.read_state()
        if state is None:
            if self.armed_at is not None:
                raise RuntimeError('Feedback is stale: stopping.')
            if key:
                self.event(now, 'Not ready: waiting for fresh joint and IMU feedback.')
            return
        q, dq = state.qpos[7:].copy(), state.qvel[6:].copy()
        if not np.isfinite(q).all() or not np.isfinite(dq).all():
            raise RuntimeError('Non-finite joint feedback.')
        if key == 'h' and self.mode == 'disarmed' and not self.read_only:
            if now - self.started_at < 5. or publisher_count != 1:
                self.event(now, 'Not ready: wait 5 s and ensure this is the only command publisher.')
                return
            if np.max(np.abs(dq)) > .1:
                self.event(now, 'Not ready: support the robot and wait for all joints to settle (<0.1 rad/s).')
                return
            self.baseline, self.armed_at = q.copy(), now
            self.wire_order = tuple(self.io.codec.feedback_order['body'])
            # Keep the first measured-pose command and arming atomic with the
            # publisher, so an earlier cached command cannot escape on arming.
            with self.io.codec.lock:
                self.io.set_control_mode('init', 1)
                self.io.write_command(q, np.zeros_like(q), np.zeros_like(q),
                                      np.zeros_like(q), np.full_like(q, self.damping))
            self.mode = 'hold'
            self.event(now, 'HOLD: captured measured positions; ramping kp to 5 over 2 s. j = probe; o/Space = stop.')
        if self.armed_at is None:
            target = q
        else:
            if publisher_count != 1:
                raise RuntimeError('Command publisher count changed; controller ownership is unclear.')
            if tuple(self.io.codec.feedback_order['body']) != self.wire_order:
                raise RuntimeError('HAL feedback array order changed after arming.')
            displacement = np.abs(q - self.baseline)
            if np.max(displacement) > self.max_displacement:
                index = int(np.argmax(displacement))
                raise RuntimeError(f'{self.cfg.joint_names[index]} moved >0.10 rad from the captured pose.')
            if np.max(np.abs(dq)) > self.max_speed:
                raise RuntimeError('A joint exceeded 0.5 rad/s.')
            if now - self.armed_at >= 25.:
                self.stop(now, 'STOP: 25 s armed deadline reached; this is not a validation pass.')
                return
            if key == 'j' and self.mode == 'hold':
                lo = self.cfg.joint_pos_lower_limit[self.joint]
                hi = self.cfg.joint_pos_upper_limit[self.joint]
                if now - self.armed_at < 3. or np.max(np.abs(dq)) > .1 or np.max(displacement) > .05:
                    self.event(now, 'Probe refused: hold must settle for 3 s, drift <0.05 rad and speed <0.1 rad/s.')
                elif not lo <= self.baseline[self.index] - self.amplitude < self.baseline[self.index] + self.amplitude <= hi:
                    self.event(now, 'Probe refused: the measured pose is too close to/outside the selected joint limits.')
                else:
                    self.mode, self.pulse_at = 'probe', now
                    self.event(now, f'PROBE: {self.joint}, +0.03 -> 0 -> -0.03 -> 0 rad over 8 s.')
            target = self.baseline.copy()
            if self.pulse_at is not None:
                elapsed = now - self.pulse_at
                if elapsed >= 9.:
                    self.stop(now, 'STOP: probe finished and damped. Inspect physical joint identity/direction and the log; no automatic PASS.')
                    return
                if elapsed < 8.:
                    segment = int(elapsed // 2.)
                    endpoints = (0., self.amplitude, 0., -self.amplitude, 0.)
                    fraction = .5 - .5 * np.cos(np.pi * (elapsed % 2.) / 2.)
                    target[self.index] += endpoints[segment] + fraction * (endpoints[segment + 1] - endpoints[segment])
            gain = self.stiffness * min((now - self.armed_at) / 2., 1.)
            self.io.write_command(target, np.zeros_like(q), np.zeros_like(q),
                                  np.full_like(q, gain), np.full_like(q, self.damping))
        if now - self.last_sample >= .1:
            self.samples.append({'time_s': now - self.started_at, 'mode': self.mode,
                'position': q.tolist(), 'velocity': dq.tolist(), 'target': target.tolist()})
            self.last_sample = now


@contextmanager
def keyboard(enabled):
    if not enabled:
        yield lambda: ''
        return
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)  # Preserve Ctrl+C/SIGINT.
        yield lambda: os.read(fd, 1).decode('ascii', errors='ignore') if select.select([fd], [], [], 0)[0] else ''
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def main(args):
    cfg = get_robot_cfg('q1')
    if args.joint not in cfg.joint_names:
        raise ValueError(f'Unknown Q1 joint: {args.joint}')
    if not np.isfinite(args.duration_s) or not 0 < args.duration_s <= 300.:
        raise ValueError('duration_s must be in (0, 300]')
    if not args.read_only and not sys.stdin.isatty():
        raise ValueError('Use an interactive terminal, or --read-only. Piped arming is disabled.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '.json')
    print('Q1 JOINT CHECK. Gantry must carry the weight; feet/legs must be free and uncrossed.')
    print(f'Selected: {args.joint}; + and - mean encoder coordinates, not a promised physical direction.')
    print('DISARMED: no motor commands. h = measured-pose hold; j = one ±0.03 rad cycle.')
    print('o / Space / q / Esc / Ctrl+C = damping and exit. Keep the vendor stop available.')
    print('Gains: kp <= 5, kd = 2; drift >0.10 rad or speed >0.5 rad/s stops. No policy is loaded.')
    io = Q1RobotIO(cfg, imu_topic='/aima/hal/imu/state', imu_to_pelvis_wxyz=(1., 0., 0., 0.))
    probe = Probe(io, cfg, args.joint, read_only=args.read_only)
    error, next_print, next_graph, publisher_count = None, 0., 0., None
    try:
        with keyboard(not args.read_only) as read_key:
            while not probe.done:
                now = time.monotonic()
                if now - probe.started_at >= args.duration_s:
                    probe.stop(now, 'STOP: session deadline reached.')
                    break
                if now >= next_graph:
                    publisher_count = io._node.count_publishers('/aima/hal/joint/command')
                    next_graph = now + .25
                probe.step(now, read_key(), publisher_count)
                if now >= next_print:
                    state = io.read_state()
                    if state is None:
                        print('Waiting for fresh joint + IMU feedback...', flush=True)
                    else:
                        q, dq = state.qpos[7:], state.qvel[6:]
                        delta = q - probe.baseline if probe.baseline is not None else np.zeros_like(q)
                        other = np.abs(delta).copy()
                        other[probe.index] = -1.
                        index = int(np.argmax(other))
                        print(f'{probe.mode.upper()} publishers={publisher_count} selected_q={q[probe.index]:+.4f} '
                              f'selected_delta={delta[probe.index]:+.4f} max_speed={np.max(np.abs(dq)):.3f} '
                              f'largest_other={cfg.joint_names[index]}:{delta[index]:+.4f}', flush=True)
                    next_print = now + .5
                time.sleep(.02)
        if not probe.samples:
            raise RuntimeError('No fresh joint + IMU feedback was recorded.')
    except (Exception, KeyboardInterrupt) as exc:
        error = f'{type(exc).__name__}: {exc}'
        probe.stop(time.monotonic(), 'STOP: ' + error)
    finally:
        # Production close emits damping only if the operator armed the adapter.
        io.close()
        output.write_text(json.dumps({'joint_names': list(cfg.joint_names), 'selected_joint': args.joint,
            'read_only': args.read_only, 'armed': probe.armed_at is not None,
            'hardware_validation_passed': False, 'amplitude_rad': probe.amplitude,
            'stiffness': probe.stiffness, 'damping': probe.damping,
            'feedback_wire_order': list(probe.wire_order or io.codec.feedback_order.get('body', ())),
            'error': error, 'events': probe.events, 'samples': probe.samples}, indent=2) + '\n')
        print(f'Log: {output}', flush=True)
    if error:
        raise SystemExit(1)


if __name__ == '__main__':
    main(tyro.cli(Args))
