"""Motor-command arbitration independent of inference and SDK callbacks.

All timestamps use the local PC's monotonic clock. Callers serialize access.
Standby emits no command. A ready, fresh Tracker and a new physical A press
are both required before releasing high-level control. Faults latch.
"""
from __future__ import annotations

import numpy as np

from sim2real.config.robots.base import RobotCfg
from sim2real.utils.common import LowCmdMessage, LowStateMessage
from sim2real.utils.strings import resolve_matching_names_values

# Proteus trains without clipping position targets: the G1 ankles push them up to about 0.8 rad past the
# joint range to get torque. A target past a mechanical stop only raises the PD torque, which the motor bounds.
PROTEUS_TARGET_MARGIN_RAD = 1.0
# Holding through a gap is for dropped samples; a longer silence (a robot reboot took 117 s) needs a fresh A.
MAX_HOLD_GAP_NS = 1_000_000_000


class BridgeSafety:
    def __init__(self, cfg: RobotCfg, timeout_s: float = 0.1, target_margin_rad: float = 0.0,
                 hold_through_gaps: bool = False):
        """``target_margin_rad``: how far past the joint range a position target may point.
        ``hold_through_gaps``: once active, a LowState or Tracker command gap keeps sending the last command (its PD
        targets) instead of latching damping, and control resumes when fresh data returns; late or repeated commands
        are dropped. A gap longer than ``MAX_HOLD_GAP_NS`` or a restarted robot tick still latches: the robot may have
        rebooted, and old targets must never resume on it. Invalid states and commands, Select and shutdown still
        stop."""
        if not 0 < timeout_s <= 1:
            raise ValueError("timeout_s must be in (0, 1]")
        if not 0 <= target_margin_rad <= 1.5:
            raise ValueError("target_margin_rad must be in [0, 1.5]")
        self.count = len(cfg.joint_names)
        self.timeout_ns = int(timeout_s * 1e9)
        self.hold_through_gaps = hold_through_gaps
        self.lower = np.array([cfg.joint_pos_lower_limit[n] for n in cfg.joint_names]) - target_margin_rad
        self.upper = np.array([cfg.joint_pos_upper_limit[n] for n in cfg.joint_names]) + target_margin_rad
        # An unspecified velocity limit permits only a zero velocity target.
        self.velocity = np.array([cfg.joint_velocity_limit.get(n, 0.0) for n in cfg.joint_names])
        self.effort = np.array([cfg.joint_effort_limit[n] for n in cfg.joint_names])
        ids, _, kd = resolve_matching_names_values(cfg.safe_joint_kd, cfg.joint_names, preserve_order=True)
        self.damping_kd = np.zeros(self.count, dtype=np.float32)
        self.damping_kd[ids] = kd
        if not np.isfinite(self.damping_kd).all() or np.any(self.damping_kd <= 0):
            raise ValueError("Fault damping gains must be finite and positive")
        self.state = None
        self.state_time_ns = 0
        self.command = None
        self.last_sequence = 0
        self.last_source_ns = 0
        self.stop_epoch = 0
        self.reason = "standby"
        self.latched = False
        self.fault = False
        self.active = False
        self.arm_pending = False
        self.arm_time_ns = 0
        self._selected = False
        self._a = False
        self._zero_ack_generation = None
        self._stop_generation = -1

    def latch(self, reason: str, *, fault: bool = True):
        # Fatal faults upgrade a software stop to damping.
        if not self.latched or (fault and not self.fault):
            self.stop_epoch += 1
            self._zero_ack_generation = None
            self._stop_generation = self.command.control_generation if self.command else -1
        self.latched = True
        self.fault |= fault
        self.reason = reason
        self.arm_pending = False
        self.arm_time_ns = 0

    def observe_state(self, state: LowStateMessage, keys: int, now_ns: int):
        arrays = (state.quaternion, state.gyroscope, state.joint_positions, state.joint_velocities)
        valid = (state.joint_positions.shape == (self.count,)
                 and state.joint_velocities.shape == (self.count,)
                 and all(np.isfinite(a).all() for a in arrays)
                 and 0.9 < np.linalg.norm(state.quaternion) < 1.1)
        if not valid:
            self.latch("invalid LowState")
            return
        if self.state is not None and state.tick < self.state.tick:
            self.latch("robot LowState tick restarted")
        if self.state is None or state.tick != self.state.tick:
            self.state_time_ns = now_ns
        self.state = state
        selected, a = bool(keys & (1 << 3)), bool(keys & (1 << 8))
        if selected and not self._selected:
            self.latch("Select", fault=False)
        if selected:
            self.arm_time_ns = 0
        elif a and not self._a:
            acknowledged = not self.latched or self._zero_ack_generation is not None
            if acknowledged and self.ready(now_ns):
                self.arm_time_ns = now_ns
        self._selected, self._a = selected, a

    def ready(self, now_ns: int) -> bool:
        return (self.state is not None and self.command is not None
                and 0 <= now_ns - self.state_time_ns <= self.timeout_ns
                and 0 <= now_ns - self.command.source_time_ns <= self.timeout_ns)

    def accept_command(self, cmd: LowCmdMessage, now_ns: int):
        arrays = (cmd.q_target, cmd.dq_target, cmd.tau_ff, cmd.kp, cmd.kd)
        follows_measured = (cmd.control_mode == "zero" and self.state is not None
                            and cmd.q_target.shape == (self.count,)
                            and np.all(np.abs(cmd.q_target - self.state.joint_positions) <= 0.05))
        lower, upper = self.lower, self.upper
        if cmd.control_mode == "init" and self.state is not None and self.state.joint_positions.shape == (self.count,):
            # Shaping starts wherever the robot is: a joint already past a limit may be held
            # there or brought back, never commanded further out than it is.
            lower = np.minimum(lower, self.state.joint_positions)
            upper = np.maximum(upper, self.state.joint_positions)
        valid = (all(a.shape == (self.count,) and np.isfinite(a).all() for a in arrays)
                 and (follows_measured or (np.all(cmd.q_target >= lower - 0.02)
                                          and np.all(cmd.q_target <= upper + 0.02)))
                 and np.all(np.abs(cmd.dq_target) <= self.velocity)
                 and np.all(np.abs(cmd.tau_ff) <= self.effort)
                 and np.all((cmd.kp >= 0) & (cmd.kp <= 400))
                 and np.all((cmd.kd >= 0) & (cmd.kd <= 10))
                 and cmd.control_mode in {"zero", "init", "policy"}
                 and cmd.source_time_ns is not None)
        fresh = (valid
                 and 0 <= now_ns - cmd.source_time_ns <= self.timeout_ns
                 and cmd.source_time_ns > self.last_source_ns
                 and (cmd.sequence > self.last_sequence
                      or (self.latched and cmd.control_mode == "zero")))
        if valid and not fresh and self.hold_through_gaps:
            return False  # A late or repeated command after a gap; the newer one stays in force.
        if not fresh:
            self.latch("invalid, expired or out-of-order command")
            return False
        self.command = cmd
        self.last_sequence, self.last_source_ns = cmd.sequence, cmd.source_time_ns
        if self.latched and cmd.control_mode == "zero" and cmd.control_generation > self._stop_generation:
            self._zero_ack_generation = cmd.control_generation
        if (not self._selected and self.arm_time_ns and cmd.control_mode == "init"
                and cmd.source_time_ns >= self.arm_time_ns
                and (not self.latched or (self._zero_ack_generation is not None
                     and cmd.control_generation > self._zero_ack_generation))
                and self.ready(now_ns)):
            self.latched = self.fault = False
            self.reason = "active" if self.active else "arming"
            self.arm_pending = not self.active
            self.arm_time_ns = 0
        return True

    def takeover_finished(self, success: bool, now_ns: int):
        self.arm_pending = False
        if success:
            self.active = True
            if self.latched or not self.ready(now_ns):
                self.latch("takeover completed without fresh ready state")
            else:
                self.reason = "active"
        else:
            self.latch("failed to release high-level mode")

    def output(self, now_ns: int) -> LowCmdMessage | None:
        if self.active or self.arm_pending:
            state_late = self.state is None or now_ns - self.state_time_ns > self.timeout_ns
            command_late = self.command is None or now_ns - self.command.source_time_ns > self.timeout_ns
            state_lost = self.state is None or now_ns - self.state_time_ns > MAX_HOLD_GAP_NS
            command_lost = self.command is None or now_ns - self.command.source_time_ns > MAX_HOLD_GAP_NS
            if (self.hold_through_gaps and self.active and not self.latched
                    and not state_lost and not command_lost):
                # Keep the last PD targets through the gap; fresh data resumes control without a new A.
                self.reason = ("LowState gap: holding last targets" if state_late
                               else "Tracker command gap: holding last targets" if command_late else "active")
            elif state_late:
                self.latch("LowState timeout")
            elif command_late:
                self.latch("Tracker command timeout")
        if not self.active:
            return None
        if not self.latched:
            return self.command
        q = self.state.joint_positions.copy() if self.state is not None else np.zeros(self.count)
        zeros = np.zeros(self.count, dtype=np.float32)
        if self.fault:
            return LowCmdMessage(q, zeros, zeros, zeros, self.damping_kd)
        return LowCmdMessage(q, zeros, zeros, self.command.kp, self.command.kd)
