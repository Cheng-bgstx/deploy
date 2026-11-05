"""Tracker action post-processing on joint position targets.

Same formula as Scalecar's ``PositionController``
(``Scalecar_v2/source/scalecar_deploy/scalecar_deploy/control.py``), which
training also applied:

    desired = clip(default + scale * action, soft_lower, soft_upper)
    target  = prev + clip(desired - prev, +-max_target_velocity_rad_s * control_dt)

``prev`` is the previous target; after ``reset()`` the next call takes it from
the measured joint positions instead.

Scalecar raises on a non-finite action or measured position. Here a non-finite
value never becomes ``prev`` (it would keep every later target non-finite until
the next reset): a joint whose raw target is not finite keeps its previous
target, and a joint without a finite previous target (its measured position was
not finite on entry) takes its soft-clipped target without the rate limit, once;
the rate limit continues from there. Both are reported in a warning at most once per second. With finite inputs the
targets are bitwise Scalecar's.

It is opt-in per policy through optional top-level YAML keys; a policy without
them gets ``None`` from ``JointTargetPostprocessor.from_policy_config`` and its
targets are left untouched:

    soft_joint_pos_limits:        # name (regex) -> [lower, upper]
      left_knee_joint: [-0.0131, 2.8056]
    max_target_velocity_rad_s: 15.0   # times the runtime control period (1 / rl_rate)

Either key alone enables only its half.

Pose targets the robot interpolates to before policy control (the
``shaping_target: motion_first_frame`` Shaping target and the
``reference_entry_s`` transition) are clipped by
``JointTargetPostprocessor.pose_limits_from_policy_config``: the optional
``shaping_joint_pos_limits`` (same format), else ``soft_joint_pos_limits``.
``shaping_joint_pos_limits`` never clips policy targets. A robot IO layer may
still clip them: ``NamedJointCodec`` (Q1) clips every command to the model joint
range under its default ``policy_limit_mode: clip``, so there the rate limit would
run on targets the robot never receives. Its ``pass`` and ``feedforward`` modes
handle out-of-range policy targets instead (``robot_io_options``).
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from loguru import logger

from sim2real.utils.strings import resolve_matching_names_values


WARNING_INTERVAL_S = 1.0
POLICY_LIMITS_KEY = "soft_joint_pos_limits"
SHAPING_LIMITS_KEY = "shaping_joint_pos_limits"


def _limits_from_config(
    limits_cfg: dict[str, Any],
    key: str,
    joint_names: list[str],
    policy_joint_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(joint_names), -np.inf)
    upper = np.full(len(joint_names), np.inf)
    indices, names, bounds = resolve_matching_names_values(
        dict(limits_cfg), joint_names, preserve_order=True, strict=True
    )
    for index, name, bound in zip(indices, names, bounds):
        low, high = (float(value) for value in bound)
        if not (np.isfinite(low) and np.isfinite(high) and low <= high):
            raise ValueError(
                f"{key}[{name}] must be finite [lower, upper] "
                f"with lower <= upper, got {bound}"
            )
        lower[index] = low
        upper[index] = high
    missing = [name for name in policy_joint_names if name not in names]
    if missing:
        raise ValueError(f"{key} must cover every policy joint; missing {missing}")
    return lower, upper


class JointTargetPostprocessor:
    """Post-processes full joint target vectors given in ``joint_names`` order."""

    def __init__(
        self,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        max_delta: float | None,
        joint_names: Sequence[str] | None = None,
    ):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.max_delta = max_delta
        self.joint_names = None if joint_names is None else list(joint_names)
        # The previous target, finite wherever ``_known``; None until the first call after reset().
        self._previous: np.ndarray | None = None
        self._known: np.ndarray | None = None
        self._next_warning = -np.inf
        self._suppressed_warnings = 0

    @classmethod
    def from_policy_config(
        cls,
        policy_config: dict[str, Any],
        *,
        joint_names: Sequence[str],
        policy_joint_names: Sequence[str],
        control_dt: float,
    ) -> "JointTargetPostprocessor | None":
        """Build from the policy YAML, or return None when neither key is set.

        ``joint_names`` is the order of the target vectors (the robot order);
        the soft limits are matched by name, so the YAML order does not matter,
        but they must cover every joint in ``policy_joint_names``. Joints
        outside the policy are left unbounded.
        """
        limits_cfg = policy_config.get(POLICY_LIMITS_KEY)
        max_velocity = policy_config.get("max_target_velocity_rad_s")
        if limits_cfg is None and max_velocity is None:
            return None

        joint_names = list(joint_names)
        lower = np.full(len(joint_names), -np.inf)
        upper = np.full(len(joint_names), np.inf)
        if limits_cfg is not None:
            lower, upper = _limits_from_config(
                limits_cfg, POLICY_LIMITS_KEY, joint_names, policy_joint_names
            )

        max_delta = None
        if max_velocity is not None:
            max_velocity = float(max_velocity)
            if not (np.isfinite(max_velocity) and max_velocity > 0.0):
                raise ValueError(
                    f"max_target_velocity_rad_s must be positive, got {max_velocity}"
                )
            max_delta = max_velocity * float(control_dt)
        return cls(lower=lower, upper=upper, max_delta=max_delta, joint_names=joint_names)

    @classmethod
    def pose_limits_from_policy_config(
        cls,
        policy_config: dict[str, Any],
        *,
        joint_names: Sequence[str],
        policy_joint_names: Sequence[str],
    ) -> "JointTargetPostprocessor | None":
        """Clip-only limits for pre-policy pose targets, or None when unset.

        Uses ``shaping_joint_pos_limits`` when set, else ``soft_joint_pos_limits``.
        Only ``clip()`` is meant to be called on the result.
        """
        key = SHAPING_LIMITS_KEY if policy_config.get(SHAPING_LIMITS_KEY) is not None else POLICY_LIMITS_KEY
        limits_cfg = policy_config.get(key)
        if limits_cfg is None:
            return None
        joint_names = list(joint_names)
        lower, upper = _limits_from_config(limits_cfg, key, joint_names, policy_joint_names)
        return cls(lower=lower, upper=upper, max_delta=None, joint_names=joint_names)

    def clip(self, joint_pos: np.ndarray) -> np.ndarray:
        """Clip joint positions to the soft limits."""
        return np.clip(joint_pos, self.lower, self.upper)

    def reset(self) -> None:
        """Take the next call's previous target from the measured joint positions."""
        self._previous = None

    def __call__(self, raw_target: np.ndarray, measured_joint_pos: np.ndarray) -> np.ndarray:
        desired = self.clip(raw_target)
        if self.max_delta is None:
            return desired
        if self._previous is None:
            self._previous = np.zeros(desired.shape)
            self._known = np.zeros(desired.shape, dtype=bool)
        if not self._known.all():
            measured = np.asarray(measured_joint_pos, dtype=np.float64)
            seed = ~self._known & np.isfinite(measured)
            self._previous[seed] = measured[seed]
            self._known |= seed
        target = self._previous + np.clip(
            desired - self._previous, -self.max_delta, self.max_delta
        )
        valid = np.isfinite(raw_target)
        if not (valid.all() and self._known.all()):
            held = self._known & ~valid
            unlimited = ~self._known
            target[held] = self._previous[held]
            target[unlimited] = desired[unlimited]
            self._warn(held, unlimited)
            self._known |= unlimited & valid
        self._previous[self._known] = target[self._known]
        return target

    def _warn(self, held: np.ndarray, unlimited: np.ndarray) -> None:
        now = time.monotonic()
        if now < self._next_warning:
            self._suppressed_warnings += 1
            return
        parts = []
        if held.any():
            parts.append(f"non-finite target for {self._names(held)}; keeping their previous targets")
        if unlimited.any():
            parts.append(
                f"no finite measured position for {self._names(unlimited)} to start the rate limit from; "
                "sending their soft-clipped targets unlimited"
            )
        suppressed = (
            f" ({self._suppressed_warnings} more since the last warning)" if self._suppressed_warnings else ""
        )
        logger.warning("Joint target post-processing: {}{}", "; ".join(parts), suppressed)
        self._next_warning = now + WARNING_INTERVAL_S
        self._suppressed_warnings = 0

    def _names(self, mask: np.ndarray) -> list:
        indices = np.flatnonzero(mask)
        if self.joint_names is None:
            return indices.tolist()
        return [self.joint_names[i] for i in indices]
