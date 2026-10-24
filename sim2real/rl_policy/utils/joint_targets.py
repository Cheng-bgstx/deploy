"""Tracker action post-processing on joint position targets.

Same formula as Scalecar's ``PositionController``
(``Scalecar_v2/source/scalecar_deploy/scalecar_deploy/control.py``), which
training also applied:

    desired = clip(default + scale * action, soft_lower, soft_upper)
    target  = prev + clip(desired - prev, +-max_target_velocity_rad_s * control_dt)

``prev`` is the previous target; after ``reset()`` the next call takes it from
the measured joint positions instead.

It is opt-in per policy through optional top-level YAML keys; a policy without
them gets ``None`` from ``JointTargetPostprocessor.from_policy_config`` and its
targets are left untouched:

    soft_joint_pos_limits:        # name (regex) -> [lower, upper]
      left_knee_joint: [-0.0131, 2.8056]
    max_target_velocity_rad_s: 15.0   # times the runtime control period (1 / rl_rate)

Either key alone enables only its half. ``BasePolicy`` also clips the
``shaping_target: motion_first_frame`` Shaping target with the same soft limits.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from sim2real.utils.strings import resolve_matching_names_values


class JointTargetPostprocessor:
    """Post-processes full joint target vectors given in ``joint_names`` order."""

    def __init__(
        self,
        *,
        lower: np.ndarray,
        upper: np.ndarray,
        max_delta: float | None,
    ):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.max_delta = max_delta
        self._previous: np.ndarray | None = None

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
        limits_cfg = policy_config.get("soft_joint_pos_limits")
        max_velocity = policy_config.get("max_target_velocity_rad_s")
        if limits_cfg is None and max_velocity is None:
            return None

        joint_names = list(joint_names)
        lower = np.full(len(joint_names), -np.inf)
        upper = np.full(len(joint_names), np.inf)
        if limits_cfg is not None:
            indices, names, bounds = resolve_matching_names_values(
                dict(limits_cfg), joint_names, preserve_order=True, strict=True
            )
            for index, name, bound in zip(indices, names, bounds):
                low, high = (float(value) for value in bound)
                if not (np.isfinite(low) and np.isfinite(high) and low <= high):
                    raise ValueError(
                        f"soft_joint_pos_limits[{name}] must be finite [lower, upper] "
                        f"with lower <= upper, got {bound}"
                    )
                lower[index] = low
                upper[index] = high
            missing = [name for name in policy_joint_names if name not in names]
            if missing:
                raise ValueError(
                    f"soft_joint_pos_limits must cover every policy joint; missing {missing}"
                )

        max_delta = None
        if max_velocity is not None:
            max_velocity = float(max_velocity)
            if not (np.isfinite(max_velocity) and max_velocity > 0.0):
                raise ValueError(
                    f"max_target_velocity_rad_s must be positive, got {max_velocity}"
                )
            max_delta = max_velocity * float(control_dt)
        return cls(lower=lower, upper=upper, max_delta=max_delta)

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
            self._previous = np.array(measured_joint_pos, dtype=np.float64, copy=True)
        self._previous = self._previous + np.clip(
            desired - self._previous, -self.max_delta, self.max_delta
        )
        return self._previous.copy()
