import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Optional

import tyro
from loguru import logger

from sim2real.rl_policy.base_policy import BasePolicyArgs, BasePolicy
from sim2real.rl_policy.controllers.keyboard import KeyboardController
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController
from sim2real.rl_policy.utils.motion import (
    MotionData,
    MotionDataset,
    motion_dataset_first_motion,
)
from sim2real.rl_policy.utils.motion_buffer import (
    RealtimeMotionBuffer,
    RealtimeSmplMotionBuffer,
)

np.set_printoptions(precision=3, suppress=True, linewidth=1000)


DEFAULT_MOTION_ZMQ_CONNECT = "tcp://127.0.0.1:28701"
DEFAULT_SMPL_MOTION_ZMQ_CONNECT = "tcp://127.0.0.1:28702"
SMPL_ENCODER_OBSERVATIONS = {
    "sonic_smpl_official_encoder_input",
    "sonic_smpl_joints_multi_future_local",
    "sonic_smpl_root_ori_b_multi_future",
    "sonic_joint_pos_multi_future_wrist_for_smpl",
}


def _policy_uses_smpl_motion(policy_config: dict[str, Any]) -> bool:
    observation_cfg = policy_config.get("observation", {})
    if not isinstance(observation_cfg, dict):
        return False

    for obs_items in observation_cfg.values():
        if not isinstance(obs_items, dict):
            continue
        for obs_name, obs_config in obs_items.items():
            target = obs_name
            if isinstance(obs_config, dict):
                target = str(obs_config.get("_target_", obs_name))
            if target.split(".")[-1] in SMPL_ENCODER_OBSERVATIONS:
                return True
    return False


def _default_motion_connect(motion_backend: str) -> str:
    if motion_backend == "smpl_zmq":
        return DEFAULT_SMPL_MOTION_ZMQ_CONNECT
    return DEFAULT_MOTION_ZMQ_CONNECT


def _resolve_motion_zmq_connect(
    *,
    motion_backend: str,
    requested_connect: Optional[str],
    configured_connect: Optional[str],
) -> str:
    connect = requested_connect or configured_connect or _default_motion_connect(motion_backend)
    if motion_backend == "smpl_zmq" and connect == DEFAULT_MOTION_ZMQ_CONNECT:
        logger.warning(
            "motion_backend=smpl_zmq received the normal motion endpoint "
            f"{DEFAULT_MOTION_ZMQ_CONNECT}; using SMPL endpoint "
            f"{DEFAULT_SMPL_MOTION_ZMQ_CONNECT} instead."
        )
        return DEFAULT_SMPL_MOTION_ZMQ_CONNECT
    return connect


def _apply_runtime_motion_config(
    policy_config,
    motion_backend: Optional[str],
    motion_path: Optional[str],
    motion_zmq_connect: Optional[str],
    motion_zmq_hwm: int,
    motion_dt_s: float,
    motion_tolerance_s: float,
    planner_checkpoint: Optional[str] = None,
):
    policy_config = deepcopy(policy_config)
    motion_cfg = policy_config.setdefault("motion", {})
    configured_backend = str(motion_cfg.get("motion_backend", "npz")).lower().strip()
    resolved_backend = (
        str(motion_backend).lower().strip()
        if motion_backend is not None
        else configured_backend
    )

    if _policy_uses_smpl_motion(policy_config) and resolved_backend == "zmq":
        logger.warning(
            "Policy uses SONIC SMPL encoder input; switching motion_backend=zmq "
            "to motion_backend=smpl_zmq."
        )
        resolved_backend = "smpl_zmq"

    motion_cfg["motion_backend"] = resolved_backend
    if resolved_backend == "npz" and motion_path is not None:
        motion_cfg["motion_path"] = motion_path
        motion_cfg["motion_dt_s"] = motion_dt_s
    if resolved_backend == "planner":
        # The Motion gives the Hold Reference (frame 0) and the Shaping target.
        if motion_path is not None:
            motion_cfg["motion_path"] = motion_path
        if planner_checkpoint is not None:
            motion_cfg.setdefault("planner", {})["checkpoint"] = planner_checkpoint
    if resolved_backend in {"zmq", "smpl_zmq"}:
        motion_cfg["motion_zmq_connect"] = _resolve_motion_zmq_connect(
            motion_backend=resolved_backend,
            requested_connect=motion_zmq_connect,
            configured_connect=motion_cfg.get("motion_zmq_connect"),
        )
        motion_cfg["motion_zmq_hwm"] = motion_zmq_hwm
        motion_cfg["motion_dt_s"] = motion_dt_s
        motion_cfg["motion_tolerance_s"] = motion_tolerance_s

    return policy_config


class Tracking(BasePolicy):
    args: "TrackingArgs"

    def prepare_policy_config(self, policy_config):
        policy_config = super().prepare_policy_config(policy_config)
        policy_config = _apply_runtime_motion_config(
            policy_config=policy_config,
            motion_backend=self.args.motion_backend,
            motion_path=self.args.motion_path,
            motion_zmq_connect=self.args.motion_zmq_connect,
            motion_zmq_hwm=self.args.motion_zmq_hwm,
            motion_dt_s=1 / self.args.rl_rate,
            motion_tolerance_s=self.args.motion_tolerance_s,
            planner_checkpoint=self.args.planner_checkpoint,
        )
        motion_cfg = policy_config.get("motion", {})
        resolved_backend = str(motion_cfg.get("motion_backend", ""))
        if resolved_backend in {"zmq", "smpl_zmq"}:
            logger.info(
                f"Using runtime motion_backend={resolved_backend} "
                f"connect={motion_cfg.get('motion_zmq_connect')}"
            )
        return policy_config

    def setup_observations(self, obs_cfg):
        self._init_motion_backend()
        super().setup_observations(obs_cfg)

    def _init_motion_backend(self) -> None:
        self.motion_config = dict(self.policy_config.get("motion", {}))
        self.motion_data: MotionData | None = None
        self.motion_future_steps = np.asarray(
            self.motion_config.get("future_steps", []),
            dtype=int,
        )
        if self.motion_future_steps.ndim != 1:
            raise ValueError(
                f"motion.future_steps must be 1D, got shape={self.motion_future_steps.shape}"
            )

        self.motion_backend = str(
            self.motion_config.get("motion_backend", "npz")
        ).lower().strip()
        self.motion_config["motion_backend"] = self.motion_backend

        if self.motion_backend == "npz":
            motion_path = self.motion_config.get("motion_path")
            if motion_path is None:
                raise ValueError("motion_path is required for npz motion backend")
            self.motion_dataset = MotionDataset.create_from_path(
                motion_path,
                robot_cfg=self.robot_cfg,
                mjcf_path=self.motion_config.get("mjcf_path"),
            )
            self.motion_dataset = motion_dataset_first_motion(self.motion_dataset)
            assert self.motion_dataset.num_motions == 1, "Only one motion is supported"
            self.motion_ids = np.asarray([0], dtype=int)
            self.motion_t = np.asarray([0], dtype=int)
            self.motion_length = self.motion_dataset.num_steps
            self.motion_joint_names = list(self.motion_dataset.joint_names)
            self.motion_body_names = list(self.motion_dataset.body_names)
            return

        if self.motion_backend == "zmq":
            self.motion_buffer = RealtimeMotionBuffer(
                robot_cfg=self.robot_cfg,
                future_steps=self.motion_future_steps,
                motion_zmq_connect=self.motion_config.get(
                    "motion_zmq_connect", DEFAULT_MOTION_ZMQ_CONNECT
                ),
                motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
                dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
                tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
                root_body_name=str(self.motion_config.get("root_body_name", "pelvis")),
            )
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = list(self.motion_buffer.body_names)
            return

        if self.motion_backend == "planner":
            self.motion_buffer = self._create_planner_reference()
            self.motion_data = self.motion_buffer.motion_data  # Hold until the first Chunk
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = list(self.motion_buffer.body_names)
            return

        if self.motion_backend == "smpl_zmq":
            self.motion_buffer = RealtimeSmplMotionBuffer(
                robot_cfg=self.robot_cfg,
                future_steps=self.motion_future_steps,
                motion_zmq_connect=self.motion_config.get(
                    "motion_zmq_connect", DEFAULT_SMPL_MOTION_ZMQ_CONNECT
                ),
                motion_zmq_hwm=int(self.motion_config.get("motion_zmq_hwm", 1)),
                dt_s=float(self.motion_config.get("motion_dt_s", 0.02)),
                tolerance_s=float(self.motion_config.get("motion_tolerance_s", 0.04)),
            )
            self.motion_joint_names = list(self.motion_buffer.joint_names)
            self.motion_body_names = []
            return

        raise ValueError(f"Unsupported motion_backend: {self.motion_backend}")

    def _create_planner_reference(self):
        """The Planner as Reference Source (``sim2real/planner``).

        Hold shows the Motion's frame 0 (``--motion-path`` / ``motion.motion_path``),
        as File Playback does while paused. ``motion.planner`` holds the worker
        options (checkpoint, checkpoint_sha256, device, seed; defaults: the pinned
        ModelScope d1146 checkpoint on cuda:0). Depth, worker and clock come from
        ``_planner_io``.
        """
        from sim2real.planner.provider import JOINT_NAMES, PlannerProvider
        from sim2real.planner.reference import PlannerReference, hold_motion_data
        from sim2real.planner.worker import worker_options

        if abs(1.0 / self.args.rl_rate - 0.02) > 1e-9:
            raise ValueError("motion_backend=planner runs at the Planner's 50 Hz (rl_rate 50)")
        missing = sorted(set(JOINT_NAMES) - set(self.robot_cfg.joint_names))
        if missing:
            raise ValueError(f"motion_backend=planner needs the robot joints {missing}")
        self._planner_joint_index = np.asarray([self.robot_cfg.joint_names.index(n) for n in JOINT_NAMES])

        motion_path = self.motion_config.get("motion_path")
        if motion_path is None:
            raise ValueError(
                "motion_backend=planner needs a Motion for the Hold Reference: "
                "pass --motion-path or set motion.motion_path"
            )
        dataset = motion_dataset_first_motion(
            MotionDataset.create_from_path(
                motion_path, robot_cfg=self.robot_cfg, mjcf_path=self.motion_config.get("mjcf_path")
            )
        )
        root = str(self.motion_config.get("root_body_name", "pelvis"))
        first = np.zeros(1, dtype=int)
        frame0 = dataset.get_slice(first, first, first)
        hold = hold_motion_data(
            joint_pos=[frame0.joint_pos[0, 0, dataset.joint_names.index(n)] for n in JOINT_NAMES],
            pelvis_pos=frame0.body_pos_w[0, 0, dataset.body_names.index(root)],
            pelvis_quat_wxyz=frame0.body_quat_w[0, 0, dataset.body_names.index(root)],
            future_steps=self.motion_future_steps,
        )
        io = self._planner_io(worker_options(self.motion_config.get("planner")))
        self._planner_clock = io.clock
        provider = PlannerProvider(JOINT_NAMES, source=io.source, worker=io.worker)
        return PlannerReference(provider, hold=hold, future_steps=self.motion_future_steps)

    def _planner_io(self, options: dict):
        """Depth source, Planner worker and clock for ``motion_backend=planner``.

        On the robot this is the Host depth receiver, ``GpuPlannerWorker(options)``
        and ``time.monotonic`` (dixiatielu/sim2real_scalecar#8); the vehicle sim2sim
        overrides it with rendered depth on simulation time.
        """
        raise NotImplementedError(
            "motion_backend=planner on the robot needs the Host depth receiver (dixiatielu/sim2real_scalecar#8)"
        )

    def close_reference_source(self) -> None:
        """Stops the Planner worker process (no-op for the other backends)."""
        if self.motion_backend == "planner":
            self.motion_buffer.close()

    def run(self):
        try:
            super().run()
        finally:
            self.close_reference_source()

    def motion_first_frame_joint_pos(self) -> np.ndarray:
        """Frame 0 of the Motion, in robot joint order (Shaping target).

        File Playback uses the Motion it plays. A streamed Reference (zmq
        backends, the Planner) still needs a Motion file for its first frame:
        ``--motion-path`` or ``motion.motion_path``.
        """
        if self.motion_backend == "npz":
            dataset = self.motion_dataset
        else:
            motion_path = self.args.motion_path or self.motion_config.get("motion_path")
            if motion_path is None:
                raise ValueError(
                    "shaping_target: motion_first_frame with "
                    f"motion_backend={self.motion_backend} needs a Motion for its "
                    "first frame: pass --motion-path or set motion.motion_path"
                )
            dataset = motion_dataset_first_motion(
                MotionDataset.create_from_path(
                    motion_path,
                    robot_cfg=self.robot_cfg,
                    mjcf_path=self.motion_config.get("mjcf_path"),
                )
            )
        first = np.zeros(1, dtype=int)
        joint_pos = dataset.get_slice(first, first, first).joint_pos[0, 0]
        q = self.default_dof_angles.copy()
        for index, name in enumerate(self.action_manager.joint_names):
            if name in dataset.joint_names:
                q[index] = joint_pos[dataset.joint_names.index(name)]
        return q

    def _update_motion_data(self, *, paused: bool = False) -> None:
        if self.motion_backend == "npz":
            motion_steps = self.motion_future_steps
            if paused:
                motion_steps = np.zeros_like(self.motion_future_steps)
            self.motion_data = self.motion_dataset.get_slice(
                self.motion_ids,
                self.motion_t,
                motion_steps,
            )
            if paused:
                self.motion_data.joint_vel[:] = 0.0
                self.motion_data.body_lin_vel_w[:] = 0.0
                self.motion_data.body_ang_vel_w[:] = 0.0
            return

        if self.motion_backend == "planner":
            from sim2real.planner.provider import PlannerState

            state = PlannerState(
                joint_pos=np.asarray(self.state_processor.joint_pos, dtype=np.float64)[self._planner_joint_index],
                root_quat_wxyz=np.asarray(self.state_processor.root_quat_w, dtype=np.float64).copy(),
            )
            self.motion_data = self.motion_buffer.tick(state, self._planner_clock())
            return

        self.motion_data = self.motion_buffer.get_obs()
        self.motion_joint_names = list(self.motion_buffer.joint_names)
        if self.motion_backend == "zmq":
            self.motion_body_names = list(self.motion_buffer.body_names)
        else:
            self.motion_body_names = []

    def _reset_motion(self) -> None:
        if self.motion_backend == "planner":
            # Entering the Tracker does not run a Planner cycle; the next tick does.
            self.motion_data = self.motion_buffer.motion_data
            return
        if self.motion_backend == "npz":
            self.motion_t[:] = 0
        self._update_motion_data(paused=True)

    def restart_motion(self) -> None:
        if self.motion_backend != "npz":
            return
        self.motion_t[:] = 0
        self._update_motion_data(paused=True)

    def reset(self) -> None:
        self._reset_motion()
        super().reset()

    def toggle_paused(self, *, source: str) -> None:
        paused = not bool(self.state_dict.get("paused", False))
        self.state_dict["paused"] = paused
        logger.info(f"Paused state toggled to {paused} via {source}")

    def _motion_update_state(self) -> dict[str, Any]:
        if self.motion_backend in {"zmq", "smpl_zmq"}:
            state = dict(self.state_dict)
            state["paused"] = False
            return state
        return self.state_dict

    def update(self):
        state = self._motion_update_state()
        paused = bool(state.get("paused", False))
        if not paused and self.motion_backend == "npz":
            self.motion_t += 1
            if self.motion_length > 0 and self.motion_t[0] >= self.motion_length:
                self.motion_t[:] = self.motion_length - 1
                state["paused"] = True
                self.state_dict["paused"] = True
        self._update_motion_data(paused=paused)
        for obs_group in self.observations.values():
            for obs_func in obs_group.funcs.values():
                obs_func.update(state)

    def _record_runtime_fields(self) -> dict[str, Any]:
        motion_t = getattr(self, "motion_t", None)
        value = -1
        if motion_t is not None:
            value = int(np.asarray(motion_t).reshape(-1)[0])
        return {"motion_t": value}

    def _record_metadata_fields(self) -> dict[str, Any]:
        return {
            "motion_backend": self.motion_backend,
            "motion_path": str(self.motion_config.get("motion_path", "")),
        }

    def process_controllers(self) -> None:
        super().process_controllers()

        extra_keys = self.controller.get_extra_keys()
        if not extra_keys:
            return

        if isinstance(self.controller, KeyboardController) and (
            "space" in extra_keys or " " in extra_keys
        ):
            self.toggle_paused(source="keyboard:space")
        elif isinstance(self.controller, UnitreeJoystickController) and "B" in extra_keys:
            self.toggle_paused(source="joystick:B")


@dataclass
class TrackingArgs(BasePolicyArgs):
    """Robot."""

    motion_backend: Optional[Literal["npz", "zmq", "smpl_zmq", "planner"]] = None
    motion_path: Optional[str] = None
    planner_checkpoint: Optional[str] = None
    """motion_backend=planner: checkpoint path (default: the pinned ModelScope d1146 file)."""
    motion_zmq_connect: Optional[str] = None
    motion_zmq_hwm: int = 1
    motion_tolerance_s: float = 0.04


if __name__ == "__main__":
    args = tyro.cli(TrackingArgs)
    policy = Tracking(args=args)
    policy.run()
