from __future__ import annotations

from typing import Literal

from sim2real.config.robots.base import RobotCfg, normalize_robot_name
from sim2real.rl_policy.robot_io.base import RobotIO
from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO


def create_robot_io(
    *,
    mode: Literal["zmq", "inline"],
    robot_name: str,
    robot_cfg: RobotCfg,
    interface: str,
    robot_ip: str = "127.0.0.1",
    options: dict | None = None,
) -> RobotIO:
    if mode == "zmq":
        return ZMQRobotIO(robot_cfg)

    if mode != "inline":
        raise ValueError(f"Unsupported robot_io: {mode}")

    normalized_name = normalize_robot_name(robot_name)
    if robot_cfg.name == "a3":
        from sim2real.rl_policy.robot_io.a3 import A3RobotIO

        return A3RobotIO(robot_cfg, **(options or {}))
    if robot_cfg.name == "pm01":
        from sim2real.rl_policy.robot_io.pm01 import PM01RobotIO

        return PM01RobotIO(robot_cfg, **(options or {}))
    if robot_cfg.name == "t800":
        from sim2real.rl_policy.robot_io.t800 import T800RobotIO

        return T800RobotIO(robot_cfg, **(options or {}))
    if robot_cfg.name == "x2":
        from sim2real.rl_policy.robot_io.x2 import X2RobotIO

        return X2RobotIO(robot_cfg, **(options or {}))
    if robot_cfg.name == "oli":
        from sim2real.rl_policy.robot_io.oli import OliRobotIO

        return OliRobotIO(robot_cfg, robot_ip=robot_ip)
    if normalized_name == "g1":
        from sim2real.rl_policy.robot_io.g1 import G1RobotIO

        return G1RobotIO(robot_cfg, interface=interface)

    raise NotImplementedError(
        f"robot_io='inline' is not implemented for robot={robot_name!r}"
    )
