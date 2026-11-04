"""Read-only LowState reception monitor, independent of the Bridge.

Logs every gap between LowState arrivals longer than ``gap_ms`` with the robot tick jump. Run beside the Bridge:
a gap seen here as well points at the robot or the network; a gap only the Bridge logs points at the Bridge
process. It never publishes anything.
"""
import signal
import time
from dataclasses import dataclass

import tyro
from loguru import logger

from sim2real.config.robots import get_robot_cfg
from sim2real.config.robots.base import get_unitree_dds_family


@dataclass
class Args:
    interface: str = "enp5s0"
    robot: str = "g1"
    gap_ms: float = 30.0
    report_s: float = 30.0


def main(args: Args):
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

    if get_unitree_dds_family(get_robot_cfg(args.robot).name) != "hg":
        raise NotImplementedError("only the hg LowState is supported")
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    def stop_on_signal(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_signal)
    ChannelFactoryInitialize(0, args.interface)
    gap_ns = int(args.gap_ms * 1e6)
    last = [None, None]  # arrival ns, tick
    counts = dict(samples=0, gaps=0, worst_ms=0.0)

    def on_state(msg):
        now = time.monotonic_ns()
        if last[0] is not None and now - last[0] > gap_ns:
            ms = (now - last[0]) / 1e6
            counts["gaps"] += 1
            counts["worst_ms"] = max(counts["worst_ms"], ms)
            logger.warning("LowState gap {:.0f} ms ({} robot ticks)", ms, int(msg.tick) - last[1])
        last[0], last[1] = now, int(msg.tick)
        counts["samples"] += 1

    reader = ChannelSubscriber("rt/lowstate", LowState_)
    reader.Init(on_state, 0)
    logger.info("Monitoring LowState on {}; logging gaps over {:g} ms", args.interface, args.gap_ms)
    try:
        while True:
            time.sleep(args.report_s)
            logger.info("{samples} samples, {gaps} gaps, worst {worst_ms:.0f} ms", **counts)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(tyro.cli(Args))
