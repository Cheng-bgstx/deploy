"""Operator key messages for robot bridges that have no handset in reach (ET1 keyboard runs).

Senders publish to PORTS["operator"] on the bridge host; the bridge ORs the fresh keys of every
sender into the handset, keyed by sender id so one sender's "released" never splits another's press.
"""
from __future__ import annotations

import os
import struct
import time

import zmq

from sim2real.utils.common import PORTS

# Sender monotonic ns (same host as the bridge), Unitree keys, sender id.
OPERATOR_FORMAT = "<QIQ"
# Unitree wireless_remote key bits.
KEY_A, KEY_START, KEY_SELECT = 1 << 8, 1 << 2, 1 << 3


class OperatorKeyPublisher:
    def __init__(self, host: str = "127.0.0.1", port: int = PORTS["operator"]):
        self.sender = int.from_bytes(os.urandom(8), "little")
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{port}")

    def send(self, keys: int) -> None:
        self.socket.send(struct.pack(OPERATOR_FORMAT, time.monotonic_ns(), keys, self.sender), flags=zmq.DONTWAIT)

    def press(self, keys: int, *, settle_s: float = 0.6, hold_s: float = 0.15, period_s: float = 0.05) -> None:
        """One press: released until the subscription is up, then held for hold_s, then released."""
        for value, duration in ((0, settle_s), (keys, hold_s)):
            end = time.monotonic() + duration
            while time.monotonic() < end:
                self.send(value)
                time.sleep(period_s)
        self.send(0)

    def close(self) -> None:
        self.socket.close(linger=0)
