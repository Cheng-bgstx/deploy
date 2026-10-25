"""Bounded latest-value IPC without a Queue feeder thread in the control process."""

from __future__ import annotations

import pickle
from typing import Any


class LatestMailbox:
    """One producer and one consumer; snapshots never wait for a busy reader."""

    def __init__(self, context: Any, capacity: int = 262144):
        self.capacity = capacity
        self.data = context.RawArray("B", capacity)
        self.size = context.RawValue("I", 0)
        self.version = context.RawValue("Q", 0)
        self.lock = context.Lock()

    def publish(self, value: Any) -> bool:
        payload = pickle.dumps(value, protocol=5)
        if len(payload) > self.capacity:
            raise ValueError(f"IPC snapshot exceeds {self.capacity} bytes")
        if not self.lock.acquire(block=False):
            return False
        try:
            memoryview(self.data).cast("B")[: len(payload)] = payload
            self.size.value = len(payload)
            self.version.value += 1
            return True
        finally:
            self.lock.release()

    def read(self, after: int = 0) -> tuple[int, Any | None]:
        if not self.lock.acquire(block=False):
            return after, None
        try:
            version = self.version.value
            if version <= after:
                return after, None
            payload = bytes(memoryview(self.data).cast("B")[: self.size.value])
        finally:
            self.lock.release()
        return version, pickle.loads(payload)
