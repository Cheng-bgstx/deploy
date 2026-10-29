"""Handset re-published by a robot bridge over ZMQ (scripts/et1/hdep_bridge.py).

Same A -> A -> Start / Select flow as the G1 Unitree joystick; the bridge already maps
the handset to the Unitree key layout, so decoding is shared with UnitreeJoystickController.
"""
from __future__ import annotations

import struct
import threading
from types import SimpleNamespace

import zmq
from loguru import logger

from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController, UnitreeJoystickState
from sim2real.utils.common import PORTS

# HDEP gamepad stamp (ns), Unitree keys, lx, ly, rx, ry; see scripts/et1/hdep_bridge.py.
GAMEPAD_FORMAT = "<QIffff"


class BridgeJoystickController(UnitreeJoystickController):
    name = "bridge_joystick"

    def __init__(self, perf_dict: dict[str, float], host: str = "127.0.0.1", port: int = PORTS["gamepad"]) -> None:
        self.perf_dict = perf_dict
        self._wc_lock = threading.Lock()
        self._joystick_thread_stop = threading.Event()
        self._pending_extra_keys: list[str] = []
        self._wc_msg: UnitreeJoystickState | None = None
        self._last_wc_msg = UnitreeJoystickState()
        self._closing = False
        self._g1_controls = True
        self._socket = zmq.Context.instance().socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._socket.connect(f"tcp://{host}:{port}")
        self._joystick_thread = threading.Thread(target=self._receive, daemon=True)
        self._joystick_thread.start()

    def _receive(self) -> None:
        while not self._joystick_thread_stop.is_set():
            try:
                data = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break
            try:
                _, keys, lx, ly, rx, ry = struct.unpack(GAMEPAD_FORMAT, data)
            except struct.error as exc:
                logger.warning("Malformed bridge gamepad message: {}", exc)
                continue
            decoded = self._decode_wireless_controller(SimpleNamespace(keys=keys, lx=lx, ly=ly, rx=rx, ry=ry))
            with self._wc_lock:
                self._wc_msg = decoded

    def close(self) -> None:
        self._closing = True
        self._joystick_thread_stop.set()
        self._joystick_thread.join(timeout=1.0)
        self._socket.close(linger=0)
