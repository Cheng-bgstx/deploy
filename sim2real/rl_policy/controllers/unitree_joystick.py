from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import struct
import threading
import time
from types import SimpleNamespace

from loguru import logger

from sim2real.config.robots.base import RobotCfg, get_unitree_dds_family
from sim2real.rl_policy.control_mode import resolve_unitree_joystick_control_mode
from sim2real.rl_policy.controllers.base import ControllerBase
from sim2real.rl_policy.inference import Timer


@dataclass(frozen=True)
class UnitreeJoystickState:
    A: bool = False
    B: bool = False
    X: bool = False
    Y: bool = False
    L1: bool = False
    L2: bool = False
    R1: bool = False
    R2: bool = False
    Start: bool = False
    Select: bool = False
    left_stick: tuple[float, float] = (0.0, 0.0)
    right_stick: tuple[float, float] = (0.0, 0.0)


class UnitreeJoystickController(ControllerBase):
    name = "unitree_joystick"

    def __init__(self, robot_cfg: RobotCfg, perf_dict: dict[str, float]) -> None:
        self.robot_cfg = robot_cfg
        self.perf_dict = perf_dict
        self._wc_lock = threading.Lock()
        self._joystick_thread_stop = threading.Event()
        self._pending_extra_keys: list[str] = []
        self._wc_msg: UnitreeJoystickState | None = None
        self._last_wc_msg = UnitreeJoystickState()

        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

        if self.robot_cfg.interface:
            ChannelFactoryInitialize(self.robot_cfg.domain_id, self.robot_cfg.interface)
        else:
            ChannelFactoryInitialize(self.robot_cfg.domain_id)

        self._joystick_thread = None
        if get_unitree_dds_family(self.robot_cfg.name) == "hg":
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

            # G1 carries the handset's 40-byte payload inside lowstate, even
            # when no publisher exists for rt/wirelesscontroller. Decode in
            # the callback so the 1 kHz state stream cannot queue stale keys.
            self.wireless_controller_sub = ChannelSubscriber("rt/lowstate", LowState_)
            self.wireless_controller_sub.Init(self._on_low_state, 0)
        else:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

            self.wireless_controller_sub = ChannelSubscriber(
                "rt/wirelesscontroller", WirelessController_,
            )
            self.wireless_controller_sub.Init(None, 0)
            self._joystick_thread = threading.Thread(
                target=self._poll_wireless_controller,
                daemon=True,
            )
            self._joystick_thread.start()

    @property
    def state(self) -> UnitreeJoystickState | None:
        with self._wc_lock:
            if self._wc_msg is None:
                return None
            return deepcopy(self._wc_msg)

    def _decode_wireless_controller(self, msg) -> UnitreeJoystickState:
        key_bits = {
            "R1": 0,
            "L1": 1,
            "Start": 2,
            "Select": 3,
            "R2": 4,
            "L2": 5,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
        }
        keys = getattr(msg, "keys", 0)
        return UnitreeJoystickState(
            A=bool(keys & (1 << key_bits["A"])),
            B=bool(keys & (1 << key_bits["B"])),
            X=bool(keys & (1 << key_bits["X"])),
            Y=bool(keys & (1 << key_bits["Y"])),
            L1=bool(keys & (1 << key_bits["L1"])),
            L2=bool(keys & (1 << key_bits["L2"])),
            R1=bool(keys & (1 << key_bits["R1"])),
            R2=bool(keys & (1 << key_bits["R2"])),
            Start=bool(keys & (1 << key_bits["Start"])),
            Select=bool(keys & (1 << key_bits["Select"])),
            left_stick=(getattr(msg, "lx", 0.0), getattr(msg, "ly", 0.0)),
            right_stick=(getattr(msg, "rx", 0.0), getattr(msg, "ry", 0.0)),
        )

    def _on_low_state(self, msg) -> None:
        payload = bytes(msg.wireless_remote)
        if len(payload) != 40:
            return
        # Unitree SDK layout: keys@2, lx@4, rx@8, ry@12, ly@20.
        keys = struct.unpack_from("<H", payload, 2)[0]
        lx, rx, ry = struct.unpack_from("<3f", payload, 4)
        ly = struct.unpack_from("<f", payload, 20)[0]
        decoded = self._decode_wireless_controller(
            SimpleNamespace(keys=keys, lx=lx, ly=ly, rx=rx, ry=ry)
        )
        with self._wc_lock:
            self._wc_msg = decoded

    def _poll_wireless_controller(self) -> None:
        poll_interval = 0.2
        while not self._joystick_thread_stop.is_set():
            try:
                with Timer(self.perf_dict, "read_wireless_controller"):
                    raw_msg = self.wireless_controller_sub.Read()
                if raw_msg is not None:
                    with Timer(self.perf_dict, "decode_wireless_controller"):
                        decoded = self._decode_wireless_controller(raw_msg)
                    with self._wc_lock:
                        self._wc_msg = decoded
            except Exception as exc:
                logger.debug(f"Joystick poll error: {exc}")
            finally:
                time.sleep(poll_interval)

    def get_control_mode(self):
        wc_local = self.state
        if wc_local is None:
            return None

        # Stop wins over every simultaneous press and clears queued starts.
        # Holding Select also prevents a new start until it is released.
        if wc_local.Select:
            self._pending_extra_keys.clear()
            mode = (
                resolve_unitree_joystick_control_mode("Select")
                if not self._last_wc_msg.Select else None
            )
            self._last_wc_msg = wc_local
            return mode

        mode = None
        if wc_local.A and not self._last_wc_msg.A:
            mode = resolve_unitree_joystick_control_mode("A")
        if wc_local.Start and not self._last_wc_msg.Start:
            self._pending_extra_keys.append("Start")
        if wc_local.X and not self._last_wc_msg.X:
            self._pending_extra_keys.append("X")
        if wc_local.Y and not self._last_wc_msg.Y:
            self._pending_extra_keys.append("Y")
        if wc_local.L1 and not self._last_wc_msg.L1:
            self._pending_extra_keys.append("L1")
        if wc_local.L2 and not self._last_wc_msg.L2:
            self._pending_extra_keys.append("L2")
        self._last_wc_msg = wc_local
        return mode

    def get_extra_keys(self) -> list[str]:
        extra_keys = list(self._pending_extra_keys)
        self._pending_extra_keys.clear()
        return extra_keys

    def close(self) -> None:
        try:
            self._joystick_thread_stop.set()
            if self._joystick_thread is not None and self._joystick_thread.is_alive():
                self._joystick_thread.join(timeout=1.0)
            self.wireless_controller_sub.Close()
        except Exception as exc:
            logger.debug(f"Failed to stop joystick listener cleanly: {exc}")
