"""Verify G1 lowstate handset decoding without any DDS or motor connection."""
import struct
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from sim2real.config.robots import G1_CFG
from sim2real.rl_policy.controllers.unitree_joystick import UnitreeJoystickController, UnitreeJoystickState


def decoder():
    controller = UnitreeJoystickController.__new__(UnitreeJoystickController)
    controller._wc_lock = threading.Lock()
    controller._wc_msg = None
    controller._last_wc_msg = UnitreeJoystickState()
    controller._pending_extra_keys = []
    return controller


@pytest.mark.parametrize("button,bit", [
    ("R1", 0), ("L1", 1), ("R2", 4), ("L2", 5),
    ("Start", 2), ("Select", 3),
    ("A", 8), ("B", 9), ("X", 10), ("Y", 11),
])
def test_lowstate_button_bits_match_wireless_topic(button, bit):
    controller = decoder()
    payload = bytearray(40)
    struct.pack_into("<H", payload, 2, 1 << bit)
    controller._on_low_state(SimpleNamespace(wireless_remote=payload))
    assert getattr(controller.state, button)
    assert controller.state == controller._decode_wireless_controller(
        SimpleNamespace(keys=1 << bit)
    )
    controller._on_low_state(SimpleNamespace(wireless_remote=bytes(40)))
    assert not getattr(controller.state, button)


def test_lowstate_axes_use_sdk_offsets_and_ignore_trigger():
    controller = decoder()
    payload = bytearray(40)
    struct.pack_into("<5f", payload, 4, .25, -.5, .75, 99., -1.)
    controller._on_low_state(SimpleNamespace(wireless_remote=payload))
    assert controller.state.left_stick == (.25, -1.)
    assert controller.state.right_stick == (-.5, .75)
    controller._on_low_state(SimpleNamespace(wireless_remote=bytes(24)))
    assert controller.state.left_stick == (.25, -1.)


def test_held_buttons_have_one_edge_and_select_clears_pending_start():
    controller = decoder()
    controller._wc_msg = UnitreeJoystickState(A=True)
    assert controller.get_control_mode() == "init"
    assert controller.get_control_mode() is None  # holding A is one press
    controller._wc_msg = UnitreeJoystickState(Start=True)
    assert controller.get_control_mode() is None
    assert controller.get_extra_keys() == ["Start"]
    assert controller.get_control_mode() is None
    assert controller.get_extra_keys() == []
    controller._pending_extra_keys.append("Start")
    controller._wc_msg = UnitreeJoystickState(A=True, Start=True, Select=True)
    assert controller.get_control_mode() == "zero"
    assert controller.get_extra_keys() == []
    assert controller.get_control_mode() is None
    controller._wc_msg = UnitreeJoystickState(Start=True)  # still held after Select release
    assert controller.get_control_mode() is None
    assert controller.get_extra_keys() == []
    controller._wc_msg = UnitreeJoystickState()
    controller.get_control_mode()
    controller._wc_msg = UnitreeJoystickState(Start=True)
    controller.get_control_mode()
    assert controller.get_extra_keys() == ["Start"]


def test_g1_subscribes_lowstate_callback_and_closes(monkeypatch):
    initialized = []
    readers = []
    class Subscriber:
        def __init__(self, topic, message_type):
            self.topic, self.message_type = topic, message_type
            self.closed = False
            readers.append(self)
        def Init(self, handler, queueLen):
            self.handler, self.queue_length = handler, queueLen
        def Close(self):
            self.closed = True
    channel = ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = lambda *args: initialized.append(args)
    channel.ChannelSubscriber = Subscriber
    hg = ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    hg.LowState_ = object()
    monkeypatch.setitem(sys.modules, channel.__name__, channel)
    monkeypatch.setitem(sys.modules, hg.__name__, hg)
    from dataclasses import replace
    controller = UnitreeJoystickController(replace(G1_CFG, interface="enp5s0"), {})
    assert initialized == [(0, "enp5s0")]
    assert readers[0].topic == "rt/lowstate"
    assert readers[0].message_type is hg.LowState_
    assert readers[0].queue_length == 0
    assert controller._joystick_thread is None
    readers[0].handler(SimpleNamespace(wireless_remote=bytes(40)))
    assert controller.state is not None
    controller.close()
    assert readers[0].closed
