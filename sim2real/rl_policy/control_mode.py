from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ControlMode = Literal["init", "zero", "policy"]


@dataclass(frozen=True)
class PicoButtonState:
    A: bool = False
    B: bool = False


KeyboardLayout = Literal["default", "et1"]

# Key that plays or pauses the motion from Hold (or Go with the Planner), per layout.
KEYBOARD_PLAY_KEYS: dict[str, tuple[str, ...]] = {"default": ("space", " "), "et1": ("s",)}


def resolve_keyboard_control_mode(keycode: str, layout: KeyboardLayout = "default") -> ControlMode | None:
    if layout == "et1":
        # ET1 operator keys: a shapes, a again (after shaping) enters Hold, s plays, space or x stops.
        return {"a": "init", "space": "zero", " ": "zero", "x": "zero"}.get(keycode.lower())
    return {
        "i": "init",
        "o": "zero",
        "]": "policy",
    }.get(keycode)


def resolve_unitree_joystick_control_mode(cur_key: str) -> ControlMode | None:
    return {
        "A": "init",
        "Select": "zero",
        "R1": "policy",
        "R2": "zero",
    }.get(cur_key)


def resolve_pico_control_mode(
    current: PicoButtonState,
    previous: PicoButtonState,
) -> ControlMode | None:
    current_combo = current.A and current.B
    previous_combo = previous.A and previous.B
    if current_combo and not previous_combo:
        return "policy"
    if current.A and not previous.A and not current.B:
        return "init"
    if current.B and not previous.B and not current.A:
        return "zero"
    return None
