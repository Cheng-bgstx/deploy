"""Keyboard operator for the ET1 bridge, for runs without a handset.

Run it in an SSH terminal on the board (ssh -t). Keys become handset presses that the bridge ORs into
the HDEP gamepad, so BridgeSafety and the Tracker see the same A -> A -> Start / Select flow:

    a      A       first: stand (shaping); second, after shaping: reference transition and Hold
    s      Start   play the motion from Hold
    space  Select  stop: latch damping (also x)
    Ctrl-C quit (no key is sent)

Each press is held for HOLD_S so both 200 Hz bridge and 50 Hz Tracker see one rising edge. Messages
repeat at 20 Hz; the bridge ignores keys older than its freshness window, so a dropped SSH session
releases every key. It never stops the robot by itself: keep the gantry and the hardware stop ready.
"""
from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty

from sim2real.utils.operator_keys import KEY_A, KEY_SELECT, KEY_START, OperatorKeyPublisher

KEYS = {"a": ("A", KEY_A), "s": ("Start", KEY_START), " ": ("Select", KEY_SELECT), "x": ("Select", KEY_SELECT)}
HOLD_S = 0.15
PERIOD_S = 0.05


def main() -> int:
    if not sys.stdin.isatty():
        print("Run in an interactive terminal (ssh -t).", file=sys.stderr)
        return 2
    pub = OperatorKeyPublisher()
    print(__doc__.split("\n\n")[2], flush=True)  # the key table
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    held, held_until = 0, 0.0
    try:
        tty.setcbreak(fd)
        while True:
            ready, _, _ = select.select([fd], [], [], PERIOD_S)
            now = time.monotonic()
            if ready:
                char = os.read(fd, 1).decode(errors="ignore").lower()
                if char in KEYS:
                    name, bit = KEYS[char]
                    held, held_until = bit, now + HOLD_S
                    print(f"\r{time.strftime('%H:%M:%S')}  {name:<6s}", flush=True)
            if now >= held_until:
                held = 0
            pub.send(held)
    except KeyboardInterrupt:
        return 0
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        pub.close()


if __name__ == "__main__":
    raise SystemExit(main())
