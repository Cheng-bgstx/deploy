"""Send one operator key press to the ET1 bridge on this board: python press_key.py {a,s,space}.

For scripted or remote runs. It is merged with any keyboard console per sender, so both can be open.
"""
from __future__ import annotations

import sys
import time

from sim2real.utils.operator_keys import KEY_A, KEY_SELECT, KEY_START, OperatorKeyPublisher

KEYS = {"a": KEY_A, "s": KEY_START, "space": KEY_SELECT}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in KEYS:
        print(__doc__, file=sys.stderr)
        return 2
    pub = OperatorKeyPublisher()
    try:
        pub.press(KEYS[sys.argv[1]])
    finally:
        pub.close()
    print(time.strftime("%H:%M:%S"), "pressed", sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
