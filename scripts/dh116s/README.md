# DH116S dexterous hand

[中文](README_zh.md) · Generic launcher overview: [../README.md](../README.md)

Drives one or two DH116S (LHandPro) six-motor hands from normalized grip commands. `control.py` subscribes to the
PICO publisher's hand-grip stream (left/right trigger → grip 0–1 over ZMQ) and sends joint targets through the vendor
SDK over USB-CANFD; `keyboard_control.py` drives the same mapping from the keyboard. It runs on whichever computer has
the CANFD adapters plugged in (PC or robot Orin), in its own `venv/dh116s` environment, independent of the body policy.

## Status

| Item | Value |
| --- | --- |
| Hardware status | No dated hardware result is recorded in the repo. The SDK notes ([third_party/dh116s_sdk/README.md](../../third_party/dh116s_sdk/README.md), `gs_usb/README.md`) come from bring-up on a Jetson 5.15 tegra kernel. |
| `--robot` alias | None; the hand is not a `RobotCfg`. Select hands with `--hand-dir left \| right \| double` (default `double`). |
| Motors | 6 per hand: `thumb_abd`, `thumb_flex`, `index`, `middle`, `ring`, `pinky` |
| Transport | ZMQ SUB `tcp://127.0.0.1:5593` (`PORTS["hand_grip"]`, `HandGripMessage`: timestamp, left/right grip in [0, 1]) → LHandPro SDK → USB-CANFD adapters on `gs_usb` SocketCAN (`can0`/`can1`, 1 Mbit/s, 5 Mbit/s data, FD) |
| Runs on | The computer with the adapters; the grip publisher (`sim2real/teleop/pico_retarget_pub.py`) may run elsewhere |
| Defaults | Left: device index 0, node 1; right: device index 1, node 1; SDK dir `third_party/dh116s_sdk/python` |

## Files

| File | Purpose |
| --- | --- |
| `control.py` | ZMQ grip subscriber → hand driver; `--dry-run` decodes and maps without importing the SDK or moving hardware. |
| `keyboard_control.py` | Interactive open/close with `,` / `.` for both selected hands (same grip on each). |
| `__init__.py` | Makes the folder importable for tests (`scripts.dh116s.keyboard_control`). |

Related files elsewhere:

| Area | Path |
| --- | --- |
| Driver and mapping | `sim2real/utils/dh116s.py` (`DH116SHandDriver`, `DryRunDH116SHandDriver`, `HandGripConsumer`, `grip_to_joint_targets`) |
| Message and port | `sim2real/utils/common.py` (`HandGripMessage`, `PORTS["hand_grip"] = 5593`) |
| Grip publisher | `sim2real/teleop/pico_retarget_pub.py` (`--hand-grip-bind`, `--hand-grip-hwm`) |
| SDK install | `third_party/dh116s_sdk/` (`install.sh`, `compat/`, `gs_usb/` kernel module notes) |
| Environment | `venv/dh116s/pyproject.toml` (Python 3.10, `numpy<2`, `python-can~=4.6.1`, `pyzmq`) |
| Tests | `tests/test_hand_grip.py`, `tests/test_dh116s_keyboard_control.py` |
| Docs | [Pico Teleoperation → DH116S hand control](../../docs/tutorials/pico-teleoperation.md#dh116s-hand-control) |

## Setup

1. **SDK and environment** (on the computer with the adapters):

   ```bash
   ./third_party/dh116s_sdk/install.sh          # detects aarch64 / x86_64; --source <dir> avoids GitHub
   uv sync --project venv/dh116s
   ```

   `install.sh` downloads the private vendor package with the authenticated `gh` CLI unless `--source` is given; its
   output (`vendor/`, `python/`) is gitignored. On a Jetson whose kernel lacks `gs_usb`, build the module per
   `third_party/dh116s_sdk/gs_usb/README.md`.
2. **CAN interfaces**, once per boot (the driver does not reload `gs_usb` itself):

   ```bash
   sudo modprobe gs_usb
   echo 'a8fa 8598' | sudo tee /sys/bus/usb/drivers/gs_usb/new_id
   for i in can0 can1; do
     sudo ip link set $i down
     sudo ip link set $i type can bitrate 1000000 dbitrate 5000000 fd on
     sudo ip link set $i up
   done
   ```

3. **Network.** The grip stream needs TCP 5593 reachable from the PICO publisher host.
4. If the same shell also runs policy or teleop processes, export `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`
   (the hand process itself makes no Hugging Face requests).

## Adjustments

| Setting | Default | Where | Effect |
| --- | --- | --- | --- |
| `--left-device-index` / `--right-device-index` | 0 / 1 | CLI (both scripts) | SDK CAN device index per hand. On Jetson the onboard MTT CAN may own `can0`: check `ip -details link show type can` and pass the real indices. |
| `--left-node-id` / `--right-node-id` | 1 / 1 | CLI | CANFD node of each hand. |
| `--max-current` | 800 (range 0–1000) | CLI | Per-motor current limit sent with every target. |
| `--angular-velocity` | 200.0 | CLI | Per-motor speed sent with every target. |
| `--home-wait-time` | 2.0 s | CLI | Wait after homing at startup. |
| `--stale-timeout-s` | 1.0 s | `control.py` | After this long without a message the stream is reported stale; the last pose is held. |
| `--status-interval-s`, `--hwm` | 2.0 s, 1 | `control.py` | Status print period; ZMQ receive high-water mark (conflated). |
| `--connect` | `tcp://127.0.0.1:5593` | `control.py` | Grip publisher endpoint. |
| `--step` | 0.05 | `keyboard_control.py` | Grip change per key press, in (0, 1]. |
| `--sdk-dir` | `third_party/dh116s_sdk/python` | CLI | LHandPro Python SDK location. |
| `TRIGGER_CLOSE_RATIOS` | `[1.0, 0.4, 0.4, 0.4, 0.4, 0.4]` | `sim2real/utils/dh116s.py` | Grip 1.0 closes each finger to 40% of its range; `thumb_abd` is held at its upper limit (1.588 rad) at every grip. |

## Usage

**Control flow.** At startup each requested hand connects over CANFD, reads its SDK angle limits (falls back to URDF
ranges), clears any alarm, enables the motors and **homes**. `control.py` then polls the grip stream every 5 ms and
sends the mapped targets on every message; a malformed message is ignored. In `double` mode, if either hand fails to
initialize the process disconnects and exits. Ctrl+C disconnects without commanding the hands open.

1. **Dry run** (no SDK import, no hardware), with the PICO publisher running:

   ```bash
   uv run --project venv/dh116s --no-sync scripts/dh116s/control.py --dry-run --hand-dir double
   ```

   It prints `[DH116S dry-run] left=… right=…` every 2 s.
2. **Staged hardware bring-up**, one hand first, workspace clear:

   ```bash
   uv run --project venv/dh116s --no-sync scripts/dh116s/keyboard_control.py --hand-dir left
   ```

   Starts at grip 0.0. Keys: `,` open one step, `.` close one step, `0` back to 0, `q` (or Ctrl+C) quit.
3. **PICO teleop.** Start the publisher (`uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py`), then:

   ```bash
   uv run --project venv/dh116s --no-sync scripts/dh116s/control.py \
     --hand-dir double --connect tcp://<pico-publisher-ip>:5593
   ```

   Left trigger drives the left hand, right trigger the right hand.
4. **Shutdown.** Ctrl+C. The hands stay where they are.
5. **Tests:** `uv run --with pytest python -m pytest tests/test_hand_grip.py tests/test_dh116s_keyboard_control.py`.

## Safety notes & known issues

- Every non-dry-run start enables and homes all requested hands. Clear the space around both hands first.
- Stopping never opens the hands; a closed hand stays closed until commanded.
- A stale stream holds the last pose rather than opening.
- The vendor `home()` can report completion while its home gate still rejects the first move (SDK Error 11); the
  driver releases the gate after homing.
- `new_id` may say the device is already registered; continue once `can0` and `can1` exist.
- Do not assume the default device indices on a new host until the CAN mapping is verified.

## Detailed docs

- [Pico Teleoperation → DH116S hand control](../../docs/tutorials/pico-teleoperation.md#dh116s-hand-control)
- [DH116S LHandPro SDK](../../third_party/dh116s_sdk/README.md) · [Jetson gs_usb module](../../third_party/dh116s_sdk/gs_usb/README.md)
