# Proteus G1 on the robot's Orin

The same Bridge (200 Hz, BridgeSafety) and Tracking (50 Hz, CPU ONNX) as [the external PC deployment](proteus-g1-pc.md), run on the G1's own Orin NX (`ssh g1`, 192.168.123.164, robot LAN on `eth0`). Control code, safety behaviour and the remote flow are unchanged: A → A → wait for Hold ready → lower the gantry → Start, Select to stop. The PC only deploys, starts and collects logs, so a PC sleep, a PC network-card fault or a pulled cable no longer cuts the control loop.

The Orin runs JetPack 5.1.1 (Ubuntu 20.04, glibc 2.31, system Python 3.8) and has no internet. Everything lives in one directory, `/home/unitree/proteus_g1_onboard`, with nothing installed system-wide and no root needed:

| Path | Contents |
| --- | --- |
| `python/` | standalone CPython 3.10.22 (aarch64) |
| `site-packages/` | offline packages resolved from `uv.lock` for aarch64 / manylinux_2_31 (torch CPU, mujoco, onnxruntime, …) |
| `cyclonedds/`, `cyclonedds-py/` | CycloneDDS 0.10.2, the `unitree_sdk2py` pin, built on the Orin (no aarch64 wheel exists) |
| `repo/` | `sim2real`, `scripts/g1`, the wsq62 and ht500 policies, the motion clips; runs in `repo/outputs/proteus-pc/<time>/` |
| `hf/hub/` | the G1 MJCF cache (`g1_xmls`) and `h2_model` |

## Deploy

```bash
scripts/g1/onboard.sh deploy
```

`scripts/g1/deploy_onboard.sh` resolves the packages and the Python build once per `uv.lock` (uv binary: `UV`, default `outputs/g1-onboard-env/dl/uv/...`), downloads and checks the CycloneDDS sources, rsyncs a whitelist at 20 MB/s (`G1_RSYNC_BWLIMIT`) so a running PC session keeps the robot LAN, and builds CycloneDDS on the Orin when missing (`scripts/g1/onboard_build_cyclonedds.sh`). It never starts a controller. `G1_POLICY_DIRS` selects the policies to push.

## Run

```bash
PROTEUS_READ_ONLY=1 PROTEUS_READ_ONLY_S=60 scripts/g1/onboard.sh start   # real feedback and inference, no motor command
scripts/g1/onboard.sh start                                              # protected hardware session (gantry + e-stop)
scripts/g1/onboard.sh tail      # follow bridge.log / tracker.log; Ctrl+C leaves the robot running
scripts/g1/onboard.sh status
scripts/g1/onboard.sh stop      # SIGTERM: orderly shutdown with the Bridge's damping sequence
scripts/g1/onboard.sh logs      # copy the Orin's runs to outputs/proteus-onboard/
```

`start` launches `scripts/g1/start_proteus_onboard.sh` detached from SSH and forwards `PROTEUS_POLICY_DIR`, `PROTEUS_CLIP`, `PROTEUS_PLAYBACK_LIMIT_S`, `PROTEUS_IO_TIMEOUT_S`, `PROTEUS_HOLD_THROUGH_GAPS`, `PROTEUS_MOTION_PATH`, `SIM2REAL_ORT_NUM_THREADS` and the read-only settings. It refuses to start a writing session while a PC Proteus controller runs on this PC. Never run the PC and onboard deployments against the robot at the same time. The Orin's clock runs about 12 minutes behind the PC's, so run directory names use Orin time.

`start_proteus_onboard.sh` sets the environment and pins the processes, then runs `start_proteus_pc.sh`: Bridge on core 1, Tracking on cores 2–5 with four ONNX threads, the read-only LowState gap monitor on core 6 at nice 10. Tracking uses `--controller bridge_joystick`: the Bridge re-publishes the handset on `PORTS["gamepad"]`, so the Tracker has no DDS subscription of its own. With the PC's default `joystick` controller, the Tracker's own CycloneDDS receive thread decoded 1 kHz LowState in Python and took about 70% of a core, which stretched a step to 25–33 ms (31–41 Hz). The PC launcher still defaults to `joystick` (`PROTEUS_CONTROLLER` overrides it).

## Measured timing

60-second read-only runs on 2026-11-05 (wsq62, `walk_straight_short`):

| | per step | policy | observation | rate | steps > 25 ms |
| --- | --- | --- | --- | --- | --- |
| Orin, Tracker DDS joystick | 25–33 ms | 13–17 ms | 5–8 ms | 31–41 Hz | about half |
| Orin, `bridge_joystick` | 9.7 ms | 6.4 ms | 2.0 ms | 50.00 Hz | 13 of 2455 (worst 30 ms) |
| external PC | 5.2 ms | 3.6 ms | 1.05 ms | 50.00 Hz | 19 of 46595 (worst 107 ms) |

Standalone, the WAM policy takes 13.2 / 9.1 / 6.9 ms on the Orin with 1 / 2 / 4 ONNX threads (wsq62 and ht500 alike). The Orin timing was measured after `sudo jetson_clocks` (CPU 1984 MHz, GPU 918 MHz). Locking clocks alone barely changed the DDS-joystick numbers, and the setting reverts at reboot. Apart from four steps at startup, the remaining slow steps come about every 5 seconds as an 18–19 ms policy call, consistent with another periodic task sharing cores 2–5; the robot's depth-stream process moves across them. The onboard path has not yet been tested on hardware.
