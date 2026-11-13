---
title: Root Project
sidebar_position: 2
---

Use the root project for inference, tracking policy, MuJoCo simulation, and robot I/O.

## Setup

```bash
uv sync --extra inference-cpu
```

If this machine will run `--robot-io inline` or `scripts/g1/real_bridge_cpp.py`,
install the G1 robot extra as well:

```bash
uv sync --extra inference-cpu --extra robot-g1
```

If `unitree_sdk2py` setup cannot locate `cyclonedds`, refer to the upstream
FAQ:

[https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)

Typical error:

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

Compile and install CycloneDDS first:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

Then rerun your environment setup.

## Verify Installation

### Test ankle swing

This drives the real G1 over SDK2 DDS: it releases high-level mode, moves every joint to zero over 3 seconds, then swings the ankles until Ctrl+C. Keep the robot suspended. The optional argument is the robot-LAN network interface (for example `enp5s0`).

```bash
uv run scripts/ankle_swing.py <network-interface>
```

### Test inference time

```bash
uv run scripts/test_policy_inference.py \
  --policy_config checkpoints/mimic-lite/roa/policy.yaml \
  --inference_backend onnx-cpu
```

## Next Steps

- [Offline Motion Tracking](/tutorials/offline-motion-tracking)
- [Robot I/O](/reference/robot-io)
