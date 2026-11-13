# Root Project

root project 用来跑 inference、tracking policy、MuJoCo simulation，以及 robot I/O。

## Setup

```bash
uv sync --extra inference-cpu
```

如果这台机器要跑 `--robot-io inline` 或 `scripts/g1/real_bridge_cpp.py`，还需要安装
G1 robot extra：

```bash
uv sync --extra inference-cpu --extra robot-g1
```

如果 `unitree_sdk2py` setup 找不到 `cyclonedds`，可以参考上游 FAQ：

[https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)

常见报错：

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

先编译并安装 CycloneDDS：

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

然后重新执行环境 setup。

## Verify Installation

### Test ankle swing

该脚本通过 SDK2 DDS 直接控制真实 G1：释放高层模式，3 秒内把所有关节移到零位，然后持续摆动脚踝直到 Ctrl+C。机器人需保持吊起。可选参数为连接机器人局域网的网卡（例如 `enp5s0`）。

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

- [Offline Motion Tracking](../tutorials/offline-motion-tracking.md)
- [Robot I/O](/reference/robot-io)
