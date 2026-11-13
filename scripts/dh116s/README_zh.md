# DH116S 灵巧手

[English](README.md) · 启动脚本总览：[../README_zh.md](../README_zh.md)

用归一化的握持指令驱动一只或两只 DH116S（LHandPro）六电机灵巧手。`control.py` 订阅 PICO publisher 发出的握持数据流
（左/右扳机 → 0–1 的握持量，经 ZMQ 传输），再通过厂商 SDK 经 USB-CANFD 发送关节目标；`keyboard_control.py` 用键盘驱动同一套映射。
它运行在插着 CANFD 适配器的那台电脑上（PC 或机器人 Orin），使用独立的 `venv/dh116s` 环境，与本体策略互不依赖。

## 状态

| 项目 | 内容 |
| --- | --- |
| 硬件状态 | 仓库中没有带日期的硬件测试记录。SDK 说明（[third_party/dh116s_sdk/README.md](../../third_party/dh116s_sdk/README.md)、`gs_usb/README.md`）来自在 Jetson 5.15 tegra 内核上的联调。 |
| `--robot` 别名 | 无；灵巧手不是 `RobotCfg`。用 `--hand-dir left \| right \| double` 选择手（默认 `double`）。 |
| 电机 | 每只手 6 个：`thumb_abd`、`thumb_flex`、`index`、`middle`、`ring`、`pinky` |
| 传输 | ZMQ SUB `tcp://127.0.0.1:5593`（`PORTS["hand_grip"]`，`HandGripMessage`：时间戳、左/右握持量 [0, 1]）→ LHandPro SDK → `gs_usb` SocketCAN 上的 USB-CANFD 适配器（`can0`/`can1`，仲裁段 1 Mbit/s，数据段 5 Mbit/s，FD） |
| 运行位置 | 插着适配器的电脑；握持 publisher（`sim2real/teleop/pico_retarget_pub.py`）可以在别的机器上 |
| 默认值 | 左手：设备索引 0、节点 1；右手：设备索引 1、节点 1；SDK 目录 `third_party/dh116s_sdk/python` |

## 文件

| 文件 | 用途 |
| --- | --- |
| `control.py` | ZMQ 握持订阅 → 手部驱动；`--dry-run` 只解码和映射，不导入 SDK、不动硬件。 |
| `keyboard_control.py` | 用 `,` / `.` 交互式张开/闭合所选的手（每只手握持量相同）。 |
| `__init__.py` | 让该目录可被测试导入（`scripts.dh116s.keyboard_control`）。 |

其他相关文件：

| 方面 | 路径 |
| --- | --- |
| 驱动与映射 | `sim2real/utils/dh116s.py`（`DH116SHandDriver`、`DryRunDH116SHandDriver`、`HandGripConsumer`、`grip_to_joint_targets`） |
| 消息与端口 | `sim2real/utils/common.py`（`HandGripMessage`、`PORTS["hand_grip"] = 5593`） |
| 握持 publisher | `sim2real/teleop/pico_retarget_pub.py`（`--hand-grip-bind`、`--hand-grip-hwm`） |
| SDK 安装 | `third_party/dh116s_sdk/`（`install.sh`、`compat/`、`gs_usb/` 内核模块说明） |
| 环境 | `venv/dh116s/pyproject.toml`（Python 3.10、`numpy<2`、`python-can~=4.6.1`、`pyzmq`） |
| 测试 | `tests/test_hand_grip.py`、`tests/test_dh116s_keyboard_control.py` |
| 文档 | [Pico 遥操作 → DH116S 灵巧手控制](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/pico-teleoperation.md) |

## 准备

1. **SDK 与环境**（在插着适配器的电脑上）：

   ```bash
   ./third_party/dh116s_sdk/install.sh          # 自动识别 aarch64 / x86_64；用 --source <目录> 可不访问 GitHub
   uv sync --project venv/dh116s
   ```

   未指定 `--source` 时，`install.sh` 通过已登录的 `gh` CLI 下载私有的厂商包；生成的 `vendor/`、`python/` 已被 git 忽略。
   如果 Jetson 内核缺少 `gs_usb`，按 `third_party/dh116s_sdk/gs_usb/README.md` 编译该模块。
2. **CAN 接口**，每次开机配置一次（驱动不会自行重新加载 `gs_usb`）：

   ```bash
   sudo modprobe gs_usb
   echo 'a8fa 8598' | sudo tee /sys/bus/usb/drivers/gs_usb/new_id
   for i in can0 can1; do
     sudo ip link set $i down
     sudo ip link set $i type can bitrate 1000000 dbitrate 5000000 fd on
     sudo ip link set $i up
   done
   ```

3. **网络。** PICO publisher 所在主机需要能访问本机 TCP 5593。
4. 如果同一个 shell 还要运行策略或遥操作进程，导出 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`
   （手部进程本身不会访问 Hugging Face）。

## 可调参数

| 设置 | 默认值 | 位置 | 作用 |
| --- | --- | --- | --- |
| `--left-device-index` / `--right-device-index` | 0 / 1 | 命令行（两个脚本） | 每只手的 SDK CAN 设备索引。Jetson 板载 MTT CAN 可能占用 `can0`：先用 `ip -details link show type can` 确认，再传入实际索引。 |
| `--left-node-id` / `--right-node-id` | 1 / 1 | 命令行 | 每只手的 CANFD 节点号。 |
| `--max-current` | 800（范围 0–1000） | 命令行 | 每次发送目标时附带的单电机电流上限。 |
| `--angular-velocity` | 200.0 | 命令行 | 每次发送目标时附带的单电机速度。 |
| `--home-wait-time` | 2.0 秒 | 命令行 | 启动回零后的等待时间。 |
| `--stale-timeout-s` | 1.0 秒 | `control.py` | 超过该时间没收到消息即报告数据流过期，并保持最后的姿态。 |
| `--status-interval-s`、`--hwm` | 2.0 秒、1 | `control.py` | 状态打印周期；ZMQ 接收高水位（conflate）。 |
| `--connect` | `tcp://127.0.0.1:5593` | `control.py` | 握持 publisher 地址。 |
| `--step` | 0.05 | `keyboard_control.py` | 每次按键的握持变化量，范围 (0, 1]。 |
| `--sdk-dir` | `third_party/dh116s_sdk/python` | 命令行 | LHandPro Python SDK 位置。 |
| `TRIGGER_CLOSE_RATIOS` | `[1.0, 0.4, 0.4, 0.4, 0.4, 0.4]` | `sim2real/utils/dh116s.py` | 握持量 1.0 时各手指闭合到自身行程的 40%；`thumb_abd` 在任何握持量下都保持在上限（1.588 rad）。 |

## 使用

**控制流程。** 启动时每只被选中的手经 CANFD 连接，读取 SDK 角度限位（失败则退回 URDF 范围），清除报警，使能电机并**回零**。
随后 `control.py` 每 5 ms 轮询一次握持数据流，每收到一条消息就发送映射后的目标；格式错误的消息会被忽略。
`double` 模式下任一只手初始化失败，进程都会断开连接并退出。Ctrl+C 断开连接，但不会下发张开指令。

1. **Dry run**（不导入 SDK、不动硬件），PICO publisher 运行时执行：

   ```bash
   uv run --project venv/dh116s --no-sync scripts/dh116s/control.py --dry-run --hand-dir double
   ```

   每 2 秒打印一次 `[DH116S dry-run] left=… right=…`。
2. **分步硬件联调**，先接一只手，并清空周围空间：

   ```bash
   uv run --project venv/dh116s --no-sync scripts/dh116s/keyboard_control.py --hand-dir left
   ```

   启动后握持量为 0.0。按键：`,` 张开一步，`.` 闭合一步，`0` 回到 0，`q`（或 Ctrl+C）退出。
3. **PICO 遥操作。** 先启动 publisher（`uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py`），再运行：

   ```bash
   uv run --project venv/dh116s --no-sync scripts/dh116s/control.py \
     --hand-dir double --connect tcp://<pico-publisher-ip>:5593
   ```

   左扳机控制左手，右扳机控制右手。
4. **关闭。** Ctrl+C。手保持在原位。
5. **测试：** `uv run --with pytest python -m pytest tests/test_hand_grip.py tests/test_dh116s_keyboard_control.py`。

## 安全提示与已知问题

- 每次非 dry-run 启动都会使能并回零所有被选中的手。先清空两只手周围的空间。
- 停止时不会张开手；已闭合的手会一直闭合，直到收到新指令。
- 数据流过期时保持最后姿态，不会自动张开。
- 厂商 `home()` 可能在内部回零闸门仍拒绝第一次运动（SDK Error 11）时就报告完成；驱动在回零后会解除该闸门。
- `new_id` 可能提示设备已注册；确认 `can0` 和 `can1` 存在后继续即可。
- 在新主机上确认 CAN 映射之前，不要假设默认的设备索引正确。

## 详细文档

- [Pico 遥操作 → DH116S 灵巧手控制](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/pico-teleoperation.md)
- [DH116S LHandPro SDK（英文）](../../third_party/dh116s_sdk/README.md) · [Jetson gs_usb 模块（英文）](../../third_party/dh116s_sdk/gs_usb/README.md)
