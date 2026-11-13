# Unitree G1 启动脚本

[English](README.md) · 通用流程（Shaping → Hold → Go → Final Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

本目录存放在 Unitree G1（29 自由度）上运行 Proteus 跟踪策略的启动脚本和工具。实际工作由两个进程完成：
**Bridge**（`real_bridge.py`，200 Hz，经 DDS 使用 Python `unitree_sdk2py`，由 `BridgeSafety` 仲裁）和
**Tracker**（`sim2real/rl_policy/tracking.py`，50 Hz，CPU ONNX），两者通过本地 ZMQ low_state / low_cmd 协议通信。
它们既可以跑在有线连接机器人的外部 PC 上（`start_proteus_pc.sh`），也可以跑在 G1 自带的 Orin NX 上
（`onboard.sh` → `start_proteus_onboard.sh`）；遥控器操作相同：A → A → Start，Select 停止。

## 状态

| 项目 | 内容 |
| --- | --- |
| 硬件状态 | 外部 PC：默认的 wsq62 actor 已在真机测试（[README](../../README_zh.md)、[PC 文档](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-g1-pc.md)；记录了 2026-11-04 的硬件事故）。板载 Orin：2026-11-05 做过只读计时测试；[板载文档](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-g1-onboard.md)仍写明板载路径尚未在真机上测试。 |
| `--robot` 别名 | `g1`（`G1_CFG.name`，见 `sim2real/config/robots/__init__.py`） |
| 关节数 | 29（每条腿 6 个、腰 3 个、每条手臂 7 个） |
| 机器人 I/O | `--robot-io zmq` + `scripts/g1/real_bridge.py`（`unitree_sdk2py`、CycloneDDS 0.10.2、`hg` LowState/LowCmd）。另有 inline 模式 `--robot-io inline`（`unitree_interface`，仅 aarch64），但 Proteus 启动脚本不使用。 |
| Tracker 运行位置 | PC（`start_proteus_pc.sh`，网卡 `enp5s0`）或板载 Orin（`start_proteus_onboard.sh`，网卡 `eth0`） |
| 默认策略 | `checkpoints/proteus/proteus_wamseq_62/unitree_g1/`（`policy.onnx`、`policy.yaml`、生成的 `policy-pc.yaml`），可用 `PROTEUS_POLICY_DIR` 覆盖。备选：`checkpoints/proteus/wsq61/unitree_g1`（硬件结果不覆盖它）、`checkpoints/proteus/proteus_60/unitree_g1`（旧 actor）。 |
| 默认动作 | `PROTEUS_CLIP=walk_straight_short` → `checkpoints/proteus/proteus_60/unitree_g1/motions/walk_straight_short` |

## 文件

| 文件 | 用途 |
| --- | --- |
| `start_proteus_pc.sh` | PC 启动脚本：预检后启动 Bridge、LowState 间隙监视器和 Tracker 并绑核；日志在 `outputs/proteus-pc/<时间>/`。 |
| `start_proteus_onboard.sh` | Orin 启动脚本：设置离线 Python/CycloneDDS/HF 环境和绑核，然后运行 `start_proteus_pc.sh`。 |
| `onboard.sh` | 在 PC 上控制板载运行：`deploy`、`start`、`status`、`tail`、`stop`、`logs`。 |
| `deploy_onboard.sh` | 把 CPython 3.10、离线 aarch64 包、代码、策略、动作和 HF 缓存推送到 Orin；从不启动控制器。 |
| `onboard_build_cyclonedds.sh` | 由 `deploy_onboard.sh` 在 Orin 上调用：离线编译 CycloneDDS 0.10.2 及其 Python 绑定。 |
| `prepare_proteus_deployment.py` | 在导出的 `policy.yaml` 旁生成部署覆盖文件 `policy-pc.yaml`（ET1 通过 `--robot` 也用它）。 |
| `prepare_proteus_motion.py` | 从固定版本的 `elijahgalahad/any4hdmi-g1-lafan` 数据集截取测试片段（`--clip`）；不会覆盖已有文件。 |
| `validate_proteus_deployment.py` | 无界面 MuJoCo 演练：真实 Tracker + `BridgeSafety`，不创建 DDS。 |
| `real_bridge.py` | SDK2 ⇄ ZMQ Bridge：延迟接管、停止仲裁、`--hold-through-gaps`、`--read-only`。 |
| `real_bridge_cpp.py` | 同一 ZMQ 协议，底层改用 C++ `unitree_interface` 绑定（aarch64）。 |
| `lowstate_gap_monitor.py` | 只读 LowState 接收器，记录超过 30 ms 的接收间隙；从不发布任何内容。 |
| `joint_response_test.py` | 腿部关节正弦/阶跃响应测试，可在悬吊的真机（`--target real`）或 MuJoCo（`--target sim`）中运行。 |
| `scalecar/start_bridge.sh`、`scalecar/start_tracker.sh` | Scalecar 上车任务的 Bridge 与 Tracker（Planner、手柄）启动脚本。 |
| `depth_stream/` | Orin 上的 RealSense 深度采集程序（`build.sh` → `build/scalecar-depth`），从 Scalecar_v2 原样复制（`SOURCES.json`）。 |

其他相关文件：

| 方面 | 路径 |
| --- | --- |
| 机器人配置 | `sim2real/config/robots/g1.py`（关节限位、`safe_joint_kp/kd`、电枢惯量、MJCF `hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml`） |
| 机器人 I/O | `sim2real/rl_policy/robot_io/zmq.py`、`bridge_safety.py`、`g1.py`（inline）、`g1_debug_mode.py`（MotionSwitcher 释放） |
| 控制器 | `sim2real/rl_policy/controllers/unitree_joystick.py`（`joystick`）、`bridge_joystick.py`（板载）、`keyboard.py`（仿真） |
| 遥操作 | `sim2real/teleop/pico_retarget_pub.py`（`venv/pico` 中 GMR 的 `xrobot_to_g1`），[PICO 到 G1 映射](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/pico-to-g1.md) |
| Scalecar 工具 | `scripts/scalecar/`（场景构建、动作转换）、`docs/adr/` |
| 测试 | `tests/test_g1_bridge_safety.py`、`tests/test_planner_real_robot.py` |
| 配置技能 | `.agents/skills/configure-g1-sim2real/`（`scripts/inspect_host.sh`、`scripts/verify_install.py`） |

## 准备

1. **根项目环境（策略、Bridge、仿真）。** 在 PC 上：

   ```bash
   uv sync --extra inference-cpu --extra robot-g1
   ```

   启动脚本使用 `.venv/bin/python`（可用 `SIM2REAL_PYTHON` 覆盖）。若 `unitree_sdk2py` 编译找不到 CycloneDDS，参见
   [根项目](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/root-project.md)。
   遥操作使用 `venv/pico`（`uv sync --project venv/pico`）。
2. **网络。** PC 网口配置 `192.168.123.x` 接入机器人局域网（用 `ip -br link` 查网卡名；启动脚本默认 `enp5s0`，
   用 `G1_INTERFACE` 覆盖）。Orin 地址为 `192.168.123.164`（用户 `unitree`）；`deploy_onboard.sh` 和 `onboard.sh`
   需要 `~/.ssh/config` 中有 SSH 别名 `g1`（用 `G1_HOST` 覆盖）。参见
   [网络配置](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/network-configuration.md)。
3. **运行时产物（不在 git 中）。** `checkpoints/proteus/...` 和 `third_party/` 来自共享 Google Drive 目录
   （`gdrive:projects/simple-tracking/sim2real`，见[下载产物](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/artifacts.md)）。
   联网时先缓存一次 G1 MJCF 和动作数据集；`prepare_proteus_motion.py` 会下载固定版本的源动作。
4. **Hugging Face 离线。** 每个 shell 中都设置，并写入机器人上的 `~/.bashrc`：

   ```bash
   export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
   ```

   启动脚本自身也会导出这两个变量。`start_proteus_onboard.sh` 会把 `HF_HOME` 指向推送过去的缓存。
5. **仅板载。** `deploy_onboard.sh` 需要一个 x86 `uv` 可执行文件（`UV`，默认
   `outputs/g1-onboard-env/dl/uv/uv-x86_64-unknown-linux-gnu/uv`）。所有内容都放在
   `/home/unitree/proteus_g1_onboard`（`G1_TARGET`），Orin 上不需要 root。

## 可调参数

**策略覆盖文件**：`prepare_proteus_deployment.py` 写出 `policy-pc.yaml`，不改动 `policy.yaml`。启动脚本只在文件缺失时生成，
因此修改脚本后要删除或重新生成它。

| `policy-pc.yaml` 中的键 | G1 取值 | 作用 |
| --- | --- | --- |
| 腰部 roll、pitch 的 `joint_kp` / `joint_kd` | 100 / 2.5（yaw 保持 `policy.yaml` 中的 100 / 2） | 训练时的腰部增益在真机上引起高频抖动。 |
| `soft_joint_pos_limits` | 关节限位 ± `PROTEUS_TARGET_MARGIN_RAD` = 1.0 rad | Proteus 训练时不裁剪目标，踝关节目标会超出范围约 0.8 rad。其他机器人使用 5% 内缩。 |
| `max_target_velocity_rad_s` | 15.0 | 目标变化速率上限。 |
| `shaping_target` | `default_joint_pos` | Shaping 到双臂下垂的默认站姿（10 秒，写死在 `base_policy.py`）。 |
| `reference_entry_s` | 2.0 | 第二次 A 插值到片段首帧，然后进入 Hold。 |
| 锁定关节（`--locked-joints`） | G1 无 | 在默认姿态处设置零宽软限位（ET1 锁定头部）。 |

`proteus.reference_forcing` 必须保持 `true`（启动脚本会断言）。G1 没有设置 `robot_io_options` 或 IMU 外参。

**Bridge / 安全**（`real_bridge.py`、`sim2real/rl_policy/robot_io/bridge_safety.py`）：

| 设置 | 默认值 | 说明 |
| --- | --- | --- |
| `--timeout-s` / `PROTEUS_IO_TIMEOUT_S` | 单独运行 0.1，启动脚本中 0.25 | 同一时间窗口以 `SIM2REAL_IO_TIMEOUT_S` 传给 Tracker；范围 (0, 1]。 |
| `--hold-through-gaps` / `PROTEUS_HOLD_THROUGH_GAPS` | 单独运行关闭，启动脚本中为 `1` | 间隙期间保持上一组 PD 目标；超过 1 秒（`MAX_HOLD_GAP_NS`）或 tick 回退仍会锁存阻尼。设为 `0` 恢复锁存阻尼。 |
| `--target-margin-rad` | 1.0 | 必须覆盖策略的软限位；允许范围 [0, 1.5]。 |
| 指令边界 | kp ≤ 400，kd ≤ 10 | 超出即锁存故障。 |
| 故障阻尼 | `g1.py` 中的 `safe_joint_kd` | 故障后以 kp = 0 施加。 |

**启动脚本环境变量**（未注明者均属 `start_proteus_pc.sh`）：

| 变量 | 默认值（PC / 板载） | 用途 |
| --- | --- | --- |
| `PROTEUS_POLICY_DIR` | `checkpoints/proteus/proteus_wamseq_62/unitree_g1` | 策略目录。 |
| `PROTEUS_CLIP` | `walk_straight_short` | 另有 `walk_forward_to_stand`、`dance_in_place`、`dance_steps`、`jump_once`。 |
| `PROTEUS_MOTION_PATH`、`PROTEUS_PLAYBACK_LIMIT_S` | 片段目录、片段时长 | 覆盖参考动作 / 回放上限。 |
| `PROTEUS_READ_ONLY`、`PROTEUS_READ_ONLY_S` | `0`、`5` | Bridge `--read-only --max-runtime-s`：真实反馈和推理，不发电机指令。 |
| `PROTEUS_CONTROLLER` | `joystick` / `bridge_joystick` | 板载使用 Bridge 转发的手柄数据，省出一个核。 |
| `PROTEUS_BRIDGE_CPUS`、`PROTEUS_TRACKER_CPUS`、`PROTEUS_MONITOR_CPUS` | `0-3,6-15`、`4,5` / `1`、`2-5`、`6` | `taskset` 绑核。 |
| `SIM2REAL_ORT_NUM_THREADS` | `1` / `4` | ONNX 线程数。 |
| `G1_INTERFACE` | `enp5s0` / `eth0` | DDS 网卡。 |
| `PROTEUS_RUN_DIR` | `outputs/proteus-pc/<时间>` | 日志目录。 |
| `G1_HOST`、`G1_TARGET`、`G1_POLICY_DIRS`、`G1_RSYNC_BWLIMIT`（部署） | `g1`、`/home/unitree/proteus_g1_onboard`、wsq62 + ht500、`20000` KiB/s | 板载部署。 |

## 使用

**1. 准备（每个策略 / 片段一次；文件缺失时启动脚本会自动完成）。**

```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
uv run python scripts/g1/prepare_proteus_deployment.py \
  --policy-config checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy.yaml
uv run python scripts/g1/prepare_proteus_motion.py --clip walk_straight_short
```

**2. Sim2sim 演练。** 无界面，使用真实 Tracker 和 `BridgeSafety`：

```bash
uv run python scripts/g1/validate_proteus_deployment.py --scenario walk-hold \
  --policy-config checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy-pc.yaml \
  --motion-path checkpoints/proteus/proteus_60/unitree_g1/motions/walk_straight_short \
  --npz-playback-limit-s 5.1
```

可加 `--push-velocity-m-s`、`--action-delay-ms`、`--state-gap-s` 或 `--joint-friction <yaml>` 做压力测试。
交互式做法：一个终端运行 `uv run sim2real/sim_env/base_sim.py`，另一个终端运行
`uv run sim2real/rl_policy/tracking.py --robot g1 --robot-io zmq --controller keyboard --policy-config <policy-pc.yaml> --motion-path <片段>`
（按键 `i`、`]`、空格、`o`）。

**3a. 从 PC 上真机。** 机器人挂在龙门架上，操作员手持遥控器并能随时按下硬件急停。

```bash
G1_INTERFACE=enp5s0 PROTEUS_READ_ONLY=1 scripts/g1/start_proteus_pc.sh   # 只读联调
G1_INTERFACE=enp5s0 scripts/g1/start_proteus_pc.sh                       # 受保护的真机会话
```

**3b. 在 Orin 上真机。**

```bash
scripts/g1/onboard.sh deploy
PROTEUS_READ_ONLY=1 PROTEUS_READ_ONLY_S=60 scripts/g1/onboard.sh start
scripts/g1/onboard.sh start      # 受保护的会话；PC 控制器运行时会拒绝启动
scripts/g1/onboard.sh tail       # Ctrl+C 只退出查看，机器人继续运行
scripts/g1/onboard.sh status
```

**4. 操作顺序**（先等 `bridge.log` 出现 `Tracker and LowState ready`）：

| 遥控器（仿真键盘） | 结果 |
| --- | --- |
| A（`i`） | 释放高层模式；10 秒内 Shaping 到双臂下垂站姿。等待 `Joystick: Shaping complete`。 |
| 再按 A（`]`） | 2 秒过渡到片段首帧，然后进入 Hold。等待 `Hold ready`。 |
| 放下龙门架 | Hold 已生效时让双脚落地，保留保护。 |
| Start（空格） | 播放 / 暂停片段；到达回放上限后停在 Final Hold。 |
| Select（`o`） | 锁存的软件停止：目标跟随实测关节，保留 PD 增益，不再主动支撑姿态。 |

Select 或故障后，用 A → A → Start 重新开始。

**5. 关闭。** PC：对启动脚本按 Ctrl+C（或发 SIGTERM），Bridge 会发送一段短暂的阻尼序列。板载：
`scripts/g1/onboard.sh stop`，再用 `scripts/g1/onboard.sh logs` 把运行记录复制到 `outputs/proteus-onboard/`。

**其他任务**

- *Scalecar 上车*：`bash scripts/g1/scalecar/start_bridge.sh` 和 `bash scripts/g1/scalecar/start_tracker.sh`
  （可覆盖 `SCALECAR_PYTHON`、`SCALECAR_POLICY_CONFIG`、`SCALECAR_MOTION_PATH`、`G1_SSH_HOST`、`G1_DEPTH_EXECUTABLE`）；
  产物生成和深度程序编译见 [Scalecar 上车](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/scalecar-vehicle-entry.md)。尚未在真机上运行。
- *关节响应*：`uv run python scripts/g1/joint_response_test.py --target sim`；`--target real --interface enp5s0`
  只能在机器人悬空且没有 Bridge/Tracker 运行时使用。
- *PICO 遥操作*：[Pico 遥操作](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/pico-teleoperation.md)。
- *测试*：`uv run --with pytest python -m pytest tests/test_g1_bridge_safety.py tests/test_planner_real_robot.py`。

## 安全提示与已知问题

- **按 A 后无反应（电机未使能）。** 若 Bridge 显示 active 但机器人不动，且 LowState 中每个 `motor_state[i].motorstate`
  都是 `0x40000000`（tau_est ≈ 0），说明电机未使能。先停止启动脚本，再重启机器人；motorstate 读数恢复为 0 后再启动。
  会话进行中电机重新使能，机器人会猛地跳到当前目标。
- **会话中重启机器人。** 重启前先停止启动脚本：曾有一次 117 秒的重启让 Bridge 在新开机的机器人上恢复旧目标约 3 秒。
  现在超过 1 秒的间隙或 tick 回退都会锁存。
- **接收间隙。** 2026-11-04 行走中出现 343 ms 的 LowState 间隙，锁存阻尼后机器人摔倒。保持目标对站立 Hold 有效；
  仿真中行走时超过约 0.3 秒的盲区仍通常会摔。对比 `bridge.log` 和 `lowstate-monitor.log`，区分机器人/网络问题与 Bridge 进程问题。
- **始终不出现 "Tracker and LowState ready"。** 先确认网卡确实在收发包（看 `ip -s link show <网卡>` 计数），
  再怀疑机器人；测试期间不要让 PC 休眠。
- 每台机器人只能有一个写入者：PC 和板载会话绝不能同时运行，启动脚本会拒绝第二个控制器。
- Select 和被杀掉的进程都不是硬件急停；Select 会失去主动姿态支撑。Shaping 和 Hold 期间保持龙门架保护，硬件急停随手可及。
- 本地参考强制（local forcing）没有根部定位，会漂移（一次行走横向约 0.5 m）。
- 环境变量修改只在下次启动时生效。Orin 时钟比 PC 慢约 12 分钟，运行目录名使用 Orin 时间。

## 详细文档

- [从外部 PC 运行 Proteus G1](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-g1-pc.md)
- [在机器人 Orin 上运行 Proteus G1](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-g1-onboard.md)
- [机器人 I/O](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/robot_io.md) ·
  [根项目](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/root-project.md) ·
  [网络配置](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/network-configuration.md)
- [Scalecar 上车](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/scalecar-vehicle-entry.md) · [ADR（英文）](../../docs/adr)
- [Pico 遥操作](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/pico-teleoperation.md) ·
  [PICO 到 G1 映射](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/pico-to-g1.md)
