# 启元 Q1 启动脚本

本目录包含在启元 Q1（`primebot_q1` contract，22 个策略关节）上运行 Proteus
WAMSeq 跟踪策略的启动脚本和工具。Tracker 通过 inline robot I/O，经 ROS 2
（AimDK 消息）与 Q1 HAL 通信；同一策略也可在 MuJoCo sim2sim 中运行，并支持
PICO 遥操作。本页只讲 Q1 特有的内容。通用流程（Shaping → Hold → Go → Final
Hold、急停、新机器人检查清单）见 [`../README_zh.md`](../README_zh.md)。

## 状态

| 项目 | Q1 |
| --- | --- |
| 硬件状态 | 软件、MuJoCo 与只读实机反馈已测试。首次实机 Shaping 出现双腿交叉；指令顺序修正仅有离线覆盖。策略接管与头显实时操作**尚未验证**。 |
| `--robot` 别名 | `q1`、`primebot_q1`（`sim2real/config/robots/__init__.py`） |
| 关节 | 22 个策略关节，单一线路分组 `body` |
| Robot I/O | `--robot-io inline` → `Q1RobotIO`（`robot_io/q1.py`），基于 `NamedRosRobotIO`（`robot_io/ros2_named.py`）；`aimdk_msgs` 的 `JointStateArray` / `JointCommandArray`，话题 `/aima/hal/joint/state` / `/aima/hal/joint/command`；`sensor_msgs/Imu` 话题 `/aima/hal/imu/state` |
| Tracker 运行位置 | 机器人 ROS 2 网络上的主机 PC（教程中的“已检查 PC”，有线 DDS）。文档中没有板载部署方案。 |
| 默认 checkpoint | `wsq61`（`wsq61_wam_61robots_30000.pt`）：`checkpoints/proteus/wsq61/primebot_q1/` |
| 策略 YAML | `policy-sim.yaml`（sim2sim、PICO 仿真）、`policy.yaml`（实机基础配置、PICO `robot` 模式）、`policy-pc.yaml`（由 `prepare_deployment.py` 生成的实机覆盖配置，`start_proteus.sh` 使用） |

## 文件

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 构建厂商模型，导出 WAMSeq ONNX（`--release`，默认 `wsq61`），生成 PICO 重定向配置、`pico_hold` 数据集和三段 LAFAN1 动作。 |
| `prepare_deployment.py` | 在 `policy.yaml` 旁写入实机覆盖配置（默认 `policy-pc.yaml`）：IMU 话题/外参、关节限位、限位模式、进入门限。 |
| `start_proteus.sh` | 实机文件回放：inline I/O、键盘控制器、NPZ 动作、`policy-pc.yaml`。 |
| `start_pico.sh` | PICO 遥操作：`publisher`、`sim`、`policy-sim`、`robot`。 |
| `validate_deployment.py` | 离线演练（不连 ROS）：用真实的 Tracker、codec 和序列化器驱动带吊架焊接约束的 MuJoCo。 |
| `commission_joints.py` | 默认未上电的实测位姿保持与单关节 ±0.03 rad 探测，走生产传输链路，不加载策略。 |
| `inspect_imu.py` | 只读 IMU 订阅；打印假定骨盆系下的 RPY、陀螺仪和重力偏差。 |

其他位置的相关文件：

- 机器人配置：`sim2real/config/robots/q1.py`（来自 `sim2real/config/robots/specs/primebot_q1.yaml`）。
- Robot I/O：`sim2real/rl_policy/robot_io/q1.py`、`sim2real/rl_policy/robot_io/ros2_named.py`，分发在 `sim2real/rl_policy/robot_io/factory.py`。
- 目标后处理：`sim2real/rl_policy/utils/joint_targets.py`。
- 策略 YAML 导出（Q1 的 `robot_io_options`）：`scripts/export_proteus_onnx.py`。
- 遥操作配置：`sim2real/teleop/configs/xrobot_to_q1.json`、`bvh_lafan1_to_q1.json`。
- 共享 Proteus 工具：`scripts/proteus/`（`evaluate.py`、`prepare_assets.py`、`build_retarget_profiles.py`、`retarget_lafan.py`）、`scripts/make_hold_motion.py`。
- 测试：`tests/test_q1_commission.py`、`tests/test_proteus_transport.py` 中的 Q1 用例、`tests/test_proteus_sim_resolution.py`、`tests/test_tracker_action_postprocess.py`、`tests/test_pico_retarget_profiles.py`。
- 实验记录：`docs/experiments/proteus-q1.json`。

## 环境准备

- **ROS 2 与厂商消息。** source ROS 2 Humble 以及提供 `aimdk_msgs` 的厂商工作空间
  （[primebot-q1-sdk](https://github.com/PrimebotTech/primebot-q1-sdk)）。
  Python 环境不会安装机器人 SDK。
  ```bash
  source /opt/ros/humble/setup.bash
  source /path/to/vendor_workspace/install/setup.bash
  ```
- **网络。** 主机需能通过 DDS 看到 Q1 HAL 话题。在已检查的 PC 上，有线 DDS 设置位于
  本地的 `outputs/q1/source_env.sh`（不在 git 中）。
- **厂商控制模式。** 将厂商控制器切到 Advanced / 底层开发模式，并停止其他指令发布程序。
- **Python 环境。** 策略、仿真和工具使用根目录 `uv` 项目；PICO IK 与离线重定向使用
  `venv/pico`。启动脚本会自动选择。
- **HF 离线。** 设置 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`。两个启动脚本已自动设置。
- **运行时产物（不在 git 中）。** ONNX、厂商资源、动作和保持数据集需在本地重建。
  `--sources-root` 下需有：`Proteus-wamseq`（Proteus `wamseq` 工作树）、
  `primebot-q1-urdf`、`GMR`、`ubisoft-laforge-animation-dataset`。然后运行：
  ```bash
  uv run --no-sync python -m scripts.q1.prepare --sources-root /path/to/projects
  # 可选：--release <name>、--proteus-root <dir>、--skip-motions
  ```
  输出：`checkpoints/proteus/<release>/primebot_q1/`、
  `checkpoints/primebot_q1/motions/`、`checkpoints/primebot_q1/pico_hold/`。

## 可调项

实机可调项位于 `policy-pc.yaml`（由 `prepare_deployment.py` 写入）以及
`Q1RobotIO` / `NamedJointCodec` 的默认值中。

| 设置 | 当前值 | 位置 |
| --- | --- | --- |
| `policy_limit_mode` | `clip`（默认）；`pass`、`feedforward` 为实验性 | `prepare_deployment.py --policy-limit-mode` → `robot_io_options` |
| `limit_feedforward_band_rad` | `0.1`（前馈在距限位最后 0.1 rad 内线性衰减到零） | `NamedJointCodec` 默认值 |
| 关节限位 | `clip` → `soft_joint_pos_limits`（Tracker 同时裁剪策略和 Shaping 目标）；`pass`/`feedforward` → `shaping_joint_pos_limits`（仅限 Shaping） | `prepare_deployment.py` |
| `max_target_velocity_rad_s` | `15.0` | `prepare_deployment.py` |
| 进入门限 | 守护模式：`policy_entry_max_joint_error_rad 0.10`、`policy_entry_max_joint_velocity_rad_s 0.5`；`--manual-policy-entry` 移除二者 | `prepare_deployment.py` |
| `shaping_target` / `reference_entry_s` | `motion_first_frame` / `0.0` | `prepare_deployment.py` |
| Shaping 重力前馈 | 关闭；`--enable-shaping-gravity` 启用，力矩上限的 10%，2 s 爬升 | `prepare_deployment.py` |
| `allow_init_from_measured` | `True`：实测位姿超出模型范围时，Shaping 逐步拉回 | `Q1RobotIO` |
| `start_disarmed` | `True`：首次 `i` / `]` 之前不发送任何电机指令（退出时也不发） | `Q1RobotIO` |
| `preserve_feedback_order` | `True`：所有指令字段按 HAL 反馈的槽位顺序发送 | `Q1RobotIO` |
| IMU | 话题 `/aima/hal/imu/state`，`imu_to_pelvis_wxyz: [1, 0, 0, 0]` | 覆盖配置参数；`imu_topic` 未设置时回退到 `SIM2REAL_Q1_IMU_TOPIC` |
| `publish_hz` | `1000`（策略 50 Hz） | `Q1RobotIO`、`export_proteus_onnx.py` |
| `timeout_s` | `0.25`（反馈/指令过期 → 阻尼） | `export_proteus_onnx.py` |
| 阻尼 | 零模式、超时或出错时 kd `5`、kp `0` | `Q1RobotIO(damping=5.)` |
| `max_position_step` | 每次发布 `0.2` rad | `Q1RobotIO` |
| 仿真步长 | `0.000625` s（`evaluate.py` 对 Q1 的默认值；`start_pico.sh sim` 使用 `--decimation 32`） | `scripts/proteus/evaluate.py`、`start_pico.sh` |

说明：

- **限位模式。** Shaping 目标始终被裁剪。`pass` 原样发送策略目标；在 Q1 上只应配合
  手臂下垂起始的动作使用（2026-11-06，walk1 的 T-pose 起始使右臂甩到 -2.28 rad；
  肩部横滚在模型限位处没有硬限位）。`feedforward` 额外施加力矩 kp·(target − bound)；
  已有证据表明 Q1 硬件链路可能忽略 `effort`，此时其效果等同 `clip`。
- **IMU。** 厂商 V3 MJCF 将 `imu_0` 无旋转地放在 `pelvis` 上，因此使用单位四元数。
  实时 ROS IMU 坐标系是否与模型一致仍需在实机上验证。不要为了得到“直立”读数而抵消
  机器人当前的倾斜。

## 使用

**Sim2sim 评估**（MuJoCo，默认 0.000625 s）：

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot q1 \
  --policy-config checkpoints/proteus/wsq61/primebot_q1/policy-sim.yaml \
  --dataset checkpoints/primebot_q1/motions --output outputs/q1/eval-new
```

**生成实机覆盖配置**（写入 `policy-pc.yaml`；如需变体可加
`--policy-limit-mode pass --output-name policy-pc-pass.yaml`）：

```bash
uv run --no-sync python -m scripts.q1.prepare_deployment \
  --imu-topic /aima/hal/imu/state --imu-to-pelvis-wxyz 1 0 0 0 --manual-policy-entry
```

**离线演练**（不连 ROS；失败时非零退出；写入 `outputs/q1/deployment-rehearsal.{json,npz}`）：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
  uv run --no-sync python -m scripts.q1.validate_deployment \
  --policy-config checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml
# 可选：--initial-feedback <ros-readonly.json> | --initial-tracking-record <record.npz>
#       --no-use-recorded-orientation
```

**关节调试**（吊架承重、双腿不交叉、厂商急停在手边）：

```bash
uv run --no-sync python -m scripts.q1.commission_joints --joint left_elbow_joint
uv run --no-sync python -m scripts.q1.commission_joints --read-only --duration-s 15
```

启动时未上电。`h` 捕获并保持实测位姿（kp 在 2 s 内从 0 升到 5，kd 2）；`j` 让所选
关节在 8 s 内走一次 `+0.03 → 0 → -0.03 → 0` rad；`o` / 空格 / `q` / Esc / Ctrl+C
进入阻尼并退出。漂移 > 0.10 rad、速度 > 0.5 rad/s、反馈过期、出现第二个指令发布者或
线路顺序变化时自动停止；上电后最长 25 s。日志写入 `outputs/q1/commission/`。不会
自动判定通过：测下一个关节前，先确认物理关节身份与方向。

**IMU 检查**（仅订阅，不发指令）：

```bash
uv run --no-sync python -m scripts.q1.inspect_imu --duration-s 30
# --topic、--imu-to-pelvis-wxyz、--output <file.json>
```

**实机运行**（文件回放，键盘）：

```bash
bash scripts/q1/start_proteus.sh checkpoints/primebot_q1/motions/motions/walk1_subject1.npz
# PROTEUS_POLICY_CONFIG=<yaml> 可覆盖 policy-pc.yaml
```

按键：`i` Shaping（10 s，手动进入模式下可选）；`]` 进入策略 Hold；空格播放/暂停参考
动作（不是停止）；`o` 阻尼；Ctrl+C 退出。Hold 是主动策略控制：参考暂停时肢体仍可能运动。

**PICO 遥操作：**

```bash
bash scripts/q1/start_pico.sh publisher --actual-human-height 1.8   # 终端 1
bash scripts/q1/start_pico.sh sim                                   # 终端 2
bash scripts/q1/start_pico.sh policy-sim 127.0.0.1                  # 终端 3
# 实机替代终端 3：
PROTEUS_POLICY_CONFIG=checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml \
  bash scripts/q1/start_pico.sh robot PUBLISHER_IP
```

动作走 TCP 28701，按键走 TCP 5592。**A** Shaping 到中立位；**A+B** 进入策略 Hold；
**X** 暂停/恢复实时参考；单按 **B** 进入阻尼。数据流中断时保持最后一帧参考，不会
自动阻尼。`robot` 模式默认使用 `policy.yaml`，其中 `imu_topic` 未设置；请按上面方式
使用覆盖配置，或设置 `SIM2REAL_Q1_IMU_TOPIC`。

**关闭。** `o`（或 PICO **B**）进入阻尼；Ctrl+C 退出。上电后正常退出会连续发送几次
阻尼指令。这些只是进程层面的检查，进程被杀或断电时并非固件级保护；请确保厂商急停
随时可用。

## 安全说明与已知问题

- 软件、MuJoCo 与只读实机反馈已测试。2026-11-04 的 501 tick 只读 wsq61 探测中，
  22 个关节名全部匹配，无反馈过期。
- **双腿交叉（2026-11-04）。** 首次仅按 `i` 的 Shaping 试验中双腿交叉：适配器按
  腿在前的顺序发送，而 HAL 反馈是手臂在前，且看起来按槽位消费指令。
  `preserve_feedback_order` 已在软件层面修正，并有离线覆盖，但**尚未**通过新的实机
  试验。下一步应先运行 `commission_joints.py`，而不是全身 Shaping。
- 策略接管与头显实时操作仍未验证。
- 低 kp 下观察到持续的肩部下垂；重力前馈未体现在上报的力矩中。
- `validate_deployment.py` 将吊架理想化为焊接约束；通过演练不代表实机释放安全。
- 开发者模式和空的发布者列表并不能证明厂商控制器已交出控制权。

## 详细文档

- [启元 Q1 上的 Proteus WAMSeq](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-q1.md)
  （[English](../../docs/tutorials/proteus-q1.md)）
- 通用流程：[`../README_zh.md`](../README_zh.md)
