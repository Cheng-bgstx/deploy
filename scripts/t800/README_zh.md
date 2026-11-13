# 众擎 T800 启动脚本

[English](README.md) · 通用流程（整形 → Hold → Go → 末帧 Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

在标准 25 电机众擎 T800（`engineai_t800` 契约；不是任一 T800 Pro 契约）上运行 Proteus WAMSeq。发布的共享
actor 原样导出为 ONNX，无需训练或微调。Tracker（`sim2real/rl_policy/tracking.py`）以 50 Hz 在 CPU 上做
ONNX 推理，通过众擎 `joint_bridge` ROS 2 接口（`interface_protocol`）驱动机器人；另有一个 500 Hz 的 ROS
定时器以 `parallel_parser_type=1`（RL_PARSER）持续发送最新命令。**软件和 MuJoCo 已测试；真机和真实 PICO
头显均未测试。**

## 状态

| 项目 | 值 |
| --- | --- |
| 真机 | 未测试。仅有软件测试和 MuJoCo 评测（`docs/experiments/proteus-t800.json`：`hardware_tested: false`、`physical_pico_tested: false`） |
| `--robot` | `t800`（别名 `engineai_t800`）—— `sim2real/config/robots/__init__.py` |
| 关节数 | 25（SDK 顺序 `J00_HIP_PITCH_L` … `J24_HEAD_YAW`，按名映射到策略顺序） |
| Robot I/O | `--robot-io inline` → `T800RobotIO`（`interface_protocol` 的 JointState/JointCommand/ImuInfo/MotionState）。话题 `/hardware/joint_state`、`/hardware/joint_command`、`/hardware/imu_info`、`/motion/motion_state` |
| Tracker 运行位置 | 任意接入机器人 ROS 2 网络、已 source ROS Humble 和厂商工作空间的主机；启动脚本不指定主机 |
| 默认策略（启动脚本） | `checkpoints/proteus/wsq61/engineai_t800/policy.yaml`（真机）和 `policy-sim.yaml`（仿真） |
| 已验证版本 | `checkpoints/proteus/proteus_wamseq_62/engineai_t800/` —— 2026-11-03 的评测表和实验记录针对它，而非 wsq61 |
| 默认动作 | `checkpoints/engineai_t800/motions/motions/walk1_subject1.npz`；PICO 保持数据集 `checkpoints/engineai_t800/pico_hold` |

## 文件

`scripts/t800/`：

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 一次性构建资产：从 `GMR-engineai` 拷贝并校验 `t800_mocap.xml`，导出 ONNX 和 YAML，把众擎的 `smplx_to_t800.json` 转换为 `xrobot_to_t800.json` 并复制 `bvh_lafan1_to_t800.json`，生成 PICO 保持数据集，重定向四段 LAFAN1 片段。参数 `--sources-root`、`--proteus-root`、`--release`（默认 `wsq61`）、`--skip-motions`。 |
| `start_proteus.sh` | 真机文件回放：`tracking --robot t800 --robot-io inline --controller keyboard --motion-backend npz`。第一个可选参数为动作 NPZ。 |
| `start_pico.sh` | PICO 遥操作：`publisher`、`sim`（附加 `--sim-dt 0.000625 --decimation 32`）、`policy-sim [publisher-ip]`、`robot [publisher-ip]`。 |
| `__init__.py` | 使 `python -m scripts.t800.prepare` 可导入。 |

其他相关文件：

- `sim2real/config/robots/t800.py` + `sim2real/config/robots/specs/engineai_t800.yaml`（经 `proteus_spec.py` 加载）。
- `sim2real/rl_policy/robot_io/t800.py` —— SDK 关节名；`T800Codec` 继承 RL_PARSER。
- `sim2real/rl_policy/robot_io/engineai.py` —— 众擎共享传输层（PM01 也用）：`joint_bridge` 模式归属、1 s 模式心跳、仅在存在命令订阅者时发布。
- `sim2real/rl_policy/robot_io/ros2_named.py` —— IMU 外参、250 ms 看门狗、限位裁剪、0.2 rad 步长限制、停止时阻尼。
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_t800.json` —— 实时 PICO IK。
- `scripts/proteus/`（`prepare_assets.py`、`retarget_lafan.py`、`evaluate.py`）、`scripts/build_xrobot_profile.py`、`scripts/export_proteus_onnx.py`、`scripts/make_hold_motion.py`。
- 测试：`tests/test_proteus_transport.py`、`tests/test_proteus_sim_resolution.py`、`tests/test_pm01.py`（`test_pm01_uses_classic_parser_while_t800_keeps_rl_parser`）、`tests/test_pico_retarget_profiles.py`。
- 实验记录：`docs/experiments/proteus-t800.json`。

## 准备

1. 环境：推理/仿真用根项目（`uv`），发布端和离线 IK 用 `venv/pico`。启动脚本使用 `--no-sync`，须先装好。
2. 源码放在仓库旁：`Proteus-wamseq`、`GMR-engineai`（engineai-robotics/GMR）、`GMR`（YanjieZe）、
   `ubisoft-laforge-animation-dataset`。克隆命令见教程。
3. 构建运行时产物（不在 git 中）。默认 release 与启动脚本一致：
   ```bash
   uv run --no-sync python -m scripts.t800.prepare --sources-root "$sources_root"
   ```
   教程中的命令传了 `--release proteus_wamseq_62`；若构建该版本，请用 `PROTEUS_POLICY_CONFIG` 让启动脚本
   指向它。输出位于 `third_party/engineai_t800/`、`checkpoints/proteus/<release>/engineai_t800/`、
   `checkpoints/engineai_t800/{motions,pico_hold}`。
4. 仅真机：需要实现了文档所述 `joint_bridge` 命令订阅的固件
   （[engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace)）。检查过的原生
   SDK 版本 `335c60e` 只有遥测，没有关节命令订阅者，无法运行本适配器。
   source `/opt/ros/humble/setup.bash` 以及提供 `interface_protocol` 的厂商工作空间。
5. 启动脚本会导出 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`，并在未设置时设 `SIM2REAL_ORT_NUM_THREADS=1`。

## 需调整项

| 参数 | 当前值 | 设置位置 |
| --- | --- | --- |
| IMU 话题 | `/hardware/imu_info`（`ImuInfo`） | `policy.yaml` 中的 `robot_io_options.imu_topic`（由 `scripts/export_proteus_onnx.py` 写入）；默认值在 `robot_io/engineai.py` |
| IMU → 骨盆外参 | `imu_to_pelvis_wxyz: [1, 0, 0, 0]` | `robot_io_options` |
| 命令发布频率 | `publish_hz: 500` | `robot_io_options` |
| 反馈/命令超时 | `timeout_s: 0.25`；模式心跳 1 s | `robot_io_options`；心跳在 `EngineAICodec._fresh` |
| 停止阻尼 / 步长限制 | `damping` 2.0，`max_position_step` 每次发布 0.2 rad | `NamedJointCodec` 默认值；可经 `robot_io_options` 覆盖 |
| 踝关节 parser | `parallel_parser_type = 1`（RL_PARSER：由 SDK 转换串联俯仰/横滚踝 PD） | `robot_io/engineai.py` 中的 `EngineAICodec` |
| 踝关节增益 | 冻结 spec 中 kp 107.2198、kd 8.2957、力矩 160 Nm | `specs/engineai_t800.yaml`；实际下发的 `joint_kp`/`joint_kd` 来自 `policy.yaml` 中导出的契约 |
| 关节限位 | 仅用训练限位；厂商模型与训练一致，无物理限位覆盖 | `robot_io/t800.py`（未传入） |
| 仿真步长 | **0.000625 s**，decimation 32（策略 50 Hz） | `start_pico.sh sim`；`scripts/proteus/evaluate.py` 对 T800 的默认值（可用 `--sim-dt` 覆盖） |
| 策略配置 | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | 两个启动脚本 |

0.625 ms 步长很关键：在旧的 5 ms 外部 PD 循环里，T800 跟踪明显更差（匹配样本身体误差 12.72 cm 对 5.62 cm）；
`tests/test_proteus_sim_resolution.py` 对此做回归检查。

## 使用

离线评测（自动使用 0.000625 s）：

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot t800 \
  --policy-config checkpoints/proteus/wsq61/engineai_t800/policy-sim.yaml \
  --dataset checkpoints/engineai_t800/motions --output outputs/t800/eval-new
```

仿真中的 PICO 遥操作：

```bash
# 终端 1：XRoboToolkit PC Service + 已标定的 PICO 身体/腿部追踪器
bash scripts/t800/start_pico.sh publisher --actual-human-height 1.8
# 终端 2：MuJoCo 仿真器（打开其 mjviser URL；评估平衡前取消勾选 Elastic Band）
bash scripts/t800/start_pico.sh sim
# 终端 3：Tracker，从终端 1 读取动作（TCP 28701）和按键（TCP 5592）
bash scripts/t800/start_pico.sh policy-sim 127.0.0.1
```

PICO 按键：**A** 整形到中立姿态；松开后 **A+B** 进入策略 Hold；**X** 暂停/恢复实时参考；单独按 **B** 进入阻尼。

真机：

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
# 先用厂商遥控器进入 joint_bridge，并检查 /motion/motion_state。
bash scripts/t800/start_proteus.sh checkpoints/engineai_t800/motions/motions/walk1_subject1.npz
# 或改用 PICO 遥操作（发布端在其他机器上）：
bash scripts/t800/start_pico.sh robot PUBLISHER_IP
```

键盘：**i** 在 10 s 内整形到第 0 帧，**]** 进入策略 Hold，**空格** 播放/暂停，**o** 阻尼，**Ctrl+C** 退出。
额外参数会转发给 `tracking`（例如 `--record`）。

退出：按 **o**（或 **B**），再 Ctrl+C；关闭时传输层会重复发送阻尼。启动脚本从不切换机器人模式——
请用厂商遥控器退出 `joint_bridge`。

## 安全提示与已知问题

- 尚无真机运行记录。首次尝试务必使用吊架，急停触手可及。
- 只有当 `/motion/motion_state` 报告 `joint_bridge`（1 s 内更新）且存在命令订阅者时传输层才发布，否则什么都不发。
  显式切换模式会把控制权交还固件。
- 命令/反馈超过 250 ms、出错或 zero 模式 → 刚度、速度和前馈归零，阻尼为 2。仅为进程内检查：
  进程被杀或断电后不保证有固件保护。
- wsq61 默认版本不在已发布的评测表覆盖范围内（评测用的是 `proteus_wamseq_62`）。
- PM01 共用此传输层，但因真机踝关节振荡已把 parser 改为 classic（`0`）；T800 仍用 RL_PARSER，且未经真机验证。
  首次 T800 真机运行前请阅读 `scripts/pm01/README_zh.md`。
- PICO 数据流中断时保持最后的参考，只要机器人状态正常就不会自动阻尼。

## 详细文档

- English：[`docs/tutorials/proteus-t800.md`](../../docs/tutorials/proteus-t800.md)
- 中文：[`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-t800.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-t800.md)
- PICO 配置：`docs/tutorials/pico-teleoperation.md`
