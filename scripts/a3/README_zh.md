# 智元远征 A3 启动脚本

[English](README.md) · 通用流程（整形 → Hold → Go → 末帧 Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

在标准版智元远征 A3（`agibot_a3` 契约，公开的 T2.5 模型；不是 A3 Ultra，也不是 T3.0 版本）上运行 Proteus WAMSeq。
发布的共享 actor 原样导出为 ONNX，无需训练或微调。Tracker（`sim2real/rl_policy/tracking.py`）以 50 Hz 在
CPU 上做 ONNX 推理，通过官方 `joint_msgs` ROS 2 接口与机器人通信；另有一个 500 Hz 的 ROS 定时器持续发送
最新命令。**软件和 MuJoCo 已测试；真机和真实 PICO 头显均未测试。**

## 状态

| 项目 | 值 |
| --- | --- |
| 真机 | 未测试。仅有软件测试和 MuJoCo 评测（`docs/experiments/proteus-a3.json`：`hardware_tested: false`、`physical_pico_tested: false`） |
| `--robot` | `a3`（别名 `agibot_a3`）—— `sim2real/config/robots/__init__.py` |
| 关节数 | 30 个策略关节；厂商机器人有 31 个电机。`head_pitch_joint` 不在契约中，固定为 0 |
| Robot I/O | `--robot-io inline` → `A3RobotIO`（ROS 2，`joint_msgs` 的 Command/JointCommand、State/JointState）。话题 `/body_drive/{waist,neck,arm,leg}_joint_{state,command}`，IMU 为 `/body_drive/pelvis_imu/data`（`sensor_msgs/Imu`） |
| Tracker 运行位置 | 任意接入机器人 ROS 2 网络、已 source ROS Humble 和厂商 `joint_msgs` 工作空间的主机；启动脚本不指定主机 |
| 默认策略 | `checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml`（真机，参考锚点作根）和 `policy-sim.yaml`（sim2sim，实测世界根） |
| 默认动作 | `checkpoints/agibot_a3/motions/motions/walk1_subject1.npz`；PICO 保持数据集 `checkpoints/agibot_a3/pico_hold` |

## 文件

`scripts/a3/`：

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 一次性构建资产：拷贝并校验厂商 MJCF（固定 `head_pitch_joint`），导出 WAMSeq ONNX 和 YAML，生成 `xrobot_to_a3.json` / `bvh_lafan1_to_a3.json`，生成 PICO 保持数据集，重定向三段 LAFAN1 片段。参数 `--sources-root`、`--proteus-root`、`--release`（默认 `proteus_wamseq_62`）、`--skip-motions`。 |
| `start_proteus.sh` | 真机文件回放：`tracking --robot a3 --robot-io inline --controller keyboard --motion-backend npz`。第一个可选参数为动作 NPZ。 |
| `start_pico.sh` | PICO 遥操作，每个终端一种模式：`publisher`、`sim`、`policy-sim [publisher-ip]`、`robot [publisher-ip]`。 |
| `__init__.py` | 使 `python -m scripts.a3.prepare` 可导入。 |

其他相关文件：

- `sim2real/config/robots/a3.py` + `sim2real/config/robots/specs/agibot_a3.yaml`（经 `proteus_spec.py` 加载）—— 关节顺序、限位、增益、电枢惯量、MJCF 路径。
- `sim2real/rl_policy/robot_io/a3.py` —— 关节分组、话题、膝关节物理限位、固定的头部俯仰。
- `sim2real/rl_policy/robot_io/ros2_named.py` —— 共享的按名关节编解码器：IMU 外参、看门狗、限位裁剪、每次发布 0.2 rad 步长限制、停止时阻尼。
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_a3.json` —— 实时 PICO IK（关节速度上限 12 rad/s）。
- `scripts/proteus/` —— 共享的 `prepare_assets.py`、`build_retarget_profiles.py`、`retarget_lafan.py`、`evaluate.py`；以及 `scripts/export_proteus_onnx.py`、`scripts/make_hold_motion.py`。
- 测试：`tests/test_proteus_transport.py`、`tests/test_proteus_observations.py`、`tests/test_robot_io.py`、`tests/test_command_sender.py`、`tests/test_pico_retarget_profiles.py`。
- 实验记录：`docs/experiments/proteus-a3.json`。

## 准备

1. 环境：推理/仿真用根项目（`uv`），PICO 发布端和离线 IK 用 `venv/pico`。启动脚本使用 `--no-sync`，
   两个环境须提前装好（见 Getting Started）。
2. 源码放在仓库旁（默认是 git common dir 的上级目录）：`Proteus-wamseq`（`Proteus` 在 `origin/wamseq`
   的 worktree）、`A3-A3U-robot-model`、`GMR`（YanjieZe）、`ubisoft-laforge-animation-dataset`。
   具体克隆命令见教程。
3. 构建运行时产物（不在 git 中）：
   ```bash
   uv run --no-sync python -m scripts.a3.prepare --sources-root "$sources_root"
   ```
   输出：`third_party/agibot_a3/`（模型 + `validation.json`）、`checkpoints/proteus/proteus_wamseq_62/agibot_a3/`、
   `checkpoints/agibot_a3/{motions,pico_hold}`。
4. 仅真机：安装提供 `joint_msgs` 的厂商 ROS 2 工作空间（线缆协议来自
   [sonic_for_a3](https://github.com/AgibotTech/sonic_for_a3)），并与 `/opt/ros/humble/setup.bash` 一起 source。
   `prepare.py` 不安装机器人 SDK。
5. 每个启动脚本都会导出 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`，并在未设置时设
   `SIM2REAL_ORT_NUM_THREADS=1`。手动运行 Python 的 shell 中也请设置同样的变量。

## 需调整项

| 参数 | 当前值 | 设置位置 |
| --- | --- | --- |
| IMU 话题 | `/body_drive/pelvis_imu/data` | `policy.yaml` 中的 `robot_io_options.imu_topic`（由 `scripts/export_proteus_onnx.py` 写入）；默认值在 `robot_io/a3.py` |
| IMU → 骨盆外参 | `imu_to_pelvis_wxyz: [1, 0, 0, 0]` —— **按实际安装的传感器设置** | `policy.yaml` 的 `robot_io_options` |
| 命令发布频率 | `publish_hz: 500` | `robot_io_options` |
| 反馈/命令超时 | `timeout_s: 0.25` | `robot_io_options`（编解码器默认值在 `ros2_named.py`） |
| 停止阻尼 / 步长限制 | `damping` 2.0，`max_position_step` 每次发布 0.2 rad | `ros2_named.py` 中 `NamedJointCodec` 默认值；可经 `robot_io_options` 覆盖 |
| 策略模式下的限位处理 | `policy_limit_mode: clip`（另有 `pass`、`feedforward`） | `NamedJointCodec`；可经 `robot_io_options` 覆盖 |
| 真机膝关节限位 | 训练 ∩ 物理：上限 2.49582 rad（训练为 2.53073） | `robot_io/a3.py` 中的 `physical_limits` |
| 固定头部俯仰 | `head_pitch_joint`：q=0、dq=0、τ=0、kp=3、kd=0.3 | `robot_io/a3.py` 中的 `fixed_joints`；`prepare.py` 在仿真模型中去掉该关节 |
| 整形目标 | `shaping_target: motion_first_frame` | `policy.yaml` |
| 踝关节增益 | 冻结 spec 中 kp 64.6461、kd 4.708 | `specs/agibot_a3.yaml`；实际下发的 `joint_kp`/`joint_kd` 来自 `policy.yaml` 中导出的契约 |
| 仿真步长 | 0.005 s，decimation 4（base_sim/`evaluate.py` 默认值） | `start_pico.sh sim`、`scripts/proteus/evaluate.py` |
| 策略配置 | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | 两个启动脚本 |

厂商模型质量与训练相差约 0.194 kg，保留不改；有五个关节范围与厂商不同（仿真使用训练限位）。
运行 `prepare.py` 后详见 `third_party/agibot_a3/validation.json`。

## 使用

离线评测（MuJoCo，实测根）：

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot a3 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy-sim.yaml \
  --dataset checkpoints/agibot_a3/motions --output outputs/a3/eval-new
```

仿真中的 PICO 遥操作（发布端在同一台 PC）：

```bash
# 终端 1：XRoboToolkit PC Service + 已标定的 PICO 身体/腿部追踪器
bash scripts/a3/start_pico.sh publisher --actual-human-height 1.8
# 终端 2：MuJoCo 仿真器（打开其 mjviser URL；评估平衡前取消勾选 Elastic Band）
bash scripts/a3/start_pico.sh sim
# 终端 3：Tracker，从终端 1 读取动作（TCP 28701）和按键（TCP 5592）
bash scripts/a3/start_pico.sh policy-sim 127.0.0.1
```

PICO 按键：**A** 整形到中立姿态；松开后 **A+B** 进入策略 Hold；**X** 暂停/恢复实时参考（发布端处理）；
单独按 **B** 进入阻尼。控制前先在发布端查看器中检查 IK。

真机（先按厂商流程释放原生控制器）：

```bash
# 终端 1
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
bash scripts/a3/start_proteus.sh checkpoints/agibot_a3/motions/motions/walk1_subject1.npz
# 或改用 PICO 遥操作：终端 1 运行 `start_pico.sh publisher`，然后
bash scripts/a3/start_pico.sh robot PUBLISHER_IP
```

键盘（`start_proteus.sh`）：**i** 在 10 s 内整形到第 0 帧，**]** 进入策略 Hold，**空格** 播放/暂停，
**o** 阻尼，**Ctrl+C** 退出。额外参数会转发给 `tracking`，例如 `--record`。

退出：先按 **o**（PICO 上按 **B**），再 Ctrl+C。关闭时传输层会重复发送阻尼命令。

## 安全提示与已知问题

- 尚无真机运行记录。在 A3 真机上运行时必须有吊架保护，急停触手可及。
- 反馈或命令超过 250 ms、出错、非有限数值或 zero 模式 → 刚度、速度和前馈力矩归零并保留阻尼。
  这些是进程内检查；不代表进程被杀或断电后有固件保护。
- IMU 外参默认为单位四元数；安装方向不对会悄无声息地破坏重力/姿态观测。
- PICO 数据流中断时保持最后的参考，只要机器人状态正常就**不会**自动进入阻尼。
  每套默认端口（28701/5592；手部扳机仍为 5593，这里没有手部驱动）只能对应一台机器人。
- `policy.yaml` 使用参考锚点（无定位）；`policy-sim.yaml` 需要实测世界根，仅用于仿真。
- 文件跟踪指标（三段默认片段均在 0.5 m 阈值内通过）不能代表实时控制质量。

## 详细文档

- English：[`docs/tutorials/proteus-a3.md`](../../docs/tutorials/proteus-a3.md)
- 中文：[`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-a3.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-a3.md)
- PICO 配置：`docs/tutorials/pico-teleoperation.md`
