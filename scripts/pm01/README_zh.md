# 众擎 PM01 启动脚本

[English](README.md) · 通用流程（整形 → Hold → Go → 末帧 Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

在 24 关节众擎 PM01 / `pm_v2`（`engineai_pm01` 契约；不是 25 关节的 T800，也不是 `engineai_pm01_edu_official`）上
运行 Proteus WAMSeq（wsq61 actor）。由 PC 运行 Tracker（`sim2real/rl_policy/tracking.py`，CPU ONNX 推理 50 Hz），
经以太网通过众擎 `joint_bridge` ROS 2 接口驱动机器人；独立的 500 Hz 发送器持续发布最新命令。传输层与 T800
相同，只有两处 PM01 专属改动：**classic 踝关节 parser**（`parallel_parser_type=0`），以及接受固件 `76445d09`
的零时间戳。本目录还包含只读检查脚本，以及 2026-11 真机排查时使用的限时 relay/bench 工具。

## 状态

| 项目 | 值 |
| --- | --- |
| 真机 | 2026-11-05 首次在吊架保护下行走：398 帧回放全部完成，无复位。**承重状态下的退出未验证**；踝俯仰力矩峰值 106 Nm，训练上限为 61 Nm。真实 PICO 上真机未测试 |
| `--robot` | `pm01`（别名 `pm_v2`、`engineai_pm01`）—— `sim2real/config/robots/__init__.py` |
| 关节数 | 24，SDK 顺序 `J00`–`J23`（`J12_WAIST_YAW`，`J23_HEAD_YAW` 是唯一的头部关节），按名映射 |
| Robot I/O | `--robot-io inline` → `PM01RobotIO`（`interface_protocol` 由机器人上已安装的协议构建）。话题 `/hardware/joint_state`、`/hardware/joint_command`、`/hardware/imu_info`、`/motion/motion_state`；Cyclone DDS，域 69，网口 `eno1` |
| Tracker 运行位置 | PC（机器人上无需安装推理环境）。运动控制计算机 `user@192.168.0.163`，Orin `ubuntu@192.168.0.162` |
| 默认策略 | `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml`（真机）和 `policy-sim.yaml`（仿真） |
| 默认动作 | `checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz`；PICO 保持数据集 `checkpoints/engineai_pm01/pico_hold` |

## 文件

`scripts/pm01/` —— 启动与准备：

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 从 `GMR-engineai` 拷贝并校验 `pm_v2.xml`（轴容差 1e-4），导出 wsq61 ONNX 和 YAML，生成 `xrobot_to_pm01.json` / `bvh_lafan1_to_pm01.json`，生成 PICO 保持数据集，重定向四段 LAFAN1 片段。参数 `--sources-root`、`--proteus-root`、`--release`（默认 `wsq61`）、`--skip-motions`。 |
| `prepare_ros.sh` | 从机器人只读拷贝（rsync）**已安装的** `interface_protocol` msg/srv/action 定义到 `third_party/engineai_pm01/ros2`，并用 colcon 编译。参数：`user@host`（默认 `user@192.168.0.163`）。 |
| `env.sh` | 在 PC shell 中 source：ROS Humble + 编译好的消息、`ROS_DOMAIN_ID`、绑定网口的 Cyclone DDS、HF 离线变量。 |
| `check_robot.py` | 只读就绪检查（只订阅）。当 24 关节反馈、IMU 和 `joint_bridge` 均新鲜，存在兼容的 BEST_EFFORT 命令订阅者且没有其他命令发布者时返回 0，否则返回 2。`--duration`（默认 5 s）。 |
| `start_proteus.sh` | 真机文件回放：source `env.sh`，运行 `check_robot`（带 `--help` 时跳过），然后 `tracking --robot pm01 --robot-io inline --controller keyboard`。第一个可选参数为动作 NPZ。 |
| `start_pico.sh` | PICO 遥操作：`publisher`、`sim`（`--sim-dt 0.000625 --decimation 32`）、`policy-sim [ip]`、`robot [ip]`（先 source `env.sh` 并运行 `check_robot`）。 |

`scripts/pm01/` —— 排查工具（实验性；不是部署路径）：

| 文件 | 用途 |
| --- | --- |
| `shadow_policy.py` | 用真实 PM01 反馈跑完整 Tracker/ONNX 链路 775 帧，不创建任何命令发布者。 |
| `trial_policy.py` | 限时 PC 客户端：只向 `/sim2real/pm01/request` 发布候选命令，需要机载 relay。`--walk-once` 需要新的 `--policy-signal` / `--play-signal` 文件。 |
| `policy_bridge.py` | 机载 relay（运动控制计算机）：试验期间唯一的硬件写入者，附带独立的监督进程，出错时锁存 classic 阻尼。主动命令使用 **RL_PARSER**。 |
| `bench_damping.py` | 机载踝关节阻尼测试；默认只读，`--execute` 允许 ≤1 s 的低踝阻尼。需要 root 权限，且电机驱动二进制哈希必须完全一致。试验阶段使用 RL_PARSER。 |
| `rehearse_trial.py` | 不需要 ROS：在 MuJoCo 中用 mocap weld 模拟吊架，从录制的悬挂姿态复演 relay 闸门。 |

其他相关文件：

- `sim2real/config/robots/pm01.py` + `sim2real/config/robots/specs/engineai_pm01.yaml`（经 `proteus_spec.py` 加载）。
- `sim2real/rl_policy/robot_io/pm01.py` —— `PM01_SDK_JOINT_NAMES`、`PM01Codec`（classic parser、零时间戳处理）。
- `sim2real/rl_policy/robot_io/engineai.py`（与 T800 共用）和 `sim2real/rl_policy/robot_io/ros2_named.py`。
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_pm01.json`。
- `scripts/proteus/`（`prepare_assets.py`、`retarget_lafan.py`、`evaluate.py`）、`scripts/build_xrobot_profile.py`、`scripts/export_proteus_onnx.py`、`scripts/make_hold_motion.py`。
- 测试：`tests/test_pm01.py`、`tests/test_pm01_launch.py`（需显式开启）、`tests/test_pm01_policy_bridge.py`、`tests/test_pm01_bench.py`、`tests/test_proteus_transport.py`、`tests/test_pico_retarget_profiles.py`。
- 实验记录：`docs/experiments/proteus-pm01.json`、`docs/experiments/proteus-pm01-recovery.json`。

## 准备

1. 环境：推理/仿真用根项目（`uv`），发布端和离线 IK 用 `venv/pico`（`--no-sync`）。
2. 源码放在同一上级目录：`Proteus-wamseq`、`GMR-engineai`（含 `assets/engineai_pm01` 和 PM01 IK 配置）、
   `ubisoft-laforge-animation-dataset`。然后构建产物（不在 git 中）：
   ```bash
   uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects
   ```
   输出：`third_party/engineai_pm01/`（模型 + `validation.json`）、`checkpoints/proteus/wsq61/engineai_pm01/`、
   `checkpoints/engineai_pm01/{motions,pico_hold}`。
3. 使用**这台机器人**上的 ROS 消息（旧固件字段为 `current_motion_task_name`；DDS 两端定义必须一致）：
   ```bash
   bash scripts/pm01/prepare_ros.sh user@192.168.0.163
   source scripts/pm01/env.sh
   uv run --no-sync python -m scripts.pm01.check_robot
   ```
4. `env.sh` 和启动脚本会导出 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1` 和 `SIM2REAL_ORT_NUM_THREADS=1`。

## 需调整项

| 参数 | 当前值 | 设置位置 |
| --- | --- | --- |
| 踝关节 parser | `start_proteus.sh` / `start_pico.sh robot` 使用 `parallel_parser_type = 0`（classic，PD 在电机驱动内） | `robot_io/pm01.py` 中的 `PM01Codec`（T800 保持 `1`） |
| 踝关节增益 | kp 37.701、kd 2.4916（训练值不变）。classic 下作用于电机侧：关节侧俯仰约 4.09 倍、横滚约 1.11 倍 | `specs/engineai_pm01.yaml` / `policy.yaml` |
| 零时间戳 | 接受；新鲜度用本地单调接收时间。出现过非零时间戳后再收到零会被拒绝 | `PM01Codec._new_stamp` |
| IMU 话题 / 外参 | `/hardware/imu_info`，`imu_to_pelvis_wxyz: [1, 0, 0, 0]`（IMU 位置与骨盆对齐） | `policy.yaml` 中的 `robot_io_options` |
| 发布频率 / 超时 | `publish_hz: 500`，`timeout_s: 0.25`；模式心跳 1 s | `robot_io_options`；`EngineAICodec._fresh` |
| 停止阻尼 / 步长限制 | 阻尼 2.0，每次发布 0.2 rad（500 Hz 下即 100 rad/s） | `ros2_named.py` 中 `NamedJointCodec` 默认值 |
| 关节限位 | 训练限位；无需厂商限位覆盖 | `prepare.py` → `validation.json` |
| 模型轴容差 | `axis_atol=1e-4`（spec 轴四舍五入到 4 位小数；实测最大 8.76e-5） | `prepare.py` |
| 仿真步长 | **0.000625 s**，decimation 32 | `start_pico.sh sim`；`scripts/proteus/evaluate.py` 对 PM01 的默认值 |
| DDS / 网络 | `PM01_NETWORK_INTERFACE`（`eno1`）、`PM01_ROS_DOMAIN_ID`（69）、`PM01_ROS_SETUP`；已有的 `CYCLONEDDS_URI` 优先 | `env.sh` |
| 远端消息路径 | `PM01_REMOTE_PROTOCOL`（默认 `/apps/engineai_robotics/_install/share/interface_protocol`） | `prepare_ros.sh` |
| 策略配置 | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | 启动脚本 |

## 使用

离线评测与测试：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
source scripts/pm01/env.sh   # ROS 回环测试会自行隔离到 localhost、域 171
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py -q
PM01_RUN_LAUNCH_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync \
  python -m pytest tests/test_pm01_launch.py -q -s
```

机器人上电与模式切换（吊架就位，急停触手可及；启动脚本从不切换模式）：

1. 先给计算机上电，再按白色上电按钮给电机上电（两声短鸣，背部/髋部白灯亮）。不要用颈部红色急停上电。
2. 在运动控制计算机上，若服务未运行：`sudo systemctl start robotics.service`（电机未上电时它可能卡在
   `monitor.service` 的电机检查上）。
3. 固件 `76445d09` 的遥控器操作：**LB + Start**（`idle`，电机使能握手）→ **LB + RB**（`passive`）
   → **LB + A**（`pd_stand`，姿态偏离站立超过 1.2 rad 会拒绝）→ **LB + 十字键左**（`joint_bridge`）。
   新版厂商文档中的 BACK + A 在此版本上无效。检查关节读数，而不只看外观。

真机运行（PC）：

```bash
# 终端 1
bash scripts/pm01/start_proteus.sh checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

1. 双脚悬空，10 s 内不按任何键：机器人应和 `passive` 一样安静。
2. **i** 在 10 s 内整形到 walk 第 0 帧（双臂抬到接近水平）。
3. 降低吊架到双脚轻触地面，然后按 **]** 进入策略 Hold。
4. 逐步转移负载，再按 **空格** 播放。**o** 阻尼；**Ctrl+C** 退出。

PICO 遥操作（发布端在同一台 PC）：

```bash
# 终端 1：XRoboToolkit PC Service + 已标定的身体/腿部追踪器
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# 终端 2：真机策略（MuJoCo 则用 `sim` + `policy-sim 127.0.0.1`）
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

按键：**A** 整形到中立姿态；松开后 **A+B** 进入 Hold；**X** 暂停/恢复；单独按 **B** 进入阻尼。
端口 28701（动作）/ 5592（按键）；每套端口只对应一台机器人。

退出：按 **o**，保持吊架承载，Ctrl+C（关闭时重复发送阻尼），然后 **LB + RB** 回到 `passive`。

排查工具（需要与测试时完全一致的增益、固件哈希和悬挂条件；使用前先读教程）：

```bash
uv run --no-sync python -m scripts.pm01.shadow_policy --policy-config <policy.yaml> --output <new.npz>
uv run --no-sync python -m scripts.pm01.rehearse_trial --policy-config <yaml> --pose-record <rec.npz> \
  --bridge-config <bridge.json> --output <out>
# 在运动控制计算机上以 root 运行，并把 bench_damping.py 复制到同一目录：
python3 policy_bridge.py --execute --config <bridge.json> --output <new-log.jsonl>
# 然后在 PC 上（先 `source scripts/pm01/env.sh`）：
uv run --no-sync python -m scripts.pm01.trial_policy --policy-config <yaml> --motion <static.npz> --output <new.npz>
```

`policy_bridge.py` 选项：`--shaping-only --shaping-seconds {1,3,5,10}`、`--hold-seconds {1,3}`、
`--walk-shaping`、`--walk-once`。`bench_damping.py`：`--execute`、`--ankle-kd {0.05,0.1,0.2,0.3}`、
`--ankle-kp {0,…,37.701}`、`--duration`、`--output`。

## 安全提示与已知问题

- RL_PARSER（`1`）在 2026-11-04 和 2026-11-05 使悬空踝关节振荡并导致整机复位（估计回路延迟 8–9 ms，
  真实踝横滚惯量 0.011–0.016 kg m²，仿真为 0.04）。classic `0` 是有意偏离厂商建议的做法；
  电机读回只验证过 `kp=0` 的 classic 命令。
- relay/bench 工具的主动命令仍使用 RL_PARSER，不要把它们当作部署路径。
- 仅有阻尼无法支撑站立的机器人（`walk-deploy-005`）：负载转移中的反馈超时撤掉了刚度，躯干随之倾斜。
  吊架必须在每一步都能接住机器人。
- 首次行走：踝俯仰力矩 106 Nm（训练上限 61 Nm；仿真在 61 Nm 截断），踝横滚 5 Hz 以上的运动约为仿真的
  2.5 倍，末帧 Hold 中摇晃约 33 s。根位置未测量。
- 零时间戳反馈无法区分重发的旧值与新测量值。`check_robot` 的 `ready: true` 不能证明电机已使能
  （曾出现报告 `joint_bridge` 时 24 个电机禁用标志全部置位的情况）。
- 进程退出或断电需要固件侧保护；bridge runner 是否有命令超时尚未确认。

## 详细文档

- English：[`docs/tutorials/proteus-pm01.md`](../../docs/tutorials/proteus-pm01.md)
- 中文：[`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md)
- 厂商：[engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace)、[PM01 并联踝说明](https://en.engineai.com.cn/open/docs/pm01/04/source-code-analysis)
