# 众擎 PM01 启动脚本

[English](README.md) · 通用流程（整形 → Hold → Go → 末帧 Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

在 24 关节众擎 PM01 / `pm_v2`（`engineai_pm01` 契约；不是 25 关节的 T800，也不是 `engineai_pm01_edu_official`）上
运行 Proteus WAMSeq。Tracker（`sim2real/rl_policy/tracking.py`，CPU ONNX 推理 50 Hz）可以在 PC 上运行、经以太网
驱动机器人，也可以**机载**运行在 PM01 的运动控制计算机（Nezha）上。两种方式都通过众擎 `joint_bridge` ROS 2 接口
控制，由独立的发送器持续发布最新命令。传输层与 T800 相同，PM01 专属改动有：**classic 踝关节 parser**
（`parallel_parser_type=0`）、接受固件 `76445d09` 的零时间戳，以及可选的踝关节和手臂增益缩放（由机载启动脚本
设置）。本目录还包含只读检查脚本，以及 2026-11 真机排查时使用的限时 relay/bench 工具。

## 状态

| 项目 | 值 |
| --- | --- |
| 真机（PC） | 2026-11-05 首次在吊架保护下行走（`wsq61`，PC 方案）：398 帧回放全部完成，无复位。**承重状态下的退出未验证**；踝俯仰力矩峰值 106 Nm，训练上限为 61 Nm。真实 PICO 上真机未测试 |
| 真机（机载） | 2026-11-13 首次机载吊架测试（`proteus_wamseq_62_full` 跑在 Nezha 上，`joint_bridge` 模式）。确认了机载方案能用机器人自带的 ROS 运行，并发现两个问题：未限速的 ROS 回调线程把一步策略拉长到 22–28 ms（改为在 CPU 0 上以 200 Hz 限速后：推理约 12 ms，平均每步约 17.5 ms，约 10% 的步略超 20 ms）；吊绳绷紧、承担大部分体重时，Hold 和挥手过程中骨盆以 4.3–4.7 Hz 晃动（对策是踝关节 kd ×1.5，**只在仿真中验证过**）。没有机载行走的运行记录；承重状态下的表现仍未验证 |
| `--robot` | `pm01`（别名 `pm_v2`、`engineai_pm01`）—— `sim2real/config/robots/__init__.py` |
| 关节数 | 24，SDK 顺序 `J00`–`J23`（`J12_WAIST_YAW`，`J23_HEAD_YAW` 是唯一的头部关节），按名映射 |
| Robot I/O | `--robot-io inline` → `PM01RobotIO`（`interface_protocol` 由机器人上已安装的协议构建）。话题 `/hardware/joint_state`、`/hardware/joint_command`、`/hardware/imu_info`、`/motion/motion_state`；Cyclone DDS，域 69（PC 网口 `eno1`） |
| Tracker 运行位置 | PC（`start_proteus.sh`、`start_pico.sh`），或运动控制计算机 `user@192.168.0.163`（`start_proteus_onboard.sh`，部署到 `~/proteus_pm01`）。这些启动脚本不使用 Orin `ubuntu@192.168.0.162` |
| 默认策略 | PC 启动脚本（`start_proteus.sh`、`start_pico.sh robot`）：`checkpoints/proteus/wsq61/engineai_pm01/policy.yaml`；`start_pico.sh policy-sim`：同一版本的 `policy-sim.yaml`。机载（`start_proteus_onboard.sh`、`deploy_onboard.sh`）：`checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.yaml`。均可用 `PROTEUS_POLICY_CONFIG` 覆盖 |
| 默认动作 | PC：`checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz`；PICO 保持数据集 `checkpoints/engineai_pm01/pico_hold`。机载：`wave` → `checkpoints/engineai_pm01/wave/motions/wave_right.npz`（默认），`walk` → 同一段行走片段 |

## 文件

`scripts/pm01/` —— 启动与准备（PC）：

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 从 `GMR-engineai` 拷贝并校验 `pm_v2.xml`（轴容差 1e-4），导出 ONNX 和 YAML 到 `checkpoints/proteus/<release>/engineai_pm01`，生成 `xrobot_to_pm01.json` / `bvh_lafan1_to_pm01.json`，生成 PICO 保持数据集，重定向四段 LAFAN1 片段。参数 `--sources-root`、`--proteus-root`、`--release`（默认 `wsq61`）、`--skip-motions`。 |
| `prepare_ros.sh` | 从机器人只读拷贝（rsync）**已安装的** `interface_protocol` msg/srv/action 定义到 `third_party/engineai_pm01/ros2`，并用 colcon 编译。参数：`user@host`（默认 `user@192.168.0.163`）。 |
| `env.sh` | 在 PC shell 中 source：ROS Humble + 编译好的消息、`ROS_DOMAIN_ID`、绑定网口的 Cyclone DDS、HF 离线变量。 |
| `check_robot.py` | 只读就绪检查（只订阅）。当 24 关节反馈、IMU 和 `joint_bridge` 均新鲜，存在兼容的 BEST_EFFORT 命令订阅者且没有其他命令发布者时返回 0，否则返回 2。`--duration`（默认 5 s）。 |
| `start_proteus.sh` | 真机文件回放：source `env.sh`，运行 `check_robot`（带 `--help` 时跳过），然后 `tracking --robot pm01 --robot-io inline --controller keyboard --keyboard-layout et1`。第一个可选参数为动作 NPZ。 |
| `start_pico.sh` | PICO 遥操作：`publisher`、`sim`（`--sim-dt 0.000625 --decimation 32`）、`policy-sim [ip]`、`robot [ip]`（先 source `env.sh` 并运行 `check_robot`）。 |
| `make_wave_motion.py` | 生成合成的挥手片段（站立、抬起一只手臂、用肘关节挥手、放下、站立），并检查软关节限位和手臂与身体的间距。参数 `--output`（默认 `checkpoints/engineai_pm01/wave`）、`--arm {right,left}`、`--duration`（10 s）、`--freq`（1 Hz）、`--ramp`（1.5 s）、`--hold`（1 s）、`--min-clearance`（0.05 m）。 |

`scripts/pm01/` —— 机载部署（运动控制计算机 Nezha：Ubuntu 22.04、x86_64、CPython 3.10、ROS Humble）：

| 文件 | 用途 |
| --- | --- |
| `deploy_onboard.sh` | 在 PC 上运行。根据 `uv.lock` 解析离线的 x86_64 / CPython 3.10 依赖包（仅 CPU 版 torch，缓存在 `outputs/pm01-onboard-env`），用 rsync 把代码、策略导出、行走和挥手动作、准备好的模型以及两个 Hugging Face 缓存拷到机器人上的 `<target>/{repo,site-packages,hf,logs}`，最后在机器人上做导入检查。不会启动、停止或重新配置机器人软件。`--build-only` 在连接机器人之前停止。 |
| `env_onboard.sh` | 在机器人上 source：使用机器人自带的 ROS Humble 和已安装的 `interface_protocol`，域 69，Cyclone DDS，把随附的依赖包加入 `PYTHONPATH`，HF 离线（`HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1`），ONNX Runtime / BLAS 各一个线程。 |
| `start_proteus_onboard.sh` | `start_proteus.sh` 的机载版本，需在交互式 SSH 终端中运行。第一个参数为 `wave`（默认）、`walk` 或动作 `.npz`，其余参数传给 `tracking`。要求策略开启 `reference_forcing`，在 `logs/` 下加锁，若已有其他 PM01 控制进程则拒绝启动，运行 `check_robot`，设置增益缩放和回调限速，用 `taskset` 绑定 Tracking 的 CPU，并录制到 `logs/run-<时间>/`（`check_robot.json`、`tracking.npz`）。 |
| `watch_sway.py` | 只读晃动监视器（订阅 IMU、关节状态和运动状态，不发布任何消息）。每 0.5 s 打印骨盆俯仰/横滚角、最近 1 s 俯仰/横滚角速度的振荡分量（deg/s rms）、速度最快的关节、两侧踝俯仰速度以及固件模式。`--seconds`（默认 120）。 |

`scripts/pm01/` —— 排查工具（实验性；不是部署路径）：

| 文件 | 用途 |
| --- | --- |
| `shadow_policy.py` | 用真实 PM01 反馈跑完整 Tracker/ONNX 链路 775 帧（15.5 s），不创建任何命令发布者，并按 20 ms 控制周期报告每步耗时（`policy_step_ms_p50`、`policy_step_ms_p99`、`policy_step_ms_max`、`steps_over_20ms`）。参数 `--policy-config`、`--motion`、`--output`。 |
| `trial_policy.py` | 限时 PC 客户端：只向 `/sim2real/pm01/request` 发布候选命令，需要机载 relay。`--walk-once` 需要新的 `--policy-signal` / `--play-signal` 文件。 |
| `policy_bridge.py` | 机载 relay（运动控制计算机）：试验期间唯一的硬件写入者，附带独立的监督进程，出错时锁存 classic 阻尼。主动命令使用 **RL_PARSER**。 |
| `bench_damping.py` | 机载踝关节阻尼测试；默认只读，`--execute` 允许 ≤1 s 的低踝阻尼。需要 root 权限，且电机驱动二进制哈希必须完全一致。试验阶段使用 RL_PARSER。 |
| `rehearse_trial.py` | 不需要 ROS：在 MuJoCo 中用 mocap weld 模拟吊架，从录制的悬挂姿态复演 relay 闸门。 |

其他相关文件：

- `sim2real/config/robots/pm01.py` + `sim2real/config/robots/specs/engineai_pm01.yaml`（经 `proteus_spec.py` 加载）。
- `sim2real/rl_policy/robot_io/pm01.py` —— `PM01_SDK_JOINT_NAMES`、`PM01Codec`（classic parser、零时间戳处理、踝/臂增益缩放）、`PM01RobotIO`（`PM01_HAL_HZ` / `PM01_HAL_CPU` → `spin_hz` / `spin_cpu`）。
- `sim2real/rl_policy/robot_io/engineai.py`（与 T800 共用）和 `sim2real/rl_policy/robot_io/ros2_named.py`（回调线程限速）。
- `sim2real/rl_policy/control_mode.py` + `sim2real/rl_policy/controllers/keyboard.py` —— `et1` 键盘布局。
- `sim2real/teleop/proteus_retarget.py` + `sim2real/teleop/configs/xrobot_to_pm01.json`。
- `scripts/proteus/`（`prepare_assets.py`、`retarget_lafan.py`、`evaluate.py`）、`scripts/build_xrobot_profile.py`、`scripts/export_proteus_onnx.py`、`scripts/make_hold_motion.py`。
- 测试：`tests/test_pm01.py`、`tests/test_pm01_wave.py`、`tests/test_pm01_launch.py`（需显式开启）、`tests/test_pm01_policy_bridge.py`、`tests/test_pm01_bench.py`、`tests/test_keyboard_layout.py`、`tests/test_proteus_transport.py`、`tests/test_pico_retarget_profiles.py`。
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
5. 机载资产（在 PC 上、运行 `deploy_onboard.sh` 之前准备；这次准备时没有 `wsq61` 权重）：
   ```bash
   python -m scripts.pm01.prepare --sources-root .. --release proteus_wamseq_62_full --skip-motions
   uv run --project venv/pico --no-sync python -m scripts.proteus.retarget_lafan --robot pm01 --duration 8 \
     --output checkpoints/engineai_pm01/motions --bvh /path/to/lafan1/walk1_subject1.bvh
   python -m scripts.pm01.make_wave_motion --output checkpoints/engineai_pm01/wave
   ```
   缺少 `checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.{yaml,onnx}`、
   `third_party/engineai_pm01/model/pm_v2_tracking.xml` 或两个 `manifest.json`（motions、wave）时，
   `deploy_onboard.sh` 会直接退出。机器人上不需要编译本仓库的 ROS 消息：`env_onboard.sh` 直接使用固件已安装的消息。

## 需调整项

| 参数 | 当前值 | 设置位置 |
| --- | --- | --- |
| 踝关节 parser | `start_proteus.sh`、`start_pico.sh robot` 和 `start_proteus_onboard.sh` 使用 `parallel_parser_type = 0`（classic，PD 在电机驱动内） | `robot_io/pm01.py` 中的 `PM01Codec`（T800 保持 `1`） |
| 踝关节增益 | kp 37.701、kd 2.4916（策略中的训练值）。classic 下作用于电机侧：关节侧俯仰约 4.09 倍、横滚约 1.11 倍 | `specs/engineai_pm01.yaml` / `policy.yaml` |
| 踝俯仰增益缩放 | `PM01_ANKLE_PITCH_GAIN_SCALE`：同时乘到下发的踝俯仰 kp **和** kd 上。机载默认 **0.306**（= 1.25 / 4.09，即使 4.09 的估计有偏差，关节侧刚度也保持在训练值的 0.8–2 倍内）；允许范围 0.1–1 | `start_proteus_onboard.sh` → `PM01Codec` |
| 手臂 kp 缩放 | `PM01_ARM_KP_SCALE`：乘到肩/肘 kp（`J13`–`J22`）上。机载默认 **0.7**（仿真中把延迟引起的全身振荡起点从观测到目标约 20 ms 延迟推到约 30 ms；机载延迟约 15–18 ms）；允许范围 0.5–1 | `start_proteus_onboard.sh` → `PM01Codec` |
| 踝关节 kd 缩放 | `PM01_ANKLE_KD_SCALE`：在俯仰缩放之后乘到踝俯仰和横滚 kd 上。机载默认 **1.5**（抑制双脚轻载时的晃动；仿真中 80% 支撑下挥手晃动从 8.8 降到 0.4 deg/s）；允许范围 1–3 | `start_proteus_onboard.sh` → `PM01Codec` |
| 增益缩放作用范围 | 只作用于策略命令，阻尼不变。PC 启动脚本不设置这些变量，所以 PC 方案下发的是训练增益（未设置即为 1） | `PM01Codec.write_command` |
| ROS 回调限速 | `PM01_HAL_HZ` → `spin_hz`，`PM01_HAL_CPU` → `spin_cpu`。机载默认 **CPU 0 上 200 Hz**；设为空可分别关闭。PC 上不设置（不限速）。限速后每个周期对每个订阅和定时器最多执行一次回调，因此 500 Hz 的发布定时器每个周期也最多运行一次 | `start_proteus_onboard.sh` → `PM01RobotIO` → `ros2_named.py` |
| Tracker CPU | `PM01_TRACKER_CPU`（默认 `3`；Nezha 以 `isolcpus=1-3` 启动，厂商实时线程在 CPU 1–2 上；设为空则不用 `taskset`） | `start_proteus_onboard.sh` |
| 键盘布局 | `--keyboard-layout et1`（`start_proteus.sh` 和 `start_proteus_onboard.sh` 都使用）：**a** 整形、再按 **a** 进入 Hold、**s** 播放/暂停、**空格**或 **x** 停止。其他机器人仍用 `default`（i、]、空格、o） | `base_policy.py` 中的 `BasePolicyArgs.keyboard_layout`；`control_mode.py` |
| 零时间戳 | 接受；新鲜度用本地单调接收时间。出现过非零时间戳后再收到零会被拒绝 | `PM01Codec._new_stamp` |
| IMU 话题 / 外参 | `/hardware/imu_info`，`imu_to_pelvis_wxyz: [1, 0, 0, 0]`（IMU 位置与骨盆对齐） | `policy.yaml` 中的 `robot_io_options` |
| 发布频率 / 超时 | `publish_hz: 500`，`timeout_s: 0.25`；模式心跳 1 s | `robot_io_options`；`EngineAICodec._fresh` |
| 停止阻尼 / 步长限制 | 阻尼 2.0，每次发布 0.2 rad | `ros2_named.py` 中 `NamedJointCodec` 默认值 |
| 关节限位 | 训练限位；无需厂商限位覆盖 | `prepare.py` → `validation.json` |
| 模型轴容差 | `axis_atol=1e-4`（spec 轴四舍五入到 4 位小数；实测最大 8.76e-5） | `prepare.py` |
| 仿真步长 | **0.000625 s**，decimation 32 | `start_pico.sh sim`；`scripts/proteus/evaluate.py` 对 PM01 的默认值 |
| DDS / 网络（PC） | `PM01_NETWORK_INTERFACE`（`eno1`）、`PM01_ROS_DOMAIN_ID`（69）、`PM01_ROS_SETUP`；已有的 `CYCLONEDDS_URI` 优先 | `env.sh` |
| DDS / 路径（机载） | `PM01_ROS_OVERLAY`（默认 `/apps/engineai_robotics/_install/share/interface_protocol/local_setup.bash`）、`PM01_ROS_DOMAIN_ID`（69）、`PM01_CYCLONEDDS_URI`（不设置则保持厂商默认）、`PROTEUS_ROOT` | `env_onboard.sh` |
| 部署目标 | `PM01_HOST`（`user@192.168.0.163`）、`PM01_TARGET`（`proteus_pm01`，相对远端 home 目录）、`PM01_RELEASE`（`proteus_wamseq_62_full`）、`UV`、`PM01_ENV_CACHE` | `deploy_onboard.sh` |
| 远端消息路径 | `PM01_REMOTE_PROTOCOL`（默认 `/apps/engineai_robotics/_install/share/interface_protocol`） | `prepare_ros.sh` |
| 策略配置 | `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` | 启动脚本 |

## 使用

离线评测与测试：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
source scripts/pm01/env.sh   # ROS 回环测试会自行隔离到 localhost、域 171
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py tests/test_pm01_wave.py -q
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

### PC 上的真机运行

```bash
# 终端 1
bash scripts/pm01/start_proteus.sh checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

1. 双脚悬空，10 s 内不按任何键：机器人应和 `passive` 一样安静。
2. **a** 在 10 s 内整形到 walk 第 0 帧（双臂抬到接近水平）。
3. 降低吊架，直到双脚承担体重（吊绳松弛，但仍能接住摔倒），然后再按 **a** 进入策略 Hold。
4. **s** 播放。**空格**或 **x** 进入阻尼；**Ctrl+C** 退出。

### 在运动控制计算机上机载运行

```bash
# 在 PC 上，只需一次：安装 SSH 密钥，然后部署（或用 --build-only 只解析依赖包、不连接机器人）
ssh-copy-id user@192.168.0.163
bash scripts/pm01/deploy_onboard.sh
# 在 passive 或 pd_stand 下试运行（只读，无命令发布者）；检查 policy_step_ms_p99 和 steps_over_20ms
ssh -t user@192.168.0.163 'source proteus_pm01/repo/scripts/pm01/env_onboard.sh && cd proteus_pm01/repo && \
  python3 -m scripts.pm01.shadow_policy --policy-config checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.yaml \
  --motion checkpoints/engineai_pm01/wave/motions/wave_right.npz --output ../logs/shadow-wave.npz'
# 进入 joint_bridge，双脚悬空在吊架上：运行挥手（或 `walk`，或某个动作 .npz）
ssh -t user@192.168.0.163 proteus_pm01/repo/scripts/pm01/start_proteus_onboard.sh wave
# 第二个终端：只读的骨盆监视器
ssh user@192.168.0.163 'source proteus_pm01/repo/scripts/pm01/env_onboard.sh && cd proteus_pm01/repo && python3 -m scripts.pm01.watch_sway'
```

1. 只有当试运行的 `policy_step_ms_p99` 明显低于 20、`steps_over_20ms` 接近 0 时才继续；否则用
   `PM01_TRACKER_CPU` 把 Tracking 换到别的核上。启动脚本会打印所用的增益缩放和回调限速，`check_robot`
   不通过时直接退出。
2. 10 s 内不按任何键（应和 `passive` 一样安静），然后按 **a** 在 10 s 内整形到第一帧。
3. 降低吊架，直到双脚平放在地面并承担体重（吊绳松弛，但仍能接住摔倒），再按 **a** 进入策略 Hold。保持约 10 s，
   同时看 `watch_sway`：如果骨盆晃动，按**空格**。
4. **s** 播放或暂停。**空格**或 **x** 停止（阻尼）；**Ctrl+C** 退出。行走（`walk`）会向前移动约 3 m：
   吊架必须跟随，路径必须清空。
5. 日志：`~/proteus_pm01/logs/run-*/`（`check_robot.json`、`tracking.npz`）。

### PICO 遥操作（发布端在同一台 PC）

```bash
# 终端 1：XRoboToolkit PC Service + 已标定的身体/腿部追踪器
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# 终端 2：真机策略（MuJoCo 则用 `sim` + `policy-sim 127.0.0.1`）
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

按键：**A** 整形到中立姿态；松开后 **A+B** 进入 Hold；**X** 暂停/恢复；单独按 **B** 进入阻尼。
端口 28701（动作）/ 5592（按键）；每套端口只对应一台机器人。

退出（PC 或机载）：按**空格**或 **x**（PICO：**B**），保持吊架承载，Ctrl+C（关闭时重复发送阻尼），
然后 **LB + RB** 回到 `passive`。

### 排查工具

需要与测试时完全一致的增益、固件哈希和悬挂条件；使用前先读教程：

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
- 2026-11-13 机载测试：不限速时，ROS 回调线程在策略所在的 CPU 上不停解码固件的关节/IMU 数据流，在
  `joint_bridge` 下把一步拉长到 22–28 ms。在 `joint_bridge` 之外做试运行发现不了这个问题（那时没有数据发布）。
  务必保留 `PM01_HAL_HZ` / `PM01_HAL_CPU`；即使限速，仍有约 10% 的步略超 20 ms（GIL 竞争）。
- 2026-11-13 机载测试：吊绳绷紧、承担大部分体重时（膝关节力矩只有 5–7 Nm），Hold 和挥手中骨盆以
  4.3–4.7 Hz 晃动（俯仰角速度阵发性达到 15–40 deg/s rms）。进入 Hold 前要让双脚承担体重，绝不要在双脚悬空时
  进入 Hold（策略会把它当成摔倒）。踝关节 kd ×1.5 的修正和三项增益缩放都来自仿真，classic parser 的 4.09
  倍系数也只是估计值。
- 零时间戳反馈无法区分重发的旧值与新测量值。`check_robot` 的 `ready: true` 不能证明电机已使能
  （曾出现报告 `joint_bridge` 时 24 个电机禁用标志全部置位的情况）。
- 进程退出或断电需要固件侧保护（PC 和机载都一样）；bridge runner 是否有命令超时尚未确认。

## 详细文档

- English：[`docs/tutorials/proteus-pm01.md`](../../docs/tutorials/proteus-pm01.md)
- 中文：[`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md)
- 厂商：[engineai_ros2_workspace](https://github.com/engineai-robotics/engineai_ros2_workspace)、[PM01 并联踝说明](https://en.engineai.com.cn/open/docs/pm01/04/source-code-analysis)
