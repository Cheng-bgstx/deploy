# 部署脚本

[English](README.md)

本目录存放把策略部署到真机所用的启动脚本和工具。所有机器人共用同一套运行时（`sim2real/`）和同一套
部署流程，差别只在机器人 I/O、部署覆盖配置（deployment overlay）和安全细节上。本页介绍通用流程，以及
如何接入一台新机器人。每个机器人目录下都有自己的 README（`./<robot>/README_zh.md`），写明该机器人的具体命令。

术语以 [`GLOSSARY.md`](../GLOSSARY.md) 为准：**Tracker**（跟踪策略）、**Reference**（参考）、
**Motion**（动作文件）、**File Playback**（文件回放）、**Planner**（规划器）、**Shaping**（塑形）、
**Hold**（冻结保持）、**Go**、**Final Hold**（末帧保持）、**E-stop**（急停）、
**Controller**（操作输入设备）、**Bridge**（桥接进程）。

> **安全须知。** 每一次真机运行都要当作危险操作。在 Hold 稳定之前，机器人始终挂在龙门架或吊架上；
> 硬件急停按钮由第二个人拿在手里。下表中硬件状态为"未测试"的机器人，没有厂商安全流程和现场监护时
> 绝不能运行。软件 E-stop、被杀掉的进程或断开的网络都**不是**硬件急停。

## 目录结构

### 机器人目录

| 目录 | 机器人 | `--robot` 名称 | 内容 |
| --- | --- | --- | --- |
| [`g1/`](./g1/README_zh.md) | Unitree G1 | `g1` | 外置 PC 或机载 Orin 上运行 Proteus：SDK2 Bridge、验证脚本、机载部署 |
| [`et1/`](./et1/README_zh.md) | Galbot ET1 | `galbot_et1`、`et1` | 机载 HDEP Bridge、键盘操作端、Galbot 跟踪策略导入 |
| [`pm01/`](./pm01/README_zh.md) | EngineAI PM01 | `pm01`、`pm_v2`、`engineai_pm01` | ROS 2 启动脚本（PC 与板载）、只读检查、晃动监视、受限试验中继工具 |
| [`q1/`](./q1/README_zh.md) | Primebot Q1 | `q1`、`primebot_q1` | 覆盖配置生成、离线验证、关节调试、IMU 检查 |
| [`a3/`](./a3/README_zh.md) | AgiBot A3 | `a3`、`agibot_a3` | 资产准备、文件回放与 PICO 启动脚本 |
| [`t800/`](./t800/README_zh.md) | EngineAI T800 | `t800`、`engineai_t800` | 资产准备、文件回放与 PICO 启动脚本 |
| [`x2/`](./x2/README_zh.md) | AgiBot 灵犀 X2 | `x2`、`lingxi_x2`、`agibot_x2` | 板载 AimDK Bridge（真机实际使用的路线）、加阻尼 overlay、探测工具；PC/ROS 2 启动脚本 |
| [`oli/`](./oli/README_zh.md) | LimX OLI | `oli`、`limx_oli_edu`、`hu_d04_01` | 资产、动作转换与重定向、sim2sim 渲染 |

名称来自 `sim2real/config/robots/__init__.py`。Unitree H2（`h2`、`unitree_h2`）有机器人配置和 PICO
重定向配置，但没有机器人 I/O 后端，也没有对应目录。

### 其他目录

| 目录 | 内容 |
| --- | --- |
| [`dh116s/`](./dh116s/README_zh.md) | 由 PICO 扳机握持控制的 DH116S 灵巧手（`control.py`、`keyboard_control.py`） |
| `scalecar/` | G1 上车任务工具：构建车辆场景、转换 Scalecar 动作、比对 Reference Token。G1 启动脚本在 `g1/scalecar/` |
| `proteus/` | 多机器人共用的 Proteus 工具：`prepare_assets.py`（导入并校验 MJCF）、`retarget_lafan.py`、`build_retarget_profiles.py`、`evaluate.py`（在动作库上跑 sim2sim） |
| `setup/` | G1 网络辅助脚本（热点、笔记本网关）和 XRoboToolkit pybind 安装 |
| `tracking_experiment/` | 跨代码库跟踪评测与排行榜数据 |
| `publishers/`、`vis/` | 关节状态 / 动捕发布器和可视化工具 |

### 共用工具（`scripts/*.py`）

| 脚本 | 用途 |
| --- | --- |
| `export_proteus_onnx.py` | 把已发布的 Proteus actor 包装成单个机器人的策略：`policy.onnx`、`policy.yaml`、`policy-sim.yaml`、`comparison.json` |
| `build_proteus_contract.py` | 为发布版之外的机器人生成 Proteus contract，依据其 MJCF 和一个已发布的模板机器人 |
| `prepare_proteus_test_motions.py` | 把 Proteus ScaleBridge 测试片段（G1、ET1）转换为 Motion 数据集 |
| `lafan_tracking_suite.py` | 生成机器人专用的 LAFAN Reference，可选跑 sim2sim |
| `make_hold_motion.py` | 生成 PICO 实时流之前塑形用的中性起始 Motion |
| `summarize_proteus_sim2sim.py` | 汇总渲染 Proteus sim2sim 视频时记录的轨迹 |
| `build_xrobot_profile.py` | 把标定好的 SMPL-X GMR 配置转换到 PICO/XRobot 身体坐标系 |
| `export_scalecar_tracker_onnx.py`、`export_scalebfm_onnx.py` | Scalecar Tracker 与 ScaleBFM 的导出脚本 |
| `test_policy_inference.py` | 推理后端基准测试 |
| `record_motion.py`、`view_motion.py`、`process_qpos_data.py` | 录制、查看、平滑 Motion |

## 运行时架构

```text
 参考源 (Reference Source)        Tracker (50 Hz)                 机器人 I/O                     机器人
 ------------------------        ---------------                 ----------                     ------
 File Playback  --motion-backend npz ---+
 Planner        --motion-backend planner+-> sim2real/rl_policy/ -> --robot-io inline ---------> 厂商 SDK / ROS 2
 PICO 实时流    --motion-backend zmq ---+   tracking.py            (后端在 Tracker 进程内)
   (sim2real/teleop/pico_retarget_pub.py)   策略 ONNX              --robot-io zmq ---> Bridge -> 厂商 SDK
                                            阶段状态机             (127.0.0.1 上的 ZMQ) BridgeSafety
                                                  ^
                                      Controller: keyboard / joystick / bridge_joystick / pico
```

- **Tracker。** `sim2real/rl_policy/tracking.py`（基于 `base_policy.py`）每 `1 / --rl-rate` 秒（50 Hz）
  执行一个控制步：读取状态、更新 Reference、构造观测（`sim2real/rl_policy/observations/`）、运行 ONNX
  策略（`--inference-backend onnx-cpu`、`onnx-gpu` 或 `tensorrt`）、对关节目标做后处理
  （`sim2real/rl_policy/utils/joint_targets.py`），再按 YAML 中的 PD 增益下发位置目标。
- **inline 机器人 I/O**（`--robot-io inline`）。`sim2real/rl_policy/robot_io/factory.py` 在 Tracker 进程内
  创建机器人后端。A3、Q1、T800、PM01（ROS 2，`ros2_named.py` / `engineai.py`）、X2（`x2.py`，AimDK
  ROS 2）和 OLI（`oli.py`，LimX SDK）都走这条路。这些后端有自己的发布循环和指令看门狗，反馈或指令
  过期时退回阻尼。
- **ZMQ 机器人 I/O**（`--robot-io zmq`）。Tracker 通过本地 ZMQ（`robot_io/zmq.py`）与独立的 Bridge 进程
  交换 LowState/LowCmd。G1 用 `scripts/g1/real_bridge.py`（SDK2），ET1 用 `scripts/et1/hdep_bridge.py`
  （HDEP；ET1 没有 inline 模式）。X2 的板载路线同样如此，使用 `scripts/x2/aimdk_bridge.py`。它们都用 `robot_io/bridge_safety.py`（BridgeSafety）仲裁指令：启动时不发
  任何电机指令；接管需要 Tracker 就绪并有一次新的实体 A 键按下；Select 和故障在 Bridge 内独立锁存停止，
  不依赖 Tracker。用 `sim2real/sim_env/base_sim.py` 做 sim2sim 时同样使用 `--robot-io zmq`。
- **运行位置。** Tracker 跑在能访问机器人 SDK 的那台机器上：机器人局域网中的外置主机（Host），或机器人
  自带的计算机。G1 启动脚本两种都支持（`g1/start_proteus_pc.sh`、`g1/onboard.sh`；只有 PC 路径经过真机测试）；ET1 只在机载运行。像启动脚本
  那样把 Tracker 和 Bridge 绑在不同核上，并限制 ONNX 线程数（`SIM2REAL_ORT_NUM_THREADS`）。

## 部署流程

### 1. 适配并导出策略

Proteus actor 只做包装，不重新训练。`export_proteus_onnx.py` 读取
`<proteus-root>/ScaleTrack/checkpoints/<release>`（或 `--checkpoint-dir`）下的发布版，输出到
`checkpoints/proteus/<release>/<contract>/`：

```bash
uv run python scripts/export_proteus_onnx.py --release wsq61 --contract engineai_pm01 --robot pm01
# 发布版之外的机器人：先生成 contract，再让导出脚本读取它。
uv run python scripts/build_proteus_contract.py --robot et1 --name galbot_et1 --template unitree_r1
uv run python scripts/export_proteus_onnx.py --release proteus_60 --contract galbot_et1 --robot et1 \
  --contract-dir checkpoints/proteus/contracts
```

| 文件 | 用途 |
| --- | --- |
| `policy.yaml` | 真机用。`proteus.reference_forcing: true`：机器人没有根位置，根节点取 Reference 骨盆位置。需要改动时重新生成，不要手改 |
| `policy-sim.yaml` | 仅限仿真。使用实测世界坐标根位置（`reference_forcing: false`）。没有定位的真机上不要用 |
| `policy-pc.yaml` | 真机部署覆盖配置，由各机器人的准备脚本生成（第 2 步），不是导出产物 |
| `comparison.json` | 包装模型与原发布 actor 在随机输入上的对比；误差超过 `1e-6` 时导出失败 |

多数机器人目录把这一步封装进准备脚本（`python -m scripts.a3.prepare`、`scripts.t800.prepare`、
`scripts.pm01.prepare`、`scripts.q1.prepare`、`scripts/x2/export_policy.py`、`scripts/oli/prepare_assets.py`），
同时准备 MJCF 和 Motion。非 Proteus 系列的策略用 `.agents/skills/adapt-policy-to-sim2real` 技能适配。

### 2. 部署覆盖配置

导出的 `policy.yaml` 保持不动，真机相关的修改写进单独的覆盖配置
（`g1/prepare_proteus_deployment.py --robot g1|galbot_et1`、`python -m scripts.q1.prepare_deployment`、
`et1/prepare_galbot_track_policy.py`）。运行时读取的键：

| 键 | 代码中的实际行为 |
| --- | --- |
| `joint_kp`、`joint_kd` | 每个阶段的每条指令都附带的 PD 增益（`utils/command_sender.py`）。在覆盖配置里修改，例如 G1 腰部 roll/pitch 改为 kp 100、kd 2.5 |
| `soft_joint_pos_limits` | 关节名（正则）→ `[lower, upper]`。裁剪 Tracker 目标，必须覆盖所有策略关节。零宽区间 `[d, d]` 把关节锁在 `d`（ET1 头部） |
| `max_target_velocity_rad_s` | Tracker 目标的变化率限制，每步最多 `max × 控制周期`；每次进入 Hold 都从实测姿态重新开始 |
| `shaping_target` | `default_joint_pos`（默认）或 `motion_first_frame`：Shaping 插值的目标 |
| `shaping_joint_pos_limits` | 只裁剪策略接管之前的姿态目标（Shaping、`reference_entry_s`），从不裁剪 Tracker 目标 |
| `reference_entry_s` | 仅文件回放：请求 Hold 时先在这段时间内插值到 Motion 首帧，再进入 Hold（G1 和 ET1 覆盖配置为 2 s） |
| `policy_entry_max_joint_error_rad`、`policy_entry_max_joint_velocity_rad_s` | 可选的 Hold 门槛：Shaping 未完成或有关节超出误差（设置了速度键时还要求已静止）时拒绝进入 Hold |
| `shaping_gravity_compensation` | Shaping 期间可选的前馈力矩（Q1，实验性） |
| `robot_io_options` | 传给 inline 后端的参数：话题、`publish_hz`、`timeout_s`、IMU 外参 `imu_to_pelvis_wxyz`，以及 `ros2_named.py` 系机器人的 `policy_limit_mode` |

Shaping 目标的裁剪优先用 `shaping_joint_pos_limits`，否则用 `soft_joint_pos_limits`；两者都没有时，直接使用
未裁剪的 Motion 首帧。`policy_limit_mode`（ROS 2 按名关节后端）决定超出关节范围的 Tracker 目标如何处理：
`clip`（默认）发送边界值，`pass` 原样发送，`feedforward` 发送边界值并附加 `kp × 超出量` 的力矩，接近
限位时逐渐减为零。该后端总是裁剪 Shaping 目标。锁定关节有三种方式：contract 中不分配 slot
（`build_proteus_contract.py --lock`）、零宽软限位、后端的固定电机（`robot_io/a3.py` 中的 A3 头部 pitch）。
导出脚本已经为 OLI、X2、A3、T800、PM01 和 Q1 设置 `shaping_target: motion_first_frame` 和默认
`robot_io_options`。

### 3. sim2sim 演练

上真机之前，在 MuJoCo 中用同样的 YAML、Motion 和 Controller 流程跑一遍。

```bash
# 在动作库上评测策略（实测根位置）；未跑完、出现非有限值或身体误差 > 0.5 m 即失败
uv run --no-sync python -m scripts.proteus.evaluate --robot a3 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy-sim.yaml \
  --dataset checkpoints/agibot_a3/motions --output outputs/a3/eval
# 在集成仿真器中跑单个 Motion（带可视化；批量运行加 --headless）
uv run sim2real/sim_env/integrated_sim2sim.py --robot g1 --policy-config <policy.yaml> --motion-path <motion>
# 用真实 Tracker、阶段状态机和 BridgeSafety 做部署演练（G1、ET1）
uv run python scripts/g1/validate_proteus_deployment.py --robot g1 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy-pc.yaml --npz-playback-limit-s 3
```

`validate_proteus_deployment.py` 在 Shaping 期间固定根节点（理想龙门架），随后放下，再按 Controller 顺序
操作：过早的 Start 和过早的第二次 A 必须被忽略，然后依次 Shaping、Hold、播放、暂停、继续、Select。
结果写入 `<output>.json` 和 `<output>.npz`；`passed` 要求状态全部有限、Bridge 无故障、阶段顺序符合预期、
放下后根高度始终高于站立高度的 75% 且保持直立。`--scenario walk-hold` 还检查前进距离、横向误差和
Final Hold 中的漂移。Q1 有单独的离线演练 `scripts/q1/validate_deployment.py`。通过只说明仿真结果：
它不模拟龙门架吊索、机器人通信超时和硬件急停。PICO 实时流可以用各机器人的 `start_pico.sh sim` +
`start_pico.sh policy-sim` 对接 `base_sim.py` 演练。

### 4. 真机上电前检查

- [ ] `checkpoints/` 和 `third_party/` 已就位（运行时产物，不在 git 中；见
      [`docs/artifacts.md`](../docs/artifacts.md)）。推理用根项目环境，遥操作用 `venv/pico`。
- [ ] shell 中已导出 `HF_HUB_OFFLINE=1` 和 `HF_HUB_DISABLE_TELEMETRY=1`，并写入机器人用户的 `~/.bashrc`。
      启动脚本会导出这两个变量；手动执行的命令也要带上。
- [ ] 网络：运行 Tracker 的机器能访问机器人局域网，网卡设置正确（`G1_INTERFACE`、`--robot-interface`、
      `--robot-ip`）。见 [`docs/getting-started/network-configuration.md`](../docs/getting-started/network-configuration.md)。
- [ ] 已按机器人 README 加载厂商 SDK 或 ROS 2 环境（例如 X2 需要 ROS 2 Humble 和 AimDK，PM01 需要
      `scripts/pm01/env.sh`）。
- [ ] 只有一个电机指令写入方：厂商的运动控制栈已停止或已交出控制权。G1 和 ET1 的启动脚本检测到正在运行的
      控制器时会拒绝启动。
- [ ] 只读反馈检查已通过：G1 和 ET1 用 `PROTEUS_READ_ONLY=1`，PM01 用 `scripts/pm01/check_robot.py`，
      X2 用 `scripts/x2/probe.sh`（板载）或 `scripts/x2/check_feedback.py`（PC），Q1 用 `scripts/q1/inspect_imu.py`。核对关节顺序和正负方向，
      确认 IMU 测得的重力方向正确。
- [ ] 机器人挂在龙门架或吊架上，双脚离地；操作者清楚何时放下。
- [ ] 硬件急停已测试，由第二个人拿在手里。
- [ ] Tracker 所在机器上没有在跑基准测试、仿真或其他重负载进程。

### 5. 真机操作阶段

```text
 启动 --> E-stop (zero) --塑形--> Shaping --保持--> Hold --Go--> 播放中 --结束--> Final Hold
              ^                     (10 s)          (Reference 冻结)          (末帧)
              +------------ 任意阶段均可 E-stop；从 Shaping 重新开始 ------------+
```

| 阶段 | 机器人行为 | 进入条件 | 离开方式 |
| --- | --- | --- | --- |
| E-stop（`zero`） | Tracker 目标跟随实测关节角。ROS 2、X2 和 OLI 后端只发阻尼（kp = 0）。BridgeSafety：Select 保留上一条指令的 kp/kd，目标为实测位置；故障则锁存为纯阻尼 | 启动时；E-stop 输入；Bridge 停止或故障 | Shaping |
| Shaping（`init`） | 从进入时的姿态在 10 s 内插值到 `shaping_target`，之后保持 | Shaping 输入 | Hold、E-stop，或再次 Shaping（重新开始） |
| Hold（`policy`，暂停） | Tracker 接管；历史清零；Reference 冻结在 Motion 首帧（使用 Planner 时持续到第一个规划块到达） | Hold 输入（手柄：仅在 Shaping 完成后） | Go、E-stop、Shaping |
| Go（播放中） | Reference 前进；文件回放时每次 Go 输入切换播放/暂停；Planner 只接受第一次 Go | Hold 中的 Go 输入 | Motion 结束、E-stop、Shaping |
| Final Hold | Reference 停在末帧（或 `--npz-playback-limit-s` 处），Go 无法延长。Motion 必须以站立结束 | 文件回放结束 | E-stop、Shaping（新一轮） |

Controller 按键映射（来自 `sim2real/rl_policy/control_mode.py`、`controllers/` 和 `tracking.py`）：

| `--controller` | Shaping | Hold | Go | E-stop |
| --- | --- | --- | --- | --- |
| `keyboard`（Tracker 所在终端） | `i` | `]` | `space` | `o` |
| `keyboard` 加 `--keyboard-layout et1`（PM01 启动脚本） | `a` | 再按 `a` | `s` | 空格或 `x` |
| `joystick`，Unitree HG 系（G1） | `A` | Shaping 完成后再按 `A` | `Start` | `Select` |
| `bridge_joystick`（手柄经 Bridge 转发：G1 机载、ET1） | `A` | Shaping 完成后再按 `A` | `Start` | `Select`（ET1 Bridge 另支持 `L1` + `Down`） |
| `pico`（右手柄） | 单按 `A` | `A` + `B` 同时按（先松开） | 发布端按 `X`（暂停/继续实时流） | 单按 `B` |
| ET1 / X2 板载 `scripts/<robot>/operator.sh`（按键发给 Bridge） | `a` | 再按 `a` | `s` | `space` 或 `x` |

说明：

- `Select` 按边沿触发，同时按下其他键时以它为准。在使用 BridgeSafety 的机器人上，它在 Bridge 内部锁存，
  即使 Tracker 卡住也有效。任何停止之后都要按 Shaping → Hold → Go 重新开始。
- `keyboard` 和 `pico` 不会等 Shaping 结束才允许进入 Hold，除非覆盖配置设置了
  `policy_entry_max_joint_error_rad`。请等到日志出现 `Shaping interpolation complete`。
- 使用 PICO 实时流时，Tracker 从不暂停 Reference：它跟随发布端发送的内容，在 `X` 启动实时流之前发布端
  发送中性姿态。实时流中断时保持最后一个 Reference。
- Unitree `go` DDS 系（旧布局）映射为 `A` / `R1` / `B` / `R2`；本仓库没有机器人使用。
- G1 或 ET1 挂在龙门架上时：Shaping，然后 Hold，等待 `Hold ready`，放下龙门架，再 Go。

### 6. 记录、回看与关闭

- `--record --record-output <file>.npz` 在 Tracker 退出时保存每个控制步。启动脚本把它和日志放在同一个运行
  目录：`outputs/proteus-pc/<time>/`（G1：`tracking.npz`、`tracker.log`、`bridge.log`、`bridge-state.json`、
  `lowstate-monitor.log`）、`<root>/logs/run-<time>/`（ET1）、`outputs/real_runs/<time>/`（Scalecar）。
  不指定 `--record-output` 时文件为 `./policy_tracking_record_<policy>_<time>.npz`。
- 读取记录：

  ```python
  import numpy as np
  rec = np.load("tracking.npz", allow_pickle=True)["data"].item()  # schema policy_tracking_record_v1
  rec["robot"]["joint_pos"], rec["policy"]["q_target"], rec["runtime"]["control_mode"], rec["runtime"]["motion_t"]
  ```

- 关闭时先在机器人有支撑的情况下 E-stop，再按 Ctrl+C 或发送 SIGTERM。Tracker 会保存记录并关闭 Controller
  和机器人 I/O；G1 和 ET1 的 Bridge 正常退出时会发送阻尼。G1 机载运行用 `scripts/g1/onboard.sh stop` 和
  `scripts/g1/onboard.sh logs`。重启机器人之前先停掉启动脚本。

## 接入新机器人

1. **机器人配置。** 新增 `sim2real/config/robots/<robot>.py`，定义 `RobotCfg`（关节和刚体名、限位、电枢、
   `safe_joint_kp`/`kd`、`mjcf_path`、`default_qpos`）。Proteus 机器人可以用 `from_proteus_spec()` 读取
   `sim2real/config/robots/specs/<contract>.yaml` 中冻结的规格。在 `sim2real/config/robots/__init__.py` 的
   `_ROBOT_CFGS` 中注册名称和别名。
2. **MJCF 与资产。** 用 `scripts/proteus/prepare_assets.py` 把厂商模型导入 `third_party/<contract>/model`，
   并解决它报告的每一处关节轴不一致。`third_party/` 和 `checkpoints/` 是运行时产物，不进 git：转换后的
   产物按 `AGENTS.md` 的说明用 `rclone` 上传到共享云盘。
3. **机器人 I/O。** 新增 `sim2real/rl_policy/robot_io/<robot>.py`，并在 `robot_io/factory.py` 中分发。
   优先复用现有代码：
   - `ros2_named.py`（`NamedRosRobotIO`）：按名称寻址的 ROS 2 关节分组（A3、Q1）；
   - `engineai.py`：EngineAI 的 `interface_protocol`，带 `joint_bridge` 模式握手（T800、PM01）；
   - `--robot-io zmq` 加使用 `BridgeSafety` 的 Bridge：SDK 无法放进 Tracker 进程，或需要在 Tracker 卡死时
     仍然有效的停止（G1、ET1）。
   每个后端都必须按名称映射关节、拒绝过期反馈、指令超时时发送阻尼，并通过 `set_control_mode` 接收
   `zero` / `init` / `policy` 模式。
4. **Proteus contract 与导出。** 使用该机器人已发布的 contract，或用
   `build_proteus_contract.py --template <已发布机器人>` 生成，然后导出（第 1 步）。如果需要
   `shaping_target` 或默认 `robot_io_options`，把机器人加入 `export_proteus_onnx.py` 中 `policy_yaml()` 的名称集合。
5. **部署覆盖配置。** 写一个准备脚本，从导出的 `policy.yaml` 生成 `policy-pc.yaml`（或在文档中写明各个键）；
   不要原地修改导出文件。
6. **PICO 重定向配置。** 新增 `sim2real/teleop/configs/xrobot_to_<robot>.json`（用
   `scripts/proteus/build_retarget_profiles.py` 或 `scripts/build_xrobot_profile.py` 生成），并把机器人加入
   `sim2real/teleop/pico_retarget_pub.py` 中的 GMR 目标映射和机器人集合。用
   `python -m scripts.make_hold_motion --robot <robot> --output <dir>` 生成起始 Motion。
7. **测试。** 参照 `tests/test_proteus_transport.py`（反馈映射、限位、看门狗、阻尼、factory 参数）、
   `tests/test_pico_retarget_profiles.py`（配置有效性、FK 和 ZMQ 缓冲），使用 Bridge 时参照
   `tests/test_g1_bridge_safety.py`（接管、Select、超时）。把机器人加入这些测试的参数列表；新测试文件要在
   `.gitignore` 中加白名单（`/tests/*` 默认被忽略）。
8. **机器人目录。** 新建 `scripts/<robot>/`，包含导出 HF 环境变量的 `start_proteus.sh`（支持 PICO 时还有
   `start_pico.sh`），以及 `README.md` 和 `README_zh.md`，章节为 **Status、Files、Setup、Adjustments、
   Usage、Safety notes**（中文版对应：状态、文件、准备、调整项、使用、安全须知）。把机器人加入本页的表格。
9. **文档。** 中英文保持同步：本页与 `README.md`、机器人 README 中英文对、顶层 `README.md` / `README_zh.md`，
   以及 `docs/` 与 `docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/`。
10. **验证。** 运行时文件用 `uv run python -m py_compile <files>`，遥操作文件用
    `uv run --project venv/pico python -m py_compile <files>`；然后跑测试、sim2sim 演练、真机只读检查，
    最后才做有龙门架保护的 Shaping 试验。

## 机器人状态

| 机器人 | 硬件状态 | 默认策略（启动脚本） | README |
| --- | --- | --- | --- |
| Unitree G1 | 外置 PC 部署（`start_proteus_pc.sh`）已真机测试。机载 Orin：只做过只读计时运行，尚未真机测试 | `checkpoints/proteus/proteus_wamseq_62/unitree_g1`（`policy-pc.yaml`） | [g1](./g1/README_zh.md) |
| Galbot ET1 | 机载真机测试：2026-10-29 和 2026-11-11 的运行（proteus_60；WAM-62 已能站立并开始行走） | `checkpoints/proteus/proteus_60/galbot_et1`（`policy-pc.yaml`）；可用 `PROTEUS_POLICY_DIR` 选 `proteus_wamseq_62/galbot_et1` | [et1](./et1/README_zh.md) |
| EngineAI PM01 | 2026-11-05 完成首次龙门架保护下的真机行走；2026-11-13 板载吊架运行 | `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml` | [pm01](./pm01/README_zh.md) |
| Primebot Q1 | 只读反馈已测试；首次 Shaping 试验双腿交叉；修复仅离线验证；策略接管未验证 | `checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml` | [q1](./q1/README_zh.md) |
| AgiBot A3 | 仿真已测试；真机未测试 | `checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml` | [a3](./a3/README_zh.md) |
| EngineAI T800 | 仿真已测试；真机未测试 | `checkpoints/proteus/wsq61/engineai_t800/policy.yaml` | [t800](./t800/README_zh.md) |
| AgiBot 灵犀 X2 | 2026-11-12 板载吊架试验（Shaping 正常；手臂/髋部振荡，加阻尼 overlay 仅在仿真中验证） | `checkpoints/proteus/proteus_wamseq_62/lingxi_x2`（`policy-pc-damped.yaml`，`start_proteus_onboard.sh`） | [x2](./x2/README_zh.md) |
| LimX OLI | 仿真已测试；真机未测试 | `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml` | [oli](./oli/README_zh.md) |
| Unitree H2 | 仅有机器人模型和 PICO 重定向；无硬件 I/O | — | [`docs/h2.md`](../docs/h2.md) |

默认策略指各机器人 `start_proteus.sh`（G1 为 `start_proteus_pc.sh`，ET1 为 `start_proteus_onboard.sh`）
使用的策略；启动脚本支持时，可用 `PROTEUS_POLICY_CONFIG` 或 `PROTEUS_POLICY_DIR` 覆盖。
