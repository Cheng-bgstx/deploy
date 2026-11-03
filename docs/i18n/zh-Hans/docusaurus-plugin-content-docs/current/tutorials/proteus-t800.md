---
title: 众擎 T800 上的 Proteus WAMSeq
slug: /tutorials/proteus-t800
---

# 众擎 T800 上的 Proteus WAMSeq

分支 `feat/proteus-wamseq-engineai-t800` 对应 **`engineai_t800`（25 个策略关节）**，机器人别名为 `t800`、`engineai_t800`。使用已发布的跨本体共享 actor，不需要训练或微调。包含模型、带名文件 tracking、PICO IK/ZMQ 和 inline 真机接口。**已验证软件与 MuJoCo，未连接真机或真实头显测试。**

## 准备

```bash
# From this branch's repository/worktree root. Reuse existing source checkouts.
sources_root="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
git -C "$sources_root/Proteus" worktree add --detach "$sources_root/Proteus-wamseq" origin/wamseq
gh repo clone engineai-robotics/GMR "$sources_root/GMR-engineai" -- --depth 1
gh repo clone YanjieZe/GMR "$sources_root/GMR" -- --depth 1
gh repo clone ubisoft/ubisoft-laforge-animation-dataset "$sources_root/ubisoft-laforge-animation-dataset" -- --depth 1
uv run --no-sync python -m scripts.t800.prepare --sources-root "$sources_root"
```

可用 `--sources-root /path/to/projects` 指定源仓库父目录（兼容 git worktree），`--proteus-root` 指定已有 `wamseq` 检出。`--skip-motions` 只准备模型、策略和 PICO 静态参考。推理用根环境，遥操用 `venv/pico`；准备脚本会在 `venv/pico` 子进程中生成 IK 数据，不负责安装真机 SDK。先按已有 Getting Started 页面配置两个环境；`--no-sync` 复用本机环境。

核对的源版本：[Proteus wamseq](https://github.com/Hxxxz0/Proteus/tree/1e0e0381bdec73652f8aa90b871f2980586bb52a) `1e0e038`、[厂商模型](https://github.com/engineai-robotics/GMR) `71e02c0`（`assets/engineai_t800/t800_mocap.xml`）、GMR `bb1bbe4`、LAFAN1 `9408460`。准备过程检查训练默认姿态下所有关节轴，把质量/限位差异写入 `third_party/engineai_t800/validation.json`，展开 MJCF include，并改为可迁移的相对资源路径。关节参数来自 `sim2real/config/robots/specs/` 中冻结的训练 spec。仿真只与地面碰撞，摩擦系数 1.0，采用 implicitfast，去除关节被动阻尼/摩擦，保留 contract 的增益、力矩限制和 armature。

`checkpoints/proteus/proteus_wamseq_62/engineai_t800/` 包含单个完整 `policy.onnx`、原始 contract、数值比较报告和两个 YAML。`policy-sim.yaml` 使用世界根位置，供 sim2sim；`policy.yaml` 在没有定位反馈时使用参考根位置，供真机。输入均为 `proprioception[10,68]`、`task[6,253]`、`action_history[10,31]`、`history_valid[10]`；输出为 `action[25]`、`next_slot_action[31]`。十步历史从旧到新，无效帧及其前序动作按训练语义屏蔽。相对于原始 actor，64 组 seed-0 输入的动作输出误差为 **0**（IR 8，opset 17）。`--release proteus_wamseq_62_full` 可导出另一份已发布 actor。

## 多动作数据与验证

T800 使用众擎 GMR `71e02c0` 中的 `bvh_lafan1_to_t800.json` 和 `smplx_to_t800.json`。后者按 `R_smplx = R_xrobot * Ry(pi)` 转换为 XRobot 人体坐标系。离线 BVH 与厂商一致，采用 Rx(90) 基变换且不取四元数共轭。有界求解器分别使用两个 IK 阶段的偏移，并用关键字传入 Mink 限位。

`checkpoints/engineai_t800/motions/` 是含多个带名 NPZ 和 manifest 的 any4hdmi 库。源数据是真实 LAFAN1 人体 mocap，在目标机器人上重定向，不是合成静态保持。manifest 标记输入 30 Hz，tracking 重采样为 50 Hz；根 XY 以首帧归零，四元数为 WXYZ，按脚底碰撞几何贴地。IK 关节速度上限 12 rad/s。独立 `pico_hold/` 数据集只用于实时遥操启动整形。

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot t800 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/engineai_t800/policy-sim.yaml \
  --dataset checkpoints/engineai_t800/motions --output outputs/t800/eval-new
```

本地验证（2026-11-03）：seed 0，策略 50 Hz，物理步长 0.005 s；开始 1 s 策略激活保持，播放每条 8 s mocap 摘录至末帧，再保持 1 s。通过条件为播放完成、数值有限，且每个播放帧在 14 个 contract body 上的平均位置误差不超过 0.5 m。下列默认动作全部通过。指标由实际记录轨迹计算，表中根部终点误差采用同一世界坐标系；NPZ 还包含双方各自在初始根坐标系表达的位移误差，两者定义不同。

| 动作 | 平均世界 body 误差（cm） | 平均局部 body 误差（cm） | 世界根部终点误差（cm） |
| --- | ---: | ---: | ---: |
| `aiming1_subject1` | 13.49 | 6.96 | 3.03 |
| `dance2_subject1` | 12.73 | 5.04 | 24.71 |
| `fightAndSports1_subject4` | 14.19 | 6.15 | 13.19 |
| `walk1_subject1` | 12.63 | 7.12 | 10.70 |

包含源/产物哈希与准确命令的机器可读记录见 `docs/experiments/proteus-t800.json`。重跑需要新的输出目录。阈值通过仅表示离线 tracking 结果，不等于真机或实时遥操结果。第一条舞蹈 `dance1_subject1` 在标准 actor 下摔倒，因此不属于默认库；full actor 通过了该压力片段，`eval-full` 与对应比较报告另行记录。默认四条动作使用标准 actor，舞蹈为 `dance2_subject1`。

## 真机入口

这里是标准 25 电机 T800，不对应两个 T800 Pro contract。厂商模型与训练模型仅有数值序列化精度差异。inline 适配使用官方 ROS `joint_bridge` 接口：`interface_protocol/JointState`、`JointCommand`、`ImuInfo`、`MotionState`。SDK 数组顺序为 J00–J24，与策略状态按名称显式映射；命令设置 `RL_PARSER=1`，由 SDK 转换串联 pitch/roll 踝关节命令。必须收到新鲜的 `current_motion_task=joint_bridge` 且存在命令订阅者。所检查的 native SDK 版本 `335c60e` 只有 ROS 遥测，没有关节命令订阅，单独启动其 ROS bridge 无法部署本适配。需要支持文档中 joint bridge 的固件，并用厂商控制器进入该模式；启动脚本不切换机器人模式。

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
# Use the vendor controller to enter joint_bridge and verify /motion/motion_state.
bash scripts/t800/start_proteus.sh checkpoints/engineai_t800/motions/motions/walk1_subject1.npz
```

[厂商 SDK/接口来源](https://github.com/engineai-robotics/engineai_ros2_workspace)。CPU 策略推理为 50 Hz，独立 ROS 定时器持续发送最新命令。必须有新鲜关节/IMU 反馈、有限数值命令和非负增益。命令/反馈超过 250 ms、错误或 zero 模式时，位置刚度、目标速度和前馈力矩归零，并保留阻尼。Q1 采用阻尼 5，单次发布的位置步进不超过 0.2 rad。正常退出会重复发送阻尼命令。这些属于进程内检查，不代表进程被杀或断电后有固件保护。IMU 姿态和角速度通过显式外参转换为骨盆 WXYZ/body 坐标。控制 generation 变化时丢弃旧命令。键位：`i` 在 10 s 内整形到首帧，`]` 开启策略 Hold，空格播放/暂停，`o` 阻尼，Ctrl+C 退出。可用 `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` 指定部署配置。

## PICO

```bash
# Terminal 1: calibrated PICO body/leg trackers and XRoboToolkit PC Service
bash scripts/t800/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: simulator (open its mjviser URL)
bash scripts/t800/start_pico.sh sim
# Terminal 3: policy, motion + button stream from terminal 1
bash scripts/t800/start_pico.sh policy-sim 127.0.0.1
# Hardware alternative to terminal 3, after SDK/IMU/ownership setup:
bash scripts/t800/start_pico.sh robot PUBLISHER_IP
```

启动器设置两个 HF 离线环境变量，选择正确环境，并在缺失时创建静态整形数据。动作端口 TCP 28701，按钮端口 TCP 5592（扳机保留 5593，本次没有新增手部驱动）。默认端口组一次运行一个本体。**A** 整形到中立位；松开后 **A+B** 开启策略 Hold；**X** 开始/暂停实时参考；**单独 B** 阻尼。先在 publisher viewer 中检查 IK，评估平衡前关闭仿真弹力绳。实时 XR 的比例按假定身高/实际身高归一化，关节速度限制为 12 rad/s。

软件测试用已知姿态替代 XR 设备，在 1.6/1.8/2.0 m 身高下验证 IK/FK、带名 ZMQ 动作、按钮、暂停/重连、语义观测和完整 ONNX 推理。ONNX 留在根环境，IK 在 `venv/pico`。身体流丢失时保持最后参考；若机器人状态通道还在，不会自动进入阻尼。PICO 没有未来帧，实时 buffer 用已接收姿态补齐缺失的未来参考。文件 tracking 指标不能代表实时遥操质量。

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m pytest \
  tests/test_proteus_transport.py tests/test_proteus_observations.py \
  tests/test_robot_io.py tests/test_command_sender.py -q
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py -q
```

大型 ONNX、厂商资产、动作和轨迹不进入 git。可以按上面的步骤重建；不要假定它们已经上传远端。
