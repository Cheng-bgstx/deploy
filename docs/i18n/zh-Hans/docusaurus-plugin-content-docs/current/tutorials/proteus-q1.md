---
title: 启元 Q1 上的 Proteus WAMSeq
slug: /tutorials/proteus-q1
---

# 启元 Q1 上的 Proteus WAMSeq

## 默认 checkpoint（2026-11-04）

启动脚本现默认使用 **wsq61**，即
[ModelScope Hzzzz001/Proteus](https://modelscope.cn/datasets/Hzzzz001/Proteus)
中的 `wsq61_wam_61robots_30000.pt`。当前策略路径为
`checkpoints/proteus/wsq61/primebot_q1/policy.yaml`，使用测得根位置的仿真选择
`policy-sim.yaml`。
checkpoint SHA256 为 `1d1c65ddf5706b940744d87e2f86f7cea453c9240bcccc12d31923211f694bba`。
保留原关节/形态合约，actor 使用十步 WAMSeq 窗口。旧产物仍可使用。
下文配方及性能数据除特别标明 wsq61 外均对应旧版本，不能作为 wsq61 的验收数据。
本地 wsq61 导出来源和逐机器人对比记录位于
`checkpoints/proteus/wsq61/source/export.json` 与 `<robot>/comparison.json`。

分支 `feat/proteus-wamseq-primebot-q1` 对应 **`primebot_q1`（22 个策略关节）**，机器人别名为 `q1`、`primebot_q1`。使用已发布的跨本体共享 actor，不需要训练或微调。包含模型、带名文件 tracking、PICO IK/ZMQ 和 inline 真机接口。**已验证软件与 MuJoCo，未连接真机或真实头显测试。**

## 准备

```bash
# From this branch's repository/worktree root. Reuse existing source checkouts.
sources_root="$(dirname "$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")")"
git -C "$sources_root/Proteus" worktree add --detach "$sources_root/Proteus-wamseq" origin/wamseq
gh repo clone PrimebotTech/primebot-q1-urdf "$sources_root/primebot-q1-urdf" -- --depth 1
gh repo clone YanjieZe/GMR "$sources_root/GMR" -- --depth 1
gh repo clone ubisoft/ubisoft-laforge-animation-dataset "$sources_root/ubisoft-laforge-animation-dataset" -- --depth 1
uv run --no-sync python -m scripts.q1.prepare --release proteus_wamseq_62 --sources-root "$sources_root"
```

可用 `--sources-root /path/to/projects` 指定源仓库父目录（兼容 git worktree），`--proteus-root` 指定已有 `wamseq` 检出。`--skip-motions` 只准备模型、策略和 PICO 静态参考。推理用根环境，遥操用 `venv/pico`；准备脚本会在 `venv/pico` 子进程中生成 IK 数据，不负责安装真机 SDK。先按已有 Getting Started 页面配置两个环境；`--no-sync` 复用本机环境。

核对的源版本：[Proteus wamseq](https://github.com/Hxxxz0/Proteus/tree/1e0e0381bdec73652f8aa90b871f2980586bb52a) `1e0e038`、[厂商模型](https://github.com/PrimebotTech/primebot-q1-urdf) `9657817`（`xml/q1_v3.xml`）、GMR `bb1bbe4`、LAFAN1 `9408460`。准备过程检查训练默认姿态下所有关节轴，把质量/限位差异写入 `third_party/primebot_q1/validation.json`，展开 MJCF include，并改为可迁移的相对资源路径。关节参数来自 `sim2real/config/robots/specs/` 中冻结的训练 spec。仿真只与地面碰撞，摩擦系数 1.0，采用 implicitfast，去除关节被动阻尼/摩擦，保留 contract 的增益、力矩限制和 armature。

`checkpoints/proteus/proteus_wamseq_62/primebot_q1/` 包含单个完整 `policy.onnx`、原始 contract、数值比较报告和两个 YAML。`policy-sim.yaml` 使用世界根位置，供 sim2sim；`policy.yaml` 在没有定位反馈时使用参考根位置，供真机。输入均为 `proprioception[10,68]`、`task[6,253]`、`action_history[10,31]`、`history_valid[10]`；输出为 `action[22]`、`next_slot_action[31]`。十步历史从旧到新，无效帧及其前序动作按训练语义屏蔽。相对于原始 actor，64 组 seed-0 输入的动作输出误差为 **0**（IR 8，opset 17）。`--release proteus_wamseq_62_full` 可导出另一份已发布 actor。

## 多动作数据与验证

所检查的公开源中没有 Q1 厂商 GMR 配置。这里从 G1 已标定的人体坐标系与 Q1 自身中立姿态 FK 生成 BVH、XRobot 表，包括目标 link 轴、根部/肢体比例和位置偏移。动作始终在 Q1 模型上求 IK，没有复用 G1 关节轨迹。配置中记录了源哈希和生成方法。

`checkpoints/primebot_q1/motions/` 是含多个带名 NPZ 和 manifest 的 any4hdmi 库。源数据是真实 LAFAN1 人体 mocap，在目标机器人上重定向，不是合成静态保持。manifest 标记输入 30 Hz，tracking 重采样为 50 Hz；根 XY 以首帧归零，四元数为 WXYZ，按脚底碰撞几何贴地。IK 关节速度上限 12 rad/s。独立 `pico_hold/` 数据集只用于实时遥操启动整形。

```bash
uv run --no-sync python -m scripts.proteus.evaluate --robot q1 \
  --policy-config checkpoints/proteus/proteus_wamseq_62/primebot_q1/policy-sim.yaml \
  --dataset checkpoints/primebot_q1/motions --output outputs/q1/eval-new
```

本地验证（2026-11-03）：seed 0，策略 50 Hz，物理步长 0.000625 s；开始 1 s 策略激活保持，播放每条 8 s mocap 摘录至末帧，再保持 1 s。通过条件为播放完成、数值有限，且每个播放帧在 14 个 contract body 上的平均位置误差不超过 0.5 m。下列默认动作全部通过。指标由实际记录轨迹计算，表中根部终点误差采用同一世界坐标系；NPZ 还包含双方各自在初始根坐标系表达的位移误差，两者定义不同。

| 动作 | 平均世界 body 误差（cm） | 平均局部 body 误差（cm） | 世界根部终点误差（cm） |
| --- | ---: | ---: | ---: |
| `dance1_subject1` | 4.88 | 2.31 | 8.46 |
| `fightAndSports1_subject4` | 4.06 | 2.63 | 4.39 |
| `walk1_subject1` | 4.46 | 1.81 | 1.74 |

包含源/产物哈希与准确命令的机器可读记录见 `docs/experiments/proteus-q1.json`。重跑需要新的输出目录。阈值通过仅表示离线 tracking 结果，不等于真机或实时遥操结果。T800 舞蹈压力测试结果若出现在记录中，不属于默认数据集。

## 仿真步长修正

已使用同一份 WAMSeq 权重、验证过的本体模型和完全相同的 50 Hz 参考数组，
直接运行未修改的 Proteus `deploy/sim2sim.py`。Q1、A3 的平均位置误差在两边接近，
T800 原来的 5 ms 外部 PD 仿真则明显更差。统一按 50 Hz 播放帧采样时，
T800 本地 body 误差为 12.72 cm，原版为 5.62 cm；Q1 虽然位置误差接近，
关节加速度 RMS 却为 87.11，对应原版 33.83 rad/s²。

原版使用 0.625 ms 物理步长和隐式关节伺服。MuJoCo 的 `implicitfast` 并不会让
在积分器外按上一步速度计算的反馈力矩自动成为隐式反馈。仅将本地物理步长改为
0.625 ms，T800 的统一采样误差就降至 5.84 cm，Q1 加速度 RMS 降至 28.42 rad/s²。
单独去掉启动 Hold 不能解决 T800 差距；在 5 ms 下改用隐式伺服也能解决。
实验记录保留了各项单变量对照。

`evaluate.py` 现在对 **Q1/T800 默认使用 0.000625 s**，可用 `--sim-dt` 覆盖。
PICO 仿真启动器传入 `--sim-dt 0.000625 --decimation 32`，策略推理和 viewer 更新
仍为 50 Hz。直接使用 integrated 仿真命令时，也应显式加 `--sim-dt 0.000625`。
开始和末帧 Hold 均保留。上表已按修正步长重新评测。

回归测试 `tests/test_proteus_sim_resolution.py` 使用实际 checkpoint 和步行动作，
修正前失败、修正后通过，可捕获 Q1 加速度 RMS >=60 rad/s² 或 T800 平均 body
误差 >=8 cm。源仓库和 ONNX 权重均未修改。本修正针对仿真数值稳定性，
不代表真机性能已验证。

## 真机入口

公开 V3 MJCF 的质量、限位和全部关节轴与训练一致。22 关节策略没有虚构腕部或头部电机。Q1 HAL 使用全身带名 `aimdk_msgs/JointStateArray` / `JointCommandArray`，topic 为 `/aima/hal/joint/state` 和 `/aima/hal/joint/command`，命令频率 1000 Hz。公开 SDK **没有给出根部 IMU topic**；必须用 `SIM2REAL_Q1_IMU_TOPIC` 或 `robot_io_options.imu_topic` 指向已确认的 `sensor_msgs/Imu` 骨盆数据流，缺失时会在连接前报错。`imu_to_pelvis_wxyz` 需要按实际安装方向配置。本次没有验证真机 IMU 地址或标定。

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
# Configure the verified pelvis IMU topic before either hardware launcher.
export SIM2REAL_Q1_IMU_TOPIC=/your/verified/pelvis_imu_topic
bash scripts/q1/start_proteus.sh checkpoints/primebot_q1/motions/motions/walk1_subject1.npz
```

[厂商 SDK/接口来源](https://github.com/PrimebotTech/primebot-q1-sdk)。CPU 策略推理为 50 Hz，独立 ROS 定时器持续发送最新命令。必须有新鲜关节/IMU 反馈、有限数值命令和非负增益。命令/反馈超过 250 ms、错误或 zero 模式时，位置刚度、目标速度和前馈力矩归零，并保留阻尼。Q1 采用阻尼 5，单次发布的位置步进不超过 0.2 rad。正常退出会重复发送阻尼命令。这些属于进程内检查，不代表进程被杀或断电后有固件保护。IMU 姿态和角速度通过显式外参转换为骨盆 WXYZ/body 坐标。控制 generation 变化时丢弃旧命令。键位：`i` 在 10 s 内整形到首帧，`]` 开启策略 Hold，空格播放/暂停，`o` 阻尼，Ctrl+C 退出。可用 `PROTEUS_POLICY_CONFIG=/path/to/custom.yaml` 指定部署配置。

## PICO

```bash
# Terminal 1: calibrated PICO body/leg trackers and XRoboToolkit PC Service
bash scripts/q1/start_pico.sh publisher --actual-human-height 1.8
# Terminal 2: simulator (open its mjviser URL)
bash scripts/q1/start_pico.sh sim
# Terminal 3: policy, motion + button stream from terminal 1
bash scripts/q1/start_pico.sh policy-sim 127.0.0.1
# Hardware alternative to terminal 3, after SDK/IMU/ownership setup:
bash scripts/q1/start_pico.sh robot PUBLISHER_IP
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
