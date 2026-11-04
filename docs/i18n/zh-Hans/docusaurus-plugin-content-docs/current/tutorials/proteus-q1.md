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

分支 `feat/proteus-wamseq-primebot-q1` 对应 **`primebot_q1`（22 个策略关节）**，机器人别名为 `q1`、`primebot_q1`。使用已发布的跨本体共享 actor，不需要训练或微调。包含模型、带名文件 tracking、PICO IK/ZMQ 和 inline 真机接口。**已验证软件、MuJoCo 和真机只读反馈。首次实机塑形出现双腿交叉；命令顺序修正已有离线测试覆盖，但未再次通过实机验证。策略接管与真实头显操作仍未验收。**

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

Q1 HAL 使用 22 个带名关节，消息为 `aimdk_msgs/JointStateArray` / `JointCommandArray`，topic 为 `/aima/hal/joint/state` 和 `/aima/hal/joint/command`。公开 V3 模型与训练合约匹配；每台实机的零位与物理限位仍需核对。2026-11-04，本机连接的 Q1 在 **`/aima/hal/imu/state`** 发布 `sensor_msgs/Imu`，22 个关节名称全部匹配。wsq61 完成了 501 次只读推理，没有反馈超时或非有限值，观测计算加 ONNX 平均 3.66 ms，全程没有发送电机命令。

厂商 V3 模型明确将 `imu_0` 直接定义在 **`pelvis`（骨盆）** 上，相对该刚体没有旋转，见 [site 定义](https://github.com/PrimebotTech/primebot-q1-urdf/blob/9657817ba375f41b0e5d2b62d8d6b72b757cdf4f/xml/q1_v3_mc.xml#L40-L41)。[姿态、陀螺仪和加速度计](https://github.com/PrimebotTech/primebot-q1-urdf/blob/9657817ba375f41b0e5d2b62d8d6b72b757cdf4f/xml/q1_v3_mc.xml#L270-L276) 都引用这个 site。加载厂商 MJCF 后确认所属刚体为 `pelvis`、局部位置为 `[0, 0, 0]`、四元数为 `[1, 0, 0, 0]`。因此按官方模型使用 `imu_to_pelvis_wxyz: [1, 0, 0, 0]`，本机当前配置已经如此，无需腰关节补偿。公开 SDK 文字没有单独说明实机 ROS IMU 输出坐标系是否遵循该模型，实机及驱动与模型的一致性仍待核对；这不意味着模型中的 IMU 所属刚体未知。不要把机器人当前倾斜角当成安装误差消除。

生成独立的实机配置，保留导出 actor、合约、默认姿态和 PD 增益：

```bash
source /opt/ros/humble/setup.bash
source /path/to/vendor_workspace/install/setup.bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
# 厂商 V3 模型中 imu_0 与 pelvis 坐标轴对齐。
uv run --no-sync python -m scripts.q1.prepare_deployment \
  --imu-topic /aima/hal/imu/state --imu-to-pelvis-wxyz 1 0 0 0 --manual-policy-entry
# 本次检查的 PC 可 source outputs/q1/source_env.sh 加载有线 DDS 配置。
bash scripts/q1/start_proteus.sh checkpoints/primebot_q1/motions/motions/walk1_subject1.npz
```

`start_proteus.sh` 选择 `checkpoints/proteus/wsq61/primebot_q1/policy-pc.yaml`，可用 `PROTEUS_POLICY_CONFIG` 覆盖。该配置提供 `shaping_target: motion_first_frame` 所需的软限位及 15 rad/s 策略目标速度限制。上方命令按本机操作者要求明确选择**手动策略接管**：按 `]` 直接进入暂停回放的 policy Hold，不要求塑形完成、关节误差小于 0.10 rad 或速度小于 0.5 rad/s。`i` 仍可选择执行十秒塑形。启动仍未使能，反馈检查、超时阻尼、目标限位及 `o` 停止仍保留。这是操作者手动试机，不代表实机验收通过。

重力前馈**默认关闭**。最新实机记录虽然包含非零前馈命令，反馈力矩却接近单独的 PD 项，肩部下垂仍然存在。离线 SDK/CDR 序列化保留了 `effort`；具体是硬件执行链路还是反馈语义存在差异，尚未确定。显式传入 `--enable-shaping-gravity` 才启用实验性的限幅模型前馈（两秒渐增、关节力矩上限的 10%、仅塑形阶段）。不传 `--manual-policy-entry` 则恢复塑形完成、姿态和速度接管门槛。这些选项均不改变导出的 actor 和 PD 增益。

### 实机命令顺序问题（2026-11-04）

实机仅按 `i` 的塑形尝试出现双腿交叉，操作者随后进入阻尼并退出，没有进入策略。适配器按“腿在前”的策略顺序序列化，而实机全身 HAL 反馈为“手臂在前”；厂商 `publish_full_command` 示例保持反馈顺序。停止后的若干读数接近其数组位置上收到的目标，而不是按名称给出的目标，例如左肩 yaw 实测 0.0545，错误槽位目标 0.0515，原定目标 -0.5864 rad。这强烈支持硬件路径按数组位置取值的解释，但尚未独立确认固件行为。

Q1 现将**所有命令字段**按最新、完整且验证通过的反馈顺序发布，包括阻尼命令；内部观测与策略输出仍按名称映射。新增回归测试将实测布局输入真实序列化器，再交给按数组位置接收的模拟 HAL。离线演练也已包含序列化；之前直接将带名 codec 数值施加给 MuJoCo，漏掉了这一接口错误。修正后的演练通过**不代表实机故障已修复验收**。下一步不应直接重试全身塑形，应先在可靠支撑、控制权明确且可停止的条件下，验证实测位置保持和单关节小幅响应。

### 可选的当前位置保持与单关节检查

保持“高级开发／低层运控开发”，停止其他发命令的程序。在阻尼状态下确认吊架承担机器人重量、双腿已解开交叉、脚底不会限制测试运动。这组低增益不能支持机器人站立。原厂停止操作应随时可用；网络断开时，主机键盘无法停止机器人。

```bash
cd /home/galbot/projects/sim2real  # 本次检查的 PC
source outputs/q1/source_env.sh
uv run --no-sync python -m scripts.q1.commission_joints --joint left_elbow_joint
```

此入口使用正式 Q1 传输层及按反馈顺序排列的序列化器，不加载策略。启动时未使能，按 **`h`** 前不发送电机命令。等到反馈有效、显示 `publishers=1` 后：

1. 按 **`h`**，捕获并保持当前实测关节位置。刚度在两秒内从零升至 5，阻尼为 2，不向动作首帧移动。至少观察三秒；出现意外扭转立即按 **`o`**。
2. 有支撑的姿态保持稳定后，按 **`j`**。只有选定关节的目标变化：相对捕获姿态，用八秒完成 `+0.03 → 0 → -0.03 → 0` rad，约 ±1.7°。左右均按机器人自身定义。结合编码器读数，确认实物运动的关节及方向；其余关节目标保持不变。
3. 往返结束一秒后程序自动阻尼并退出；**`o`**、**空格**、**`q`**、**Esc** 或 **Ctrl+C** 可提前停止。**仅此诊断程序中空格表示停止；策略启动程序中的空格表示开始／暂停动作回放。**

关节仍在运动或发现其他命令发布者时拒绝使能。测试中反馈超时、反馈数组顺序或发布者数量变化、任一关节偏离捕获姿态超过 0.10 rad、速度超过 0.5 rad/s 都会停止；使能后最长运行 25 秒。选定关节接近模型限位或低刚度保持发生漂移时拒绝往返，不要通过提高增益强行继续。发布者检查不能替代原厂控制器释放控制权。

每次运行将目标、实测位置／速度、消息顺序及停止原因保存到 `outputs/q1/commission/`。附加 `--read-only --duration-s 15` 可在不使能的情况下检查反馈。往返结束不会自动标记实机验收通过；先检查日志及实物响应，再更换关节测试（`--joint right_elbow_joint`，之后检查本次异常涉及的腰、髋关节）。完成关节对应关系与方向检查后，才恢复全身塑形和策略接管。

### IMU 数据一致性的只读检查

```bash
source outputs/q1/source_env.sh  # 仅本次检查的 PC；其他机器加载各自 ROS/厂商环境
uv run --no-sync python -m scripts.q1.inspect_imu --duration-s 30
```

此工具仅创建订阅，默认采用官方模型中与骨盆对齐的 IMU 坐标系。这是可选的实机及驱动一致性检查，不是选择官方模型外参的前提。实机的 `q1_imu_state` 标签本身不说明变换，本机八秒检查期间也没有收到可用的 IMU→骨盆 TF；这两点均不推翻厂商模型中的定义。

对于刚性安装在骨盆上的 IMU，应将实际轴向与模型核对：x 向前、y 向左、z 向上。停止驱动并保持可靠支撑后，小幅已知转动应主要出现在对应角速度轴：向右倾（左侧抬高）对应 +x、前倾对应 +y、向左转对应 +z。骨盆实际水平时，变换后的 roll/pitch 应接近零，静止比力应接近 `[0, 0, +9.81]` m/s²。仅凭重力无法确定安装 yaw，还需核对轴标记或多个已知转轴。坐标约定见 [ROS REP 103](https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst) 和 [IMU REP 145](https://github.com/ros-infrastructure/rep/blob/master/rep-0145.rst)。

本次原始 IMU 显示约 18.9° pitch；由四元数预测的重力方向与实测方向相差 0.25°。这说明数据内部一致，不代表安装标定已完成。工具中的 `assumed_pelvis_*` 字段使用传入的 `--imu-to-pelvis-wxyz`，不会将外参标记为已验证，也不会修改部署配置。不要因为照片中躯干看起来竖直就直接减掉这 18.9°。

手动接管的操作顺序（对应上方配置生成命令）：

1. 保持吊架承重，确认原厂高层运控已释放控制权，并启用“高级开发／低层运控开发”。首次按 **`i` 或 `]`** 前，Q1 不发送电机命令，退出时也不发送。
2. 可选按 **`i`** 执行十秒塑形。关闭重力前馈后可能仍存在静态下垂；手动接管模式不要求完成此步骤。
3. 按 **`]`** 直接进入 **policy Hold**。历史重置，目标速度限制从实测关节位置开始，参考停在第 0 帧。Hold 是策略主动控制，不是阻尼；即使暂停回放，肢体也可能运动。
4. 首次 Hold 保留吊架保护。**`o`** 进入阻尼，**Ctrl+C** 退出。**空格播放／暂停参考动作，不是电机停止键。**检查有支撑的 Hold 和脚底接触后再播放。手动模式下阻尼后也可直接按 `]`，不强制重做 `i`。

[厂商接口来源](https://github.com/PrimebotTech/primebot-q1-sdk)。策略推理为 50 Hz，ROS 定时器请求以 1000 Hz 发布命令。接管后，命令／反馈超时（>250 ms）、错误和 zero 模式均使用零位置刚度、零目标速度、零前馈力矩和阻尼 5；单次发布的位置步进限制为 0.2 rad。接管后的正常退出会重复发送阻尼命令。这些是进程内检查，未验证进程被杀或断电后的固件保护。开发模式开启、ROS 命令发布者数量为零，都不能单独证明原厂内部控制器已释放控制权。

可以离线演练实际 Tracking 启动、命令编解码和塑形到 Hold 的切换，整个过程不连接 ROS：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1 \
  uv run --no-sync python -m scripts.q1.validate_deployment
# 可选：--initial-feedback outputs/q1/preflight-20261104/ros-readonly.json
# 或：--initial-tracking-record <本地-tracking-记录.npz>
# 加 --no-use-recorded-orientation 可对比动作首帧的根方向。
```

之前重置根位姿的演练没有正确模拟支撑重力：每个子步重置自由根位姿并不会产生支撑力，因此漏检了肩部下垂。现在使用实际承重的 mocap weld 约束、正式命令序列化器及 0.625 ms 物理步长。回放 2026-11-04 实机关节姿态和约 19° 骨盆 pitch 时，不加补偿可复现接管前 0.305 rad 的误差；补偿后降至 0.002 rad。保留该前倾姿态的演练虽能进入 Hold，但解除支撑阶段未达到标准（最低高度 0.280 m、最小直立余弦 0.776）。使用相同初始关节、改为参考根方向后通过：10.4 s 进入 Hold，首目标零跳变，解除支撑后最低根高度 0.405 m、最小直立余弦 0.993。以上均为离线结果，weld 仍简化了吊绳力学，不能证明实机可解除保护。

对应的实机记录显示肩部持续下垂（左肩 roll 在 kp 4.2032 时误差 0.340 rad），实测力矩接近模型重力力矩。策略接管前降低吊架落地后，双踝 pitch 误差增大，右踝达到 0.351 rad，同时骨盆 pitch 增至约 27°。之后启用前馈的实机尝试仍存在肩部下垂，始终未进入策略。操作者随后选择关闭前馈、采用手动接管，绕过可选的 0.10 rad 接管门槛；之前的仿真结果不能视为本次手动接管试验已通过验证。

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
