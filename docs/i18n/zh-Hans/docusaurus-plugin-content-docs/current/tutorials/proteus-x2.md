---
title: 在智元灵犀 X2 上部署 Proteus WAMSeq
slug: /tutorials/proteus-x2
---

# 在智元灵犀 X2 上部署 Proteus WAMSeq

本适配针对 Proteus **`lingxi_x2` 31 关节合约**，使用公开的 X2 Ultra v1.3 模型，
接入本仓库的 Tracking 和集成 sim2sim。机器人别名为 `x2`、`lingxi_x2`、`agibot_x2`。
策略是原始共享 WAMSeq actor，使用合约中的形态描述和带符号动作映射固定为 X2，无需
微调。已检查两个完整动态动作的 sim2sim，真机尚未验证。

## 准备模型和策略

使用根项目 Python 3.10 环境，需要 MuJoCo 3.11、ONNX / CPU ONNX Runtime、Pillow、
joblib 和 `ffmpeg`。`--no-sync` 用于绕过当前仓库已有 ARM GPU 依赖的全平台解析问题。

```bash
git -C ../Proteus worktree add --detach ../Proteus-wamseq origin/wamseq
gh repo clone AgibotTech/agibot_x2_urdf ../agibot_x2_urdf -- --depth 1
uv run --no-sync python scripts/x2/prepare_assets.py \
  --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
uv run --no-sync python scripts/x2/export_policy.py --proteus-root ../Proteus-wamseq
```

已有工作区直接复用，不必重复克隆。本次验证版本：Proteus `1e0e038`、官方模型
`575cc6b`。模型为 `X2_URDF-v1.3.0/x2_ultra.xml`，官方 v1.4 是不同版本，参见
[官方模型说明](https://github.com/AgibotTech/agibot_x2_urdf)。

`prepare_assets.py` 生成 `third_party/lingxi_x2/model/scene_tracking.xml`，复制网格
和许可证，移除原 actuator/sensor 块，并应用训练限位、armature 和零被动阻尼/摩擦。
集成仿真添加 31 个按关节命名的力矩执行器，并按训练设置关闭自碰撞。训练默认姿态下
各关节轴误差均为零。官方模型总质量 **41.966473 kg**，训练描述为 **41.80159766 kg**，
差值 **0.16487534 kg**；保留官方模型惯量。12 个关节范围与训练不同，仿真模型显式
覆盖为训练范围。差异与文件哈希均记录于 `validation.json`。AimDK 后端独立使用官方
物理限位与训练限位的交集，不能用更宽的训练范围扩大硬件范围。

导出结果位于 `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/`，包含一个完整的
`policy.onnx`、原始合约、`comparison.json` 和两个 YAML：

| 配置 | 根部平移来源 | 用途 |
| --- | --- | --- |
| `policy-sim.yaml` | 仿真测得的世界系根部位置，`reference_forcing: false` | 与原仓库 MuJoCo 观测语义一致的 sim2sim |
| `policy.yaml` | 参考锚点，`reference_forcing: true` | 没有世界系定位反馈的 SDK 局部跟踪 |

两者绑定 `proprioception[10,68]`、`action_history[10,31]`、`history_valid[10]` 和
`task[6,253]`，输出 `action[31]`、`next_slot_action[31]`。历史、有效掩码、关节符号、
动作缩放和增益均来自原合约。64 组 seed=0 输入对比中，封装前后输出误差为零；ONNX
为 IR 8、opset 17。`--release proteus_wamseq_62_full` 可导出另一份 actor。

## 动态参考与视频

本次使用 [sonic-x2](https://github.com/meetsitaram/sonic-x2) 公开的 X2 动作姿态，
版本 `c995944`，仅使用动作 PKL 和显式关节列顺序，不加载其中的 SONIC 策略或部署
调参。转换器按名称映射 31 列关节，将源根部四元数 XYZW 转为 WXYZ，保留绝对根部
高度，将首帧 XY 平移到零。any4hdmi manifest 保存源帧率，运行时重采样到 50 Hz。

```bash
gh repo clone meetsitaram/sonic-x2 ../sonic-x2-reference -- --depth 1 --filter=blob:none --sparse
git -C ../sonic-x2-reference sparse-checkout set motions scripts assets/mjcf
uv run --no-sync python scripts/x2/convert_reference.py \
  --clip gangam_dance --output outputs/x2/gangam_dance
uv run --no-sync python scripts/x2/convert_reference.py \
  --clip relaxed_walk --output outputs/x2/relaxed_walk
uv run --no-sync python scripts/x2/render_sim2sim.py \
  --motion-path outputs/x2/gangam_dance --duration 11.2 \
  --output outputs/x2/x2_wamseq_gangam_dance.mp4 \
  --label 'X2 Gangnam dance reference | Proteus WAMSeq | measured world root'
uv run --no-sync python scripts/x2/render_sim2sim.py \
  --motion-path outputs/x2/relaxed_walk --duration 17 \
  --output outputs/x2/x2_wamseq_relaxed_walk.mp4 \
  --label 'X2 relaxed walk reference | Proteus WAMSeq | measured world root'
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_relaxed_walk.mp4
```

视频右侧为集成物理仿真，左侧为参考姿态播放，共享同一相机，原速、25 fps、1280×720。
seed=0，策略 50 Hz，物理步长 0.005 秒。初始暂停 1 秒和结束保持末帧期间策略均持续
运行。每个视频旁保存同次运行的 `.root.npz`、`.trajectory.npz`、`.json`、
`.metrics.json` 和 JPEG 预览。

本地验证记录（2026-11-02），误差统计对象为播放期间的 14 个合约跟踪身体：

| 参考动作 | 动作 / 视频时长 | 世界系 / 局部平均身体位置误差 | 世界系根部末端位置误差 |
| --- | --- | --- | --- |
| 江南舞 | 9.2 / 11.2 秒 | 8.24 / 5.42 厘米 | 1.00 厘米 |
| 放松行走 | 14.82 / 17 秒 | 4.60 / 4.05 厘米 | 1.72 厘米 |

两段都完整跑完，没有跌倒。舞蹈期间根部高度 0.608–0.665 米，行走期间为
0.636–0.667 米。世界系误差直接比较身体位置；局部误差将两套身体分别变换到各自的
骨盆坐标系。根部记录还包含旧版“分别在各自初始根坐标系中表达位移”的误差（舞蹈
2.46 厘米，行走 45.54 厘米），因初始骨盆朝向不同，与世界系末端误差不是同一指标。
指标 JSON 对这两类量作了明确命名。

## AimDK 部署

inline 后端遵循 [AimDK 关节接口](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html)
和[骨盆 IMU 说明](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/hal/sensor.html)。
订阅四组关节状态及 `/aima/hal/imu/torso/state`（虽然叫 torso，实际为胯部 IMU），
按组/名称映射数组，将 ROS XYZW 四元数转为 WXYZ。独立 ROS 定时器以 500 Hz 复发
最新命令，策略推理为 50 Hz。反馈或命令超过 250 ms、异常反馈以及 `zero` 模式均将
位置刚度、速度和前馈力矩清零，保留阻尼。正常退出时发送 5 次阻尼命令。HAL 本身
没有发布者死亡超时保护，因此此线程不能覆盖进程被强杀或电脑断电的情况。

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/setup.bash
# 真机取得 HAL 控制权前，需要显式释放原生控制器：
# aima em stop-app mc
bash scripts/x2/start_proteus.sh /path/to/x2/any4hdmi_dataset
```

接管前设置正确的 ROS domain/网络并确认反馈正常。脚本设置离线环境、CPU 推理、
`robot=x2`、`robot_io=inline` 和键盘控制：`i` 用 10 秒插值到第 0 帧，`]` 启用策略
Hold，空格播放/暂停，`o` 阻尼，Ctrl+C 退出。首次真机测试需支撑机器人。

`policy.yaml` 中的 `robot_io_options` 配置频率、超时、IMU 话题、四元数
`imu_to_pelvis_wxyz` 和 `head_pitch_enabled`。默认单位 IMU 旋转对应模型；真机必须
使用出厂标定外参，不能只依赖话题 `frame_id`。旧版 X2 Ultra 仅头部 yaw 有效，因此
默认不驱动 head pitch。本适配没有验证其他 X2 修订版本。没有世界系定位时，不能在
真机使用 `policy-sim.yaml`。

回归检查：`uv run --no-sync python -m pytest tests/test_x2.py tests/test_oli.py
tests/test_proteus_observations.py tests/test_et1_proteus.py tests/test_robot_io.py
tests/test_command_sender.py -q`，并通过 Python 语法和 shell 参数检查。
`third_party/`、`checkpoints/`、`outputs/` 下的运行资源不纳入 Git。


## PICO 全身遥操

发布端使用 **`venv/pico`**，策略与仿真使用 **根项目**。发布 PC 和策略计算机均按上文
准备 X2 模型。按 [PC 遥操安装](../getting-started/teleop-x86-64.md)安装 XRoboToolkit
PC Service，佩戴腿部追踪器，完成 PICO 全身标定并开启 body/controller 数据流。首次安装
先克隆两个 SDK 仓库，再执行 `uv sync --project venv/pico --no-install-package xrobotoolkit-sdk`，
随后运行 `bash scripts/setup/setup_xrobot_pybind.sh`。

内置配置来自社区 [TaiE64/AgibotX2-Retargeter](https://github.com/TaiE64/AgibotX2-Retargeter)
的 `bb582d9`（Apache-2.0），不是智元官方提供的 PICO 标定。SMPL-X 刚体偏移复合
`Ry(pi)` 转为 XRobot 坐标；源配置中的 `*_toe_link` 辅助连杆在厂商模型中不存在，因此改用
已标定的 LAFAN 踝关节任务，位置来自 XR Ankle、朝向来自 XR Foot。IK/FK 均使用准备好的
31 关节模型，不复制 G1 关节角。求解器加入肩部/腰部中立姿态约束，分别使用两阶段 IK 的
偏移，以关键字传入 Mink 限位，并按发布频率把连续关节目标限制在 **12 rad/s**。
实时人体尺寸按“标定身高 / 操作者实测身高”归一化。

配置已提交 Git；生成器记录输入配置 SHA256。可选的重新生成步骤：

```bash
gh repo clone TaiE64/AgibotX2-Retargeter ../AgibotX2-Retargeter -- --depth 1
# 必须使用原版 GMR 的源坐标配置，不能混入 LimX 新版 BVH 基变换。
uv run --no-sync python scripts/x2/build_pico_profile.py \
  --retarget-root ../AgibotX2-Retargeter \
  --gmr-configs ../GMR/general_motion_retargeting/ik_configs
```

在策略计算机上生成一次中立启动参考。流式 tracking 的
`shaping_target: motion_first_frame` 仍然需要一个动作文件：

```bash
uv run --no-sync python scripts/make_hold_motion.py --robot x2 --output outputs/x2/pico_hold
```

本地 sim2sim 开 **三个终端**：

```bash
# 1：发布 PC，实测身高，单位米
bash scripts/x2/start_pico.sh publisher --actual-human-height 1.70
# 2：根项目 MuJoCo / mjviser
bash scripts/x2/start_pico.sh sim
# 3：根项目 tracking，policy-sim.yaml 使用测得的仿真根位置
bash scripts/x2/start_pico.sh policy-sim 127.0.0.1
```

先检查发布器中的重定向。**A** 用 10 秒整形到中立姿态，松开后 **A+B** 启用策略 Hold。
**X** 开始实时参考，X 再次按下回到当前 XY/朝向的中立 Hold。**单独 B** 切换阻尼。
自由平衡测试前在仿真 GUI 关闭弹力带。X 暂停参考时策略继续运行。头部目标保持中立，
真机 head pitch 仍按默认设置禁用。

真机测试时，在 X2 策略计算机上将终端 3 替换为：

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/setup.bash
# 接管 HAL 前按上文释放原生 mc 控制权。
# aima em stop-app mc
bash scripts/x2/start_pico.sh robot PUBLISHER_IP
```

脚本设置 HF 离线参数、CPU 推理、50 Hz 策略、`robot=x2`、`robot_io=inline` 和真机
`policy.yaml`；`policy-sim` 使用 ZMQ RobotIO 与 `policy-sim.yaml`。两者均连接发布 PC 的
动作 **TCP 28701** 和手柄 **TCP 5592**，两端需放通。位置参数后可追加 tracking 参数，
覆盖策略或中立 `--motion-path`。同一组端口只运行一个发布器/仿真，OLI 与 X2 默认端口
不能在同一主机并行占用。扳机流仍在 5593，此适配未加入 X2 灵巧手驱动。

body 断流时接收端保持最后参考，**不会自动切换阻尼**；重连时重新对齐动作段。
手柄链路正常时可按 B，网络/进程失败则使用机器人独立停止路径。AimDK 控制权、IMU
标定和 HAL watchdog 限制仍按上文处理。实时 PICO 没有未来动作样本，现有 buffer 使用
已收到的姿态填充不可用的未来参考，因此下文文件 tracking 结果不代表实时遥操性能。

本地离线 XR 测试已覆盖 OLI/X2、1.6/1.8/2.0 米操作者身高：源坐标变换、IK/FK 关节顺序、
动作和手柄消息、X 暂停/恢复、重连、具名 ZMQ 接收、X2 目标速度限制。SDK 在这台 x86 PC
已编译且可导入；本次未连接物理 PICO 头显或 X2 真机。

```bash
# 测试时在 Pico 环境中导入根项目 motion buffer 所需的依赖：
uv pip install --python venv/pico/.venv/bin/python pytest \
  'any4hdmi @ git+https://github.com/EGalahad/any4hdmi.git@5d0ae1915c904c64f2480a055aa1693523b9e4b4'
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py tests/test_pico_hand_grip.py -q
```

## 新增 LAFAN tracking 动作

公共配方 `scripts/lafan_tracking_suite.json` 新增 **walk 步行、aiming 瞄准、dance2 舞蹈**，
均取 subject 1 前 8 秒，BVH 原始 30 Hz，策略重采样为 50 Hz。可复用本地数据，或从
[Ubisoft 源仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)取得
`lafan1/lafan1.zip`，解压到 `lafan1/data` 并保留数据许可。

```bash
gh repo clone limxdynamics/GMR ../GMR-limx -- --depth 1
uv run --no-sync python scripts/lafan_tracking_suite.py --robot x2 \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
# 原键盘启动器可播放生成的真机参考：
# bash scripts/x2/start_proteus.sh outputs/x2/lafan_suite/walk
```

已有 GMR 仓库可直接复用。脚本使用内置社区 LAFAN 标定和准备好的 X2 模型、中立姿态
约束、12 rad/s 关节目标限速、逐帧脚底接触点贴地。明确复现该标定使用的 **原版 Rx(90)
左乘、不对四元数做共轭** 的坐标约定，避免混用 LimX 新版 loader 的不同基变换。
不需要给外部 GMR 打 X2 补丁，也不在 X2 上回放 G1 qpos。两个头部关节保持中立。

数据集位于 `outputs/x2/lafan_suite/{walk,aiming,dance2}`，保存 BVH SHA256、截取区间、
标定/GMR 版本、qpos 名称和贴地信息。修改转换参数后指定新的 `--output`，已有数据会被
复用。视频 `x2_wamseq_lafan_<name>.mp4` 含 1 秒初始策略 Hold 和末帧 Hold，并保存
实际轨迹及 `.metrics.json`。seed=0，物理步长 0.005 秒，策略 50 Hz，测得的世界根位置，
没有弹力带。

2026-11-02 播放阶段、14 个合约跟踪刚体上的结果：

| LAFAN 片段 | 平均世界 / 骨盆局部刚体误差 | 世界根位置末端误差 |
| --- | --- | --- |
| walk1_subject1 | 4.49 / 3.38 cm | 3.72 cm |
| aiming1_subject1 | 3.75 / 3.64 cm | 1.34 cm |
| dance2_subject1 | 4.23 / 3.59 cm | 1.59 cm |

三条 10 秒视频均完整保持站立，最低根高度分别为 0.620 / 0.641 / 0.620 m。加上已有的
relaxed walk 和 Gangnam dance，目前有五类参考。这些结果只验证了相应短片段的仿真；
模型仍保留厂商惯性与训练质量的差异，真机和实时 PICO 效果需另行验证。

## Proteus 原版仿真对照（2026-11-03）

在发现 Q1/T800 的仿真步长问题后，已将本分支与 **未修改** 的 Proteus `wamseq`
`1e0e038` 仿真器对照。两边使用相同的已准备本体模型、共享 WAMSeq 权重和完全相同的
50 Hz 参考数组；参考只打包成原生 `MotionLibrary`，没有重新 retarget。
每条原版动作也通过实际 `deploy/sim2sim.py` CLI 运行。
这里控制了本体模型版本，并未重建缺失的原始训练资产。

seed 为 0，策略 50 Hz。对双方共有的播放帧 1..T-1 各取首个物理采样，在 contract 的
14 个 tracking body 上计算平均世界位置误差。本地保留启动时 1 s 策略 Hold，
原版保留原生启动方式和参考速度；最后一列仅改变本地物理步长。
该统一采样指标与历史视频指标可能因采样时相略有差异，较早 OLI 实验还使用了不同的
根部参考强制设置。本次全部使用 `policy-sim.yaml` 和实测世界根部平移。

| 参考动作 | 本地 5 ms（cm） | 原版 0.625 ms（cm） | 本地 0.625 ms（cm） |
| --- | ---: | ---: | ---: |
| walk | 4.70 | 4.38 | 4.82 |
| aiming | 3.76 | 3.64 | 3.75 |
| dance2 | 4.25 | 4.25 | 4.29 |
| gangam | 8.18 | 8.27 | 8.23 |
| relaxed | 4.87 | 4.60 | 5.06 |

这些参考上 **未发现类似的步长数值不稳定或明显运行时差距**。江南舞在两边的误差均约为
8.2 cm，本地/原版关节加速度 RMS 为 24.02/24.24 rad/s²。
减小本地步长并未改善这些参考上的平均位置精度，因此保留 X2 的现有仿真默认值。
此前记录的官方模型/训练模型 0.16487534 kg 质量差仍保留。

已在 399 个共同物理状态上对照原版的 proprioception、action history 和有效位。
统一使用即时 FK 和单位参考四元数后，task/action 最大差分别为
5.66e-07/3.96e-05，action history 和有效位完全一致。
未经诊断调整的原版 body 变换缓存比 qpos 滞后一个物理步；实际原版 rollout 保留了
这一小幅观测时序差异。

机器可读记录为 `docs/experiments/proteus-x2-runtime-audit.json`，包含源/产物哈希、
准确命令、原版 CLI 结果、三种运行设置和观测诊断。调试脚本及实际物理轨迹/视频归档于
`outputs/proteus_oli_x2_audit/20261103_171600/`。差分检查器能够检出已知有问题的
Q1/T800 轨迹，而 OLI/X2 的九条参考在两种本地步长下均未触发该差距检查。
该检查针对已观测到的数值退化，不是绝对跟踪质量阈值。
本次排查没有修改运行时代码、策略权重或真机/PICO 行为。
