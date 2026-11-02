---
title: 在 LimX OLI 上部署 Proteus WAMSeq
slug: /tutorials/proteus-oli
---

# 在 LimX OLI 上部署 Proteus WAMSeq

本适配面向 **HU_D04_01 / limx_oli_edu，31 个虚拟 PR 关节**，使用 Proteus 的
`wamseq` 分支、本仓库的 Tracking / 集成 sim2sim 运行时和
[LimX 官方 SDK](https://github.com/limxdynamics/limxsdk-lowlevel)。
已验证本地导出、历史观测、通信映射、静止/合成挥手仿真和一个 LAFAN1 舞蹈片段；真机仍需验证。

## 准备模型和策略

以下命令在 sim2real 根目录运行，使用已有根项目 `.venv`（Python 3.10、MuJoCo 3.11、
ONNX、CPU ONNX Runtime）。使用 `--no-sync` 是因为当前仓库已有的 ARM GPU 依赖可能
导致全平台依赖解析失败。

```bash
# 保留原 Proteus 工作区。
git -C ../Proteus worktree add --detach ../Proteus-wamseq origin/wamseq
gh repo clone limxdynamics/humanoid-description ../humanoid-description -- --depth 1
gh repo clone limxdynamics/limxsdk-lowlevel ../limxsdk-lowlevel -- --depth 1

uv run --no-sync python scripts/oli/prepare_assets.py \
  --description-root ../humanoid-description --proteus-root ../Proteus-wamseq
uv run --no-sync python scripts/export_proteus_onnx.py \
  --proteus-root ../Proteus-wamseq --release proteus_wamseq_62 \
  --contract limx_oli_edu --robot oli

# x86_64；机载 ARM 使用 python3/aarch64 下的 wheel。
uv pip install --no-deps ../limxsdk-lowlevel/python3/amd64/limxsdk-*.whl
```

`prepare_assets.py` 将官方串联模型 `HU_D04_01_vis.xml` 和网格复制到
`third_party/limx_oli_edu/model/`，对照 Proteus 训练描述检查 31 个关节、限位、默认
姿态下的关节轴和总质量。也可用 `--asset-root <SCALETRACK_ROBOT_ASSETS>` 从训练
资源目录准备模型。sim2real 的 FK 和集成仿真应使用串联模型，不能替换成含并联连杆的
`HU_D04_01.xml`。

导出结果位于 `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/`，包含
`policy.onnx`、`policy.yaml`、原始机器人合约和 `comparison.json`。
第二个 WAMSeq 模型使用 `--release proteus_wamseq_62_full`；不指定 release 时仍导出
原来的 `proteus_60`。集成 OLI 仿真按训练描述关闭自碰撞和关节被动阻尼。

| 无 batch 维的 ONNX 输入 | 内容 |
| --- | --- |
| `proprioception[10,68]` | 重力、基座角速度、带符号的槽位关节偏移、带符号的速度 × 0.05 |
| `action_history[10,31]` | 每个观测之前的原始槽位动作 |
| `history_valid[10]` | 从旧到新；重置时仅最新观测有效 |
| `task[6,253]` | 锚点坐标系中的参考身体位置和切线/法线旋转，含绝对/相对量和偏移 `[0,1,2,3,4,16]` |

ONNX 固化形态描述，按合约关节顺序输出 `action[31]`，同时保留
`next_slot_action[31]` 用于反馈。共享观测核心每个策略步只推进一次，无效历史观测和
动作均置零。关节目标为 `default + signed_action_scale * action`，使用原训练增益。
由于硬件 SDK 不提供世界系根部位置，`reference_forcing: true` 用参考锚点位置替代
根部平移，因此验证的是局部跟踪。

## 动作与本地仿真

将 OLI 的打包动作库（`index.json` 和 `*.npy`）中的一个片段转换成 any4hdmi 数据集：

```bash
uv run --no-sync python scripts/oli/convert_motion.py \
  --library /path/to/libraries/limx_oli_edu/eval --motion-index 0 \
  --output outputs/oli/clip0
```

转换保留 WXYZ 的 articulation-root 位姿，按名称映射关节，并抽样对比 FK 与原始身体
参考数据。输入应为 50 Hz，输出目录不能已有数据集。已有 OLI any4hdmi 数据集可直接使用。

没有动作库时，可以生成**合成静止参考**来做基础检查：

```bash
uv run --no-sync python scripts/oli/convert_motion.py \
  --standing-smoke --duration 4 --output outputs/oli/wamseq_smoke
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 SIM2REAL_ORT_NUM_THREADS=1
uv run --no-sync python -m sim2real.sim_env.integrated_sim2sim \
  --robot oli --policy-config checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml \
  --motion-path outputs/oli/wamseq_smoke --headless --initial-pause-s 1 \
  --max-runtime-s 6 --seed 0 --root-trajectory-output outputs/oli/root_smoke_hold.npz
```

初始暂停期间策略保持运行，动作结束后保持末帧。加 `--run-once` 可在到达末帧时退出。
静止片段用于检查集成链路，不能代表动态跟踪性能。

可以录制真实仿真与参考姿态的左右对照视频（需要 MuJoCo EGL、PATH 中的 `ffmpeg`
和 Pillow）：

```bash
uv run --no-sync python scripts/oli/render_sim2sim.py \
  --motion-path outputs/oli/wamseq_smoke \
  --output outputs/oli/oli_wamseq_standing_sim2sim.mp4 \
  --label 'Synthetic stationary reference | smoke test only'
```

默认生成 6 秒、25 fps、原速视频：初始保持 1 秒，参考播放 4 秒，随后保持末帧。
同时输出 JPEG 预览与 JSON 元数据。录制动态跟踪时替换为 OLI 动作数据集。

也可以生成并录制合成的动态挥手动作：

```bash
uv run --no-sync python scripts/oli/make_wave_motion.py --output outputs/oli/wave_motion
uv run --no-sync python scripts/oli/render_sim2sim.py \
  --motion-path outputs/oli/wave_motion --duration 10 \
  --output outputs/oli/oli_wamseq_wave_sim2sim.mp4 \
  --label 'Synthetic dynamic reference | raise arm - wave - lower arm'
```

参考动作长 8 秒（抬左臂、挥手、放下），视频前后各保持 1 秒。动作由关节曲线生成，
不属于人体动捕或重定向数据。同配置、seed=0 的集成仿真保存了
`outputs/oli/wave_root.npz` 和 `outputs/oli/wave_trajectory.npz`；播放期间根部高度
保持在 0.9096–0.9119 m，所有已记录模型身体的世界系平均位置误差为 0.0363 m。
这仅验证了该合成上肢动作，更广泛的动作库表现和真机仍需验证。
指标来自同一策略和动作下的 `integrated_sim2sim`，参数为 `--headless
--initial-pause-s 1 --max-runtime-s 10 --seed 0 --root-trajectory-output
outputs/oli/wave_root.npz --trajectory-output outputs/oli/wave_trajectory.npz`。

## 用官方 LimX GMR 重定向 LAFAN1 动捕动作

[LimX 官方 GMR](https://github.com/limxdynamics/GMR/blob/master/README.zh-CN.md)
包含同一串联 HU_D04_01 模型的 `bvh_lafan1_to_oli_edu.json`。
[Ubisoft LAFAN1 仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
提供原始人体 BVH。不能将已重定向到 G1 的关节轨迹直接复制给 OLI。
`limxdynamics/luna-beyondmimic` 附带的舞蹈示例对应 27 关节 Luna，不是 31 关节 OLI。

```bash
gh repo clone limxdynamics/GMR ../GMR-limx -- --depth 1
uv pip install mink daqp loop-rate-limiters
uv run --no-sync python scripts/oli/retarget_lafan.py \
  --gmr-root ../GMR-limx \
  --bvh ../ubisoft-laforge-animation-dataset/lafan1/data/dance1_subject1.bvh \
  --duration 8 --output outputs/oli/lafan_dance1
uv run --no-sync python scripts/oli/render_sim2sim.py \
  --motion-path outputs/oli/lafan_dance1 --duration 10 \
  --output outputs/oli/oli_wamseq_lafan_dance1_sim2sim.mp4 \
  --label 'LAFAN1 dance1_subject1 | official LimX GMR retarget'
```

无界面适配器调用官方 GMR API，使用已准备的机器人模型，每帧按最低脚底碰撞盒对齐
地面，将首帧根部 XY 平移到零。any4hdmi manifest 保留 BVH 帧间隔，运行时重采样到
50 Hz。数据集旁保存原始 GMR qpos、源文件哈希、IK 配置哈希、提交号和地面对齐偏移。
本次使用 GMR **86e9546** 和 BVH 第 0–239 帧（约 8 秒）。视频两侧共享同一相机，
跟随参考与实际根部的中点，因此不会隐藏漂移。

seed=0 的集成仿真（初始保持 1 秒，总时长 10 秒）没有跌倒，播放期间根部高度
0.8526–0.9189 m。所有已记录模型身体的世界系平均位置误差为 **0.2433 m**，最差帧
平均误差 **0.5238 m**，此片段有可见的平移漂移。这是世界系诊断量，不是训练的局部
身体成功判据。原始记录为 `outputs/oli/lafan_dance1_{root,trajectory}.npz`。
生成记录的完整参数为 `integrated_sim2sim --robot oli --policy-config
checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/policy.yaml --motion-path
outputs/oli/lafan_dance1 --headless --initial-pause-s 1 --max-runtime-s 10 --seed 0
--root-trajectory-output outputs/oli/lafan_dance1_root.npz --trajectory-output
outputs/oli/lafan_dance1_trajectory.npz`。

## 通过 SDK 部署

将 `checkpoints/proteus/`、`third_party/limx_oli_edu/` 和动作数据集复制到部署电脑；
这些运行资源不纳入 Git。在根项目环境安装对应架构的 SDK wheel。启动前先释放官方
stand/walk/mimic 控制器，避免两个命令发布者同时控制机器人。首次真机实验应吊装机器人。

```bash
# 官方 SDK 仿真器使用 127.0.0.1，真机填实际 IP。
bash scripts/oli/start_proteus.sh /path/to/oli/any4hdmi_dataset 127.0.0.1
```

启动脚本设置 `ROBOT_TYPE=HU_D04_01`、离线环境变量、CPU 推理、50 Hz 策略频率、
`--robot-io inline`；生成的 YAML 设置 `shaping_target: motion_first_frame`。
键盘：`i` 用 10 秒插值到第 0 帧，`]` 启用策略并保持参考，空格开始/暂停动作，`o` 进入
阻尼模式，Ctrl+C 退出。此入口使用键盘控制器，Unitree 手柄控制器不能作为 OLI 遥控器使用。

`OliRobotIO` 订阅 RobotState 和 IMU，保留 SDK 的 WXYZ 四元数和机体系角速度，将
SDK 的 head-pitch/head-yaw 顺序转换为 MJCF 的 head-yaw/head-pitch 顺序，并以 1 kHz
发布混合控制命令。每个关节均设置 `parallel_solve_required=True`，踝和腰依赖 SDK 的
PR→AB 转换（[官方映射说明](https://github.com/limxdynamics/humanoid-rl-deploy-python/blob/main/doc/parallel_joint_mapping_cn.md)）。
没有关节名时使用官方 HU_D04_01 数组顺序。异常数据会被拒绝；反馈或命令超过 250 ms
未更新，以及 `zero` 模式，均切换为阻尼。正常退出时发送阻尼命令。Python 发布线程不提供
硬实时保证，仍需在硬件上验证时序和 SDK/固件兼容性。

## 本地验证记录（2026-11-02）

来源：Proteus `origin/wamseq` **1e0e038**，官方模型 **d7d6a01**，SDK **0572427**
（wheel 4.1.2），官方 Python 部署仓库 **f799b88**。
模型总质量 52.547359 kg，与训练描述的关节轴/限位误差均为零。两个 WAMSeq actor 分别
使用 64 组 seed=0 随机输入（包含二值有效掩码）比较，封装前后输出误差均为零；ONNX
为 IR 8、opset 17。

以上命令使用 `../Proteus-wamseq` 和官方模型工作区，动作是
`outputs/oli/wamseq_smoke`（200 帧静止参考），seed=0，初始暂停 1 秒，仿真总计 6 秒。
`outputs/oli/root_smoke_hold.npz` 记录末端根部相对位置误差 **0.00713 m**，播放期间根部
高度 **0.9096–0.9105 m**。该记录对应 `proteus_wamseq_62`，不代表 full actor 或真机结果。

另外以 seed=0，对照源仓库 `ObservationBuilder` 和 `ActingWindow` 比较 3 次重置、
每次 30 个随机状态：本体观测和历史窗口误差为零，任务观测最大误差 1.33e-15
（`outputs/oli/observation_comparison.json`）。以下 40 项相关回归测试、语法检查和启动
参数解析均通过。

回归检查：

```bash
uv run --no-sync python -m pytest tests/test_oli.py tests/test_proteus_observations.py \
  tests/test_et1_proteus.py tests/test_robot_io.py tests/test_command_sender.py \
  -q
uv run --no-sync python -m py_compile scripts/export_proteus_onnx.py scripts/oli/*.py \
  sim2real/config/robots/oli.py sim2real/rl_policy/robot_io/oli.py
bash -n scripts/oli/start_proteus.sh
```

另外尝试了 SONIC 资源测试；本机缺少 `checkpoints/sonic/{release/g1,release/smpl,v1_1/g1}`，
三个 release 用例因缺少这些文件无法执行。


## PICO 全身遥操

发布 PC 使用独立的 `venv/pico` 环境，策略和仿真使用根项目环境。需要模型的两端均先按
上文准备 OLI 模型和 actor。按[遥操安装](../getting-started/teleop-x86-64.md)安装
XRoboToolkit PC Service 与 Python SDK。首次安装先克隆两个 SDK 仓库，再执行
`uv sync --project venv/pico --no-install-package xrobotoolkit-sdk`，随后运行
`bash scripts/setup/setup_xrobot_pybind.sh`。佩戴腿部追踪器，完成 PICO 全身标定，向发布
PC 开启 body 和 controller 数据流。

内置 `xrobot_to_oli.json` 来自 LimX GMR `86e9546` 的 `smplx_to_oli_edu.json`：保留
OLI 的任务权重、缩放和连杆位置偏移，把人体名称转换为 XRobot 名称，并对旋转偏移
复合 `Ry(pi)`。该坐标转换由 G1 的 SMPL-X / XRobot 对应配置验证，不复制 G1 的关节角。
实时发布器按“标定身高 / 操作者实测身高”归一化人体尺寸。
IK 和 FK 均注册准备好的 **HU_D04_01** 模型，发布 31 个具名关节与模型刚体、WXYZ 四元数。
重新生成配置和启动参考：

```bash
uv run --no-sync python scripts/build_xrobot_profile.py \
  --smplx-config ../GMR-limx/general_motion_retargeting/ik_configs/smplx_to_oli_edu.json \
  --output sim2real/teleop/configs/xrobot_to_oli.json
uv run --no-sync python scripts/make_hold_motion.py --robot oli --output outputs/oli/pico_hold
```

在策略计算机上生成一次中立参考。流式 tracking 的启动整形仍需要动作文件
（`shaping_target: motion_first_frame`）；发布器暂停时使用同一个 RobotCfg 默认姿态。
本地 sim2sim 开 **三个终端**：

```bash
# 1：发布 PC，身高改成操作者的实测米数
bash scripts/oli/start_pico.sh publisher --actual-human-height 1.70
# 2：根项目仿真，打开其 mjviser 地址
bash scripts/oli/start_pico.sh sim
# 3：根项目 tracking，使用 policy-sim.yaml 和真实仿真根位置
bash scripts/oli/start_pico.sh policy-sim 127.0.0.1
```

先在发布器查看 IK。按 **A** 用 10 秒整形至中立姿态，松开按键，再同时按 **A+B**
启用策略 Hold。按 **X** 开始实时参考，X 再按一次回到当前 XY/朝向的中立 Hold。
**单独 B** 切换阻尼。评估自由平衡前在仿真 GUI 关闭弹力带。X 控制参考生成的暂停，
策略仍然运行；A/B 与动作数据走独立通道。

发布 PC 和 OLI 计算机分开时，将第三条命令替换为：

```bash
# 策略 / SDK 计算机：PUBLISHER_IP 是 PC，ROBOT_IP 是 LimX 控制器
bash scripts/oli/start_pico.sh robot PUBLISHER_IP ROBOT_IP
```

脚本设置 `ROBOT_TYPE=HU_D04_01`、HF 离线参数、CPU 推理、50 Hz 策略、
`robot_io=inline`、真机 `policy.yaml`，并同时连接动作 **TCP 28701** 和手柄 **TCP 5592**。
两端放通这两个端口。`policy-sim` 则使用 `robot_io=zmq` 连接 BaseSimulator。
位置参数后可追加 tracking 参数，例如自定义 `--policy-config` 或 `--motion-path`。
同一组端口同时只运行一个发布器/仿真实例。扳机流保持在 5593，此处没有新增 OLI 灵巧手驱动。

body 断流时发布器停止发实时姿态，接收端保持最后参考，**不会自动切换阻尼**。
重连后创建重新对齐的动作段。手柄链路正常时可按 B；网络或进程失效时使用 SDK 独立停止
路径。真机控制权与 IMU 要求仍按上文部署章节处理。

本地使用离线 XR 测试输入验证了各机器人 IK、FK 关节顺序、暂停/运行/重连标记、具名
ZMQ 接收和原有按钮映射，Python SDK 在此 x86 PC 上也已编译并可导入。本次未连接
PICO 头显或 OLI 真机；真实遥操质量仍需操作者标定及有支撑的首次验证。

## 新增 LAFAN tracking 动作

`scripts/lafan_tracking_suite.json` 新增 **步行 walk、瞄准 aiming、舞蹈 dance2**，均取
subject 1 的前 8 秒、原始 30 Hz。这些是真实人体 mocap，经官方 LimX GMR 直接重定向到
OLI，运行时重采样到 50 Hz。可复用本地 LAFAN 数据，也可从
[Ubisoft 源仓库](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)取得
`lafan1/lafan1.zip`，将 BVH 解压至 `lafan1/data`，保留数据许可。

```bash
# 重新导出，同时生成仿真 policy-sim.yaml 和真机 policy.yaml
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python scripts/export_proteus_onnx.py \
  --proteus-root ../Proteus-wamseq --release proteus_wamseq_62 --contract limx_oli_edu --robot oli
uv run --no-sync python scripts/lafan_tracking_suite.py --robot oli \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
# 使用原键盘启动器播放任一真机参考：
# bash scripts/oli/start_proteus.sh outputs/oli/lafan_suite/walk ROBOT_IP
```

动作目录为 `outputs/oli/lafan_suite/{walk,aiming,dance2}`，各自保存 BVH SHA256、截取
区间、GMR 提交和配置哈希、qpos 名称、脚底贴地信息。脚本会复用已有数据集，修改重定向
设置后请指定新的 `--output`。同一父目录保存 `oli_wamseq_lafan_<name>.mp4`、根轨迹、
全身轨迹、误差 JSON 以及实际策略/动作路径。每条视频 10 秒，包含 1 秒初始策略 Hold
和末帧 Hold；seed=0，策略 50 Hz，物理步长 0.005 秒，使用测得的世界根位置、无弹力带。

2026-11-02 在播放阶段、14 个合约跟踪刚体上的结果：

| LAFAN 片段 | 平均世界 / 骨盆局部刚体误差 | 世界根位置末端误差 |
| --- | --- | --- |
| walk1_subject1 | 13.29 / 16.61 cm | 13.26 cm |
| aiming1_subject1 | 12.35 / 17.29 cm | 11.05 cm |
| dance2_subject1 | 13.74 / 16.91 cm | 18.11 cm |

三条都完整保持站立，最低根高度分别为 0.864 / 0.894 / 0.879 m。姿态和朝向有可见偏差，
这些是验证参考，不代表真机就绪。`policy-sim.yaml` 关闭 reference forcing 以匹配训练
仿真；没有世界定位的真机仍使用 `policy.yaml`。

```bash
# 无需连接 SDK 的集成测试也会导入根项目的 motion buffer 依赖：
uv pip install --python venv/pico/.venv/bin/python pytest \
  'any4hdmi @ git+https://github.com/EGalahad/any4hdmi.git@5d0ae1915c904c64f2480a055aa1693523b9e4b4'
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py tests/test_pico_hand_grip.py -q
```
