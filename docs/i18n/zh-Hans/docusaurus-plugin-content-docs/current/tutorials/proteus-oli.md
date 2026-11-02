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
