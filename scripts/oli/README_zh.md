# LimX OLI（HU_D04_01）启动脚本

`scripts/oli/` 通过 LimX 官方 Python SDK，在 LimX OLI EDU 人形机器人（HU_D04_01，
Proteus 契约 `limx_oli_edu`）上运行 Proteus WAMSeq Tracker；同时负责准备机器人模型、
转换与重定向动作，以及录制 sim2sim 视频。通用流程（Shaping → Hold → Go → Final Hold、
急停、新机器人检查清单）见 [`../README_zh.md`](../README_zh.md)，本页只介绍 OLI 特有内容。
[English](README.md)

## 状态

| 项目 | 内容 |
| --- | --- |
| 真机 | **未测试。** 仅在仿真中检查了本地导出、观测历史、SDK 传输映射、静止 / 合成挥手 sim2sim、一段 LAFAN1 舞蹈，以及 LAFAN walk / aiming / dance2 套件 |
| `--robot` 别名 | `oli`、`limx_oli_edu`、`hu_d04_01`（`sim2real/config/robots/__init__.py`） |
| 关节数 | 31 个虚拟 pitch/roll（PR）关节，串联模型 `HU_D04_01_vis.xml` |
| Robot I/O | `--robot-io inline --robot-ip <IP>` → `OliRobotIO`（LimX `limxsdk` wheel）；要求 `ROBOT_TYPE=HU_D04_01` |
| Tracker 运行位置 | 任何装有对应架构 LimX SDK wheel、能访问机器人 IP 的电脑（SDK 仿真器用 `127.0.0.1`）。教程给出了 x86_64 与机载 ARM 两种 wheel，均未在真机上运行过 |
| 默认 checkpoint | `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/`（2026-11-03 运行时对照所用的 actor） |
| 策略 YAML | `policy.yaml`：真机，`reference_forcing: true`（SDK 不提供世界系根位置）。`policy-sim.yaml`：使用测得根位置的仿真，`reference_forcing: false` |
| 备选 actor | wsq61 **不是** OLI 默认值，OLI 的结果也不能证明它可用；需另行导出并显式传入其 `policy.yaml` |

## 文件

| 文件 | 用途 |
| --- | --- |
| `start_proteus.sh` | 用键盘控制器在真机 / SDK 仿真器上播放一个 any4hdmi 数据集 |
| `start_pico.sh` | PICO 遥操：`publisher`、`sim`、`policy-sim` 或 `robot` 模式 |
| `prepare_assets.py` | 将官方串联 HU_D04_01 模型复制到 `third_party/limx_oli_edu/model/`，并按 Proteus spec 检查关节、限位、轴向和质量 |
| `convert_motion.py` | Proteus 打包的 OLI 动作库片段 → any4hdmi 数据集（含 FK 校验），或用 `--standing-smoke` 生成静止片段 |
| `make_wave_motion.py` | 生成抬臂 / 挥手 / 放下的合成参考，用于动态冒烟测试 |
| `retarget_lafan.py` | 用 LimX 官方 GMR 无界面地将 LAFAN1 BVH 重定向到 OLI，并做脚底贴地 |
| `render_sim2sim.py` | 录制集成 sim2sim 与参考并排的视频（MuJoCo EGL + `ffmpeg`）；`scripts/x2/render_sim2sim.py` 也复用它 |

其他相关文件：

- 机器人配置：`sim2real/config/robots/oli.py`（`OLI_CFG`、`OLI_SDK_JOINT_NAMES`、`OLI_MJCF_PATH`）。
- Robot I/O：`sim2real/rl_policy/robot_io/oli.py`；分发逻辑在 `sim2real/rl_policy/robot_io/factory.py`。
- 仿真模型调整（只保留地面接触、关节被动阻尼清零）：`sim2real/sim_env/utils/mjcf.py`。
- 策略导出：`scripts/export_proteus_onnx.py`（生成 `policy.yaml` 与 `policy-sim.yaml`）。
- 遥操：`sim2real/teleop/pico_retarget_pub.py`（GMR，目标机器人 `limx_oli_edu`），配置
  `sim2real/teleop/configs/xrobot_to_oli.json`，可用 `scripts/build_xrobot_profile.py` 重新生成。
- 共享工具：`scripts/make_hold_motion.py`、`scripts/lafan_tracking_suite.py`（及 `.json`）、
  `scripts/summarize_proteus_sim2sim.py`。
- 测试：`tests/test_oli.py`、`tests/test_pico_retarget_profiles.py`。
- 记录：`docs/experiments/proteus-oli-runtime-audit.json`。

## 准备

所有命令都在仓库根目录执行。Tracker、仿真器和资源工具使用根目录 `uv` 项目；PICO
发布端使用 `venv/pico`。教程中的命令均带 `uv run --no-sync`。

1. 外部仓库（与本仓库同级）：Proteus `wamseq` worktree（`../Proteus-wamseq`）、
   `limxdynamics/humanoid-description`、`limxdynamics/limxsdk-lowlevel`；LAFAN 与遥操配置
   还需要 `limxdynamics/GMR`（`../GMR-limx`）。
2. 准备模型并导出 actor（二者都是 Git 忽略的运行时产物）：

   ```bash
   uv run --no-sync python scripts/oli/prepare_assets.py \
     --description-root ../humanoid-description --proteus-root ../Proteus-wamseq
   uv run --no-sync python scripts/export_proteus_onnx.py \
     --proteus-root ../Proteus-wamseq --release proteus_wamseq_62 \
     --contract limx_oli_edu --robot oli
   ```

   使用串联模型 `HU_D04_01_vis.xml`，不要用并联连杆的 `HU_D04_01.xml`。
   如需 wsq61，用 `--release wsq61` 导出，输出位于 `checkpoints/proteus/wsq61/limx_oli_edu/`。
3. 在根环境中安装 SDK wheel（以下为 x86_64；机载 ARM 电脑使用 `python3/aarch64`）：
   `uv pip install --no-deps ../limxsdk-lowlevel/python3/amd64/limxsdk-*.whl`
4. 将 `checkpoints/proteus/proteus_wamseq_62/limx_oli_edu/`（ONNX、YAML、契约 JSON）、
   `third_party/limx_oli_edu/` 和动作数据集复制到部署电脑。
5. 两个启动脚本都会导出 `HF_HUB_OFFLINE=1` 与 `HF_HUB_DISABLE_TELEMETRY=1`；手动执行
   其他命令时也请在 shell 中设置。
6. 仅 PICO：按 `docs/getting-started/teleop-x86-64.md` 安装 XRoboToolkit PC Service 与
   Python SDK，然后执行 `uv sync --project venv/pico --no-install-package xrobotoolkit-sdk`
   和 `bash scripts/setup/setup_xrobot_pybind.sh`。在策略电脑上生成中立启动参考：
   `uv run --no-sync python scripts/make_hold_motion.py --robot oli --output outputs/oli/pico_hold`

## 可调项

| 可调项 | 当前值 | 设置位置 |
| --- | --- | --- |
| 关节限位、kp/kd、armature、速度 / 力矩上限、默认姿态 | Proteus `limx_oli_edu` spec（如膝关节 `-0.087..2.583` rad，根高 `0.9282` m） | `sim2real/config/robots/oli.py` 的 `_JOINTS`、`default_qpos` |
| SDK 关节顺序 | 除头部外与 MJCF 相同：SDK 中 `head_pitch` 在 `head_yaw` 之前 | `oli.py` 的 `OLI_SDK_JOINT_NAMES` |
| 命令频率 / 看门狗 | 1000 Hz 发布线程；反馈 / 命令超过 250 ms → 阻尼 | `OliRobotIO.__init__` 默认值（`publish_hz`、`timeout_s`）；factory 只传 `robot_ip`，OLI 不使用 `robot_io_options` |
| PR → AB 转换 | 每个关节都设 `parallel_solve_required=True`（踝和腰必需） | `OliRobotIO._publish` |
| IMU | 直接把 SDK 的 WXYZ 四元数和陀螺仪当作骨盆系，无外参旋转 | `OliRobotIO._on_imu` |
| Shaping 目标 | `shaping_target: motion_first_frame`；OLI 不写 soft / shaping 限位，因此第 0 帧不做截断 | `scripts/export_proteus_onnx.py` 的 `policy_yaml` |
| 根平移 | `reference_forcing: true`（真机）/ `false`（`policy-sim.yaml`） | `scripts/export_proteus_onnx.py` |
| 仿真 | 只保留地面接触、关节被动阻尼为零；`sim_dt` 0.005 s、策略 50 Hz（共享默认值，对照审计认为无需修改） | `sim2real/sim_env/utils/mjcf.py`、`IntegratedSim2SimArgs` |
| 推理 | CPU ONNX Runtime，`SIM2REAL_ORT_NUM_THREADS` 默认 1 | 两个启动脚本 |
| 机器人 / 发布端 IP | 默认 `127.0.0.1` | 启动脚本的位置参数 |

## 使用

### Sim2sim 预演

```bash
# 静止冒烟片段；或用 --library DIR --motion-index N 转换动作库片段
uv run --no-sync python scripts/oli/convert_motion.py --standing-smoke --duration 4 --output outputs/oli/wamseq_smoke
uv run --no-sync python scripts/oli/make_wave_motion.py --output outputs/oli/wave_motion
uv run --no-sync python scripts/oli/render_sim2sim.py --motion-path outputs/oli/wave_motion \
  --duration 10 --output outputs/oli/oli_wamseq_wave_sim2sim.mp4
# 通过 LimX GMR 使用真实 LAFAN1 动作
uv run --no-sync python scripts/oli/retarget_lafan.py --gmr-root ../GMR-limx \
  --bvh ../ubisoft-laforge-animation-dataset/lafan1/data/dance1_subject1.bvh \
  --duration 8 --output outputs/oli/lafan_dance1
uv run --no-sync python scripts/lafan_tracking_suite.py --robot oli \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
```

`render_sim2sim.py` 默认使用 `proteus_wamseq_62/limx_oli_edu/policy-sim.yaml`，可用
`--policy-config` 覆盖。输出目录必须是新的：转换脚本拒绝覆盖已有数据集。

### 真机（或 LimX SDK 仿真器）运行

先释放官方站立 / 行走 / mimic 控制器，并吊起机器人。

```bash
# start_proteus.sh <动作> [robot-ip=127.0.0.1] [policy.yaml]
bash scripts/oli/start_proteus.sh /path/to/oli/any4hdmi_dataset ROBOT_IP
```

键盘：`i` 用 10 秒 Shaping 到第 0 帧，`]` 进入策略 Hold，空格播放 / 暂停，`o` 阻尼，
Ctrl+C 退出。OLI 不使用 Unitree 手柄控制器。

### PICO 遥操

```bash
bash scripts/oli/start_pico.sh publisher --actual-human-height 1.70   # 发布 PC（venv/pico）
bash scripts/oli/start_pico.sh sim                                     # 根项目仿真器
bash scripts/oli/start_pico.sh policy-sim 127.0.0.1                    # Tracker，ZMQ robot I/O
# 真机时替换后两条：在策略 / SDK 电脑上执行
bash scripts/oli/start_pico.sh robot PUBLISHER_IP ROBOT_IP
```

按键：**A** Shaping 到中立姿态（10 秒），**A+B** 进入策略 Hold，**X** 开始 / 暂停实时
参考（暂停时发布端在当前 XY / 朝向回到中立姿态），**单独按 B** 阻尼。两台电脑间需放通
TCP 28701（动作）和 5592（按键）。额外的 Tracker 参数（如 `--policy-config`、
`--motion-path`）放在 IP 位置参数之后。评估自由平衡前，先在仿真器 GUI 中关闭弹性绳。

### 关停

`o`（键盘）或 **B**（PICO）切换阻尼；Ctrl+C 退出。正常关闭时 `OliRobotIO` 会发送一次
阻尼命令。请随时准备好 SDK 的独立停止手段。

## 安全须知与已知问题

- **真机未测试。** 验证期间没有连接 OLI 或 PICO 头显。
- 不能有两个命令发布者同时控制机器人：先释放官方控制器。
- 1 kHz 发布器是 Python 线程，不是硬实时控制器；需在真机上验证时序及 SDK / 固件兼容性。
- 超时或 `zero` 模式下的阻尼是以 `kp=0`、保留 `kd` 保持测得位置；无法应对进程被强杀或断电。
- PICO 身体数据流中断时，接收端保持最后一帧参考，不会自动切到阻尼。
- `policy-sim.yaml` 需要测得的世界系根位置，不要在真机上使用。
- 仿真中 LAFAN walk / aiming / dance2 在 14 个跟踪刚体上的平均世界系误差约 12–14 cm，
  并有明显姿态 / 朝向漂移；这不代表可以上真机。

## 详细文档

- English：[`docs/tutorials/proteus-oli.md`](../../docs/tutorials/proteus-oli.md)
- 中文：[`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-oli.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-oli.md)
- 通用流程：[`../README_zh.md`](../README_zh.md)
