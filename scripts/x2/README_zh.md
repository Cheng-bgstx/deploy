# 智元灵犀 X2 启动脚本

`scripts/x2/` 通过 AimDK ROS 2 HAL，在智元灵犀 X2 Ultra（Proteus 契约 `lingxi_x2`，公开
v1.3 模型）上运行 Proteus WAMSeq Tracker；同时负责准备模型、导出策略、转换 / 重定向参考
动作、录制 sim2sim 视频，并提供只读的反馈与系统模式预检工具。通用流程（Shaping → Hold →
Go → Final Hold、急停、新机器人检查清单）见 [`../README_zh.md`](../README_zh.md)，本页只介绍
X2 特有内容。[English](README.md)

## 状态

| 项目 | 内容 |
| --- | --- |
| 真机 | **控制未测试。** 在一台 Ultra 上（2026-11-04）有线只读反馈和只读系统状态查询正常；该机 `Develop_MC` 迁移因未配置被拒绝，因此没有接管控制 |
| `--robot` 别名 | `x2`、`lingxi_x2`、`agibot_x2`（`sim2real/config/robots/__init__.py`） |
| 关节数 | 31（腿 12、腰 3、臂 14、头 2） |
| Robot I/O | `--robot-io inline` → `X2RobotIO`：ROS 2 Humble + AimDK 消息，`/aima/hal/joint/{leg,waist,arm,head}/{state,command}` 与 `/aima/hal/imu/torso/state` |
| DDS | HAL 启动脚本：CycloneDDS 绑定 `10.0.1.2`。系统状态工具：通过 `fastdds_host.xml` 使用 Fast DDS |
| Tracker 运行位置 | 接入机器人有线网络、装有 ROS 2 Humble 与 AimDK 的上位机 |
| 默认 checkpoint | `checkpoints/proteus/wsq61/lingxi_x2/`（所有启动脚本、`check_feedback.py`、`render_sim2sim.py`；`export_policy.py --release` 默认 `wsq61`） |
| 策略 YAML | `policy.yaml`：真机，`reference_forcing: true`，含 `robot_io_options` 和 `soft_joint_pos_limits`。`policy-sim.yaml`：使用测得根位置的仿真，`reference_forcing: false`，不含 soft 限位 |
| 早期结果 | sim2sim 表格和 2026-11-03 运行时对照用的是 `proteus_wamseq_62`，不能证明 wsq61 可用 |

## 文件

| 文件 | 用途 |
| --- | --- |
| `start_proteus.sh` | 用键盘控制器在真机上播放一个 any4hdmi 数据集（inline AimDK I/O） |
| `start_first_hold.sh` | 首次真机 Hold：要求已处于 `Develop_MC`，运行带控制权 / 限位检查的 `check_feedback.py`，然后跟踪静态参考 `outputs/x2/pico_hold` 并录制 |
| `start_pico.sh` | PICO 遥操：`publisher`、`sim`、`policy-sim` 或 `robot` 模式 |
| `check_feedback.py` | 只订阅的 HAL 反馈检查（频率、名称、新鲜度、IMU、关节位置）；可选 `--require-exclusive-control`、`--require-joint-limits` |
| `system_state.sh` / `system_state.py` | 只读查询系统状态；`--target Develop_MC` / `--target Ready` 发起迁移；`--require-develop-mc` 为只读门控 |
| `fastdds_host.xml` | Fast DDS 配置，只允许上位机接口 `10.0.1.2` |
| `prepare_assets.py` | 将厂商 `X2_URDF-v1.3.0/x2_ultra.xml` 准备为 `third_party/lingxi_x2/model/scene_tracking.xml`，套用训练限位 / armature，写出 `validation.json` |
| `export_policy.py` | 导出 actor 及 `policy-sim.yaml`（`--release`，默认 `wsq61`） |
| `convert_reference.py` | sonic-x2 PKL（`relaxed_walk`、`gangam_dance`、`idle_stand`）→ any4hdmi |
| `retarget_lafan.py` | 用内置标定和 `X2Retargeter` 将 LAFAN BVH 重定向到 X2 |
| `render_sim2sim.py` | `scripts/oli/render_sim2sim.py` 的 X2 封装（默认 wsq61 的 `policy-sim.yaml`） |
| `summarize_sim2sim.py` | 根据渲染视频旁的 `.trajectory.npz` / `.root.npz` 计算指标 |
| `build_pico_profile.py` | 基于 TaiE64/AgibotX2-Retargeter 重新生成 `sim2real/teleop/configs/xrobot_to_x2.json` |

其他相关文件：

- 机器人配置：`sim2real/config/robots/x2.py`（`X2_CFG`、`X2_AIMDK_GROUPS`、`X2_VENDOR_LIMITS`、`x2_deploy_joint_limits`）。
- Robot I/O：`sim2real/rl_policy/robot_io/x2.py`（`X2StateCommand` 编解码、`X2RobotIO`）；分发在 `robot_io/factory.py`。
- 策略 YAML 生成：`scripts/export_proteus_onnx.py`（X2 的 `robot_io_options`、`soft_joint_pos_limits`）。
- 仿真模型调整：`sim2real/sim_env/utils/mjcf.py`。
- 遥操：`sim2real/teleop/x2_retarget.py`、`sim2real/teleop/pico_retarget_pub.py`、
  `sim2real/teleop/configs/{xrobot_to_x2.json,bvh_lafan1_to_x2.json,X2-RETARGET-LICENSE}`。
- 共享工具：`scripts/make_hold_motion.py`、`scripts/lafan_tracking_suite.py`、`scripts/summarize_proteus_sim2sim.py`。
- 测试：`tests/test_x2.py`、`tests/test_pico_retarget_profiles.py`。
- 记录：`docs/experiments/proteus-x2-runtime-audit.json`、`docs/experiments/proteus-wsq61.json`。

## 准备

在仓库根目录执行。Tracker、仿真器和工具使用根目录 `uv` 项目（`uv run --no-sync`）；PICO
发布端使用 `venv/pico`。

1. 外部仓库：Proteus `wamseq` worktree（`../Proteus-wamseq`）、`AgibotTech/agibot_x2_urdf`
   （`../agibot_x2_urdf`）；参考动作需要 `meetsitaram/sonic-x2`（`../sonic-x2-reference`）；
   LAFAN 需要 `limxdynamics/GMR`（`../GMR-limx`）。
2. 准备模型并导出默认（wsq61）策略，二者都被 Git 忽略：

   ```bash
   uv run --no-sync python scripts/x2/prepare_assets.py \
     --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
   uv run --no-sync python scripts/x2/export_policy.py --proteus-root ../Proteus-wamseq
   ```

   使用 `--release proteus_wamseq_62` 时输出到 `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/`。
3. 上位机：ROS 2 Humble 及与机器人固件匹配的 AimDK 消息包。自行 source 的启动脚本默认使用
   `third_party/aimdk_x2_ros_v082/install/setup.bash`（可用 `X2_AIMDK_SETUP` 覆盖）；
   `start_proteus.sh` 和 `start_pico.sh robot` 需要先在当前 shell 中 source ROS 2 + AimDK。
4. 网络（仅为一台 Ultra 上的实测，不是通用要求）：带 VLAN 200 标签，上位机 `eno1.200`
   上配置 `10.0.1.2/24`，机器人 `10.0.1.40`。上位机地址写死在 `fastdds_host.xml` 和
   `start_first_hold.sh` 的 `CYCLONEDDS_URI` 中，地址不同时两处都要改。`ROS_DOMAIN_ID=0`。
5. 将 `checkpoints/proteus/wsq61/lingxi_x2/`（ONNX、YAML、契约 JSON）、
   `third_party/lingxi_x2/` 和动作数据集复制到上位机。
6. 所有启动脚本都会导出 `HF_HUB_OFFLINE=1` 与 `HF_HUB_DISABLE_TELEMETRY=1`；手动执行命令时
   也请在 shell 中设置。
7. PICO / 首次 Hold：在策略电脑上生成中立参考：
   `uv run --no-sync python scripts/make_hold_motion.py --robot x2 --output outputs/x2/pico_hold`。
   PICO 发布端按 `docs/getting-started/teleop-x86-64.md` 配置。

## 可调项

| 可调项 | 当前值 | 设置位置 |
| --- | --- | --- |
| 训练关节限位、kp/kd、armature、默认姿态 | Proteus `lingxi_x2` spec，根高 `0.6735` m | `sim2real/config/robots/x2.py` 的 `_JOINTS` |
| 厂商（物理 v1.3）限位 | 比训练限位更窄的有：`waist_yaw`（上限 2.2078）、腕 pitch（±0.5236）、腕 roll 外侧限位（1.5097，训练为 1.571）、`head_yaw`（±0.349） | `X2_VENDOR_LIMITS` |
| 真机目标截断 | 训练限位与厂商限位的交集，绝不放宽 | `x2_deploy_joint_limits`，在 `X2StateCommand.write_command` 中应用 |
| Soft 限位（Shaping 截断） | `soft_joint_pos_limits` 取同一交集，仅在 `policy.yaml` 中 | `scripts/export_proteus_onnx.py` |
| `robot_io_options` | `publish_hz: 500`、`timeout_s: 0.25`、`head_pitch_enabled: false`、`imu_topic: /aima/hal/imu/torso/state`、`imu_to_pelvis_wxyz: [1,0,0,0]` | `policy.yaml`（由 `scripts/export_proteus_onnx.py` 生成） |
| IMU 外参 | 单位旋转对应模型；真机必须使用出厂标定 | `imu_to_pelvis_wxyz` |
| 头部 pitch | 禁用：`head_pitch_joint` 的所有命令字段均发送 0 | `head_pitch_enabled` |
| Shaping 目标 | `motion_first_frame` | `policy.yaml` / `policy-sim.yaml` |
| 预检限位余量 | 超出部署限位 0.01 rad 即报错 | `check_feedback.py --require-joint-limits` |
| 仿真 | 只保留地面接触、被动阻尼为零，`sim_dt` 0.005 s，策略 50 Hz | `sim2real/sim_env/utils/mjcf.py`、`IntegratedSim2SimArgs` |
| 遥操关节速度上限 | 相邻目标之间 12 rad/s | `X2Retargeter.max_joint_speed` |

## 使用

### Sim2sim 预演

```bash
uv run --no-sync python scripts/x2/convert_reference.py --clip gangam_dance --output outputs/x2/gangam_dance
uv run --no-sync python scripts/x2/render_sim2sim.py --motion-path outputs/x2/gangam_dance \
  --duration 11.2 --output outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/lafan_tracking_suite.py --robot x2 \
  --lafan-data ../ubisoft-laforge-animation-dataset/lafan1/data --gmr-root ../GMR-limx --render
```

### 真机预检与接管

```bash
# 按 start_first_hold.sh 设置 ROS 环境（CycloneDDS 绑定 10.0.1.2），然后：
uv run --no-sync python scripts/x2/check_feedback.py --duration 10 --output outputs/x2/feedback.json
bash scripts/x2/system_state.sh                       # 只读查询
bash scripts/x2/system_state.sh --target Develop_MC  # 仅当机器人已配置该模式
bash scripts/x2/start_first_hold.sh                  # 静态 Hold，记录到 outputs/x2/first_hold_<time>.npz
bash scripts/x2/start_proteus.sh /path/to/x2/any4hdmi_dataset [policy.yaml]
```

键盘：`i` 用 10 秒 Shaping 到第 0 帧，`]` 进入策略 Hold，空格播放 / 暂停，`o` 阻尼，
Ctrl+C 退出。Tracking 一启动就发送阻尼命令，不必等按键。

### PICO 遥操

```bash
bash scripts/x2/start_pico.sh publisher --actual-human-height 1.70   # venv/pico
bash scripts/x2/start_pico.sh sim
bash scripts/x2/start_pico.sh policy-sim 127.0.0.1
bash scripts/x2/start_pico.sh robot PUBLISHER_IP                      # 真机，替换后两条
```

按键：**A** Shaping 到中立姿态（10 秒），**A+B** 进入策略 Hold，**X** 开始 / 暂停实时参考，
**单独按 B** 阻尼。放通 TCP 28701 和 5592；OLI 与 X2 的 PICO 默认端口相同，同一主机上一次
只能运行一个。

### 关停

停止 Tracking（先 `o`，再 Ctrl+C）；正常退出时后端连续发送 5 次阻尼命令。保持支撑，然后用
`bash scripts/x2/system_state.sh --target Ready` 恢复原生控制（可能自动进入 Business）。

## 安全须知与已知问题

- **真机控制未测试。** v0.8.2 反馈格式匹配并不能确认固件版本，也不提供电机故障监测（该
  格式没有 `error_code` 字段）。
- 已测试的机器人未配置 `Develop_MC`；不要猜测其他模式名，也不要在原生 MC 占有 HAL 时启动 Tracking。
- 预检只是启动检查，不是控制权互斥锁。
- HAL 没有命令超时保护：进程被强杀或断电时会保留最后一条命令。请保持吊装支撑和独立停止手段。
- 先核对 IMU 轴向与骨盆实际倾斜、编码器零位；不要为绕过拒绝而放宽限位。本适配不覆盖其他
  X2 修订版本（如 v1.4 模型）。
- `policy-sim.yaml` 需要世界系定位，不要在真机上使用。
- PICO 身体数据流中断时保持最后一帧参考，不会切到阻尼。

## 详细文档

- English：[`docs/tutorials/proteus-x2.md`](../../docs/tutorials/proteus-x2.md)
- 中文：[`docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2.md`](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2.md)
- 通用流程：[`../README_zh.md`](../README_zh.md)
