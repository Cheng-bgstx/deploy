# 智元灵犀 X2 启动脚本

[English](README.md) · 通用流程（Shaping → Hold → Go → Final Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

X2 在自身的运控主板（RK3588，aarch64，CPython 3.10）上运行 Proteus Tracker，由电脑通过 SSH 操作，布局与
Galbot ET1 相同：`aimdk_bridge.py` 独占 AimDK ROS 2 HAL，并在本地提供 ZMQ LowState/LowCmd 协议，指令由
`BridgeSafety` 仲裁；共用的 Tracker（`sim2real/rl_policy/tracking.py`）在旁边以 50 Hz 做 CPU ONNX 推理；
第二个 SSH 会话中的 ET1 操作键盘发送 `a`、`a`、`s` 和空格。这条机载路线是真机上实际运行过的路线，也是本文档
唯一介绍的真机路线。

## 状态

| 项目 | 内容 |
| --- | --- |
| 真机 | **2026-11-12 在龙门架上试验**，机器人为 X2 Basic（固件 v0.8.2.16），使用 `proteus_wamseq_62` 和 `policy-pc-damped.yaml`：接管、整形和 Hold（分别站立 23 秒和 42 秒），`21_wave_right` 手臂和腰部无振荡，`15_walk_short` 完整播放并站立停下。每次运行中绳子都承担了大部分重量：**尚未完成经验证的独立站立运行。** 详见[机载教程](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2-onboard.md) |
| `--robot` 别名 | `x2`、`lingxi_x2`、`agibot_x2`（`sim2real/config/robots/__init__.py`） |
| 关节数 | 31（腿 12、腰 3、手臂 14、头 2）；头部由 overlay 锁定 |
| Robot I/O | `--robot-io zmq`（127.0.0.1）⇄ `aimdk_bridge.py`（500 Hz）⇄ `/aima/hal/joint/{leg,waist,arm,head}/{state,command}`、`/aima/hal/imu/torso/state`（即骨盆 IMU） |
| 控制器 | `--controller bridge_joystick`（bridge 以 Unitree 键位转发操作按键） |
| Tracker 运行位置 | X2 主板：bridge 绑定核 5，Tracking 绑定核 7（`isolcpus=5-7`） |
| 默认策略 | `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc-damped.yaml`（45 次扰动仿真 0 次摔倒） |
| 其他策略 | `proteus_60`、`proteus_60_bigdata`（目录结构相同）；`wsq61` 可通过 `export_policy.py --release wsq61` 或显式路径选用 |
| 默认动作 | `outputs/x2/wave/21_wave_right`（站立挥右臂） |

## 文件

机载路线（`scripts/x2/`）：

| 文件 | 用途 |
| --- | --- |
| `deploy_onboard.sh` | 电脑 → 主板：rsync 代码、带 overlay 的 X2 actor、模型、动作、HF 资源和 aarch64 site-packages。不会停止 mc，也不会启动控制器。 |
| `env.sh` | 主板脚本共用：主板 ROS 2 环境、`aimdk_msgs`、`PYTHONPATH`、HF 离线、ORT/BLAS 单线程。 |
| `start_proteus_onboard.sh` | 主板启动脚本：拒绝第二个 supervisor 或控制器，按 CPU 绑定启动 bridge + Tracker，日志写入 `logs/run-<UTC>/`。 |
| `aimdk_bridge.py` | AimDK HAL ⇄ ZMQ bridge，`BridgeSafety`（`max_kd=25`）；就绪后的操作员 A 触发接管。 |
| `release_mc.sh` | 接管命令：`aima em stop-app mc`，然后等待 `mc_main` 退出。 |
| `operator.sh` | 在主板上运行 `scripts/et1/operator_keyboard.py`（`ssh -t`）。 |
| `probe.sh` / `state_probe.py` | 只读：关节/IMU 反馈频率和数值，以及核 7 上的 ONNX 计时。 |
| `watch.sh` / `state_now.py` | 只读实时行（骨盆倾角、腿部误差、膝关节力矩），读取 bridge 的 LowState，运行在核 0-3。 |
| `imu_compare.py` | 只读：结合腰部关节比较 torso 和 chest 两个 IMU 话题。 |
| `make_damped_config.py` | 生成 `policy-pc-damped.yaml`（手臂 kd ×5，腿/腰 kd ×2，kp 不变）。 |
| `make_wave_motion.py` / `make_short_walk.py` | 合成挥臂动作；把 11 号片段截成约 1.4 米、以站立结束的短走。 |
| `eval_sim2sim.py` | 无界面 sim2sim，支持扰动和指令延迟（`--cmd-delay-ms`、`--sim-dt`）。 |
| `bridge_sim_test.py` | 演练：真实的 bridge 和 Tracking 进程对接 MuJoCo，带虚拟龙门架。 |
| `render_delay_compare.py` | 在指令延迟下并排渲染多个 overlay 的 sim2sim 视频。 |

模型、仿真和遥操文件：`prepare_assets.py`（生成 `third_party/lingxi_x2/model/scene_tracking.xml`）、
`export_policy.py`（`--release`，默认 `proteus_wamseq_62`；同时写出 `policy-sim.yaml`）、`retarget_lafan.py`、
`convert_reference.py`、`render_sim2sim.py`、`summarize_sim2sim.py`、`build_pico_profile.py`、`start_pico.sh`
（PICO 遥操作，见下文）。

其他位置的相关文件：

- `sim2real/config/robots/x2.py`：关节、限位、增益、`X2_AIMDK_GROUPS`、`X2_VENDOR_LIMITS`、`x2_deploy_joint_limits`。
- `sim2real/rl_policy/robot_io/x2.py`：`X2StateCommand` 编解码（`head_present`，与 bridge 共用）、`X2RobotIO`（inline 电脑路径，仅 `start_pico.sh robot` 使用；从未在真机上运行）。
- `sim2real/rl_policy/robot_io/bridge_safety.py`：与 G1/ET1 bridge 共用的指令仲裁。
- `scripts/export_proteus_onnx.py`、`scripts/g1/prepare_proteus_deployment.py`（`--robot x2`）、`scripts/make_hold_motion.py`。
- 测试：`tests/test_x2.py`、`tests/test_x2_bridge.py`、`tests/test_pico_retarget_profiles.py`。

## 准备

电脑上一次性完成（根 `uv` 项目）：

1. 网络：机器人的 `10.0.1.0/24` 是**带标签的 VLAN 200**（不带标签的 IP 收不到回复）。在 VLAN 接口上给电脑配置
   `10.0.1.2/24`（`nmcli` 命令见教程），然后执行 `ssh-copy-id agi@10.0.1.40`（运控主板 soc0，用户 `agi`）。
2. 模型和 actor（需要 `../agibot_x2_urdf` 和 Proteus `wamseq` 检出）：

   ```bash
   uv run --no-sync python scripts/x2/prepare_assets.py --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
   uv run --no-sync python scripts/export_proteus_onnx.py --contract lingxi_x2 --robot x2 \
     --checkpoint-dir ../Proteus-wamseq/ScaleTrack/checkpoints/proteus_wamseq_62 \
     --output-root checkpoints/proteus/proteus_wamseq_62
   uv run --no-sync python scripts/g1/prepare_proteus_deployment.py \
     --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy.yaml --robot x2   # -> policy-pc.yaml
   uv run --no-sync python scripts/x2/make_damped_config.py checkpoints/proteus/proteus_wamseq_62/lingxi_x2
   ```

3. 参考动作：`scripts/make_hold_motion.py --robot x2 --output outputs/x2/pico_hold`、
   `scripts/x2/make_wave_motion.py --output outputs/x2/wave/21_wave_right`，可选 LAFAN 片段
   （`retarget_lafan.py`，见教程）以及 `make_short_walk.py --output outputs/x2/lafan/15_walk_short`。
4. 按 `deploy_onboard.sh` 文件头的命令，从 `uv.lock` 一次性生成 aarch64 / CPython 3.10 site-packages 到
   `outputs/x2-onboard-env/site`（`X2_ENV_CACHE`；也可复用由同一 `uv.lock` 解析的 ET1 包）。目录中必须有
   `.lock-<sha256(uv.lock) 前 16 位>`。
5. `~/.cache/huggingface/hub` 中需有 HF 缓存 `models--elijahgalahad--h2_model` 和 `models--elijahgalahad--g1_xmls`。

`deploy_onboard.sh` 要求 `X2_RELEASES`（默认 `proteus_wamseq_62 proteus_60_bigdata proteus_60`）中每个 release
都有 `policy.onnx`、`policy-pc.yaml` 和 `policy-pc-damped.yaml`，且模型、`pico_hold` 和 `21_wave_right` 都存在，
否则拒绝运行。设置 `X2_RELEASES=proteus_wamseq_62` 可只部署默认 actor。

主板上：`agi` 用户主目录下为 `proteus_x2/{repo,site-packages,hf,logs}`。`env.sh` 加载
`/agibot/software/entry/cfg/basic_env.sh` 和 `/agibot/software/hal_ethercat` 中的 `aimdk_msgs`
（`X2_AIMDK_PKG`），并导出 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=<root>/hf`。

## 可调项

| 项目 | 当前值 | 设置位置 |
| --- | --- | --- |
| Overlay | `policy-pc.yaml`：整形到站姿，2 秒进入参考，目标限制在训练 ∩ 厂商范围内缩 5%，目标速度上限 15 rad/s，锁定头部 | `scripts/g1/prepare_proteus_deployment.py` |
| 真机增益 | `policy-pc-damped.yaml`：手臂 kd ×5（1.1 → 5.5），腿/腰 kd ×2（膝 7.7 → 15.3），kp 不变 | `make_damped_config.py`（`--kd-scale`、`--leg-kd-scale`） |
| Bridge kd 上限 | `max_kd=25.0`（`BridgeSafety` 默认 10） | `aimdk_bridge.py` |
| Bridge 频率 / 裁剪 | 500 Hz；目标裁剪到训练 ∩ 厂商范围 | `aimdk_bridge.py`、`x2_deploy_joint_limits` |
| CPU 绑定 | `PROTEUS_BRIDGE_CPU=5`、`PROTEUS_TRACKER_CPU=7`；核 4 运行 EtherCAT HAL，核 6 运行智元 DDS 线程 | `start_proteus_onboard.sh` |
| I/O 超时 | `PROTEUS_IO_TIMEOUT_S=0.2` → `SIM2REAL_IO_TIMEOUT_S` 和 bridge `--timeout-s` | `start_proteus_onboard.sh` |
| 操作按键有效期 | 0.3 秒（`OPERATOR_FRESH_S`） | `aimdk_bridge.py` |
| 接管 | `--takeover-cmd`（`bash release_mc.sh`）、`--takeover-timeout-s 70` | `aimdk_bridge.py` |
| 头部关节组 | `X2_NO_HEAD=0`（默认）：bridge 订阅并下发头部关节组。这台 X2 Basic 的 HAL 会发布头部关节组，读数恒为 0，因此默认值可用。仅当 `probe.sh` 输出 `head : no messages` 时才设置 `X2_NO_HEAD=1` → `--no-head`（不订阅、不下发头部话题） | `start_proteus_onboard.sh` |
| 主板地址 / 目录 | `X2_HOST=agi@10.0.1.40`、`X2_TARGET=proteus_x2` | `deploy_onboard.sh` |

主板启动脚本的选择项：

- `PROTEUS_POLICY_DIR`（默认 `checkpoints/proteus/proteus_wamseq_62/lingxi_x2`）和 `PROTEUS_POLICY_FILE`
  （默认 `policy-pc-damped.yaml`；`policy-pc.yaml` 为训练增益）。YAML 必须设置 `proteus.reference_forcing: true`。
- `PROTEUS_MOTION`：`outputs/x2/wave/21_wave_right`（默认）、`outputs/x2/pico_hold`（仅站立）、
  `outputs/x2/lafan/15_walk_short`（向前短走）、`outputs/x2/lafan/1x_*`（较长的 LAFAN 片段）。
- `PROTEUS_PLAYBACK_LIMIT_S`：参考时钟停在该时刻并保持。
- `PROTEUS_READ_ONLY=1`（可加 `PROTEUS_MAX_RUNTIME_S`）：空跑，不创建指令发布器，不接管。
- `PROTEUS_RUN_DIR`、`PROTEUS_ROOT`：日志目录 / 安装根目录。

## 使用

1. 在电脑上用真实的 bridge 和 Tracking 进程演练。`--lower-before-hold` 对应真机按键顺序；
   `--gantry-gravity 0.22,0.10,-0.97` 复现机器人倾斜悬挂的情况：

   ```bash
   uv run --no-sync python scripts/x2/bridge_sim_test.py --lower-before-hold \
     --motion outputs/x2/wave/21_wave_right --play-s 12
   uv run --no-sync python scripts/x2/eval_sim2sim.py --sim-dt 0.001 --cmd-delay-ms 40 \
     --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy-pc-damped.yaml \
     --motion-path outputs/x2/lafan/15_walk_short --perturbation push60 --out outputs/x2/eval/walk_push.json
   ```

2. 部署（只改动主板上的 `X2_TARGET`）：`bash scripts/x2/deploy_onboard.sh`
3. 只读探测，mc 仍在运行。**把关节读数和机器人实际姿态对照**；接管前先用手把四肢摆到接近站姿：

   ```bash
   ssh agi@10.0.1.40 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 proteus_x2/repo/scripts/x2/probe.sh
   ```

4. mc 运行时空跑（bridge 不创建指令发布器）：

   ```bash
   ssh -t agi@10.0.1.40 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PROTEUS_READ_ONLY=1 \
     proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ```

5. 龙门架上运行。终端 1（启动脚本）、终端 2（操作员）、终端 3（可选监视）：

   ```bash
   ssh -t agi@10.0.1.40 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/operator.sh
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/watch.sh
   ```

   | 步骤 | 按键 | 作用 |
   | --- | --- | --- |
   | 等待 | — | `bridge.log` 出现 `Tracker and LowState ready` |
   | 1 | `a` | 接管（`release_mc.sh`）并整形到站姿 |
   | — | — | 放下龙门架，直到双脚平放、机器人直立，绳子仍收紧（`watch.sh` 显示 `UPRIGHT`） |
   | 2 | `a` | 2 秒过渡到参考；策略接管（Hold） |
   | — | — | 松绳，直到两侧膝关节读数约 +10 到 +20 Nm |
   | 3 | `s` | 播放 |
   | 停止 | 空格或 `x` | 锁存停止（阻尼） |

   如果 `mc_main` 正在运行（例如主板重启后），第一次 `a` 会释放 mc，随后 bridge 锁存阻尼并提示
   "takeover took … s; mc is released, restart the launcher and press a again"。Ctrl-C 后重新启动并按 `a`：
   mc 已停止时接管立即完成。`mc_main` 运行时启动脚本会给出提示。

6. 关机：保持机器人吊在龙门架上，在启动脚本终端按 Ctrl-C；bridge 锁存并发送 0.2 秒阻尼。用
   `aima em start-app mc` 重启原厂控制器。`proteus_x2/logs/run-<UTC>/` 中有 `bridge.log`、`tracker.log`、
   `bridge-state.json` 和 `tracking.npz`。

### PICO 遥操作

```bash
bash scripts/x2/start_pico.sh publisher --actual-human-height 1.70   # venv/pico
bash scripts/x2/start_pico.sh sim
bash scripts/x2/start_pico.sh policy-sim 127.0.0.1
```

`start_pico.sh robot PUBLISHER_IP` 用 inline 电脑路径替代后两条（`--robot-io inline`、`X2RobotIO`、`policy.yaml`，
需已加载 ROS 2 + AimDK 并释放 mc）。**该模式未在真机上测试。**

按键：**A** 整形到中立姿态，**A+B** 策略 Hold，**X** 开始 / 暂停实时参考，**B** 单独按下进入阻尼。
需开放 TCP 28701 和 5592；OLI 与 X2 的 PICO 共用这些端口，每台主机只运行一个。需要 `outputs/x2/pico_hold`。

## 安全注意事项与已知问题

- **HAL 没有看门狗。** bridge 被 SIGKILL 或断电后，最后一条指令持续生效。龙门架和物理急停要随时可及；
  仅靠键盘无法让机器人停下。
- **机器人悬空时绝不要按第二次 `a`。** 2026-11-12 机器人倾斜悬挂（前倾 13°、侧倾 7°），双脚离地，策略猛烈迈步，
  在绳上乱摆并旋转，直到按下空格。请按上面的顺序操作。
- **接管较慢。** 在 X2 Basic 上 `aima em stop-app mc` 用时超过 20 秒；`release_mc.sh` 以 `mc_main` 退出判定成功
  （最多等待 60 秒）。
- **按 `a` 前检查读数。** 在龙门架上处于被动状态时，腿部曾读到髋 yaw ±1.7 rad。整形不会把关节指令推到比按 `a`
  以来到过的位置更远；被拒绝的指令会锁存，并在 `bridge.log` 中记录关节信息。
- **阻尼。** 使用训练增益时手臂以 4.4 Hz 振荡（→ 手臂 kd ×5）；之后髋部以 3.7 Hz 摆动（→ 腿/腰 kd ×2，在 50 ms
  指令延迟的 sim2sim 中找到；真机上随后 23 秒 Hold 中腿部关节速度为 0.01–0.03 rad/s）。真机延迟估计为 40–50 ms，
  可能出现一次恢复性迈步。
- **脚上的承重。** 绳子承担了重量（膝关节约 0 Nm）；`15_walk_short` 走了约 0.7 米（参考为 1.33 米）。
  双脚完全承重的行走尚未测试。
- **GC 卡顿。** 一次垃圾回收曾让 RK3588 上的循环卡住约 600 ms；Tracker 现在调用 `gc.freeze()`。
- **运行中不要订阅 1000 Hz 的 ROS 话题**（关节状态 1000 Hz，IMU 500 Hz）：用 `watch.sh`，它读取 bridge 的
  LowState。不要把任何进程绑到核 4（EtherCAT HAL）。
- **头部。** `start_proteus_onboard.sh` 默认 `X2_NO_HEAD=0`。这台 X2 Basic 的 HAL 会发布头部关节组，读数恒为 0，
  因此默认值可用。仅当 `probe.sh` 输出 `head : no messages` 时才设置 `X2_NO_HEAD=1`。
- **仿真负载。** 31 GB 内存的电脑上最多并行运行 4 个仿真。
- 停止、超过 0.2 秒的指令或状态超时、无效指令都会锁存阻尼；Ctrl-C 会发送阻尼。

## 延伸阅读

- 机载教程：[proteus-x2-onboard.md](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2-onboard.md) ·
  [English](../../docs/tutorials/proteus-x2-onboard.md)
- 模型、sim2sim 与 PICO 遥操作：[proteus-x2.md](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-x2.md) ·
  [English](../../docs/tutorials/proteus-x2.md)
- 通用流程：[../README_zh.md](../README_zh.md)
