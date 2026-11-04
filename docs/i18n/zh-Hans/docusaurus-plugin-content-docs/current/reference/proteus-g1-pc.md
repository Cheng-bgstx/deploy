# Proteus G1 外接电脑部署

PC 上运行 50 Hz 的 CPU ONNX Tracking 和 200 Hz 的 Python SDK2 Bridge。Tracking 从本地缓存读取 NPZ 参考动作；G1 通过 DDS 接收关节位置、速度目标、前馈力矩和 PD 增益。沿用现有 Python 3.10、SDK2、CycloneDDS 和 ZMQ 环境，无需 aarch64 inline 路径的 `unitree_interface`。

启动器默认将 ONNX、OpenMP 和 BLAS 限制为一个 CPU 线程。未限制的 ONNX 线程池可能在 50 Hz 推理间隔期间占用多个核心，与 DDS 回调竞争。之前的日志也记录过一次 174 ms 的指令发送延迟。启动器现将统一 I/O 超时窗口从 100 ms 改为 250 ms，短时间隔不会锁定停止。这两项设置在新进程启动时生效，现有进程不会读取启动脚本的修改。真机测试期间不要并行运行策略基准测试或仿真。

模型放在 `checkpoints/proteus/proteus_60/unitree_g1/`。保留导出的 `policy.yaml`，单独生成部署配置：

```bash
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$PWD"
export SIM2REAL_PYTHON=/absolute/path/to/existing/.venv/bin/python
"$SIM2REAL_PYTHON" scripts/g1/prepare_proteus_deployment.py
```

模型和动作产物不进入 Git。导出的 actor 已就绪，并已缓存固定版本的源动作和 G1 MJCF 后，可一次性生成向前走片段：

```bash
"$SIM2REAL_PYTHON" scripts/g1/prepare_proteus_motion.py
```

脚本以 30 Hz 原样复制源动作第 90–330 帧（含两端），不重采样、不修改 qpos，且拒绝覆盖已有产物。默认片段目录不存在时，启动器也会离线生成。部署前先缓存源动作和 MJCF；若已有目录不完整，应检查后处理，不能静默覆盖。

`policy-pc.yaml` 保留 `proteus.reference_forcing: true`，塑形到双臂自然下垂的默认站姿，将腰部 roll/pitch 的增益设为 kp=100、kd=2.5（yaw 保持 100/2），允许目标超出关节限位最多 1 rad（Bridge 接受同一范围），并将目标变化率限制为 15 rad/s。Proteus 训练时不裁剪目标，ankle 目标会超出关节范围约 0.8 rad；原来向内收 5% 的裁剪在大部分行走步里截掉了 ankle 目标，导致仿真中横向漂移和 Hold 时摔倒。原始 actor 和观测协议保持不变。初始手臂目标与之前部署的默认站姿一致。第二次 A 用 2 秒平滑过渡到参考起始姿态（包含稳定等待），之后才重置历史并进入 Hold。直接从站姿切入策略，以及先放下龙门架再进入 Hold 的流程，在仿真中失败，因此要等 Hold ready 后再着地。

使用实际 Tracking 控制器进行无 DDS、无真机的仿真：

```bash
"$SIM2REAL_PYTHON" scripts/g1/validate_proteus_deployment.py \
  --policy-config checkpoints/proteus/proteus_60/unitree_g1/policy-pc.yaml \
  --npz-playback-limit-s 3
```

仿真覆盖提前按键拒绝、塑形、Hold、理想龙门架下降、动作回放和 Select 锁定停止。固定 root 仅近似吊装保护，不模拟绳索动力学。Select 不能维持站立，稳定性指标截至 Select 前。`tests/test_g1_bridge_safety.py` 使用模拟 SDK 通道和真实本机 ZMQ，验证异常或乱序指令、重复 tick、超时和停止优先级。

接入真实 G1 状态与 ONNX 推理进行只读预检：

```bash
G1_INTERFACE=enp5s0 PROTEUS_READ_ONLY=1 scripts/g1/start_proteus_pc.sh
```

随后启动受保护的真机部署：

```bash
G1_INTERFACE=enp5s0 scripts/g1/start_proteus_pc.sh
```

等待 Bridge 日志出现 `Tracker and LowState ready`。启动时不发送 DDS 电机指令，也不释放高层模式。只有 Tracker 正常输出后，新的遥控器 A 按键和 Tracker 新的 init 指令共同请求接管。机器人保持吊装保护，现场操作员持遥控器和硬件急停。

| 遥控器 / 操作 | 行为 |
| --- | --- |
| 第一次 A | 释放高层模式，在龙门架保护下用 10 秒塑形到双臂下垂的默认站姿。 |
| 塑形完成后第二次 A | 用 2 秒平滑过渡到参考姿态，随后重置历史和航向并进入 Hold；仍受支撑时等待 Hold ready。 |
| 降低龙门架 | Hold 已工作后让双脚着地，继续保留保护。 |
| Start | 播放或暂停参考动作；启动器默认原速向前走 8 秒，然后保持，策略控制继续运行。 |
| Select | 最高优先级、锁定的软件停止；目标跟随实测关节角，保留 PD 增益，主动站姿支撑会丢失。 |

Select 或故障后必须重新走 A → A → Start。达到回放限制后，Start 不能延长片段；重新塑形会重置参考时间。`PROTEUS_PLAYBACK_LIMIT_S` 可改变新会话的回放上限，但不能超过所选动作自身的长度。

启动器现默认使用 `PROTEUS_CLIP=walk_straight_short`（`checkpoints/proteus/proteus_60/unitree_g1/motions/walk_straight_short/`，回放上限 5.1 秒），通过 `/home/galbot/Documents/Proteus-g1/start-pc.sh` 启动时同样生效。它取自 `walk3_subject1` 第 6670–6823 帧：从站立出发，以 0.52 m/s 直线向前走 2.2 m，停在双脚支撑的站姿。在数据集 12 个 walk 动作里扫描到 4 段"站立 → 直线向前走 → 站立"的片段；WAM 仿真中（local forcing，含 0.15 m/s 扰动）这一段平均横向漂移 0.05 m（最大 0.15 m），`walk_forward_to_stand` 为 0.53 m，均未摔倒。原地舞蹈片段都从站立开始、以站立结束：`PROTEUS_CLIP=dance_in_place`（`dance1_subject3` 第 345–525 帧，6 秒，脚着地，手臂动作最丰富）和 `dance_steps`（`dance1_subject3` 第 165–405 帧，8 秒，两次抬脚）。它们来自对 8 个 dance 动作的扫描（活动范围 0.6 m 以内、没有腾空、每一秒都有手臂动作；之前选的 `dance_arms` 前 6 秒几乎不动，已删除），在加入实测关节摩擦和 30 ms 指令延迟的仿真中均未摔倒，但 `dance_steps` 结束后的 Hold 位移最大 0.55 m。`PROTEUS_CLIP=walk_forward_to_stand` 选择较长的走路片段（回放上限 8.63 秒），该片段保留 `walk1_subject1` 第 3–11.63 秒（第 90–349 帧）的原速动作，跳过开头准备段，结束在人已停稳、双脚平放并齐的时刻，早于 12.2 秒开始的转身。Hold 会冻结最后一帧参考，所以片段必须以站立结束：原来的 `walk_forward_original`（第 3–11 秒）结束时右脚正在迈步，Hold 期间右脚一直离地约 3 cm，两脚前后错开 12 cm。用 WAM actor 仿真，新片段播完后 10 秒 Hold 双脚平放（local forcing 下 pelvis 漂移 0.2 cm/s），0.15 m/s 扰动后同样如此。可用 `prepare_proteus_motion.py` 重新生成。`PROTEUS_MOTION_PATH` 可为预检和 Tracking 覆盖参考动作。Local forcing 在行走中仍横向漂移约 0.5 m，该片段尚未在真机上测试。播放和 Hold 期间均保留龙门架保护。修改在下次启动生效，不改变正在运行的会话。

启动器默认 `PROTEUS_IO_TIMEOUT_S=0.25`，为 Bridge（`--timeout-s`）和 Tracking（`SIM2REAL_IO_TIMEOUT_S`）设置相同窗口。窗口内，Tracking 使用最近有效反馈继续正常推理和参考动作播放；短时指令间隔内，Bridge 以 200 Hz 重发最近有效指令。新鲜更新到达后立即继续，不冻结目标、不锁定停止代次，也不要求按 A。重复的 LowState tick 不会重置反馈新鲜度时钟。这是有限窗口内的运行容忍，并不证明真实步态能承受 250 ms 的观测或指令延迟。

反馈或指令中断超过窗口后，默认不再让机器人软倒：`PROTEUS_HOLD_THROUGH_GAPS=1`（启动器默认，Bridge `--hold-through-gaps`）时，Bridge 继续发送最后一组有效 PD 目标，Tracking 暂停推理和参考时钟，等新的 LowState 回来后自动继续控制，不需要重新按 A；中断后迟到或重复的指令会被丢弃。中断超过 1 秒，或 LowState tick 倒退，仍会锁定阻尼，需要重新 A → A：2026-11-04 在会话运行中重启机器人（静默 117 秒），Bridge 在刚开机的机器人上恢复了旧会话约 3 秒才被停掉。重启机器人前请先停止启动器。2026-11-04 行走 4.6 秒时出现 343 ms 的 LowState 接收中断（机器人 tick 仍在增长，说明样本没有送到电脑上的进程），当时锁定 `kp=0` 阻尼导致摔倒。保持目标并不能根治：仿真中行走期间任何超过约 0.3 秒的"失明"（保持目标或用旧状态推理）大多仍会摔，0.15 秒的中断可以撑过；站立 Hold 可以撑过。`PROTEUS_HOLD_THROUGH_GAPS=0` 恢复锁定纯阻尼（`kp=0`，`kd` 使用机器人配置）。Bridge 现在在 200 Hz 主循环里解码 LowState（DDS 回调只保存最新样本，不加锁），运行在大核 0-3、6-15 上，并记录每次超过 30 ms 的接收间隔和对应的机器人 tick 跳变；`scripts/g1/lowstate_gap_monitor.py` 作为独立只读接收进程同时运行（`lowstate-monitor.log`）：两边都看到的中断指向机器人或网络，只有 Bridge 记录到的指向 Bridge 进程。异常反馈或指令、模式或 DDS 写入故障，以及正常退出，保留原有故障处理。Select 保留更高优先级的软件停止行为。待机接管前不会发送电机指令。若需其他窗口，在新启动前设置 `PROTEUS_IO_TIMEOUT_S`；允许范围为大于零且不超过一秒。单独启动 Bridge 或 ZMQ I/O 时默认仍为 100 ms，手动启动必须配套设置两端。zero 指令若在实测角 0.05 rad 内，可在机械限位附近用于确认停止。更新后的 Python Bridge 和 Tracking 使用可选的控制代次及停止代次 ZMQ 元数据，需要配套部署。

局部 forcing 没有全局 root 定位，允许漂移。60 秒诊断保持直立，但 root XY 误差达到 3.16 m；修改后的两段 10 秒回放仿真达到 1.35 m；腰部 roll/pitch PD=100/2.5、双臂下垂站姿到参考姿态平滑过渡的最新 3 秒受限回放仿真达到 0.138 m，相对身体误差均值 0.042 m。仿真不等于真机安全验证。网线断开或 PC 故障后，本机 watchdog 无法把停止指令送到机器人，机器人侧通信超时行为和独立硬件急停仍需现场验证。

日志、PID 和 `tracking.npz` 保存到 `outputs/proteus-pc/<timestamp>/`。Ctrl+C 或 SIGTERM 会停止两进程；Bridge 正常退出前短时发送阻尼命令。强杀进程、断网或退出 Python 均不能当作硬件急停。

退出时限制原生 SDK 通道销毁的等待时间；即使析构超时需要直接结束进程，也先完成记录保存与电机阻尼收尾。
