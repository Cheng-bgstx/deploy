---
title: 在智元灵犀 X2 上运行 Proteus（机载）
sidebar_position: 7
---

Proteus 策略运行在 X2 的运控主板（RK3588）上，由电脑通过 SSH 操作。流程与 Galbot ET1 相同：bridge 进程负责
机器人收发和停止仲裁，Tracking 进程运行策略，第二个 SSH 会话中的操作键盘发送 a、a、s 和空格。

**已在真机上验证**（2026-11-12）。机器人自报为 **X2 Basic**，固件 v0.8.2.16，吊在龙门架上，使用
`proteus_wamseq_62` 和 `policy-pc-damped.yaml`：

- 接管、整形到站姿、第二次 `a` 进入 Hold；策略分别稳定站立 23 秒和 42 秒（腿和腰关节 0.01–0.03 rad/s，前倾保持在
  0.5–1.4°）；
- `21_wave_right` 播放时手臂和腰部没有振荡（手臂速度 0.38 rad/s，sim2sim 为 0.32 rad/s，前倾 −0.1° 到 +1.3°）；
- `15_walk_short` 完整播放（10 秒）并站立停下。

这些运行中绳子仍承担了机器人大部分重量（见下文“脚上的承重”）；双脚完全承重时的行走尚未测试。接管和各条动作也
用同样的 bridge 与 Tracking 进程在 MuJoCo 中演练过（见“在电脑上演练”）。

## 网络

电脑连接机器人背部网口。机器人的 `10.0.1.0/24` 网段**带 VLAN 200 标签**：ARP 以不带标签的方式应答，但不带
标签的 IP 包没有回应。创建 VLAN 接口，电脑地址为 `10.0.1.2`：

```bash
nmcli connection add type vlan con-name x2-vlan200 ifname enp5s0.200 dev enp5s0 id 200 \
  ipv4.method manual ipv4.addresses 10.0.1.2/24 ipv6.method disabled
nmcli connection up x2-vlan200
```

若激活时报 “Failed to find a compatible device”，把 `vlan.parent` 设为 `enp5s0` 上当前活动连接的 UUID（netplan
会改写接口名）。运控主板为 `agi@10.0.1.40`，先用 `ssh-copy-id agi@10.0.1.40` 装一次公钥。

## 策略与参考动作

```bash
python scripts/export_proteus_onnx.py --contract lingxi_x2 --robot x2 \
  --checkpoint-dir ../Proteus-wamseq/ScaleTrack/checkpoints/proteus_wamseq_62 \
  --output-root checkpoints/proteus/proteus_wamseq_62
python scripts/g1/prepare_proteus_deployment.py \
  --policy-config checkpoints/proteus/proteus_wamseq_62/lingxi_x2/policy.yaml --robot x2
```

`policy-pc.yaml` 整形到默认站姿，用 2 秒进入参考，目标限制在训练 ∩ 厂家关节范围并留 5% 余量，目标速度限制
15 rad/s，并锁定头部。`proteus_60` 和 `proteus_60_bigdata` 以同样方式导出（不加 `--checkpoint-dir`，或使用
`.../proteus_60_bigdata --output-root checkpoints/proteus/proteus_60_bigdata`）。

MuJoCo 模型由 `scripts/x2/prepare_assets.py` 生成（厂家模型 `X2_URDF-v1.3.0`，AgibotTech/agibot_x2_urdf
`575cc6b`）。行走参考取自 LAFAN1 `walk1_subject1.bvh` 中从站立开始的片段，用 GMR 和已标定的 X2 IK 表重定向；
`--hold-s` 在开头加一段首帧静止：

```bash
python scripts/x2/retarget_lafan.py --gmr-root <GMR-limx> --bvh walk1_subject1.bvh \
  --start-s 0 --duration 10 --output outputs/x2/lafan/11_stand_walk_straight
python scripts/x2/retarget_lafan.py --gmr-root <GMR-limx> --bvh walk1_subject1.bvh \
  --start-s 119.8 --duration 9.7 --hold-s 2 --output outputs/x2/lafan/13_stand_walk_c
```

挥手动作是合成的，不是动捕：`scripts/x2/make_wave_motion.py` 让腿、骨盆和腰保持站姿，把右臂向侧前方举起、
小臂竖在头旁，用肩滚转和肘关节左右摆动。脚本会检查每个关节都在部署用的软限位之内（关节峰值速度 2.2 rad/s）：

```bash
python scripts/x2/make_wave_motion.py --output outputs/x2/wave/21_wave_right
```

在龙门架上做短距离向前走，可以用 `scripts/x2/make_short_walk.py` 把 11 号片段缩短到约 1.4 米。脚本在步态相位
最接近的地方，把行走接到片段末尾的减速段上。它在两脚都踩平、重心位于两脚之间的那一帧结束，最后 1 秒把播放速度
平滑降到零，再保持这一帧 3 秒。11 号片段本身结束时后脚跟正在抬起、准备迈下一步，所以如果跟踪器保持这一帧，
机器人会继续往前走。

```bash
python scripts/x2/make_short_walk.py --output outputs/x2/lafan/15_walk_short
```

在 `policy-pc-damped.yaml`、`--sim-dt 0.001`、40–60 ms 指令延迟下，`proteus_wamseq_62` 跑 `15_walk_short` 的
所有测试都没有跌倒（nominal、kp0.8、mass1.1、friction0.5、push60）。有延迟时它会比参考多走约 0.5 米
（共 1.8–1.9 米），之后站稳不动。kp0.8 时停下的瞬间骨盆降到 0.44 米，并后退了一步。用 `--lower-before-hold`
做的 bridge 彩排也通过了。

| 参考 | 源片段 | 动作 |
| --- | --- | --- |
| `outputs/x2/pico_hold` | — | 站立（`scripts/make_hold_motion.py --robot x2`） |
| `11_stand_walk_straight` | 0.0–10.0 s | 站立 3.6 秒，然后行走约 4.8 米 |
| `12_stand_walk_b` | 10.6–20.0 s，静止 2 秒 | 行走约 4 米，略微转向 |
| `13_stand_walk_c` | 119.8–129.5 s，静止 2 秒 | 约 0.6 m/s 行走 |
| `14_stand_walk_d` | 237.0–246.5 s，静止 2 秒 | 约 0.65 m/s 行走，中间举一次手臂 |
| `15_walk_short` | 11 号片段在 5.7 → 8.6 秒处拼接，9.4 秒处停下 | 站立 3.5 秒，向前走 1.4 米，停下，站立 3 秒 |
| `outputs/x2/wave/21_wave_right` | 合成 | 站立，举起右臂以 1 Hz 挥手约 5 秒，再放下 |

### 真机上的阻尼

启动脚本加载 `policy-pc-damped.yaml`（`PROTEUS_POLICY_FILE`），由 `scripts/x2/make_damped_config.py` 生成：在
`policy-pc.yaml` 基础上把手臂关节的 kd 乘 5（1.1 → 5.5），腿和腰关节的 kd 乘 2（膝 7.7 → 15.3），kp 不变。X2 的 bridge
允许 kd 最大 25（其他机器人为 10）。2026-11-12 使用训练增益（手臂 kp 17.5 / kd 1.1）时，
策略接管后 3 秒内手臂出现 4.4 Hz 且幅度不断增大的振荡，腿和骨盆保持平稳。

仿真中让指令比策略看到的状态晚 40 ms 到达电机即可复现（`scripts/x2/eval_sim2sim.py --sim-dt 0.001 --cmd-delay-ms 40`；
主板上仅策略计算就需 16 ms）。站立时手臂关节速度均方根，第一秒 → 最后一秒：

| 额外延迟 | 训练增益 | 手臂 kd ×3 | 手臂 kd ×5 |
| --- | --- | --- | --- |
| 40 ms | 0.21 → 0.48，4.3 Hz | 0.03 → 0.00 | — |
| 50 ms | 1.13 → 3.80 | 0.24 → 0.34 | 0.15 → 0.19 |
| 60 ms | 3.49 → 5.37，跌倒 | 0.41 → 0.45 | 0.23 → 0.36 |

挥手在 50 ms 延迟下，训练增益会跌倒（误差 14.5 厘米），kd ×5 误差 3.1 厘米。没有额外延迟时，kd ×5 在 kp ×0.8、推力和
一步延迟下，站立、挥手、行走误差与训练增益相差不超过 0.2 厘米。厂商的手臂增益（kp 50 / kd 3）在延迟下振荡更严重
（40 ms 时站立 5.2 rad/s）。

真机上手臂 kd ×5 后，手臂保持平静（5 秒内 0.11–0.13 rad/s），但髋和骨盆出现 3.7 Hz、幅度稳定的前后摆动（髋俯仰
±0.08 rad）。仿真在 50 ms 延迟下出现同样的站立腿部振荡（0.88 → 0.72 rad/s）；腿和腰 kd ×2 后消失（0.04 → 0.01），
60 ms 时站立仍平稳（0.05），行走和挥手误差不变（3.8 / 2.2 厘米）。真机上腿和腰 kd ×2 消除了摆动：Hold 的 23 秒内
腿部关节速度 0.01–0.03 rad/s，腰部 0.01 rad/s。

使用 `policy-pc-damped.yaml` 时，挥手在 45 ms 以内的延迟下双脚保持不动（脚移动不超过 6 厘米）。从 50 ms 起会迈出纠正步：
举臂时向前约 40 厘米，只站立时也会迈 16 厘米。真机延迟估计在 40–50 ms，所以吊架上可能出现一次恢复性的迈步。在主板上
增加 ONNX Runtime 线程不会缩短推理（两个隔离核上 38 ms，核 7 单线程 10.4 ms）。

## 仿真检查

`scripts/x2/eval_sim2sim.py` 用运行时自身的 IntegratedSim2Sim 加载 `policy-pc.yaml`（局部跟踪，无世界系定位），
在标称条件和以下扰动下运行：kp 或 kd ×0.8 / ×1.2、质量 ×1.1、地面摩擦减半、一个控制周期的动作延迟、每 3 秒
一次 60 N 骨盆侧推。指标为骨盆坐标系下跟踪刚体的平均误差。2026-11-12 结果（每格 9 次）：

| 策略 | 站立 | 11 | 12 | 13 | 14 | 跌倒 |
| --- | --- | --- | --- | --- | --- | --- |
| `proteus_wamseq_62` | 1.8 / 2.0 | 3.7 / 7.5 | 3.9 / 21.7 | 3.2 / 8.4 | 4.2 / 10.1 | 45 次中 0 次 |
| `proteus_60_bigdata` | 2.3 / 13.8 | 3.8 / 15.4 | 4.7 / 20.7 | 3.5 / 26.5 | 4.6 / 11.6 | 45 次中 2 次 |
| `proteus_60` | 1.3 / 18.5 | 4.1 / 10.9 | 4.8 / 16.6 | 3.7 / 7.0 | 7.2 / 15.6 | 45 次中 0 次 |

每格为标称平均误差 / 最差最大误差，单位厘米。部署默认使用 `proteus_wamseq_62`。`12_stand_walk_b` 在 kp ×0.8 时
骨盆降到 0.45 米，因此先跑站立参考，再跑 11、13、14。

在 `21_wave_right` 上，`proteus_wamseq_62` 为 2.1 / 3.1，9 次无跌倒，右手左右摆幅 12.7 厘米（参考为 15.3 厘米）。
`proteus_60_bigdata` 为 2.5 / 13.8（kp ×0.8 时骨盆降到 0.50 米），`proteus_60` 在 kp ×0.8 时跌倒。早先一版把手臂向前
举起的挥手，在 kp ×0.8 时会让 `proteus_wamseq_62` 向前迈步约 1.9 米，因此部署用的挥手改为向侧面举臂。

## 在电脑上演练

`scripts/x2/bridge_sim_test.py` 用 MuJoCo 代替 AimDK HAL，运行真实的 `aimdk_bridge.py` 和 `tracking.py` 进程，
带虚拟龙门架，并按操作键顺序执行：a、a、放下龙门架、s、空格（`--lower-before-hold`：a、放到地面且绳子收紧、a、
松绳、s、空格；`--gantry-gravity` 和 `--gantry-z-offset` 设置悬挂姿态）。它检查 bridge 在第一次 a 之前不发任何指令，并报告
放下龙门架后的最低骨盆高度和实时系数。使用 `proteus_wamseq_62` 时六条参考全部通过，实时系数 1.00，骨盆最低
0.546 米（挥手时为 0.654 米）。

## 真机操作

在电脑上部署（不改变机器人状态）。aarch64 / CPython 3.10 依赖包复用由同一 `uv.lock` 解析的 ET1 依赖：

```bash
bash scripts/x2/deploy_onboard.sh
```

主板上的脚本加载 `/agibot/software/entry/cfg/basic_env.sh` 和 `/agibot/software/hal_ethercat` 中的 `aimdk_msgs`。
HAL 以 1000 Hz 发布关节状态、500 Hz 发布 IMU。主板设置了 `isolcpus=5-7`，核 4 运行 `hal_ethercat_x2`，因此
bridge 运行在核 5，Tracking 运行在核 7（ONNX 单独 10.8 ms，完整循环中 13.4 ms）。

1. 只读检查（mc 仍在运行）：

   ```bash
   ssh agi@10.0.1.40 proteus_x2/repo/scripts/x2/probe.sh
   ```

   **把关节读数和机器人实际姿态对照。** 吊在龙门架上处于零力矩模式时，髋关节可能被转到远离站姿的位置（曾测得
   髋偏航 ±1.7 rad）。接管前先用手把四肢摆到接近站姿。

2. mc 运行时空跑（bridge 不创建指令发布者）：

   ```bash
   ssh -t agi@10.0.1.40 PROTEUS_READ_ONLY=1 proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ```

3. 机器人吊在龙门架上运行。终端 1：

   ```bash
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/start_proteus_onboard.sh
   ```

   默认参考是挥手 `outputs/x2/wave/21_wave_right`。它的第一帧就是站姿，所以接管、整形和 Hold 与 `pico_hold` 完全
   相同，按 `s` 后手臂才开始动。用 `PROTEUS_MOTION=outputs/x2/pico_hold`（只站立）或
   `PROTEUS_MOTION=outputs/x2/lafan/15_walk_short`（短距离向前走，已在真机验证）选择其他参考。

   终端 2（操作员）：

   ```bash
   ssh -t agi@10.0.1.40 proteus_x2/repo/scripts/x2/operator.sh
   ```

   等 `logs/run-*/bridge.log` 中出现 “Tracker and LowState ready”。然后：

   - `a`：接管并整形到站姿。接管时运行 `scripts/x2/release_mc.sh`：停止 mc，并等到 `mc_main` 退出
     （`aima em stop-app mc` 可能超过 20 秒才返回，最多等 60 秒）
   - 放低龙门架，直到双脚平踩地面、机器人站直，绳子仍保持收紧
   - 再按 `a`：用 2 秒过渡到参考，由策略接管（Hold）
   - 放松绳子
   - `s`：播放
   - 空格：锁存停止

   **机器人悬空时绝不能按第二次 `a`。** Hold 阶段策略已经在运行。2026-11-12 第一次真机运行时，按第二次 `a` 那一刻
   机器人吊着，前倾 13°、侧倾 7°，双脚离地。策略把这当作要摔倒，用力迈步；脚下没有支撑，于是乱蹬并绕绳子打转
   （偏航角速度最高 7 rad/s），直到按下空格。电脑演练用 `--gantry-gravity 0.22,0.10,-0.97` 能复现这一现象（腿部
   关节速度均方根 3.0 rad/s，真机为 3.5），按上面的顺序 Hold 则保持平稳（0.4 rad/s）。

   **脚上的承重。** Hold 时监视窗口（`scripts/x2/watch.sh`）会显示膝盖力矩。X2（42 kg）自己站立时每个膝盖约
   +15 Nm。按 `s` 之前把绳子放松，直到两边都在约 +10 到 +20 Nm，并让龙门架跟着机器人走。2026-11-12 膝盖读数约
   0 Nm（行走时平均 −1.9 / −1.4 Nm）：重量在绳子上，脚底摩擦很小，按腿式里程计 `15_walk_short` 只走了约 0.7 米，
   参考为 1.33 米（电脑演练 1.26 米），横滚最大摆到 −12°。

   **主板重启后** `mc_main` 会重新启动，此时 `aima em stop-app mc` 约需 15 秒。这期间 bridge 循环被阻塞，
   Tracking 的整形计时继续走，因此 bridge 会锁存阻尼，并提示 “takeover took … s; mc is released, restart the
   launcher and press a again”。退出并重新启动即可：mc 已停止，下一次接管是即时的。`mc_main` 在运行时启动脚本会给出
   提醒。

   **超出限位的关节。** 软吊在龙门架上的机器人可能有关节停在限位之外。整形可以保持或拉回这样的关节，但不会把它
   指令到比按 `a` 以来更远的位置。被拒绝的指令会锁存，并在 `bridge.log` 中写明关节、目标、限位和实测值。

   停止、指令或状态超时超过 0.2 秒、或收到无效指令，都会锁存阻尼。退出时 bridge 会发送阻尼。用
   `aima em start-app mc` 重新启动原生控制器。日志、bridge 状态和 Tracking 录制保存在 `proteus_x2/logs/run-*/`。

HAL 自身没有指令超时：如果 bridge 进程被强杀或主板断电，最后一条指令会一直生效。龙门架和实体急停要始终在手边。
