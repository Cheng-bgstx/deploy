---
title: 众擎 PM01 上的 Proteus WAMSeq
slug: /tutorials/proteus-pm01
---

# 众擎 PM01 上的 Proteus WAMSeq

## 在运控计算机上机载运行：wamseq_62_full、挥手与行走（2026-11-13）

策略也可以像 Galbot ET1 一样，直接在 PM01 自带的运控计算机上运行。运控计算机是 Nezha（`user@192.168.0.163`），配置为 Ubuntu 22.04、x86_64、CPython 3.10、ROS Humble。机载运行使用机器人自己的 ROS 和已安装的 `interface_protocol`，因此 PC 上不需要装 ROS。Tracking 进程、通信传输、看门狗和按键都与下文 PC 方案相同，变化的只是运行位置。**目前只做过一次吊架测试（2026-11-13，见下文）：在 `joint_bridge` 下运行了 Hold 和挥手，当时吊绳承担了大部分体重。此后加入的踝关节 kd 放大、双脚承重时的 Hold 以及行走都只在仿真中验证过。**

### 资产（在 PC 上）

2026-11-05 真机试验用的是 `wsq61` 权重，这台电脑上没有，所以这里导出的是 `proteus_wamseq_62_full`。两者的 WAMSeq 输入格式相同，PM01 契约中的参数也一致。为什么选 `_full` 而不是 `proteus_wamseq_62`，见下文“骨盆晃动”一节。需要把以下源码放在仓库旁边：`Proteus-wamseq`、`GMR-engineai`（engineai-robotics/GMR），以及 LAFAN1 的 `walk1_subject1.bvh`。

```bash
python -m scripts.pm01.prepare --sources-root .. --release proteus_wamseq_62_full --skip-motions
uv run --project venv/pico --no-sync python -m scripts.proteus.retarget_lafan --robot pm01 --duration 8 \
  --output checkpoints/engineai_pm01/motions --bvh /path/to/lafan1/walk1_subject1.bvh
python -m scripts.pm01.make_wave_motion --output checkpoints/engineai_pm01/wave
```

挥手动作是合成的，不是动作捕捉。整个过程是：站立；用 1.5 秒把右臂向侧面抬起，小臂竖直立在头旁边；以 1 Hz 用肘部挥手（±0.4 rad），持续约 5 秒；放下手臂；站立。首帧和末帧都是契约中的默认站姿。关节峰值速度为 2.5 rad/s，挥动的手臂与躯干、头部至少相距 11.8 cm。行走使用的仍是之前那段 8 秒的 LAFAN 片段，首帧双臂接近水平。

MuJoCo 集成评测（`scripts.proteus.evaluate`，使用实测根位置，前后各保持一秒）的世界坐标身体误差，格式为平均 / 最差帧：

| 动作 | 训练增益（0.625 ms） | classic 踝增益（2 ms） |
| --- | --- | --- |
| walk1_subject1 | 2.30 / 3.57 cm | 4.29 / 11.6 cm |
| wave_right | 5.17 / 13.7 cm | 5.09 / 9.0 cm |

classic 一列是把踝关节的 kp 和 kd 乘以下文的关节侧系数（俯仰 4.09，横滚 1.113）得到的。挥手的世界误差主要来自评测开始时 12 cm 的位置偏差，行走评测中也有同样的偏差，机器人会挪动双脚 6–9 cm 去追上参考。在骨盆坐标系下，右手与参考的偏差在 1–2 cm 以内，左右摆幅 10.6 cm，参考为 12.1 cm。两组仿真都没有包含实测的踝关节惯量和回路延迟。

### 骨盆晃动：classic 踝关节增益与控制延迟

最初的仿真视频里，骨盆以约 2 Hz 前后晃动，挥手时也一样，而挥手的参考动作身体是不动的。用贴近真机的仿真可以把原因分开：这种仿真使用集成仿真，和真机一样开启 `reference_forcing`，并加入从观测到新目标生效的命令延迟。

- **classic 踝关节增益。** 通信层把训练增益原样转发，经过 classic parser 后，踝关节俯仰在关节侧约硬 4.09 倍。用 `proteus_wamseq_62`、15 ms 延迟时，挥手中骨盆晃动 0.92° rms，双脚滑动 10 cm；没有延迟时滑动 40 cm，相当于迈了一步。加大 kd 没有帮助。
- **控制延迟。** `proteus_wamseq_62` 在延迟 20 ms 时挥手开始抖动（腰部约 1°），22.5 ms 时无论踝关节怎么设置都会迈步。`proteus_wamseq_62_full`（同一模型，用全部动作训练）到 20 ms 仍然平稳，22.5 ms 才开始抖动。在 Nezha 上，仅推理就要约 12.5 ms。
- **版本。** `proteus_wamseq_62_full` 在踝关节增益有偏差且完全没有延迟时会迈步，但真机不可能没有延迟；在 15–20 ms 区间内，它是更稳的版本。

- **手臂伺服与延迟。** 到了延迟临界点，所有关节组都会振荡，挥动的那只手臂最明显。手臂伺服比较软（kp 18.9，kd 1.25）；把手臂 kp 乘以 0.7，临界点从约 20 ms 推到约 30 ms。手臂 kd 乘以 1.5 能到 25 ms；只加腰部或腿部阻尼没有帮助。不需要再提高推理速度：int8 量化快 3 倍，但闭环跟踪变差；在 Nezha 上用 CPU 1 和 3 跑两个 ONNX Runtime 线程实测 26 ms，反而更慢（CPU 3 单线程为 12.5 ms，因此从观测到目标约 15–18 ms）。

因此机载启动脚本默认使用 `proteus_wamseq_62_full`，用 `PM01_ANKLE_PITCH_GAIN_SCALE=0.306`（1.25 / 4.09）缩放指令中的踝关节俯仰 kp 和 kd，并用 `PM01_ARM_KP_SCALE=0.7` 缩放肩、肘的 kp。两者都只作用于策略命令，阻尼不变；设为 1 即恢复训练增益。考虑到 4.09 这个估计可能不准，测试覆盖了关节侧踝关节系数 0.8–2 倍、延迟 15–25 ms：挥手时骨盆晃动不超过 0.18° rms，双脚滑动不超过 5.6 cm；唯一例外是 2 倍与 20–25 ms 同时出现时（0.88°，25 ms 时迈了 46 cm 一步）。行走从未摔倒，骨盆坐标系下的最差帧误差小于 4 cm。

第一次吊架测试（2026-11-13）及由此做的修改：

- **回调线程延迟。** 进入 `joint_bridge` 后，每步策略计算从 12.5 ms 变成 22–28 ms。固件发布关节和 IMU 状态的速度比 Python 解析得快，ROS 回调线程在策略所在的 CPU 上一直满负荷运行（py-spy：约 40% 在我们的解析代码，60% 在 rclpy）。启动脚本现在把这个线程固定为 200 Hz 节奏（`PM01_HAL_HZ`），并放到 CPU 0（`PM01_HAL_CPU`）：推理恢复到约 12 ms，每步平均约 17.5 ms（由于 GIL 争用，约 10% 的步骤略超 20 ms）。之前空跑测得的 12.5 ms 没有发现这个问题，因为不在 `joint_bridge` 时不会发布命令。
- **轻载晃动。** Hold 和挥手时骨盆以 4.3–4.7 Hz 前后晃动（俯仰角速度阵发性达到 15–40°/s rms），而静止的 Hold 很安静（0.1°/s）。膝关节力矩只有 5–7 Nm：张紧的吊绳承担了大部分体重。仿真中在骨盆处托住 80% 体重即可复现（5.4 Hz），满载时则很安静。启动脚本现在用 `PM01_ANKLE_KD_SCALE=1.5` 放大踝关节俯仰和横滚的 kd：托住 80% 体重时挥手晃动从 8.8 降到 0.4°/s，托住 50% 时 Hold 很安静；在踝关节系数 0.8–2 倍、延迟 15–25 ms 范围内，挥手和行走结果与上文一致（2 倍 / 25 ms 时挥手迈出的一步从 46 cm 缩小到 13 cm）。进入 Hold 前要让双脚承担体重（见“在机器人上运行”）。

### 部署（在 PC 上）

```bash
ssh-copy-id user@192.168.0.163    # 只需一次
bash scripts/pm01/deploy_onboard.sh
```

脚本会按 `uv.lock` 解析出 x86_64 / CPython 3.10 的依赖包（torch 用仅 CPU 版本），然后把代码、`proteus_wamseq_62_full` 导出、两个动作、准备好的模型和两个缓存的 Hugging Face 资产复制到机器人的 `~/proteus_pm01`，最后在机器人的 ROS 环境下做一次导入检查。它不会启动、停止或重新配置机器人上的任何软件。整个目录约占 1.6 GB，其中依赖包 1.4 GB，大部分是 Tracking 需要导入的 CPU 版 torch，部署前请先在机器人上用 `df -h ~` 确认空间足够。加上 `--build-only` 参数时，只在 PC 上解析依赖包，不连接机器人。用这套依赖包加独立的 CPython 3.10、离线资产重新跑上面的挥手评测，结果与之前相同（5.17 cm），说明要部署到机器人上的 ONNX Runtime 和 MuJoCo 都能正常工作；rclpy 和机器人的消息定义只能在机器人上检查。

### 在机器人上运行

1. 按下文方式上电：先开计算机电源，再按白色按钮给关节上电，等 `systemctl status monitor.service robotics.service` 显示 monitor 已退出、robotics 正在运行（`journalctl -b -u robotics.service | grep 在线设备有` 应报告 25 个设备）。
2. 机器人挂在吊架上：按 **LB + RB** 进入 `passive`，把每个关节都扳到离站姿 1.2 rad 以内（悬挂时髋、肩、肘的偏航关节会转得很远），再按 **LB + A** 进入 `pd_stand`。如果 `pd_stand` 被拒绝，`journalctl -u robotics.service` 会列出超限的关节。
3. 只读演练，在 `passive` 或 `pd_stand` 状态下进行：完整的 Tracking 流程会用实时反馈运行 15.5 秒，不创建任何命令发布者，并报告每一步耗时与 20 ms 控制周期的对比。

   ```bash
   ssh -t user@192.168.0.163 'source proteus_pm01/repo/scripts/pm01/env_onboard.sh && cd proteus_pm01/repo && \
     python3 -m scripts.pm01.shadow_policy --policy-config checkpoints/proteus/proteus_wamseq_62_full/engineai_pm01/policy.yaml \
     --motion checkpoints/engineai_pm01/wave/motions/wave_right.npz --output ../logs/shadow-wave.npz'
   ```

   只有当 `policy_step_ms_p99` 明显小于 20、`steps_over_20ms` 接近 0 时才继续。否则用 `PM01_TRACKER_CPU` 把 Tracking 绑定到厂商控制器不使用的 CPU 核上。
4. 在机器人悬挂、双脚离地时按 **LB + 十字键左**，进入 `joint_bridge`。然后运行：

   ```bash
   ssh -t user@192.168.0.163 proteus_pm01/repo/scripts/pm01/start_proteus_onboard.sh wave
   ```

   只读检查不通过时，启动脚本会直接退出。按键与 ET1 操作一致：**a** 整形，整形完成后再按 **a** 进入策略 Hold，**s** 播放或暂停，**空格** 或 **x** 停止（进入阻尼），**Ctrl+C** 退出。启动后先 10 秒什么都不按，机器人应该和在 `passive` 时一样安静。按 **a**，用 10 秒整形到首帧。放低吊架，直到双脚平稳踩在地上并承担体重，吊绳放松但仍能接住摔倒，再按 **a** 进入策略 Hold。如果吊绳张紧、承担了大部分体重，策略会让骨盆以 4–5 Hz 晃动（见上文）。先保持约 10 秒观察骨盆：如果晃动，按 **空格** 并停止。双脚悬空时绝不能进入 Hold：策略会把悬空当成正在摔倒（X2 上就出现过机器人在吊绳上乱甩的情况）。然后按 **s** 开始播放。停止后进入阻尼，但阻尼撑不住站立的机器人，所以吊架要随时能接住。可以在另一个终端观察骨盆：
   `ssh user@192.168.0.163 'source proteus_pm01/repo/scripts/pm01/env_onboard.sh && cd proteus_pm01/repo && python3 -m scripts.pm01.watch_sway'`。
5. 行走：步骤相同，改用 `start_proteus_onboard.sh walk`。行走动作会向前移动约 3 米，吊架必须能跟着走，路径上也要清空。

每次运行都会写入 `~/proteus_pm01/logs/run-*/`，包括 `check_robot.json` 和 `tracking.npz`。除踝关节和手臂增益缩放以及按固定节奏运行的回调线程外，通信传输与 PC 方案相同：使用 classic 踝关节 parser；250 ms 收不到命令或反馈就进入阻尼；每次位置变化最多 0.2 rad。进程崩溃或断电时，仍然只能依靠固件自身的保护。

## 当前基线：classic 踝关节 parser（2026-11-05）

`start_proteus.sh` 和 `start_pico.sh robot` 现在发送 `parallel_parser_type=0`（`PM01Codec.parallel_parser_type`）。`policy.onnx`、`policy.yaml` 和所有下发的 kp/kd 数值都没有改变。T800 继续使用 `RL_PARSER=1`，relay 工具（`policy_bridge.py`、`trial_policy.py`、`bench_damping.py`）的主动命令也仍使用 `RL_PARSER=1`。**2026-11-05 已在吊架保护下完成首次硬件行走，结果见下文。**

更换 parser 的原因：

- 2026-11-05 19:35，移除启动防护后的首次通电运行使机器人剧烈抖动，并在 `tracking` 启动约 3.4 秒后整机重启。掉线后的 PC 网口事件与 2026-11-04 那次重启相差不到一秒。这次运行没有录制，其关节层面的原因是根据上述时序和未改变的通信层推断的。
- 没有发现与之竞争的固件写入者。`JointCommandCallback` 只在 `JointBridgeRunner::Run()` 中运行，且 `joint_bridge` 只有一个 runner。18 次试验共 38,277 个 RL 阶段的电机驱动样本都与下发增益一致，踝电机 kp 和 kd 均为零。
- 在同一话题、同一模式下，classic 的脚踝 `kd=3` 在全部 23 次记录的试验中都保持稳定。在 `walk-deploy-006` 中，它在 24 ms 内把踝 roll 速度从 9.39 rad/s 降到 0.42 rad/s。
- 由 `walk-deploy-006` 得到，从 PD 律到实际加速度约 8-9 ms，踝 roll 惯量为 0.011-0.016 kg m^2。仿真模型中为 0.04，因此仿真一直稳定。用 0.0107 kg m^2 和 8 ms 的延迟模型可以复现事故：±40.8 rad/s、±0.24 rad、28 Hz，录制值为 ±39 rad/s、±0.24 rad、25-28 Hz。这些是基于一次振荡和一份录制的估计值。
- 在该模型中，`kp=37.701` 时脚悬空的 RL 稳定范围约为 `kd` 0.54-1.23。训练值 `kd=2.4916` 高于该范围，之前的 `kd=0.3` 低于该范围。classic 把 PD 保留在电机驱动内部，不经过这段回路。

代价：

- classic 增益作用在电机上，因此关节侧踝增益为 `J^T K J`：根据已安装固件的踝关节数学库在录制姿态下的结果，pitch 约为下发值的 4.09 倍，roll 约为 1.11 倍。
- 厂家文档建议以串联关节训练的策略使用 RL parser，这里是有意偏离。
- 电机回读确认 classic 的脚踝 kd 原样到达电机，但这里只测试过 `kp=0` 的 classic 命令。`kp>0` 的行为依据厂家文档。

使用未改动的 ONNX 和上述关节侧踝增益，在 2 ms 步长下四个动作全部通过仿真：

| 动作 | 平均误差，RL 基线 | 平均误差，classic | 最大帧误差，RL 基线 | 最大帧误差，classic |
| --- | --- | --- | --- | --- |
| aiming | 2.59 cm | 3.21 cm | 4.43 cm | 5.56 cm |
| dance | 2.93 cm | 3.84 cm | 6.63 cm | 6.61 cm |
| fight | 4.79 cm | 4.44 cm | 19.37 cm | 10.54 cm |
| walk | 2.29 cm | 4.29 cm | 3.51 cm | 11.60 cm |

walk 的跟踪变差了。此仿真既没有实测的踝惯量，也没有回路延迟，因此不能证明硬件稳定。产物：`outputs/pm01/classic-equivalent-2ms/summary.json` 和 `outputs/pm01/classic-equivalent-config/policy-sim.yaml`。

硬件运行，全程保持吊架和急停就绪：

1. 检查关节读数，不要只看外观。2026-11-05 机器人看起来站得很正，骨盆却相对躯干转了约 180 度（`J12` 约 -3.4，`J08` 约 -3.1，`J00` 约 -3.1，`J01` 约 +2.3）。要经由 `pd_stand` 进入 `joint_bridge`，它会拒绝偏离站立姿态超过 1.2 rad 的姿态；不要从 `passive` 直接进入。
2. 让机器人悬空、脚离地。运行 `bash scripts/pm01/start_proteus.sh --record`，10 秒内不按任何键。它应该和 `passive` 一样安静。
3. 按 **a** 进行插值。双臂会抬到接近水平，这是 walk 第一帧。
4. 放低吊架直到双脚承担体重（吊绳放松，但仍能接住摔倒），再按一次 **a** 进入策略 Hold。
5. 按 **s** 播放。

单靠阻尼撑不住站立的机器人（见 `walk-deploy-005`），因此每一步吊架都必须能接住它。

首次硬件行走（2026-11-05 21:01，`policy_tracking_record_policy_20261105_210134.npz`）：398 帧全部播放完成，没有重启，结束后电机温度不超过约 43 C。与使用相同踝增益和 reference forcing 的仿真相比，膝关节幅度为 1.14/1.04 rad（仿真 1.18/1.01），抬脚高度为 17.1/16.9 cm（仿真 17.8/18.1），骨盆平均倾角为 5.2 度（仿真 5.9）。与该仿真的差异：

- 上报的踝 pitch 力矩峰值达到 106 Nm，训练限制为 61 Nm。仿真在 61 Nm 处截断，因此看不到这一点。
- 踝 roll 5 Hz 以上的运动约为仿真的 2.5 倍。
- 踝 pitch 相对参考的跟踪误差约为仿真中理想 RL 基线的两倍（RMS 0.145 与 0.074 rad）。classic 仿真也有同样的损失（0.147）。

播放结束后，机器人在最后一帧 Hold 中继续晃动约 33 秒，仿真中同样会晃动；约 110 s 处出现 18.27 rad/s 和 133.7 Nm 的峰值。根位置没有测量，因此未比较落脚点和行走距离。

## 已被取代的基线：使用 RL_PARSER 的原训练输出（2026-11-05）

该基线使用从原始 wsq61 contract 重新导出的 `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml`。下文恢复候选均为历史实验，不是当前部署配置。

- 恢复原 actor 的形态特征和全部训练增益，包括脚踝 `kp=37.701, kd=2.4916`；kd=0.3 和 kd=0.5 候选不再作为基线。
- 删除策略软位置裁剪、身体/手臂/头部目标限速、两秒接管混合和限速器状态重置。首帧整形不再强制启用策略裁剪。进入策略之前的十秒整形插值仍作为接管步骤保留。
- PM01 使用标准 EngineAI 通信层：策略输出的 q/dq/前馈/kp/kd 按 SDK 顺序原样下发，共用位置/速度/力矩边界和每包 0.2 rad 的目标限幅（500 Hz 发布下相当于 100 rad/s）。不再有 PM01 专属的裁剪、限速或停机锁存。
- 各处都使用训练增益：zero 模式下发实测姿态、零刚度与基础被动阻尼，parser 保持 `RL_PARSER=1`。有限值检查、反馈/命令看门狗、按发布版执行器 effort 的关节速度/力矩限制保留。客户端会拒绝过期限速/混合配置，避免悄悄忽略旧参数。
- 导出模型与源模型比较 64 组输入，最大绝对误差为 0。四个参考动作均通过 2 ms 集成评测，原 walk 平均世界坐标身体误差 0.02288 m；仿真不包含物理并联电机反馈环。
- 从 `walk-deploy-006` 实测姿态出发、不经过实验命令门控的复演完成原 walk。原有支撑保持第 2～12 秒质量检查因左踝横滚摆幅 0.1604 rad 超过其 0.12 rad 标准而失败，不能将该报告标为全通过。第 5～12 秒全身最大摆幅为 0.02893 rad、最大速度 RMS 为 0.00430 rad/s，说明早期过渡逐渐收敛，并非整个保持段持续振荡。
- 真机只读尝试因 `eno1` 已 DOWN、无 IPv4 地址而无法建立 DDS；未发送新的电机命令。恢复训练输出不等于已完成真机部署。

产物：`outputs/pm01/pure-training-2ms/summary.json`、`outputs/pm01/pure-training-pose006.json` 和 `docs/experiments/proteus-pm01-recovery.json`。


:::warning 真机指令已重新开放（需监护）
2026-11-04 已使能 PM01 在策略启动后、尚未按任何键时出现剧烈振荡，操作者报告整机
自动重启。最新录制共 128 帧，全部为 `zero` 模式；通信层发送了 `kp=0`、`kd=2`、
`RL_PARSER=1` 的阻尼命令。录制没有实际线上增益或机载命令反馈，物理根因仍未确定。

现已移除代码级封锁、停机后锁存的经典阻尼以及 kd=0.3 资格门——它们改变的是
闭环增益和踝关节控制律，与训练对齐的方向相反，并不是部署所需的保护。保留的只有
不改变控制律的部分：有限值检查、命令/反馈/IMU 新鲜度看门狗、`joint_bridge` 模式
归属、发布版执行器 effort/速度边界，以及中继的监护进程。

首次上电请按受控试验对待：用吊架或吊带支撑机器人，手边保留急停，并重点观察事故
中的启动振荡。`walk-deploy-005` 之后承重退出仍未验证（承重转移时反馈超时，
退出移除位置刚度，随后屈膝并出现较大机身倾斜）。此前回环通过仅验证软件集成。
下文历史内容保留作排查背景，不代表已验证的部署流程。
:::

2026-11-05 重新连接后的只读调查进一步缩小了问题范围：

- 事故录制持续 2.539 秒，全程为 `zero` 模式。左右踝横滚反馈速度峰值分别为
  41.44 和 39.44 rad/s，反馈力矩峰值分别为 70.51 和 72.83 Nm。
  这些是反馈值，不是抓取到的电机命令。
- 对实际安装固件的反汇编确认：`RL_PARSER=1` 先计算关节空间 PD 力矩，
  经并联踝雅可比变换后写入电机前馈力矩，并清零踝电机的刚度和阻尼。
  因而发送 `kp=0, kd=2` 不等于电机侧阻尼为 2。
  已确认启动阶段会通过这条路径发送命令；振荡的具体物理机制（包括控制环延迟和
  变换行为）仍未验证。单独更换 parser 或阻尼数值不构成经过验证的修复。
- 启动记录确认运控主机重启，未见正常关机记录。上次启动的 journal、pstore 和
  电机日志均未确定重启原因。电机日志最后条目早于本次事故，不能把其中的电源板
  错误归因到此次事件。检查时机器人时钟比 PC 快约 12 分钟，不能直接按原始墙钟
  时间对齐。电源保护、硬件复位和软件故障等原因仍待区分。

进一步的离线核对与防护：

- 早期的防护曾让 PM01 codec 在初始 `zero` 模式、加载模型和开始控制前退出时不产生命令，
  并在 zero/退出、反馈过期、命令过期或反馈异常时锁定为原厂 classic 阻尼。
  该防护同时替换了训练脚踝阻尼和 parser，因此已移除：通信层现在使用上文
  “当前基线”所述的标准 EngineAI 行为。
- 已在 PC 上使用实际安装的脚踝数学库和已加载几何参数，核对左右脚各 128 个事故姿态。
  因为没有录制原始电机反馈，电机角度由串联关节姿态重建。
  雅可比与数值差分的最大相对误差为 0.173%，最大条件数为 2.002，所有样本的
  重建阻尼功率均非正。这不支持该重建条件下简单的几何符号反转或奇异点假设；
  不代表实际传感器标定、离散控制环延迟、饱和或稳定性已通过验证。
- 已加载的电机任务周期为 2 毫秒。RL parser 的脚踝阻尼在该机载任务中根据机载反馈
  计算，控制环不经过 PC 策略推理往返。已检查的 bridge runner 会在退出时恢复
  classic parser，其运行路径仅处理 ROS 回调；尚未确认存在命令超时保护。
  不能假设停止 PC 发布就会停止已经受控的机器人。
- 同一台机器已加载的 `rl_basic/vae/default` 使用脚踝 `kd=0.3`，
  `rl_amp/walk_stand/release` 使用 `kd=0.2`。事故启动阶段的 `kd=2.0`
  是这些参考值的 6.7～10 倍，导出的 Proteus 脚踝 `kd=2.4916` 是其 8.3～12.5 倍。
  这些是特定控制器的参考配置，不是已验证的通用安全上限。
  原生 passive 的脚踝 `kd=3` 使用另一条控制路径，不能据此证明 RL parser 下
  阻尼 2 安全。离散力矩环的增益不匹配成为主要假设，尚未证明完整因果链。
  修改策略增益也会改变训练时的控制器动力学，需要单独验证。
- 运控主机和 Orin 都发生了重启。分别读取 `/proc/uptime`，推算出两台机器在事故后
  的启动时间相差约 7.9 秒。这支持共同供电或复位事件，但不能确定触发原因。
  Orin 的墙钟未设置（显示 1970 年），且没有持久化保存上次启动日志；比较使用的是
  运行时长和 PC 时钟，不是这些墙钟日期。

离线恢复候选将四个脚踝阻尼改为 `0.3`，并同步更新 Proteus 形态描述中的对数阻尼特征。
如果只修改 YAML，actor 仍会以原始动力学参数作为条件。重新导出的模型在 64 组输入上
与源 actor 完全一致。采用 2 ms 仿真步长时，四个参考动作均通过；平均世界坐标身体误差
分别为瞄准 2.54 cm、舞蹈 3.77 cm、搏击 8.00 cm、行走 2.58 cm。这些是离线跟踪结果：
仿真的串联关节 armature 和力矩限幅不能证明实机并联电机环稳定。
命令、哈希和完整指标保存在 `docs/experiments/proteus-pm01-recovery.json`。

独立工具 `scripts/pm01/bench_damping.py` 默认只读记录。其实验执行路径最多测试一秒
低脚踝阻尼；可选 `--ankle-kp` 保持测得的初始脚踝姿态，默认不施加位置刚度。
两种模式都不请求关节前馈力矩；要求新鲜反馈，开始前检查 classic 阻尼，
结束后恢复并检查 classic 阻尼。另一个进程负责处理测试进程卡死；两个进程都必须
运行在机载电脑，才能在 PC 断线后继续退出流程。它不是部署看门狗，也不能处理 Linux、
DDS 或电机总线故障。实机执行还要求只读访问已核对的电机驱动
实际增益数组，通过精确二进制哈希拒绝其他版本。机载 `JointCommand` 回读没有填写
解析器字段，不能仅靠该字段确认解析器切换。`MotorDebug` 数组为 24 个数值加上
`DataMonitor` 追加的时间值，并不存在第 25 个电机。

## 限时真机恢复测试结果（2026-11-05）

机器人由吊架完全悬空时，已完成十二轮短时脚踝测试：阻尼从 `0.05` 到 `0.3`，
然后保持当前姿态，刚度逐步到 `kp=37.701, kd=0.3`；最长脚踝保持为一秒。
另完成一秒和三秒全身整形测试；三秒测试中双臂实际移动约 19 度，最大关节反馈力矩
3.20 Nm。每轮结束均通过实际电机增益回读确认恢复 classic 阻尼。

首次实际 **policy 模式**测试也已完成：510 条整形请求后，运行约一秒、51 条策略请求，
随后锁定恢复 classic 阻尼。脚踝峰值速度 0.186 rad/s，脚踝反馈力矩 0.729 Nm，
全身反馈力矩峰值 2.184 Nm。运控主机 boot ID 未改变；测试后回读为零刚度、零前馈、
原厂阻尼，无新增已报告故障标志。这仅验证短时静态接管，**不代表行走、无支撑站立、
长时间稳定性通过，也没有证明原重启事故的完整根因**。
操作者确认没有异常抖动；随后只读检查确认已回到 `passive`，没有硬件命令发布者。

随后三秒真机保持也已完成：510 条整形请求、151 条策略请求，策略请求持续 3.009 秒。
脚踝峰值速度 0.200 rad/s、反馈力矩 1.149 Nm，全身力矩峰值 4.253 Nm（右髋横滚）。
监护进程确认恢复 classic 阻尼；运控主机没有重启，没有新增已报告电机故障。
操作者确认无异常抖动或声音并返回 `passive`，只读反馈确认该模式且无命令发布者。
日志为 `.scratch/pm01-recovery/static-policy-hold-003.*`。

另外，对同一静态参考进行约 20 秒地面仿真，2 ms 物理步长下通过：平均身体误差
3.14 cm，最大逐帧平均身体误差 3.82 cm，播放期间骨盆最低高度 0.812 m。
这里使用不含调试限速的 `kd03/policy-sim.yaml` 候选、仿真测得根位置，且机器人直接
初始化在参考姿态。它**没有**验证真机接管、接触承重过程或部署配置中参考强制的根位置估计。
不能仅因该离线结果通过，就把此配置替换进悬空真机测试。

测试使用独立调试配置：重新导出的 `kd=0.3` actor、目标变化速度限制 `0.05 rad/s`，
以及从当前悬挂姿态采集的静态参考。这样的慢速限幅不适合地面平衡，采用这些严格限幅的
地面预演发生跌倒，不能拿此配置行走。切入 policy 时保留整形的最后目标，避免突然撤掉
双臂用于抵抗重力的 PD 力矩。

`rehearse_trial.py` 现用 MuJoCo mocap weld 支撑根节点。逐子步重置自由根位置和速度
仍会让动力学求解出现自由落体加速度，从而低估关节重力负载；此前此类结果不再作为依据。
采用 weld 后，原行走片段首帧双臂水平展开，在整形中触发 5 Nm 手臂保护；静态悬挂参考
在不提高保护阈值的情况下通过，仿真脚踝峰值速度 0.165 rad/s。weld 仍是近似，
不等同于实际吊索或电机控制环动力学。

实验工具与常规入口分开：

- `shadow_policy.py` 用真机反馈运行 ONNX，不创建硬件命令发布者，已检查 775 帧。
- `trial_policy.py --policy-config <commissioning.yaml> --motion <static.npz>
  --output <new-record.npz>` 只向 `/sim2real/pm01/request` 发请求。
  在根项目环境中先 source `scripts/pm01/env.sh` 再运行。
- `policy_bridge.py --execute --config <bridge-config.json> --output <new-log.jsonl>`
  必须在**已核对的运控主机**运行，使用机载 ROS 环境和 root 只读回读权限，旁边放置
  `bench_damping.py`。独立机载监护进程允许约 10 秒整形，默认一秒策略保持；PC 命令中断、
  非法目标或速度/力矩过大都会锁定恢复 classic 阻尼。先启动 PC 客户端，它会等待 bridge。
  添加 `--shaping-only --shaping-seconds 1` 则在进入 policy 前结束。工具不会播放参考动作。
  `--hold-seconds 3` 选择下一阶段的三秒保持，速度、力矩和断线保护保持不变。
  三秒静态 weld 仿真已经通过，最大关节力矩 3.728 Nm，随后完成了上述限时真机测试。

这些命令要求匹配已测增益、关节顺序、静态数据集 manifest、固件哈希及悬挂工况，
不是不受限的部署方案。监护进程在退出时核对实际原厂阻尼，PC 断线后仍可执行，
但不能覆盖操作系统、DDS 或电机总线故障。本地证据保存在
`.scratch/pm01-recovery/static-policy-hold-001.*`，配置、产物哈希、仿真及实机指标
索引在 recovery 实验 JSON 中。

### 原 walk 调试入口（实验性）

原 walk 参考在起始朝向下约向前移动 3.00 m、横向移动 0.21 m，需要可随行的吊架及
对应净空，不能把固定悬空测试等同于地面行走。操作者已确认本现场吊架可随行且路径净空。

双脚轻触地、吊带承重后，首次原 walk 首帧整形在约 6.8 秒退出：左踝俯仰触发原有
2 Nm 保护，随后确认恢复原厂阻尼。操作者确认没有异常抖动或声音并回到 passive。
有效整形期间全身最大速度为 0.214 rad/s；整个日志中较大的 1.245 rad/s 峰值出现在
恢复阻尼阶段。本轮没有进入 policy。接触负载需求估计显示脚踝约需 3.06 Nm，
肩横滚约需 6.30 Nm；这只是模型估计，不代表可以绕过故障保护。

独立 `--walk-once` 入口实现：整形、显式策略激活、暂停在首帧的策略保持、独立播放条件、
原速播放一次、锁定恢复原厂阻尼。它采用重新导出的脚踝 `kd=0.3` actor，身体及手臂
目标限速 `6 rad/s`、头部 `1 rad/s`，并用 2 秒从最后整形指令渐入策略。
地面静态 `0.5 rad/s` 限速版本，以及四倍慢放搭配 `2 rad/s` 限速版本均离线失败，
不得用于部署。

此入口使用独立工作范围：整形时脚踝保护为 4 Nm、肩横滚为 8 Nm；policy/播放期间
踝俯仰上限 50 Nm、踝横滚 20 Nm、关节速度 8 rad/s、目标步进 0.12001 rad；其他
关节力矩上限见 `WALK_TORQUE_LIMIT`。这些值低于准备模型的限值，**不构成真机安全认证**。
还会额外检查实际踝电机前馈数组，单电机不得超过 25 Nm。必须收到新鲜机载 IMU，
倾斜超过 0.35 rad 或角速度超过 3 rad/s 会停止。原有反馈、温度、电机故障、命令断流、
单发布者及独立监护检查继续生效。

三次真机接管（`walk-deploy-001` 至 `003`）均在播放前退出，依次触发头部请求力矩、
右肩请求力矩、以及私有请求漏一帧后的头部目标步进保护（接收时间戳间隔从 20 ms
变成 40 ms）。各次均确认恢复原厂阻尼。操作者修正第三次反馈为**确实存在持续抖动**。
该次策略保持约 10 秒，双肩约 1.95 Hz 振荡，峰峰摆幅 0.51/0.48 rad。
未观察到新增电机故障或再次重启。

将仿真的吊架支撑保持延长至 12 秒后复现了抖动。新增的手臂 `2 rad/s` 限速导致
持续振荡，仿真双肩峰峰摆幅 0.43/0.53 rad；离线恢复 `6 rad/s` 后降到
0.0084/0.0048 rad，加入 2 秒接管渐入后为 0.0212/0.0135 rad。
这些对照定位了仿真闭环中的限速诱发振荡。不要使用
手臂限速 2 的配置。此前仅以不摔倒、完整播放作为仿真通过条件并不充分。
relay 新增首帧静态检查：策略保持满 4 秒后，检查最近 2 秒双肩运动，峰峰幅度超过
0.25 rad 或速度 RMS 超过 0.6 rad/s 就锁定恢复阻尼，播放前必须通过该检查。
私有请求改为有限队列的可靠传输，超时、顺序和步进保护保持有效。

随后修正版真机 `walk-deploy-004` 在吊架支撑下保持原动作首帧 45.02 秒。
前 4 秒之后，双肩峰峰摆幅为 0.00191/0.00038 rad，速度 RMS 为
0.00224/0.00210 rad/s，全身最大关节速度 0.0117 rad/s；未触发振荡或电机保护。
由于等待承重转移确认超时，本轮未播放动作，退出后确认恢复原厂阻尼。
**这验证了受支撑首帧保持，不代表自主承重或行走已通过。** 操作者确认窗口现改为
180 秒；100 ms 命令断流及其他电机保护保持不变。

`walk-deploy-005` 策略保持约 57.28 秒后触发关节反馈超时。切回 classic 阻尼前
最后 200 ms 的最大关节速度仅 0.0076 rad/s，最后录制的策略状态倾斜为 0.067 rad。
第一条阻尼指令后约 18 ms 开始出现关节运动，100 ms 内膝关节弯曲增加约 0.05 rad；
随后的只读监测观察到机身倾斜约 0.70 rad。操作者确认当时正在转移承重，已急停，
吊架托住机器人且人员安全。没有发送播放信号。relay 与监护进程均确认恢复原厂阻尼，
运控电脑未重启。证据确定了**反馈超时 → 移除位置刚度 → 丧失站姿支撑**的顺序，
反馈间隙本身的原因尚未确定。纯阻尼不是站立保护；此前仿真只验证正常播放，
没有覆盖这条承重退出路径。实机 walk 仍未验证。

离线故障注入已复现此前漏测的退出工况：承重转移后将位置刚度归零并使用原厂阻尼，
根部高度在 2 秒内从 0.810 m 降到 0.254 m；加入理想刚性支撑约束后则保持在
0.810 m。串联关节模型与理想 weld 不能验证实际吊带，持续机械支撑仍须单独建立。
只读时序测试还记录到轻量订阅的一次 20.06 ms 间隙，以及带完整负载接收循环的
66.02 ms 最大单轮耗时。批量处理回调降低了反馈年龄，但这些测量尚不能确定原始
间隙来源，也不能据此认可新的真机时限。该轮时限尚未调整。后续实验增加了 20～100 ms 短反馈间隙保持：保留上一目标和增益、暂停 PC 推理，并用 epoch 丢弃旧请求；超过 100 ms 仍退出。006 轮未触发此保持，仍出现连续颤动。

采用实际部署观测、完整 Tracking 客户端、命令保护、首帧整形、12 秒支撑策略保持、
理想化支撑释放及原速播放，修正渐入后的版本从两个实测接触姿态出发完成了仿真。
两次脚踝峰值速度不超过 6.19 rad/s、力矩不超过 45.97 Nm，稳定保持时双肩峰峰摆幅
均低于 0.026 rad。支撑释放以竖直支撑力替代 weld，并在 1 秒内降至零，只是实际
承重转移的近似。较早配置曾用机载脚踝数学库估计单电机最大指令 21.86 Nm，
其中电机角度由模型重建；这不是新配置的实测电机结果。
在记录到完整实机结果前，仍不能宣称原 walk 已通过真机播放。

PC 客户端加 `--walk-once --policy-signal <此前不存在的新文件>
--play-signal <另一个此前不存在的新文件>`，机载 relay 加 `--walk-once`。
客户端检查原动作、目标限速和接管渐入配置。整形后打印 `SHAPING_COMPLETE` 并保持 PD。
双脚接触地面、吊带继续承担主要重量时，向策略信号文件写入 `policy`，进入 `POLICY_HOLD`。
不要在仅有 PD 保持时先卸掉吊架承重：这个顺序已在仿真中失败。策略稳定并通过静态检查后，
在吊架保护下逐渐转移承重。只有确认双脚稳定承重、吊架可随行后，才向播放文件写入 `play`。
动作只播放一次，机载播放时限为 8.5 秒。进入 policy 后 180 秒仍无播放信号，或发生
停止/故障，都会恢复阻尼；任何退出时吊架都必须能继续支撑机器人。独立监护进程总时限
270 秒，旧信号文件或迟到请求不能重新启动一轮动作。

厂商的[并联踝控制说明](https://en.engineai.com.cn/open/docs/pm01/04/source-code-analysis)
也区分了 classic 电机侧 PD 与清零电机 PD 增益的 RL 力矩变换。
新版文档支持这一区别，但不能据此认定机载 2025 固件或替代增益已经验证。

分支 `feat/proteus-wamseq-engineai-pm01` 面向 **24 关节 PM01/pm_v2**，使用已发布的
`engineai_pm01` 合约和 wsq61 WAMSeq actor。机器人别名为 `pm01`、`pm_v2`、
`engineai_pm01`。它与 25 关节 T800 不同；也不使用默认姿态和动力学参数不同的
`engineai_pm01_edu_official` 合约。

## 已验证范围

2026-11-04 对已连接 PM01 的 SSH 检查发现，机载安装包为
`76445d09e967a16eb191d10aa05fe2ac5df6c19f`（2025-10-16），启动配置为 `pm_v2`。
机载 `serial_pm_v2.urdf` 有 24 个活动关节，与厂商 ROS 仓库的 URDF 逐关节一致。
SDK 顺序为 J00–J23，其中 J12 是 `J12_WAIST_YAW`，J23 是唯一的头部关节
`J23_HEAD_YAW`。模型 IMU 坐标轴与骨盆对齐；若实际安装不同，部署 YAML 中可配置
经验证的 `imu_to_pelvis_wxyz` 外参。

机载 `MotionState` 只有 **`current_motion_task_name`** 字段，新版公开接口则使用
`current_motion_task` 和 `available_transition_motions`。codec 兼容两种字段，但 DDS
两端仍须使用匹配的消息定义；仅增加 Python 字段兼容无法解决线上消息结构不一致。
已核对机载 `JointState`、`JointCommand`、`ImuInfo` 和 `RL_PARSER=1`。

此固件的关节和 IMU 消息头时间戳恒为零。PM01 codec 对这类数据使用本机单调接收时间，
保留 250 毫秒反馈断流看门狗。非零时间戳仍拒绝重复和倒退；出现非零时间戳后再回到零也会被拒绝。
零时间戳数据无法区分“重新发布的旧传感器值”和新测量。只读检查同时输出消息头时间戳和
已接受反馈的年龄，便于排查。

已完成：完整 ONNX 数值等价检查、四动作 integrated MuJoCo 验证、不依赖 SDK 的
映射/看门狗测试、使用机载消息定义的真实 ROS 回环收发，以及以已知 XR 姿态替代设备的
PICO IK/ZMQ/ONNX 测试。实际 `start_proteus.sh --record` 启动脚本也通过了隔离回环测试：
使用零时间戳测试反馈，完成 10 秒整形、首帧保持、播放、阻尼、正常退出，保存 878 帧录制，
动作值均为有限数。该测试使用已安装的 ROS 消息和 ONNX 策略，不验证机器人动力学。
**尚未验证持续真机运动跟踪或真实 PICO 设备。** 限时静态真机结果见上文。初次检查时
`robotics.service` 未运行；操作者开启电机电源并切到 `joint_bridge` 后，PC 收到了约
500 Hz 的关节/IMU 数据和 5 Hz 模式反馈，发现一个兼容命令订阅者且没有其他命令发布者。
零时间戳兼容修复后，真机只读就绪检查通过；该检查没有发送 Proteus 控制命令。

## 准备资产

推理使用根项目，IK/遥操使用 `venv/pico`。`--no-sync` 复用已安装的环境。把
`Proteus-wamseq`、`GMR-engineai` 和 `ubisoft-laforge-animation-dataset` 源仓库放在
同一父目录；GMR 检出需要包含 `assets/engineai_pm01` 及 PM01 的 BVH/SMPL-X IK 配置。

```bash
uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects
# 只准备模型、策略和 PICO 中立参考：
uv run --no-sync python -m scripts.pm01.prepare --sources-root /path/to/projects --skip-motions
```

模型、ONNX 和动作属于本地忽略的运行产物。冻结 spec 保留训练默认姿态、增益、限位、
armature 和腕关节动作符号。源 spec 的世界关节轴只保留四位小数，因此验证显式使用
`1e-4` 容差，实测最大差异 `8.76e-5`；没有覆盖任何关节限位。厂商质量与舍入后的 spec
相差 0.000189 kg。来源和产物哈希记录在 `third_party/engineai_pm01/validation.json`。

`checkpoints/proteus/wsq61/engineai_pm01/` 包含单个完整 `policy.onnx`、原样复制的合约、
数值比较报告、真机 `policy.yaml` 和使用测得根位置的仿真 `policy-sim.yaml`。
输入是 `proprioception[10,68]`、`task[6,253]`、`action_history[10,31]`、
`history_valid[10]`；输出是 `action[24]` 和 `next_slot_action[31]`。十步历史从旧到新。
与发布 actor 的 64 组 seed-0 随机输入比较，动作误差均为 0；导出为 IR 8/opset 17。

PM01 导出配置显式包含全部 24 个关节的 `soft_joint_pos_limits`，取自已核对的模型限位，
同时裁剪策略目标和首帧整形目标。若旧 YAML 报 `shaping_target: motion_first_frame ...
set them in the policy YAML`，用更新后的导出器重新生成：

```bash
uv run --no-sync python -m scripts.export_proteus_onnx \
  --contract engineai_pm01 --robot pm01 --release wsq61 \
  --proteus-root ../Proteus-wamseq
```

## PC 连接真机

官方 SSH 地址为 `user@192.168.0.163`（运控电脑）和 `ubuntu@192.168.0.162`（Orin），
详见[厂商连接配置](https://github.com/engineai-robotics/engineai_ros2_workspace/blob/community/scripts/sync_src.sh)。
策略在 PC 上通过以太网控制机器人，无需在 Orin 安装推理环境。

先在 PC 编译**机器人实际安装的**消息定义。下列脚本只复制消息并在本地编译，不修改或
启动机载程序：

```bash
cd /path/to/sim2real
bash scripts/pm01/prepare_ros.sh user@192.168.0.163
source scripts/pm01/env.sh
uv run --no-sync python -m scripts.pm01.check_robot
```

`env.sh` 设置 ROS Humble、Cyclone DDS、域 69、以太网口 `eno1`、本地匹配的消息环境和
两个 Hugging Face 离线变量。可在 source 前设置 `PM01_NETWORK_INTERFACE`、
`PM01_ROS_DOMAIN_ID` 或 `PM01_ROS_SETUP`；已存在的 `CYCLONEDDS_URI` 优先。
`PM01_REMOTE_PROTOCOL` 可覆盖 `prepare_ros.sh` 的机载消息包路径。
不要在同一 shell 混用不兼容的消息 overlay。

检查程序只订阅：收到新鲜的 24 关节、IMU 和 `joint_bridge` 状态，发现兼容的命令订阅者，
且不存在其他命令发布者时返回 0；否则输出缺失条件并返回 2。服务未运行时应当检查失败。
`ready: true` 只验证 ROS 通信和模式归属。这些消息没有电机使能状态，因此不能证明
电机已使能或正在执行目标。

## 真机运行

用吊架保护 PM01，并准备好硬件停止手段。电脑上电和电机上电是两个步骤。
对于配备单独白色急停/上电按钮的 PM01，[官方使用手册第 6、8 页](https://cn-engineai-1304599088.cos.ap-guangzhou.myqcloud.com/uploads/upload/files/20251031/c15bec68ae58e2f5cd2901aad04e0750.pdf#page=6)
要求在电脑开机后按白色按钮；两声短鸣、臀部白灯亮起表示关节已上电。
不要使用颈部红色急停按钮来使能，也不要在成功上电后反复按白色按钮，它也能切断关节电源。

若运控服务未启动，在运控电脑启动现有厂商服务：

```bash
ssh user@192.168.0.163
sudo systemctl start robotics.service
systemctl status monitor.service robotics.service --no-pager
```

实测 PM01 固件会先通过 `monitor.service` 检查电机，再启动 `robotics.service`。
电机未上电时，检查可能无限等待，使 `systemctl start` 一直阻塞。可在另一个 SSH 终端查看
`systemctl list-jobs` 和 `journalctl -b -u monitor.service --no-pager`。
电机上电后，已排队的启动请求可自动继续，不必重复执行，也不要绕过电机检查。
正常启动后，monitor 应为 `active (exited)`，robotics 应为 `active (running)`。

电机上电并不等于控制使能。先退出外部策略，确保吊架支撑、人手离开，再按 **LB + Start**
执行厂商启动（`idle`）流程，随后 **LB + RB** 进入 `passive`，再进入 `pd_stand`。
若当前在 `joint_bridge`，先按 LB + RB 回到 `passive`：此版本只允许从 `passive` 进入 `idle`。
机载 IdleRunner 在离开 `idle` 时请求电机使能；单独进入 `pd_stand`/`joint_bridge`
不能替代这个启动握手。一次机器人不动的只读检查发现：电机已上电、模式为 `joint_bridge`，
但全部 24 个电机的 disable 标志仍为真。重新启动外部策略前应先确认使能；不要在策略
仍针对不动的机器人持续产生目标时恢复使能。

实测机载版本 `76445d09`（2025-10-16）加载的 `robot_release` 模式使用
`task_motion/release`，实际手柄键位如下：

| 按键 | 目标模式 |
| --- | --- |
| LB + Start | `idle` 启动 / 电机使能握手 |
| LB + RB | `passive` |
| LB + A | `pd_stand` |
| LB + 十字方向键左 | `joint_bridge`（`CROSS_Y_LEFT`） |

新版厂商 ROS2 文档中的 BACK + A 在此版本不会触发透传切换。
保持吊架支撑，每一步都通过 `/motion/motion_state` 验证；此固件的字段名为
`current_motion_task_name`。启动器不切换机器人模式。

如果进入 `pd_stand` 后退回 `passive`，查看服务日志。此版本会在任一关节离站立目标
超过 1.2 弧度时拒绝进入。保持吊架支撑和 `passive`，缓慢扶正折叠、扭转的初始姿态，
不要强扳关节；人手离开后再试。如果外观姿态正常但角度读数不符，先查反馈和标定，
不要绕过进入检查。

回到 PC：

```bash
cd /path/to/sim2real
bash scripts/pm01/start_proteus.sh \
  checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

启动器自动加载 PM01 环境，先执行只读检查，再创建控制接口。默认 CPU 策略频率 50 Hz，
独立 ROS 命令线程 500 Hz。`PROTEUS_POLICY_CONFIG` 可覆盖策略路径。

键位（与 ET1 一致）：**a** 用 10 秒整形到动作首帧；再按 **a** 开启策略 Hold；**s** 播放/暂停；
**空格** 或 **x** 进入阻尼；**Ctrl+C** 退出。整形和初次 Hold 时保持吊架保护。控制接口在
`joint_bridge` 下启动后、整形前发送的是阻尼，不能独立支撑站立。显式切换离开该模式后，
接口把控制权交还固件。

发送器按名称映射关节，使用 classic parser（`parallel_parser_type=0`）转换串联踝关节命令，并拒绝维度错误或非有限值。
命令/关节/IMU 超过 250 ms、模式心跳超过 1 s 或错误锁存时，刚度、目标速度和前馈力矩
归零，保留阻尼 2。每次发布的位置步进不超过 0.2 rad。正常退出重复发送阻尼；进程被杀或
断电仍需要固件保护。

## PICO

```bash
# PC 终端 1：XRoboToolkit PC Service，以及已标定的身体/腿部追踪器
bash scripts/pm01/start_pico.sh publisher --actual-human-height 1.8
# PC 终端 2：真机策略；publisher 在同一 PC 上
bash scripts/pm01/start_pico.sh robot 127.0.0.1
```

仿真使用 `start_pico.sh sim` 和 `start_pico.sh policy-sim 127.0.0.1`。
**A** 整形到中立位；松开后 **A+B** 开启 Hold；**X** 播放/暂停；**单独 B** 阻尼。
动作/按钮 TCP 端口为 28701/5592，同一端口组一次运行一个本体。实时 XR 按操作者身高归一化，
IK 关节速度上限 12 rad/s；未来帧不足时用已收到参考补齐。身体流丢失但机器人反馈健康时，
保持最后参考，不自动进入阻尼。本次没有新增手部驱动。

## 复现验证

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m scripts.proteus.evaluate \
  --robot pm01 --policy-config checkpoints/proteus/wsq61/engineai_pm01/policy-sim.yaml \
  --dataset checkpoints/engineai_pm01/motions --output outputs/pm01/new-eval
```

seed 0，策略 50 Hz，物理步长 0.000625 s，初始和末帧各保持 1 秒，播放 8 秒真实 LAFAN1
重定向动作。全部播放完成、数值有限，每帧平均 tracked-body 误差均小于 0.5 m。
以下指标只代表仿真结果。

| 动作 | 平均世界 body 误差（cm） | 平均局部 body 误差（cm） | 世界根部终点误差（cm） |
| --- | ---: | ---: | ---: |
| aiming1_subject1 | 2.60 | 1.71 | 3.97 |
| dance2_subject1 | 3.00 | 1.98 | 2.29 |
| fightAndSports1_subject4 | 4.77 | 2.01 | 7.01 |
| walk1_subject1 | 2.30 | 1.85 | 2.42 |

准确命令、哈希和轨迹见 `docs/experiments/proteus-pm01.json`。

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --no-sync python -m pytest \
  tests/test_pm01.py tests/test_proteus_transport.py tests/test_proteus_observations.py \
  tests/test_robot_io.py tests/test_command_sender.py -q
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run --project venv/pico --no-sync python -m pytest \
  tests/test_pico_retarget_profiles.py -q -k pm01
# ROS 回环测试自动隔离到 localhost 和域 171，不连接真机。
source scripts/pm01/env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync python -m pytest tests/test_pm01.py -q
# 完整启动脚本、键盘整形/保持/播放/阻尼、推理和录制。
# 需要已准备的资产；强制使用回环网口和域 171，反馈由测试节点生成。
PM01_RUN_LAUNCH_SMOKE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync \
  python -m pytest tests/test_pm01_launch.py -q -s
```
