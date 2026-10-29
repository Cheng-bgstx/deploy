---
title: Proteus 部署到 Galbot ET1
slug: /tutorials/proteus-et1
---

# Proteus 部署到 Galbot ET1

本教程用 sim2real 自己的运行时（而不是 et1-dep 的 Loco），把多机器人
[Proteus](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleTrack#multi-robot-training-proteus)
actor 部署到 Galbot ET1 上。内容包括：适配了什么、为什么这样适配，产物生成，
sim2sim 验收，以及上真机运行。

:::warning
ET1 不在 Proteus 训练用的 60 台机器人里。它的 contract 由 Unitree R1 的
contract 推导而来（见下文），目前只在 sim2sim 中验证过。前几次真机运行请按实验
对待：机器人挂在吊架上，先跑短的 Motion。
:::

## 适配内容

ET1 源自 Unitree R1：26 个关节相同，在 Proteus 的参考姿态下关节轴也完全一致。
厂商模型重 31.8 kg（R1 为 28.9 kg），腰、手臂和头的部分限位更窄。它的底层是
Unitree HG DDS：et1-dep 的 `libhdep_bridge_galbot_et1` 发布 `rt/lowcmd`、读取
`rt/lowstate`（`unitree_hg` 消息），并通过 MotionSwitcher 释放内置控制器。
因此 sim2real 用和 G1 相同的 Unitree Bridge 驱动它，只是电机槽位不同。

| 组成 | 位置 | 说明 |
| --- | --- | --- |
| 机器人配置 `et1`（别名 `galbot_et1`） | `sim2real/config/robots/et1.py` | 关节顺序、限位、力矩上限、HG 电机槽位（槽位 14、20、21、27、28 为空）。电枢惯量由 et1-dep Walker/Tracker 的增益反推（`kp = armature * (2π·10 Hz)²`）。 |
| MJCF | `third_party/galbot_et1/galbot_et1_26dof_unitree_frames.xml` | et1-dep 厂商模型，把肘和腕的 body 坐标系改回 Unitree R1 的定义（厂商模型把它们转了 90°）。物理完全不变；Galbot 自己重定向的 Motion 也用这套坐标系。由 `scripts/et1/prepare_mjcf.py` 生成。 |
| Proteus contract | `checkpoints/proteus/contracts/galbot_et1.json` | 由 `scripts/build_proteus_contract.py` 以 R1 为模板生成，见下文。 |
| Policy | `checkpoints/proteus/proteus_60/galbot_et1/` | 和 G1 policy 一样，由 `scripts/export_proteus_onnx.py` 包装发布的 actor。 |
| Motion | `checkpoints/galbot_et1/motions/` | 由 `scripts/et1/convert_motions.py` 把 et1-dep 的 Motion 转成 any4hdmi 数据集。 |
| Bridge | `scripts/g1/real_bridge.py --robot et1` | G1 的安全 Bridge，把 26 个关节映射到各自的 HG 电机槽位；`mode_machine` 取自第一帧 `rt/lowstate`，之后若变化则锁存阻尼。 |
| 部署 overlay | `checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml` | 由 `scripts/g1/prepare_proteus_deployment.py --robot et1 --limit-inset 0.01` 生成，见[上真机运行](#上真机运行)。 |
| 启动脚本 | `scripts/et1/start_proteus.sh` | 和 G1 的 `scripts/g1/start_proteus_pc.sh` 一样，在同一台机器上启动 Bridge 和 Tracking。 |

contract 直接在 ET1 MJCF 上运行 Proteus 自己的描述生成代码
（`ScaleTrack/scripts/multi_robot/build_robot_spec.py`）和形态描述子
（`supermimic/morphology.py`）：

- 槽位、符号和跟踪 body 由 ET1 几何得出，并且必须和 R1 一致（实际一致：24 个
  驱动关节都保持 R1 的槽位，符号均为 +1）；
- 默认姿态和站立髋高 `h` 沿用 R1：Proteus 从重定向的行走片段推导这两项，而 ET1
  没有这些片段；
- 刚度、阻尼和速度上限与所有已发布机器人一样，取 G1 同槽位数值，再按 ET1 的
  质量和 `h` 缩放，所以 ET1 的 kp/kd 等于 R1 的乘以质量比（约 1.1）；
- 力矩上限取自 MJCF 执行器，电枢惯量取自机器人配置；
- 头部关节锁定：不占槽位，和 Loco 一样用 et1-dep Loco 的 uncovered-joint 增益
  （kp 15.71、kd 1.00）保持在 0；
- 脚踝跟踪 body 是 `ankle_roll_link`（R1 跟踪的 toe link 在 ET1 模型里不存在）。

## 前置条件

把 et1-dep 交付仓库（Galbot 的 ET1 部署包，含 Loco）和 Proteus 仓库放在本仓库
旁边（`../et1-dep`、`../Proteus`）。下面的命令都在 sim2real 根目录、用根项目环境
运行：

```bash
uv sync --extra inference-cpu
```

## 生成产物

1. MJCF（mesh 和 DDS 关节表会一起拷到旁边）：

   ```bash
   uv run python scripts/et1/prepare_mjcf.py \
     --source ../et1-dep/storage/humanoid_description/galbot_et1
   ```

   脚本会对照厂商模型检查 body 位置、质量和质量矩阵，并打印最大差异（约 1e-12）。

2. Contract 和 policy：

   ```bash
   uv run python scripts/build_proteus_contract.py \
     --robot et1 --name galbot_et1 --template unitree_r1
   uv run python scripts/export_proteus_onnx.py \
     --contract galbot_et1 --robot et1 --contract-dir checkpoints/proteus/contracts
   ```

   第一条命令打印每个关节的槽位、默认值、增益和动作缩放，并在 contract 旁边写出
   `galbot_et1.yaml` 供审阅。第二条把包装后的模型和发布的 actor 做对比
   （`comparison.json`）。

3. Motion。两种 et1-dep 格式都支持：一个 `.npz`，或一个包含 `meta.json` 和
   `.npy` 数组的目录。文件名不是普通标识符时，用 `=<name>` 命名。数组属于不可信
   数据，所以用 `python -I` 和绝对路径运行：

   ```bash
   E=$PWD/../et1-dep/storage/motions/galbot_et1
   uv run --frozen python -I scripts/et1/convert_motions.py \
     --output-dir "$PWD/checkpoints/galbot_et1/motions" \
     --motions "$E/1010test/2-太极大师-20000轮.npz=taichi" \
       "$E/liedown_getup_data/0814getup_liedown/liedown_002/liedown_002_yanwei"
   ```

   每一帧转换结果的正运动学都必须复现存储的 body 位置和朝向；重定向到其他模型的
   Motion 会被拒绝。

## Sim2sim 验收

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run sim2real/sim_env/integrated_sim2sim.py \
  --robot et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml \
  --motion-path checkpoints/galbot_et1/motions/motions/taichi.npz
```

加上 `--headless --run-once --trajectory-output <file>.npz` 可以记录机器人和
Reference 的 body。也可以用分进程方式：`sim2real/sim_env/base_sim.py --robot et1`
加 `sim2real/rl_policy/tracking.py --robot et1`。

`taichi`（6894 帧，138 s）上的结果，误差为 14 个跟踪 body 的平均值：

| `proteus.reference_forcing` | 全局 body 误差 | 相对根节点 body 误差 | 最大倾角 | 根节点漂移 |
| --- | ---: | ---: | ---: | ---: |
| `true`（局部跟踪，真机用法） | 3.51 m | 0.038 m | 12.8° | 向后 5.1 m |
| `false`（全局跟踪，仅仿真） | 0.12 m（最大 0.20 m） | 0.033 m | 20.1° | 无 |

- 机器人全程没有摔倒；全局模式下没有一帧超过 Proteus 的 0.5 m 失败阈值。
- 局部跟踪会漂移（这个 Motion 上约 3.7 cm/s 向后）。这种模式下 Proteus 看不到
  根节点位置误差；G1 移植版在局部模式下同样会漂移。
- 这个 Motion 的脚比地面低约 3.5 cm（踝关节高度中位数 0.02 到 0.03 m，机器人上为
  0.06 m），所以机器人比 Reference 骨盆最多高约 8 cm。局部跟踪不受影响。

## 上真机运行

机器人侧沿用 [Proteus G1 PC 部署](../reference/proteus-g1-pc.md)：在 Tracker 就绪
且机器人遥控器上有一次实际的 A 按下之前，Bridge 不发出任何电机指令；超出关节限位
（±0.02 rad）的指令一律拒绝；出现故障，或状态、指令中断超过 I/O 窗口时，锁存阻尼。
G1 上得到的经验同样适用于 ET1。

### 部署 overlay

```bash
uv run python scripts/g1/prepare_proteus_deployment.py --robot et1 --limit-inset 0.01 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml
```

它在 policy 旁边写出 `policy-pc.yaml`，导出的 actor contract 保持不变。它把腰 roll 的
PD 设为 100/2.5（yaw 不变），Shaping 到 contract 的默认姿态，进入 Hold 前用 2 s 过渡到
Motion 第一帧。它把目标裁剪到关节限位以内 1%，并把目标变化率限制在 15 rad/s。ET1 的
踝 roll 范围只有 ±0.26 rad，所以用 1% 而不是 G1 的 5%。

### 演练

演练在 MuJoCo 中运行真实的 Tracking 控制器和 Bridge 的 `BridgeSafety`，不经过 DDS。
Shaping 期间根节点像理想吊架一样被固定，随后下放。演练依次按 A、A、Start、Start
（暂停）、Start 和 Select：

```bash
PYTHONPATH=$PWD SIM2REAL_ORT_NUM_THREADS=1 uv run python scripts/g1/validate_proteus_deployment.py \
  --robot et1 --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml \
  --motion-path checkpoints/galbot_et1/motions/motions/taichi.npz --walking-s 30
```

`taichi` 上的演练结果：

| Policy | 结果 |
| --- | --- |
| `policy.yaml`（不加 overlay） | 吊架在 Hold 中下放时，踝的目标超出关节限位（pitch 0.63 > 0.576，roll −0.45 < −0.26）；Bridge 锁存阻尼，机器人摔倒。 |
| overlay 使用 G1 的 5% 余量 | Bridge 没有故障，但深蹲姿态下踝的目标一直贴在软限位上，机器人在 Motion 第 24 秒摔倒。 |
| 5% 余量加 `policy.yaml` 的腰部 PD | 在 Motion 第 16 秒摔倒。 |
| `policy-pc.yaml`（1% 余量），Motion 60 s | 通过：无故障，无被拒指令，最大倾角 19°，相对 body 误差 8.6 cm，漂移 2.4 m。 |
| 在关节限位处裁剪，完整 Motion | 132 s 全程通过。 |

吊架模型不含绳索动力学；演练不能证明真机安全。

### 机器与安全

policy 和 Bridge 可以跑在 ET1 的机载计算机上，也可以跑在机器人网络
（`192.168.123.x`）里的另一台机器上。把 `third_party/galbot_et1/`、
`checkpoints/proteus/proteus_60/galbot_et1/` 和 `checkpoints/galbot_et1/` 拷到那台
机器的 checkout；Motion 数据集按相对路径找到 MJCF。安装 Unitree SDK：

```bash
uv sync --extra inference-cpu --extra robot-g1
```

- 把机器人挂在吊架上，硬件急停随时在手边。
- 在机器人上停掉 Loco 和所有 et1-dep 的 Controller
  （`./dist/cpp/loco/aarch64/tmux.sh stop`）。同一时间只能有一个进程发布
  `rt/lowcmd`；启动脚本在本机看到它们时会拒绝启动。
- 用 ET1 遥控器操作。Bridge 只认从 `rt/lowstate`（`wireless_remote`）读到的 A，
  所以键盘无法开始一次会话。

### 只读启动

```bash
ET1_INTERFACE=<nic> PROTEUS_READ_ONLY=1 scripts/et1/start_proteus.sh
```

Bridge 读取 `rt/lowstate` 并运行 Tracker，但不写任何电机指令，也不释放内置控制器
（30 s，`PROTEUS_READ_ONLY_S`）。在 `outputs/proteus-et1/<run>/bridge.log` 中检查：

- 先出现 `et1 mode_machine <n> from LowState`，再出现 `Tracker and LowState ready`；
- 按 A 出现 `Bridge state: arming`，按 Select 出现 `Bridge state: Select`。这说明遥控器
  的数据布局就是 Bridge 解码的 Unitree 布局（按键位于第 2 字节，A 是 bit 8，Select 是
  bit 3，Start 是 bit 2）；
- `bridge-state.json`（退出时写出）里的关节位置，按 `sim2real/config/robots/et1.py` 的
  关节顺序与机器人当前姿态一致。

### 受保护的会话

```bash
ET1_INTERFACE=<nic> scripts/et1/start_proteus.sh
```

启动脚本把 ONNX 限制为单线程，两个进程共用 250 ms 的 I/O 窗口
（`PROTEUS_IO_TIMEOUT_S`），并把记录写到 `outputs/proteus-et1/<run>/`。默认播放 `taichi`
的前 8 s（`PROTEUS_MOTION_PATH`、`PROTEUS_PLAYBACK_LIMIT_S`）。等出现
`Tracker and LowState ready` 后：

| 遥控器 | 行为 |
| --- | --- |
| 第一次 A | 释放内置控制器，在吊架上用 10 s Shaping 到默认姿态。 |
| Shaping 完成后第二次 A | 用 2 s 过渡到 Motion 第一帧，重置历史并进入 Hold。等出现 `Hold ready`。 |
| 放低吊架 | 这时才让双脚承重，保护仍然保留。G1 上在 Hold 之前下放失败过。 |
| Start | 播放 / 暂停 Motion。到达播放上限后 Tracker 保持最后一帧。 |
| Select | 优先级最高的软件停止：目标跟随实测关节角，增益保持不变；机器人不再主动平衡。 |

Select 或故障之后，用 A、A、Start 重新开始。Ctrl+C 或 SIGTERM 会停止两个进程，Bridge
会发送一小段阻尼。进程被杀、网线断开或 PC 死机都不是急停：机器人侧的通信超时和硬件
急停必须在现场验证。

这条路径的限制：

- 尚未在真机上测试。
- ET1 不支持 `--robot-io inline`：`unitree_interface` 只认识 G1。
- 局部跟踪会漂移（演练中 60 s 漂移 2.4 m），头部保持锁定。
- policy 使用 `rt/lowstate` 里的骨盆 IMU（MJCF 的 `imu` site 在 `pelvis_link` 上）。
