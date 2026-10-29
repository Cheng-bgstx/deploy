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
| Bridge | `scripts/g1/real_bridge.py --robot et1` | 把 26 个关节映射到 HG 电机槽位，并从 `rt/lowstate` 复制 `mode_machine`。 |

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

policy 和 Bridge 可以跑在 ET1 的机载计算机上，也可以跑在机器人网络
（`192.168.123.x`）里的另一台机器上。把 `third_party/galbot_et1/`、
`checkpoints/proteus/proteus_60/galbot_et1/` 和 `checkpoints/galbot_et1/` 拷到那台
机器的 checkout；Motion 数据集按相对路径找到 MJCF。安装 Unitree SDK：

```bash
uv sync --extra inference-cpu --extra robot-g1
```

开始前：

- 把机器人挂在吊架上。
- 停掉 Loco 和所有 et1-dep 的 Controller（`./dist/cpp/loco/aarch64/tmux.sh stop`）。
  同一时间只能有一个进程发布 `rt/lowcmd`。
- Bridge 会像 Loco 一样，通过 MotionSwitcher 释放内置控制器。

终端 1，Bridge（`--interface` 设为连接机器人网络的网卡）：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run scripts/g1/real_bridge.py \
  --robot et1 --interface eth0
```

终端 2，policy：

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 uv run sim2real/rl_policy/tracking.py \
  --robot et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml \
  --motion-path checkpoints/galbot_et1/motions/motions/taichi.npz
```

policy 进程一启动，所有关节就以 policy 的增益保持在实测位置（E-stop）。然后在
键盘上：

1. `i`：Shaping，10 s 内过渡到 contract 的默认姿态。
2. 放低吊架，直到双脚承重，绳子留有余量。
3. `]`：Tracker 接管，进入 Hold，停在 Motion 第一帧。
4. `Space`：Go。再按一次 `Space` 暂停；Motion 结束后 Tracker 保持最后一帧。
5. `o`：随时 E-stop。关节跟随各自的实测位置，机器人变柔顺，所以吊架要随时待命。

这条路径的限制：

- ET1 不支持 `--robot-io inline`：`unitree_interface` 只认识 G1。请用 Bridge。
- et1-dep 从 `rt/lowstate`（`wireless_remote`）读取 ET1 遥控器。
  `--controller joystick` 读的是 `rt/wirelesscontroller`，尚未在 ET1 上确认。
  请用键盘。
- policy 使用 `rt/lowstate` 里的骨盆 IMU（MJCF 的 `imu` site 在 `pelvis_link` 上）。
