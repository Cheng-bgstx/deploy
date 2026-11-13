# Galbot ET1 板载运行 Proteus

ET1 在自身板卡（aarch64、CPython 3.10）上运行 Proteus actor（60 机器人 actor 或 WAM-62）以及 Galbot 自己的 et1-dep 跟踪策略：HDEP Bridge 200 Hz 绑定 CPU 7，CPU ONNX Tracking 50 Hz 绑定 CPU 5–6（A76 大核；5、6 号核跑 WAM actor 需 9.7 ms，p99 10.5；7 号核需 14 ms，4 号核要分担中断；A55 小核需 12–22 ms）。Galbot Loco 及其手柄程序不能同时运行，启动脚本会拒绝启动。

## 模型、配置与动作

`sim2real/config/robots/et1.py`（`galbot_et1`，别名 `et1`）以 et1-dep MJCF 为准：关节顺序和限位，其中腰部 yaw 为 ±0.785，而不是 Proteus 训练 spec 的 ±π/2。armature 和增益与 Proteus 及 Galbot 自己的 tracker 一致。MJCF 不放进仓库，通过 `ET1_DEP_ROOT` 指定（默认 `~/et1-dep`），其运动学与 Proteus 训练 MJCF 在所有共有 body 上误差小于 1e-7 m。

```bash
python scripts/export_proteus_onnx.py --contract galbot_et1 --robot galbot_et1   # 64 组输入误差为 0
python scripts/g1/prepare_proteus_deployment.py --robot galbot_et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml          # 生成 policy-pc.yaml
python scripts/prepare_proteus_test_motions.py --robot galbot_et1                # 转换 Proteus ScaleBridge 动作
python scripts/et1/slow_motion_tail.py \
  --source checkpoints/proteus/proteus_60/galbot_et1/motions/11_stand_walk_straight \
  --output checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close
```

`policy-pc.yaml` 用零宽软限位把头部锁在 0：actor 不跟踪头部 body，不锁时会把头转到约 1 rad。Proteus 测试动作里只有 `10_walk_b` 和 `11`–`14` 是向前走（`07`–`09` 相对骨盆是后退），只有 `11`–`14` 从站立起步。原速播放 `11_stand_walk_straight` 时，ET1 在结尾收脚处会深蹲；`11_slow_close` 把 10.4 s 之后放慢 2 倍并在结尾静止，机器人能停稳站立（仿真：前进 3.66 m，最小竖直度 0.941，保持期间漂移 0.01 m）。

MuJoCo 演练与真机使用同一套 Tracking、BridgeSafety 和 A → A → Start 流程：

```bash
python scripts/g1/validate_proteus_deployment.py --robot galbot_et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml \
  --motion-path checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close --walking-s 20
```

误差指标与策略看到的、按朝向对齐后的参考比较。动作短于演练日程时会在脚本暂停之前进入 Hold，`passed` 可能仅因模式序列为 false，请看稳定性字段。

## 板载组件

- `scripts/et1/hdep_shim.cpp`：给 ctypes 用的 HDEP C 接口（`hdep_py` 只有 x86_64 / CPython 3.11 版本），在板卡上编译。
- `scripts/et1/hdep_bridge.py`：HDEP 与 G1 Bridge 相同的 ZMQ LowState/LowCmd 协议互转，由 `BridgeSafety` 仲裁。启动只读；Tracker 就绪且有新的 A 才调用 `enter_debug_mode()`（真机 0.11–0.86 s），调用期间 Bridge 循环被阻塞。返回后 Bridge 先丢弃阻塞期间排队的命令（它们和阻塞一样旧，之前会被判为过期而锁定接管），继续发布状态，最多等 0.5 s 等 Tracker 的下一条命令。Select、Galbot 的 L1+下、超时和非法命令都会锁定。
- `sim2real/rl_policy/controllers/bridge_joystick.py`：`--controller bridge_joystick`，读取 Bridge 转发的手柄（Unitree 键位）。
- `scripts/et1/operator_keyboard.py`、`operator.sh`、`press_key.py`：无手柄时用键盘或脚本按键。按发送方分别合并，0.3 s 未更新即视为松开，SSH 断开会释放按键，但无法再停止机器人，务必保留龙门架和硬件急停。
- `scripts/et1/full_state_probe.cpp`：只读读取全部关节、IMU、手柄和机器人状态。

Tracker 进入控制循环前会冻结启动阶段创建的对象（`gc.freeze()`）。动作数据集和 FK 缓存会留下约 35 万个对象，在板卡上对它们做一次完整垃圾回收要 442 ms，曾让一步超过 0.2 s 的命令超时并锁定阻尼；冻结后一次完整回收约 1 ms。

`BridgeSafety` 对塑形（`init`）命令允许在实测位置与限位之间，因此停在模型限位以外的机器人（ET1 曾停在腰部 yaw 1.32 rad、左手腕 roll −1.93 rad）按 A 也能复位，但绝不会往外推；策略命令仍使用原限位。

## 部署与运行

在电脑上（首次用 `uv` 解析 aarch64 wheel，之后 rsync；只改动目标目录）：

```bash
ET1_HOST=192.168.121.42 ET1_TARGET=/root/proteus_et1_claude scripts/et1/deploy_onboard.sh
```

在板卡上开一个 SSH 终端启动（`PROTEUS_READ_ONLY=1` 为只读联调，不进入调试模式、不写电机）：

```bash
ssh -t 192.168.121.42 'cd /root/proteus_et1_claude/repo && \
  PROTEUS_MOTION=checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close bash scripts/et1/start_proteus_onboard.sh'
```

另开一个终端运行键盘：

```bash
ssh -t 192.168.121.42 /root/proteus_et1_claude/repo/scripts/et1/operator.sh
```

等待 `Tracker and LowState ready`。在龙门架上：`a` 用 10 s 塑形到站姿；再按 `a` 用 2 s 过渡到参考起始姿态并进入 Hold；看到 `Hold ready` 后放下龙门架；`s` 播放动作，结束后保持站立。空格（Select）锁定停止并失去主动支撑。在控制程序终端按 Ctrl-C 会停止两个进程，Bridge 退出时发阻尼，先用龙门架吊稳。

`PROTEUS_POLICY_DIR` 选择 actor：`checkpoints/proteus/proteus_60/galbot_et1`（默认）、`checkpoints/proteus/proteus_wamseq_62/galbot_et1`，或 Galbot 跟踪策略导出目录（见下）。`PROTEUS_PLAYBACK_LIMIT_S` 提前停止参考时钟并保持该帧。

## Galbot et1-dep 跟踪策略

`scripts/et1/prepare_galbot_track_policy.py` 把 et1-dep 跟踪 checkpoint（policy 协议 3.2：`policy.toml`、`inference.toml`、内含观测归一化的 `policy.onnx`、`motion.npz`）转成 Tracker 策略：`policy.yaml`、部署覆盖配置 `policy-pc.yaml`（限位内缩 5%、头部锁 0、15 rad/s 目标限速、2 s 参考过渡），以及 qpos 数据集格式的动作。111 维观测由 `sim2real/rl_policy/observations/galbot_track.py` 实现，其约定（6D 姿态展开方式、参考关节为绝对值、关节位置相对策略默认姿态）已用 checkpoint 的归一化统计核对。

```bash
python scripts/et1/prepare_galbot_track_policy.py \
  --checkpoint <addon>/storage/ckpt-shanshenbu --output checkpoints/galbot_track/galbot_et1/shanshenbu
python scripts/g1/validate_proteus_deployment.py --robot galbot_et1 --scenario walk-hold \
  --policy-config checkpoints/galbot_track/galbot_et1/shanshenbu/policy-pc.yaml \
  --motion-path checkpoints/galbot_track/galbot_et1/shanshenbu/motions/shanshenbu \
  --npz-playback-limit-s 49.78 --min-forward-m -100 --max-lateral-error-m 100 --action-delay-ms 10
```

`shanshenbu` 舞蹈（49.8 s）训练时没有指令延迟。仿真中延迟 0–10 ms 能跳完，从 12 ms 起约在第 8 s 摔倒（换成训练模型的 armature 0.01、无关节摩擦，10 ms 就摔）；动作开始后停在任何一帧都会走开或摔倒，因此只能截取开头的站立段（前 2.5 s）。由于没有位置反馈，仿真中整段舞蹈漂移约 4 m（Galbot 训练预览视频中约 0.5 m）。

## 真机状态

- 2026-10-29：proteus_60 先跑 `01_idle`，再跑 `11_slow_close`，均无 Bridge 故障。只读联调测得控制周期中位数 20.07 ms（p99 26 ms，最大 28 ms）。
- WAM-62 在真机上站立过（`01_idle`），也开始走过 `11_slow_close`；其中一次因 241 ms 的垃圾回收卡顿而锁定，现已由 `gc.freeze()` 解决。
- 2026-11-11，接管与垃圾回收修复之后：每次接管都成功（`enter_debug_mode` 0.11–0.14 s），没有卡顿或超时。proteus_60 `11_slow_close` 完整播放 14.5 s 并保持站立（行走中最小竖直度 0.965，保持时 0.993，朝向变化 −6.6°）。
- 2026-11-11，`shanshenbu`：开头站立段（截到 2.5 s）播放并保持正常。两次完整运行都跟住了前 8 s（包括快速转身），之后每次转身多转或少转 10–37°，身体最多倾斜约 30°，操作员分别在第 12.9 s 和 12.4 s 停止。板载被动监听测得软件链路平均约 8 ms（状态发布到命令发出中位数 0.8 ms，加上等待 200 Hz Bridge 与 50 Hz Tracker 周期），尚未计入软件无法测量的 HDEP/DDS 到驱动器这一段。
