# Galbot ET1 启动脚本

[English](README.md) · 通用流程（Shaping → Hold → Go → Final Hold、急停、新机器人检查清单）：[../README_zh.md](../README_zh.md)

Galbot ET1 的整套部署都跑在机器人自身板卡上（aarch64、CPython 3.10）：HDEP Bridge（`hdep_bridge.py`，200 Hz）
通过一个小型 C++ shim 与机器人通信，并在本地提供与 G1 Bridge 相同的 ZMQ LowState/LowCmd 协议；共用的 Tracker
（`sim2real/rl_policy/tracking.py`）在旁边以 50 Hz 做 CPU ONNX 推理。可运行 Proteus actor（proteus_60、WAM-62）
以及 Galbot 自己的 et1-dep 跟踪策略。受支持的路线就是这套 HDEP 板载栈；另一分支上仅用于仿真的 Unitree-HG-DDS
ET1 路线在本次合并中有意未保留。

## 状态

| 项目 | 内容 |
| --- | --- |
| 真机 | 已上机测试。修复接管与 GC 卡顿（提交 `8da006e`）后，每次接管均成功，proteus_60 的 `11_slow_close` 完整播放并保持站立；WAM-62 已能站立（`01_idle`）；`shanshenbu` 开头站立可保持，完整舞蹈约 8 s 后转身过冲。详见[真机状态](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-et1-onboard.md#真机状态) |
| `--robot` | `galbot_et1`（别名 `et1`），见 `sim2real/config/robots/__init__.py` |
| 关节数 | 26：每条腿 6 个、腰 roll/yaw、每条手臂 5 个、头 pitch/yaw |
| Robot I/O | `--robot-io zmq`（127.0.0.1）⇄ `hdep_bridge.py` ⇄ HDEP（`libhdep_bridge_galbot_et1.so.2.2.0`） |
| 控制器 | `--controller bridge_joystick`（Bridge 转发的手柄，Unitree 键位） |
| Tracker 运行位置 | ET1 板卡，绑定 CPU 5,6；Bridge 绑定 CPU 7 |
| 默认策略 | `checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml` |
| 其他策略 | `checkpoints/proteus/proteus_wamseq_62/galbot_et1/policy-pc.yaml`、`checkpoints/galbot_track/galbot_et1/<name>/policy-pc.yaml` |
| 默认动作 | `checkpoints/proteus/proteus_60/galbot_et1/motions/01_idle` |

## 文件

`scripts/et1/`：

| 文件 | 用途 |
| --- | --- |
| `deploy_onboard.sh` | PC → 板卡：一次性解析 aarch64/CPython 3.10 依赖，rsync 代码、checkpoint、MJCF 和 HF 资源，在板卡上编译 shim 与 probe。不会启动控制器。 |
| `start_proteus_onboard.sh` | 板卡启动脚本：检查是否有其他控制器，按 CPU 绑定启动 Bridge 和 Tracker，日志写到 `logs/run-<UTC>/`。 |
| `hdep_bridge.py` | 200 Hz 的 HDEP ⇄ ZMQ Bridge，由 `BridgeSafety` 仲裁；通过 `enter_debug_mode()` 延迟接管。 |
| `hdep_shim.cpp` | 给 ctypes 用的 HDEP C 接口（`hdep_py` 只有 x86_64 / CPython 3.11 版本），在板卡上编译为 `libet1_hdep_shim.so`。 |
| `full_state_probe.cpp` | 只读输出全部关节电机、IMU、手柄和机器人状态，在板卡上编译为 `probe/full_state_probe`。 |
| `operator.sh` | 在板卡上运行 `operator_keyboard.py` 的包装脚本（`ssh -t`）。 |
| `operator_keyboard.py` | 没有手柄时的键盘操作端（`a`、`s`、空格/`x`）。 |
| `press_key.py` | 发送一次按键（`a`、`s`、`space`），用于脚本或远程操作。 |
| `prepare_galbot_track_policy.py` | 把 et1-dep 跟踪 checkpoint（策略协议 3.2）转换为 Tracker 策略和 qpos 动作。 |
| `slow_motion_tail.py` | 放慢 qpos 动作的结尾并追加静止段（用于生成 `11_slow_close`）。 |

仓库其他位置的相关文件：

- `sim2real/config/robots/et1.py`：关节顺序、限位、armature、增益、站立姿态、MJCF 路径。
- `sim2real/rl_policy/robot_io/zmq.py`：Tracker 侧的 ZMQ 协议（`SIM2REAL_IO_TIMEOUT_S`）。
- `sim2real/rl_policy/robot_io/bridge_safety.py`：与 G1 Bridge 共用的命令仲裁。
- `sim2real/rl_policy/controllers/bridge_joystick.py`：从 `PORTS["gamepad"]`（5594）读取手柄。
- `sim2real/utils/operator_keys.py`：`PORTS["operator"]`（5595）上的操作按键消息。
- `sim2real/rl_policy/observations/galbot_track.py`：Galbot 跟踪 actor 的 111 维观测。
- `scripts/g1/prepare_proteus_deployment.py`、`scripts/g1/validate_proteus_deployment.py`、`scripts/export_proteus_onnx.py`、`scripts/prepare_proteus_test_motions.py`：通用 Proteus 工具，配合 `--robot galbot_et1` 使用。
- 测试：`tests/test_et1_bridge.py`、`tests/test_et1_config.py`、`tests/test_galbot_track_obs.py`。
- 文档：`docs/proteus_et1_onboard.md` 及其中文版。

## 准备

PC 上一次性准备：

1. 获取 Galbot 的 et1-dep 交付包，并用 `ET1_DEP_ROOT` 指向它（默认 `~/et1-dep`）。MJCF
   `storage/humanoid_description/galbot_et1/galbot_et1_26dof.xml` 不放进仓库，`et1.py` 从这里读取。
   `deploy_onboard.sh` 也从这里拷贝 aarch64 的 HDEP 头文件/库、CycloneDDS 库以及 `galbot_et1` 的 HDEP bridge 库。
2. 运行时产物放在 `checkpoints/` 下（已被 git 忽略；共享 Drive 目录见 `AGENTS.md`）。部署脚本会同步
   `checkpoints/proteus/proteus_60/galbot_et1`、`checkpoints/proteus/proteus_wamseq_62/galbot_et1`，
   以及存在时的 `checkpoints/galbot_track/galbot_et1`。
3. PC 的 `~/.cache/huggingface/hub` 中需要有 `models--elijahgalahad--h2_model` 和 `models--elijahgalahad--g1_xmls`；
   模块加载时会导入它们，部署脚本会 rsync 到板卡。
4. PC 上有 `uv`（可用 `UV=` 覆盖），并能 `ssh`/`rsync` 到板卡（用户名和端口由 `~/.ssh/config` 提供）。

板卡上（由 `deploy_onboard.sh` 完成）：

- 使用系统 `python3`（CPython 3.10，aarch64），依赖装在 `<target>/site-packages`
  （`uv export --extra inference-cpu`、`--python-platform aarch64-manylinux_2_35`、CPU 版 PyTorch 源）。
  `outputs/et1-onboard-env` 中的缓存在 `uv.lock` 不变时复用（可用 `ET1_ENV_CACHE` 覆盖）。
- 用 `g++`（C++17）基于 `probe/include` 和 `probe/lib` 编译 `probe/libet1_hdep_shim.so` 与 `probe/full_state_probe`。
- 目标目录结构：`repo/ site-packages/ et1-dep/storage/ hf/ probe/{lib,bridge,...} logs/`。
- `start_proteus_onboard.sh` 会导出 `HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME=<target>/hf`。
  手工准备的命令里也要带上这两个 HF 变量。

## 可调项

| 项目 | 当前值 | 设置位置 |
| --- | --- | --- |
| Armature 分组（大关节 / 踝 pitch / 踝 roll / 小关节） | 0.01890 / 0.01492 / 0.00619 / 0.00398 | `sim2real/config/robots/et1.py` |
| 安全 PD kp | 74.6 / 58.9 / 24.4 / 15.7 | `et1.py`（`_KP`） |
| 安全 PD kd（也是故障阻尼） | 4.75 / 3.75 / 1.55 / 1.00 | `et1.py`（`_KD`） |
| 关节范围 | 取自 et1-dep MJCF；腰 yaw ±0.7854（不是 Proteus spec 的 ±π/2） | `et1.py`（`_LIMITS`） |
| 关节摩擦 | 0.1（et1-dep MJCF 数值） | `et1.py` |
| 站立姿态 / 根高度 | Galbot walker 姿态，根 z 0.74 | `et1.py`（`_STAND_JOINT_POS`、`default_qpos`） |
| 头部锁定 | 头 pitch/yaw 软限位 `[0, 0]`（actor 不跟踪头部，不锁会转到约 1 rad） | `scripts/g1/prepare_proteus_deployment.py`（`DEFAULT_LOCKED_JOINTS`、`--locked-joints`） |
| 部署 overlay | 软限位内缩 5%、目标转速上限 15 rad/s、2 s 进入参考 | `prepare_proteus_deployment.py`、`prepare_galbot_track_policy.py` |
| CPU 绑定 | Bridge `PROTEUS_BRIDGE_CPU=7`，Tracker `PROTEUS_TRACKER_CPU=5,6`；ORT/BLAS 单线程 | `start_proteus_onboard.sh` |
| I/O 超时 | `PROTEUS_IO_TIMEOUT_S=0.2` → `SIM2REAL_IO_TIMEOUT_S` 和 Bridge `--timeout-s` | `start_proteus_onboard.sh` |
| 操作按键有效期 | 0.3 s（`OPERATOR_FRESH_S`） | `hdep_bridge.py` |
| 接管后等待恢复 | 0.5 s（`TAKEOVER_RESUME_S`） | `hdep_bridge.py` |
| HDEP 网段 / 频率 | `--interface-prefix 192.168.123`、`--rate 200` | `hdep_bridge.py` |
| 板卡地址 / 目标目录 | `ET1_HOST=192.168.121.42`、`ET1_TARGET=/root/proteus_et1_claude` | `deploy_onboard.sh` |

策略与动作选择（板卡启动脚本）：

- `PROTEUS_POLICY_DIR`：`checkpoints/proteus/proteus_60/galbot_et1`（默认）、
  `checkpoints/proteus/proteus_wamseq_62/galbot_et1` 或 `checkpoints/galbot_track/galbot_et1/<name>`。
  启动脚本会检查 YAML 含有 `galbot_track` 段或 `proteus.reference_forcing: true`。
- `PROTEUS_MOTION`：动作数据集目录（需含 `manifest.json`）。
- `PROTEUS_PLAYBACK_LIMIT_S`：参考时钟停在该时刻并保持（`--npz-playback-limit-s`）。
- `PROTEUS_READ_ONLY=1`（可加 `PROTEUS_MAX_RUNTIME_S`）：空跑，不进入 debug 模式，也不写电机。
- `PROTEUS_RUN_DIR`、`PROTEUS_ROOT`：覆盖日志目录 / 安装根目录。

重新生成产物（PC）：

```bash
python scripts/g1/prepare_proteus_deployment.py --robot galbot_et1 \
  --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy.yaml          # 生成 policy-pc.yaml
python scripts/prepare_proteus_test_motions.py --robot galbot_et1                # Proteus 测试动作
python scripts/et1/slow_motion_tail.py \
  --source checkpoints/proteus/proteus_60/galbot_et1/motions/11_stand_walk_straight \
  --output checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close       # 10.4 s 后放慢 2 倍，结尾静止 1 s
```

`slow_motion_tail.py` 参数：`--after-s 10.4`、`--factor 2.0`、`--still-s 1.0`；输出已存在时拒绝覆盖。

et1-dep 跟踪策略：`prepare_galbot_track_policy.py` 需要包含 `policy.toml`（协议 3.2、50 Hz）、`inference.toml`、
`policy.onnx` 和 `motion.npz` 的 checkpoint 目录，并需要系统 `python3 >= 3.11` 来读取 TOML。它会生成 `policy.onnx`、
`policy.yaml`、`policy-pc.yaml`（锁定 fixed 关节、内缩 5%、`--max-target-velocity-rad-s 15`、`--reference-entry-s 2`）
以及 `motions/<name>`：

```bash
python scripts/et1/prepare_galbot_track_policy.py \
  --checkpoint checkpoints/et1-dep-addon/26-1111/storage/ckpt-shanshenbu \
  --output checkpoints/galbot_track/galbot_et1/shanshenbu
```

## 使用

1. 在 MuJoCo 中演练（PC），与真机使用同一套 Tracking、BridgeSafety 和 A → A → Start 流程：

   ```bash
   python scripts/g1/validate_proteus_deployment.py --robot galbot_et1 \
     --policy-config checkpoints/proteus/proteus_60/galbot_et1/policy-pc.yaml \
     --motion-path checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close --walking-s 20
   ```

   动作短于演练日程时会提前进入 Hold，`passed` 可能仅因模式序列为 false，请看稳定性字段。
   Galbot 跟踪策略的演练见文档中的 `--scenario walk-hold` 示例。

2. 从 PC 部署（只改动板卡上的 `ET1_TARGET`）：

   ```bash
   ET1_HOST=192.168.121.42 ET1_TARGET=/root/proteus_et1_claude scripts/et1/deploy_onboard.sh
   ```

3. 停掉板卡上的 Galbot Loco、其手柄程序和其他 tracker。在一个 SSH 终端中启动：

   ```bash
   ssh -t 192.168.121.42 'cd /root/proteus_et1_claude/repo && HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
     PROTEUS_MOTION=checkpoints/proteus/proteus_60/galbot_et1/motions/11_slow_close \
     bash scripts/et1/start_proteus_onboard.sh'
   ```

   Galbot 跟踪策略，仅开头站立段：
   `PROTEUS_POLICY_DIR=checkpoints/galbot_track/galbot_et1/shanshenbu PROTEUS_MOTION=checkpoints/galbot_track/galbot_et1/shanshenbu/motions/shanshenbu PROTEUS_PLAYBACK_LIMIT_S=2.5`。

4. 没有手柄时，在第二个终端打开键盘操作端：

   ```bash
   ssh -t 192.168.121.42 /root/proteus_et1_claude/repo/scripts/et1/operator.sh
   ```

   在板卡上用脚本按键：`python scripts/et1/press_key.py {a,s,space}`（`PYTHONPATH` 与 `operator.sh` 相同）。

5. 操作顺序（机器人挂在龙门架上）：

   | 步骤 | 手柄 | 键盘 | 效果 |
   | --- | --- | --- | --- |
   | 等待 | — | — | `bridge.log` 出现 `Tracker and LowState ready` |
   | 1 | A | `a` | Bridge 进入 debug 模式（接管）；10 s 内 Shaping 到站立姿态 |
   | 2 | A | `a` | 2 s 过渡到参考起始姿态，`tracker.log` 出现 `Hold ready` |
   | — | — | — | 放下龙门架，但保持随时可用 |
   | 3 | Start | `s` | 播放动作；`11_slow_close` 结束时保持站立 |
   | 停止 | Select，或 L1 + 下 | 空格或 `x` | 锁定停止，撤去主动支撑 |

   启动脚本会打印本次运行目录，可用 `tail -f <run>/bridge.log <run>/tracker.log` 跟踪日志。

6. 关闭：先用龙门架托住机器人，再在启动终端按 Ctrl-C。两个进程都会停止；Bridge 会锁定并在退出前发送约 0.2 s 阻尼。
   日志、`bridge-state.json` 和 `tracking.npz` 在 `<target>/logs/run-<UTC>/`。

不接管、只读查看板卡状态：`probe/full_state_probe 192.168.123 probe/bridge/libhdep_bridge_galbot_et1.so.2.2.0 [SAMPLES] [INTERVAL_MS]`
（在 `<target>` 下运行，并设置 `LD_LIBRARY_PATH=probe/lib`）。

## 安全注意事项与已知问题

- **只能有一个电机写入方。** Galbot Loco、其手柄程序（`gamepad-recover*`、`motion-sender`）以及其他 tracker
  不能同时运行。启动脚本发现它们会拒绝启动，`logs/active.lock` 防止同时运行第二个 supervisor。
- **键盘本身无法停住机器人。** SSH 断开只会释放按键（0.3 s 后过期）。每次运行都要准备好龙门架和硬件急停。
- **接管会阻塞 Bridge。** 真机上 `enter_debug_mode()` 耗时 0.11–0.86 s。Bridge 会丢弃阻塞期间排队的命令，并最多等待
  0.5 s 等 Tracker 发来新命令；Tracker 一直不发则接管失败。只有在 Tracker 就绪且有新的 A 按下时才会请求接管。
- **GC 冻结。** 板卡上对启动阶段约 35 万个对象做一次完整垃圾回收要 442 ms，曾触发 0.2 s 超时并锁定阻尼；
  现在 Tracker 在进入控制循环前调用 `gc.freeze()`（`base_policy.py`）。
- **超限位时的 Shaping。** `BridgeSafety` 接受介于实测位置和限位之间的 Shaping（`init`）目标，因此静止时超出模型限位的
  机器人（实测腰 yaw 1.32 rad）会被拉回，而不会被推得更远。
- **头部默认锁定。** proteus_60 另有 `policy-pc-head-free.yaml`，但不是默认配置。
- **`shanshenbu`** 训练时没有命令延迟；真机上约 8 s 后转身过冲并倾斜，只有开头站立段（前 2.5 s）可以截断后保持。
- Bridge 退出时不做 HDEP teardown（`os._exit`），因为销毁 HDEP bridge 可能与 CycloneDDS 回调竞争。

## 延伸阅读

- 中文：[proteus-et1-onboard.md](../../docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-et1-onboard.md)
- English: [docs/proteus_et1_onboard.md](../../docs/proteus_et1_onboard.md)
- 通用流程：[../README_zh.md](../README_zh.md)
