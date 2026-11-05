# Proteus G1 在机器人 Orin 上运行

与[外接电脑部署](proteus-g1-pc.md)相同的 Bridge（200 Hz，BridgeSafety）和 Tracking（50 Hz，CPU ONNX），运行在 G1 自带的 Orin NX 上（`ssh g1`，192.168.123.164，机器人局域网为 `eth0`）。控制代码、安全行为和遥控器流程都不变：A → A → 等 Hold ready → 放下龙门架 → Start，Select 停止。电脑只负责部署、启动和收日志，电脑睡眠、电脑网卡故障或拔网线都不会再中断控制回路。

Orin 为 JetPack 5.1.1（Ubuntu 20.04、glibc 2.31、系统 Python 3.8），不能上网。所有内容都在一个目录 `/home/unitree/proteus_g1_onboard` 下，不做系统级安装，不需要 root：

| 路径 | 内容 |
| --- | --- |
| `python/` | 独立的 CPython 3.10.22（aarch64） |
| `site-packages/` | 按 `uv.lock` 为 aarch64 / manylinux_2_31 解析的离线依赖（torch CPU、mujoco、onnxruntime 等） |
| `cyclonedds/`、`cyclonedds-py/` | CycloneDDS 0.10.2（`unitree_sdk2py` 锁定的版本），在 Orin 上编译（没有 aarch64 预编译包） |
| `repo/` | `sim2real`、`scripts/g1`、wsq62 与 ht500 权重、动作片段；运行记录在 `repo/outputs/proteus-pc/<时间>/` |
| `hf/hub/` | G1 MJCF 缓存（`g1_xmls`）和 `h2_model` |

## 部署

```bash
scripts/g1/onboard.sh deploy
```

`scripts/g1/deploy_onboard.sh` 每个 `uv.lock` 只解析一次依赖和 Python（uv 由 `UV` 指定，默认 `outputs/g1-onboard-env/dl/uv/...`），下载并校验 CycloneDDS 源码，以 20 MB/s（`G1_RSYNC_BWLIMIT`）按白名单 rsync，避免挤占正在运行的电脑会话的机器人局域网；缺少时在 Orin 上编译 CycloneDDS（`scripts/g1/onboard_build_cyclonedds.sh`）。脚本从不启动控制器。`G1_POLICY_DIRS` 选择要同步的权重。

## 运行

```bash
PROTEUS_READ_ONLY=1 PROTEUS_READ_ONLY_S=60 scripts/g1/onboard.sh start   # 真实反馈加推理，不发电机指令
scripts/g1/onboard.sh start                                              # 受保护的真机会话（龙门架 + 急停）
scripts/g1/onboard.sh tail      # 跟踪 bridge.log / tracker.log；Ctrl+C 不影响机器人
scripts/g1/onboard.sh status
scripts/g1/onboard.sh stop      # SIGTERM：正常退出，Bridge 发送阻尼收尾
scripts/g1/onboard.sh logs      # 把 Orin 上的运行记录拷到 outputs/proteus-onboard/
```

`start` 让 `scripts/g1/start_proteus_onboard.sh` 脱离 SSH 运行，并转发 `PROTEUS_POLICY_DIR`、`PROTEUS_CLIP`、`PROTEUS_PLAYBACK_LIMIT_S`、`PROTEUS_IO_TIMEOUT_S`、`PROTEUS_HOLD_THROUGH_GAPS`、`PROTEUS_MOTION_PATH`、`SIM2REAL_ORT_NUM_THREADS` 和只读相关设置。本机电脑上有 Proteus 控制器在运行时，它拒绝启动会写电机的会话。电脑部署和板载部署绝不能同时控制机器人。Orin 的时钟比电脑慢约 12 分钟，运行目录名用的是 Orin 时间。

`start_proteus_onboard.sh` 设好环境并绑核后运行 `start_proteus_pc.sh`：Bridge 在 1 核，Tracking 在 2–5 核并用 4 个 ONNX 线程，只读的 LowState 中断监视器在 6 核、nice 10。Tracking 使用 `--controller bridge_joystick`：Bridge 在 `PORTS["gamepad"]` 上转发手柄，Tracker 自己不再订阅 DDS。用电脑默认的 `joystick` 控制器时，Tracker 自带的 CycloneDDS 接收线程在 Python 里解码 1 kHz 的 LowState，占约 70% 的一个核，每步被拉长到 25–33 ms（31–41 Hz）。电脑启动脚本仍默认 `joystick`（可用 `PROTEUS_CONTROLLER` 覆盖）。

## 实测耗时

2026-11-05 的 60 秒只读运行（wsq62，`walk_straight_short`）：

| | 每步 | 推理 | 观测构造 | 频率 | 超过 25 ms 的步 |
| --- | --- | --- | --- | --- | --- |
| Orin，Tracker 自带 DDS 手柄 | 25–33 ms | 13–17 ms | 5–8 ms | 31–41 Hz | 约一半 |
| Orin，`bridge_joystick` | 9.7 ms | 6.4 ms | 2.0 ms | 50.00 Hz | 2455 步中 13 步（最长 30 ms） |
| 外接电脑 | 5.2 ms | 3.6 ms | 1.05 ms | 50.00 Hz | 46595 步中 19 步（最长 107 ms） |

单独测试时，WAM 推理在 Orin 上 1 / 2 / 4 个 ONNX 线程分别需要 13.2 / 9.1 / 6.9 ms（wsq62 和 ht500 相同）。Orin 的数据是在 `sudo jetson_clocks`（CPU 1984 MHz、GPU 918 MHz）之后测的；单独锁频对 DDS 手柄的结果几乎没有改善，且重启后失效。除启动时的 4 步外，其余慢步约每 5 秒出现一次，表现为推理耗时 18–19 ms，像是有其他周期性任务与 2–5 核共用 CPU；机器人的深度推流进程会在这些核之间切换。板载路径尚未在真机上测试。
