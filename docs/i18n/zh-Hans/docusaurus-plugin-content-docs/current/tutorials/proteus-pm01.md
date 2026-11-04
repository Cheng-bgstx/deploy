---
title: 众擎 PM01 上的 Proteus WAMSeq
slug: /tutorials/proteus-pm01
---

# 众擎 PM01 上的 Proteus WAMSeq

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

已完成：完整 ONNX 数值等价检查、四动作 integrated MuJoCo 验证、不依赖 SDK 的
映射/看门狗测试、使用机载消息定义的真实 ROS 回环收发，以及以已知 XR 姿态替代设备的
PICO IK/ZMQ/ONNX 测试。**尚未验证真机策略运动或真实 PICO 设备。** 检查时
`robotics.service` 未运行，只读检查没有反馈或命令订阅者。本次没有启动机器人服务或电机控制。

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

## 真机运行

用吊架保护 PM01，并准备好硬件停止手段。若运控服务未启动，在运控电脑启动现有厂商服务：

```bash
ssh user@192.168.0.163
sudo systemctl start robotics.service
systemctl status robotics.service --no-pager
```

按照[厂商部署说明](https://github.com/engineai-robotics/engineai_ros2_workspace/blob/community/README_CN.md)，
用遥控器进入 `pd_stand`，再进入 `joint_bridge`。启动器不切换机器人模式。回到 PC：

```bash
cd /path/to/sim2real
bash scripts/pm01/start_proteus.sh \
  checkpoints/engineai_pm01/motions/motions/walk1_subject1.npz --record
```

启动器自动加载 PM01 环境，先执行只读检查，再创建控制接口。默认 CPU 策略频率 50 Hz，
独立 ROS 命令线程 500 Hz。`PROTEUS_POLICY_CONFIG` 可覆盖策略路径。

键位：**i** 用 10 秒整形到动作首帧；**]** 开启策略 Hold；**空格**播放/暂停；
**o** 进入阻尼；**Ctrl+C** 退出。整形和初次 Hold 时保持吊架保护。控制接口在
`joint_bridge` 下启动后、整形前发送的是阻尼，不能独立支撑站立。显式切换离开该模式后，
接口把控制权交还固件。

发送器按名称映射关节，使用 `RL_PARSER=1` 转换串联踝关节命令，并拒绝维度错误或非有限值。
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
```
