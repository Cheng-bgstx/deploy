---
title: 在智元灵犀 X2 上部署 Proteus WAMSeq
slug: /tutorials/proteus-x2
---

# 在智元灵犀 X2 上部署 Proteus WAMSeq

本适配针对 Proteus **`lingxi_x2` 31 关节合约**，使用公开的 X2 Ultra v1.3 模型，
接入本仓库的 Tracking 和集成 sim2sim。机器人别名为 `x2`、`lingxi_x2`、`agibot_x2`。
策略是原始共享 WAMSeq actor，使用合约中的形态描述和带符号动作映射固定为 X2，无需
微调。已检查两个完整动态动作的 sim2sim，真机尚未验证。

## 准备模型和策略

使用根项目 Python 3.10 环境，需要 MuJoCo 3.11、ONNX / CPU ONNX Runtime、Pillow、
joblib 和 `ffmpeg`。`--no-sync` 用于绕过当前仓库已有 ARM GPU 依赖的全平台解析问题。

```bash
git -C ../Proteus worktree add --detach ../Proteus-wamseq origin/wamseq
gh repo clone AgibotTech/agibot_x2_urdf ../agibot_x2_urdf -- --depth 1
uv run --no-sync python scripts/x2/prepare_assets.py \
  --description-root ../agibot_x2_urdf --proteus-root ../Proteus-wamseq
uv run --no-sync python scripts/x2/export_policy.py --proteus-root ../Proteus-wamseq
```

已有工作区直接复用，不必重复克隆。本次验证版本：Proteus `1e0e038`、官方模型
`575cc6b`。模型为 `X2_URDF-v1.3.0/x2_ultra.xml`，官方 v1.4 是不同版本，参见
[官方模型说明](https://github.com/AgibotTech/agibot_x2_urdf)。

`prepare_assets.py` 生成 `third_party/lingxi_x2/model/scene_tracking.xml`，复制网格
和许可证，移除原 actuator/sensor 块，并应用训练限位、armature 和零被动阻尼/摩擦。
集成仿真添加 31 个按关节命名的力矩执行器，并按训练设置关闭自碰撞。训练默认姿态下
各关节轴误差均为零。官方模型总质量 **41.966473 kg**，训练描述为 **41.80159766 kg**，
差值 **0.16487534 kg**；保留官方模型惯量。12 个关节范围与训练不同，仿真模型显式
覆盖为训练范围。差异与文件哈希均记录于 `validation.json`。AimDK 后端独立使用官方
物理限位与训练限位的交集，不能用更宽的训练范围扩大硬件范围。

导出结果位于 `checkpoints/proteus/proteus_wamseq_62/lingxi_x2/`，包含一个完整的
`policy.onnx`、原始合约、`comparison.json` 和两个 YAML：

| 配置 | 根部平移来源 | 用途 |
| --- | --- | --- |
| `policy-sim.yaml` | 仿真测得的世界系根部位置，`reference_forcing: false` | 与原仓库 MuJoCo 观测语义一致的 sim2sim |
| `policy.yaml` | 参考锚点，`reference_forcing: true` | 没有世界系定位反馈的 SDK 局部跟踪 |

两者绑定 `proprioception[10,68]`、`action_history[10,31]`、`history_valid[10]` 和
`task[6,253]`，输出 `action[31]`、`next_slot_action[31]`。历史、有效掩码、关节符号、
动作缩放和增益均来自原合约。64 组 seed=0 输入对比中，封装前后输出误差为零；ONNX
为 IR 8、opset 17。`--release proteus_wamseq_62_full` 可导出另一份 actor。

## 动态参考与视频

本次使用 [sonic-x2](https://github.com/meetsitaram/sonic-x2) 公开的 X2 动作姿态，
版本 `c995944`，仅使用动作 PKL 和显式关节列顺序，不加载其中的 SONIC 策略或部署
调参。转换器按名称映射 31 列关节，将源根部四元数 XYZW 转为 WXYZ，保留绝对根部
高度，将首帧 XY 平移到零。any4hdmi manifest 保存源帧率，运行时重采样到 50 Hz。

```bash
gh repo clone meetsitaram/sonic-x2 ../sonic-x2-reference -- --depth 1 --filter=blob:none --sparse
git -C ../sonic-x2-reference sparse-checkout set motions scripts assets/mjcf
uv run --no-sync python scripts/x2/convert_reference.py \
  --clip gangam_dance --output outputs/x2/gangam_dance
uv run --no-sync python scripts/x2/convert_reference.py \
  --clip relaxed_walk --output outputs/x2/relaxed_walk
uv run --no-sync python scripts/x2/render_sim2sim.py \
  --motion-path outputs/x2/gangam_dance --duration 11.2 \
  --output outputs/x2/x2_wamseq_gangam_dance.mp4 \
  --label 'X2 Gangnam dance reference | Proteus WAMSeq | measured world root'
uv run --no-sync python scripts/x2/render_sim2sim.py \
  --motion-path outputs/x2/relaxed_walk --duration 17 \
  --output outputs/x2/x2_wamseq_relaxed_walk.mp4 \
  --label 'X2 relaxed walk reference | Proteus WAMSeq | measured world root'
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_gangam_dance.mp4
uv run --no-sync python scripts/x2/summarize_sim2sim.py outputs/x2/x2_wamseq_relaxed_walk.mp4
```

视频右侧为集成物理仿真，左侧为参考姿态播放，共享同一相机，原速、25 fps、1280×720。
seed=0，策略 50 Hz，物理步长 0.005 秒。初始暂停 1 秒和结束保持末帧期间策略均持续
运行。每个视频旁保存同次运行的 `.root.npz`、`.trajectory.npz`、`.json`、
`.metrics.json` 和 JPEG 预览。

本地验证记录（2026-11-02），误差统计对象为播放期间的 14 个合约跟踪身体：

| 参考动作 | 动作 / 视频时长 | 世界系 / 局部平均身体位置误差 | 世界系根部末端位置误差 |
| --- | --- | --- | --- |
| 江南舞 | 9.2 / 11.2 秒 | 8.24 / 5.42 厘米 | 1.00 厘米 |
| 放松行走 | 14.82 / 17 秒 | 4.60 / 4.05 厘米 | 1.72 厘米 |

两段都完整跑完，没有跌倒。舞蹈期间根部高度 0.608–0.665 米，行走期间为
0.636–0.667 米。世界系误差直接比较身体位置；局部误差将两套身体分别变换到各自的
骨盆坐标系。根部记录还包含旧版“分别在各自初始根坐标系中表达位移”的误差（舞蹈
2.46 厘米，行走 45.54 厘米），因初始骨盆朝向不同，与世界系末端误差不是同一指标。
指标 JSON 对这两类量作了明确命名。

## AimDK 部署

inline 后端遵循 [AimDK 关节接口](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/control_mod/joint_control.html)
和[骨盆 IMU 说明](https://x2-aimdk.agibot.com/zh-cn/latest/Interface/hal/sensor.html)。
订阅四组关节状态及 `/aima/hal/imu/torso/state`（虽然叫 torso，实际为胯部 IMU），
按组/名称映射数组，将 ROS XYZW 四元数转为 WXYZ。独立 ROS 定时器以 500 Hz 复发
最新命令，策略推理为 50 Hz。反馈或命令超过 250 ms、异常反馈以及 `zero` 模式均将
位置刚度、速度和前馈力矩清零，保留阻尼。正常退出时发送 5 次阻尼命令。HAL 本身
没有发布者死亡超时保护，因此此线程不能覆盖进程被强杀或电脑断电的情况。

```bash
source /opt/ros/humble/setup.bash
source /path/to/aimdk/install/setup.bash
# 真机取得 HAL 控制权前，需要显式释放原生控制器：
# aima em stop-app mc
bash scripts/x2/start_proteus.sh /path/to/x2/any4hdmi_dataset
```

接管前设置正确的 ROS domain/网络并确认反馈正常。脚本设置离线环境、CPU 推理、
`robot=x2`、`robot_io=inline` 和键盘控制：`i` 用 10 秒插值到第 0 帧，`]` 启用策略
Hold，空格播放/暂停，`o` 阻尼，Ctrl+C 退出。首次真机测试需支撑机器人。

`policy.yaml` 中的 `robot_io_options` 配置频率、超时、IMU 话题、四元数
`imu_to_pelvis_wxyz` 和 `head_pitch_enabled`。默认单位 IMU 旋转对应模型；真机必须
使用出厂标定外参，不能只依赖话题 `frame_id`。旧版 X2 Ultra 仅头部 yaw 有效，因此
默认不驱动 head pitch。本适配没有验证其他 X2 修订版本。没有世界系定位时，不能在
真机使用 `policy-sim.yaml`。

回归检查：`uv run --no-sync python -m pytest tests/test_x2.py tests/test_oli.py
tests/test_proteus_observations.py tests/test_et1_proteus.py tests/test_robot_io.py
tests/test_command_sender.py -q`，并通过 Python 语法和 shell 参数检查。
`third_party/`、`checkpoints/`、`outputs/` 下的运行资源不纳入 Git。
