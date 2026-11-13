# sim2real

root project 负责 inference、tracking policy，以及 MuJoCo 的 sim / sim2real runtime。Pico / XR teleoperation 工具请使用 `venv/pico`。

English version: [README.md](./README.md)

本仓库是我们各款人形机器人（G1、ET1、灵犀 X2、PM01 等）的真机部署代码，fork 自
[EGalahad/sim2real](https://github.com/EGalahad/sim2real)。请从[真机部署](#真机部署)开始；文档源文件在
[`docs/`](./docs/README.md)。

## Runtime Artifacts

大文件不放在 git 里。先从共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
下载，把 `checkpoints/` 和 `third_party/` 放到 repo 根目录。

目录结构和 onboard 依赖说明见 [Download Artifacts](./docs/artifacts.md)。

## 快速开始

```bash
uv sync --extra inference-cpu
```

在 G1 上安装或修复环境时，可以调用 repo 内置的 Codex skill
`$configure-g1-sim2real`；它位于 `.agents/skills/configure-g1-sim2real`。

运行离线动作跟踪（sim2sim）：

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy_config checkpoints/mimic-lite/v1_1/policy.yaml
```

两个进程都启动后，在 policy 终端按 `]` 开始跟踪，然后打开 `base_sim.py` 打印出来的 mjviser URL。虚拟 gantry / elastic band 的开关和长度在 viewer UI 里调。

## Migrating to sim2real

这个 repo 内置了一个 Codex skill，用来把外部训练 codebase 里的 policy 适配到 `sim2real`：

```text
.agents/skills/adapt-policy-to-sim2real
```

已经转好的 checkpoints 统一放在共享的
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
目录里。

目前已经支持的 adapted / distributed checkpoint：

| Policy family | Config path(s) | 说明 |
| --- | --- | --- |
| Mimic Lite v1.1 | `checkpoints/mimic-lite/v1_1/policy.yaml` | 使用 student-only reference-motion noise 的 8x8192 PPO-ROA student。 |
| MimicLite-ROA | `checkpoints/mimic-lite/roa/policy.yaml` | 最新 16x16384 PPO-ROA student release。 |
| MimicLite-PPO | `checkpoints/mimic-lite/ppo/policy.yaml` | 最新 16x16384 Huge PPO release。 |
| HEFT | `checkpoints/heft` | PMG 和 compliance 两个版本。 |
| HoloMotion v1.4.0 | `checkpoints/holomotion/v1_4_0/policy.yaml` | 使用官方未修改 ONNX：[HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx)，下载后放到 `checkpoints/holomotion/v1_4_0/policy.onnx`。 |
| SONIC release | `checkpoints/sonic/release` | Release G1 和 SMPL encoder variants。 |
| SONIC low-latency | `checkpoints/sonic/low_latency` | Low-latency G1 和 SMPL variants。 |
| SONIC v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | 使用 heading-normalized reference orientation 的 G1 policy。 |
| GRIT v0.0.1 | `checkpoints/grit/v0_0_1/policy.yaml` | 九帧参考上下文和十帧本体感知历史。 |
| ScaleBFM | `checkpoints/scalebfm` | [WeishuaiZeng/ScaleBFM](https://huggingface.co/WeishuaiZeng/ScaleBFM) 的 Humanoid Transformer XL 和 M ONNX exports。 |
| Proteus | `checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy.yaml` | [ScaleBFM](https://github.com/zengweishuai/ScaleBFM) 的 62 本体 Proteus WAMSeq actor（已在 G1 真机测试；wsq61 actor 可作为替代），由 `scripts/export_proteus_onnx.py` 包装成 G1 policy；见 [Run External Policies](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/run-external-policies.md#proteus)。 |
| Proteus ET1 | `checkpoints/proteus/proteus_60/galbot_et1/policy.yaml` | Galbot ET1（`--robot et1`）上的 Proteus，在 ET1 板载运行；见 [Proteus 在 Galbot ET1 板载运行](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-et1-onboard.md)。 |
| Proteus WAMSeq AgiBot Expedition A3 | `checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml` | 30 关节、多条 LAFAN 动作和 PICO；[教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-a3.md)。 |
| Proteus WAMSeq EngineAI T800 | `checkpoints/proteus/wsq61/engineai_t800/policy.yaml` | 25 关节、多条 LAFAN 动作和 PICO；[教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-t800.md)。 |
| Proteus WAMSeq EngineAI PM01 | `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml` | 24 关节 PM01/pm_v2、旧版 ROS 接口、LAFAN 动作和 PICO；[教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-pm01.md)。 |
| Proteus WAMSeq Primebot Q1 | `checkpoints/proteus/wsq61/primebot_q1/policy.yaml` | 22 关节、多条 LAFAN 动作和 PICO；[教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/proteus-q1.md)。 |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Latent-conditioned motion tracker。 |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | TeleopIT policy wrapper。 |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Humanoid-GPT policy wrapper。 |
| TWIST2 | `checkpoints/twist2/policy.yaml` | TWIST2 policy wrapper。 |

## 真机部署

每款机器人在 [`scripts/`](./scripts/README_zh.md) 下都有自己的文件夹，其中的 README 说明需要调整什么、如何运行。
[`scripts/README_zh.md`](./scripts/README_zh.md) 介绍所有机器人共用的部署流程（导出、部署 overlay、sim2sim 预演、
上机前检查、Shaping → Hold → Go → Final Hold 各阶段以及急停），以及接入新机器人的检查清单。

| 机器人 | 文件夹 | 真机状态 |
| --- | --- | --- |
| Unitree G1 | [`scripts/g1`](./scripts/g1/README_zh.md) | PC 部署已真机测试；机载 Orin 路线尚未真机测试 |
| Galbot ET1 | [`scripts/et1`](./scripts/et1/README_zh.md) | 已机载真机测试 |
| EngineAI PM01 | [`scripts/pm01`](./scripts/pm01/README_zh.md) | 吊架保护行走（PC 与板载） |
| Primebot Q1 | [`scripts/q1`](./scripts/q1/README_zh.md) | 仅验证反馈读取；策略接管未验证 |
| AgiBot A3 | [`scripts/a3`](./scripts/a3/README_zh.md) | 仅仿真 |
| EngineAI T800 | [`scripts/t800`](./scripts/t800/README_zh.md) | 仅仿真 |
| 智元灵犀 X2 | [`scripts/x2`](./scripts/x2/README_zh.md) | 板载吊架试验，尚未稳定 |
| LimX OLI | [`scripts/oli`](./scripts/oli/README_zh.md) | 仅仿真 |
| DH116S 灵巧手 | [`scripts/dh116s`](./scripts/dh116s/README_zh.md) | PICO 抓握控制 |

## 真机环境

机器人 SDK 不安装进通用 root 环境。G1 inline 部署使用
`uv sync --extra inference-cpu --extra robot-g1`。安装与部署命令见
[Robot I/O 模式](./docs/robot_io.md)。

Repo skills 统一放在 `.agents/skills/`，无需手动复制到
`~/.codex/skills/`。可以在 Codex 中显式调用
`$adapt-policy-to-sim2real`。

## 下一步

- [文档首页](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/intro.md)
- [快速上手](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/overview.md)
- [Root Project Setup](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/root-project.md)
- [离线动作跟踪教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation 教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/pico-teleoperation.md)
- [Scalecar 进车部署教程](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/scalecar-vehicle-entry.md)
- [Proteus 在 Galbot ET1 板载运行](./docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/reference/proteus-et1-onboard.md)

## Citation

如果 sim2real 对你的研究有所帮助，请引用：

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
