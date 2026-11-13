# sim2real

A lightweight and modular sim2sim and sim2real deployment stack.

Chinese version: [README_zh.md](./README_zh.md)

This repository is the real-robot deployment stack for our humanoids (G1, ET1, Lingxi X2, PM01, …). It is
forked from [EGalahad/sim2real](https://github.com/EGalahad/sim2real). Start with
[Deploying On Robots](#deploying-on-robots); the documentation sources are under [`docs/`](./docs/README.md).

## Runtime Artifacts

Large runtime artifacts are not stored in git. Download the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder and place `checkpoints/` and `third_party/` at the repo root.

See [Download Artifacts](./docs/reference/artifacts.md) for the expected directory
layout and onboard dependency notes.

## Quick Start

```bash
uv sync --extra inference-cpu
```

For G1 onboard installation or repair, invoke the repository Codex skill
`$configure-g1-sim2real` from `.agents/skills/configure-g1-sim2real`.

Run offline motion tracking (sim2sim):

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
uv run sim2real/rl_policy/tracking.py --robot g1 \
  --policy_config checkpoints/mimic-lite/v1_1/policy.yaml \
  --motion_path hf://elijahgalahad/any4hdmi-g1-lafan/motions/walk1_subject1.npz
```

After both processes are up, press `]` in the policy terminal to start. Open the mjviser URL printed by `base_sim.py`, then use the Elastic Band controls in the viewer UI to disable or tune the virtual gantry.

## Migrating to sim2real

This repo includes a Codex skill for adapting policies trained in external codebases into `sim2real`:

```text
.agents/skills/adapt-policy-to-sim2real
```

Converted checkpoints are distributed through the shared
[sim2real artifacts](https://drive.google.com/drive/folders/1lrPyiiy7anyG3P4wHNIQQQlydboLPd9e)
folder.

Currently supported adapted / distributed checkpoint families:

| Policy family | Config path(s) | Notes |
| --- | --- | --- |
| Mimic Lite v1.1 | `checkpoints/mimic-lite/v1_1/policy.yaml` | 8x8192 PPO-ROA student with student-only reference-motion noise. |
| MimicLite-ROA | `checkpoints/mimic-lite/roa/policy.yaml` | Latest 16x16384 PPO-ROA student release. |
| MimicLite-PPO | `checkpoints/mimic-lite/ppo/policy.yaml` | Latest 16x16384 Huge PPO release. |
| HEFT | `checkpoints/heft` | PMG and compliance variants. |
| HoloMotion v1.4.0 | `checkpoints/holomotion/v1_4_0/policy.yaml` | Uses the official unmodified ONNX from [HorizonRobotics/HoloMotion_models](https://huggingface.co/HorizonRobotics/HoloMotion_models/resolve/main/HoloMotion_motion_tracking_model_v1.4.0/exported/model_14000.onnx); place it at `checkpoints/holomotion/v1_4_0/policy.onnx`. |
| SONIC release | `checkpoints/sonic/release` | Release G1 and SMPL encoder variants. |
| SONIC low-latency | `checkpoints/sonic/low_latency` | Low-latency G1 and SMPL variants. |
| SONIC v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | G1 policy with heading-normalized reference orientation. |
| GRIT v0.0.1 | `checkpoints/grit/v0_0_1/policy.yaml` | Nine-frame reference context and ten-frame proprioceptive history. |
| ScaleBFM | `checkpoints/scalebfm` | ScaleBFM Humanoid Transformer XL and M ONNX exports from [WeishuaiZeng/ScaleBFM](https://huggingface.co/WeishuaiZeng/ScaleBFM). |
| Proteus | `checkpoints/proteus/proteus_wamseq_62/unitree_g1/policy.yaml` | The 62-robot Proteus WAMSeq actor (hardware-tested on G1; the wsq61 actor is an alternative) from [ScaleBFM](https://github.com/zengweishuai/ScaleBFM), wrapped for the G1 by `scripts/export_proteus_onnx.py`; see [Run External Policies](./docs/tutorials/run-external-policies.md#proteus). |
| Proteus ET1 | `checkpoints/proteus/proteus_60/galbot_et1/policy.yaml` | Proteus on the Galbot ET1 (`--robot et1`), run on the ET1 board; see [Proteus on the Galbot ET1 board](./docs/reference/proteus-et1-onboard.md). |
| Proteus WAMSeq AgiBot Expedition A3 | `checkpoints/proteus/proteus_wamseq_62/agibot_a3/policy.yaml` | 30 joints, multiple LAFAN motions and PICO; [guide](./docs/tutorials/proteus-a3.md). |
| Proteus WAMSeq EngineAI T800 | `checkpoints/proteus/wsq61/engineai_t800/policy.yaml` | 25 joints, multiple LAFAN motions and PICO; [guide](./docs/tutorials/proteus-t800.md). |
| Proteus WAMSeq EngineAI PM01 | `checkpoints/proteus/wsq61/engineai_pm01/policy.yaml` | 24-joint PM01/pm_v2, legacy ROS interface, LAFAN motions and PICO; [guide](./docs/tutorials/proteus-pm01.md). |
| Proteus WAMSeq Primebot Q1 | `checkpoints/proteus/wsq61/primebot_q1/policy.yaml` | 22 joints, multiple LAFAN motions and PICO; [guide](./docs/tutorials/proteus-q1.md). |
| BFM-Zero | `checkpoints/bfm-zero/exp_lafan40-100style_update_z10/policy.yaml` | Latent-conditioned motion tracker. |
| TeleopIT | `checkpoints/teleopit/policy.yaml` | TeleopIT policy wrapper. |
| Humanoid-GPT | `checkpoints/humanoid-gpt/policy.yaml` | Humanoid-GPT policy wrapper. |
| TWIST2 | `checkpoints/twist2/policy.yaml` | TWIST2 policy wrapper. |

## Deploying On Robots

Every robot has its own folder under [`scripts/`](./scripts/README.md) with a README on what to adjust and how
to run it. [`scripts/README.md`](./scripts/README.md) describes the deployment workflow shared by all of them
(export, deployment overlay, sim2sim rehearsal, hardware preflight, the Shaping → Hold → Go → Final Hold stages
and E-stop) and the checklist for bringing up a new robot.

| Robot | Folder | Hardware status |
| --- | --- | --- |
| Unitree G1 | [`scripts/g1`](./scripts/g1/README.md) | Tested on PC; onboard Orin path not yet hardware-tested |
| Galbot ET1 | [`scripts/et1`](./scripts/et1/README.md) | Tested onboard |
| EngineAI PM01 | [`scripts/pm01`](./scripts/pm01/README.md) | Gantry-protected walks (PC and onboard) |
| Primebot Q1 | [`scripts/q1`](./scripts/q1/README.md) | Feedback only; policy takeover unvalidated |
| AgiBot A3 | [`scripts/a3`](./scripts/a3/README.md) | Simulation only |
| EngineAI T800 | [`scripts/t800`](./scripts/t800/README.md) | Simulation only |
| AgiBot Lingxi X2 | [`scripts/x2`](./scripts/x2/README.md) | Onboard gantry trials; not yet stable |
| LimX OLI | [`scripts/oli`](./scripts/oli/README.md) | Simulation only |
| DH116S hand | [`scripts/dh116s`](./scripts/dh116s/README.md) | PICO grip control |

## Real-robot Environments

Robot SDKs are kept out of the generic root environment. G1 inline deployment
uses `uv sync --extra inference-cpu --extra robot-g1`. See
[Robot I/O Modes](./docs/robot_io.md) for setup and deployment commands.

## Next Steps

- [Docs Home](./docs/README.md)
- [Getting Started](./docs/getting-started/README.md)
- [Offline Motion Tracking Tutorial](./docs/tutorials/offline-motion-tracking.md)
- [Pico Teleoperation Tutorial](./docs/tutorials/pico-teleoperation.md)
- [Scalecar Vehicle Entry Tutorial](./docs/tutorials/scalecar-vehicle-entry.md)
- [Proteus on the Galbot ET1 board](./docs/reference/proteus-et1-onboard.md)

## Citation

If you find sim2real useful in your research, please cite:

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
