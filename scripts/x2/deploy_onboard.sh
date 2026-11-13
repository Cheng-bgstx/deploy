#!/usr/bin/env bash
# Push the Proteus X2 stack from this PC to the X2 board. Touches only $X2_TARGET on the robot.
# Never stops mc and never starts the controller.
#   X2_HOST       ssh destination (default agi@10.0.1.40, the motion-control board; the PC needs 10.0.1.2/24 on VLAN 200)
#   X2_TARGET     robot directory (default proteus_x2, relative to the remote $HOME, which survives firmware updates)
#   X2_ENV_CACHE  offline aarch64 / CPython 3.10 site-packages resolved from uv.lock (see below)
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
host="${X2_HOST:-agi@10.0.1.40}"
target="${X2_TARGET:-proteus_x2}"
site="${X2_ENV_CACHE:-$repo/outputs/x2-onboard-env/site}"
cd "$repo"

# 1. Offline Python packages, resolved for the board from this uv.lock. Build once with:
#   uv export --frozen --no-hashes --extra inference-cpu --no-emit-project --no-dev > req.txt
#   uv pip install --target "$site" --python-platform aarch64-manylinux_2_35 --python-version 3.10 \
#     --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --no-deps -r req.txt
#   touch "$site/.lock-$(sha256sum uv.lock | cut -c1-16)"
lock_hash="$(sha256sum uv.lock | cut -c1-16)"
[[ -f "$site/.lock-$lock_hash" ]] || { echo "No aarch64 site-packages for uv.lock $lock_hash at $site" >&2; exit 1; }

# 2. Code, the X2 actors (with their policy-pc.yaml overlays), the prepared model, the reference motions,
#    and the HF assets that sim2real.config.robots resolves at import time.
releases="${X2_RELEASES:-proteus_wamseq_62 proteus_60_bigdata proteus_60}"
for r in $releases; do
  for f in policy.onnx policy-pc.yaml policy-pc-damped.yaml; do
    [[ -e "checkpoints/proteus/$r/lingxi_x2/$f" ]] || { echo "Missing checkpoints/proteus/$r/lingxi_x2/$f" >&2; exit 1; }
  done
done
for f in third_party/lingxi_x2/model/scene_tracking.xml outputs/x2/pico_hold/manifest.json \
    outputs/x2/wave/21_wave_right/manifest.json; do
  [[ -e "$f" ]] || { echo "Missing $f (see docs/tutorials/proteus-x2-onboard.md)" >&2; exit 1; }
done
files="$(mktemp)"
trap 'rm -f "$files"' EXIT
{
  git ls-files sim2real scripts/x2 scripts/et1/operator_keyboard.py pyproject.toml
  git ls-files --others --exclude-standard sim2real scripts/x2
  for r in $releases; do find "checkpoints/proteus/$r/lingxi_x2" -type f; done
  find third_party/lingxi_x2 outputs/x2/pico_hold -type f
  for d in outputs/x2/lafan outputs/x2/wave; do
    [[ ! -d $d ]] || find $d -type f \( -name manifest.json -o -path '*/motions/*' \)
  done
} | sort -u > "$files"
ssh "$host" "mkdir -p '$target/repo' '$target/hf/hub' '$target/logs'"
rsync -a --delete "$site/" "$host:$target/site-packages/"
rsync -a --files-from="$files" ./ "$host:$target/repo/"
rsync -a ~/.cache/huggingface/hub/models--elijahgalahad--h2_model ~/.cache/huggingface/hub/models--elijahgalahad--g1_xmls \
  "$host:$target/hf/hub/"
ssh "$host" "chmod +x '$target/repo/scripts/x2/'*.sh; echo 'Deployed to $target on' \$(hostname) \$(uname -m)"
