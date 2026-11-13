#!/usr/bin/env bash
# Push the Proteus PM01 onboard stack from this PC to the PM01 motion computer (Nezha, Ubuntu 22.04, x86_64,
# CPython 3.10, ROS Humble). Touches only $PM01_TARGET on the robot; never starts, stops or reconfigures
# robot software. Prepare the assets on the PC first (scripts/pm01/prepare.py, make_wave_motion.py).
#   PM01_HOST     ssh destination (default user@192.168.0.163; install a key with ssh-copy-id first)
#   PM01_TARGET   robot directory, relative to the remote home (default proteus_pm01)
#   PM01_RELEASE  exported Proteus release (default proteus_wamseq_62_full)
#   UV            uv binary used to resolve the x86_64 / CPython 3.10 packages
# --build-only resolves the packages on the PC and stops before contacting the robot.
# Layout on the robot: <target>/{repo, site-packages, hf, logs}.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
host="${PM01_HOST:-user@192.168.0.163}"
target="${PM01_TARGET:-proteus_pm01}"
release="${PM01_RELEASE:-proteus_wamseq_62_full}"
uv="${UV:-$(command -v uv || echo "$HOME/.local/share/proteus-tools/bin/uv")}"
cache="${PM01_ENV_CACHE:-$repo/outputs/pm01-onboard-env}"
cd "$repo"

policy_dir="checkpoints/proteus/$release/engineai_pm01"
for f in "$policy_dir/policy.yaml" "$policy_dir/policy.onnx" third_party/engineai_pm01/model/pm_v2_tracking.xml \
         checkpoints/engineai_pm01/motions/manifest.json checkpoints/engineai_pm01/wave/manifest.json; do
  [[ -e "$f" ]] || { echo "Missing $f; prepare the PM01 assets on this PC first (see docs/tutorials/proteus-pm01.md)" >&2; exit 1; }
done

# 1. Offline Python packages for the robot (resolved once, reused while uv.lock is unchanged). manylinux_2_35 is
#    Ubuntu 22.04's glibc; the CPU-only torch wheel keeps the tree small. rclpy and the messages come from ROS.
lock_hash="$(sha256sum uv.lock | cut -c1-16)"
if [[ ! -f "$cache/site/.lock-$lock_hash" ]]; then
  rm -rf "$cache" && mkdir -p "$cache"
  # uv.lock resolves the CUDA torch build for x86_64 (2.7 GB of nvidia-*, cuda-* and triton wheels); the robot has
  # no GPU, so install the same torch version's CPU build and drop the GPU-only packages.
  "$uv" export --frozen --no-hashes --extra inference-cpu --no-emit-project --no-dev \
    | grep -v -E '^(nvidia-|cuda-|triton==)' \
    | sed -E "s/^torch==([0-9.]+) ; platform_machine != 'aarch64'$/torch==\1+cpu/; /^torch==[0-9.]+\+cpu ; platform_machine == 'aarch64'$/d" \
    > "$cache/requirements.txt"
  grep -q '^torch==[0-9.]*+cpu$' "$cache/requirements.txt" || { echo 'Could not select the CPU torch build' >&2; exit 1; }
  "$uv" pip install --target "$cache/site" --python-platform x86_64-manylinux_2_35 --python-version 3.10 \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --no-deps \
    -r "$cache/requirements.txt"
  touch "$cache/site/.lock-$lock_hash"
fi
if [[ "${1:-}" == --build-only ]]; then
  echo "Packages in $cache/site ($(du -sh "$cache/site" | cut -f1)); nothing sent to the robot"
  exit 0
fi

# 2. Code, the policy, the motions, the prepared model and the Hugging Face assets imported at module load.
files="$cache/files.txt"
{
  git ls-files sim2real scripts/__init__.py scripts/pm01 pyproject.toml
  git ls-files --others --exclude-standard sim2real scripts/pm01
  find "$policy_dir" checkpoints/engineai_pm01/motions checkpoints/engineai_pm01/wave third_party/engineai_pm01/model \
    -type f -not -path '*/ros2/*'
  for f in third_party/engineai_pm01/LICENSE third_party/engineai_pm01/validation.json; do [[ ! -f "$f" ]] || echo "$f"; done
} | sort -u > "$files"
ssh "$host" "mkdir -p '$target/repo' '$target/hf/hub' '$target/logs'"
rsync -a --delete "$cache/site/" "$host:$target/site-packages/"
rsync -a --files-from="$files" ./ "$host:$target/repo/"
rsync -a ~/.cache/huggingface/hub/models--elijahgalahad--h2_model ~/.cache/huggingface/hub/models--elijahgalahad--g1_xmls \
  "$host:$target/hf/hub/"

# 3. Import check on the robot with its ROS and messages; reads nothing from the robot's topics.
ssh "$host" bash -s <<EOF
set -e
source "$target/repo/scripts/pm01/env_onboard.sh"
cd "$target/repo"
python3 - <<'PY'
import socket
import onnxruntime, rclpy, interface_protocol.msg  # noqa: F401
import sim2real.rl_policy.tracking  # noqa: F401
from sim2real.rl_policy.robot_io.pm01 import PM01RobotIO  # noqa: F401
print('imports ok on', socket.gethostname())
PY
EOF
echo "Deployed to $host:$target"
