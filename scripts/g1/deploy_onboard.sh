#!/usr/bin/env bash
# Push the Proteus G1 onboard stack from this PC to the robot's Orin and build CycloneDDS there.
# Touches only $G1_TARGET on the Orin. Never starts the controller.
#   G1_HOST          ssh destination (default g1; ~/.ssh/config supplies 192.168.123.164 / unitree)
#   G1_TARGET        Orin directory (default /home/unitree/proteus_g1_onboard)
#   G1_POLICY_DIRS   policy directories to push (default: wsq62 and ht500)
#   G1_RSYNC_BWLIMIT rsync KiB/s (default 20000), so a running PC session keeps the robot LAN
#   UV               uv binary that resolves the aarch64 / CPython 3.10 packages
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
host="${G1_HOST:-g1}"
target="${G1_TARGET:-/home/unitree/proteus_g1_onboard}"
cache="${G1_ENV_CACHE:-$repo/outputs/g1-onboard-env}"
uv="${UV:-$cache/dl/uv/uv-x86_64-unknown-linux-gnu/uv}"
policies="${G1_POLICY_DIRS:-checkpoints/proteus/proteus_wamseq_62/unitree_g1 checkpoints/proteus/ht500_k10_anchor01_lr001_t250/unitree_g1}"
rsync=(rsync -a --bwlimit="${G1_RSYNC_BWLIMIT:-20000}")
cd "$repo"
[[ -x "$uv" ]] || { echo "Set UV to a uv binary (https://github.com/astral-sh/uv/releases)." >&2; exit 1; }

# 1. CPython 3.10 and offline packages for the Orin (resolved once, reused while uv.lock is unchanged). The Orin runs
# JetPack 5 (Ubuntu 20.04, glibc 2.31, system Python 3.8), so wheels must be manylinux_2_31 or older. cyclonedds 0.10.2,
# unitree_sdk2py's pin, has no aarch64 wheel and is built on the Orin in step 3; unitree-interface (inline I/O) and
# opencv-python are not used by this deployment.
UV_PYTHON_INSTALL_DIR="$cache/python" UV_PYTHON_BIN_DIR="$cache/python-bin" \
  "$uv" python install cpython-3.10.22-linux-aarch64-gnu
lock_hash="$(sha256sum uv.lock | cut -c1-16)"
if [[ ! -f "$cache/site/.lock-$lock_hash" ]]; then
  rm -rf "$cache/site" && mkdir -p "$cache"
  "$uv" export --frozen --no-hashes --extra inference-cpu --extra robot-g1 --no-emit-project --no-dev \
    | grep -vE '^(cyclonedds|unitree-interface|opencv-python)==' > "$cache/requirements.txt"
  "$uv" pip install --target "$cache/site" --python-platform aarch64-manylinux_2_31 --python-version 3.10 \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --no-deps \
    -r "$cache/requirements.txt"
  touch "$cache/site/.lock-$lock_hash"
fi
mkdir -p "$cache/dl/cyclonedds"
python3 - "$cache/dl/cyclonedds" <<'EOF'
import hashlib, pathlib, sys, urllib.request
sources = {
    "cyclonedds-c-0.10.2.tar.gz": ("https://github.com/eclipse-cyclonedds/cyclonedds/archive/refs/tags/0.10.2.tar.gz",
                                   "bc84e137e0c8a055b8cd97fbeafec94e36de1b0c2e88800896a82384fd867ae5"),
    "cyclonedds-python-0.10.2.tar.gz": ("https://files.pythonhosted.org/packages/62/88/4b440f5916976234838989c3214222a5"
                                        "abc6fd48009a992c4dc22a86c04f/cyclonedds-0.10.2.tar.gz",
                                        "f834962eabbdcdf4e9cd75cf87222f3c5ef22d9cb9e7ed651d9a8710fe984a30"),
}
for name, (url, sha) in sources.items():
    path = pathlib.Path(sys.argv[1]) / name
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        path.unlink()
        raise SystemExit(f"{name}: sha256 mismatch")
EOF

# 2. Code, the policies and motions, and the HF assets (G1 MJCF; h2_model is resolved at module load).
files="$cache/files.txt"
{
  git ls-files sim2real scripts/g1 scripts/test_policy_inference.py pyproject.toml \
    .agents/skills/configure-g1-sim2real/scripts/verify_install.py
  git ls-files --others --exclude-standard sim2real scripts/g1
  for policy in $policies; do find "$policy" -maxdepth 1 -type f \( -name 'policy*' -o -name '*.json' \); done
  find checkpoints/proteus/proteus_60/unitree_g1/motions -type f
} | sort -u > "$files"
ssh "$host" "mkdir -p '$target/repo' '$target/hf/hub' '$target/src'"
"${rsync[@]}" "$cache/python/cpython-3.10.22-linux-aarch64-gnu/" "$host:$target/python/"
"${rsync[@]}" --delete "$cache/site/" "$host:$target/site-packages/"
"${rsync[@]}" "$cache/dl/cyclonedds/" "$host:$target/src/"
"${rsync[@]}" --files-from="$files" ./ "$host:$target/repo/"
"${rsync[@]}" ~/.cache/huggingface/hub/models--elijahgalahad--g1_xmls ~/.cache/huggingface/hub/models--elijahgalahad--h2_model \
  "$host:$target/hf/hub/"

# 3. CycloneDDS 0.10.2 (C library, then the Python binding against it) on the Orin, once.
ssh "$host" "bash '$target/repo/scripts/g1/onboard_build_cyclonedds.sh' '$target'"
echo "Deployed to $host:$target"
