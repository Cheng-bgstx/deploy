#!/usr/bin/env bash
# Push the Proteus ET1 onboard stack from this PC to the robot board and build the HDEP pieces there.
# Touches only $ET1_TARGET on the robot. Never starts the controller.
#   ET1_HOST     ssh destination (default 192.168.121.42; ~/.ssh/config supplies user/port)
#   ET1_TARGET   robot directory (default /root/proteus_et1_claude)
#   ET1_DEP_ROOT local et1-dep checkout (default ~/et1-dep)
#   UV           uv binary used to resolve the aarch64 / CPython 3.10 packages
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
host="${ET1_HOST:-192.168.121.42}"
target="${ET1_TARGET:-/root/proteus_et1_claude}"
dep="${ET1_DEP_ROOT:-$HOME/et1-dep}"
uv="${UV:-uv}"
cache="${ET1_ENV_CACHE:-$repo/outputs/et1-onboard-env}"
cd "$repo"

# 1. Offline Python packages for the board (resolved once, reused while uv.lock is unchanged).
lock_hash="$(sha256sum uv.lock | cut -c1-16)"
if [[ ! -f "$cache/site/.lock-$lock_hash" ]]; then
  rm -rf "$cache" && mkdir -p "$cache"
  "$uv" export --frozen --no-hashes --extra inference-cpu --no-emit-project --no-dev > "$cache/requirements.txt"
  "$uv" pip install --target "$cache/site" --python-platform aarch64-manylinux_2_35 --python-version 3.10 \
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --no-deps \
    -r "$cache/requirements.txt"
  touch "$cache/site/.lock-$lock_hash"
fi

# 2. HDEP headers and aarch64 libraries from et1-dep (dereferenced; et1-dep uses relative symlinks).
stage="$cache/probe"
rm -rf "$stage" && mkdir -p "$stage/include/hdep" "$stage/lib" "$stage/bridge"
cp "$dep/thirdparty/aarch64/hdep/include/hdep/hdep.hpp" "$stage/include/hdep/"
cp -L "$dep/thirdparty/aarch64/hdep/lib/libhdep.so.2.2" "$stage/lib/"
ln -s libhdep.so.2.2 "$stage/lib/libhdep.so"
cp "$dep"/dist/cpp/loco/aarch64/lib/libddsc.so.0 "$dep"/dist/cpp/loco/aarch64/lib/libddscxx.so.0 "$stage/lib/"
cp "$dep/dist/cpp/loco/aarch64/thirdparty/aarch64/bridges/galbot_et1/lib/libhdep_bridge_galbot_et1.so.2.2.0" "$stage/bridge/"

# 3. Code, the ET1 policy and motions, the MJCF, and the HF assets imported at module load.
files="$cache/files.txt"
{
  git ls-files sim2real scripts/et1 pyproject.toml
  git ls-files --others --exclude-standard sim2real scripts/et1
  find checkpoints/proteus/proteus_60/galbot_et1 checkpoints/proteus/proteus_wamseq_62/galbot_et1 -type f
  [[ ! -d checkpoints/galbot_track/galbot_et1 ]] || find checkpoints/galbot_track/galbot_et1 -type f
} | sort -u > "$files"
ssh "$host" "mkdir -p '$target/repo' '$target/et1-dep/storage' '$target/hf/hub' '$target/logs'"
rsync -a --delete "$cache/site/" "$host:$target/site-packages/"
rsync -a "$stage/" "$host:$target/probe/"
rsync -a --files-from="$files" ./ "$host:$target/repo/"
rsync -a "$dep/storage/humanoid_description" "$host:$target/et1-dep/storage/"
rsync -a ~/.cache/huggingface/hub/models--elijahgalahad--h2_model ~/.cache/huggingface/hub/models--elijahgalahad--g1_xmls \
  "$host:$target/hf/hub/"

# 4. Build the ctypes shim and the read-only state probe on the board.
ssh "$host" "set -e; cd '$target/probe'
g++ -std=c++17 -O2 -fPIC -shared -Wall -Iinclude '$target/repo/scripts/et1/hdep_shim.cpp' -Llib -lhdep \
  -Wl,-rpath,'\$ORIGIN/lib' -o libet1_hdep_shim.so
g++ -std=c++17 -O2 -Wall -Iinclude '$target/repo/scripts/et1/full_state_probe.cpp' -Llib -lhdep \
  -Wl,-rpath,'\$ORIGIN/lib' -o full_state_probe
echo 'Deployed to $target on' \$(hostname)"
