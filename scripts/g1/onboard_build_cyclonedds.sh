#!/usr/bin/env bash
# Run on the Orin by deploy_onboard.sh: build CycloneDDS 0.10.2 into <root>/cyclonedds and its Python binding into
# <root>/cyclonedds-py (kept out of site-packages, which deploy_onboard.sh syncs with --delete). Offline; skips work
# that is already done. Security, SSL and shared memory are off: the G1's DDS uses none of them.
set -euo pipefail
root="$1"
src="$root/src"
if [[ ! -f "$root/cyclonedds/lib/libddsc.so.0.10.2" ]]; then
  rm -rf "$src/build-c" "$src/cyclonedds-0.10.2"
  tar xzf "$src/cyclonedds-c-0.10.2.tar.gz" -C "$src"
  cmake -S "$src/cyclonedds-0.10.2" -B "$src/build-c" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$root/cyclonedds" \
    -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DENABLE_SSL=OFF -DENABLE_SECURITY=OFF -DENABLE_SHM=OFF > "$src/cmake.log"
  nice -n 10 make -C "$src/build-c" -j4 > "$src/make.log"
  make -C "$src/build-c" install > "$src/install.log"
fi
if [[ ! -d "$root/cyclonedds-py/cyclonedds-0.10.2.dist-info" ]]; then
  rm -rf "$src/cyclonedds-py" "$root/cyclonedds-py"
  mkdir -p "$src/cyclonedds-py"
  tar xzf "$src/cyclonedds-python-0.10.2.tar.gz" -C "$src/cyclonedds-py"
  # setuptools comes from the synced site-packages; nothing is downloaded.
  CYCLONEDDS_HOME="$root/cyclonedds" PYTHONPATH="$root/site-packages" nice -n 10 "$root/python/bin/python3" -m pip install \
    --no-index --no-build-isolation --no-deps --target "$root/cyclonedds-py" "$src/cyclonedds-py/cyclonedds-0.10.2" \
    > "$src/pip-cyclonedds.log"
fi
echo "CycloneDDS 0.10.2 ready in $root/cyclonedds and $root/cyclonedds-py"
