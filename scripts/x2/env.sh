# Sourced by the X2 onboard scripts. Layout under PROTEUS_ROOT (from scripts/x2/deploy_onboard.sh):
# repo/ site-packages/ hf/ logs/. Firmware v0.8.2 ROS 2 / FastDDS environment plus aimdk_msgs from an app.
root="${PROTEUS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
repo="$root/repo"
export AGIBOT_HOME=""
set +u
source /agibot/software/entry/cfg/basic_env.sh
aimdk="${X2_AIMDK_PKG:-/agibot/software/hal_ethercat}"
source "$aimdk/share/aimdk_msgs/local_setup.bash"
set -u
export PYTHONPATH="$root/site-packages:$repo:$aimdk/local/lib/python3.10/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$aimdk/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" HF_HOME="$root/hf"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONDONTWRITEBYTECODE=1 ORT_LOGGING_LEVEL=3
export SIM2REAL_ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
