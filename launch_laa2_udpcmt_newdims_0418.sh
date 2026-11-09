#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"

declare -A CONFIGS=(
  [laa2_udpcmt_0418a]="${ROOT}/laa2_udpcmt_ratio10_lr8e4_wd5e5_0418a.yaml"
  [laa2_udpcmt_0418b]="${ROOT}/laa2_udpcmt_ratio10_do015_t008_0418b.yaml"
  [laa2_udpcmt_0418c]="${ROOT}/laa2_udpcmt_ratio20_lr8e4_wd5e5_0418c.yaml"
  [laa2_udpcmt_0418d]="${ROOT}/laa2_udpcmt_ratio20_do015_t008_0418d.yaml"
)

cd "${ROOT}"

for session in "${!CONFIGS[@]}"; do
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "screen session already exists: ${session}"
    exit 1
  fi
done

launch_one() {
  local session="$1"
  local config_path="$2"
  local log_path="${ROOT}/outputs/runs/${session}.log"

  screen -dmS "${session}" bash -lc "
    cd '${ROOT}'
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python train.py --config '${config_path}' 2>&1 | tee '${log_path}'
  "
  echo "launched ${session}"
}

launch_one "laa2_udpcmt_0418a" "${CONFIGS[laa2_udpcmt_0418a]}"
launch_one "laa2_udpcmt_0418b" "${CONFIGS[laa2_udpcmt_0418b]}"
launch_one "laa2_udpcmt_0418c" "${CONFIGS[laa2_udpcmt_0418c]}"
launch_one "laa2_udpcmt_0418d" "${CONFIGS[laa2_udpcmt_0418d]}"
