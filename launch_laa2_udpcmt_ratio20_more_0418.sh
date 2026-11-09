#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"

declare -A CONFIGS=(
  [laa2_udpcmt_0418e]="${ROOT}/laa2_udpcmt_ratio20_combo_lr8e4_wd5e5_do015_t008_0418e.yaml"
  [laa2_udpcmt_0418f]="${ROOT}/laa2_udpcmt_ratio20_opt_lr7e4_wd5e5_cw09_0418f.yaml"
  [laa2_udpcmt_0418g]="${ROOT}/laa2_udpcmt_ratio20_repr_do010_t007_0418g.yaml"
  [laa2_udpcmt_0418h]="${ROOT}/laa2_udpcmt_ratio20_repr_do015_t012_0418h.yaml"
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

launch_one "laa2_udpcmt_0418e" "${CONFIGS[laa2_udpcmt_0418e]}"
launch_one "laa2_udpcmt_0418f" "${CONFIGS[laa2_udpcmt_0418f]}"
launch_one "laa2_udpcmt_0418g" "${CONFIGS[laa2_udpcmt_0418g]}"
launch_one "laa2_udpcmt_0418h" "${CONFIGS[laa2_udpcmt_0418h]}"
