#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"

launch_one() {
  local session="$1"
  local config_path="$2"
  local run_dir="$3"
  local log_path="$4"

  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "screen session already exists: ${session}"
    return 1
  fi

  screen -dmS "${session}" bash -lc "
    cd '${ROOT}'
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python train.py --config '${config_path}' --run-dir '${run_dir}' 2>&1 | tee '${log_path}'
  "
  echo "launched ${session}"
}

cd "${ROOT}"

launch_one "laa2_baseline10_0413" "${ROOT}/laa2_baseline_ratio10_0413.yaml" "${ROOT}/outputs/runs/laa2_baseline_ratio10_0413" "${ROOT}/outputs/runs/laa2_baseline_ratio10_0413.log"
launch_one "laa2_baseline20_0413" "${ROOT}/laa2_baseline_ratio20_0413.yaml" "${ROOT}/outputs/runs/laa2_baseline_ratio20_0413" "${ROOT}/outputs/runs/laa2_baseline_ratio20_0413.log"
launch_one "laa2_mt10_0413" "${ROOT}/laa2_mt_ratio10_0413.yaml" "${ROOT}/outputs/runs/laa2_mt_ratio10_0413" "${ROOT}/outputs/runs/laa2_mt_ratio10_0413.log"
launch_one "laa2_mt20_0413" "${ROOT}/laa2_mt_ratio20_0413.yaml" "${ROOT}/outputs/runs/laa2_mt_ratio20_0413" "${ROOT}/outputs/runs/laa2_mt_ratio20_0413.log"
