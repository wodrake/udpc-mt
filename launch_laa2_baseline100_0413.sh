#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"
SESSION="laa2_baseline100_0413"
CONFIG_PATH="${ROOT}/laa2_baseline_ratio100_0413.yaml"
RUN_DIR="${ROOT}/outputs/runs/laa2_baseline_ratio100_0413"
LOG_PATH="${ROOT}/outputs/runs/laa2_baseline_ratio100_0413.log"

cd "${ROOT}"

if screen -ls | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: ${SESSION}"
  exit 1
fi

if [ -e "${RUN_DIR}" ]; then
  echo "run directory already exists: ${RUN_DIR}"
  exit 1
fi

screen -dmS "${SESSION}" bash -lc "
  cd '${ROOT}'
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python train.py --config '${CONFIG_PATH}' --run-dir '${RUN_DIR}' 2>&1 | tee '${LOG_PATH}'
"

echo "launched ${SESSION}"
