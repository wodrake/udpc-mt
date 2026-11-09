#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"
SESSION="laa2_udpcmt_0413c"
CONFIG_PATH="${ROOT}/laa2_udpcmt_ratio10_ema995_uq03_mc8_proto012_0413c.yaml"
LOG_PATH="${ROOT}/outputs/runs/laa2_udpcmt_ratio10_ema995_uq03_mc8_proto012_0413c.log"

cd "${ROOT}"

if screen -ls | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: ${SESSION}"
  exit 1
fi

screen -dmS "${SESSION}" bash -lc "
  cd '${ROOT}'
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python train.py --config '${CONFIG_PATH}' 2>&1 | tee '${LOG_PATH}'
"

echo "launched ${SESSION}"
