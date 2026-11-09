#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"
SESSION="busi_udpcmt10_0413"
SPLIT_DIR="${ROOT}/data/processed/busi/splits_udpc_mt_ratio10"
BASE_SPLITS="${ROOT}/outputs/runs/busi_udpc_mt_20260324_005245/splits.json"
CONFIG_PATH="${ROOT}/busi_r10_ema995_uq015_proto008_min16_gpu1_20260413.yaml"
LOG_PATH="${ROOT}/outputs/runs/busi_r10_ema995_uq015_proto008_min16_gpu1_20260413.log"

cd "${ROOT}"

if screen -ls | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session already exists: ${SESSION}"
  exit 1
fi

screen -dmS "${SESSION}" bash -lc "
  cd '${ROOT}'
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python derive_busi_ratio_split_from_run.py \
    --dataset-root data/processed/busi \
    --base-splits '${BASE_SPLITS}' \
    --ratio 0.1 \
    --seed 42 \
    --output-dir '${SPLIT_DIR}'
  PYTHONUNBUFFERED=1 python train.py --config '${CONFIG_PATH}' 2>&1 | tee '${LOG_PATH}'
"

echo "launched ${SESSION}"
