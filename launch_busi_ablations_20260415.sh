#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"
SPLIT_DIR="${ROOT}/data/processed/busi/splits_udpc_mt_ratio20"
BASE_SPLITS="${ROOT}/outputs/runs/busi_udpc_mt_20260324_005245/splits.json"

SESSION_NO_UQ="busi_ablate20_no_uq_20260415"
SESSION_NO_PROTO="busi_ablate20_no_proto_20260415"
SESSION_MT_RECIPE="busi_ablate20_mt_same_recipe_20260415"

CONFIG_NO_UQ="${ROOT}/busi_ablate20_no_uq_20260415.yaml"
CONFIG_NO_PROTO="${ROOT}/busi_ablate20_no_proto_20260415.yaml"
CONFIG_MT_RECIPE="${ROOT}/busi_ablate20_mt_same_recipe_20260415.yaml"

LOG_NO_UQ="${ROOT}/outputs/runs/${SESSION_NO_UQ}.log"
LOG_NO_PROTO="${ROOT}/outputs/runs/${SESSION_NO_PROTO}.log"
LOG_MT_RECIPE="${ROOT}/outputs/runs/${SESSION_MT_RECIPE}.log"

cd "${ROOT}"

source .venv/bin/activate

if [ ! -f "${SPLIT_DIR}/summary.json" ]; then
  PYTHONUNBUFFERED=1 python derive_busi_ratio_split_from_run.py \
    --dataset-root data/processed/busi \
    --base-splits "${BASE_SPLITS}" \
    --ratio 0.2 \
    --seed 42 \
    --output-dir "${SPLIT_DIR}"
fi

for session in "${SESSION_NO_UQ}" "${SESSION_NO_PROTO}" "${SESSION_MT_RECIPE}"; do
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "screen session already exists: ${session}"
    exit 1
  fi
done

screen -dmS "${SESSION_NO_UQ}" bash -lc "
  cd '${ROOT}'
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python train.py --config '${CONFIG_NO_UQ}' 2>&1 | tee '${LOG_NO_UQ}'
"

screen -dmS "${SESSION_NO_PROTO}" bash -lc "
  cd '${ROOT}'
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python train.py --config '${CONFIG_NO_PROTO}' 2>&1 | tee '${LOG_NO_PROTO}'
"

screen -dmS "${SESSION_MT_RECIPE}" bash -lc "
  cd '${ROOT}'
  source .venv/bin/activate
  PYTHONUNBUFFERED=1 python train.py --config '${CONFIG_MT_RECIPE}' 2>&1 | tee '${LOG_MT_RECIPE}'
"

echo "launched ${SESSION_NO_UQ}"
echo "launched ${SESSION_NO_PROTO}"
echo "launched ${SESSION_MT_RECIPE}"
