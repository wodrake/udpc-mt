#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"
BASE_CONFIG="${ROOT}/laa2_udpcmt_ratio20_repr_do015_t012_0418h.yaml"

declare -A COMMANDS

COMMANDS[laa2_cw04_0420a]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_cw04_0420a' --override 'device=cuda:0' --override 'loss.consistency_weight=0.4' --override \"wandb.tags=['laa2','ratio20','coeff0420','consistency_weight','cw04']\""
COMMANDS[laa2_cw07_0420b]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_cw07_0420b' --override 'device=cuda:1' --override 'loss.consistency_weight=0.7' --override \"wandb.tags=['laa2','ratio20','coeff0420','consistency_weight','cw07']\""
COMMANDS[laa2_cw13_0420c]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_cw13_0420c' --override 'device=cuda:2' --override 'loss.consistency_weight=1.3' --override \"wandb.tags=['laa2','ratio20','coeff0420','consistency_weight','cw13']\""
COMMANDS[laa2_cw16_0420d]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_cw16_0420d' --override 'device=cuda:3' --override 'loss.consistency_weight=1.6' --override \"wandb.tags=['laa2','ratio20','coeff0420','consistency_weight','cw16']\""
COMMANDS[laa2_pw04_0420e]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_pw04_0420e' --override 'device=cuda:0' --override 'loss.prototype_weight=0.04' --override \"wandb.tags=['laa2','ratio20','coeff0420','prototype_weight','pw04']\""
COMMANDS[laa2_pw08_0420f]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_pw08_0420f' --override 'device=cuda:1' --override 'loss.prototype_weight=0.08' --override \"wandb.tags=['laa2','ratio20','coeff0420','prototype_weight','pw08']\""
COMMANDS[laa2_pw16_0420g]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_pw16_0420g' --override 'device=cuda:2' --override 'loss.prototype_weight=0.16' --override \"wandb.tags=['laa2','ratio20','coeff0420','prototype_weight','pw16']\""
COMMANDS[laa2_pw20_0420h]="PYTHONUNBUFFERED=1 python train.py --config '${BASE_CONFIG}' --override 'project_name=laa2_coeff20_pw20_0420h' --override 'device=cuda:3' --override 'loss.prototype_weight=0.20' --override \"wandb.tags=['laa2','ratio20','coeff0420','prototype_weight','pw20']\""

cd "${ROOT}"

for session in "${!COMMANDS[@]}"; do
  if screen -ls | grep -q "[.]${session}[[:space:]]"; then
    echo "screen session already exists: ${session}"
    exit 1
  fi
done

launch_one() {
  local session="$1"
  local cmd="$2"
  local log_path="${ROOT}/outputs/runs/${session}.log"

  screen -dmS "${session}" bash -lc "
    cd '${ROOT}'
    source .venv/bin/activate
    ${cmd} 2>&1 | tee '${log_path}'
  "
  echo "launched ${session}"
}

launch_one "laa2_cw04_0420a" "${COMMANDS[laa2_cw04_0420a]}"
launch_one "laa2_cw07_0420b" "${COMMANDS[laa2_cw07_0420b]}"
launch_one "laa2_cw13_0420c" "${COMMANDS[laa2_cw13_0420c]}"
launch_one "laa2_cw16_0420d" "${COMMANDS[laa2_cw16_0420d]}"
launch_one "laa2_pw04_0420e" "${COMMANDS[laa2_pw04_0420e]}"
launch_one "laa2_pw08_0420f" "${COMMANDS[laa2_pw08_0420f]}"
launch_one "laa2_pw16_0420g" "${COMMANDS[laa2_pw16_0420g]}"
launch_one "laa2_pw20_0420h" "${COMMANDS[laa2_pw20_0420h]}"
