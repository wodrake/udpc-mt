#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"

launch_one() {
  local session="$1"
  local config_path="$2"
  local log_path="$3"

  if ssh server "screen -ls | grep -q '[.]${session}[[:space:]]'"; then
    echo "screen session already exists: ${session}"
    return 1
  fi

  ssh server "screen -dmS '${session}' bash -lc \"
    cd '${ROOT}'
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python train.py --config '${config_path}' 2>&1 | tee '${log_path}'
  \""
  echo "launched ${session}"
}

launch_one \
  "laa2_udpcmt_pf0414a" \
  "${ROOT}/laa2_udpcmt_ratio20_precfocus_uq035_proto010_r16_mp64_ema995_mc8_0414a.yaml" \
  "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_precfocus_uq035_proto010_r16_mp64_ema995_mc8_0414a.log"

launch_one \
  "laa2_udpcmt_pf0414b" \
  "${ROOT}/laa2_udpcmt_ratio20_precfocus_uq040_proto010_r16_mp64_ema995_mc8_0414b.yaml" \
  "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_precfocus_uq040_proto010_r16_mp64_ema995_mc8_0414b.log"

launch_one \
  "laa2_udpcmt_pf0414c" \
  "${ROOT}/laa2_udpcmt_ratio20_precfocus_uq035_proto008_r20_mp96_cw08_ema995_mc8_0414c.yaml" \
  "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_precfocus_uq035_proto008_r20_mp96_cw08_ema995_mc8_0414c.log"

launch_one \
  "laa2_udpcmt_pf0414d" \
  "${ROOT}/laa2_udpcmt_ratio20_precfocus_uq035_proto010_r16_mp64_ema99_mc4_0414d.yaml" \
  "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_precfocus_uq035_proto010_r16_mp64_ema99_mc4_0414d.log"
