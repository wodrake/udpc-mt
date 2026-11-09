#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"

launch_one() {
  local session="$1"
  local config_path="$2"
  local log_path="$3"

  if ssh server "screen -ls | grep -q '[.]${session}[[:space:]]'"; then
    echo "skip existing session: ${session}"
    return 0
  fi

  ssh server "screen -dmS '${session}' bash -lc \"
    cd '${ROOT}'
    source .venv/bin/activate
    PYTHONUNBUFFERED=1 python train.py --config '${config_path}' 2>&1 | tee '${log_path}'
  \""
  echo "launched ${session}"
}

launch_one "laa2_udpcmt_0415a" "${ROOT}/laa2_udpcmt_ratio20_search_uq032_proto011_r10_mp48_ema995_mc8_0415a.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_search_uq032_proto011_r10_mp48_ema995_mc8_0415a.log"
launch_one "laa2_udpcmt_0415e" "${ROOT}/laa2_udpcmt_ratio10_search_uq028_proto010_r14_mp24_ema995_mc8_0415e.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio10_search_uq028_proto010_r14_mp24_ema995_mc8_0415e.log"

launch_one "laa2_udpcmt_0415b" "${ROOT}/laa2_udpcmt_ratio20_search_uq035_proto013_r10_mp48_ema995_mc6_0415b.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_search_uq035_proto013_r10_mp48_ema995_mc6_0415b.log"
launch_one "laa2_udpcmt_0415f" "${ROOT}/laa2_udpcmt_ratio10_search_uq030_proto011_r12_mp24_ema995_mc8_0415f.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio10_search_uq030_proto011_r12_mp24_ema995_mc8_0415f.log"

launch_one "laa2_udpcmt_0415c" "${ROOT}/laa2_udpcmt_ratio20_search_uq031_proto012_r8_mp40_ema99_mc6_0415c.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_search_uq031_proto012_r8_mp40_ema99_mc6_0415c.log"
launch_one "laa2_udpcmt_0415g" "${ROOT}/laa2_udpcmt_ratio10_search_uq033_proto013_r10_mp24_ema995_mc6_0415g.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio10_search_uq033_proto013_r10_mp24_ema995_mc6_0415g.log"

launch_one "laa2_udpcmt_0415d" "${ROOT}/laa2_udpcmt_ratio20_search_uq033_proto010_r14_mp64_cw09_ema995_mc8_0415d.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio20_search_uq033_proto010_r14_mp64_cw09_ema995_mc8_0415d.log"
launch_one "laa2_udpcmt_0415h" "${ROOT}/laa2_udpcmt_ratio10_search_uq030_proto010_r10_mp16_ema99_mc6_0415h.yaml" "${ROOT}/outputs/runs/laa2_udpcmt_ratio10_search_uq030_proto010_r10_mp16_ema99_mc6_0415h.log"
