#!/usr/bin/env bash
set -euo pipefail

PCCS_ROOT="/data1/houshengdi/PCCS-master/PCCS-master"
UDPC_ROOT="/home/houshengdi/houshengdi_data1/udpc-mt"
CONDA_ACTIVATE="/data1/houshengdi/anaconda3/bin/activate"
ASSET_ROOT="${UDPC_ROOT}/outputs/qualitative/busi_figure7_4methods"
MINI_ROOT="${ASSET_ROOT}/mini_busi_root"

CASES=(
  "benign_breast_046"
  "benign_breast_164"
  "malignant_breast_481"
)

UNET_CONFIG="${PCCS_ROOT}/runs/BUSI_udpc_split_full/unet20_0410/resolved_config.yaml"
UNET_CKPT="${PCCS_ROOT}/runs/BUSI_udpc_split_full/unet20_0410/unet_best_model1.pth"
MT_CONFIG="${PCCS_ROOT}/runs/BUSI_udpc_split_full/mt20_0410/resolved_config.yaml"
MT_CKPT="${PCCS_ROOT}/runs/BUSI_udpc_split_full/mt20_0410/unet_mt_best_model1.pth"
PCCS_CONFIG="${PCCS_ROOT}/runs/BUSI_udpc_split_full/pccs20_0410/resolved_config.yaml"
PCCS_CKPT="${PCCS_ROOT}/runs/BUSI_udpc_split_full/pccs20_0410/unet_pccs_best_model1.pth"
UDPC_CKPT="${UDPC_ROOT}/outputs/runs/busi_r20_ema995_uq015_proto008_min16_gpu3_20260410/checkpoints/best.pt"

mkdir -p \
  "${MINI_ROOT}/test/image" \
  "${MINI_ROOT}/test/mask" \
  "${ASSET_ROOT}/images" \
  "${ASSET_ROOT}/masks" \
  "${ASSET_ROOT}/preds/unet" \
  "${ASSET_ROOT}/preds/mt" \
  "${ASSET_ROOT}/preds/pccs" \
  "${ASSET_ROOT}/preds/udpc_mt"

for case_id in "${CASES[@]}"; do
  src_img="${UDPC_ROOT}/data/processed/busi/images/${case_id}.png"
  src_mask="${UDPC_ROOT}/data/processed/busi/masks/${case_id}.png"
  cp -f "${src_img}" "${ASSET_ROOT}/images/${case_id}.png"
  cp -f "${src_mask}" "${ASSET_ROOT}/masks/${case_id}.png"
  ln -sfn "${src_img}" "${MINI_ROOT}/test/image/${case_id}.png"
  ln -sfn "${src_mask}" "${MINI_ROOT}/test/mask/${case_id}.png"
done

run_pccs_eval() {
  local name="$1"
  local config_path="$2"
  local ckpt_path="$3"
  local gpu_id="$4"
  local save_dir="${ASSET_ROOT}/eval_${name}"

  rm -rf "${save_dir}"
  mkdir -p "${save_dir}"

  source "${CONDA_ACTIVATE}" pccs
  cd "${PCCS_ROOT}"
  PYTHONUNBUFFERED=1 python eval_busi.py \
    --config "${config_path}" \
    --checkpoint "${ckpt_path}" \
    --root-dir "${MINI_ROOT}" \
    --split test \
    --gpu "${gpu_id}" \
    --save-dir "${save_dir}" \
    --save-preds

  cp -f "${save_dir}/predictions/"*_pred.png "${ASSET_ROOT}/preds/${name}/"
}

run_pccs_eval "unet" "${UNET_CONFIG}" "${UNET_CKPT}" "0"
run_pccs_eval "mt" "${MT_CONFIG}" "${MT_CKPT}" "1"
run_pccs_eval "pccs" "${PCCS_CONFIG}" "${PCCS_CKPT}" "2"

cd "${UDPC_ROOT}"
source .venv/bin/activate
PYTHONUNBUFFERED=1 python infer.py \
  --checkpoint "${UDPC_CKPT}" \
  --input "${ASSET_ROOT}/images" \
  --gt-dir "${ASSET_ROOT}/masks" \
  --output-dir "${ASSET_ROOT}/preds/udpc_mt" \
  --threshold 0.5 \
  --override "device=cuda:3"

cat > "${ASSET_ROOT}/manifest.json" <<EOF
{
  "cases": [
    "benign_breast_046",
    "benign_breast_164",
    "malignant_breast_481"
  ],
  "methods": [
    "unet",
    "mt",
    "pccs",
    "udpc_mt"
  ]
}
EOF

echo "prepared BUSI qualitative assets under ${ASSET_ROOT}"
