#!/usr/bin/env bash
set -euo pipefail

# Clean architecture screening workflow using existing repo scripts:
#   1) Build manifests with make_manifest.py:
#        - train: non-AU
#        - val:   AU (via locked-country output)
#   2) Train each architecture for 6 epochs
#   3) Evaluate each architecture on AU validation
#
# Required:
#   PATCH_TABLE=<path-to-patch-summary.{csv,json,geojson}>
#
# Optional overrides:
#   DATA_ROOT=/data/training
#   VENV_PATH=/home/gorelick/venv-cu128
#   OUTPUT_ROOT=./runs
#   RUN_NAME=arch_screen_non_au_vs_au_YYYYmmdd_HHMMSS
#   EPOCHS=6
#   BATCH_SIZE=32
#   WORKERS=3
#   USE_AMP=1

PATCH_TABLE="${PATCH_TABLE:-}"
DATA_ROOT="${DATA_ROOT:-/data/training}"
VENV_PATH="${VENV_PATH:-/home/gorelick/venv-cu128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./runs}"
RUN_NAME="${RUN_NAME:-arch_screen_non_au_vs_au_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-3}"
USE_AMP="${USE_AMP:-1}"

ARCHES=(
  "film_unet"
  "gated_unet"
  "xattn_unet"
  "hybrid_tf_unet"
  "rcan_ae_unet"
)

if [[ -z "${PATCH_TABLE}" ]]; then
  echo "ERROR: PATCH_TABLE is required." >&2
  echo "Example: PATCH_TABLE=/path/to/patch_summary.csv ./run_arch_non_au_vs_au.sh" >&2
  exit 1
fi

if [[ ! -f "${PATCH_TABLE}" ]]; then
  echo "ERROR: PATCH_TABLE file not found: ${PATCH_TABLE}" >&2
  exit 1
fi

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "ERROR: VENV_PATH does not exist: ${VENV_PATH}" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
source "${VENV_PATH}/bin/activate"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
MANIFEST_DIR="${RUN_DIR}/manifests"
CKPT_DIR="${RUN_DIR}/checkpoints"
EVAL_DIR="${RUN_DIR}/eval"
mkdir -p "${MANIFEST_DIR}" "${CKPT_DIR}" "${EVAL_DIR}"

TRAIN_MANIFEST="${MANIFEST_DIR}/train_non_au_manifest.txt"
VAL_MANIFEST="${MANIFEST_DIR}/val_au_manifest.txt"
SUMMARY_JSON="${MANIFEST_DIR}/manifest_summary_non_au_vs_au.json"

echo "==> Building manifests with make_manifest.py (locked country: AU)"
python3 make_manifest.py \
  --data-root "${DATA_ROOT}" \
  --patch-table "${PATCH_TABLE}" \
  --locked-country AU \
  --val-fraction 0 \
  --holdout-out "${VAL_MANIFEST}" \
  --train-out "${TRAIN_MANIFEST}" \
  --summary-json "${SUMMARY_JSON}"

if [[ ! -s "${TRAIN_MANIFEST}" ]]; then
  echo "ERROR: Train manifest is missing or empty: ${TRAIN_MANIFEST}" >&2
  exit 1
fi
if [[ ! -s "${VAL_MANIFEST}" ]]; then
  echo "ERROR: Validation manifest is missing or empty: ${VAL_MANIFEST}" >&2
  exit 1
fi

AMP_FLAG=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_FLAG=(--amp)
fi

echo "==> Evaluating z_lr baseline on AU validation"
python3 eval_experiment.py \
  --experiment baseline \
  --prediction-source z_lr \
  --data-root "${DATA_ROOT}" \
  --manifest "${VAL_MANIFEST}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --output-json "${EVAL_DIR}/eval_z_lr.json"

for ARCH in "${ARCHES[@]}"; do
  CKPT_PATH="${CKPT_DIR}/dem_${ARCH}_${EPOCHS}ep.pt"
  ARCH_EVAL_JSON="${EVAL_DIR}/eval_${ARCH}.json"

  echo "==> Training ${ARCH} (${EPOCHS} epochs, batch=${BATCH_SIZE}, workers=${WORKERS})"
  python3 train_experiment.py \
    --experiment baseline \
    --arch "${ARCH}" \
    --data-root "${DATA_ROOT}" \
    --manifest "${TRAIN_MANIFEST}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    "${AMP_FLAG[@]}" \
    --checkpoint-out "${CKPT_PATH}"

  echo "==> Evaluating ${ARCH} on AU validation"
  python3 eval_experiment.py \
    --experiment baseline \
    --prediction-source model \
    --arch "${ARCH}" \
    --checkpoint "${CKPT_PATH}" \
    --manifest "${VAL_MANIFEST}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    "${AMP_FLAG[@]}" \
    --output-json "${ARCH_EVAL_JSON}"
done

echo "==> Run complete"
echo "Run directory: ${RUN_DIR}"
echo "Train manifest: ${TRAIN_MANIFEST}"
echo "Val manifest:   ${VAL_MANIFEST}"
echo "Eval outputs:   ${EVAL_DIR}"
