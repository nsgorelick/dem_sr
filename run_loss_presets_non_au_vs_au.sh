#!/usr/bin/env bash
set -euo pipefail

# Loss-preset screening workflow: same non-AU/AU split as run_arch_non_au_vs_au.sh,
# but fixes --arch film_unet and iterates the four contour-aware --loss-preset options:
#
#   baseline  | elev + slope (reference; matches current loss_dem)
#   geom      | + gradient dx/dy L1 + Laplacian L1 + 2x multi-scale elev L1
#   multitask | geom + contour SDF L1 + soft contour-indicator L1
#   contour   | baseline + contour SDF L1 only
#
# Required:
#   PATCH_TABLE=<path-to-patch-summary.{csv,json,geojson}>
#
# Optional overrides:
#   DATA_ROOT=/data/training
#   VENV_PATH=/home/gorelick/venv-cu128
#   OUTPUT_ROOT=./runs
#   RUN_NAME=loss_preset_non_au_vs_au_YYYYmmdd_HHMMSS
#   EPOCHS=3
#   BATCH_SIZE=32
#   WORKERS=3
#   USE_AMP=1
#   ARCH=film_unet
#   CONTOUR_INTERVAL=10.0
#   PRESETS="baseline geom multitask contour"
#   HARD_FRACTION=0.10             (set to 1.0 to use the full non-AU train manifest)
#   VAL_HARD_FRACTION=0.10         (set to 1.0 to use the full AU val manifest)
#   HARD_SCORE_FIELD=resid_scale   (p90_slope | relief | mean_uncert also work)
#   HARD_MIN_MEAN_W=0.4
#   HARD_MIN_VALID_FRAC=0.8
#   HARD_MIN_GT_COVERAGE=0.8
#   HARD_MIN_RELIEF=0.5
#   HARD_MAX_FRAC_WATER=0.25

PATCH_TABLE="${PATCH_TABLE:-}"
DATA_ROOT="${DATA_ROOT:-/data/training}"
VENV_PATH="${VENV_PATH:-/home/gorelick/venv-cu128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./runs}"
RUN_NAME="${RUN_NAME:-loss_preset_non_au_vs_au_$(date +%Y%m%d_%H%M%S)}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-3}"
USE_AMP="${USE_AMP:-1}"
ARCH="${ARCH:-film_unet}"
CONTOUR_INTERVAL="${CONTOUR_INTERVAL:-10.0}"
PRESETS="${PRESETS:-baseline geom multitask contour}"
HARD_FRACTION="${HARD_FRACTION:-0.10}"
VAL_HARD_FRACTION="${VAL_HARD_FRACTION:-0.10}"
HARD_SCORE_FIELD="${HARD_SCORE_FIELD:-resid_scale}"
HARD_MIN_MEAN_W="${HARD_MIN_MEAN_W:-0.4}"
HARD_MIN_VALID_FRAC="${HARD_MIN_VALID_FRAC:-0.8}"
HARD_MIN_GT_COVERAGE="${HARD_MIN_GT_COVERAGE:-0.8}"
HARD_MIN_RELIEF="${HARD_MIN_RELIEF:-0.5}"
HARD_MAX_FRAC_WATER="${HARD_MAX_FRAC_WATER:-0.25}"

if [[ -z "${PATCH_TABLE}" ]]; then
  echo "ERROR: PATCH_TABLE is required." >&2
  echo "Example: PATCH_TABLE=/path/to/patch_summary.csv ./run_loss_presets_non_au_vs_au.sh" >&2
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

TRAIN_MANIFEST_FULL="${MANIFEST_DIR}/train_non_au_manifest.txt"
VAL_MANIFEST_FULL="${MANIFEST_DIR}/val_au_manifest.txt"
SUMMARY_JSON="${MANIFEST_DIR}/manifest_summary_non_au_vs_au.json"

echo "==> Building manifests with make_manifest.py (locked country: AU)"
python3 make_manifest.py \
  --data-root "${DATA_ROOT}" \
  --patch-table "${PATCH_TABLE}" \
  --locked-country AU \
  --val-fraction 0 \
  --holdout-out "${VAL_MANIFEST_FULL}" \
  --train-out "${TRAIN_MANIFEST_FULL}" \
  --summary-json "${SUMMARY_JSON}"

if [[ ! -s "${TRAIN_MANIFEST_FULL}" ]]; then
  echo "ERROR: Train manifest is missing or empty: ${TRAIN_MANIFEST_FULL}" >&2
  exit 1
fi
if [[ ! -s "${VAL_MANIFEST_FULL}" ]]; then
  echo "ERROR: Validation manifest is missing or empty: ${VAL_MANIFEST_FULL}" >&2
  exit 1
fi

subsample_hard() {
  # $1 role label (train|val), $2 input manifest, $3 fraction, outputs path via echo
  local role="$1"
  local src="$2"
  local fraction="$3"
  local use_hard
  use_hard=$(python3 -c "import sys; f=float(sys.argv[1]); print('1' if 0 < f < 1 else '0')" "${fraction}")
  if [[ "${use_hard}" != "1" ]]; then
    echo "${src}"
    return 0
  fi
  local pct
  pct=$(python3 -c "import sys; print(int(round(float(sys.argv[1]) * 100)))" "${fraction}")
  local region
  if [[ "${role}" == "train" ]]; then
    region="non_au"
  else
    region="au"
  fi
  local out="${MANIFEST_DIR}/${role}_${region}_hard_${HARD_SCORE_FIELD}_p${pct}_manifest.txt"
  local summary="${MANIFEST_DIR}/${role}_${region}_hard_${HARD_SCORE_FIELD}_p${pct}_summary.json"
  echo "==> Selecting hardest ${pct}% of ${role} (${region}) patches by ${HARD_SCORE_FIELD}" >&2
  python3 select_hard_patches.py \
    --manifest "${src}" \
    --patch-table "${PATCH_TABLE}" \
    --fraction "${fraction}" \
    --score-field "${HARD_SCORE_FIELD}" \
    --min-mean-w "${HARD_MIN_MEAN_W}" \
    --min-valid-frac "${HARD_MIN_VALID_FRAC}" \
    --min-gt-coverage "${HARD_MIN_GT_COVERAGE}" \
    --min-relief "${HARD_MIN_RELIEF}" \
    --max-frac-water "${HARD_MAX_FRAC_WATER}" \
    --out "${out}" \
    --summary-json "${summary}" >&2
  if [[ ! -s "${out}" ]]; then
    echo "ERROR: Hard ${role} manifest is missing or empty: ${out}" >&2
    exit 1
  fi
  echo "${out}"
}

TRAIN_MANIFEST="$(subsample_hard train "${TRAIN_MANIFEST_FULL}" "${HARD_FRACTION}")"
VAL_MANIFEST="$(subsample_hard val "${VAL_MANIFEST_FULL}" "${VAL_HARD_FRACTION}")"
echo "==> Train manifest: ${TRAIN_MANIFEST}"
echo "==> Val manifest:   ${VAL_MANIFEST}"

AMP_FLAG=()
if [[ "${USE_AMP}" == "1" ]]; then
  AMP_FLAG=(--amp)
fi

echo "==> Evaluating z_lr baseline on AU validation"
python3 eval_dem.py \
  --prediction-source z_lr \
  --data-root "${DATA_ROOT}" \
  --manifest "${VAL_MANIFEST}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --contour-interval "${CONTOUR_INTERVAL}" \
  --output-json "${EVAL_DIR}/eval_z_lr.json"

for PRESET in ${PRESETS}; do
  CKPT_PATH="${CKPT_DIR}/dem_${ARCH}_${PRESET}_${EPOCHS}ep.pt"
  PRESET_EVAL_JSON="${EVAL_DIR}/eval_${ARCH}_${PRESET}.json"

  echo "==> Training ${ARCH} loss-preset=${PRESET} (${EPOCHS} epochs, batch=${BATCH_SIZE}, workers=${WORKERS})"
  python3 train_dem.py \
    --arch "${ARCH}" \
    --loss-preset "${PRESET}" \
    --contour-interval "${CONTOUR_INTERVAL}" \
    --data-root "${DATA_ROOT}" \
    --manifest "${TRAIN_MANIFEST}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    "${AMP_FLAG[@]}" \
    --checkpoint-out "${CKPT_PATH}"

  echo "==> Evaluating ${ARCH}/${PRESET} on AU validation"
  python3 eval_dem.py \
    --prediction-source model \
    --arch "${ARCH}" \
    --checkpoint "${CKPT_PATH}" \
    --manifest "${VAL_MANIFEST}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --contour-interval "${CONTOUR_INTERVAL}" \
    "${AMP_FLAG[@]}" \
    --output-json "${PRESET_EVAL_JSON}"
done

echo "==> Run complete"
echo "Run directory:  ${RUN_DIR}"
echo "Train manifest: ${TRAIN_MANIFEST}"
echo "Val manifest:   ${VAL_MANIFEST}"
echo "Eval outputs:   ${EVAL_DIR}"
