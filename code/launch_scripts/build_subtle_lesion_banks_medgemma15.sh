#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL_PATH="${MODEL_PATH:-}"
BANK_SOURCE_ROOT="${BANK_SOURCE_ROOT:-}"
BANK_ROOT="${BANK_ROOT:-${ROOT_DIR}/banks/subtle_lesion_medgemma15}"
DEVICE="${DEVICE:-cuda}"
VISION_LAYER="${VISION_LAYER:-1}"
FORCE="${FORCE:-0}"

NUM_LESION_PROTOTYPES="${NUM_LESION_PROTOTYPES:-8}"
NUM_NORMAL_PROTOTYPES="${NUM_NORMAL_PROTOTYPES:-8}"
MAX_LESION_PER_IMAGE="${MAX_LESION_PER_IMAGE:-256}"
MAX_BG_PER_IMAGE="${MAX_BG_PER_IMAGE:-0}"
MAX_LESION_TOKENS="${MAX_LESION_TOKENS:-30000}"
MAX_BG_PER_POSITION="${MAX_BG_PER_POSITION:-0}"
MAX_BG_GLOBAL_TOKENS="${MAX_BG_GLOBAL_TOKENS:-60000}"
BODY_INTENSITY_THRESHOLD="${BODY_INTENSITY_THRESHOLD:-5.0}"
SHAPE_RADIUS="${SHAPE_RADIUS:-2}"
REX_RAW_ROOT="${REX_RAW_ROOT:-}"
LIDC_SOURCE_MANIFEST="${LIDC_SOURCE_MANIFEST:-}"
MASK_OVERLAP_THRESHOLD="${MASK_OVERLAP_THRESHOLD:-0.0}"
KMEANS_ITERS="${KMEANS_ITERS:-50}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-0}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required, for example: MODEL_PATH=/path/to/medgemma-1.5-4b-it $0" >&2
  exit 2
fi

if [[ -z "${BANK_SOURCE_ROOT}" ]]; then
  echo "BANK_SOURCE_ROOT is required and must point to the independent bank/train split, not to the test benchmark root." >&2
  exit 2
fi

export PYTHONPATH="${ROOT_DIR}/code/runners:${ROOT_DIR}/code/method:${ROOT_DIR}/code/dependencies:${PYTHONPATH:-}"
mkdir -p "${BANK_ROOT}"

RUNNER="${ROOT_DIR}/code/runners/evaluate_rex_easylens_model_native.py"

build_bank() {
  local dataset_name="$1"
  local bank_file="$2"
  local source_dir="${BANK_SOURCE_ROOT}/${dataset_name}"
  local output_path="${BANK_ROOT}/${bank_file}"
  local extra_args=()

  if [[ "${dataset_name}" == "rex" && -n "${REX_RAW_ROOT}" ]]; then
    extra_args+=(--raw-root "${REX_RAW_ROOT}" --mask-overlap-threshold "${MASK_OVERLAP_THRESHOLD}")
  fi
  if [[ "${dataset_name}" == "lidc" && -n "${LIDC_SOURCE_MANIFEST}" ]]; then
    extra_args+=(--lidc-source-manifest "${LIDC_SOURCE_MANIFEST}" --mask-overlap-threshold "${MASK_OVERLAP_THRESHOLD}")
  fi

  if [[ -s "${output_path}" && "${FORCE}" != "1" ]]; then
    echo "[SKIP] ${bank_file} exists"
    return
  fi

  if [[ ! -d "${source_dir}" ]]; then
    echo "Missing bank source directory: ${source_dir}" >&2
    exit 3
  fi

  echo "[BUILD] ${source_dir} -> ${output_path}"
  "${PYTHON_BIN}" "${RUNNER}" build-bank \
    --model-kind medgemma1.5 \
    --model-path "${MODEL_PATH}" \
    --benchmark-dir "${source_dir}" \
    --output-path "${output_path}" \
    --vision-layer "${VISION_LAYER}" \
    --device "${DEVICE}" \
    --num-lesion-prototypes "${NUM_LESION_PROTOTYPES}" \
    --num-normal-prototypes "${NUM_NORMAL_PROTOTYPES}" \
    --max-lesion-per-image "${MAX_LESION_PER_IMAGE}" \
    --max-bg-per-image "${MAX_BG_PER_IMAGE}" \
    --max-lesion-tokens "${MAX_LESION_TOKENS}" \
    --max-bg-per-position "${MAX_BG_PER_POSITION}" \
    --max-bg-global-tokens "${MAX_BG_GLOBAL_TOKENS}" \
    --body-intensity-threshold "${BODY_INTENSITY_THRESHOLD}" \
    --shape-radius "${SHAPE_RADIUS}" \
    --kmeans-iters "${KMEANS_ITERS}" \
    --seed "${SEED}" \
    --limit "${LIMIT}" \
    "${extra_args[@]}"
}

build_bank rex rex_shape_bank_v2.pt
build_bank lidc lidc_shape_bank_v2.pt
build_bank abdomen abdomen_shape_bank_v2.pt

echo "[DONE] subtle-lesion MedGemma1.5 banks written to ${BANK_ROOT}"
