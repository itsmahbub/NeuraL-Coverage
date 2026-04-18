#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-ImageNet}"
MODEL="${MODEL:-resnet50}"
CRITERION="${CRITERION:-NLC}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/data/output/Coverage/Fuzzer/}"
RESULTS_JSON="${RESULTS_JSON:-${SCRIPT_DIR}/results.json}"
CLIP_THRESHOLD="${CLIP_THRESHOLD:-0.80}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-ViT-B-32}"
CLIP_PRETRAINED="${CLIP_PRETRAINED:-openai}"

run_variant() {
  local name="$1"
  shift

  echo
  echo "=== Running variant: ${name} ==="

  # "${PYTHON_BIN}" "${SCRIPT_DIR}/fuzz.py" \
  #   --dataset "${DATASET}" \
  #   --model "${MODEL}" \
  #   --criterion "${CRITERION}" \
  #   --output_dir "${OUTPUT_DIR}" \
  #   --random_seed 0 \
  #   "$@"

  local exp_name="${DATASET}-${MODEL}-${CRITERION}"
  if [[ "${name}" == "rounding" ]]; then
    exp_name="${exp_name}-rounding"
  elif [[ "${name}" == "enforce-plausibility" ]]; then
    exp_name="${exp_name}-enforce-plausibility"
  elif [[ "${name}" == "rounding-enforce-plausibility" ]]; then
    exp_name="${exp_name}-rounding-enforce-plausibility"
  fi

  local image_root="${OUTPUT_DIR%/}/${exp_name}/image"
  local drifted_json="${OUTPUT_DIR%/}/${exp_name}/clip_drifted.json"
  local clip_details_json="${OUTPUT_DIR%/}/${exp_name}/clip_details.json"
  local repro_details_json="${OUTPUT_DIR%/}/${exp_name}/reproducibility_details.json"

  echo "--- Computing naturalness for ${name} ---"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_naturalness.py" \
    --image-root "${image_root}" \
    --output-json "${RESULTS_JSON}"

  echo "--- Computing reproducibility for ${name} ---"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_reproducibility.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --image-root "${image_root}" \
    --output-json "${RESULTS_JSON}" \
    --details-json "${repro_details_json}"

  echo "--- Computing CLIP drift for ${name} ---"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_clip_drift.py" \
    --image-root "${image_root}" \
    --output-json "${RESULTS_JSON}" \
    --threshold "${CLIP_THRESHOLD}" \
    --model-name "${CLIP_MODEL_NAME}" \
    --pretrained "${CLIP_PRETRAINED}" \
    --details-json "${clip_details_json}" \
    --drifted-json "${drifted_json}"

  echo "Saved drifted cases to: ${drifted_json}"
}

run_variant "baseline"
# run_variant "rounding" --use_rounding
# run_variant "enforce-plausibility" --enforce_plausibility
# run_variant "rounding-enforce-plausibility" --use_rounding --enforce_plausibility

echo
echo "All experiments complete."
echo "Shared results JSON: ${RESULTS_JSON}"
