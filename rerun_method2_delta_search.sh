#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/jupyter-iec2024iot06/anaconda3/envs/wsss/bin/python3}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python3"
fi

BASELINE_RUN_DIR="experiments/pcre_sat_20261117_200400_39e382/baseline"
BASELINE_EXPORT_DIR="experiments/inference_voc_val_20261119/full_baseline_rerun"
BASE_MASKS_DIR="${BASELINE_EXPORT_DIR}/baseline/crf/masks"
OUTPUT_DIR="experiments/sam_advanced_prompting_20261121/evaluation_uniform"

DELTA_INIT="${DELTA_INIT:-0.2}"
DELTA_GRID="${DELTA_GRID:-0.0 0.1 0.2 0.3 0.4 0.5}"
LIMIT="${LIMIT:-0}"
CRF_WORKERS="${CRF_WORKERS:-12}"

echo "========================================================================"
echo "  RERUN METHOD 2: SAM ADVANCED PROMPTING (UNIFORM NP + MMD)"
echo "  Delta Init: $DELTA_INIT | Delta Grid: $DELTA_GRID"
echo "========================================================================"

# Step 1: Export Baseline PCRE DenseCRF Masks (1,449 VOC Val images) if needed
if [[ ! -d "$BASE_MASKS_DIR" ]] || [[ $(ls -1 "$BASE_MASKS_DIR"/*.png 2>/dev/null | wc -l) -lt 1449 && "$LIMIT" -eq 0 ]]; then
    echo "[Step 1/2] Exporting Baseline PCRE DenseCRF masks from $BASELINE_RUN_DIR..."
    rm -rf "${BASELINE_EXPORT_DIR}/baseline"
    "$PYTHON_BIN" tools/infer_voc_run.py \
        --run-dir "$BASELINE_RUN_DIR" \
        --output-dir "$BASELINE_EXPORT_DIR" \
        --mode baseline \
        --crf-workers "$CRF_WORKERS" \
        --limit "$LIMIT"
else
    echo "[Step 1/2] Found existing Baseline PCRE DenseCRF masks in $BASE_MASKS_DIR."
fi

# Step 2: Run Method 2 with Delta Parameter Search
echo "[Step 2/2] Running Method 2 (Uniform NP + MMD) with Delta Grid Search..."
"$PYTHON_BIN" experiments/sam_advanced_prompting_20261121/run_sam_advanced_prompting.py \
    --base_masks_dir "$BASE_MASKS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --delta_init "$DELTA_INIT" \
    --delta_grid $DELTA_GRID \
    --search_delta \
    --limit "$LIMIT"

echo "========================================================================"
echo "  METHOD 2 RERUN COMPLETE. Results saved in: $OUTPUT_DIR"
echo "========================================================================"
