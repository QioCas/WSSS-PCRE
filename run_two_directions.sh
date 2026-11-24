#!/usr/bin/env bash
set -e

WORKSPACE_DIR="/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE"
cd "$WORKSPACE_DIR"

echo "========================================================================"
echo "STARTING TWO-PHASE ADVANCED FOUNDATION MODEL EVALUATION"
echo "Phase 1: SAM Advanced Prompting (Negative Points + Multi-Mask Disambiguation)"
echo "Phase 2: SAM + CLIP Zero-Shot Co-occurrence Verification"
echo "Timestamp: $(date)"
echo "========================================================================"

# --- Phase 1 ---
echo ""
echo ">>> [Phase 1/2] Executing SAM Advanced Prompting..."
python3 experiments/sam_advanced_prompting_20261121/run_sam_advanced_prompting.py

echo ""
echo ">>> Phase 1 completed successfully!"

# --- Phase 2 ---
echo ""
echo ">>> [Phase 2/2] Executing SAM + CLIP Zero-Shot Co-occurrence Verification..."
python3 experiments/sam_clip_synergy_20261121/run_sam_clip_synergy.py

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "Timestamp: $(date)"
echo "========================================================================"
