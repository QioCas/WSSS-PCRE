#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPUs detected. Check CUDA_VISIBLE_DEVICES and your PyTorch installation." >&2
    exit 1
fi

exec "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train_voc_sam_pcre_v2.py \
    --spg 4 \
    --crop_size 448 \
    --max_iters 20000 \
    --eval_iters 5000 \
    --log_iters 100 \
    --save_ckpt \
    --data_folder ./dataset/VOC2012 \
    --list_folder datasets/voc \
    --sam_cache_dir experiments/representation_sam_20261120/cache/sam/labels \
    --work_dir experiments/voc_sam_pcre_v2 \
    --enable_sam_shrinkage True \
    --enable_sam_crme_barrier True \
    --enable_sat_cr True \
    --sat_cr_purity 0.85 \
    --enable_sbf_distill True \
    --boundary_weight_alpha 2.0 \
    "$@"
