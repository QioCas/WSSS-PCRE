#!/usr/bin/env bash
set -euo pipefail
trap '' HUP

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON="${PYTHON:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
if ! [[ "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPUs detected. Check CUDA_VISIBLE_DEVICES and your PyTorch installation." >&2
    exit 1
fi

exec "$PYTHON" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train_coco.py \
    --spg 2 \
    --grad_accum_steps 4 \
    --img_folder dataset/COCO2014/coco \
    --label_folder dataset/COCO2014/coco/SegmentationClass \
    --list_folder datasets/coco \
    --work_dir experiments/coco_baseline \
    --save_ckpt \
    "$@"
