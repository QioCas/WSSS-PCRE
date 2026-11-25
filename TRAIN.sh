#!/usr/bin/env bash
# ==============================================================================
# TRAIN.sh - Kịch bản điều phối huấn luyện tự động cho các tiến trình chưa xong
# Quản lý và thực thi tuần tự các thực nghiệm Pascal VOC:
#   - Giai đoạn 1: SAT-CR (SAM-Affinity Token Contrastive Regularization)
#   - Giai đoạn 2: SBF-Distill (SAM-Guided Boundary-Focal Density Distillation)
#   - Giai đoạn 3: Paradigm 2 Full Synergy (SAT-CR + SBF-Distill kết hợp)
#   - (Tùy chọn) Giai đoạn 4: Đánh giá COCO Baseline 80k (--coco-eval)
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

if [ -z "${PYTHON:-}" ]; then
    if [ -x "/home/jupyter-iec2024iot06/anaconda3/envs/wsss/bin/python3" ]; then
        PYTHON="/home/jupyter-iec2024iot06/anaconda3/envs/wsss/bin/python3"
    elif command -v conda &>/dev/null && conda info --envs | grep -q "/envs/wsss"; then
        PYTHON="$(conda info --base)/envs/wsss/bin/python3"
    else
        PYTHON="python3"
    fi
fi
export PYTHON
LOG_FILE="$ROOT_DIR/experiments/train_pipeline.log"
mkdir -p "$ROOT_DIR/experiments"

log() {
    local now
    now="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$now] $*" | tee -a "$LOG_FILE"
}

show_help() {
    echo "Cách sử dụng: ./TRAIN.sh [TÙY CHỌN]"
    echo ""
    echo "Tùy chọn:"
    echo "  --all            (Mặc định) Chạy tuần tự tất cả các tiến trình VOC chưa hoàn thành (Stages 1, 2, 3)."
    echo "  --stage 1        Chỉ chạy Stage 1: SAT-CR (experiments/voc_sam_sat_cr)"
    echo "  --stage 2        Chỉ chạy Stage 2: SBF-Distill (experiments/voc_sam_sbf_distill)"
    echo "  --stage 3        Chỉ chạy Stage 3: Full Paradigm 2 (experiments/voc_sam_pcre_v2)"
    echo "  --coco-eval      (Tùy chọn) Chạy đánh giá COCO 80k (Multi-scale + DenseCRF)"
    echo "  --status         Kiểm tra trạng thái của từng giai đoạn (Đã xong / Chưa xong)"
    echo "  --wait           Tự động chờ GPU có đủ VRAM trống (>= 16 GB) thay vì thoát nếu bận"
    echo "  --help           Hiển thị trợ giúp này"
    exit 0
}

check_gpu() {
    log "Kiểm tra trạng thái phần cứng GPU..."
    local cuda_ok
    cuda_ok=$($PYTHON -c "import torch; print(torch.cuda.is_available() and torch.cuda.device_count() > 0)" 2>/dev/null || echo "False")
    
    if [ "$cuda_ok" != "True" ]; then
        log "⚠️ CẢNH BÁO: PyTorch không nhận diện được GPU CUDA (cuda.is_available = False)."
        log "   Nguyên nhân phổ biến: Driver NVIDIA cần khởi động lại hoặc máy chủ vừa restart."
        log "   Hãy kiểm tra bằng lệnh 'nvidia-smi' trên terminal."
        return 1
    fi
    local gpu_info
    gpu_info=$("$PYTHON" -c 'import torch; print(f"{torch.cuda.get_device_name(0)} (Tổng {torch.cuda.device_count()} GPU)")') || return 1
    log "✓ GPU CUDA đã sẵn sàng: $gpu_info"
    return 0
}

get_free_vram_mb() {
    local vram
    vram=$(nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader 2>/dev/null | head -n 1 || echo "0")
    echo "$vram" | tr -d '[:space:]'
}

wait_for_vram() {
    local min_vram=14000
    log "Đang kiểm tra dung lượng VRAM trống (yêu cầu tối thiểu ${min_vram} MiB)..."
    while true; do
        local free_vram
        free_vram=$(get_free_vram_mb)
        if [[ "$free_vram" =~ ^[0-9]+$ ]] && [ "$free_vram" -ge "$min_vram" ]; then
            log "✓ VRAM trống $free_vram MiB >= $min_vram MiB. Tiếp tục thực thi."
            break
        fi
        log "Đang chờ GPU... Hiện có: ${free_vram} MiB trống (cần: ${min_vram} MiB). Kiểm tra lại sau 30 giây..."
        sleep 30
    done
}

run_stage() {
    local name="$1"
    local script="$2"
    local target_ckpt="$3"
    local work_dir="$4"

    log "------------------------------------------------------------------------"
    log "▶ KIỂM TRA: $name"
    log "------------------------------------------------------------------------"

    if [ -f "$target_ckpt" ]; then
        log "✓ [BỎ QUA] $name đã có kết quả hoàn chỉnh tại:"
        log "   $target_ckpt"
        return 0
    fi

    log "Khởi chạy $name bằng lệnh: bash $script"
    if [ "${AUTO_WAIT:-0}" -eq 1 ]; then
        wait_for_vram
    fi

    local start_time
    start_time=$(date +%s)
    
    mkdir -p "$work_dir"
    
    # Thực thi script và ghi log song song
    set +e
    bash "$ROOT_DIR/$script" 2>&1 | tee -a "$work_dir/live_run.log" | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time
    end_time=$(date +%s)
    local duration=$(( (end_time - start_time) / 60 ))

    if [ $exit_code -eq 0 ]; then
        log "✓ [THÀNH CÔNG] $name hoàn tất sau $duration phút (Mã thoát: 0)."
    else
        log "❌ [LỖI] $name kết thúc với mã lỗi $exit_code sau $duration phút."
        return $exit_code
    fi
}

run_coco_eval() {
    log "------------------------------------------------------------------------"
    log "▶ KIỂM TRA: Đánh giá chính thức COCO Baseline 80k (Multi-scale + DenseCRF)"
    log "------------------------------------------------------------------------"

    local coco_ckpt="$ROOT_DIR/experiments/coco_baseline/2026-11/train_coco_22-17-53-40/checkpoints/model_iter_80000.pth"
    local out_dir="$ROOT_DIR/experiments/coco_baseline/full_eval_80k"

    if [ ! -f "$coco_ckpt" ]; then
        log "❌ Không tìm thấy checkpoint COCO 80k tại: $coco_ckpt"
        return 1
    fi

    if [ -f "$out_dir/evaluation_completed.flag" ]; then
        log "✓ [BỎ QUA] Đánh giá COCO 80k đã hoàn tất trước đó tại: $out_dir"
        return 0
    fi

    mkdir -p "$out_dir"
    log "Bắt đầu chạy tools/infer_seg_coco.py trên toàn bộ tập Val COCO..."
    
    set +e
    $PYTHON tools/infer_seg_coco.py \
        --model_path "$coco_ckpt" \
        --img_folder dataset/COCO2014/coco \
        --label_folder dataset/COCO2014/coco/SegmentationClass \
        --list_folder datasets/coco \
        --infer_set val \
        2>&1 | tee -a "$out_dir/eval.log" | tee -a "$LOG_FILE"
    local ret=${PIPESTATUS[0]}
    set -e

    if [ $ret -eq 0 ]; then
        touch "$out_dir/evaluation_completed.flag"
        log "✓ [THÀNH CÔNG] Đánh giá COCO 80k hoàn tất."
    else
        log "❌ [LỖI] Đánh giá COCO 80k thất bại (Mã lỗi: $ret)."
        return $ret
    fi
}

show_status() {
    echo "========================================================================"
    echo "📊 TRẠNG THÁI CÁC TIẾN TRÌNH THỰC NGHIỆM"
    echo "========================================================================"
    
    # Check Stage 1
    local s1_ckpt="experiments/voc_sam_sat_cr/checkpoints/model_iter_20000.pth"
    if [ -f "$s1_ckpt" ]; then
        echo "  [✓ HOÀN THÀNH] Stage 1 (SAT-CR): $s1_ckpt"
    else
        echo "  [⌛ CHƯA CHẠY] Stage 1 (SAT-CR): Cần chạy train_voc_sam_sat_cr.sh"
    fi

    # Check Stage 2
    local s2_ckpt="experiments/voc_sam_sbf_distill/checkpoints/model_iter_20000.pth"
    if [ -f "$s2_ckpt" ]; then
        echo "  [✓ HOÀN THÀNH] Stage 2 (SBF-Distill): $s2_ckpt"
    else
        echo "  [⌛ CHƯA CHẠY] Stage 2 (SBF-Distill): Cần chạy train_voc_sam_sbf_distill.sh"
    fi

    # Check Stage 3
    local s3_ckpt="experiments/voc_sam_pcre_v2/checkpoints/model_iter_20000.pth"
    if [ -f "$s3_ckpt" ]; then
        echo "  [✓ HOÀN THÀNH] Stage 3 (Paradigm 2 Full): $s3_ckpt"
    else
        echo "  [⌛ CHƯA CHẠY] Stage 3 (Paradigm 2 Full): Cần chạy train_voc_sam_pcre_v2.sh"
    fi

    # Check COCO Eval
    local coco_flag="experiments/coco_baseline/full_eval_80k/evaluation_completed.flag"
    if [ -f "$coco_flag" ]; then
        echo "  [✓ HOÀN THÀNH] COCO Baseline 80k Evaluation (Multi-scale + DenseCRF)"
    else
        echo "  [⌛ CHƯA CHẠY] COCO Baseline 80k Evaluation: Sẵn sàng chạy qua infer_seg_coco.py"
    fi
    echo "========================================================================"
    exit 0
}

# ------------------------------------------------------------------------------
# XỬ LÝ THAM SỐ DÒNG LỆNH
# ------------------------------------------------------------------------------
MODE="all"
AUTO_WAIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            show_help
            ;;
        --status)
            show_status
            ;;
        --wait)
            AUTO_WAIT=1
            shift
            ;;
        --stage)
            MODE="stage_$2"
            shift 2
            ;;
        --sat-cr)
            MODE="stage_1"
            shift
            ;;
        --sbf-distill)
            MODE="stage_2"
            shift
            ;;
        --pcre-v2)
            MODE="stage_3"
            shift
            ;;
        --coco-eval)
            MODE="coco_eval"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        *)
            echo "Tham số không hợp lệ: $1. Dùng --help để xem hướng dẫn."
            exit 1
            ;;
    esac
done

# Kiểm tra GPU trước khi chạy
if ! check_gpu; then
    if [ "$AUTO_WAIT" -eq 1 ]; then
        log "Chế độ --wait được kích hoạt. Sẽ tiếp tục thăm dò GPU..."
        wait_for_vram
    else
        log "Dừng thực thi vì GPU chưa sẵn sàng. (Dùng thêm tham số --wait để tự động chờ GPU)."
        exit 1
    fi
fi

log "========================================================================"
log "🚀 BẮT ĐẦU ĐIỀU PHỐI HUẤN LUYỆN (Chế độ: $MODE)"
log "========================================================================"

case "$MODE" in
    stage_1)
        run_stage "Stage 1: SAT-CR" "train_voc_sam_sat_cr.sh" \
            "experiments/voc_sam_sat_cr/checkpoints/model_iter_20000.pth" "experiments/voc_sam_sat_cr"
        ;;
    stage_2)
        run_stage "Stage 2: SBF-Distill" "train_voc_sam_sbf_distill.sh" \
            "experiments/voc_sam_sbf_distill/checkpoints/model_iter_20000.pth" "experiments/voc_sam_sbf_distill"
        ;;
    stage_3)
        run_stage "Stage 3: Full Paradigm 2 (SAT-CR + SBF-Distill)" "train_voc_sam_pcre_v2.sh" \
            "experiments/voc_sam_pcre_v2/checkpoints/model_iter_20000.pth" "experiments/voc_sam_pcre_v2"
        ;;
    coco_eval)
        run_coco_eval
        ;;
    all)
        log "Chạy tuần tự tất cả các giai đoạn VOC còn thiếu..."
        run_stage "Stage 1: SAT-CR" "train_voc_sam_sat_cr.sh" \
            "experiments/voc_sam_sat_cr/checkpoints/model_iter_20000.pth" "experiments/voc_sam_sat_cr"
        
        run_stage "Stage 2: SBF-Distill" "train_voc_sam_sbf_distill.sh" \
            "experiments/voc_sam_sbf_distill/checkpoints/model_iter_20000.pth" "experiments/voc_sam_sbf_distill"
        
        run_stage "Stage 3: Full Paradigm 2 (SAT-CR + SBF-Distill)" "train_voc_sam_pcre_v2.sh" \
            "experiments/voc_sam_pcre_v2/checkpoints/model_iter_20000.pth" "experiments/voc_sam_pcre_v2"
        ;;
esac

log "========================================================================"
log "🎉 TẤT CẢ TIẾN TRÌNH YÊU CẦU ĐÃ HOÀN TẤT THÀNH CÔNG!"
log "========================================================================"
