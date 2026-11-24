#!/usr/bin/env bash
# ==============================================================================
# ALL.sh - Kịch bản cài đặt và thiết lập toàn diện tự động cho WSSS-PCRE
# Tự động kiểm tra và cài đặt:
#   1. Môi trường Python & Thư viện phụ thuộc (requirements.txt, clip, sam, crf)
#   2. Cấu trúc thư mục (datasets, pretrained, experiments)
#   3. Trọng số tiền huấn luyện (ViT-B ImageNet, SAM ViT-B)
#   4. Bộ dữ liệu & Danh sách phân chia (PASCAL VOC 2012, MS COCO 2014)
#   5. Phân quyền và kiểm thử đơn vị tự động (Sanity Check)
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python3}"

echo "========================================================================"
echo "🚀 BẮT ĐẦU CÀI ĐẶT & THIẾT LẬP HỆ THỐNG WSSS-PCRE TỰ ĐỘNG"
echo "Thư mục gốc: $ROOT_DIR"
echo "Python thực thi: $($PYTHON --version)"
echo "Thời gian: $(date)"
echo "========================================================================"

# ------------------------------------------------------------------------------
# 1. CÀI ĐẶT CÁC GÓI PYTHON CƠ BẢN TỪ REQUIREMENTS.TXT
# ------------------------------------------------------------------------------
echo ""
echo ">>> [1/6] Cài đặt & cập nhật các thư viện phụ thuộc Python..."
$PYTHON -m pip install --upgrade pip setuptools wheel --quiet

if [ -f "requirements.txt" ]; then
    echo "Đang cài đặt từ requirements.txt..."
    $PYTHON -m pip install -r requirements.txt --quiet
    echo "✓ Đã cài đặt xong requirements.txt."
fi

# ------------------------------------------------------------------------------
# 2. CÀI ĐẶT CÁC THƯ VIỆN NỀN TẢNG (FOUNDATION MODELS & CRF)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [2/6] Kiểm tra & cài đặt các thư viện Foundation Models (SAM, CLIP, CRF)..."

# PyDenseCRF
if ! $PYTHON -c "import pydensecrf" &>/dev/null; then
    echo "Đang cài đặt pydensecrf2..."
    $PYTHON -m pip install pydensecrf2 --quiet || echo "⚠️ Không thể tự cài pydensecrf2, bỏ qua nếu chỉ chạy train."
else
    echo "✓ pydensecrf đã sẵn sàng."
fi

# pycocotools
if ! $PYTHON -c "import pycocotools" &>/dev/null; then
    echo "Đang cài đặt pycocotools..."
    $PYTHON -m pip install pycocotools --quiet
else
    echo "✓ pycocotools đã sẵn sàng."
fi

# Segment Anything (SAM)
if ! $PYTHON -c "import segment_anything" &>/dev/null; then
    echo "Đang cài đặt Segment Anything Model (SAM)..."
    $PYTHON -m pip install git+https://github.com/facebookresearch/segment-anything.git --quiet
else
    echo "✓ segment_anything đã sẵn sàng."
fi

# OpenAI CLIP
if ! $PYTHON -c "import clip" &>/dev/null; then
    echo "Đang cài đặt OpenAI CLIP..."
    $PYTHON -m pip install ftfy regex git+https://github.com/openai/CLIP.git --quiet
else
    echo "✓ OpenAI clip đã sẵn sàng."
fi

# Bilateral Filter (rloss)
if $PYTHON -c "import bilateralfilter" &>/dev/null; then
    echo "✓ bilateralfilter extension đã sẵn sàng."
else
    echo "⚠️ Module bilateralfilter chưa được cài đặt."
    echo "   Nếu cần dùng regularized loss, hãy biên dịch theo hướng dẫn rloss."
fi

# ------------------------------------------------------------------------------
# 3. THIẾT LẬP CẤU TRÚC THƯ MỤC
# ------------------------------------------------------------------------------
echo ""
echo ">>> [3/6] Thiết lập cấu trúc thư mục chuẩn..."
mkdir -p dataset
mkdir -p datasets/voc datasets/coco
mkdir -p pretrained/checkpoints
mkdir -p experiments
echo "✓ Đã tạo đầy đủ các thư mục cần thiết."

# ------------------------------------------------------------------------------
# 4. KIỂM TRA & TẢI TRỌNG SỐ TIỀN HUẤN LUYỆN (PRETRAINED WEIGHTS)
# ------------------------------------------------------------------------------
echo ""
echo ">>> [4/6] Kiểm tra trọng số tiền huấn luyện (Pretrained Weights)..."

# ViT-B_16 ImageNet Pretrained
VIT_PTH="pretrained/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"
if [ ! -f "$VIT_PTH" ]; then
    echo "Đang tải trọng số ViT-B ImageNet pretrained (khoảng 346 MB)..."
    wget -c "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth" \
        -O "$VIT_PTH" || echo "⚠️ Tải ViT-B thất bại, bạn có thể tải thủ công vào $VIT_PTH."
else
    echo "✓ ViT-B ImageNet pretrained đã sẵn sàng tại $VIT_PTH."
fi

# SAM ViT-B Checkpoint
SAM_PTH="pretrained/checkpoints/sam_vit_b_01ec64.pth"
SAM_SRC="/home/jupyter-iec2024iot08/Workspace/.deps/pcre_sam/sam_vit_b_01ec64.pth"

if [ ! -f "$SAM_PTH" ]; then
    if [ -f "$SAM_SRC" ]; then
        echo "Tạo liên kết mềm cho SAM ViT-B từ $SAM_SRC..."
        ln -sf "$SAM_SRC" "$SAM_PTH"
    else
        echo "Đang tải trọng số SAM ViT-B (khoảng 375 MB)..."
        wget -c "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
            -O "$SAM_PTH" || echo "⚠️ Tải SAM weights thất bại, bạn có thể tải thủ công vào $SAM_PTH."
    fi
else
    echo "✓ SAM ViT-B checkpoint đã sẵn sàng tại $SAM_PTH."
fi

# ------------------------------------------------------------------------------
# 5. KIỂM TRA TẬP DỮ LIỆU & TỰ ĐỘNG SINH METADATA
# ------------------------------------------------------------------------------
echo ""
echo ">>> [5/6] Kiểm tra tập dữ liệu & Metadata..."

# Pascal VOC 2012
if [ -d "dataset/VOC2012" ]; then
    echo "✓ Đã tìm thấy thư mục dataset/VOC2012."
    if [ ! -f "datasets/voc/train_aug.txt" ] || [ ! -f "datasets/voc/cls_labels_onehot.npy" ]; then
        echo "Đang sinh danh sách phân chia và nhãn cấp ảnh cho VOC2012..."
        $PYTHON scripts/prepare_voc.py \
            --data_folder ./dataset/VOC2012 \
            --train_list ./datasets/voc/train_aug.txt \
            --output datasets/voc || true
    else
        echo "✓ Metadata datasets/voc đã đầy đủ."
    fi
else
    echo "ℹ️ Chưa tìm thấy dataset/VOC2012. Hãy tải hoặc tạo symlink:"
    echo "   ln -s /path/to/VOC2012 dataset/VOC2012"
fi

# MS COCO 2014
if [ -d "dataset/COCO2014" ]; then
    echo "✓ Đã tìm thấy thư mục dataset/COCO2014."
    if [ ! -f "datasets/coco/val.txt" ] && [ -f "dataset/COCO2014/annotations/instances_val2014.json" ]; then
        echo "Đang sinh metadata cho COCO 2014..."
        $PYTHON scripts/prepare_coco.py || true
    else
        echo "✓ Metadata datasets/coco đã sẵn sàng."
    fi
else
    echo "ℹ️ Chưa tìm thấy dataset/COCO2014. Xem README.md để biết cách tải."
fi

# ------------------------------------------------------------------------------
# 6. PHÂN QUYỀN VÀ KIỂM THỬ ĐƠN VỊ TỰ ĐỘNG
# ------------------------------------------------------------------------------
echo ""
echo ">>> [6/6] Cấp quyền thực thi và chạy kiểm thử tự động (Unit Test)..."
chmod +x *.sh 2>/dev/null || true

echo "Chạy kiểm thử đơn vị cho các module SAM-PCRE..."
$PYTHON -m unittest tests/test_sam_innovations.py

echo ""
echo "========================================================================"
echo "🎉 HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG TOÀN DIỆN THÀNH CÔNG!"
echo "Giờ bạn có thể chạy ./TRAIN.sh để thực hiện các tiến trình huấn luyện."
echo "========================================================================"
