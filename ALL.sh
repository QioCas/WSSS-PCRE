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
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python3}"

# ------------------------------------------------------------------------------
# HỖ TRỢ THAM SỐ DÒNG LỆNH
# ------------------------------------------------------------------------------
show_help() {
    echo "Cách sử dụng: ./ALL.sh [TÙY CHỌN]"
    echo ""
    echo "Tùy chọn:"
    echo "  --create-env     Tự động tạo môi trường Conda 'wsss' (python=3.12) và hướng dẫn kích hoạt"
    echo "  --help, -h       Hiển thị hướng dẫn này"
    echo ""
    echo "Quy trình thiết lập môi trường chuẩn:"
    echo "  1️⃣  Tạo môi trường:     conda create -n wsss python=3.12 -y  (hoặc ./ALL.sh --create-env)"
    echo "  2️⃣  Kích hoạt:          conda activate wsss"
    echo "  3️⃣  Cài đặt toàn bộ:    ./ALL.sh"
    exit 0
}

create_conda_env() {
    echo "========================================================================"
    echo "🔧 ĐANG TỰ ĐỘNG TẠO MÔI TRƯỜNG CONDA 'wsss' (Python 3.12)..."
    echo "========================================================================"
    if command -v conda &>/dev/null; then
        conda create -n wsss python=3.12 -y
        echo ""
        echo "========================================================================"
        echo "🎉 ĐÃ TẠO THÀNH CÔNG MÔI TRƯỜNG CONDA 'wsss'!"
        echo "👉 Bây giờ bạn chỉ cần chạy 2 lệnh sau trên terminal để kích hoạt và cài đặt:"
        echo ""
        echo "   conda activate wsss"
        echo "   ./ALL.sh"
        echo "========================================================================"
        exit 0
    else
        echo "❌ Không tìm thấy lệnh 'conda' trong biến môi trường PATH."
        echo "   Vui lòng kiểm tra lại cài đặt Anaconda hoặc Miniconda trên máy của bạn."
        exit 1
    fi
}

# Xử lý tham số
if [ $# -gt 0 ]; then
    case "$1" in
        --help|-h)
            show_help
            ;;
        --create-env)
            create_conda_env
            ;;
        *)
            echo "Tham số không hợp lệ: $1. Dùng ./ALL.sh --help để xem hướng dẫn."
            exit 1
            ;;
    esac
fi

CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
PY_VER=$($PYTHON --version 2>&1 | awk '{print $2}')
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")

echo "========================================================================"
echo "🚀 BẮT ĐẦU CÀI ĐẶT & THIẾT LẬP HỆ THỐNG WSSS-PCRE TỰ ĐỘNG"
echo "Thư mục gốc: $ROOT_DIR"
echo "Môi trường: ${CONDA_ENV:-[Hệ thống / Chưa kích hoạt Conda]}"
echo "Python thực thi: $($PYTHON --version)"
echo "Thời gian: $(date)"
echo "========================================================================"

# Hướng dẫn thiết lập môi trường Conda nếu chưa vào môi trường riêng hoặc dùng Python >= 3.13
if [ -z "$CONDA_ENV" ] || [ "$CONDA_ENV" = "base" ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; }; then
    echo ""
    echo "┌────────────────────────────────────────────────────────────────────────┐"
    echo "│ 💡 HƯỚNG DẪN THIẾT LẬP MÔI TRƯỜNG CONDA (QUAN TRỌNG TRƯỚC KHI CÀI ĐẶT) │"
    echo "├────────────────────────────────────────────────────────────────────────┤"
    echo "│ Bạn hiện đang ở: ${CONDA_ENV:-[Môi trường hệ thống]} (Python $PY_VER)"
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
    echo "│ ⚠️  CẢNH BÁO: Python >= 3.13 đã bỏ C-API cũ, không tương thích pydensecrf2!"
    fi
    echo "│ Để tránh xung đột gói và bảo đảm 100% tương thích với các module C++,   │"
    echo "│ khuyến nghị bạn nên tạo & kích hoạt môi trường Conda riêng trước:     │"
    echo "│                                                                        │"
    echo "│   1️⃣  Tạo môi trường:     conda create -n wsss python=3.12 -y          │"
    echo "│   2️⃣  Kích hoạt môi trường: conda activate wsss                          │"
    echo "│   3️⃣  Chạy lại script:      ./ALL.sh                                   │"
    echo "│                                                                        │"
    echo "│ (Hoặc gõ: ./ALL.sh --create-env để kịch bản tự tạo môi trường giúp bạn)│"
    echo "└────────────────────────────────────────────────────────────────────────┘"
    echo ""

    if [ -t 0 ]; then
        ans="y"
        read -t 7 -p "👉 Bạn có muốn tiếp tục cài đặt vào môi trường hiện tại không? [Y/n] (Tự động tiếp tục sau 7s): " ans || ans="y"
        echo ""
        ans=${ans:-y}
        if [[ "$ans" =~ ^[nN] ]]; then
            echo "Đã dừng cài đặt."
            echo "Hãy gõ 2 lệnh sau để thiết lập môi trường Conda rồi chạy lại:"
            echo "   conda create -n wsss python=3.12 -y"
            echo "   conda activate wsss && ./ALL.sh"
            exit 0
        fi
    fi
else
    echo "✓ Đang chạy trong môi trường Conda cách ly: '$CONDA_ENV' (Python $PY_VER)."
fi

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
    if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -ge 13 ]; then
        echo "⚠️ Bỏ qua pydensecrf2 vì Python 3.13 không hỗ trợ C-API cũ của pydensecrf (dùng Python <= 3.12 nếu cần DenseCRF)."
    else
        echo "Đang cài đặt pydensecrf2..."
        $PYTHON -m pip install pydensecrf2 --quiet || echo "⚠️ Không thể tự cài pydensecrf2, bỏ qua nếu chỉ chạy train."
    fi
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

# SAM VOC train_aug offline labels cache (cho SAT-CR và SBF-Distill)
SAM_CACHE_DIR="experiments/representation_sam_20261120/cache/sam/labels"
SAM_CACHE_TAR="pretrained/checkpoints/sam_voc_train_aug_cache.tar.gz"
SAM_CACHE_COUNT=$(ls "$SAM_CACHE_DIR" 2>/dev/null | wc -l || echo "0")

if [ "$SAM_CACHE_COUNT" -lt 1000 ]; then
    echo "Thư mục SAM cache ($SAM_CACHE_DIR) chưa đầy đủ (hiện có $SAM_CACHE_COUNT mặt nạ)."
    if [ -f "$SAM_CACHE_TAR" ]; then
        echo "Đang tự động giải nén SAM VOC train_aug cache từ $SAM_CACHE_TAR..."
        mkdir -p "experiments/representation_sam_20261120/cache/sam"
        tar -xzf "$SAM_CACHE_TAR" -C "experiments/representation_sam_20261120/cache/sam/"
        echo "✓ Đã giải nén thành công SAM offline cache ($(ls "$SAM_CACHE_DIR" | wc -l) mặt nạ)."
    elif [ -f "/tmp/sam_voc_train_aug_cache.tar.gz" ]; then
        echo "Đang tự động giải nén SAM VOC train_aug cache từ /tmp/sam_voc_train_aug_cache.tar.gz..."
        mkdir -p "experiments/representation_sam_20261120/cache/sam"
        tar -xzf "/tmp/sam_voc_train_aug_cache.tar.gz" -C "experiments/representation_sam_20261120/cache/sam/"
        echo "✓ Đã giải nén thành công SAM offline cache ($(ls "$SAM_CACHE_DIR" | wc -l) mặt nạ)."
    elif [ -d "/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE/experiments/representation_sam_20261120/cache/sam/labels" ]; then
        echo "Đang liên kết mềm SAM cache từ máy chủ..."
        mkdir -p "experiments/representation_sam_20261120/cache/sam"
        rm -rf "$SAM_CACHE_DIR" 2>/dev/null || true
        ln -sf "/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE/experiments/representation_sam_20261120/cache/sam/labels" "$SAM_CACHE_DIR"
        echo "✓ Đã liên kết mềm SAM offline cache thành công."
    else
        echo "⚠️ Không tìm thấy $SAM_CACHE_TAR. Các module SAT-CR sẽ chạy với cơ chế an toàn fallback sang CAM pseudo-label."
    fi
else
    echo "✓ SAM offline mask cache đã sẵn sàng ($SAM_CACHE_COUNT mặt nạ tại $SAM_CACHE_DIR)."
fi

# ------------------------------------------------------------------------------
# 5. KIỂM TRA TẬP DỮ LIỆU & TỰ ĐỘNG SINH METADATA
# ------------------------------------------------------------------------------
echo ""
echo ">>> [5/6] Kiểm tra tập dữ liệu & Metadata..."

# Pascal VOC 2012
VOC_DIR="dataset/VOC2012"
VOC_READY=true
if [ ! -d "$VOC_DIR/JPEGImages" ] || [ ! -d "$VOC_DIR/ImageSets/Segmentation" ]; then
    VOC_READY=false
fi

if [ "$VOC_READY" = false ]; then
    echo "Chưa tìm thấy tập dữ liệu Pascal VOC 2012 đầy đủ."
    echo "Đang tự động tải bộ dữ liệu Pascal VOC 2012 (VOCtrainval ~2 GB)..."
    mkdir -p dataset
    VOC_TAR="dataset/VOCtrainval_11-May-2012.tar"
    if [ ! -f "$VOC_TAR" ]; then
        wget -c "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" -O "$VOC_TAR" || \
        wget -c "https://data.pjreddie.com/files/VOCtrainval_11-May-2012.tar" -O "$VOC_TAR" || {
            echo "⚠️ Tải VOCtrainval_11-May-2012.tar thất bại. Vui lòng tải thủ công hoặc kiểm tra mạng."
        }
    fi

    if [ -f "$VOC_TAR" ]; then
        echo "Đang giải nén VOCtrainval_11-May-2012.tar..."
        tar -xf "$VOC_TAR" -C dataset/
        if [ -d "dataset/VOCdevkit/VOC2012" ]; then
            [ -L "$VOC_DIR" ] && rm -f "$VOC_DIR"
            [ -d "$VOC_DIR" ] && rm -rf "$VOC_DIR"
            mv dataset/VOCdevkit/VOC2012 "$VOC_DIR"
            rm -rf dataset/VOCdevkit "$VOC_TAR"
            echo "✓ Đã giải nén Pascal VOC 2012 vào $VOC_DIR."
        fi
    fi
else
    echo "✓ Đã tìm thấy thư mục $VOC_DIR (JPEGImages & ImageSets)."
fi

# Nhãn mở rộng SegmentationClassAug (SBD Benchmark)
if [ -d "$VOC_DIR" ] && [ ! -d "$VOC_DIR/SegmentationClassAug" ]; then
    echo "Đang tự động tải nhãn mở rộng SegmentationClassAug (SBD Benchmark)..."
    AUG_ZIP="dataset/SegmentationClassAug.zip"
    wget -c "https://www.dropbox.com/s/g3twnwcrsmhpaze/SegmentationClassAug.zip?dl=1" -O "$AUG_ZIP" || {
        echo "⚠️ Không thể tải SegmentationClassAug.zip từ Dropbox. Vui lòng xem README.md để tải thủ công."
    }
    if [ -f "$AUG_ZIP" ]; then
        echo "Đang giải nén SegmentationClassAug..."
        unzip -q -o "$AUG_ZIP" -d "$VOC_DIR/" || true
        rm -f "$AUG_ZIP"
        echo "✓ Đã giải nén SegmentationClassAug vào $VOC_DIR."
    fi
fi

# Sinh metadata VOC nếu thiếu
if [ -d "$VOC_DIR/JPEGImages" ] && [ -d "$VOC_DIR/SegmentationClassAug" ]; then
    if [ ! -f "datasets/voc/train_aug.txt" ] || [ ! -f "datasets/voc/cls_labels_onehot.npy" ]; then
        echo "Đang sinh danh sách phân chia và nhãn cấp ảnh cho VOC2012..."
        $PYTHON scripts/prepare_voc.py \
            --data_folder "$VOC_DIR" \
            --train_list ./datasets/voc/train_aug.txt \
            --output datasets/voc || true
    else
        echo "✓ Metadata datasets/voc đã đầy đủ."
    fi
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
    echo "ℹ️ Bỏ qua dataset/COCO2014 (không bắt buộc tải nếu chỉ huấn luyện VOC/Paradigm 2)."
fi

# ------------------------------------------------------------------------------
# 6. PHÂN QUYỀN VÀ KIỂM THỬ ĐƠN VỊ TỰ ĐỘNG
# ------------------------------------------------------------------------------
echo ""
echo ">>> [6/6] Cấp quyền thực thi và chạy kiểm thử tự động (Unit Test)..."
chmod +x *.sh 2>/dev/null || true

echo "Chạy kiểm thử đơn vị cho các module SAM-PCRE..."
$PYTHON -m unittest discover -s tests -p "test_sam_innovations.py"

echo ""
echo "========================================================================"
echo "🎉 HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG TOÀN DIỆN THÀNH CÔNG!"
echo "Giờ bạn có thể chạy ./TRAIN.sh để thực hiện các tiến trình huấn luyện."
echo "========================================================================"
