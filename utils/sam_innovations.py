"""
SAM-PCRE Paradigm 2 Innovations:
1. SAT-CR (SAM-Affinity Token Contrastive Regularization)
   Purifies the ViT Backbone patch-token affinity matrix (aff_mask) using offline SAM patch purity.
2. SBF-Distill (SAM-Guided Boundary-Focal Density Distillation)
   Weights the segmentation cross-entropy loss with a smooth boundary-focal Gaussian band around SAM edges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.camutils import label_to_aff_mask


def compute_sam_patch_affinity(sam_mask, pseudo_label_aux, purity_thresh=0.85, ignore_index=255):
    """
    Method 1: SAT-CR (SAM-Affinity Token Contrastive Regularization)
    
    Args:
        sam_mask: (B, H, W) tensor with class IDs or 0 (bg), 255 (ignore)
        pseudo_label_aux: (B, h, w) tensor of downsampled CAM pseudo labels (e.g. 28x28)
        purity_thresh: minimum ratio of homogeneous pixels to consider a patch pure
        ignore_index: ignore index (255)
    Returns:
        aff_mask: (B, h*w, h*w) token affinity mask where boundary/noisy tokens are set to 255
        refined_aux: (B, h, w) purified token pseudo labels
    """
    b, H, W = sam_mask.shape
    _, h, w = pseudo_label_aux.shape
    kh, kw = H // h, W // w

    refined_aux = pseudo_label_aux.clone()

    # Kiểm tra những ảnh nào trong batch thực sự có SAM mask hợp lệ (khác ignore_index và có đối tượng)
    has_sam = ((sam_mask != ignore_index) & (sam_mask > 0)).sum(dim=(-2, -1)) > 0
    if not has_sam.any():
        # Fallback an toàn: Khi hoàn toàn không có SAM mask (hoặc cache rỗng / toàn 255),
        # bảo toàn nguyên vẹn pseudo_label_aux và sinh aff_mask chuẩn từ CAM.
        aff_mask = label_to_aff_mask(refined_aux, ignore_index=ignore_index)
        return aff_mask, refined_aux

    sam_fg = ((sam_mask > 0) & (sam_mask != ignore_index)).float().unsqueeze(1)
    sam_bg = (sam_mask == 0).float().unsqueeze(1)

    # Average pool to obtain patch foreground and background ratios
    fg_ratio = F.avg_pool2d(sam_fg, kernel_size=(kh, kw), stride=(kh, kw)).squeeze(1)
    bg_ratio = F.avg_pool2d(sam_bg, kernel_size=(kh, kw), stride=(kh, kw)).squeeze(1)
    valid_ratio = fg_ratio + bg_ratio

    # Điểm boundary thực sự:
    # 1. SAM phải có dữ liệu trong patch (valid_ratio >= 0.5)
    # 2. Cả FG và BG đều xuất hiện trong patch (ranh giới vật thể)
    # 3. Không có bên nào đạt ngưỡng thuần khiết (purity_thresh)
    boundary_patch = (
        (valid_ratio >= 0.5) &
        (fg_ratio > 0.05) &
        (bg_ratio > 0.05) &
        (fg_ratio < purity_thresh) &
        (bg_ratio < purity_thresh)
    )

    # 1. Boundary patches receive ignore_index (eliminates false affinity pairs in self-attention)
    refined_aux[boundary_patch] = ignore_index

    # 2. Pure background patches are anchored to 0
    pure_bg = (bg_ratio >= purity_thresh)
    refined_aux[pure_bg] = 0

    # Đối với những ảnh không có SAM trong batch, bảo toàn pseudo_label_aux gốc
    no_sam_indices = (~has_sam).nonzero(as_tuple=True)[0]
    if len(no_sam_indices) > 0:
        refined_aux[no_sam_indices] = pseudo_label_aux[no_sam_indices]

    # Generate purified affinity mask (B, h*w, h*w)
    aff_mask = label_to_aff_mask(refined_aux, ignore_index=ignore_index)

    return aff_mask, refined_aux


def compute_boundary_weight_map(sam_mask, alpha=2.0, kernel_size=9, ignore_index=255):
    """
    Method 2: SBF-Distill (SAM-Guided Boundary-Focal Density Distillation)
    
    Computes a GPU-accelerated boundary weight map from SAM masks using morphological pooling.
    
    Args:
        sam_mask: (B, H, W) tensor
        alpha: boundary penalty scaling factor (e.g. 2.0 -> boundary pixels have 3.0x weight)
        kernel_size: smoothing kernel size for the Gaussian boundary band
        ignore_index: ignore index (255)
    Returns:
        weight_map: (B, H, W) tensor with values in [1.0, 1.0 + alpha]
    """
    fg_binary = ((sam_mask > 0) & (sam_mask != ignore_index)).float().unsqueeze(1)

    # Dilation and erosion to extract exact 1-pixel boundary
    dilated = F.max_pool2d(fg_binary, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-fg_binary, kernel_size=3, stride=1, padding=1)
    boundary_raw = (dilated - eroded).clamp(min=0.0)

    # Multi-pixel smooth boundary band via average pooling
    pad = kernel_size // 2
    boundary_band = F.avg_pool2d(boundary_raw, kernel_size=kernel_size, stride=1, padding=pad)

    # Normalize band per image so peak weight at boundary is 1.0 + alpha
    band_max = boundary_band.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-5)
    normalized_band = (boundary_band / band_max).squeeze(1)

    weight_map = 1.0 + alpha * normalized_band
    return weight_map


def get_boundary_focal_seg_loss(pred, label, weight_map, ignore_index=255):
    """
    Boundary-focal weighted cross entropy loss for segmentation decoder.
    
    Args:
        pred: (B, num_classes, H, W) predicted logits
        label: (B, H, W) target class labels
        weight_map: (B, H, W) boundary focal weights (>= 1.0)
        ignore_index: ignore index (255)
    Returns:
        loss: scalar tensor
    """
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    loss_pixel = ce(pred, label.type(torch.long))

    valid_bg = (label == 0).float()
    valid_fg = ((label > 0) & (label != ignore_index)).float()

    bg_weight = weight_map * valid_bg
    fg_weight = weight_map * valid_fg

    bg_loss = (loss_pixel * bg_weight).sum() / (bg_weight.sum() + 1e-6)
    fg_loss = (loss_pixel * fg_weight).sum() / (fg_weight.sum() + 1e-6)

    return 0.5 * (bg_loss + fg_loss)
