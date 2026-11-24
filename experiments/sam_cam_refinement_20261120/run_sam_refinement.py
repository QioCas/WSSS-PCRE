"""SAM-CAM Consensus Semantic Refinement (SCSR) on Pascal VOC 2012 Val.

Leverages the segment boundary precision and part-whole grouping of SAM
prompted by PCRE baseline peaks and bounding boxes, subject to semantic
consensus filtering.
"""
import os
import sys
import time
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import torch

# Paths
ROOT = Path("/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE")
SAM_DIR = Path("/home/jupyter-iec2024iot08/Workspace/.deps/pcre_sam/segment-anything")
SAM_CKPT = Path("/home/jupyter-iec2024iot08/Workspace/.deps/pcre_sam/sam_vit_b_01ec64.pth")
DATA_DIR = Path("/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
BASE_MASKS_DIR = ROOT / "experiments/inference_voc_val_20261119/full/baseline/crf/masks"
EXP_DIR = ROOT / "experiments/sam_cam_refinement_20261120"
OUT_DIR = EXP_DIR / "evaluation"

sys.path.insert(0, str(SAM_DIR))
from segment_anything import sam_model_registry, SamPredictor

VOC_CLASSES = [
    "_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def palette():
    colors = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        value = i
        for j in range(8):
            for channel in range(3):
                colors[i, channel] |= ((value >> channel) & 1) << (7-j)
            value >>= 3
    colors[255] = (224, 224, 224)
    return colors

def compute_hist(pred, gt, nclasses=21):
    valid = (gt >= 0) & (gt < nclasses)
    return np.bincount(nclasses * gt[valid].astype(np.int64) + pred[valid],
                       minlength=nclasses * nclasses).reshape(nclasses, nclasses)

def compute_scores(hist):
    tp = np.diag(hist).astype(float)
    fn = hist.sum(1) - tp
    fp = hist.sum(0) - tp
    union = tp + fp + fn
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    present = union > 0
    miou = float(np.mean(iou[present]))
    return miou, iou, precision, recall

def main():
    print("=== Starting SAM-CAM Semantic Refinement (SCSR) Pipeline ===")
    start_time = time.time()
    
    val_list_file = ROOT / "datasets/voc/val.txt"
    val_names = val_list_file.read_text().split()
    total_images = len(val_names)
    assert total_images == 1449, f"Expected 1449 images, found {total_images}"
    print(f"Total VOC Val images: {total_images}")

    # Load SAM
    print(f"Loading SAM ViT-B from {SAM_CKPT}...")
    torch.cuda.set_device(0)
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT)).cuda()
    predictor = SamPredictor(sam)
    print("SAM loaded successfully.")

    colors = palette()
    hist_base = np.zeros((21, 21), dtype=np.int64)
    hist_sam = np.zeros((21, 21), dtype=np.int64)

    progress_file = EXP_DIR / "progress.json"
    status_file = EXP_DIR / "status.json"

    refined_components_count = 0
    total_components_checked = 0

    for idx, name in enumerate(val_names):
        img_path = DATA_DIR / "JPEGImages" / f"{name}.jpg"
        gt_path = DATA_DIR / "SegmentationClassAug" / f"{name}.png"
        base_mask_path = BASE_MASKS_DIR / f"{name}.png"

        img = np.array(Image.open(img_path).convert("RGB"))
        base_mask = np.array(Image.open(base_mask_path))
        gt = np.array(Image.open(gt_path))
        h, w, _ = img.shape

        hist_base += compute_hist(base_mask, gt)

        # Refine with SAM
        refined_mask = base_mask.copy()
        predictor.set_image(img)

        unique_classes = np.unique(base_mask)
        for c in unique_classes:
            if c == 0 or c == 255:
                continue
            class_mask = (base_mask == c)
            labeled, num_features = ndi.label(class_mask)
            for f_idx in range(1, num_features + 1):
                comp = (labeled == f_idx)
                area = comp.sum()
                if area < 180: # Filter small noisy fragments
                    continue
                total_components_checked += 1

                # Box prompt with safety padding
                y_indices, x_indices = np.where(comp)
                pad = 8
                x_min, x_max = max(0, x_indices.min() - pad), min(w - 1, x_indices.max() + pad)
                y_min, y_max = max(0, y_indices.min() - pad), min(h - 1, y_indices.max() + pad)
                box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

                # Peak point prompt
                dist = ndi.distance_transform_edt(comp)
                py, px = np.unravel_index(dist.argmax(), dist.shape)
                points = [[px, py]]
                point_labels = [1]

                masks, scores, _ = predictor.predict(
                    point_coords=np.array(points, dtype=np.float32),
                    point_labels=np.array(point_labels, dtype=np.int32),
                    box=box,
                    multimask_output=True
                )

                best_iou, best_m = 0, None
                for m, s in zip(masks, scores):
                    inter = (m & comp).sum()
                    union = (m | comp).sum()
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou and s > 0.85:
                        best_iou = iou
                        best_m = m

                # High agreement consensus: IoU > 0.65
                if best_m is not None and best_iou > 0.65:
                    refined_components_count += 1
                    roi = np.zeros((h, w), dtype=bool)
                    roi[y_min:y_max+1, x_min:x_max+1] = True

                    # 1. Recover missing object boundaries (expand into background only)
                    to_fg = best_m & roi & (base_mask == 0)
                    # 2. Trim over-activation / background leakage (within the component)
                    to_bg = (~best_m) & comp & roi

                    refined_mask[to_fg] = c
                    refined_mask[to_bg] = 0

        hist_sam += compute_hist(refined_mask, gt)

        # Save refined masks
        Image.fromarray(refined_mask.astype(np.uint8)).save(OUT_DIR / "masks" / f"{name}.png")
        Image.fromarray(colors[refined_mask]).save(OUT_DIR / "color" / f"{name}.png")

        if (idx + 1) % 50 == 0 or (idx + 1) == total_images:
            elapsed = time.time() - start_time
            cur_miou_base, _, _, _ = compute_scores(hist_base)
            cur_miou_sam, _, _, _ = compute_scores(hist_sam)
            delta = (cur_miou_sam - cur_miou_base) * 100
            
            progress_data = {
                "processed": idx + 1,
                "total": total_images,
                "elapsed_seconds": elapsed,
                "baseline_crf_miou": cur_miou_base * 100,
                "sam_refined_miou": cur_miou_sam * 100,
                "delta_pp": delta,
                "refined_components": refined_components_count,
                "total_components": total_components_checked
            }
            with open(progress_file, "w") as pf:
                json.dump(progress_data, pf, indent=2)
            
            with open(status_file, "w") as sf:
                json.dump({
                    "stage": "running" if (idx + 1) < total_images else "completed",
                    "description": f"Processed {idx+1}/{total_images} images (Current Gain: {delta:+.2f}%)",
                    "progress": progress_data
                }, sf, indent=2)
            
            print(f"[{idx+1}/{total_images}] Base: {cur_miou_base*100:.2f}% | SAM: {cur_miou_sam*100:.2f}% | Gain: {delta:+.2f}% | Elapsed: {elapsed:.1f}s", flush=True)

    # Final metrics computation
    final_miou_base, base_iou, base_prec, base_rec = compute_scores(hist_base)
    final_miou_sam, sam_iou, sam_prec, sam_rec = compute_scores(hist_sam)
    total_delta = (final_miou_sam - final_miou_base) * 100

    np.save(OUT_DIR / "crf_confusion.npy", hist_sam)
    np.save(OUT_DIR / "base_confusion.npy", hist_base)

    results = {
        "dataset": "Pascal VOC 2012 Val",
        "num_images": total_images,
        "baseline_crf_miou": final_miou_base * 100,
        "sam_cam_refined_miou": final_miou_sam * 100,
        "gain_pp": total_delta,
        "per_class": {}
    }

    csv_rows = []
    for c_id, c_name in enumerate(VOC_CLASSES):
        b_iou = base_iou[c_id] * 100
        s_iou = sam_iou[c_id] * 100
        d_iou = s_iou - b_iou
        results["per_class"][c_name] = {
            "baseline_iou": b_iou,
            "refined_iou": s_iou,
            "delta": d_iou,
            "precision": sam_prec[c_id] * 100,
            "recall": sam_rec[c_id] * 100
        }
        csv_rows.append({
            "class": c_name,
            "baseline_iou": f"{b_iou:.2f}",
            "refined_iou": f"{s_iou:.2f}",
            "delta": f"{d_iou:+.2f}",
            "sam_precision": f"{sam_prec[c_id]*100:.2f}",
            "sam_recall": f"{sam_rec[c_id]*100:.2f}"
        })

    with open(OUT_DIR / "metrics.json", "w") as mf:
        json.dump(results, mf, indent=2)

    with open(EXP_DIR / "per_class_comparison.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["class", "baseline_iou", "refined_iou", "delta", "sam_precision", "sam_recall"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "="*50)
    print("FINAL EVALUATION COMPLETE (ALL 1449 IMAGES)")
    print(f"Baseline CRF mIoU:     {final_miou_base*100:.2f}%")
    print(f"SAM-CAM Refined mIoU:  {final_miou_sam*100:.2f}%")
    print(f"Net Gain:              {total_delta:+.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
