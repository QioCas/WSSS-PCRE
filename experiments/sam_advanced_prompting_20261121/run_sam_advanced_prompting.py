"""
Experiment 1: SAM Advanced Prompting with Negative Points & Multi-Mask Disambiguation (SAM-NP-MMD)
Applied on Pascal VOC 2012 Val (1,449 images).

Key Technical Novelties:
1. Boundary-Contrastive Negative Point Prompting:
   - For every candidate component, dynamically samples negative points (label 0) in the exterior
     boundary margin and neighboring competing classes, actively repelling SAM from bleeding into background.
2. Multi-Mask Disambiguation (Whole vs. Part):
   - Rather than naive IoU or argmax score, computes a leakage-penalized score across SAM's 3 multi-granularity
     outputs (sub-part, part, whole), selecting the optimal mask level suited for the object's geometry.
3. Class-Adaptive Geometric Branching:
   - Inherits color-gradient gating (\Delta RGB < 50) for slender objects and topological hole healing for solid objects.
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

ROOT = Path("/home/jupyter-iec2024iot08/Workspace/WSSS/WSSS-PCRE")
SAM_DIR = Path("/home/jupyter-iec2024iot08/Workspace/.deps/pcre_sam/segment-anything")
SAM_CKPT = Path("/home/jupyter-iec2024iot08/Workspace/.deps/pcre_sam/sam_vit_b_01ec64.pth")
DATA_DIR = Path("/home/jupyter-iec2024iot08/Workspace/2024_IoT_08-Contrastive-Learning-for-SSS/data/VOC2012")
BASE_MASKS_DIR = ROOT / "experiments/peft_csl_gtbf_synergy_20261121/crf_masks"
EXP_DIR = ROOT / "experiments/sam_advanced_prompting_20261121"
OUT_DIR = EXP_DIR / "evaluation"

sys.path.insert(0, str(SAM_DIR))
from segment_anything import sam_model_registry, SamPredictor

VOC_CLASSES = [
    "_background_", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

SLENDER_HOLLOW = {2, 9}           # bicycle, chair
COMPOSITE = {11, 15, 18}          # diningtable, person, sofa
SOLID_RIGID = set(range(1, 21)) - SLENDER_HOLLOW - COMPOSITE


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
    print("=" * 70)
    print("=== STARTING SAM ADVANCED PROMPTING: NEGATIVE POINTS + MMD ===")
    print("=" * 70)
    start_time = time.time()

    val_list_file = ROOT / "datasets/voc/val.txt"
    val_names = val_list_file.read_text().split()
    total_images = len(val_names)
    print(f"Total VOC Val images: {total_images}")

    print(f"Loading SAM ViT-B from {SAM_CKPT}...")
    torch.cuda.set_device(0)
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT)).cuda()
    predictor = SamPredictor(sam)
    print("SAM loaded successfully.")

    colors = palette()
    hist_peft = np.zeros((21, 21), dtype=np.int64)
    hist_adv = np.zeros((21, 21), dtype=np.int64)

    progress_file = EXP_DIR / "progress.json"
    status_file = EXP_DIR / "status.json"

    refined_components_count = 0
    total_components_checked = 0

    for idx, name in enumerate(val_names):
        img_path = DATA_DIR / "JPEGImages" / f"{name}.jpg"
        gt_path = DATA_DIR / "SegmentationClassAug" / f"{name}.png"
        base_mask_path = BASE_MASKS_DIR / f"{name}.png"

        img = np.array(Image.open(img_path).convert("RGB"))
        peft_mask = np.array(Image.open(base_mask_path))
        gt = np.array(Image.open(gt_path))
        h, w, _ = img.shape

        hist_peft += compute_hist(peft_mask, gt)

        refined_mask = peft_mask.copy()
        predictor.set_image(img)

        unique_classes = np.unique(peft_mask)
        for c in unique_classes:
            if c == 0 or c == 255:
                continue
            class_mask = (peft_mask == c)
            labeled, num_features = ndi.label(class_mask)
            for f_idx in range(1, num_features + 1):
                comp = (labeled == f_idx)
                area = comp.sum()
                if area < 160:
                    continue
                total_components_checked += 1

                y_indices, x_indices = np.where(comp)
                pad = 8
                x_min, x_max = max(0, x_indices.min() - pad), min(w - 1, x_indices.max() + pad)
                y_min, y_max = max(0, y_indices.min() - pad), min(h - 1, y_indices.max() + pad)
                box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

                # 1. Positive Prompt Points (Medial Axis Skeleton)
                dist = ndi.distance_transform_edt(comp)
                cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
                coords = [[cx, cy]]
                labels = [1]

                if area > 1200:
                    temp_dist = dist.copy()
                    temp_dist[max(0, cy-20):min(h, cy+21), max(0, cx-20):min(w, cx+21)] = 0
                    if temp_dist.max() > 3:
                        cy2, cx2 = np.unravel_index(np.argmax(temp_dist), temp_dist.shape)
                        coords.append([cx2, cy2])
                        labels.append(1)

                # 2. Negative Point Prompting (Boundary-Contrastive)
                dilated_band = ndi.binary_dilation(comp, iterations=8)
                bg_margin = dilated_band & (~comp)
                if bg_margin.sum() > 0:
                    dist_out = ndi.distance_transform_edt(bg_margin)
                    ny, nx = np.unravel_index(np.argmax(dist_out), dist_out.shape)
                    coords.append([nx, ny])
                    labels.append(0)

                # Check for neighboring competing foreground classes within 12 pixels
                comp_dilated = ndi.binary_dilation(comp, iterations=12)
                neighbor_conflict = (peft_mask != 0) & (peft_mask != c) & (peft_mask != 255) & comp_dilated
                if neighbor_conflict.sum() > 50:
                    conf_y, conf_x = np.where(neighbor_conflict)
                    # Pick point closest to component boundary
                    c_idx = len(conf_y) // 2
                    coords.append([conf_x[c_idx], conf_y[c_idx]])
                    labels.append(0)

                # Branch 1: Slender hollow objects omit box prompt!
                use_box = None if c in SLENDER_HOLLOW else box

                # Predict 3 multi-granularity masks from SAM
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(coords, dtype=np.float32),
                    point_labels=np.array(labels, dtype=np.int32),
                    box=use_box,
                    multimask_output=True
                )

                # 3. Multi-Mask Disambiguation (Leakage-penalized Quality Score)
                thresh = 0.50 if c in SLENDER_HOLLOW else (0.55 if c in COMPOSITE else 0.65)
                leak_weight = 0.40 if c in SLENDER_HOLLOW else (0.25 if c in COMPOSITE else 0.10)

                roi = np.zeros((h, w), dtype=bool)
                roi[y_min:y_max+1, x_min:x_max+1] = True

                best_quality, best_m = -1.0, None
                for m, s in zip(masks, scores):
                    inter = (m & comp).sum()
                    union = (m | comp).sum()
                    iou = inter / union if union > 0 else 0

                    # Penalize pixels spilling far beyond ROI and into background
                    leakage = (m & (~comp) & (~roi)).sum() / (m.sum() + 1e-5)
                    quality = iou - leak_weight * leakage

                    if s > 0.80 and iou > thresh and quality > best_quality:
                        best_quality = quality
                        best_m = m

                if best_m is not None:
                    refined_components_count += 1
                    to_fg = best_m & roi & (peft_mask == 0)

                    # Branch 2: Color-gradient gating for slender objects
                    if c in SLENDER_HOLLOW:
                        core_color = img[comp].mean(axis=0)
                        bg_candidate_colors = img[to_fg]
                        if len(bg_candidate_colors) > 0:
                            color_diff = np.linalg.norm(bg_candidate_colors - core_color, axis=-1)
                            valid_fg = np.zeros(to_fg.shape, dtype=bool)
                            valid_indices = np.where(to_fg)
                            valid_fg[(valid_indices[0][color_diff < 50], valid_indices[1][color_diff < 50])] = True
                            to_fg = valid_fg

                    refined_mask[to_fg] = c

                    # Branch 3: Trim background overactivation for solid rigid classes
                    if c in SOLID_RIGID:
                        to_bg = (~best_m) & comp & roi
                        refined_mask[to_bg] = 0

        # Topological Hole Closing on Solid Rigid Objects
        for sol in SOLID_RIGID:
            sol_mask = (refined_mask == sol)
            if sol_mask.sum() > 200:
                filled = ndi.binary_fill_holes(sol_mask)
                refined_mask[(refined_mask == 0) & filled] = sol

        hist_adv += compute_hist(refined_mask, gt)

        # Save masks
        Image.fromarray(refined_mask.astype(np.uint8)).save(OUT_DIR / "masks" / f"{name}.png")
        Image.fromarray(colors[refined_mask]).save(OUT_DIR / "color" / f"{name}.png")

        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            elapsed = time.time() - start_time
            cur_miou_peft, _, _, _ = compute_scores(hist_peft)
            cur_miou_adv, _, _, _ = compute_scores(hist_adv)
            delta = (cur_miou_adv - cur_miou_peft) * 100

            progress_data = {
                "processed": idx + 1,
                "total": total_images,
                "elapsed_seconds": elapsed,
                "peft_csl_crf_miou": cur_miou_peft * 100,
                "adv_prompt_miou": cur_miou_adv * 100,
                "delta_pp": delta,
                "refined_components": refined_components_count,
                "total_components": total_components_checked
            }
            with open(progress_file, "w") as pf:
                json.dump(progress_data, pf, indent=2)

            with open(status_file, "w") as sf:
                json.dump({
                    "stage": "running" if (idx + 1) < total_images else "completed",
                    "description": f"Processed {idx+1}/{total_images} images (Current AdvPrompt mIoU: {cur_miou_adv*100:.2f}%)",
                    "progress": progress_data
                }, sf, indent=2)

            print(f"[{idx+1:4d}/{total_images}] PEFT-CSL: {cur_miou_peft*100:.2f}% | AdvPrompt: {cur_miou_adv*100:.2f}% | Gain: {delta:+.2f}% | Elapsed: {elapsed:.1f}s", flush=True)

    # Final scores
    final_miou_peft, peft_iou, peft_prec, peft_rec = compute_scores(hist_peft)
    final_miou_adv, adv_iou, adv_prec, adv_rec = compute_scores(hist_adv)
    total_delta = (final_miou_adv - final_miou_peft) * 100

    true_base_path = ROOT / "experiments/inference_voc_val_20261119/full/baseline/metrics.json"
    true_base_m = json.loads(true_base_path.read_text())["crf"]["miou"] * 100.0

    np.save(OUT_DIR / "adv_confusion.npy", hist_adv)
    np.save(OUT_DIR / "peft_confusion.npy", hist_peft)

    results = {
        "method": "SAM Advanced Prompting: Negative Points + MMD",
        "dataset": "Pascal VOC 2012 Val",
        "num_images": total_images,
        "true_baseline_crf_miou": true_base_m,
        "peft_csl_crf_miou": final_miou_peft * 100,
        "adv_prompt_miou": final_miou_adv * 100,
        "net_gain_vs_baseline": (final_miou_adv * 100) - true_base_m,
        "gain_vs_peft_csl": total_delta,
        "per_class": {}
    }

    csv_rows = []
    for c_id, c_name in enumerate(VOC_CLASSES):
        p_iou = peft_iou[c_id] * 100
        a_iou = adv_iou[c_id] * 100
        d_iou = a_iou - p_iou
        results["per_class"][c_name] = {
            "peft_csl_iou": p_iou,
            "adv_prompt_iou": a_iou,
            "delta_vs_peft": d_iou,
            "precision": adv_prec[c_id] * 100,
            "recall": adv_rec[c_id] * 100
        }
        csv_rows.append({
            "class": c_name,
            "peft_csl_iou": f"{p_iou:.2f}",
            "adv_prompt_iou": f"{a_iou:.2f}",
            "delta_vs_peft": f"{d_iou:+.2f}",
            "precision": f"{adv_prec[c_id]*100:.2f}",
            "recall": f"{adv_rec[c_id]*100:.2f}"
        })

    with open(OUT_DIR / "metrics.json", "w") as mf:
        json.dump(results, mf, indent=2)

    with open(EXP_DIR / "per_class_comparison.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["class", "peft_csl_iou", "adv_prompt_iou", "delta_vs_peft", "precision", "recall"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 70)
    print("SAM ADVANCED PROMPTING EVALUATION COMPLETE (ALL 1449 IMAGES)")
    print(f"True Baseline CRF mIoU:      {true_base_m:.2f}%")
    print(f"PEFT-CSL CRF mIoU:           {final_miou_peft*100:.2f}%")
    print(f"SAM AdvPrompt (NP+MMD) mIoU: {final_miou_adv*100:.2f}%")
    print(f"NET GAIN OVER BASELINE:      +{(final_miou_adv*100) - true_base_m:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
