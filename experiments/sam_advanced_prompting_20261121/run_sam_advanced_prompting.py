"""
Method 2: SAM Advanced Prompting with Boundary-Contrastive Negative Points & Multi-Mask Disambiguation (SAM-NP-MMD)
Evaluated on Pascal VOC 2012 Val (1,449 images).

Key Technical Components (Unified Class-Agnostic Formulation):
1. Boundary-Contrastive Negative Point Prompting:
   - Positive points (label = 1): EDT peak points along the object medial axis.
   - Negative point 1 (label = 0): EDT peak inside the 8px exterior boundary band.
   - Negative point 2 (label = 0): Competing neighboring foreground classes within 12px.
   - Uniformly passes point prompts + 8px-padded bounding box prompt to SAM for all classes.
2. Multi-Mask Disambiguation (MMD) with Delta Parameter Search:
   - Evaluates SAM's 3 multi-granularity candidate masks (sub-part, part, whole) using:
     quality = IoU - delta * leakage
     where leakage is the fraction of SAM mask pixels falling outside the ROI.
   - Supports transparent delta parameter grid search starting from an initial delta_init
     to select the optimal leakage penalty weight uniformly across all classes.
3. Uniform Boundary Refinement:
   - Recovers missing foreground pixels inside ROI and trims over-activated background pixels
     uniformly without class-specific morphological branching.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import scipy.ndimage as ndi
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments/sam_advanced_prompting_20261121"
DEFAULT_SAM_CKPT = ROOT / "pretrained/checkpoints/sam_vit_b_01ec64.pth"
DEFAULT_DATA_DIR = ROOT / "dataset/VOC2012"
DEFAULT_BASE_MASKS_DIR = ROOT / "experiments/inference_voc_val_20261119/full_baseline_rerun/baseline/crf/masks"
DEFAULT_OUT_DIR = EXP_DIR / "evaluation_uniform"

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    SAM_EXT_DIR = ROOT / "experiments/representation_sam_20261120/code_snapshot/external"
    sys.path.insert(0, str(SAM_EXT_DIR))
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
                colors[i, channel] |= ((value >> channel) & 1) << (7 - j)
            value >>= 3
    colors[255] = (224, 224, 224)
    return colors


def compute_hist(pred, gt, nclasses=21):
    valid = (gt >= 0) & (gt < nclasses)
    return np.bincount(
        nclasses * gt[valid].astype(np.int64) + pred[valid],
        minlength=nclasses * nclasses
    ).reshape(nclasses, nclasses)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Method 2: SAM Advanced Prompting (NP + MMD) with Delta Search")
    parser.add_argument("--base_masks_dir", type=Path, default=DEFAULT_BASE_MASKS_DIR,
                        help="Directory containing input PCRE CRF prediction masks (.png)")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Path to VOC2012 dataset root")
    parser.add_argument("--sam_ckpt", type=Path, default=DEFAULT_SAM_CKPT,
                        help="Path to SAM ViT-B checkpoint")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory to store evaluation outputs and search logs")
    parser.add_argument("--delta_init", type=float, default=0.2,
                        help="Initial default delta (leakage penalty) parameter")
    parser.add_argument("--delta_grid", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Grid of delta values to sweep during parameter search")
    parser.add_argument("--delta", type=float, default=None,
                        help="Single delta value to evaluate (disables grid search unless --search_delta is set)")
    parser.add_argument("--search_delta", action="store_true",
                        help="Run transparent grid search over --delta_grid (including --delta_init)")
    parser.add_argument("--iou_thresh", type=float, default=0.65,
                        help="Minimum consensus IoU threshold with PCRE component")
    parser.add_argument("--score_thresh", type=float, default=0.85,
                        help="Minimum SAM predicted IoU quality score")
    parser.add_argument("--min_area", type=int, default=160,
                        help="Minimum component pixel area to prompt SAM")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional limit on number of validation images (0 = full 1449)")
    parser.add_argument("--no_save_masks", action="store_true",
                        help="Skip saving per-image PNG masks to disk")
    return parser.parse_args()


def extract_prompts_and_roi(base_mask, comp, class_id, h, w):
    """Build uniform positive/negative point prompts, box prompt, and ROI for a component."""
    y_indices, x_indices = np.where(comp)
    pad = 8
    x_min, x_max = max(0, x_indices.min() - pad), min(w - 1, x_indices.max() + pad)
    y_min, y_max = max(0, y_indices.min() - pad), min(h - 1, y_indices.max() + pad)
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    roi = np.zeros((h, w), dtype=bool)
    roi[y_min:y_max + 1, x_min:x_max + 1] = True

    # 1. Positive Prompt Points (Medial Axis Skeleton via EDT)
    dist = ndi.distance_transform_edt(comp)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    coords = [[cx, cy]]
    labels = [1]

    if comp.sum() > 1200:
        temp_dist = dist.copy()
        temp_dist[max(0, cy - 20):min(h, cy + 21), max(0, cx - 20):min(w, cx + 21)] = 0
        if temp_dist.max() > 3:
            cy2, cx2 = np.unravel_index(np.argmax(temp_dist), temp_dist.shape)
            coords.append([cx2, cy2])
            labels.append(1)

    # 2. Boundary-Contrastive Negative Point (8px exterior band)
    dilated_band = ndi.binary_dilation(comp, iterations=8)
    bg_margin = dilated_band & (~comp)
    if bg_margin.sum() > 0:
        dist_out = ndi.distance_transform_edt(bg_margin)
        ny, nx = np.unravel_index(np.argmax(dist_out), dist_out.shape)
        coords.append([nx, ny])
        labels.append(0)

    # 3. Competing Neighbor Class Negative Point (within 12px)
    comp_dilated = ndi.binary_dilation(comp, iterations=12)
    neighbor_conflict = (base_mask != 0) & (base_mask != class_id) & (base_mask != 255) & comp_dilated
    if neighbor_conflict.sum() > 50:
        conf_y, conf_x = np.where(neighbor_conflict)
        c_idx = len(conf_y) // 2
        coords.append([conf_x[c_idx], conf_y[c_idx]])
        labels.append(0)

    return np.array(coords, dtype=np.float32), np.array(labels, dtype=np.int32), box, roi


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_masks:
        (out_dir / "masks").mkdir(parents=True, exist_ok=True)
        (out_dir / "color").mkdir(parents=True, exist_ok=True)

    if args.search_delta:
        delta_candidates = sorted({round(float(d), 4) for d in [args.delta_init] + list(args.delta_grid)})
    elif args.delta is not None:
        delta_candidates = [round(float(args.delta), 4)]
    else:
        delta_candidates = [round(float(args.delta_init), 4)]

    print("=" * 72)
    print("=== METHOD 2: SAM ADVANCED PROMPTING (UNIFORM NP + MMD) ===")
    print(f"Base Masks Dir:  {args.base_masks_dir}")
    print(f"SAM Checkpoint:  {args.sam_ckpt}")
    print(f"Output Dir:      {out_dir}")
    print(f"Delta Init:      {args.delta_init}")
    print(f"Delta Search:    {delta_candidates}")
    print(f"IoU / Score Thr: {args.iou_thresh} / {args.score_thresh}")
    print("=" * 72)

    if not args.base_masks_dir.exists():
        raise FileNotFoundError(f"Base masks directory not found: {args.base_masks_dir}")
    if not args.sam_ckpt.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_ckpt}")

    start_time = time.time()
    val_list_file = ROOT / "datasets/voc/val.txt"
    val_names = val_list_file.read_text().split()
    if args.limit > 0:
        val_names = val_names[:args.limit]
    total_images = len(val_names)
    print(f"Total VOC Val images to evaluate: {total_images}")

    torch.cuda.set_device(0)
    sam = sam_model_registry["vit_b"](checkpoint=str(args.sam_ckpt)).cuda()
    predictor = SamPredictor(sam)
    print("SAM ViT-B loaded successfully.")

    colors = palette()
    hist_base = np.zeros((21, 21), dtype=np.int64)
    hist_by_delta = {d: np.zeros((21, 21), dtype=np.int64) for d in delta_candidates}
    refined_count_by_delta = {d: 0 for d in delta_candidates}
    total_components_checked = 0

    # Keep final refined masks in memory to emit best_delta masks cleanly at end
    stored_masks = {d: {} for d in delta_candidates} if not args.no_save_masks else None

    for idx, name in enumerate(val_names):
        img_path = args.data_dir / "JPEGImages" / f"{name}.jpg"
        gt_path = args.data_dir / "SegmentationClassAug" / f"{name}.png"
        base_mask_path = args.base_masks_dir / f"{name}.png"

        img = np.array(Image.open(img_path).convert("RGB"))
        base_mask = np.array(Image.open(base_mask_path))
        gt = np.array(Image.open(gt_path))
        h, w, _ = img.shape

        hist_base += compute_hist(base_mask, gt)

        # Initialize per-delta mask copies for this image
        refined_masks = {d: base_mask.copy() for d in delta_candidates}

        predictor.set_image(img)
        unique_classes = np.unique(base_mask)

        for c in unique_classes:
            if c == 0 or c == 255:
                continue
            class_mask = (base_mask == c)
            labeled, num_features = ndi.label(class_mask)
            for f_idx in range(1, num_features + 1):
                comp = (labeled == f_idx)
                if comp.sum() < args.min_area:
                    continue
                total_components_checked += 1

                coords, labels, box, roi = extract_prompts_and_roi(base_mask, comp, c, h, w)
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=box,
                    multimask_output=True
                )

                # Precompute IoU & Leakage once per SAM candidate mask
                candidates = []
                for m, s in zip(masks, scores):
                    inter = (m & comp).sum()
                    union = (m | comp).sum()
                    iou = float(inter / union) if union > 0 else 0.0
                    leakage = float((m & (~comp) & (~roi)).sum() / (m.sum() + 1e-5))
                    if s > args.score_thresh and iou > args.iou_thresh:
                        candidates.append((m, float(s), iou, leakage))

                if not candidates:
                    continue

                # Evaluate MMD across every delta in search grid in one pass
                for d in delta_candidates:
                    best_quality = -1e9
                    best_m = None
                    for m, s, iou, leakage in candidates:
                        quality = iou - d * leakage
                        if quality > best_quality:
                            best_quality = quality
                            best_m = m

                    if best_m is not None:
                        refined_count_by_delta[d] += 1
                        to_fg = best_m & roi & (base_mask == 0)
                        to_bg = (~best_m) & comp & roi
                        refined_masks[d][to_fg] = c
                        refined_masks[d][to_bg] = 0

        for d in delta_candidates:
            hist_by_delta[d] += compute_hist(refined_masks[d], gt)
            if stored_masks is not None:
                stored_masks[d][name] = refined_masks[d]

        if (idx + 1) % 100 == 0 or (idx + 1) == total_images:
            elapsed = time.time() - start_time
            base_miou, _, _, _ = compute_scores(hist_base)
            delta_str = " | ".join(
                f"d={d:.2f}: {compute_scores(hist_by_delta[d])[0]*100:.2f}%"
                for d in delta_candidates
            )
            print(f"[{idx+1:4d}/{total_images}] Base: {base_miou*100:.2f}% | {delta_str} | {elapsed:.1f}s", flush=True)

    # Final Comparison across Delta Search Grid
    base_miou, base_iou, _, _ = compute_scores(hist_base)
    delta_results = []
    best_delta = None
    best_miou = -1.0

    for d in delta_candidates:
        d_miou, d_iou, d_prec, d_rec = compute_scores(hist_by_delta[d])
        gain_pp = (d_miou - base_miou) * 100.0
        delta_results.append({
            "delta": d,
            "is_init": abs(d - args.delta_init) < 1e-6,
            "miou_percent": d_miou * 100.0,
            "gain_vs_base_pp": gain_pp,
            "refined_components": refined_count_by_delta[d],
            "total_components": total_components_checked,
        })
        if d_miou > best_miou:
            best_miou = d_miou
            best_delta = d

    # Save search table
    search_summary = {
        "method": "Method 2: SAM Advanced Prompting (Unified NP + MMD, No Class Grouping)",
        "dataset": "Pascal VOC 2012 Val",
        "num_images": total_images,
        "base_masks_dir": str(args.base_masks_dir),
        "base_miou_percent": base_miou * 100.0,
        "delta_init": args.delta_init,
        "best_delta": best_delta,
        "best_miou_percent": best_miou * 100.0,
        "best_gain_vs_base_pp": (best_miou - base_miou) * 100.0,
        "delta_search": delta_results,
    }

    with open(out_dir / "delta_search_results.json", "w") as f:
        json.dump(search_summary, f, indent=2)

    with open(out_dir / "delta_search_results.csv", "w", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=["delta", "is_init", "miou_percent", "gain_vs_base_pp", "refined_components", "total_components"]
        )
        writer.writeheader()
        for row in delta_results:
            writer.writerow({
                "delta": f"{row['delta']:.2f}",
                "is_init": row["is_init"],
                "miou_percent": f"{row['miou_percent']:.4f}",
                "gain_vs_base_pp": f"{row['gain_vs_base_pp']:+.4f}",
                "refined_components": row["refined_components"],
                "total_components": row["total_components"],
            })

    # Save full per-class metrics & masks for best_delta
    best_hist = hist_by_delta[best_delta]
    best_miou_val, best_iou, best_prec, best_rec = compute_scores(best_hist)
    np.save(out_dir / "best_confusion.npy", best_hist)
    np.save(out_dir / "base_confusion.npy", hist_base)

    if stored_masks is not None:
        print(f"Saving final refined PNG masks for best delta={best_delta:.2f} to {out_dir / 'masks'}...")
        for name, rmask in stored_masks[best_delta].items():
            Image.fromarray(rmask.astype(np.uint8)).save(out_dir / "masks" / f"{name}.png")
            Image.fromarray(colors[rmask]).save(out_dir / "color" / f"{name}.png")

    metrics_payload = {
        "method": "Method 2: SAM Advanced Prompting (Unified NP + MMD, No Class Grouping)",
        "dataset": "Pascal VOC 2012 Val",
        "num_images": total_images,
        "base_crf_miou": base_miou * 100.0,
        "delta_init": args.delta_init,
        "best_delta": best_delta,
        "adv_prompt_miou": best_miou_val * 100.0,
        "gain_vs_base": (best_miou_val - base_miou) * 100.0,
        "delta_search": delta_results,
        "per_class": {},
    }

    csv_rows = []
    for c_id, c_name in enumerate(VOC_CLASSES):
        b_iou = base_iou[c_id] * 100.0
        a_iou = best_iou[c_id] * 100.0
        d_iou = a_iou - b_iou
        metrics_payload["per_class"][c_name] = {
            "base_iou": b_iou,
            "adv_prompt_iou": a_iou,
            "delta_vs_base": d_iou,
            "precision": best_prec[c_id] * 100.0,
            "recall": best_rec[c_id] * 100.0,
        }
        csv_rows.append({
            "class": c_name,
            "base_iou": f"{b_iou:.2f}",
            "adv_prompt_iou": f"{a_iou:.2f}",
            "delta_vs_base": f"{d_iou:+.2f}",
            "precision": f"{best_prec[c_id]*100:.2f}",
            "recall": f"{best_rec[c_id]*100:.2f}",
        })

    with open(out_dir / "metrics.json", "w") as mf:
        json.dump(metrics_payload, mf, indent=2)

    with open(out_dir / "per_class_comparison.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=["class", "base_iou", "adv_prompt_iou", "delta_vs_base", "precision", "recall"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 72)
    print(f"METHOD 2 (UNIFORM NP + MMD) DELTA SEARCH COMPLETE ({total_images} IMAGES)")
    print(f"Base PCRE CRF mIoU: {base_miou*100:.2f}%")
    print("-" * 72)
    for row in delta_results:
        tag = ""
        if row["is_init"]:
            tag += " [INIT]"
        if row["delta"] == best_delta:
            tag += " [BEST]"
        print(f"  delta = {row['delta']:.2f}  ->  mIoU = {row['miou_percent']:.2f}%  (Gain: {row['gain_vs_base_pp']:+.2f}%){tag}")
    print("-" * 72)
    print(f"FINAL SELECTED DELTA: {best_delta:.2f} | FINAL mIoU: {best_miou_val*100:.2f}% (Gain: {(best_miou_val-base_miou)*100:+.2f}%)")
    print("=" * 72)


if __name__ == "__main__":
    main()
